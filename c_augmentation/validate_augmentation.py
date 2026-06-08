#!/usr/bin/env python3
"""
Validate augmented kernels by comparing CPU runtime against original.

This script:
1. Loads original and augmented code from JSONL
2. Compiles both versions
3. Runs on CPU and compares:
   - Output correctness (stdout comparison)
   - Runtime similarity (should be within tolerance)
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    """Result of compiling and running a kernel."""
    success: bool
    compile_time_s: float
    run_time_s: float
    run_time_std: float  # Standard deviation of run times
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class ComparisonResult:
    """Result of comparing original vs augmented."""
    kernel_name: str
    file_name: str
    original_result: Optional[RunResult]
    augmented_result: Optional[RunResult]
    output_matches: bool
    runtime_diff_percent: Optional[float]
    transforms_applied: list[str]
    status: str  # "PASS", "FAIL", "ERROR"
    message: str


def compile_and_run(
    code: str,
    filename: str,
    work_dir: Path,
    run_args: list[str] = None,
    timeout: int = 60,
    num_runs: int = 100
) -> RunResult:
    """
    Compile and run C/C++ code on CPU, averaging runtime over multiple runs.
    
    Args:
        code: Source code content
        filename: Name for the source file
        work_dir: Directory to work in
        run_args: Arguments to pass to the executable
        timeout: Maximum execution time in seconds
        num_runs: Number of runs to average for timing (default: 100)
    
    Returns:
        RunResult with timing and output information
    """
    run_args = run_args or []
    
    # Determine compiler based on extension
    ext = Path(filename).suffix.lower()
    if ext in ['.cpp', '.cc', '.cxx']:
        compiler = 'g++'
        std_flag = '-std=c++17'
    else:
        compiler = 'gcc'
        std_flag = '-std=c11'
    
    # Write source file
    src_path = work_dir / filename
    src_path.write_text(code)
    
    # Compile
    exe_path = work_dir / 'a.out'
    compile_cmd = [
        compiler,
        std_flag,
        '-O2',  # Optimize for fair comparison
        '-Wall',
        '-o', str(exe_path),
        str(src_path),
        '-lm',  # Math library
    ]
    
    compile_start = time.perf_counter()
    try:
        compile_result = subprocess.run(
            compile_cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        compile_time = time.perf_counter() - compile_start
        
        if compile_result.returncode != 0:
            return RunResult(
                success=False,
                compile_time_s=compile_time,
                run_time_s=0,
                run_time_std=0,
                stdout='',
                stderr=f"Compilation failed:\n{compile_result.stderr}",
                exit_code=compile_result.returncode
            )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False,
            compile_time_s=timeout,
            run_time_s=0,
            run_time_std=0,
            stdout='',
            stderr='Compilation timed out',
            exit_code=-1
        )
    except Exception as e:
        return RunResult(
            success=False,
            compile_time_s=0,
            run_time_s=0,
            run_time_std=0,
            stdout='',
            stderr=f'Compilation error: {e}',
            exit_code=-1
        )
    
    # Run multiple times and average
    run_cmd = [str(exe_path)] + run_args
    run_times = []
    last_stdout = ''
    last_stderr = ''
    last_exit_code = 0
    
    try:
        # Warmup run (don't count)
        subprocess.run(run_cmd, cwd=work_dir, capture_output=True, text=True, timeout=timeout)
        
        for i in range(num_runs):
            run_start = time.perf_counter()
            run_result = subprocess.run(
                run_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            run_time = time.perf_counter() - run_start
            run_times.append(run_time)
            last_stdout = run_result.stdout
            last_stderr = run_result.stderr
            last_exit_code = run_result.returncode
            
            if run_result.returncode != 0:
                break
        
        avg_run_time = sum(run_times) / len(run_times) if run_times else 0
        
        # Calculate standard deviation
        if len(run_times) > 1:
            variance = sum((t - avg_run_time) ** 2 for t in run_times) / len(run_times)
            std_run_time = variance ** 0.5
        else:
            std_run_time = 0
        
        return RunResult(
            success=last_exit_code == 0,
            compile_time_s=compile_time,
            run_time_s=avg_run_time,
            run_time_std=std_run_time,
            stdout=last_stdout,
            stderr=last_stderr,
            exit_code=last_exit_code
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False,
            compile_time_s=compile_time,
            run_time_s=timeout,
            run_time_std=0,
            stdout='',
            stderr='Execution timed out',
            exit_code=-1
        )
    except Exception as e:
        return RunResult(
            success=False,
            compile_time_s=compile_time,
            run_time_s=0,
            run_time_std=0,
            stdout='',
            stderr=f'Execution error: {e}',
            exit_code=-1
        )


def create_standalone_main(code: str, kernel_name: str) -> str:
    """
    If the code doesn't have a main function, wrap it with a simple test harness.
    """
    if 'int main(' in code or 'void main(' in code:
        return code
    
    # Add a simple main that calls the first function found
    # This is a basic heuristic - real usage would need kernel-specific harnesses
    wrapper = f'''
// Auto-generated test harness
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

{code}

int main(int argc, char** argv) {{
    printf("Kernel {kernel_name} compiled and linked successfully\\n");
    printf("Note: No executable test - code structure validation only\\n");
    return 0;
}}
'''
    return wrapper


def compare_outputs(original_stdout: str, augmented_stdout: str, tolerance: float = 0.001) -> bool:
    """
    Compare outputs, allowing for small floating-point differences.
    """
    if original_stdout == augmented_stdout:
        return True
    
    # Try to parse as numbers and compare with tolerance
    try:
        orig_lines = original_stdout.strip().split('\n')
        aug_lines = augmented_stdout.strip().split('\n')
        
        if len(orig_lines) != len(aug_lines):
            return False
        
        for orig_line, aug_line in zip(orig_lines, aug_lines):
            if orig_line == aug_line:
                continue
            
            # Try numeric comparison
            try:
                orig_val = float(orig_line)
                aug_val = float(aug_line)
                if abs(orig_val - aug_val) > tolerance * max(abs(orig_val), 1.0):
                    return False
            except ValueError:
                # Not a number, must match exactly
                if orig_line != aug_line:
                    return False
        
        return True
    except Exception:
        return False


def validate_sample_pair(
    original: dict,
    augmented: dict,
    work_dir: Path,
    run_args: list[str] = None,
    timeout: int = 60,
    num_runs: int = 100
) -> list[ComparisonResult]:
    """
    Validate an original/augmented pair.
    """
    results = []
    kernel_name = original.get('kernel_name', 'unknown')
    transforms = augmented.get('augmentation', {}).get('transforms_applied', [])
    
    orig_code = original.get('code', {})
    aug_code = augmented.get('code', {})
    
    for filename in orig_code:
        if filename not in aug_code:
            results.append(ComparisonResult(
                kernel_name=kernel_name,
                file_name=filename,
                original_result=None,
                augmented_result=None,
                output_matches=False,
                runtime_diff_percent=None,
                transforms_applied=transforms,
                status="ERROR",
                message=f"File {filename} missing in augmented version"
            ))
            continue
        
        orig_content = orig_code[filename]
        aug_content = aug_code[filename]
        
        # Skip if content is just a filename reference
        if len(orig_content) < 100 or len(aug_content) < 100:
            results.append(ComparisonResult(
                kernel_name=kernel_name,
                file_name=filename,
                original_result=None,
                augmented_result=None,
                output_matches=True,
                runtime_diff_percent=None,
                transforms_applied=transforms,
                status="SKIP",
                message="Code content too short (likely filename reference)"
            ))
            continue
        
        # Create work directories
        orig_dir = work_dir / 'original'
        aug_dir = work_dir / 'augmented'
        orig_dir.mkdir(exist_ok=True)
        aug_dir.mkdir(exist_ok=True)
        
        # Make code standalone if needed
        orig_standalone = create_standalone_main(orig_content, kernel_name)
        aug_standalone = create_standalone_main(aug_content, kernel_name)
        
        # Compile and run original
        orig_result = compile_and_run(
            orig_standalone, filename, orig_dir, run_args, timeout, num_runs
        )
        
        # Compile and run augmented
        aug_result = compile_and_run(
            aug_standalone, filename, aug_dir, run_args, timeout, num_runs
        )
        
        # Compare results
        if not orig_result.success:
            results.append(ComparisonResult(
                kernel_name=kernel_name,
                file_name=filename,
                original_result=orig_result,
                augmented_result=aug_result,
                output_matches=False,
                runtime_diff_percent=None,
                transforms_applied=transforms,
                status="ERROR",
                message=f"Original failed: {orig_result.stderr[:200]}"
            ))
            continue
        
        if not aug_result.success:
            results.append(ComparisonResult(
                kernel_name=kernel_name,
                file_name=filename,
                original_result=orig_result,
                augmented_result=aug_result,
                output_matches=False,
                runtime_diff_percent=None,
                transforms_applied=transforms,
                status="FAIL",
                message=f"Augmented failed: {aug_result.stderr[:200]}"
            ))
            continue
        
        # Compare outputs
        output_matches = compare_outputs(orig_result.stdout, aug_result.stdout)
        
        # Calculate runtime difference
        if orig_result.run_time_s > 0:
            runtime_diff = (aug_result.run_time_s - orig_result.run_time_s) / orig_result.run_time_s * 100
        else:
            runtime_diff = 0.0
        
        # Determine status
        if output_matches and abs(runtime_diff) < 50:  # Allow 50% runtime variance
            status = "PASS"
            message = f"Output matches, runtime diff: {runtime_diff:+.1f}%"
        elif not output_matches:
            status = "FAIL"
            message = "Output mismatch"
        else:
            status = "WARN"
            message = f"Large runtime difference: {runtime_diff:+.1f}%"
        
        results.append(ComparisonResult(
            kernel_name=kernel_name,
            file_name=filename,
            original_result=orig_result,
            augmented_result=aug_result,
            output_matches=output_matches,
            runtime_diff_percent=runtime_diff,
            transforms_applied=transforms,
            status=status,
            message=message
        ))
    
    return results


def load_augmented_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Load augmented JSONL and separate originals from augmented samples.
    """
    originals = []
    augmented = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                aug_info = sample.get('augmentation', {})
                if aug_info.get('is_original', False):
                    originals.append(sample)
                else:
                    augmented.append(sample)
            except json.JSONDecodeError:
                continue
    
    return originals, augmented


def main():
    parser = argparse.ArgumentParser(
        description='Validate augmented kernels against originals'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Augmented JSONL file (containing both originals and augmented)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output JSON file for results (default: stdout)'
    )
    parser.add_argument(
        '--run-args',
        type=str,
        default='',
        help='Arguments to pass to compiled programs (space-separated)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout for compilation/execution in seconds'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of runs to average for timing (default: 100)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary compilation directories'
    )
    
    args = parser.parse_args()
    
    # Load samples
    print(f"Loading samples from {args.input}...")
    originals, augmented = load_augmented_jsonl(args.input)
    
    print(f"  Found {len(originals)} original samples")
    print(f"  Found {len(augmented)} augmented samples")
    
    if len(originals) != len(augmented):
        print("Warning: Number of originals doesn't match augmented samples")
    
    # Match originals with augmented by kernel_name
    orig_by_name = {s['kernel_name']: s for s in originals}
    
    # Create temp directory
    if args.keep_temp:
        work_dir = Path('validation_temp')
        work_dir.mkdir(exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix='augment_validate_')
        work_dir = Path(temp_dir)
    
    run_args = args.run_args.split() if args.run_args else []
    
    all_results = []
    
    print("\nValidating augmented samples...")
    print("=" * 60)
    
    for aug_sample in augmented:
        kernel_name = aug_sample['kernel_name']
        transforms = aug_sample.get('augmentation', {}).get('transforms_applied', [])
        
        if kernel_name not in orig_by_name:
            print(f"  {kernel_name}: SKIP (no matching original)")
            continue
        
        orig_sample = orig_by_name[kernel_name]
        
        # Create kernel-specific work directory
        kernel_work_dir = work_dir / kernel_name
        kernel_work_dir.mkdir(exist_ok=True)
        
        results = validate_sample_pair(
            orig_sample, aug_sample, kernel_work_dir, run_args, args.timeout, args.num_runs
        )
        
        for result in results:
            status_symbol = {
                'PASS': '[OK]',
                'FAIL': '[X]',
                'WARN': '[!]',
                'ERROR': '[E]',
                'SKIP': '[-]'
            }.get(result.status, '[?]')
            
            print(f"  {status_symbol} {result.kernel_name}/{result.file_name}")
            print(f"      Transforms: {result.transforms_applied}")
            print(f"      {result.message}")
            
            if result.original_result and result.augmented_result:
                orig_r = result.original_result
                aug_r = result.augmented_result
                print(f"      Original runtime:  {orig_r.run_time_s:.6f}s (+/- {orig_r.run_time_std:.6f}s)")
                print(f"      Augmented runtime: {aug_r.run_time_s:.6f}s (+/- {aug_r.run_time_std:.6f}s)")
            
            all_results.append({
                'kernel_name': result.kernel_name,
                'file_name': result.file_name,
                'status': result.status,
                'output_matches': result.output_matches,
                'runtime_diff_percent': result.runtime_diff_percent,
                'transforms_applied': result.transforms_applied,
                'message': result.message,
                'original_runtime_s': result.original_result.run_time_s if result.original_result else None,
                'augmented_runtime_s': result.augmented_result.run_time_s if result.augmented_result else None,
            })
        
        print()
    
    # Cleanup
    if not args.keep_temp:
        shutil.rmtree(work_dir, ignore_errors=True)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    pass_count = sum(1 for r in all_results if r['status'] == 'PASS')
    fail_count = sum(1 for r in all_results if r['status'] == 'FAIL')
    error_count = sum(1 for r in all_results if r['status'] == 'ERROR')
    skip_count = sum(1 for r in all_results if r['status'] == 'SKIP')
    warn_count = sum(1 for r in all_results if r['status'] == 'WARN')
    
    print(f"  PASS:  {pass_count}")
    print(f"  FAIL:  {fail_count}")
    print(f"  WARN:  {warn_count}")
    print(f"  ERROR: {error_count}")
    print(f"  SKIP:  {skip_count}")
    
    if pass_count + warn_count > 0:
        valid_results = [r for r in all_results if r['runtime_diff_percent'] is not None]
        if valid_results:
            avg_diff = sum(r['runtime_diff_percent'] for r in valid_results) / len(valid_results)
            print(f"  Average runtime diff: {avg_diff:+.1f}%")
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'summary': {
                    'pass': pass_count,
                    'fail': fail_count,
                    'warn': warn_count,
                    'error': error_count,
                    'skip': skip_count,
                },
                'results': all_results
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Exit code based on results
    if fail_count > 0:
        return 1
    return 0


if __name__ == '__main__':
    exit(main())

