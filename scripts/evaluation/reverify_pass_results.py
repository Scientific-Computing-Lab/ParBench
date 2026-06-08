#!/usr/bin/env python3
"""Re-verify existing PASS eval results with corrected verification strategies.

Reads translated code from result JSONs, writes it to the target working directory,
builds, runs, and verifies with the updated verifier (stdout_pattern + exit_code).
Records TRUE_PASS vs FALSE_PASS for each result.

Usage:
    python3 scripts/evaluation/reverify_pass_results.py \
        --project-root . -v
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT_DEFAULT = _SCRIPT_DIR.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT_DEFAULT))

from harness.builder import build_spec
from harness.models import Status
from harness.runner import run_spec
from harness.spec_loader import load_spec, resolve_paths
from harness.verifier import verify_run

logger = logging.getLogger("reverify")

_SUPPORT_HEADER_EXTS = frozenset({".h", ".hpp", ".cuh"})


def backup_files(file_paths: list[Path]) -> dict[Path, bool]:
    """Backup original files before writing translated code."""
    backup_info: dict[Path, bool] = {}
    for fp in file_paths:
        if fp.exists():
            shutil.copy2(fp, str(fp) + ".parbench_bak")
            backup_info[fp] = True
        else:
            backup_info[fp] = False
    return backup_info


def restore_files(backup_info: dict[Path, bool]) -> None:
    """Restore original files after verification."""
    for fp, existed in backup_info.items():
        bak = Path(str(fp) + ".parbench_bak")
        if existed and bak.exists():
            shutil.copy2(bak, fp)
            bak.unlink()
        elif not existed and fp.exists():
            fp.unlink()
        if bak.exists():
            bak.unlink()


def stage_support_headers(
    source_spec: dict, target_spec_resolved: dict, project_root: Path
) -> list[Path]:
    """Copy source header files into target build directory (only if missing)."""
    source_resolved = resolve_paths(source_spec, project_root)
    source_dir: Path = source_resolved["_resolved"]["source_dir"]
    target_dir: Path = target_spec_resolved["_resolved"]["source_dir"]
    support_names: list[str] = source_spec.get("files", {}).get("support_files", [])

    staged: list[Path] = []
    for fname in support_names:
        if Path(fname).suffix.lower() not in _SUPPORT_HEADER_EXTS:
            continue
        src_fp = source_dir / fname
        tgt_fp = target_dir / fname
        if src_fp.exists() and not tgt_fp.exists():
            tgt_fp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fp, tgt_fp)
            staged.append(tgt_fp)
            logger.debug("Staged header %s → %s", src_fp.name, target_dir)
    return staged


def unstage_support_headers(staged_files: list[Path]) -> None:
    """Remove headers that were staged into the target build directory."""
    for fp in staged_files:
        fp.unlink(missing_ok=True)


def find_pass_results(
    results_dir: Path, model_filter: str | None = None
) -> list[Path]:
    """Find all PASS result JSON files."""
    pass_files = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_filter and model_dir.name != model_filter:
            continue
        for result_file in sorted(model_dir.glob("*.json")):
            try:
                data = json.loads(result_file.read_text())
                if data.get("overall_status") == "PASS":
                    pass_files.append(result_file)
            except (json.JSONDecodeError, OSError):
                continue
    return pass_files


def reverify_one(
    result_path: Path,
    specs_dir: Path,
    project_root: Path,
    verbose: bool = False,
) -> dict:
    """Re-verify a single PASS eval result.

    Returns a dict with reverification details.
    """
    data = json.loads(result_path.read_text())

    source_spec_id = data["source_spec"]
    target_spec_id = data["target_spec"]
    model = data["model"]
    augment_level = data.get("augment_level", 0)
    translated_files = data.get("translated_files", {})

    result = {
        "result_file": str(result_path.relative_to(project_root)),
        "source_spec": source_spec_id,
        "target_spec": target_spec_id,
        "model": model,
        "augment_level": augment_level,
        "original_status": "PASS",
        "original_verify_strategy": data.get("verify_strategy", "unknown"),
    }

    # Load target spec
    target_spec_path = specs_dir / f"{target_spec_id}.json"
    if not target_spec_path.exists():
        result["reverify_status"] = "SPEC_NOT_FOUND"
        result["error"] = f"Target spec not found: {target_spec_path}"
        return result

    # Load source spec (for header staging)
    source_spec_path = specs_dir / f"{source_spec_id}.json"
    if not source_spec_path.exists():
        result["reverify_status"] = "SPEC_NOT_FOUND"
        result["error"] = f"Source spec not found: {source_spec_path}"
        return result

    if not translated_files:
        result["reverify_status"] = "NO_TRANSLATED_FILES"
        result["error"] = "Result JSON has no translated_files"
        return result

    target_spec = load_spec(target_spec_path)
    source_spec = load_spec(source_spec_path)
    target_spec_resolved = resolve_paths(target_spec, project_root)

    source_dir: Path = target_spec_resolved["_resolved"]["source_dir"]

    # Build list of files to write
    target_file_paths = [source_dir / fname for fname in translated_files]

    # Backup, stage, write, build, run, verify, restore
    backup_info = backup_files(target_file_paths)
    staged_headers: list[Path] = []

    try:
        # Stage support headers from source spec
        staged_headers = stage_support_headers(
            source_spec, target_spec_resolved, project_root
        )

        # Write translated files
        for fname, code in translated_files.items():
            fp = source_dir / fname
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(code)

        # Build
        build_result = build_spec(
            target_spec_resolved, project_root, verbose=verbose
        )
        result["build_status"] = build_result.status.name

        if build_result.status != Status.PASS:
            result["reverify_status"] = "BUILD_FAIL"
            result["error"] = (build_result.stderr or "")[:500]
            return result

        # Run
        run_result = run_spec(
            target_spec_resolved, project_root, verbose=verbose
        )
        result["run_status"] = run_result.status.name
        result["run_exit_code"] = run_result.exit_code

        if run_result.status != Status.PASS:
            result["reverify_status"] = "RUN_FAIL"
            result["error"] = (run_result.stderr or "")[:500]
            return result

        # Verify with corrected strategies
        verify_result = verify_run(
            target_spec,
            run_result,
            working_dir=target_spec_resolved["_resolved"]["working_dir"],
        )
        result["verify_status"] = verify_result.status.name
        result["verify_strategy"] = verify_result.strategy_used
        result["verify_details"] = verify_result.details

        if verify_result.status == Status.PASS:
            result["reverify_status"] = "TRUE_PASS"
        else:
            result["reverify_status"] = "FALSE_PASS"
            result["stdout_snippet"] = (run_result.stdout or "")[:500]

        return result

    except Exception as exc:
        result["reverify_status"] = "ERROR"
        result["error"] = str(exc)
        return result

    finally:
        restore_files(backup_info)
        unstage_support_headers(staged_headers)


def main():
    parser = argparse.ArgumentParser(
        description="Re-verify PASS eval results with corrected verification"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT_DEFAULT,
        help="Path to parbench project root",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter to a specific model directory",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    project_root = args.project_root.resolve()
    specs_dir = project_root / "specs"
    results_dir = project_root / "results" / "evaluation"

    # Find all PASS results
    pass_files = find_pass_results(results_dir, model_filter=args.model)
    logger.info("Found %d PASS results to re-verify", len(pass_files))

    if not pass_files:
        logger.warning("No PASS results found in %s", results_dir)
        return

    # Re-verify each result
    all_results: list[dict] = []
    true_pass = 0
    false_pass = 0
    build_fail = 0
    run_fail = 0
    errors = 0

    start_time = time.time()

    for i, result_path in enumerate(pass_files, 1):
        rel_path = result_path.relative_to(results_dir)
        logger.info("[%d/%d] %s", i, len(pass_files), rel_path)

        result = reverify_one(
            result_path, specs_dir, project_root, verbose=args.verbose
        )
        all_results.append(result)

        status = result["reverify_status"]
        if status == "TRUE_PASS":
            true_pass += 1
            logger.info("  → TRUE_PASS (%s)", result.get("verify_strategy", "?"))
        elif status == "FALSE_PASS":
            false_pass += 1
            logger.warning(
                "  → FALSE_PASS (%s): %s",
                result.get("verify_strategy", "?"),
                result.get("verify_details", "")[:100],
            )
        elif status == "BUILD_FAIL":
            build_fail += 1
            logger.warning("  → BUILD_FAIL: %s", result.get("error", "")[:100])
        elif status == "RUN_FAIL":
            run_fail += 1
            logger.warning("  → RUN_FAIL: %s", result.get("error", "")[:100])
        else:
            errors += 1
            logger.error("  → %s: %s", status, result.get("error", "")[:100])

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("REVERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total PASS results re-verified: {len(pass_files)}")
    print(f"  TRUE_PASS  (correct output):  {true_pass}")
    print(f"  FALSE_PASS (wrong output):    {false_pass}")
    print(f"  BUILD_FAIL (on re-verify):    {build_fail}")
    print(f"  RUN_FAIL   (on re-verify):    {run_fail}")
    print(f"  ERROR      (script error):    {errors}")
    print(f"Time elapsed: {elapsed:.1f}s")

    if false_pass > 0:
        corrected_rate = true_pass / len(pass_files) * 100
        print(f"\nCorrected pass rate: {true_pass}/{len(pass_files)} ({corrected_rate:.1f}%)")

    # Per-model breakdown
    print("\n--- Per-model breakdown ---")
    by_model: dict[str, dict[str, int]] = {}
    for r in all_results:
        model = r["model"]
        status = r["reverify_status"]
        by_model.setdefault(model, {}).setdefault(status, 0)
        by_model[model][status] += 1

    for model in sorted(by_model):
        counts = by_model[model]
        total = sum(counts.values())
        tp = counts.get("TRUE_PASS", 0)
        fp = counts.get("FALSE_PASS", 0)
        bf = counts.get("BUILD_FAIL", 0)
        rf = counts.get("RUN_FAIL", 0)
        er = counts.get("ERROR", 0) + counts.get("SPEC_NOT_FOUND", 0)
        print(f"  {model}: {tp} TRUE_PASS, {fp} FALSE_PASS, {bf} BUILD_FAIL, {rf} RUN_FAIL, {er} ERROR (total={total})")

    # Per-kernel breakdown of FALSE_PASS
    if false_pass > 0:
        print("\n--- FALSE_PASS by kernel ---")
        fp_by_kernel: dict[str, list[str]] = {}
        for r in all_results:
            if r["reverify_status"] == "FALSE_PASS":
                kernel = r.get("target_spec", "unknown")
                fp_by_kernel.setdefault(kernel, []).append(r["model"])
        for kernel in sorted(fp_by_kernel):
            models = fp_by_kernel[kernel]
            print(f"  {kernel}: {len(models)} FALSE_PASS ({', '.join(sorted(set(models)))})")

    # Save detailed results
    audit_path = results_dir / "reverification_audit.json"
    audit_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_reverified": len(pass_files),
        "true_pass": true_pass,
        "false_pass": false_pass,
        "build_fail": build_fail,
        "run_fail": run_fail,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "by_model": by_model,
        "results": all_results,
    }
    audit_path.write_text(json.dumps(audit_data, indent=2))
    logger.info("Detailed results saved to %s", audit_path)

    print(f"\nResults saved to: {audit_path}")


if __name__ == "__main__":
    main()
