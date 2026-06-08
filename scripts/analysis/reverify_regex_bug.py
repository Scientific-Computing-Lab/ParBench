#!/usr/bin/env python3
"""Re-verify results affected by the regex combiner (?i) flag bug.

This script is READ-ONLY — it does NOT modify any result files.

It scans Qwen evaluation results for "Invalid regex" errors, rebuilds the
combined pattern using the fixed _wrap_pattern() logic, and checks whether
the stored run_stdout_snippet would match the corrected pattern.

Usage:
    python3 scripts/analysis/reverify_regex_bug.py --project-root /path/to/parbench
"""

import argparse
import json
import glob
import os
import re
import sys


def _wrap_pattern(p: str) -> str:
    """Wrap a pattern in a non-capturing group, preserving inline flags.

    Identical to the fix in llm_evaluate.py — duplicated here so this script
    is self-contained and can run without importing the eval pipeline.
    """
    m = re.match(r'^\(\?([aimsux]+)\)', p)
    if m:
        flags = m.group(1)
        rest = p[m.end():]
        return f"(?{flags}:{rest})"
    return f"(?:{p})"


def find_regex_error_results(results_dir: str) -> list[dict]:
    """Find result JSONs with 'Invalid regex' in error_message."""
    affected = []
    pattern = os.path.join(results_dir, "*.json")
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath) as f:
            try:
                result = json.load(f)
            except json.JSONDecodeError:
                continue

        error_msg = result.get("error_message", "") or ""
        if "Invalid regex" in error_msg or "global flags" in error_msg.lower():
            affected.append({"filepath": filepath, "result": result})

    return affected


def rebuild_combined_pattern(source_spec_path: str, target_spec_path: str,
                              project_root: str) -> str | None:
    """Rebuild the combined verify pattern using the fixed logic."""
    src_path = os.path.join(project_root, "specs", f"{source_spec_path}.json")
    tgt_path = os.path.join(project_root, "specs", f"{target_spec_path}.json")

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        return None

    with open(src_path) as f:
        source_spec = json.load(f)
    with open(tgt_path) as f:
        target_spec = json.load(f)

    source_strategies = (source_spec.get("verification") or {}).get("strategies", [])
    target_strategies = (target_spec.get("verification") or {}).get("strategies", [])

    source_patterns = [
        s["pattern"] for s in source_strategies
        if s.get("type") == "stdout_pattern" and s.get("pattern")
    ]
    target_patterns = [
        s["pattern"] for s in target_strategies
        if s.get("type") == "stdout_pattern" and s.get("pattern")
    ]

    all_patterns = source_patterns + [p for p in target_patterns if p not in source_patterns]
    if not all_patterns:
        return None

    return "|".join(_wrap_pattern(p) for p in all_patterns)


def main():
    parser = argparse.ArgumentParser(description="Re-verify regex bug affected results (read-only)")
    parser.add_argument("--project-root", required=True, help="Path to project root")
    parser.add_argument("--model-dir", default="together-qwen-3.5-397b-a17b",
                        help="Model results directory name")
    args = parser.parse_args()

    results_dir = os.path.join(args.project_root, "results", "evaluation", args.model_dir)
    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {results_dir}")
    print()

    affected = find_regex_error_results(results_dir)
    print(f"Found {len(affected)} results with regex errors")
    print()

    flips = []
    still_fail = []

    for item in affected:
        result = item["result"]
        filepath = item["filepath"]
        filename = os.path.basename(filepath)

        source_spec = result.get("source_spec", "")
        target_spec = result.get("target_spec", "")
        stdout_snippet = result.get("run_stdout_snippet", "") or ""

        # Rebuild combined pattern with fix
        fixed_pattern = rebuild_combined_pattern(source_spec, target_spec, args.project_root)

        if fixed_pattern is None:
            print(f"  SKIP {filename}: could not rebuild pattern")
            still_fail.append({"file": filename, "reason": "pattern rebuild failed"})
            continue

        try:
            compiled = re.compile(fixed_pattern)
        except re.error as e:
            print(f"  ERROR {filename}: fixed pattern still invalid: {e}")
            still_fail.append({"file": filename, "reason": f"pattern still invalid: {e}"})
            continue

        match = compiled.search(stdout_snippet)
        status = result.get("overall_status", "UNKNOWN")

        if match:
            print(f"  FLIP  {filename}: {status} -> would PASS (matched: {match.group()!r})")
            flips.append({
                "file": filename,
                "source": source_spec,
                "target": target_spec,
                "old_status": status,
                "matched": match.group(),
            })
        else:
            reason = "no stdout" if not stdout_snippet else "pattern didn't match"
            print(f"  STILL {filename}: {status} ({reason})")
            still_fail.append({
                "file": filename,
                "source": source_spec,
                "target": target_spec,
                "reason": reason,
                "stdout_preview": stdout_snippet[:100] if stdout_snippet else "(empty)",
            })

    print()
    print("=" * 60)
    print(f"SUMMARY: {len(flips)} would flip to PASS, {len(still_fail)} still fail")
    print()

    if flips:
        print("Would flip to PASS:")
        for f in flips:
            print(f"  {f['file']}: {f['source']} -> {f['target']} (matched: {f['matched']!r})")

    if still_fail:
        print()
        print("Still fail:")
        for f in still_fail:
            print(f"  {f['file']}: {f.get('reason', 'unknown')}")

    print()
    print("NOTE: This is a READ-ONLY analysis. No result files were modified.")


if __name__ == "__main__":
    main()
