#!/usr/bin/env python3
"""
scripts/analysis/classify_translation_pairs.py

Classify all translation pairs (source spec → target spec) by complexity
and output a CSV for stratified reporting in the SC26 paper.

Complexity classes (docs/design/kernel_centric_translation.md §4):
    single_file      — 1 source → 1 target  (translation_targets count)
    multi_to_single  — N source → 1 target
    single_to_multi  — 1 source → 2+ targets
    multi_to_multi   — N source → M targets (N, M ≥ 2)

File counts come from files.translation_targets (or files.prompt_payload as fallback).

Output: results/evaluation/translation_complexity.csv

Usage:
    python3 scripts/analysis/classify_translation_pairs.py \\
        --project-root .
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_spec(spec_path: Path) -> dict | None:
    if not spec_path.exists():
        return None
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _translation_targets_count(spec: dict) -> int:
    """Return the number of translation target files for this spec.

    Prefers files.translation_targets; falls back to files.prompt_payload.
    """
    files = spec.get("files") or {}
    tt = files.get("translation_targets")
    if tt is not None:
        return len(tt)
    return len(files.get("prompt_payload", []))


def _classify(src_count: int, tgt_count: int) -> str:
    """Classify a (source, target) pair by file count."""
    if src_count == 1 and tgt_count == 1:
        return "single_file"
    elif src_count > 1 and tgt_count == 1:
        return "multi_to_single"
    elif src_count == 1 and tgt_count > 1:
        return "single_to_multi"
    else:
        return "multi_to_multi"


def classify_pairs(project_root: Path) -> list[dict]:
    """Build the classification table for all spec pairs."""
    manifest_path = project_root / "manifest.jsonl"
    specs_dir = project_root / "specs"

    manifest = _load_manifest(manifest_path)

    # Index: (source_suite, kernel_name, parallel_api) → spec_file
    index: dict[tuple[str, str, str], str] = {}
    for entry in manifest:
        key = (entry["source_suite"], entry["kernel_name"], entry["parallel_api"])
        # Last entry wins (manifest is append-only; later entries override earlier)
        if key not in index:
            index[key] = entry["spec_file"]

    # Collect all unique (suite, kernel) combinations with their available APIs
    kernel_apis: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in manifest:
        kernel_apis[(entry["source_suite"], entry["kernel_name"])].add(entry["parallel_api"])

    rows: list[dict] = []
    seen_pairs: set[tuple] = set()

    for (suite, kernel), apis in sorted(kernel_apis.items()):
        for src_api in sorted(apis):
            for tgt_api in sorted(apis):
                if src_api == tgt_api:
                    continue
                pair_key = (suite, kernel, src_api, tgt_api)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                src_key = (suite, kernel, src_api)
                tgt_key = (suite, kernel, tgt_api)

                if src_key not in index or tgt_key not in index:
                    continue

                src_spec_path = project_root / index[src_key]
                tgt_spec_path = project_root / index[tgt_key]

                src_spec = _load_spec(src_spec_path)
                tgt_spec = _load_spec(tgt_spec_path)

                if src_spec is None or tgt_spec is None:
                    continue

                src_count = _translation_targets_count(src_spec)
                tgt_count = _translation_targets_count(tgt_spec)
                complexity = _classify(src_count, tgt_count)

                rows.append({
                    "suite": suite,
                    "kernel": kernel,
                    "source_spec": src_spec_path.stem,
                    "target_spec": tgt_spec_path.stem,
                    "source_api": src_api,
                    "target_api": tgt_api,
                    "source_files": src_count,
                    "target_files": tgt_count,
                    "complexity_class": complexity,
                })

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite", "kernel", "source_spec", "target_spec",
        "source_api", "target_api", "source_files", "target_files", "complexity_class",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], suite_filter: str | None = None) -> None:
    """Print a summary table of pair counts by complexity class and direction."""
    from collections import Counter

    # Overall complexity counts
    print("\n=== Translation Pair Complexity Summary ===\n")
    by_class: Counter = Counter(r["complexity_class"] for r in rows)
    print(f"{'Class':<20} {'Pairs':>6}")
    print("-" * 28)
    for cls in ["single_file", "multi_to_single", "single_to_multi", "multi_to_multi"]:
        print(f"{cls:<20} {by_class.get(cls, 0):>6}")
    print(f"{'TOTAL':<20} {sum(by_class.values()):>6}")

    # By direction (src_api -> tgt_api), optionally filtered by suite
    suite_label = suite_filter.title() if suite_filter else "All Suites"
    print(f"\n=== By Direction ({suite_label}) ===\n")
    filtered_rows = [r for r in rows if r["suite"] == suite_filter] if suite_filter else rows
    direction_class: dict[str, Counter] = defaultdict(Counter)
    for r in filtered_rows:
        direction = f"{r['source_api']}→{r['target_api']}"
        direction_class[direction][r["complexity_class"]] += 1

    classes = ["single_file", "multi_to_single", "single_to_multi", "multi_to_multi"]
    header = f"{'Direction':<20}" + "".join(f"{c[:8]:>10}" for c in classes) + f"{'total':>8}"
    print(header)
    print("-" * (20 + 10 * len(classes) + 8))
    for direction in sorted(direction_class):
        counts = direction_class[direction]
        total = sum(counts.values())
        row_str = f"{direction:<20}"
        for c in classes:
            row_str += f"{counts.get(c, 0):>10}"
        row_str += f"{total:>8}"
        print(row_str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Path to parbench_sam root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results/evaluation/translation_complexity.csv)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Filter to a specific suite (default: all suites combined)",
    )
    args = parser.parse_args()

    output_path = args.output or (
        args.project_root / "results" / "evaluation" / "translation_complexity.csv"
    )

    rows = classify_pairs(args.project_root)
    write_csv(rows, output_path)
    print_summary(rows, suite_filter=args.suite)
    print(f"\nWrote {len(rows)} pairs to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
