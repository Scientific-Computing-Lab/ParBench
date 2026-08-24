#!/usr/bin/env python3
"""Verify both analysis paths produced identical direction-paired cells.

Task 6 (gate E-01) acceptance tool: statistical_analysis.py and
quantitative_findings.py must both consume the shared statistical unit in
paired_tasks.py, so the paired (model, suite, kernel) cells they report for
every direction pair must be identical. The two scripts pin different
"forward" members per pair (each to its own committed JSON), so cells are
canonicalized to the alphabetically first direction of the pair before
comparison.

Usage:
    python3 scripts/analysis/verify_direction_pairs.py \\
      --statistical <statistical_analysis.json> \\
      --quantitative <quantitative_findings.json> \\
      --expected-pairs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PairKey = tuple[str, str]
Cell = tuple[str, str, str, bool, bool]  # model, suite, kernel, first, second


def _canonical_cells(forward: str, reverse: str,
                     paired_cells: list[dict]) -> tuple[PairKey, list[Cell]]:
    """Orient each cell's passes to the alphabetically first direction."""
    key: PairKey = tuple(sorted([forward, reverse]))  # type: ignore[assignment]
    flip = forward != key[0]
    cells = [
        (
            c["model"], c["suite"], c["kernel"],
            c["reverse_pass"] if flip else c["forward_pass"],
            c["forward_pass"] if flip else c["reverse_pass"],
        )
        for c in paired_cells
    ]
    return key, sorted(cells)


def load_statistical(path: Path) -> dict[PairKey, list[Cell]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[PairKey, list[Cell]] = {}
    for entry in data["direction_asymmetry"]:
        key, cells = _canonical_cells(
            entry["forward_direction"], entry["reverse_direction"],
            entry["paired_cells"],
        )
        out[key] = cells
    return out


def load_quantitative(path: Path) -> dict[PairKey, list[Cell]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    asym = data["canonical"]["direction_asymmetry"]
    out: dict[PairKey, list[Cell]] = {}
    for finding in asym.values():
        value = finding["value"]
        key, cells = _canonical_cells(
            value["forward"], value["reverse"], value["paired_cells"],
        )
        out[key] = cells
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistical", type=Path, required=True)
    parser.add_argument("--quantitative", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    args = parser.parse_args(argv)

    stat = load_statistical(args.statistical)
    quant = load_quantitative(args.quantitative)

    problems: list[str] = []
    if set(stat) != set(quant):
        problems.append(
            f"pair sets differ: statistical={sorted(stat)} "
            f"quantitative={sorted(quant)}"
        )
    for key in sorted(set(stat) & set(quant)):
        if stat[key] != quant[key]:
            problems.append(
                f"paired cells differ for {key}:\n"
                f"  statistical:  {stat[key]}\n"
                f"  quantitative: {quant[key]}"
            )
    if len(stat) != args.expected_pairs:
        problems.append(
            f"expected {args.expected_pairs} direction pairs, "
            f"statistical has {len(stat)}"
        )
    if len(quant) != args.expected_pairs:
        problems.append(
            f"expected {args.expected_pairs} direction pairs, "
            f"quantitative has {len(quant)}"
        )

    n_cells = sum(len(v) for v in stat.values())
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        print(f"CHECK FAILED: {len(problems)} problem(s)")
        return 1
    print(f"OK: {len(stat)} direction pair(s), {n_cells} paired cell(s), "
          "identical across both analysis paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
