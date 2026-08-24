#!/usr/bin/env python3
"""Shared statistical unit for direction-paired analyses (Task 6, gate E-01).

One task cell is the (model, suite, kernel, direction) tuple. The submitted
direction-asymmetry analyses keyed their pass lookups without ``sample_id``,
so with three stochastic L0 samples per cell the dict was overwritten twice
and only the last-loaded sample survived. The corrected rule, disclosed in
the rebuttal, collapses all L0 samples of one cell by any-sample success
(pass@1-of-any, the metric already reported in eval_summary.md) and pairs
the two directions of a direction pair only within the same
(model, suite, kernel).

Every field is derived here, from the record itself, so all consumers get
identical cells regardless of how their loaders enriched the records.
Iteration orders are pinned by sorting, so output is byte-identical across
``PYTHONHASHSEED`` values.

Consumers: scripts/analysis/statistical_analysis.py,
scripts/analysis/quantitative_findings.py, and
scripts/rebuttal/suite_composition.py. Do not add a second aggregation
implementation; extend this one.
"""

from __future__ import annotations

CellKey = tuple[str, str, str, str]  # model, suite, kernel, direction


def suite_from_spec(spec_name: str) -> str:
    """Extract suite from a spec ID like 'rodinia-bfs-cuda' -> 'rodinia'."""
    parts = spec_name.split("-")
    return parts[0] if parts else spec_name


def kernel_from_spec(spec_name: str) -> str:
    """Extract kernel from a spec ID like 'rodinia-bfs-cuda' -> 'bfs'."""
    parts = spec_name.split("-")
    if len(parts) < 3:
        return spec_name
    return "-".join(parts[1:-1])


def _direction(record: dict) -> str:
    d = record.get("direction")
    if d:
        return d
    src = record.get("source_spec", "")
    tgt = record.get("target_spec", "")
    src_api = src.rsplit("-", 1)[-1] if src else "unknown"
    tgt_api = tgt.rsplit("-", 1)[-1] if tgt else "unknown"
    return f"{src_api}-to-{tgt_api}"


def cell_key(record: dict) -> CellKey:
    """The (model, suite, kernel, direction) task cell a record belongs to."""
    src = record.get("source_spec", "")
    return (
        record.get("model") or "unknown",
        suite_from_spec(src),
        kernel_from_spec(src),
        _direction(record),
    )


def _is_l0(record: dict) -> bool:
    return (record.get("augment_level") or 0) == 0


def collapse_l0_any_success(records: list[dict]) -> dict[CellKey, bool]:
    """Collapse all L0 samples in one task cell by any-sample success."""
    cells: dict[CellKey, bool] = {}
    for r in records:
        if not _is_l0(r):
            continue
        key = cell_key(r)
        cells[key] = cells.get(key, False) or (r.get("overall_status") == "PASS")
    return cells


def l0_cell_sample_counts(records: list[dict]) -> dict[CellKey, int]:
    """Number of L0 samples per task cell (display fact, not aggregation)."""
    counts: dict[CellKey, int] = {}
    for r in records:
        if not _is_l0(r):
            continue
        key = cell_key(r)
        counts[key] = counts.get(key, 0) + 1
    return counts


def paired_direction_cell_details(
    collapsed: dict[CellKey, bool], forward: str, reverse: str
) -> list[dict]:
    """One entry per (model, suite, kernel) present in BOTH directions.

    Sorted by (model, suite, kernel) so serialized output is deterministic.
    """
    details: list[dict] = []
    for (model, suite, kernel, direction) in sorted(collapsed):
        if direction != forward:
            continue
        rev = collapsed.get((model, suite, kernel, reverse))
        if rev is None:
            continue
        details.append({
            "model": model,
            "suite": suite,
            "kernel": kernel,
            "forward_pass": collapsed[(model, suite, kernel, forward)],
            "reverse_pass": rev,
        })
    return details


def paired_direction_cells(
    collapsed: dict[CellKey, bool], forward: str, reverse: str
) -> list[tuple[bool, bool]]:
    """Pair only within the same model, suite, and kernel."""
    return [
        (d["forward_pass"], d["reverse_pass"])
        for d in paired_direction_cell_details(collapsed, forward, reverse)
    ]
