"""Plan 02-10 Step 3 (A2, A4): dual-metric pass@1 reporting tests.

Verifies that `_compute_cell_pass_metrics()` correctly groups records by
(kernel, direction, level, model) cells, reports both pass@1_of_any and
pass@1_mean per cell, and aggregates per-level stats showing divergence.
Also verifies that `build_markdown()` emits the level-invariance scope
note and the dual-metric table.
"""
from __future__ import annotations

from scripts.evaluation.analyze_eval import (
    _compute_cell_pass_metrics,
    build_markdown,
    build_summary,
)


def _rec(kernel: str, direction: str, level: int, model: str,
         sample_id: int, status: str) -> dict:
    """Minimal per-task record shape for these tests."""
    api_src, api_tgt = direction.split("-to-")
    return {
        "source_spec": f"rodinia-{kernel}-{api_src}",
        "target_spec": f"rodinia-{kernel}-{api_tgt}",
        "kernel": kernel,
        "direction": direction,
        "augment_level": level,
        "model": model,
        "sample_id": sample_id,
        "overall_status": status,
    }


# ---------------------------------------------------------------------------
# _compute_cell_pass_metrics: per-cell structure
# ---------------------------------------------------------------------------

def test_cell_all_pass_both_metrics_equal_1():
    records = [_rec("bfs", "cuda-to-omp", 0, "m1", i, "PASS") for i in range(3)]
    out = _compute_cell_pass_metrics(records)
    assert len(out["cells"]) == 1
    cell = out["cells"][0]
    assert cell["samples"] == 3
    assert cell["passes"] == 3
    assert cell["pass_at_1_of_any"] == 1
    assert cell["pass_at_1_mean"] == 1.0


def test_cell_all_fail_both_metrics_equal_0():
    records = [_rec("bfs", "cuda-to-omp", 0, "m1", i, "BUILD_FAIL") for i in range(3)]
    out = _compute_cell_pass_metrics(records)
    cell = out["cells"][0]
    assert cell["pass_at_1_of_any"] == 0
    assert cell["pass_at_1_mean"] == 0.0


def test_cell_interior_rate_divergence():
    """At 1/3 and 2/3, of_any=1 but mean < 1 — structural divergence."""
    records = [
        _rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 1, "BUILD_FAIL"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 2, "BUILD_FAIL"),
    ]
    out = _compute_cell_pass_metrics(records)
    cell = out["cells"][0]
    assert cell["pass_at_1_of_any"] == 1
    assert cell["pass_at_1_mean"] == round(1 / 3, 4)


def test_multiple_cells_grouped_correctly():
    records = [
        _rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 1, "PASS"),
        _rec("nw", "cuda-to-omp", 0, "m1", 0, "BUILD_FAIL"),
        _rec("bfs", "cuda-to-omp", 1, "m1", 0, "PASS"),
    ]
    out = _compute_cell_pass_metrics(records)
    assert len(out["cells"]) == 3  # (bfs,L0), (nw,L0), (bfs,L1)


def test_by_level_aggregate_divergence_reported():
    records = [
        # L0: one cell 1/3 (interior) + one cell 3/3 (pole)
        _rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 1, "BUILD_FAIL"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 2, "BUILD_FAIL"),
        _rec("nw", "cuda-to-omp", 0, "m1", 0, "PASS"),
        _rec("nw", "cuda-to-omp", 0, "m1", 1, "PASS"),
        _rec("nw", "cuda-to-omp", 0, "m1", 2, "PASS"),
    ]
    out = _compute_cell_pass_metrics(records)
    l0 = out["by_level"]["L0"]
    assert l0["cells"] == 2
    # of_any averages 1 and 1 → 1.0
    assert l0["mean_of_any"] == 1.0
    # mean averages 1/3 and 1.0 → 2/3 ≈ 0.6667
    assert l0["mean_of_mean"] == round((1 / 3 + 1.0) / 2, 4)
    # divergence = of_any_mean - mean_of_mean > 0
    assert l0["divergence"] > 0


def test_cells_sorted_stably():
    records = [
        _rec("nw", "cuda-to-omp", 0, "m1", 0, "PASS"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS"),
    ]
    out = _compute_cell_pass_metrics(records)
    # Sorted by (model, direction, level, kernel) — bfs before nw
    assert out["cells"][0]["kernel"] == "bfs"
    assert out["cells"][1]["kernel"] == "nw"


# ---------------------------------------------------------------------------
# build_summary + build_markdown integration
# ---------------------------------------------------------------------------

def test_build_summary_includes_pass_at_1_key():
    records = [_rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS")]
    summary = build_summary(records, complexity_lookup=None)
    assert "pass_at_1" in summary
    assert "cells" in summary["pass_at_1"]
    assert "by_level" in summary["pass_at_1"]


def test_build_markdown_has_level_invariance_scope_note():
    records = [_rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS")]
    summary = build_summary(records, complexity_lookup=None)
    md = build_markdown(summary, records, complexity_lookup=None)
    assert "Scope note (level-invariance claim)" in md
    assert "L0-passer subset" in md
    assert "02-THREATS-TO-VALIDITY.md" in md


def test_build_markdown_has_dual_metric_table():
    records = [_rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS")]
    summary = build_summary(records, complexity_lookup=None)
    md = build_markdown(summary, records, complexity_lookup=None)
    assert "pass@1 Metrics (Dual Reporting)" in md
    assert "pass@1_of_any" in md
    assert "pass@1_mean" in md
    assert "divergence" in md


def test_build_markdown_dual_metric_table_row_counts():
    records = [
        _rec("bfs", "cuda-to-omp", 0, "m1", 0, "PASS"),
        _rec("bfs", "cuda-to-omp", 0, "m1", 1, "BUILD_FAIL"),
        _rec("bfs", "cuda-to-omp", 1, "m1", 0, "PASS"),
    ]
    summary = build_summary(records, complexity_lookup=None)
    md = build_markdown(summary, records, complexity_lookup=None)
    # Must have both L0 and L1 rows in the dual-metric table
    dual_section_start = md.index("pass@1 Metrics (Dual Reporting)")
    dual_section = md[dual_section_start:]
    assert "| L0 |" in dual_section
    assert "| L1 |" in dual_section
