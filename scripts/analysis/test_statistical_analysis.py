#!/usr/bin/env python3
"""
Tests for scripts/analysis/statistical_analysis.py

Covers:
  - Issue 1: pass@k must exclude deterministic (temp=0.0) records
  - Issue 2: Cochran-Armitage must use balanced groups (deterministic only,
             tasks present at ALL levels)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.statistical_analysis import (
    compute_augmentation_trends,
    compute_pass_at_k_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    kernel: str = "bfs",
    model: str = "test-model",
    direction: str = "cuda-to-omp",
    level: int = 0,
    status: str = "PASS",
    temperature: float = 0.0,
    sample_id: int | str | None = 0,
) -> dict:
    """Build a minimal eval record for testing."""
    return {
        "source_spec": f"rodinia-{kernel}-cuda",
        "model": model,
        "direction": direction,
        "augment_level": level,
        "overall_status": status,
        "temperature": temperature,
        "sample_id": sample_id,
    }


# ---------------------------------------------------------------------------
# Issue 1: pass@k stochastic-only filter
# ---------------------------------------------------------------------------

class TestPassAtKStochasticFilter:
    """pass@k must use only stochastic (temp > 0) samples, not deterministic."""

    def test_pass_at_k_excludes_deterministic(self):
        """Given 1 deterministic + 3 stochastic records for same task,
        n must be 3 (stochastic only), not 4."""
        records = [
            # Deterministic run (temp=0.0) -- should be EXCLUDED from pass@k
            _make_record(kernel="bfs", temperature=0.0, sample_id=0, status="PASS"),
            # 3 stochastic runs (temp=0.7) -- these are the i.i.d. samples
            _make_record(kernel="bfs", temperature=0.7, sample_id="s0", status="PASS"),
            _make_record(kernel="bfs", temperature=0.7, sample_id="s1", status="PASS"),
            _make_record(kernel="bfs", temperature=0.7, sample_id="s2", status="FAIL"),
        ]
        table = compute_pass_at_k_table(records, k_values=[1, 2, 3])
        # There should be exactly one group key (bfs, test-model, cuda-to-omp, 0)
        assert len(table) == 1, f"Expected 1 group, got {len(table)}"
        entry = list(table.values())[0]
        assert entry["n"] == 3, f"Expected n=3 (stochastic only), got n={entry['n']}"
        assert entry["c"] == 2, f"Expected c=2, got c={entry['c']}"

    def test_pass_at_k_all_deterministic_returns_empty(self):
        """If there are no stochastic records, pass@k table should be empty."""
        records = [
            _make_record(kernel="bfs", temperature=0.0, sample_id=0, status="PASS"),
            _make_record(kernel="nw", temperature=0.0, sample_id=0, status="FAIL"),
        ]
        table = compute_pass_at_k_table(records, k_values=[1])
        assert len(table) == 0, f"Expected empty table for deterministic-only data, got {len(table)} entries"

    def test_pass_at_k_multiple_kernels(self):
        """Each kernel's stochastic samples should be grouped independently."""
        records = [
            # bfs: 1 deterministic + 2 stochastic (1 PASS, 1 FAIL)
            _make_record(kernel="bfs", temperature=0.0, sample_id=0, status="PASS"),
            _make_record(kernel="bfs", temperature=0.7, sample_id="s0", status="PASS"),
            _make_record(kernel="bfs", temperature=0.7, sample_id="s1", status="FAIL"),
            # nw: 1 deterministic + 2 stochastic (2 PASS)
            _make_record(kernel="nw", temperature=0.0, sample_id=0, status="FAIL"),
            _make_record(kernel="nw", temperature=0.7, sample_id="s0", status="PASS"),
            _make_record(kernel="nw", temperature=0.7, sample_id="s1", status="PASS"),
        ]
        table = compute_pass_at_k_table(records, k_values=[1, 2])
        assert len(table) == 2, f"Expected 2 groups, got {len(table)}"
        for entry in table.values():
            assert entry["n"] == 2, f"Expected n=2 per kernel, got n={entry['n']}"


# ---------------------------------------------------------------------------
# Issue 2: Cochran-Armitage balanced groups
# ---------------------------------------------------------------------------

class TestAugmentationTrendsBalanced:
    """Cochran-Armitage must filter to deterministic only and balanced groups."""

    def test_augmentation_trends_excludes_stochastic(self):
        """Stochastic records (temp > 0) should NOT be counted in trend test."""
        records = []
        # L0: 5 deterministic PASS + 5 stochastic FAIL (should be excluded)
        for i in range(5):
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.0, status="PASS",
            ))
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.7,
                sample_id=f"s{i}", status="FAIL",
            ))
        # L1: 5 deterministic PASS (no stochastic)
        for i in range(5):
            records.append(_make_record(
                kernel=f"k{i}", level=1, temperature=0.0, status="PASS",
            ))

        result = compute_augmentation_trends(records)
        overall = result["overall"]

        # Without fix: L0 has 10 total (5 pass + 5 fail = 50%),
        #              L1 has 5 total (5 pass = 100%) -> spurious trend
        # With fix: L0 has 5 total (5 pass = 100%),
        #           L1 has 5 total (5 pass = 100%) -> no trend
        assert overall["total_counts"][0] == overall["total_counts"][1], (
            f"L0 total ({overall['total_counts'][0]}) != L1 total ({overall['total_counts'][1]}); "
            "stochastic records are leaking into the trend test"
        )

    def test_augmentation_trends_balanced_groups(self):
        """Only tasks present at ALL levels should be included (balanced design)."""
        records = []
        # k0-k4: present at both L0 and L1 (balanced)
        for i in range(5):
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.0, status="PASS",
            ))
            records.append(_make_record(
                kernel=f"k{i}", level=1, temperature=0.0, status="PASS",
            ))
        # k5-k9: present at L0 ONLY (unbalanced -- should be excluded)
        for i in range(5, 10):
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.0, status="FAIL",
            ))

        result = compute_augmentation_trends(records)
        overall = result["overall"]

        # Without fix: L0 has 10 total (5 pass/10 = 50%), L1 has 5 total (100%)
        #              -> spurious "increasing" trend from unbalanced groups
        # With fix: L0 has 5 total (100%), L1 has 5 total (100%)
        #           -> balanced, no trend
        assert overall["total_counts"][0] == overall["total_counts"][1], (
            f"L0 total ({overall['total_counts'][0]}) != L1 total ({overall['total_counts'][1]}); "
            "unbalanced tasks are leaking into the trend test"
        )
        assert overall["trend_direction"] != "increasing", (
            "Trend should not be 'increasing' when groups are balanced with equal pass rates"
        )

    def test_augmentation_trends_combined_confound(self):
        """Combined scenario: stochastic dilution + unbalanced groups at L0."""
        records = []
        # k0-k2: present at L0, L1, L2 (balanced), deterministic PASS
        for i in range(3):
            for lv in [0, 1, 2]:
                records.append(_make_record(
                    kernel=f"k{i}", level=lv, temperature=0.0, status="PASS",
                ))
        # L0 stochastic noise: 6 extra FAIL records at L0 only
        for i in range(3):
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.7,
                sample_id=f"s{i}a", status="FAIL",
            ))
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.7,
                sample_id=f"s{i}b", status="FAIL",
            ))
        # k3-k4: present ONLY at L0, deterministic FAIL (unbalanced)
        for i in range(3, 5):
            records.append(_make_record(
                kernel=f"k{i}", level=0, temperature=0.0, status="FAIL",
            ))

        result = compute_augmentation_trends(records)
        overall = result["overall"]

        # With fix: only k0-k2 at L0/L1/L2, deterministic only, all PASS
        # total_counts should be [3, 3, 3]
        for lv_idx, tc in enumerate(overall["total_counts"]):
            assert tc == 3, (
                f"Level {overall['levels'][lv_idx]}: expected total=3, got {tc}"
            )


# --- new tests: predicate extraction ---

def test_is_deterministic_zero():
    from scripts.analysis.statistical_analysis import is_deterministic
    assert is_deterministic({"temperature": 0.0}) is True

def test_is_deterministic_missing_key():
    from scripts.analysis.statistical_analysis import is_deterministic
    assert is_deterministic({}) is True  # missing key defaults to 0.0

def test_is_stochastic_nonzero():
    from scripts.analysis.statistical_analysis import is_stochastic
    assert is_stochastic({"temperature": 0.7}) is True

def test_is_stochastic_zero():
    from scripts.analysis.statistical_analysis import is_stochastic
    assert is_stochastic({"temperature": 0.0}) is False

def test_constants_exist():
    import scripts.analysis.statistical_analysis as sa
    assert sa.MIN_EXPECTED_CELL_COUNT == 5
    assert sa.MCNEMAR_EXACT_THRESHOLD == 25
    assert sa.SAMPLE_SIZE_ADEQUACY_THRESHOLD == 10

def test_get_direction_pairs_valid():
    from scripts.analysis.statistical_analysis import _get_direction_pairs
    directions = {"cuda-to-omp", "omp-to-cuda", "cuda-to-opencl"}
    pair = _get_direction_pairs("cuda-to-omp", directions)
    assert pair == ("cuda-to-omp", "omp-to-cuda")

def test_get_direction_pairs_no_reverse():
    from scripts.analysis.statistical_analysis import _get_direction_pairs
    directions = {"cuda-to-omp"}
    assert _get_direction_pairs("cuda-to-omp", directions) is None
