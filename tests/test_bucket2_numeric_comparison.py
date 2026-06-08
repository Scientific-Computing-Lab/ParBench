"""S7c Stage 3: bucket2 mis-labeled-strong specs upgraded to numeric_comparison.

Five specs carried `oracle_strength: "strong"` but only stdout_pattern + exit_code
strategies — effectively weak. S7b audit follow-up §1 deferred fixing this to
post-NeurIPS; S7c brings the fix forward per paper-explorer advisory.

Policy: B-per-spec (tolerance matched to the printed quantity's semantics).

| Spec | expected | tolerance | baseline captured 2026-04-19 |
|---|---|---|---|
| rodinia-hotspot3d-cuda | 4.096975e-05 | 1e-4 abs | 3/3 deterministic runs |
| rodinia-hotspot3d-omp | 4.096975e-05 | 1e-4 abs | 3/3 deterministic runs |
| rodinia-hotspot3d-opencl | 5.547907e-05 | 1e-4 abs | 3/3 deterministic runs |
| hecbench-md-cuda | 1.86265e-09 | 1e-8 abs | 3/3 deterministic runs |
| hecbench-md-omp_target | 0.0 | 1e-9 abs | 3/3 deterministic runs |

NOTE: hotspot3d Accuracy is an RMS residual (see rodinia-src/cuda/hotspot3D/3D.cu:122-132);
lower is better, 0 = perfect. Original Policy B-per-spec tolerance 1e-7 abs was tightened
to 1e-4 abs in S7c post-review to avoid rejecting strictly-better translations (Accuracy=0
must PASS). Widened window [0, baseline+1e-4] still rejects gross errors (Accuracy >= 1e-3).

`oracle_strength` upgraded from `"strong"` (mis-label) → `"medium"` (bucket2 tier).
Existing `stdout_pattern + exit_code` strategies retained; numeric_comparison
conjoins (all must PASS).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


SPEC_DIR = Path(__file__).resolve().parent.parent / "specs"

BUCKET2_UPGRADES = {
    "rodinia-hotspot3d-cuda.json": {
        "expected": 4.096975e-05,
        "tolerance": 1e-4,
        "extract_regex": r"Accuracy:\s+([+-]?\d[\d.eE+-]*)",
    },
    "rodinia-hotspot3d-omp.json": {
        "expected": 4.096975e-05,
        "tolerance": 1e-4,
        "extract_regex": r"Accuracy:\s+([+-]?\d[\d.eE+-]*)",
    },
    "rodinia-hotspot3d-opencl.json": {
        "expected": 5.547907e-05,
        "tolerance": 1e-4,
        "extract_regex": r"Accuracy:\s+([+-]?\d[\d.eE+-]*)",
    },
    "hecbench-md-cuda.json": {
        "expected": 1.86265e-09,
        "tolerance": 1e-8,
        "extract_regex": r"Max error between host and device:\s+([+-]?\d[\d.eE+-]*)",
    },
    "hecbench-md-omp_target.json": {
        "expected": 0.0,
        "tolerance": 1e-9,
        "extract_regex": r"Max error between host and device:\s+([+-]?\d[\d.eE+-]*)",
    },
}


def _load_strategies(spec_filename: str) -> list[dict]:
    spec = json.loads((SPEC_DIR / spec_filename).read_text())
    return spec["verification"]["strategies"]


def _load_oracle_strength(spec_filename: str) -> str | None:
    spec = json.loads((SPEC_DIR / spec_filename).read_text())
    return spec["verification"].get("oracle_strength")


@pytest.mark.parametrize("spec_filename", sorted(BUCKET2_UPGRADES.keys()))
def test_numeric_comparison_present(spec_filename):
    """S7c: each bucket2 spec has a numeric_comparison strategy."""
    strategies = _load_strategies(spec_filename)
    types = [s["type"] for s in strategies]
    assert "numeric_comparison" in types, (
        f"{spec_filename} missing numeric_comparison (has {types})"
    )


@pytest.mark.parametrize("spec_filename", sorted(BUCKET2_UPGRADES.keys()))
def test_numeric_comparison_expected_is_finite(spec_filename):
    """S7c: numeric_comparison.expected is a finite float (not NaN, not inf)."""
    strategies = _load_strategies(spec_filename)
    nc = next(s for s in strategies if s["type"] == "numeric_comparison")
    assert isinstance(nc["expected"], (int, float))
    assert math.isfinite(float(nc["expected"]))
    # Exact locked value per Policy B-per-spec.
    assert float(nc["expected"]) == pytest.approx(BUCKET2_UPGRADES[spec_filename]["expected"])


@pytest.mark.parametrize("spec_filename", sorted(BUCKET2_UPGRADES.keys()))
def test_numeric_comparison_tolerance_is_positive_and_locked(spec_filename):
    """S7c: numeric_comparison.tolerance > 0 and matches Policy B-per-spec."""
    strategies = _load_strategies(spec_filename)
    nc = next(s for s in strategies if s["type"] == "numeric_comparison")
    assert nc["tolerance"] > 0
    assert float(nc["tolerance"]) == pytest.approx(BUCKET2_UPGRADES[spec_filename]["tolerance"])


@pytest.mark.parametrize("spec_filename", sorted(BUCKET2_UPGRADES.keys()))
def test_numeric_comparison_extract_regex_matches_captured_stdout(spec_filename):
    """S7c: extract_regex is the documented bucket2 pattern with a capture group."""
    strategies = _load_strategies(spec_filename)
    nc = next(s for s in strategies if s["type"] == "numeric_comparison")
    assert nc["extract_regex"] == BUCKET2_UPGRADES[spec_filename]["extract_regex"]


@pytest.mark.parametrize("spec_filename", sorted(BUCKET2_UPGRADES.keys()))
def test_oracle_strength_is_medium(spec_filename):
    """S7c: bucket2 specs are `oracle_strength: "medium"` post-upgrade.

    "strong" is reserved for file_hash + reference_files (bptree pair only).
    """
    strength = _load_oracle_strength(spec_filename)
    assert strength == "medium", (
        f"{spec_filename} oracle_strength is {strength!r}, expected 'medium'"
    )


@pytest.mark.parametrize("spec_filename", sorted(BUCKET2_UPGRADES.keys()))
def test_stdout_pattern_retained(spec_filename):
    """S7c: existing stdout_pattern strategy is retained (conjunctive verification)."""
    strategies = _load_strategies(spec_filename)
    types = [s["type"] for s in strategies]
    assert "stdout_pattern" in types, (
        f"{spec_filename} must retain stdout_pattern alongside numeric_comparison"
    )
