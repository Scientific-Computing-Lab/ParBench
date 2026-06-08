#!/usr/bin/env python3
"""Tests for augmentation_analysis.py -- validates AUG-01 and AUG-02.

Tests verify the per-kernel x per-level augmentation matrix by independently
loading raw result JSONs and comparing against the script's output.

Test organization by requirement:
  - AUG-01 structure:  test_matrix_structure, test_kernel_count_matches_disk,
                       test_status_values_match_raw, test_stochastic_excluded,
                       test_omp_target_excluded_from_cuda_to_omp,
                       test_known_fail_flagged, test_aggregate_has_wilson_ci,
                       test_secondary_matrix_directions
  - AUG-02 patterns:   test_pattern_classification, test_md_summary_contains_sections
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from augmentation_analysis import _extract_api  # noqa: E402
from augmentation_analysis import _consensus_status  # noqa: E402
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation" / "together-qwen-3.5-397b-a17b"
OUTPUT_JSON = PROJECT_ROOT / "results" / "analysis" / "augmentation_per_kernel_matrix_together-qwen-3.5-397b-a17b.json"
OUTPUT_MD = PROJECT_ROOT / "results" / "analysis" / "augmentation_per_kernel_matrix_together-qwen-3.5-397b-a17b.md"


# ---------------------------------------------------------------------------
# Helpers -- independent ground truth from raw data
# ---------------------------------------------------------------------------

def _load_output() -> dict:
    """Load the output JSON, skipping if it does not exist."""
    if not OUTPUT_JSON.exists():
        pytest.skip("Run augmentation_analysis.py first")
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))


def _count_cuda_to_omp_kernels_on_disk() -> int:
    """Independently count unique cuda-to-omp kernel pairs from raw files.

    Includes stochastic seed files (L0) since the matrix now uses consensus.
    """
    seen = set()
    for f in RESULTS_DIR.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        src = data.get("source_spec", "")
        tgt = data.get("target_spec", "")
        src_api = _extract_api(src)
        tgt_api = _extract_api(tgt)
        if src_api == "cuda" and tgt_api == "omp":
            kernel = data.get("kernel", "unknown")
            seen.add(kernel)
    return len(seen)


def _load_raw_status(kernel_name: str, level: int) -> str:
    """Load overall_status for a specific kernel + level from raw JSON.

    For L0: collects all seed results and applies consensus (PASS if any passes).
    For L1+: returns single result (skips stochastic files).
    """
    if level == 0:
        statuses = []
        for f in RESULTS_DIR.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            src_api = _extract_api(data.get("source_spec", ""))
            tgt_api = _extract_api(data.get("target_spec", ""))
            if src_api != "cuda" or tgt_api != "omp":
                continue
            if data.get("kernel") != kernel_name:
                continue
            aug = data.get("augment_level", 0)
            if aug == 0:
                statuses.append(data.get("overall_status", "UNKNOWN"))
        if statuses:
            return _consensus_status(statuses)
        return "NOT_FOUND"

    for f in RESULTS_DIR.glob("*.json"):
        stem = f.stem
        if re.search(r"-s\d+$", stem):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        src_api = _extract_api(data.get("source_spec", ""))
        tgt_api = _extract_api(data.get("target_spec", ""))
        if src_api != "cuda" or tgt_api != "omp":
            continue
        if data.get("kernel") != kernel_name:
            continue
        aug = data.get("augment_level", 0)
        if aug == level:
            return data.get("overall_status", "UNKNOWN")
    return "NOT_FOUND"


# ---------------------------------------------------------------------------
# AUG-01 Structure Tests
# ---------------------------------------------------------------------------


class TestMatrixStructure:
    """Verify the JSON output has the required top-level and nested structure."""

    def test_matrix_structure(self) -> None:
        data = _load_output()
        # Top-level keys
        assert "generated_at" in data
        assert "data_source" in data
        assert "primary_matrix" in data
        assert "secondary_matrix" in data

        pm = data["primary_matrix"]
        assert pm["direction"] == "cuda-to-omp"
        assert pm["levels"] == ["L0", "L1", "L2", "L3", "L4"]
        assert pm["kernel_count"] == 26
        assert isinstance(pm["per_kernel"], dict)
        assert isinstance(pm["aggregate"], dict)
        assert isinstance(pm["pattern_summary"], dict)
        assert isinstance(pm["exceptions"], list)

    def test_kernel_count_matches_disk(self) -> None:
        data = _load_output()
        disk_count = _count_cuda_to_omp_kernels_on_disk()
        assert data["primary_matrix"]["kernel_count"] == disk_count

    def test_status_values_match_raw(self) -> None:
        """Spot-check kernels at L0 (consensus) and L2 (augmentation) against raw data."""
        data = _load_output()
        pm = data["primary_matrix"]["per_kernel"]

        # L0 consensus check for all kernels
        for kernel in ["backprop", "bfs", "hotspot"]:
            raw = _load_raw_status(kernel, 0)
            matrix_val = pm[kernel]["L0"]
            assert matrix_val == raw, (
                f"{kernel} L0: matrix={matrix_val}, raw={raw}"
            )

        # L2 augmentation check (only kernels with augmentation data)
        for kernel in ["bfs"]:
            raw = _load_raw_status(kernel, 2)
            matrix_val = pm[kernel]["L2"]
            assert matrix_val == raw, (
                f"{kernel} L2: matrix={matrix_val}, raw={raw}"
            )

    def test_stochastic_excluded(self) -> None:
        """No kernel should have more than 5 level entries (L0-L4 only)."""
        data = _load_output()
        pm = data["primary_matrix"]["per_kernel"]
        for kernel, entry in pm.items():
            level_keys = [k for k in entry if k.startswith("L") and k[1:].isdigit()]
            assert len(level_keys) <= 5, (
                f"{kernel} has {len(level_keys)} level entries: {level_keys}"
            )

    def test_omp_target_excluded_from_cuda_to_omp(self) -> None:
        """No kernel in primary_matrix has omp_target in its target spec name."""
        data = _load_output()
        pm = data["primary_matrix"]["per_kernel"]
        for kernel, entry in pm.items():
            tgt = entry.get("target_spec", "")
            assert "omp_target" not in tgt, (
                f"{kernel} target_spec contains omp_target: {tgt}"
            )

    def test_known_fail_flagged(self) -> None:
        """kmeans and mummergpu should have known_fail=True."""
        data = _load_output()
        pm = data["primary_matrix"]["per_kernel"]
        assert pm["kmeans"]["known_fail"] is True
        assert pm["mummergpu"]["known_fail"] is True

    def test_aggregate_has_wilson_ci(self) -> None:
        """Each level in aggregate has rate, ci_lower, ci_upper, n keys."""
        data = _load_output()
        agg = data["primary_matrix"]["aggregate"]
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            entry = agg[level]
            for key in ["rate", "ci_lower", "ci_upper", "n"]:
                assert key in entry, f"aggregate[{level}] missing key '{key}'"

    def test_secondary_matrix_directions(self) -> None:
        """secondary_matrix contains per_direction_aggregate with at least 7 directions."""
        data = _load_output()
        sm = data["secondary_matrix"]
        directions = sm["per_direction_aggregate"]
        assert len(directions) >= 7, (
            f"Expected >= 7 directions, got {len(directions)}: {list(directions.keys())}"
        )


# ---------------------------------------------------------------------------
# AUG-02 Pattern Tests
# ---------------------------------------------------------------------------


class TestPatternClassification:
    """Verify pattern classification matches independently observed patterns."""

    def test_pattern_classification(self) -> None:
        data = _load_output()
        ps = data["primary_matrix"]["pattern_summary"]
        assert "bfs" in ps["degradation"]
        assert "srad" in ps["stable_pass"]
        assert "backprop" in ps["other"]

    def test_md_summary_contains_sections(self) -> None:
        if not OUTPUT_MD.exists():
            pytest.skip("Run augmentation_analysis.py first")
        md = OUTPUT_MD.read_text(encoding="utf-8")
        assert "Per-Kernel Augmentation Matrix" in md
        assert "Pattern Summary" in md
        assert "Aggregate Pass Rates" in md


# ---------------------------------------------------------------------------
# AUG-04 Figure Tests
# ---------------------------------------------------------------------------

FIGURE_DIR = PROJECT_ROOT / "docs" / "paper" / "figures"


class TestFigureGeneration:
    """Verify augmentation figures exist and meet quality criteria."""

    def test_figures_exist(self) -> None:
        """AUG-04: All 4 figure files exist after generation."""
        for name in [
            "aug_heatmap.pdf",
            "aug_heatmap.png",
            "aug_trend.pdf",
            "aug_trend.png",
        ]:
            path = FIGURE_DIR / name
            if not path.exists():
                pytest.skip(
                    f"Run augmentation_analysis.py --figures first: {name} missing"
                )
            assert path.stat().st_size > 1024, (
                f"{name} is too small ({path.stat().st_size} bytes)"
            )

    def test_okabe_ito_palette(self) -> None:
        """AUG-04: Script defines correct Okabe-Ito STATUS_COLORS."""
        src = (
            PROJECT_ROOT / "scripts" / "analysis" / "augmentation_analysis.py"
        ).read_text(encoding="utf-8")
        assert '"#009E73"' in src, "Missing PASS green"
        assert '"#D55E00"' in src, "Missing BUILD_FAIL vermillion"
        assert '"#E69F00"' in src, "Missing RUN_FAIL orange"
        assert '"#0072B2"' in src, "Missing VERIFY_FAIL blue"

    def test_heatmap_dimensions(self) -> None:
        """AUG-04: Heatmap data has 26 kernels, L0 for all, L1-L4 for augmentation kernels."""
        data = _load_output()
        pm = data["primary_matrix"]
        assert pm["kernel_count"] == 26, f"Expected 26 kernels, got {pm['kernel_count']}"
        assert len(pm["levels"]) == 5, f"Expected 5 levels, got {len(pm['levels'])}"
        for kernel, entry in pm["per_kernel"].items():
            assert "L0" in entry, f"{kernel} missing L0"
        # Kernels with augmentation data must have L1-L4
        for kernel in ["bfs", "lud", "nw"]:
            if kernel in pm["per_kernel"]:
                entry = pm["per_kernel"][kernel]
                for lvl in ["L1", "L2", "L3", "L4"]:
                    assert lvl in entry, f"{kernel} missing {lvl} (augmentation data expected)"


# ---------------------------------------------------------------------------
# AUG-05: L0 consensus and multi-direction (Change 1)
# ---------------------------------------------------------------------------


class TestConsensusStatus:
    """Verify _consensus_status helper for L0 seed aggregation."""

    def test_pass_if_any_passes(self) -> None:
        assert _consensus_status(["PASS", "BUILD_FAIL", "BUILD_FAIL"]) == "PASS"

    def test_pass_if_all_pass(self) -> None:
        assert _consensus_status(["PASS", "PASS", "PASS"]) == "PASS"

    def test_most_common_on_all_fail(self) -> None:
        assert _consensus_status(["BUILD_FAIL", "RUN_FAIL", "BUILD_FAIL"]) == "BUILD_FAIL"

    def test_single_status(self) -> None:
        assert _consensus_status(["VERIFY_FAIL"]) == "VERIFY_FAIL"


class TestL0Populated:
    """After the fix, L0 should be populated for all kernels."""

    def test_l0_present_for_all_kernels(self) -> None:
        data = _load_output()
        pm = data["primary_matrix"]
        for kernel, entry in pm["per_kernel"].items():
            assert "L0" in entry, f"{kernel} missing L0"
            assert entry["L0"] not in ("", None), f"{kernel} L0 is empty"

