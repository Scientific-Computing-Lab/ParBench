#!/usr/bin/env python3
"""Tests for cross_model_comparison.py.

TDD RED phase: these tests define the expected behavior before implementation.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Will be importable after GREEN phase
sys.path.insert(0, str(Path(__file__).parent))


def test_cohens_h_identical_proportions():
    """Test 1: cohens_h(0.5, 0.5) == 0.0 (identical proportions)."""
    from cross_model_comparison import cohens_h
    assert cohens_h(0.5, 0.5) == 0.0


def test_cohens_h_maximum_effect():
    """Test 2: cohens_h(1.0, 0.0) > 0 (maximum positive effect)."""
    from cross_model_comparison import cohens_h
    assert cohens_h(1.0, 0.0) > 0


def test_cohens_h_reversed():
    """Test 3: cohens_h(0.0, 1.0) < 0 (reversed)."""
    from cross_model_comparison import cohens_h
    assert cohens_h(0.0, 1.0) < 0


def test_classify_effect_size():
    """Test 4: classify_effect_size thresholds."""
    from cross_model_comparison import classify_effect_size
    assert classify_effect_size(0.1) == "negligible"
    assert classify_effect_size(0.3) == "small"
    assert classify_effect_size(0.5) == "medium"
    assert classify_effect_size(0.9) == "large"


def test_output_json_has_required_keys():
    """Test 5: Output JSON has required keys (mcnemar when passk_estimates present)."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data_with_estimates()
    result = build_comparison(qwen_data, gpt_data)

    assert "overall" in result
    assert "mcnemar" in result["overall"]
    assert "chi_squared" not in result["overall"]
    assert "p_value" in result["overall"]["mcnemar"]
    assert "cohens_h" in result["overall"]
    assert "per_direction" in result
    assert "per_kernel_matrix" in result


def test_per_kernel_matrix_four_keys():
    """Test 6: per_kernel_matrix has exactly 4 classification keys."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data()
    result = build_comparison(qwen_data, gpt_data)

    km = result["per_kernel_matrix"]
    name_a, name_b = result["models"]
    for key in ["both_pass", "both_fail", f"{name_a}_only_pass", f"{name_b}_only_pass"]:
        assert key in km, f"Missing key: {key}"


def test_common_directions_count():
    """Test 7: common_directions has exactly 7 entries (GPT missing omp_target-to-cuda)."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data_full_directions()
    result = build_comparison(qwen_data, gpt_data)

    assert len(result["common_directions"]) == 7


def test_missing_directions():
    """Test 8: missing_directions contains omp_target-to-cuda for GPT."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data_full_directions()
    result = build_comparison(qwen_data, gpt_data)

    assert "azure-gpt-5.4" in result["missing_directions"]
    assert "omp_target-to-cuda" in result["missing_directions"]["azure-gpt-5.4"]


def test_kernel_matrix_counts_sum():
    """per_kernel_matrix counts sum to total_common_kernels."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data()
    result = build_comparison(qwen_data, gpt_data)

    km = result["per_kernel_matrix"]
    name_a, name_b = result["models"]
    total = (km["counts"]["both_pass"] + km["counts"]["both_fail"]
             + km["counts"][f"{name_a}_only_pass"] + km["counts"][f"{name_b}_only_pass"])
    assert total == km["total_common_kernels"]


# --- Helpers ---

def _make_dir_data(total, passed):
    return {
        "total": total,
        "pass": passed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "ci_lower": 0.0,
        "ci_upper": 1.0,
        "by_status": {"PASS": passed, "BUILD_FAIL": total - passed},
    }


def _make_test_data():
    """Create minimal test data with 2 common directions and 3 kernels."""
    qwen = {
        "model": "together-qwen-3.5-397b-a17b",
        "primary_campaign": {
            "total": 100,
            "overall": {
                "total": 100, "pass": 40, "pass_rate": 0.4,
                "ci_lower": 0.3, "ci_upper": 0.5,
                "by_status": {"PASS": 40, "BUILD_FAIL": 60},
            },
            "by_direction": {
                "cuda-to-omp": _make_dir_data(50, 25),
                "omp-to-cuda": _make_dir_data(50, 15),
            },
            "by_kernel": {
                "backprop": _make_dir_data(10, 5),    # passes (pass > 0)
                "bfs": _make_dir_data(10, 0),          # fails
                "cfd": _make_dir_data(10, 3),           # passes
            },
        },
    }
    gpt = {
        "model": "azure-gpt-5.4",
        "primary_campaign": {
            "total": 80,
            "overall": {
                "total": 80, "pass": 20, "pass_rate": 0.25,
                "ci_lower": 0.15, "ci_upper": 0.35,
                "by_status": {"PASS": 20, "BUILD_FAIL": 60},
            },
            "by_direction": {
                "cuda-to-omp": _make_dir_data(40, 15),
                "omp-to-cuda": _make_dir_data(40, 5),
            },
            "by_kernel": {
                "backprop": _make_dir_data(8, 4),     # passes
                "bfs": _make_dir_data(8, 2),           # passes (gpt_only_pass)
                "cfd": _make_dir_data(8, 0),           # fails (qwen_only_pass)
            },
        },
    }
    return qwen, gpt


def _make_test_data_full_directions():
    """Create test data with 8 Qwen directions and 7 GPT directions."""
    common_dirs = [
        "cuda-to-omp", "omp-to-cuda",
        "cuda-to-opencl", "opencl-to-cuda",
        "omp-to-opencl", "opencl-to-omp",
        "cuda-to-omp_target",
    ]
    qwen_dirs = {d: _make_dir_data(20, 8) for d in common_dirs}
    qwen_dirs["omp_target-to-cuda"] = _make_dir_data(20, 5)  # Qwen-only

    gpt_dirs = {d: _make_dir_data(15, 4) for d in common_dirs}

    qwen = {
        "model": "together-qwen-3.5-397b-a17b",
        "primary_campaign": {
            "total": 180,
            "overall": {
                "total": 180, "pass": 69, "pass_rate": 69 / 180,
                "ci_lower": 0.3, "ci_upper": 0.5,
                "by_status": {"PASS": 69, "BUILD_FAIL": 111},
            },
            "by_direction": qwen_dirs,
            "by_kernel": {
                "backprop": _make_dir_data(10, 5),
            },
        },
    }
    gpt = {
        "model": "azure-gpt-5.4",
        "primary_campaign": {
            "total": 105,
            "overall": {
                "total": 105, "pass": 28, "pass_rate": 28 / 105,
                "ci_lower": 0.15, "ci_upper": 0.35,
                "by_status": {"PASS": 28, "BUILD_FAIL": 77},
            },
            "by_direction": gpt_dirs,
            "by_kernel": {
                "backprop": _make_dir_data(8, 4),
            },
        },
    }
    return qwen, gpt


def test_build_comparison_has_mcnemar():
    """build_comparison output contains mcnemar (not chi_squared) when passk_estimates present."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data_with_estimates()
    result = build_comparison(qwen_data, gpt_data)

    assert "mcnemar" in result["overall"]
    assert "chi_squared" not in result["overall"]
    mcn = result["overall"]["mcnemar"]
    assert "both_pass" in mcn
    assert "both_fail" in mcn
    assert "a_only" in mcn
    assert "b_only" in mcn
    assert "mcnemar_chi2" in mcn
    assert "p_value" in mcn
    assert mcn["both_pass"] + mcn["both_fail"] + mcn["a_only"] + mcn["b_only"] == mcn["total"]
    assert "cohens_h" in result["overall"]


def _make_test_data_with_estimates():
    """Extend _make_test_data with passk_campaign including passk_estimates."""
    qwen, gpt = _make_test_data()
    # 4 tasks covering all concordance cells
    estimates_qwen = {
        "k1:cuda-to-omp": {"n": 3, "c": 2},  # pass
        "k2:cuda-to-omp": {"n": 3, "c": 0},  # fail
        "k3:omp-to-cuda": {"n": 3, "c": 1},  # pass (qwen_only)
        "k4:omp-to-cuda": {"n": 3, "c": 0},  # fail
    }
    estimates_gpt = {
        "k1:cuda-to-omp": {"n": 3, "c": 3},  # pass (both_pass)
        "k2:cuda-to-omp": {"n": 3, "c": 0},  # fail (both_fail)
        "k3:omp-to-cuda": {"n": 3, "c": 0},  # fail (qwen_only)
        "k4:omp-to-cuda": {"n": 3, "c": 2},  # pass (gpt_only)
    }
    qwen["passk_campaign"] = dict(qwen["primary_campaign"])
    qwen["passk_campaign"]["passk_estimates"] = estimates_qwen
    gpt["passk_campaign"] = dict(gpt["primary_campaign"])
    gpt["passk_campaign"]["passk_estimates"] = estimates_gpt
    return qwen, gpt


def test_compute_mcnemar_basic():
    """McNemar concordance table and chi² on synthetic paired data."""
    from cross_model_comparison import compute_mcnemar

    # 7 paired tasks with known concordance
    qwen_est = {
        "a": {"c": 3}, "b": {"c": 0}, "c": {"c": 1},
        "d": {"c": 0}, "e": {"c": 2}, "f": {"c": 0},
        "g": {"c": 0},
    }
    gpt_est = {
        "a": {"c": 2}, "b": {"c": 0}, "c": {"c": 0},
        "d": {"c": 1}, "e": {"c": 3}, "f": {"c": 1},
        "g": {"c": 0},
    }
    # a=both_pass, b=both_fail, c=qwen_only, d=gpt_only,
    # e=both_pass, f=gpt_only, g=both_fail

    result = compute_mcnemar(qwen_est, gpt_est)

    assert result["both_pass"] == 2
    assert result["both_fail"] == 2
    assert result["a_only"] == 1
    assert result["b_only"] == 2
    assert result["total"] == 7
    # Yates-corrected McNemar chi2 = (|1-2| - 1)^2 / (1+2) = 0/3 = 0
    assert result["mcnemar_chi2"] == 0.0
    assert result["p_value"] == 1.0


def test_build_comparison_fallback_chi_squared():
    """build_comparison uses chi_squared when passk_estimates is absent."""
    from cross_model_comparison import build_comparison

    qwen_data, gpt_data = _make_test_data()
    result = build_comparison(qwen_data, gpt_data)

    assert "chi_squared" in result["overall"]
    assert "mcnemar" not in result["overall"]
    assert "p_value" in result["overall"]["chi_squared"]
    assert "chi2" in result["overall"]["chi_squared"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
