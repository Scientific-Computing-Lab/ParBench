#!/usr/bin/env python3
"""Tests for VERIFY_FAIL sub-classification in build_error_taxonomy.py."""

import sys
from pathlib import Path

# Ensure the scripts/analysis directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from build_error_taxonomy import (
    classify_verify_fail,
    build_taxonomy,
    extract_direction,
    VERIFY_FAIL_CATEGORIES,
)


# ── classify_verify_fail unit tests ──


def test_classify_verify_fail_wrong_output():
    """verify_status='fail' with non-empty stdout -> wrong_numerical_output."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "fail",
        "run_stdout_snippet": "WG size of kernel:initialize = 192, ...",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "wrong_numerical_output"


def test_classify_verify_fail_missing_output():
    """verify_status='fail' with empty stdout -> missing_output."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "fail",
        "run_stdout_snippet": "",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "missing_output"


def test_classify_verify_fail_missing_output_none():
    """verify_status='fail' with None stdout -> missing_output."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "fail",
        "run_stdout_snippet": None,
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "missing_output"


def test_classify_verify_fail_verification_error():
    """verify_status='error' -> verification_error."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "error",
        "run_stdout_snippet": "Number of records: 42764\nFinding...",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "verification_error"


def test_classify_verify_fail_pass_mislabel():
    """verify_status='pass' but overall_status is VERIFY_FAIL -> pass_overall_mislabel."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "pass",
        "run_stdout_snippet": "Random number generator seed: 7\nInput layer size : 65536",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "pass_overall_mislabel"


def test_classify_verify_fail_other():
    """Unknown verify_status -> other_verify."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "something_unknown",
        "run_stdout_snippet": "some output",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "other_verify"


def test_classify_verify_fail_no_verify_status():
    """Missing verify_status entirely -> other_verify."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "run_stdout_snippet": "some output",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "other_verify"


def test_classify_verify_fail_data_file_error():
    """verify_status='error' with data file error in stdout -> data_file_error secondary."""
    data = {
        "overall_status": "VERIFY_FAIL",
        "verify_status": "error",
        "run_stdout_snippet": "error opening a db (./cane4_0.db)\nNumber of records: 0",
        "run_exit_code": 0,
    }
    primary, secondaries = classify_verify_fail(data)
    assert primary == "verification_error"
    assert "data_file_error" in secondaries


# ── build_taxonomy integration tests ──


def _make_verify_fail_record(verify_status, stdout, exit_code=0,
                              model="test-model", kernel="test-kernel",
                              source_spec="rodinia-bfs-cuda",
                              target_spec="rodinia-bfs-omp"):
    """Create a minimal VERIFY_FAIL result dict."""
    return {
        "overall_status": "VERIFY_FAIL",
        "verify_status": verify_status,
        "run_stdout_snippet": stdout,
        "run_exit_code": exit_code,
        "model": model,
        "kernel": kernel,
        "source_spec": source_spec,
        "target_spec": target_spec,
        "augment_level": 0,
        "_file_path": "/fake/path/result.json",
    }


def test_build_taxonomy_includes_verify_fail():
    """build_taxonomy should populate verify_fail_categories for VERIFY_FAIL results."""
    results = [
        _make_verify_fail_record("fail", "wrong output here"),
        _make_verify_fail_record("fail", ""),
        _make_verify_fail_record("error", "some output"),
        _make_verify_fail_record("pass", "some output"),
    ]
    taxonomy = build_taxonomy(results)

    assert "verify_fail_categories" in taxonomy
    # All 4 should be classified as failures
    assert taxonomy["total_failures"] == 4
    assert taxonomy["total_pass"] == 0

    # Check that categories got populated
    vf_cats = taxonomy["verify_fail_categories"]
    total_vf = sum(c["count"] for c in vf_cats.values())
    assert total_vf == 4

    # Check specific categories
    assert vf_cats["wrong_numerical_output"]["count"] == 1
    assert vf_cats["missing_output"]["count"] == 1
    assert vf_cats["verification_error"]["count"] == 1
    assert vf_cats["pass_overall_mislabel"]["count"] == 1


def test_build_taxonomy_verify_fail_per_kernel_tracking():
    """Verify that per_kernel tracking includes verify_fail."""
    results = [
        _make_verify_fail_record("fail", "output", kernel="hotspot"),
        _make_verify_fail_record("fail", "output", kernel="hotspot"),
        _make_verify_fail_record("fail", "", kernel="bfs"),
    ]
    taxonomy = build_taxonomy(results)

    assert taxonomy["per_kernel"]["hotspot"]["verify_fail"]["wrong_numerical_output"] == 2
    assert taxonomy["per_kernel"]["bfs"]["verify_fail"]["missing_output"] == 1


def test_build_taxonomy_verify_fail_per_model_tracking():
    """Verify that per_model tracking includes verify_fail."""
    results = [
        _make_verify_fail_record("fail", "output", model="model-a"),
        _make_verify_fail_record("error", "output", model="model-b"),
    ]
    taxonomy = build_taxonomy(results)

    assert taxonomy["per_model"]["model-a"]["verify_fail"]["wrong_numerical_output"] == 1
    assert taxonomy["per_model"]["model-b"]["verify_fail"]["verification_error"] == 1


def test_build_taxonomy_verify_fail_classified_results():
    """Verify that classified_results have non-None primary_category for VERIFY_FAIL."""
    results = [
        _make_verify_fail_record("fail", "wrong output"),
    ]
    taxonomy = build_taxonomy(results)

    classified = taxonomy["classified_results"]
    assert len(classified) == 1
    assert classified[0]["primary_category"] == "wrong_numerical_output"
    assert classified[0]["overall_status"] == "VERIFY_FAIL"


def test_verify_fail_categories_constant_exists():
    """VERIFY_FAIL_CATEGORIES constant should exist and have entries."""
    assert isinstance(VERIFY_FAIL_CATEGORIES, list)
    assert len(VERIFY_FAIL_CATEGORIES) >= 4
    # Each entry is (name, description)
    for name, desc in VERIFY_FAIL_CATEGORIES:
        assert isinstance(name, str)
        assert isinstance(desc, str)


def test_build_taxonomy_mixed_statuses():
    """VERIFY_FAIL should coexist with other failure statuses."""
    results = [
        {
            "overall_status": "PASS",
            "model": "m",
            "kernel": "k",
            "source_spec": "rodinia-bfs-cuda",
            "target_spec": "rodinia-bfs-omp",
            "augment_level": 0,
            "_file_path": "/fake/pass.json",
        },
        {
            "overall_status": "BUILD_FAIL",
            "build_error_snippet": "undefined reference to cudaMalloc",
            "model": "m",
            "kernel": "k",
            "source_spec": "rodinia-bfs-cuda",
            "target_spec": "rodinia-bfs-omp",
            "augment_level": 0,
            "_file_path": "/fake/build.json",
        },
        _make_verify_fail_record("fail", "wrong output"),
    ]
    taxonomy = build_taxonomy(results)

    assert taxonomy["total_pass"] == 1
    assert taxonomy["total_failures"] == 2
    # Both failure types should have categories
    assert sum(c["count"] for c in taxonomy["build_fail_categories"].values()) == 1
    assert sum(c["count"] for c in taxonomy["verify_fail_categories"].values()) == 1


# --- new tests: safe_percentage ---


def test_safe_percentage_normal():
    from build_error_taxonomy import safe_percentage
    assert safe_percentage(10, 20) == 50.0


def test_safe_percentage_zero_denominator():
    from build_error_taxonomy import safe_percentage
    assert safe_percentage(5, 0) == 0.0


def test_safe_percentage_full():
    from build_error_taxonomy import safe_percentage
    assert safe_percentage(20, 20) == 100.0


# --- Change 2: omp_target parsing and canonical/augmentation split ---


def test_extract_direction_omp_target():
    """extract_direction must handle omp_target correctly (not split as 'target')."""
    data = {"source_spec": "rodinia-hotspot3d-omp_target", "target_spec": "rodinia-hotspot3d-cuda"}
    assert extract_direction(data) == "omp_target-to-cuda"


def test_extract_direction_standard():
    """extract_direction handles standard APIs."""
    data = {"source_spec": "rodinia-bfs-cuda", "target_spec": "rodinia-bfs-omp"}
    assert extract_direction(data) == "cuda-to-omp"


def test_extract_direction_opencl():
    """extract_direction handles opencl."""
    data = {"source_spec": "rodinia-bfs-opencl", "target_spec": "rodinia-bfs-cuda"}
    assert extract_direction(data) == "opencl-to-cuda"


def test_taxonomy_canonical_augmentation_split():
    """Output JSON should have canonical and augmentation sections with correct counts."""
    import json
    taxonomy_path = Path(__file__).parent.parent.parent / "results" / "analysis" / "error_taxonomy.json"
    if not taxonomy_path.exists():
        import pytest
        pytest.skip("Run build_error_taxonomy.py first")
    data = json.loads(taxonomy_path.read_text())
    assert "canonical" in data, "Missing canonical section"
    assert "augmentation" in data, "Missing augmentation section"
    can_total = data["canonical"]["total_results"]
    aug_total = data["augmentation"]["total_results"]
    combined_total = data["total_results"]
    assert can_total + aug_total == combined_total, (
        f"Partition mismatch: {can_total} + {aug_total} != {combined_total}"
    )
    assert can_total > 0, "Canonical should have results"
    assert aug_total > 0, "Augmentation should have results"
    assert can_total > aug_total, "Canonical should have more results than augmentation"
