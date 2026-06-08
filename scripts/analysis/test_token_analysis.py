"""
Tests for token_analysis.py — EXCLUDED_SPECS filtering and utility functions.

Verifies that load_all_results() correctly filters out results
involving specs listed in EXCLUDED_SPECS, and that utility functions
and constants are correctly defined.
"""

import json
import tempfile
from pathlib import Path

import pytest

from token_analysis import EXCLUDED_SPECS, load_all_results

# Test fixture constants — single source of truth for default token values
DEFAULT_TEST_PROMPT_TOKENS = 1000
DEFAULT_TEST_COMPLETION_TOKENS = 500


def _make_result(source_spec: str, target_spec: str, status: str = "PASS",
                 model: str = "test-model") -> dict:
    """Create a minimal result JSON matching the fields load_all_results checks."""
    return {
        "overall_status": status,
        "source_spec": source_spec,
        "target_spec": target_spec,
        "model": model,
        "kernel": source_spec.rsplit("-", 1)[0],
        "prompt_tokens": DEFAULT_TEST_PROMPT_TOKENS,
        "completion_tokens": DEFAULT_TEST_COMPLETION_TOKENS,
        "llm_response_time_seconds": 2.5,
        "augment_level": 0,
    }


def _write_results(tmpdir: Path, model: str, results: list[dict]) -> None:
    """Write result JSONs into tmpdir/results/evaluation/{model}/."""
    model_dir = tmpdir / "results" / "evaluation" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        (model_dir / f"result_{i}.json").write_text(
            json.dumps(r), encoding="utf-8"
        )


class TestExcludedSpecsDefinition:
    """Verify EXCLUDED_SPECS is correctly defined in token_analysis.py."""

    def test_excluded_specs_is_frozenset(self):
        assert isinstance(EXCLUDED_SPECS, frozenset)

    def test_excluded_specs_has_eight_entries(self):
        assert len(EXCLUDED_SPECS) == 8

    def test_excluded_specs_matches_canonical(self):
        """Must match the canonical list from harness.constants."""
        expected = frozenset({
            "rodinia-kmeans-cuda",
            "rodinia-mummergpu-cuda",
            "rodinia-mummergpu-omp",
            "rodinia-hybridsort-cuda",
            "rodinia-nn-opencl",
            "rodinia-kmeans-opencl",
            "hecbench-stencil1d-omp_target",
            "hecbench-scan-omp_target",
        })
        assert EXCLUDED_SPECS == expected


class TestLoadAllResultsFiltering:
    """Verify load_all_results() filters out excluded specs."""

    def test_excludes_source_spec(self):
        """Results where source_spec is in EXCLUDED_SPECS should be filtered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results = [
                _make_result("rodinia-bfs-cuda", "rodinia-bfs-omp"),
                _make_result("rodinia-kmeans-cuda", "rodinia-kmeans-omp"),  # excluded source
                _make_result("rodinia-nw-cuda", "rodinia-nw-omp"),
            ]
            _write_results(tmpdir, "test-model", results)
            loaded = load_all_results(tmpdir)
            assert len(loaded) == 2
            specs = {r["source_spec"] for r in loaded}
            assert "rodinia-kmeans-cuda" not in specs

    def test_excludes_target_spec(self):
        """Results where target_spec is in EXCLUDED_SPECS should be filtered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results = [
                _make_result("rodinia-bfs-cuda", "rodinia-bfs-omp"),
                _make_result("rodinia-bfs-omp", "rodinia-nn-opencl"),  # excluded target
                _make_result("rodinia-nw-cuda", "rodinia-nw-omp"),
            ]
            _write_results(tmpdir, "test-model", results)
            loaded = load_all_results(tmpdir)
            assert len(loaded) == 2
            targets = {r["target_spec"] for r in loaded}
            assert "rodinia-nn-opencl" not in targets

    def test_keeps_non_excluded_specs(self):
        """Results for non-excluded specs should all be kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results = [
                _make_result("rodinia-bfs-cuda", "rodinia-bfs-omp"),
                _make_result("rodinia-nw-cuda", "rodinia-nw-omp"),
                _make_result("rodinia-hotspot-cuda", "rodinia-hotspot-omp"),
            ]
            _write_results(tmpdir, "test-model", results)
            loaded = load_all_results(tmpdir)
            assert len(loaded) == 3

    def test_filters_all_excluded_specs(self):
        """Every spec in EXCLUDED_SPECS should be filtered when it appears as source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # One valid result + one per excluded spec
            results = [_make_result("rodinia-bfs-cuda", "rodinia-bfs-omp")]
            for excluded in sorted(EXCLUDED_SPECS):
                results.append(_make_result(excluded, "rodinia-bfs-omp"))
            _write_results(tmpdir, "test-model", results)
            loaded = load_all_results(tmpdir)
            assert len(loaded) == 1
            assert loaded[0]["source_spec"] == "rodinia-bfs-cuda"

    def test_empty_directory(self):
        """No results directory should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "results" / "evaluation").mkdir(parents=True)
            loaded = load_all_results(tmpdir)
            assert loaded == []


class TestPassRateUsesFilteredTotal:
    """Verify that pass_rate denominator uses filtered (not raw) count."""

    def test_pass_rate_excludes_excluded_specs(self):
        """Pass rate should be computed over filtered results only.

        Setup: 3 results total, 1 excluded.
        - 2 non-excluded: 1 PASS + 1 BUILD_FAIL
        - 1 excluded: PASS (should not count)

        Expected pass_rate = 1/2 = 0.5, NOT 2/3 = 0.6667
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results = [
                _make_result("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS"),
                _make_result("rodinia-nw-cuda", "rodinia-nw-omp", "BUILD_FAIL"),
                _make_result("rodinia-kmeans-cuda", "rodinia-kmeans-omp", "PASS"),  # excluded
            ]
            _write_results(tmpdir, "test-model", results)
            loaded = load_all_results(tmpdir)
            # Only 2 results should remain
            assert len(loaded) == 2
            pass_count = sum(1 for r in loaded if r["overall_status"] == "PASS")
            pass_rate = pass_count / len(loaded)
            assert pass_rate == pytest.approx(0.5)


# --- new tests: utility extraction ---


def test_extract_token_lists_basic():
    from token_analysis import extract_token_lists
    results = [
        {"prompt_tokens": 100, "completion_tokens": 50},
        {"prompt_tokens": 200, "completion_tokens": 80},
    ]
    prompt, completion = extract_token_lists(results)
    assert prompt == [100, 200]
    assert completion == [50, 80]


def test_extract_token_lists_missing_keys_default_zero():
    from token_analysis import extract_token_lists
    results = [{"prompt_tokens": 100}]
    prompt, completion = extract_token_lists(results)
    assert prompt == [100]
    assert completion == [0]


def test_extract_token_lists_empty():
    from token_analysis import extract_token_lists
    assert extract_token_lists([]) == ([], [])


def test_precision_constants_exist():
    import token_analysis as ta
    assert ta.PRECISION_TOKENS == 1
    assert ta.PRECISION_RATE == 4
    assert ta.PRECISION_COST_DETAIL == 6


def test_field_constants_exist():
    import token_analysis as ta
    assert ta.FIELD_PROMPT_TOKENS == "prompt_tokens"
    assert ta.FIELD_OVERALL_STATUS == "overall_status"
