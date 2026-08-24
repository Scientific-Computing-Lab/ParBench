"""
Tests for token_analysis.py — CORRECTNESS_INELIGIBLE_SPECS filtering and
utility functions.

Verifies that load_all_results() correctly filters out results involving
specs listed in CORRECTNESS_INELIGIBLE_SPECS (KNOWN_FAIL union
performance_only, Task 4 E-04), and that utility functions and constants
are correctly defined.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from token_analysis import CORRECTNESS_INELIGIBLE_SPECS, load_all_results

SCRIPT = Path(__file__).with_name("token_analysis.py")

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
    """Verify the filter set is correctly defined in token_analysis.py."""

    def test_ineligible_specs_is_frozenset(self):
        assert isinstance(CORRECTNESS_INELIGIBLE_SPECS, frozenset)

    def test_ineligible_specs_matches_canonical(self):
        """Must be the canonical set from harness.constants, not a copy of it.

        A hand-copied duplicate lived here until 2026-07-24 and had drifted two
        entries behind (8 vs 10).
        """
        from harness.constants import CORRECTNESS_INELIGIBLE_SPECS as canonical
        assert CORRECTNESS_INELIGIBLE_SPECS == canonical

    def test_ineligible_covers_known_fail_and_performance_only(self):
        from harness.constants import EXCLUDED_SPECS, PERFORMANCE_ONLY_SPECS
        assert CORRECTNESS_INELIGIBLE_SPECS == (
            EXCLUDED_SPECS | PERFORMANCE_ONLY_SPECS
        )


class TestLoadAllResultsFiltering:
    """Verify load_all_results() filters out excluded specs."""

    def test_excludes_source_spec(self):
        """Results where source_spec is in CORRECTNESS_INELIGIBLE_SPECS should be filtered."""
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
        """Results where target_spec is in CORRECTNESS_INELIGIBLE_SPECS should be filtered."""
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
        """Every ineligible spec should be filtered when it appears as source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # One valid result + one per excluded spec
            results = [_make_result("rodinia-bfs-cuda", "rodinia-bfs-omp")]
            for excluded in sorted(CORRECTNESS_INELIGIBLE_SPECS):
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


# --- gate9 finding 3: tie-aware Spearman ---


class TestSpearmanTieAware:
    """spearman_correlation must handle ties like a validated implementation.

    The textbook 1 - 6*Sum(d^2)/(n*(n^2-1)) shortcut is only exact with NO
    ties; the binary pass outcome is almost all ties. Average-rank (fractional)
    ranking recovers the correct coefficient (matches scipy.stats.spearmanr).
    """

    def test_matches_scipy_on_tied_data(self):
        from token_analysis import spearman_correlation
        sp_stats = pytest.importorskip("scipy.stats")
        xs = [10, 20, 20, 30, 30, 30, 40, 40]
        ys = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        expected = float(sp_stats.spearmanr(xs, ys).statistic)
        got = spearman_correlation(xs, ys)
        assert got == pytest.approx(round(expected, 4), abs=1e-4)

    def test_matches_scipy_on_binary_outcome(self):
        from token_analysis import spearman_correlation
        sp_stats = pytest.importorskip("scipy.stats")
        # Many ties in both variables (mirrors completion_tokens vs pass/fail).
        xs = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500]
        ys = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        expected = float(sp_stats.spearmanr(xs, ys).statistic)
        got = spearman_correlation(xs, ys)
        assert got == pytest.approx(round(expected, 4), abs=1e-4)

    def test_zero_variance_returns_none(self):
        from token_analysis import spearman_correlation
        # All y identical -> undefined correlation -> None, never a crash.
        assert spearman_correlation([1, 2, 3, 4], [5.0, 5.0, 5.0, 5.0]) is None


# --- gate9 finding 2: per-model stats over evaluated models, not priced set ---


def test_per_model_stats_include_unpriced_evaluated_models(tmp_path):
    """A model evaluated but absent from MODEL_PRICING must still get its
    non-cost per-model statistics; only cost fields are null."""
    priced = "together-qwen-3.5-397b-a17b"
    unpriced = "azure-gpt-5.3-codex"
    _write_results(tmp_path, priced,
                   [_make_result("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS", priced)])
    _write_results(tmp_path, unpriced,
                   [_make_result("rodinia-nw-cuda", "rodinia-nw-omp", "PASS", unpriced),
                    _make_result("rodinia-hotspot-cuda", "rodinia-hotspot-omp", "BUILD_FAIL", unpriced)])
    out_dir = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(tmp_path),
         "--output-dir", str(out_dir)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads((out_dir / "token_analysis.json").read_text())
    assert unpriced in data["by_model"], "unpriced evaluated model dropped"
    m = data["by_model"][unpriced]
    assert m["total_results"] == 2
    assert m["pass_count"] == 1
    assert m["prompt_tokens"]["count"] == 2
    # Cost is not computable without a verified rate -> null, not a wrong 0.
    assert m["cost_usd"]["total"] is None
    assert m["cost_usd"]["cost_per_pass"] is None
    # Priced model keeps a real cost.
    assert data["by_model"][priced]["cost_usd"]["total"] is not None


# --- gate9 finding 4: no hard-coded provider billing block ---


def test_no_hardcoded_billing_block(tmp_path):
    """The unsourced Together billing literal and its 46% coverage claim must
    not be emitted as paper-reporting evidence."""
    _write_results(tmp_path, "together-qwen-3.5-397b-a17b",
                   [_make_result("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS",
                                 "together-qwen-3.5-397b-a17b")])
    out_dir = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(tmp_path),
         "--output-dir", str(out_dir)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads((out_dir / "token_analysis.json").read_text())
    assert "actual_billing" not in data
    md = (out_dir / "token_analysis.md").read_text()
    assert "145.37" not in md
    assert "46%" not in md
    assert "Actual Billing" not in md


# --- gate9 finding 1: verdict-dependent token analysis over replay records ---


def _write_replay(tmpdir: Path, model: str, name: str, replay: dict) -> None:
    d = tmpdir / "replay" / model
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(replay), encoding="utf-8")


def test_replay_load_uses_replay_verdict_and_backfills_tokens(tmp_path):
    """Over the replay root, token analysis must use the replay verdict but
    recover the immutable token fields from the referenced submitted record."""
    # Submitted (original model-call) record: PASS, carries token fields.
    sub_dir = tmp_path / "results" / "evaluation" / "m"
    sub_dir.mkdir(parents=True)
    sub = _make_result("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS", "m")
    (sub_dir / "r.json").write_text(json.dumps(sub), encoding="utf-8")
    # Replay record: DIFFERENT verdict, no top-level tokens/model/kernel.
    replay = {
        "overall_status": "VERIFY_FAIL",
        "replay_kind": "stored_translation_replay",
        "input_record": {"path": "results/evaluation/m/r.json"},
        "parent": {
            "model": "m",
            "source_spec": "rodinia-bfs-cuda",
            "target_spec": "rodinia-bfs-omp",
            "augment_level": 0,
            "sample_id": 0,
        },
    }
    _write_replay(tmp_path, "m", "r", replay)

    loaded = load_all_results(tmp_path, tmp_path / "replay")
    assert len(loaded) == 1
    rec = loaded[0]
    # Verdict is the REPLAY verdict, never the submitted PASS.
    assert rec["overall_status"] == "VERIFY_FAIL"
    # Token fields recovered from the submitted record.
    assert rec["prompt_tokens"] == DEFAULT_TEST_PROMPT_TOKENS
    assert rec["completion_tokens"] == DEFAULT_TEST_COMPLETION_TOKENS
    # Identity promoted / recovered so grouping works.
    assert rec["model"] == "m"
    assert rec["kernel"] == "rodinia-bfs"
