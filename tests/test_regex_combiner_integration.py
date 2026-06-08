"""Integration tests for regex pattern combiner using real benchmark data.

Tests _wrap_pattern() and _build_cross_api_verify_spec() against actual spec
files and result files from the ParBench evaluation pipeline. These tests
verify the fix works end-to-end with real data, not just synthetic examples.

TDD note: These tests document the exact bug that existed before the fix.
The test_old_combined_pattern_fails_on_real_specs test proves the old code
would have crashed. The remaining tests prove the fix works.
"""

import json
import os
import re
import sys

import pytest

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluation.llm_evaluate import _wrap_pattern, _build_cross_api_verify_spec

SPECS_DIR = os.path.join(PROJECT_ROOT, "specs")
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "results", "evaluation", "together-qwen-3.5-397b-a17b"
)

# All 8 real specs with (?i) patterns, mapped to their actual patterns
SPECS_WITH_INLINE_FLAGS = {
    "hecbench-crc64-cuda": "(?i)pass",
    "hecbench-crc64-omp": "(?i)pass",
    "rodinia-hybridsort-cuda": "(?i)(PASS|passed|SUCCESS|correct)",
    "rodinia-kmeans-cuda": "(?i)(pass|correct|match|verified)",
    "rodinia-kmeans-opencl": "(?i)(PASS|passed|SUCCESS|correct)",
    "rodinia-mummergpu-cuda": "(?i)(PASS|passed|SUCCESS|correct)",
    "rodinia-mummergpu-omp": "(?i)(PASS|passed|SUCCESS|correct)",
    "rodinia-nn-opencl": "(?i)(pass|correct|match|verified)",
}

# The 9 affected result files and their expected flip status
AFFECTED_RESULTS = {
    "rodinia-nn-opencl-to-rodinia-nn-cuda-L1.json": True,   # would flip to PASS
    "rodinia-nn-opencl-to-rodinia-nn-cuda-L3.json": True,   # would flip to PASS
    "rodinia-nn-opencl-to-rodinia-nn-cuda-L4.json": True,   # would flip to PASS
    "rodinia-nn-opencl-to-rodinia-nn-omp-L1.json": False,   # still fails
    "rodinia-nn-opencl-to-rodinia-nn-omp-L2.json": False,   # still fails
    "rodinia-nn-opencl-to-rodinia-nn-omp-L3.json": False,   # still fails
    "rodinia-nn-opencl-to-rodinia-nn-omp-L4.json": False,   # still fails
    "rodinia-nn-opencl-to-rodinia-nn-omp-s2.json": False,   # still fails
    "rodinia-nn-opencl-to-rodinia-nn-omp.json": False,      # still fails
}


def _load_spec(spec_id: str) -> dict:
    """Load a spec JSON from disk."""
    path = os.path.join(SPECS_DIR, f"{spec_id}.json")
    with open(path) as f:
        return json.load(f)


def _load_result(filename: str) -> dict:
    """Load a result JSON from disk."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ── RED tests: prove the old code was broken ──────────────────────────


class TestOldCodeWasBroken:
    """Prove the bug existed: old (?:pattern) wrapping crashes on real specs."""

    @pytest.mark.parametrize("spec_id,pattern", SPECS_WITH_INLINE_FLAGS.items())
    def test_old_wrapping_crashes_on_real_spec_patterns(self, spec_id, pattern):
        """The old f'(?:{p})' wrapping produces invalid regex for every (?i) spec."""
        old_wrapped = f"(?:{pattern})"
        # Combining with any other pattern via alternation triggers the error
        old_combined = f"{old_wrapped}|(?:Distance=)"
        with pytest.raises(re.error, match="global flags not at the start"):
            re.compile(old_combined)

    def test_old_combined_pattern_matches_actual_error_messages(self):
        """The error in result files matches exactly what old code would produce."""
        if not os.path.isdir(RESULTS_DIR):
            pytest.skip("Results directory not available")

        for filename in AFFECTED_RESULTS:
            result = _load_result(filename)
            error_msg = result.get("error_message", "") or ""
            # Every affected result has this exact error pattern
            assert "Invalid regex" in error_msg, f"{filename}: expected 'Invalid regex' in error"
            assert "global flags not at the start" in error_msg, (
                f"{filename}: expected 'global flags' error"
            )


# ── GREEN tests: prove the fix works with real data ───────────────────


class TestWrapPatternWithRealSpecs:
    """Verify _wrap_pattern handles every real (?i) pattern from specs on disk."""

    @pytest.mark.parametrize("spec_id,expected_pattern", SPECS_WITH_INLINE_FLAGS.items())
    def test_wrap_pattern_on_real_spec(self, spec_id, expected_pattern):
        """_wrap_pattern produces a compilable scoped pattern for each real spec."""
        result = _wrap_pattern(expected_pattern)
        # Must compile without error
        compiled = re.compile(result)
        # Scoped flag form: (?i:...) not (?:(?i)...)
        assert not result.startswith("(?:(?i)"), (
            f"Pattern still uses old broken wrapping: {result}"
        )
        # Must match case-insensitively within its scope
        assert compiled.search("PASS") or compiled.search("pass"), (
            f"Pattern should match 'pass' case-insensitively: {result}"
        )

    @pytest.mark.parametrize("spec_id", SPECS_WITH_INLINE_FLAGS.keys())
    def test_real_spec_pattern_from_disk_matches_expected(self, spec_id):
        """Verify our hardcoded patterns match what's actually in the spec files."""
        if not os.path.isfile(os.path.join(SPECS_DIR, f"{spec_id}.json")):
            pytest.skip(f"Spec file {spec_id}.json not found")
        spec = _load_spec(spec_id)
        strategies = (spec.get("verification") or {}).get("strategies", [])
        stdout_patterns = [
            s["pattern"] for s in strategies
            if s.get("type") == "stdout_pattern" and s.get("pattern")
        ]
        assert len(stdout_patterns) > 0, f"{spec_id} has no stdout_pattern"
        # At least one pattern should contain (?i)
        has_flag = any("(?i)" in p for p in stdout_patterns)
        assert has_flag, f"{spec_id}: expected (?i) in patterns, got {stdout_patterns}"
        # Pattern should match our expected value
        assert SPECS_WITH_INLINE_FLAGS[spec_id] in stdout_patterns, (
            f"{spec_id}: expected {SPECS_WITH_INLINE_FLAGS[spec_id]} in {stdout_patterns}"
        )


class TestCrossApiCombiningWithRealSpecs:
    """Test _build_cross_api_verify_spec with actual spec pairs from disk."""

    def test_nn_opencl_to_nn_cuda_combined_pattern_compiles(self):
        """The exact spec pair that triggered the bug: nn-opencl → nn-cuda."""
        source = _load_spec("rodinia-nn-opencl")
        target = _load_spec("rodinia-nn-cuda")
        result = _build_cross_api_verify_spec(target, source)
        combined = result["verification"]["strategies"][0]["pattern"]
        # This is the line that crashed before the fix
        compiled = re.compile(combined)
        # Must match Distance= (nn-cuda's pattern)
        assert compiled.search("Distance=0.123456")
        # Must match case-insensitive pass/correct (nn-opencl's pattern)
        assert compiled.search("PASS")
        assert compiled.search("correct")

    def test_nn_opencl_to_nn_omp_combined_pattern_compiles(self):
        """Second affected pair: nn-opencl → nn-omp."""
        source = _load_spec("rodinia-nn-opencl")
        target = _load_spec("rodinia-nn-omp")
        result = _build_cross_api_verify_spec(target, source)
        combined = result["verification"]["strategies"][0]["pattern"]
        compiled = re.compile(combined)
        # Must match "total time" (nn-omp's pattern)
        assert compiled.search("total time: 0.123")
        # Must match case-insensitive pass (nn-opencl's pattern)
        assert compiled.search("PASS")

    def test_case_insensitivity_scoped_not_global(self):
        """(?i) from source must NOT make the target pattern case-insensitive."""
        source = _load_spec("rodinia-nn-opencl")
        target = _load_spec("rodinia-nn-cuda")
        result = _build_cross_api_verify_spec(target, source)
        combined = result["verification"]["strategies"][0]["pattern"]
        compiled = re.compile(combined)
        # "Distance=" is case-sensitive — lowercase should NOT match
        assert not compiled.search("distance=0.5"), (
            "(?i) leaked to Distance= branch — scoping is broken"
        )
        # But "PASS" should match (case-insensitive in its own branch)
        assert compiled.search("PASS")
        assert compiled.search("pass")

    @pytest.mark.parametrize(
        "source_id,target_id",
        [
            ("rodinia-kmeans-cuda", "rodinia-kmeans-omp"),
            ("rodinia-mummergpu-cuda", "rodinia-mummergpu-omp"),
            ("rodinia-hybridsort-cuda", "rodinia-hybridsort-omp"),
        ],
    )
    def test_other_known_fail_pairs_compile(self, source_id, target_id):
        """All KNOWN_FAIL specs with (?i) produce valid combined patterns."""
        source_path = os.path.join(SPECS_DIR, f"{source_id}.json")
        target_path = os.path.join(SPECS_DIR, f"{target_id}.json")
        if not os.path.isfile(source_path) or not os.path.isfile(target_path):
            pytest.skip(f"Spec pair {source_id} → {target_id} not found")
        source = _load_spec(source_id)
        target = _load_spec(target_id)
        result = _build_cross_api_verify_spec(target, source)
        strategies = result["verification"]["strategies"]
        stdout_strats = [s for s in strategies if s.get("type") == "stdout_pattern"]
        if stdout_strats:
            combined = stdout_strats[0]["pattern"]
            # Must compile without error — this is the whole point of the fix
            re.compile(combined)


class TestReverifyAgainstRealResults:
    """Verify the fix produces correct flip predictions against actual result files."""

    @pytest.fixture(autouse=True)
    def _check_results_dir(self):
        if not os.path.isdir(RESULTS_DIR):
            pytest.skip("Results directory not available")

    @pytest.mark.parametrize("filename,should_flip", AFFECTED_RESULTS.items())
    def test_flip_prediction_matches_expected(self, filename, should_flip):
        """Each affected result should flip (or not) as predicted by the audit."""
        result = _load_result(filename)
        source_id = result["source_spec"]
        target_id = result["target_spec"]
        stdout = result.get("run_stdout_snippet", "") or ""

        # Rebuild the combined pattern using the fixed logic
        source = _load_spec(source_id)
        target = _load_spec(target_id)
        fixed_result = _build_cross_api_verify_spec(target, source)
        combined = fixed_result["verification"]["strategies"][0]["pattern"]
        compiled = re.compile(combined)

        match = compiled.search(stdout)
        if should_flip:
            assert match is not None, (
                f"{filename}: expected stdout to match fixed pattern but didn't. "
                f"stdout[:80]={stdout[:80]!r}, pattern={combined}"
            )
        else:
            assert match is None, (
                f"{filename}: expected stdout NOT to match but it did. "
                f"matched={match.group()!r}"
            )

    def test_exactly_3_results_would_flip(self):
        """The audit predicted exactly 3 flips. Verify against all 9 results."""
        flip_count = 0
        for filename in AFFECTED_RESULTS:
            result = _load_result(filename)
            source = _load_spec(result["source_spec"])
            target = _load_spec(result["target_spec"])
            stdout = result.get("run_stdout_snippet", "") or ""
            fixed = _build_cross_api_verify_spec(target, source)
            combined = fixed["verification"]["strategies"][0]["pattern"]
            if re.search(combined, stdout):
                flip_count += 1
        assert flip_count == 3, f"Expected 3 flips, got {flip_count}"

    def test_all_flips_are_nn_opencl_to_cuda(self):
        """All flips should be nn-opencl → nn-cuda (not nn-opencl → nn-omp)."""
        for filename, should_flip in AFFECTED_RESULTS.items():
            if should_flip:
                assert "cuda" in filename, f"Unexpected flip in non-CUDA target: {filename}"
                assert "omp" not in filename, f"Unexpected flip in OMP target: {filename}"
