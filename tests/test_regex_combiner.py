"""Tests for the regex pattern combiner fix in llm_evaluate.py.

Regression tests for the _wrap_pattern() helper that scopes inline regex
flags (e.g., (?i)) to prevent 'global flags not at the start of the
expression' errors when patterns are combined via alternation.
"""

import re
import sys
import os
import pytest

# Add project root to path so we can import from scripts/evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluation.llm_evaluate import _wrap_pattern, _build_cross_api_verify_spec


class TestWrapPattern:
    """Tests for _wrap_pattern() helper."""

    def test_plain_pattern_wrapped_in_noncapturing_group(self):
        """Patterns without inline flags get standard (?:...) wrapping."""
        assert _wrap_pattern("Distance=") == "(?:Distance=)"

    def test_case_insensitive_flag_converted_to_scoped(self):
        """(?i) flag is converted to scoped form (?i:...)."""
        result = _wrap_pattern("(?i)(pass|correct)")
        assert result == "(?i:(pass|correct))"
        # Must compile without error
        re.compile(result)

    def test_multiple_flags_converted_to_scoped(self):
        """Multiple inline flags like (?is) are all scoped."""
        result = _wrap_pattern("(?is)foo.*bar")
        assert result == "(?is:foo.*bar)"
        re.compile(result)

    def test_all_flag_types(self):
        """All valid Python regex flags (a, i, m, s, u, x) are handled."""
        for flag in "aimsux":
            result = _wrap_pattern(f"(?{flag})test")
            assert result == f"(?{flag}:test)"
            re.compile(result)

    def test_no_flag_pattern_compiles(self):
        """Plain wrapped pattern compiles correctly."""
        result = _wrap_pattern(r"nel=\d+, nelr=\d+")
        re.compile(result)

    def test_flag_not_at_start_left_alone(self):
        """Inline flags not at the very start pass through as (?:...) wrapping.

        NOTE: The resulting pattern (?:foo(?i)bar) is itself invalid and would
        still raise 'global flags not at the start of the expression' if compiled.
        This is an out-of-scope case: no real spec has a mid-pattern inline flag.
        _wrap_pattern only fixes the leading-flag case that actually occurs in specs.
        """
        # (?i) in the middle of a pattern is not a leading flag — returned as-is in (?:...)
        result = _wrap_pattern("foo(?i)bar")
        assert result == "(?:foo(?i)bar)"
        # Intentionally NOT calling re.compile(result) — would raise 'global flags not at start'.
        # This is a known limitation; no real spec patterns hit this case.

    def test_old_wrapping_would_have_failed(self):
        """Regression: the old (?:(?i)...) form raises 'global flags not at start'."""
        # This documents the exact bug being fixed. If _wrap_pattern ever regresses
        # to f'(?:{p})' for flag patterns, this test catches it.
        bad_combined = "(?:(?i)(pass|correct|match|verified))|(?:Distance=)"
        with pytest.raises(re.error, match="global flags not at the start"):
            re.compile(bad_combined)
        # The fixed form must compile without error
        good_combined = "(?i:(pass|correct|match|verified))|(?:Distance=)"
        re.compile(good_combined)  # must not raise

    def test_flag_with_plain_rest_no_group(self):
        """Real spec pattern: (?i)pass without grouping parens around rest."""
        # hecbench-crc64-cuda and hecbench-crc64-omp use "(?i)pass"
        result = _wrap_pattern("(?i)pass")
        assert result == "(?i:pass)"
        compiled = re.compile(result)
        assert compiled.search("PASS")
        assert compiled.search("pass")
        assert not compiled.search("fail")


class TestBuildCrossApiVerifySpec:
    """Integration tests for _build_cross_api_verify_spec with inline flags."""

    def test_combining_case_insensitive_source_with_plain_target(self):
        """Regression: combining (?i) source pattern with plain target must not crash."""
        source_spec = {
            "verification": {
                "strategies": [
                    {"type": "stdout_pattern", "pattern": "(?i)(pass|correct)"}
                ]
            }
        }
        target_spec = {
            "verification": {
                "strategies": [
                    {"type": "stdout_pattern", "pattern": "Distance="},
                    {"type": "exit_code", "expected": 0},
                ]
            },
            "files": {"translation_targets": ["main.cu"]},  # non-kernel-only
        }
        result = _build_cross_api_verify_spec(target_spec, source_spec)
        combined = result["verification"]["strategies"][0]["pattern"]

        # Must compile without 'global flags not at start' error
        compiled = re.compile(combined)

        # (?i) should only affect its own branch
        assert compiled.search("Distance=0.5")
        assert compiled.search("PASS")  # case-insensitive branch matches
        assert not compiled.search("distance=0.5")  # (?i) scoped, not global

    def test_no_inline_flags_still_works(self):
        """Patterns without inline flags continue to combine correctly."""
        source_spec = {
            "verification": {
                "strategies": [
                    {"type": "stdout_pattern", "pattern": r"Saving solution"}
                ]
            }
        }
        target_spec = {
            "verification": {
                "strategies": [
                    {"type": "stdout_pattern", "pattern": r"Done\.\.\."},
                    {"type": "exit_code", "expected": 0},
                ]
            },
            "files": {"translation_targets": ["main.cpp"]},
        }
        result = _build_cross_api_verify_spec(target_spec, source_spec)
        combined = result["verification"]["strategies"][0]["pattern"]
        compiled = re.compile(combined)
        assert compiled.search("Saving solution to file")
        assert compiled.search("Done...")

    def test_kernel_only_bypasses_combining(self):
        """Kernel-only translations (OpenCL .cl) skip pattern combining."""
        source_spec = {
            "verification": {
                "strategies": [
                    {"type": "stdout_pattern", "pattern": "(?i)(pass|correct)"}
                ]
            }
        }
        target_spec = {
            "verification": {
                "strategies": [
                    {"type": "stdout_pattern", "pattern": r"nel=\d+, nelr=\d+"},
                    {"type": "exit_code", "expected": 0},
                ]
            },
            "files": {"translation_targets": ["kernel.cl"]},  # kernel-only
        }
        result = _build_cross_api_verify_spec(target_spec, source_spec)
        # Should use target pattern unchanged (no combining)
        pattern = result["verification"]["strategies"][0]["pattern"]
        assert pattern == r"nel=\d+, nelr=\d+"
