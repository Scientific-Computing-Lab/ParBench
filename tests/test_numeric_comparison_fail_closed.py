"""Fail-closed tests for the numeric_comparison verifier strategy.

Regression for the 2026-08-12 Task A review finding 6: float() alone accepts
booleans (float(True) == 1.0) and non-finite values, so a malformed spec with
tolerance=inf or expected=true could PASS anything, and a negative tolerance
produced FAIL where the fail-closed contract requires ERROR.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.models import RunResult, Status  # noqa: E402
from harness.verifier import _verify_numeric_comparison  # noqa: E402


def _run(stdout: str = "value: 1.0\n") -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=0,
        stdout=stdout,
        stderr="",
    )


def _strategy(**overrides) -> dict:
    base = {
        "type": "numeric_comparison",
        "extract_regex": r"value: ([-\d.eE+]+)",
        "expected": 1.0,
        "tolerance": 0.0,
    }
    base.update(overrides)
    return base


def test_well_formed_declaration_still_passes():
    r = _verify_numeric_comparison(_strategy(), _run())
    assert r.status == Status.PASS


@pytest.mark.parametrize("bad,label", [
    ({"expected": True}, "boolean expected"),
    ({"expected": False}, "boolean expected (false)"),
    ({"expected": math.inf}, "infinite expected"),
    ({"expected": math.nan}, "NaN expected"),
    ({"tolerance": True}, "boolean tolerance"),
    ({"tolerance": math.inf}, "infinite tolerance"),
    ({"tolerance": -math.inf}, "negative-infinite tolerance"),
    ({"tolerance": math.nan}, "NaN tolerance"),
    ({"tolerance": -1.0}, "negative tolerance"),
    ({"tolerance": "wide"}, "non-numeric tolerance"),
    ({"expected": "right"}, "non-numeric expected"),
], ids=lambda v: v if isinstance(v, str) else None)
def test_malformed_declaration_is_error_never_pass(bad, label):
    r = _verify_numeric_comparison(_strategy(**bad), _run())
    assert r.status == Status.ERROR, (
        f"{label}: expected ERROR (fail closed), got {r.status}: {r.details}"
    )


def test_infinite_tolerance_cannot_pass_arbitrary_output():
    """The exact exploit from the finding: tolerance=inf passed any value."""
    r = _verify_numeric_comparison(
        _strategy(tolerance=math.inf), _run("value: 999999.0\n"))
    assert r.status == Status.ERROR


def test_genuine_mismatch_is_still_fail_not_error():
    r = _verify_numeric_comparison(_strategy(), _run("value: 2.0\n"))
    assert r.status == Status.FAIL
