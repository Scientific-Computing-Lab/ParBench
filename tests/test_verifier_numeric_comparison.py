"""Unit tests for the numeric_comparison verifier strategy."""

from __future__ import annotations

from harness.models import RunResult, Status
from harness.verifier import verify_run


def _make_run(stdout: str) -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=0,
        stdout=stdout,
        stderr="",
    )


def _spec_with_numeric(
    *,
    extract_regex: str | None,
    expected: float,
    tolerance: float = 0.0,
) -> dict:
    strat: dict = {"type": "numeric_comparison", "expected": expected, "tolerance": tolerance}
    if extract_regex is not None:
        strat["extract_regex"] = extract_regex
    return {"verification": {"method": "self_checking", "strategies": [strat]}}


def test_numeric_match_within_tolerance_passes():
    spec = _spec_with_numeric(extract_regex=r"Accuracy:\s*([0-9.]+)", expected=0.95, tolerance=0.01)
    run = _make_run("Accuracy: 0.949\n")
    result = verify_run(spec, run)
    assert result.status == Status.PASS
    assert result.strategy_used == "numeric_comparison"


def test_numeric_outside_tolerance_fails():
    spec = _spec_with_numeric(extract_regex=r"Accuracy:\s*([0-9.]+)", expected=0.95, tolerance=0.01)
    run = _make_run("Accuracy: 0.80\n")
    result = verify_run(spec, run)
    assert result.status == Status.FAIL
    assert result.strategy_used == "numeric_comparison"


def test_numeric_missing_extract_regex_fails():
    spec = _spec_with_numeric(extract_regex=None, expected=0.95, tolerance=0.01)
    run = _make_run("Accuracy: 0.95\n")
    result = verify_run(spec, run)
    assert result.status == Status.FAIL
    assert result.strategy_used == "numeric_comparison"


def test_numeric_no_regex_match_fails():
    spec = _spec_with_numeric(extract_regex=r"Accuracy:\s*([0-9.]+)", expected=0.95, tolerance=0.01)
    run = _make_run("Nothing interesting here.\n")
    result = verify_run(spec, run)
    assert result.status == Status.FAIL
    assert result.strategy_used == "numeric_comparison"


def test_numeric_unparseable_capture_fails():
    spec = _spec_with_numeric(extract_regex=r"Accuracy:\s*(\S+)", expected=0.95, tolerance=0.01)
    run = _make_run("Accuracy: not_a_number\n")
    result = verify_run(spec, run)
    assert result.status == Status.FAIL
    assert result.strategy_used == "numeric_comparison"


def test_numeric_zero_tolerance_exact_match_passes():
    spec = _spec_with_numeric(extract_regex=r"val=(\S+)", expected=42.0, tolerance=0.0)
    run = _make_run("val=42.0\n")
    result = verify_run(spec, run)
    assert result.status == Status.PASS


def test_numeric_invalid_regex_errors():
    spec = _spec_with_numeric(extract_regex=r"(unbalanced", expected=1.0, tolerance=0.0)
    run = _make_run("val=1.0\n")
    result = verify_run(spec, run)
    assert result.status == Status.ERROR
    assert result.strategy_used == "numeric_comparison"


def test_numeric_tolerance_null_errors():
    spec = {
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "numeric_comparison",
                    "extract_regex": r"val=(\S+)",
                    "expected": 1.0,
                    "tolerance": None,
                }
            ],
        }
    }
    run = _make_run("val=1.0\n")
    result = verify_run(spec, run)
    assert result.status == Status.ERROR
    assert result.strategy_used == "numeric_comparison"
    assert "tolerance" in result.details.lower()


def test_numeric_expected_non_numeric_errors():
    spec = {
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "numeric_comparison",
                    "extract_regex": r"val=(\S+)",
                    "expected": "abc",
                    "tolerance": 0.0,
                }
            ],
        }
    }
    run = _make_run("val=1.0\n")
    result = verify_run(spec, run)
    assert result.status == Status.ERROR
    assert result.strategy_used == "numeric_comparison"
    assert "expected" in result.details.lower()
