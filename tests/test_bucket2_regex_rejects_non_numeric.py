"""FLAG-1 fix: bucket2 stdout_pattern regexes must require a leading digit.

The S5 Batch 3 regex-tighten upgraded `Accuracy:` / `Max error between host
and device:` to capture `([0-9.eE+-]+)`. That character class matches bare
`-` or `+` — so `printf("%e", 0.0/0.0)` → `-nan` still PASSES under
`re.search` because stdout_pattern does not float-parse the capture. Under
the bucket2 contract ("narrowing to require a numeric field makes it strong"),
this is insufficient.

The tightened pattern `([+-]?\\d[\\d.eE+-]*)` requires at least one digit
after the optional sign — rejecting `-nan`, `nan`, `inf`, bare sign — while
still matching every realistic IEEE-754 fp rendering the benchmarks emit.
"""

from __future__ import annotations

from harness.models import RunResult, Status
from harness.verifier import verify_run

_OLD_ACCURACY = r"Accuracy:\s+([0-9.eE+-]+)"
_NEW_ACCURACY = r"Accuracy:\s+([+-]?\d[\d.eE+-]*)"
_OLD_MAX_ERROR = r"Max error between host and device:\s+([0-9.eE+-]+)"
_NEW_MAX_ERROR = r"Max error between host and device:\s+([+-]?\d[\d.eE+-]*)"


def _run(stdout: str) -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.0,
        exit_code=0,
        stdout=stdout,
        stderr="",
    )


def _verify(pattern: str, stdout: str) -> Status:
    spec = {"verification": {"strategies": [{"type": "stdout_pattern", "pattern": pattern}]}}
    return verify_run(spec, _run(stdout)).status


def test_old_regex_false_passes_on_negative_nan() -> None:
    """Documents the pre-fix bug: `-nan` matches the old class-only regex."""
    assert _verify(_OLD_ACCURACY, "Accuracy: -nan\n") == Status.PASS
    assert _verify(_OLD_MAX_ERROR, "Max error between host and device: -nan\n") == Status.PASS


def test_new_regex_rejects_negative_nan() -> None:
    """New regex requires a leading digit — `-nan` no longer matches."""
    assert _verify(_NEW_ACCURACY, "Accuracy: -nan\n") == Status.FAIL
    assert _verify(_NEW_MAX_ERROR, "Max error between host and device: -nan\n") == Status.FAIL


def test_new_regex_rejects_nan_inf_bare_sign() -> None:
    for stdout in ("Accuracy: nan\n", "Accuracy: inf\n", "Accuracy: -inf\n", "Accuracy: +\n", "Accuracy: -\n"):
        assert _verify(_NEW_ACCURACY, stdout) == Status.FAIL, stdout


def test_new_regex_matches_real_binary_output() -> None:
    """Real outputs from fresh binary runs."""
    assert _verify(_NEW_ACCURACY, "Time: 0.007 (s)\nAccuracy: 4.096975e-05\n") == Status.PASS
    assert _verify(_NEW_MAX_ERROR, "Max error between host and device: 1.86265e-09\n") == Status.PASS


def test_new_regex_matches_edge_fp_forms() -> None:
    for stdout in (
        "Accuracy: 0\n",
        "Accuracy: -0\n",
        "Accuracy: 1\n",
        "Accuracy: -1.5\n",
        "Accuracy: 0.000000e+00\n",
        "Accuracy: 1e-20\n",
        "Accuracy: 1.5E+10\n",
    ):
        assert _verify(_NEW_ACCURACY, stdout) == Status.PASS, stdout
