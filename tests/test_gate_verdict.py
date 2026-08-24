"""Tests for scripts/orchestration/gate_verdict.sh (rule 8 follow-up).

The point of the extraction: the 2026-08-12 incident (the conveyor matched
GATE VERDICT: PASS inside Codex's echoed prompt and sealed parbench-final-v1
with open findings) was fixed inside a script that cannot run on this Mac, so
the fix was untestable. The helper makes the last-marker parse testable.
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "orchestration" / "gate_verdict.sh"

PROMPT_ECHO = (
    "You must end with either GATE VERDICT: PASS or GATE VERDICT: FINDINGS\n"
    "some codex reasoning here\n"
)


def run_verdict(log: Path, prefix: str, wanted: str) -> int:
    return subprocess.run(
        ["bash", str(SCRIPT), str(log), prefix, wanted], capture_output=True
    ).returncode


@pytest.mark.parametrize("prefix,wanted,other", [
    ("GATE VERDICT", "PASS", "FINDINGS"),
    ("TASK 3 RESULT", "GREEN", "RED"),
], ids=["gate-verdict", "task-result"])
class TestGateVerdict:
    def test_incident_shape_prompt_echo_then_findings(self, tmp_path, prefix, wanted, other):
        # The 2026-08-12 shape: the wanted word appears early inside an echoed
        # prompt; the real, last verdict is the other word. Must be non-zero.
        log = tmp_path / "gate.log"
        log.write_text(
            f"{prefix}: {wanted} appears inside the echoed prompt\n"
            + PROMPT_ECHO
            + f"{prefix}: {other}\n"
        )
        assert run_verdict(log, prefix, wanted) != 0

    def test_genuine_pass(self, tmp_path, prefix, wanted, other):
        log = tmp_path / "gate.log"
        log.write_text(PROMPT_ECHO + f"real work\n{prefix}: {wanted}\n")
        assert run_verdict(log, prefix, wanted) == 0

    def test_no_verdict_line(self, tmp_path, prefix, wanted, other):
        log = tmp_path / "gate.log"
        log.write_text("work happened but no verdict was printed\n")
        assert run_verdict(log, prefix, wanted) != 0

    def test_missing_log_is_exit_2(self, tmp_path, prefix, wanted, other):
        assert run_verdict(tmp_path / "absent.log", prefix, wanted) == 2

    def test_empty_log_is_exit_2(self, tmp_path, prefix, wanted, other):
        log = tmp_path / "gate.log"
        log.write_text("")
        assert run_verdict(log, prefix, wanted) == 2
