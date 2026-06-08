"""Functional smoke test: `harness verify` CLI path executes without KeyError.

Complements test_verifier_caller_contract.py (static grep). This test exercises
cmd_verify() end-to-end with stubbed build/run to catch plumbing regressions
(e.g. the S1.6 `resolved["working_dir"]` vs `resolved["_resolved"]["working_dir"]`
KeyError that the grep contract test missed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from harness import cli
from harness.models import BuildResult, RunResult, Status

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = PROJECT_ROOT / "specs" / "rodinia-bfs-cuda.json"


def _fake_build(spec, project_root, verbose=False):
    return BuildResult(
        status=Status.PASS, duration_seconds=0.0, stdout="", stderr=""
    )


def _fake_run(spec, project_root, configuration="correctness", verbose=False):
    return RunResult(
        status=Status.PASS,
        configuration=configuration,
        duration_seconds=0.0,
        exit_code=0,
        stdout="Result stored in result.txt\n",
        stderr="",
    )


def test_cmd_verify_threads_resolved_working_dir(monkeypatch, capsys) -> None:
    """`cmd_verify` must reach `verify_run` without raising KeyError.

    The pre-fix bug was `resolved["working_dir"]` instead of
    `resolved["_resolved"]["working_dir"]`. With the fix, this call returns 0
    (stdout_pattern matches) and never raises.
    """
    monkeypatch.setattr(cli, "build_spec", _fake_build)
    monkeypatch.setattr(cli, "run_spec", _fake_run)

    args = argparse.Namespace(
        project_root=str(PROJECT_ROOT),
        spec_file=str(SPEC_PATH),
        config="correctness",
        verbose=False,
        json=False,
        manifest="manifest.jsonl",
    )

    rc = cli.cmd_verify(args)
    assert rc == 0, "cmd_verify should return 0 when stdout_pattern matches"
