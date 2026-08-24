"""Gate 2 finding 5 (2026-08-12): the four bounded agentic-repair protocol
deliverables (config, schema, dry-run command, redacted dry-run trajectory)
must be staged into the release artifact.

Plan Task 10 requires them in the release artifact
(docs/superpowers/plans/2026-08-10-parbench-final-engineering.md, Files
block). Both artifact builders stage them through the shared helper
``scripts/artifact_protocol_deliverables.sh``; this test materializes the
helper's staging into a temp root and asserts the archive contents.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HELPER = PROJECT_ROOT / "scripts" / "artifact_protocol_deliverables.sh"

DELIVERABLES = [
    "config/agentic_repair_protocol.json",
    "schema/agentic_repair_protocol_schema.json",
    "scripts/evaluation/dry_run_agentic_protocol.py",
    "results/analysis/agentic_protocol_dry_run.json",
]


def test_helper_materializes_all_four_deliverables(tmp_path):
    staging = tmp_path / "staging"
    proc = subprocess.run(
        ["bash", str(HELPER), str(PROJECT_ROOT), str(staging)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for rel in DELIVERABLES:
        staged = staging / rel
        assert staged.is_file(), f"missing deliverable in archive: {rel}"
        assert staged.read_bytes() == (PROJECT_ROOT / rel).read_bytes(), (
            f"staged copy differs from repo copy: {rel}"
        )


def test_staged_trajectory_passes_check_complete_against_staged_config(tmp_path):
    # The staged archive must be internally consistent: the staged trajectory
    # validates (including the config-hash binding) against the staged config.
    staging = tmp_path / "staging"
    proc = subprocess.run(
        ["bash", str(HELPER), str(PROJECT_ROOT), str(staging)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluation/dry_run_agentic_protocol.py"),
            "--protocol",
            str(staging / "config/agentic_repair_protocol.json"),
            "--schema",
            str(staging / "schema/agentic_repair_protocol_schema.json"),
            "--output",
            str(staging / "results/analysis/agentic_protocol_dry_run.json"),
            "--check-complete",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_both_artifact_builders_invoke_the_shared_helper():
    for builder in ("build_artifact.sh", "build_artifact_zip.sh"):
        text = (PROJECT_ROOT / "scripts" / builder).read_text(encoding="utf-8")
        assert "artifact_protocol_deliverables.sh" in text, (
            f"scripts/{builder} does not stage the protocol deliverables"
        )
