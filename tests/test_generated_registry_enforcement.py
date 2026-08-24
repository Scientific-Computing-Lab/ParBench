"""Task 3 (gate E-03): enforcing --fail-uncovered mode for the generated-file registry.

check_generated_registry.py gains --fail-uncovered: any tracked-or-untracked file
changed by the current work under the caller-supplied directories that no registry
row covers is a failure (exit 1), while --uncovered stays advisory (exit 0).
A supplied directory that is missing or empty (e.g. results/augmentation_final,
whose producing task is deferred) is tolerated, never an error.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_generated_registry.py"


def run_script(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True, text=True,
    )


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".claude").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts" / "gen.py").write_text("# generator\n")
    (root / "results" / "analysis").mkdir(parents=True)
    (root / "results" / "analysis" / "covered.json").write_text("{}\n")
    (root / ".claude" / "generated-outputs.tsv").write_text(
        "results/analysis/covered.json\tscripts/gen.py\tblock\n"
    )
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    return root


def test_clean_repo_passes(repo: Path) -> None:
    proc = run_script(repo, "--fail-uncovered", "--changed-only",
                      "--dirs", "results/analysis")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_uncovered_untracked_file_fails(repo: Path) -> None:
    (repo / "results" / "analysis" / "new_output.json").write_text("{}\n")
    proc = run_script(repo, "--fail-uncovered", "--changed-only",
                      "--dirs", "results/analysis")
    assert proc.returncode == 1
    assert "new_output.json" in proc.stdout


def test_uncovered_modified_tracked_file_fails(repo: Path) -> None:
    stray = repo / "results" / "analysis" / "stray.md"
    stray.write_text("old\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "add stray")
    stray.write_text("changed by current work\n")
    proc = run_script(repo, "--fail-uncovered", "--changed-only",
                      "--dirs", "results/analysis")
    assert proc.returncode == 1
    assert "stray.md" in proc.stdout


def test_changed_only_ignores_unchanged_uncovered_files(repo: Path) -> None:
    stray = repo / "results" / "analysis" / "stray.md"
    stray.write_text("committed, not changed since\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "add stray")
    proc = run_script(repo, "--fail-uncovered", "--changed-only",
                      "--dirs", "results/analysis")
    assert proc.returncode == 0
    proc2 = run_script(repo, "--fail-uncovered", "--dirs", "results/analysis")
    assert proc2.returncode == 1


def test_covered_changed_file_passes(repo: Path) -> None:
    (repo / "results" / "analysis" / "covered.json").write_text('{"new": 1}\n')
    proc = run_script(repo, "--fail-uncovered", "--changed-only",
                      "--dirs", "results/analysis")
    assert proc.returncode == 0


def test_missing_and_empty_dirs_are_tolerated(repo: Path) -> None:
    (repo / "results" / "empty_dir").mkdir()
    proc = run_script(
        repo, "--fail-uncovered", "--changed-only",
        "--dirs", "results/analysis", "results/augmentation_final",
        "results/empty_dir",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_uncovered_flag_stays_advisory(repo: Path) -> None:
    """--uncovered keeps its pre-Task-3 compatibility contract: it lists
    uncovered *tracked* files and never fails on them."""
    (repo / "results" / "analysis" / "new_output.json").write_text("{}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "add uncovered output")
    proc = run_script(repo, "--uncovered", "--dirs", "results/analysis")
    assert proc.returncode == 0
    assert "new_output.json" in proc.stdout


def test_real_registry_still_validates() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
