"""Tests for scripts/spec_tools/apply_rodinia_patches.sh (rule 26 follow-up).

The script is exercised against fixture git repos in tmp_path via its --tree
and --patch overrides; the real Rodinia tree is never touched by these tests.

Exit contract: --check exits 0 when the patch would apply cleanly OR is already
applied, 1 when neither (conflict), 2 on guard failures (no .git, missing
patch, unresolvable tree). --apply refuses (exit 2) when the tree has
modifications outside the patch's own file set.
"""

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "spec_tools" / "apply_rodinia_patches.sh"
)

ORIGINAL = "CUDA_DIR = /usr/local/cuda\nOMP_LIB = -fopenmp\n"
PATCHED = "CUDA_DIR = /opt/nvidia/cuda\nOMP_LIB = -fopenmp\n"


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True
    )


def make_tree(root: Path) -> Path:
    tree = root / "tree"
    (tree / "common").mkdir(parents=True)
    (tree / "common" / "make.config").write_text(ORIGINAL)
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tree), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "base"],
        check=True,
    )
    return tree


def make_patch(root: Path, tree: Path) -> Path:
    (tree / "common" / "make.config").write_text(PATCHED)
    diff = subprocess.run(
        ["git", "-C", str(tree), "diff"], capture_output=True, text=True, check=True
    ).stdout
    subprocess.run(["git", "-C", str(tree), "checkout", "--", "."], check=True)
    patch = root / "toolchain.diff"
    patch.write_text(diff)
    return patch


def test_check_reports_would_apply_on_clean_tree(tmp_path):
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    proc = run_script("--check", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "would apply" in proc.stdout


def test_check_reports_already_applied(tmp_path):
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    (tree / "common" / "make.config").write_text(PATCHED)
    proc = run_script("--check", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "already applied" in proc.stdout


def test_check_conflict_exits_1(tmp_path):
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    (tree / "common" / "make.config").write_text("something else entirely\n")
    proc = run_script("--check", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_apply_then_check_roundtrip(tmp_path):
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    proc = run_script("--apply", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tree / "common" / "make.config").read_text() == PATCHED
    proc2 = run_script("--check", "--tree", str(tree), "--patch", str(patch))
    assert proc2.returncode == 0
    assert "already applied" in proc2.stdout


def test_apply_refuses_dirty_tree_outside_patch_set(tmp_path):
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    stray = tree / "common" / "stray.txt"
    stray.write_text("uncommitted work\n")
    subprocess.run(["git", "-C", str(tree), "add", str(stray)], check=True)
    proc = run_script("--apply", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert (tree / "common" / "make.config").read_text() == ORIGINAL


def test_apply_ignores_untracked_files(tmp_path):
    # The Mac layout's rodinia-src self-symlink always shows as untracked (??),
    # and git apply cannot clobber an untracked file - apply must proceed.
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    (tree / "rodinia-src-stand-in").write_text("untracked, must not block\n")
    proc = run_script("--apply", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tree / "common" / "make.config").read_text() == PATCHED


def test_relative_patch_path_resolves(tmp_path, monkeypatch):
    tree = make_tree(tmp_path)
    patch = make_patch(tmp_path, tree)
    monkeypatch.chdir(tmp_path)
    proc = run_script("--check", "--tree", str(tree), "--patch", patch.name)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "would apply" in proc.stdout


def test_missing_flag_value_is_usage_error():
    proc = run_script("--check", "--tree")
    assert proc.returncode == 2
    assert "usage" in proc.stderr


def test_tree_without_git_fails_loudly(tmp_path):
    tree = tmp_path / "bare"
    (tree / "common").mkdir(parents=True)
    (tree / "common" / "make.config").write_text(ORIGINAL)
    patch = tmp_path / "p.diff"
    patch.write_text("")
    proc = run_script("--check", "--tree", str(tree), "--patch", str(patch))
    assert proc.returncode == 2
    assert ".git" in proc.stderr


def test_missing_patch_fails_loudly(tmp_path):
    tree = make_tree(tmp_path)
    proc = run_script("--check", "--tree", str(tree),
                      "--patch", str(tmp_path / "absent.diff"))
    assert proc.returncode == 2
