"""Tests for scripts/lib/artifact_release_common.sh.

The helper is the shared source of truth for the two release artifact builders
(scripts/build_artifact.sh and scripts/build_artifact_zip.sh). It holds the three
concerns that are dangerous to let drift between the builders: the five common
flags, the nonempty-output provenance guard, and a create-and-own mktemp scratch
whose cleanup trap removes only the dirs it created.

These tests source the real helper in a bash subprocess and assert on exit codes
and side effects. They build only throwaway fixture dirs under ``tmp_path`` and
never touch the live specs/ tree, never build an archive, and never publish.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "artifact_release_common.sh"


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet with the helper already sourced; return the result."""
    full = f'set -uo pipefail\nsource "{HELPER}"\n{script}'
    return subprocess.run(
        ["bash", "-c", full],
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------- #
# 1. Flag parsing.
# --------------------------------------------------------------------------- #

def test_parse_all_flags_and_extras() -> None:
    r = _bash(
        "release_parse_flags --staging-dir /s --result-root /r "
        "--evidence-manifest /e --output /o --resume --dry-run leftover\n"
        'printf "%s|%s|%s|%s|%s|%s|%s\\n" '
        '"$STAGING_DIR" "$RESULT_ROOT" "$EVIDENCE_MANIFEST" "$OUTPUT" '
        '"$RESUME" "${RELEASE_EXTRA_ARGS[0]}" "${RELEASE_EXTRA_ARGS[1]}"'
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "/s|/r|/e|/o|true|--dry-run|leftover"


def test_parse_defaults_empty() -> None:
    r = _bash(
        "release_parse_flags\n"
        'printf "[%s][%s][%s][%s]\\n" '
        '"$STAGING_DIR" "$RESULT_ROOT" "$OUTPUT" "$RESUME"'
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[][][][false]"


def test_parse_missing_value_returns_2() -> None:
    r = _bash("release_parse_flags --output")
    assert r.returncode == 2


# --------------------------------------------------------------------------- #
# 2. Output provenance guard.
# --------------------------------------------------------------------------- #

def test_guard_absent_output_ok(tmp_path: Path) -> None:
    r = _bash(f'release_guard_output "{tmp_path}/none.zip" false /r /e')
    assert r.returncode == 0, r.stderr


def test_guard_empty_output_ok(tmp_path: Path) -> None:
    (tmp_path / "empty.zip").write_text("")
    r = _bash(f'release_guard_output "{tmp_path}/empty.zip" false /r /e')
    assert r.returncode == 0, r.stderr


def test_guard_nonempty_no_resume_refuses(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_text("data")
    r = _bash(f'release_guard_output "{tmp_path}/a.zip" false /r /e')
    assert r.returncode == 1


def test_guard_resume_without_sidecar_refuses(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_text("data")
    r = _bash(f'release_guard_output "{tmp_path}/a.zip" true /r /e')
    assert r.returncode == 2


def test_guard_resume_matching_provenance_ok(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_text("data")
    r = _bash(
        f'release_write_prov "{tmp_path}/a.zip" /r /e\n'
        f'release_guard_output "{tmp_path}/a.zip" true /r /e'
    )
    assert r.returncode == 0, r.stderr
    # The sidecar is written next to the archive with the recorded provenance.
    sidecar = tmp_path / "a.zip.provenance"
    assert sidecar.read_text() == "result_root=/r\nevidence_manifest=/e\n"


def test_guard_resume_mismatch_refuses(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_text("data")
    r = _bash(
        f'release_write_prov "{tmp_path}/a.zip" /r /e\n'
        f'release_guard_output "{tmp_path}/a.zip" true /DIFFERENT /e'
    )
    assert r.returncode == 2


# --------------------------------------------------------------------------- #
# 3. Create-and-own scratch + cleanup trap.
# --------------------------------------------------------------------------- #

def test_scratch_trap_removes_only_owned_dir(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "keep").write_text("keep")
    pathfile = tmp_path / "scratchpath"

    # The subshell exits -> its EXIT trap should remove the owned scratch dir.
    r = _bash(
        "( release_make_scratch SC parbench-test\n"
        f'  echo "$SC" > "{pathfile}"\n'
        "  [ -d \"$SC\" ] || exit 9\n"
        ")"
    )
    assert r.returncode == 0, r.stderr
    scratch = Path(pathfile.read_text().strip())
    assert not scratch.exists(), "owned scratch should be trap-removed"
    assert (sentinel / "keep").exists(), "sentinel must be untouched"


def test_safe_rmtree_refuses_dangerous_paths() -> None:
    # Root, HOME and a relative path must all be spared / refused.
    r = _bash(
        '_release_safe_rmtree "/"\n'
        '_release_safe_rmtree "$HOME"\n'
        '_release_safe_rmtree "relative/path"\n'
        '[ -d / ] && [ -d "$HOME" ] && echo SPARED'
    )
    assert r.returncode == 0, r.stderr
    assert "SPARED" in r.stdout
