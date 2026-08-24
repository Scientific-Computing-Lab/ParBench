"""Tests for scripts/spec_tools/check_spec_argc.py (rules 13+14d merged package).

Fixture specs exercise the three outcomes (OK / MISMATCH / SKIP flag-parsed);
one test runs --all against the real corpus and asserts zero MISMATCH. The
26/26/8 Rodinia split is a fact about today's corpus and is deliberately not
pinned.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "spec_tools" / "check_spec_argc.py"
PYTHON = sys.executable


def make_spec(root: Path, name: str, arguments, source: str) -> Path:
    src_dir = root / "tree" / name
    src_dir.mkdir(parents=True)
    (src_dir / "main.c").write_text(source)
    spec = {
        "identity": {"unique_id": f"fix-{name}-omp"},
        "files": {"prompt_payload": ["main.c"]},
        "provenance": {"repo_root": "tree", "source_path": name},
        "run": {"input_configurations": {"correctness": {"arguments": arguments}}},
    }
    spec_path = root / f"fix-{name}-omp.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path


def run_checker(*args: str, known: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if known is not None:
        env["CHECK_SPEC_ARGC_KNOWN_JSON"] = json.dumps(known)
    return subprocess.run(
        [PYTHON, str(SCRIPT), *args], capture_output=True, text=True, env=env
    )


# The synthetic entry the mechanism tests inject: the shape of the real
# hecbench-myocyte-omp finding (found 2026-08-15, resolved by spec fix
# 2026-08-16), kept as the canonical fixture.
FIXTURE_KNOWN = {
    "hecbench-myocyte-omp": {
        "detail": "main.c checks `argc == 3` but spec provides 1 arguments (argc=2)",
        "note": "synthetic test entry",
    },
}


def test_ok_when_argc_matches(tmp_path):
    spec = make_spec(
        tmp_path, "good", ["a", "b", "c"],
        "int main(int argc, char **argv) { if (argc != 4) return 1; return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_mismatch_exits_1(tmp_path):
    spec = make_spec(
        tmp_path, "bad", ["a", "b"],
        "int main(int argc, char **argv) { if (argc != 8) return 1; return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "MISMATCH" in proc.stdout


def test_flag_parsed_is_skipped(tmp_path):
    spec = make_spec(
        tmp_path, "flags", ["-boxes1d", "10"],
        "int main(int argc, char **argv) { if (argc != 3) return 1; return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIP (flag-parsed)" in proc.stdout


def test_no_argc_check_is_skipped(tmp_path):
    spec = make_spec(
        tmp_path, "noargc", ["a"],
        "int main(int argc, char **argv) { return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIP (no-argc)" in proc.stdout


def test_lower_bound_guard_satisfied(tmp_path):
    # `if (argc < 3) usage()` guards a minimum; more args than the minimum is OK.
    spec = make_spec(
        tmp_path, "lower", ["a", "b", "c", "d"],
        "int main(int argc, char **argv) { if (argc < 3) return 1; return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_non_utf8_source_does_not_crash(tmp_path):
    # Python-side twin of the grep -ac rule: myocyte's sources are non-ISO
    # extended ASCII; a strict UTF-8 read raises.
    src_dir = tmp_path / "tree" / "enc"
    src_dir.mkdir(parents=True)
    (src_dir / "main.c").write_bytes(
        b"/* caf\xe9 */\nint main(int argc, char **argv) "
        b"{ if (argc != 2) return 1; return 0; }\n"
    )
    spec = {
        "identity": {"unique_id": "fix-enc-omp"},
        "files": {"prompt_payload": ["main.c"]},
        "provenance": {"repo_root": "tree", "source_path": "enc"},
        "run": {"input_configurations": {"correctness": {"arguments": ["a"]}}},
    }
    spec_path = tmp_path / "fix-enc-omp.json"
    spec_path.write_text(json.dumps(spec))
    proc = run_checker("--spec", str(spec_path), "--project-root", str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_missing_source_is_skipped_not_crashed(tmp_path):
    spec = make_spec(tmp_path, "gone", ["a"], "unused\n")
    (tmp_path / "tree" / "gone" / "main.c").unlink()
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIP (source-missing)" in proc.stdout


def make_named_spec(root: Path, spec_id: str, arguments, source: str) -> Path:
    src_dir = root / "tree" / spec_id
    src_dir.mkdir(parents=True)
    (src_dir / "main.c").write_text(source)
    spec = {
        "identity": {"unique_id": spec_id},
        "files": {"prompt_payload": ["main.c"]},
        "provenance": {"repo_root": "tree", "source_path": spec_id},
        "run": {"input_configurations": {"correctness": {"arguments": arguments}}},
    }
    spec_path = root / f"{spec_id}.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path


def test_known_mismatch_requires_exact_detail(tmp_path):
    # Reproduce the recorded hecbench-myocyte-omp defect shape exactly:
    # `argc == 3` against 1 argument. Must print MISMATCH (known) and exit 0.
    spec = make_named_spec(
        tmp_path, "hecbench-myocyte-omp", ["10"],
        "int main(int argc, char **argv) { if(argc==3){ return 0; } return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path),
                       known=FIXTURE_KNOWN)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MISMATCH (known)" in proc.stdout


def test_different_defect_on_known_spec_is_a_real_mismatch(tmp_path):
    # 2026-08-16 review finding: the allowlist must not absorb a DIFFERENT
    # future mismatch on the same spec id.
    spec = make_named_spec(
        tmp_path, "hecbench-myocyte-omp", ["a", "b", "c", "d"],
        "int main(int argc, char **argv) { if(argc==3){ return 0; } return 0; }\n",
    )
    proc = run_checker("--spec", str(spec), "--project-root", str(tmp_path),
                       known=FIXTURE_KNOWN)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "treating as a new defect" in proc.stdout


def test_stale_known_entry_fails_under_all(tmp_path):
    # An --all run in which a KNOWN_MISMATCHES entry never fires means the
    # spec was fixed; the stale entry must fail the run until removed.
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    src_dir = tmp_path / "tree" / "clean"
    src_dir.mkdir(parents=True)
    (src_dir / "main.c").write_text(
        "int main(int argc, char **argv) { if (argc != 2) return 1; return 0; }\n"
    )
    (specs_dir / "fix-clean-omp.json").write_text(json.dumps({
        "identity": {"unique_id": "fix-clean-omp"},
        "files": {"prompt_payload": ["main.c"]},
        "provenance": {"repo_root": "tree", "source_path": "clean"},
        "run": {"input_configurations": {"correctness": {"arguments": ["a"]}}},
    }))
    proc = run_checker("--all", "--project-root", str(tmp_path),
                       known=FIXTURE_KNOWN)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "STALE known-mismatch entry" in proc.stdout


def test_real_corpus_has_zero_unexpected_mismatches():
    proc = run_checker("--all", "--project-root", str(REPO_ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Known mismatches (a real finding awaiting an owner ruling) print loudly
    # as "MISMATCH (known)"; any other MISMATCH line is a new defect.
    unexpected = [
        line for line in proc.stdout.splitlines()
        if line.startswith("MISMATCH") and not line.startswith("MISMATCH (known)")
    ]
    assert unexpected == [], unexpected
    # The SKIP lists are a manual-review surface and must print in full.
    assert "SKIP (flag-parsed)" in proc.stdout
