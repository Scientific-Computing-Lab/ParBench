"""Tests for the pair-contract pass of scripts/spec_tools/check_spec_argc.py.

OD-3 (ruled 2026-08-20): a pair contract's translated_run.args is validated
against the argc checks of the tree that actually runs - the TARGET spec's
source for a kernel-only pair (target-native host binary runs), the SOURCE
spec's source for a full-program pair (model keeps source argument parsing).

Synthetic fixtures, like the per-spec tests: each test writes a spec root plus
a schema-valid registry, then shells out to --contracts. One production test
pins the 2 real streamcluster findings against the live corpus via the
KNOWN_CONTRACT_ARGC_MISMATCHES allowlist (exit 0, known_mismatches=2).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "spec_tools" / "check_spec_argc.py"
PYTHON = sys.executable


def write_spec(
    spec_root: Path, tree_root: Path, uid: str, *,
    targets: list[str], payload: str, source_body: str,
) -> None:
    """Write a synthetic spec and its single payload source.

    ``targets`` populates files.translation_targets (drives the kernel-only
    predicate); ``payload`` is the one prompt_payload source the argc scan
    reads, written with ``source_body`` under <tree_root>/tree/<uid>/.
    """
    src_dir = tree_root / "tree" / uid
    src_dir.mkdir(parents=True)
    (src_dir / payload).write_text(source_body)
    spec = {
        "identity": {"unique_id": uid},
        "files": {"prompt_payload": [payload], "translation_targets": targets},
        "provenance": {"repo_root": "tree", "source_path": uid},
    }
    (spec_root / f"{uid}.json").write_text(json.dumps(spec))


def make_pair(source: str, target: str, args: list[str]) -> dict:
    """A minimal schema-valid pair entry carrying translated_run.args."""
    return {
        "source_spec": source,
        "target_spec": target,
        "comparability": "eligible",
        "baseline_witness": {"status": "pass"},
        "translated_run": {"args": args},
    }


def write_registry(path: Path, pairs: list[dict]) -> None:
    path.write_text(json.dumps(
        {"contract_version": "parbench-final-v1", "pairs": pairs}
    ))


def run_contracts(
    spec_root: Path, registry: Path, project_root: Path,
    *, known: dict | None = None,
):
    """Shell out to --contracts. ``known`` injects a synthetic
    KNOWN_CONTRACT_ARGC_MISMATCHES allowlist via the script's test seam;
    an empty dict clears the production allowlist."""
    env = dict(os.environ)
    env["CHECK_CONTRACT_ARGC_KNOWN_JSON"] = json.dumps(known or {})
    return subprocess.run(
        [
            PYTHON, str(SCRIPT), "--contracts",
            "--registry", str(registry),
            "--spec-root", str(spec_root),
            "--project-root", str(project_root),
        ],
        capture_output=True, text=True, env=env,
    )


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    spec_root = tmp_path / "specs"
    spec_root.mkdir()
    return spec_root, tmp_path


# --- (a) matching count -> OK/PASS ----------------------------------------- #

def test_matching_count_is_ok(tmp_path):
    spec_root, root = _setup(tmp_path)
    # full-program pair: source governs; source checks argc != 3 (2 args)
    write_spec(spec_root, tmp_path, "fix-a-cuda", targets=["a.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=3)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-a-omp", targets=["a.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=3)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-a-cuda", "fix-a-omp", ["x", "y"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK fix-a-cuda -> fix-a-omp" in proc.stdout
    assert "mismatches=0" in proc.stdout


# --- (b) mismatched count -> flagged, exit 1 ------------------------------- #

def test_mismatched_count_is_flagged(tmp_path):
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-b-cuda", targets=["b.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=8)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-b-omp", targets=["b.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=8)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-b-cuda", "fix-b-omp", ["x", "y"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "MISMATCH fix-b-cuda -> fix-b-omp" in proc.stdout
    assert "contract declares 2 arguments (argc=3)" in proc.stdout
    assert "mismatches=1" in proc.stdout


# --- (c) flag-parsed governing source -> SKIP ------------------------------ #

def test_flag_parsed_args_are_skipped(tmp_path):
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-c-cuda", targets=["c.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=3)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-c-omp", targets=["c.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=3)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-c-cuda", "fix-c-omp", ["-boxes1d", "10"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIP (flag-parsed) fix-c-cuda -> fix-c-omp" in proc.stdout


# --- (d) kernel-only pair governed by TARGET source ------------------------ #

def test_kernel_only_pair_is_governed_by_target(tmp_path):
    """Kernel-only target (.cl): the TARGET's host parser governs.

    Constructed so SOURCE and TARGET disagree - the args PASS the target
    parser but would FAIL the source parser. Asserting OK proves target
    governs; a second run with a source-only-passing arg count proves the
    same from the other side.
    """
    spec_root, root = _setup(tmp_path)
    # source host wants exactly 3 args (argc==4); target host wants exactly 1 (argc==2)
    write_spec(spec_root, tmp_path, "fix-d-omp", targets=["d.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=4)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-d-opencl", targets=["Kernels.cl"], payload="host.cpp",
               source_body="int main(int argc,char**argv){if(argc!=2)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    # 1 arg -> argc=2: OK for target (argc==2), MISMATCH for source (argc==4)
    write_registry(reg, [make_pair("fix-d-omp", "fix-d-opencl", ["only"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK fix-d-omp -> fix-d-opencl" in proc.stdout
    assert "governed by target fix-d-opencl (kernel_only)" in proc.stdout

    # Flip: 3 args -> argc=4 satisfies the SOURCE parser but MISMATCHes the
    # target parser; the check must flag it because target governs.
    write_registry(reg, [make_pair("fix-d-omp", "fix-d-opencl", ["a", "b", "c"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "MISMATCH fix-d-omp -> fix-d-opencl" in proc.stdout
    assert "governed by target fix-d-opencl (kernel_only)" in proc.stdout


# --- (e) full-program pair governed by SOURCE source ----------------------- #

def test_full_program_pair_is_governed_by_source(tmp_path):
    """Full-program target (.cu/.c): the SOURCE parser governs.

    Args PASS the source parser but would FAIL the target parser; asserting
    OK proves source governs.
    """
    spec_root, root = _setup(tmp_path)
    # source host wants exactly 1 arg (argc==2); target host wants exactly 3 (argc==4)
    write_spec(spec_root, tmp_path, "fix-e-cuda", targets=["e.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=2)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-e-omp", targets=["e.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=4)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    # 1 arg -> argc=2: OK for source (argc==2), MISMATCH for target (argc==4)
    write_registry(reg, [make_pair("fix-e-cuda", "fix-e-omp", ["only"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK fix-e-cuda -> fix-e-omp" in proc.stdout
    assert "governed by source fix-e-cuda (full_program)" in proc.stdout


# --- edge cases ------------------------------------------------------------ #

def test_missing_spec_fails_closed(tmp_path):
    """A contract referencing an absent spec is an integrity error, not a
    successful skip (2026-08-21 Codex finding)."""
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-f-cuda", targets=["f.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=2)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    # target spec is absent from the spec root
    write_registry(reg, [make_pair("fix-f-cuda", "fix-f-omp", ["x"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "SPEC-MISSING fix-f-cuda -> fix-f-omp" in proc.stdout
    assert "fix-f-omp not in spec root" in proc.stdout
    assert "spec_missing=1" in proc.stdout


def test_no_argc_check_source_is_skipped(tmp_path):
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-g-cuda", targets=["g.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-g-omp", targets=["g.c"], payload="main.c",
               source_body="int main(int argc,char**argv){return 0;}\n")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-g-cuda", "fix-g-omp", ["x"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIP (no-argc) fix-g-cuda -> fix-g-omp" in proc.stdout


def test_bad_registry_fails_closed(tmp_path):
    """Checker-cannot-run is exit 2, distinct from finding-bearing exit 1."""
    spec_root, root = _setup(tmp_path)
    reg = tmp_path / "reg.json"
    reg.write_text("{not valid json")
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "FAIL: cannot load pair-contract registry" in proc.stdout


def test_absent_spec_root_fails_closed(tmp_path):
    """A nonexistent spec root must never read as an all-skipped success."""
    _spec_root, root = _setup(tmp_path)
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-x-cuda", "fix-x-omp", ["x"])])
    proc = run_contracts(tmp_path / "no-such-specs", reg, root)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "spec root" in proc.stdout


def test_malformed_spec_json_fails_closed(tmp_path):
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-h-cuda", targets=["h.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=2)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-h-omp", targets=["h.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=2)return 1;return 0;}\n")
    (spec_root / "broken.json").write_text("{not valid json")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-h-cuda", "fix-h-omp", ["x"])])
    proc = run_contracts(spec_root, reg, root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "SPEC-INDEX malformed spec JSON broken.json" in proc.stdout
    assert "index_problems=1" in proc.stdout


# --- allowlist mechanism (KNOWN_CONTRACT_ARGC_MISMATCHES) ------------------ #

def _mismatching_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    """A full-program pair whose contract mismatches the governing source.

    Returns (spec_root, registry, root, exact_detail) so allowlist tests can
    pin the detail string the checker prints.
    """
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-k-cuda", targets=["k.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=8)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-k-omp", targets=["k.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=8)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-k-cuda", "fix-k-omp", ["x", "y"])])
    detail = ("governed by source fix-k-cuda (full_program): main.c checks "
              "`argc != 8` but contract declares 2 arguments (argc=3)")
    return spec_root, reg, root, detail


def test_known_mismatch_is_absorbed(tmp_path):
    spec_root, reg, root, detail = _mismatching_fixture(tmp_path)
    known = {"fix-k-cuda -> fix-k-omp": {"detail": detail, "note": "synthetic"}}
    proc = run_contracts(spec_root, reg, root, known=known)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MISMATCH (known) fix-k-cuda -> fix-k-omp" in proc.stdout
    assert "known_mismatches=1" in proc.stdout
    assert "mismatches=0" in proc.stdout


def test_different_defect_is_not_absorbed(tmp_path):
    """Same pair, different detail: the allowlist must not absorb it."""
    spec_root, reg, root, _detail = _mismatching_fixture(tmp_path)
    known = {"fix-k-cuda -> fix-k-omp": {"detail": "some other defect", "note": "n"}}
    proc = run_contracts(spec_root, reg, root, known=known)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "treating as a new defect" in proc.stdout
    assert "mismatches=1" in proc.stdout


def test_stale_allowlist_entry_fails(tmp_path):
    """An allowlist entry that no longer fires must fail until removed."""
    spec_root, root = _setup(tmp_path)
    write_spec(spec_root, tmp_path, "fix-m-cuda", targets=["m.cu"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=3)return 1;return 0;}\n")
    write_spec(spec_root, tmp_path, "fix-m-omp", targets=["m.c"], payload="main.c",
               source_body="int main(int argc,char**argv){if(argc!=3)return 1;return 0;}\n")
    reg = tmp_path / "reg.json"
    write_registry(reg, [make_pair("fix-m-cuda", "fix-m-omp", ["x", "y"])])
    known = {"fix-m-cuda -> fix-m-omp": {"detail": "was fixed", "note": "n"}}
    proc = run_contracts(spec_root, reg, root, known=known)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "STALE known-mismatch entry: fix-m-cuda -> fix-m-omp" in proc.stdout
    assert "stale_known=1" in proc.stdout


# --- production corpus pins ------------------------------------------------ #

def _summary_counts(stdout: str) -> dict[str, int]:
    """Parse the final summary line into exact integer fields, so a count of
    20 can never satisfy an assertion meant for 2."""
    for line in stdout.splitlines():
        if line.startswith("contracts_checked="):
            return {k: int(v) for k, v in
                    (part.split("=") for part in line.split())}
    raise AssertionError(f"no summary line in output:\n{stdout[-2000:]}")


def test_production_current_registry_is_clean():
    """The CURRENT registry (config/pair_contracts_v2.json, the checker's
    default) is fully clean: exit 0, no mismatches, no allowlist absorption.

    Runs WITHOUT the test seam so the production allowlist applies. Any new
    finding, stale allowlist entry, or registry integrity problem flips the
    exit code."""
    env = dict(os.environ)
    env.pop("CHECK_CONTRACT_ARGC_KNOWN_JSON", None)
    proc = subprocess.run(
        [PYTHON, str(SCRIPT), "--contracts"],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    counts = _summary_counts(proc.stdout)
    assert counts["mismatches"] == 0
    assert counts["known_mismatches"] == 0
    assert counts["spec_missing"] == 0
    assert counts["index_problems"] == 0


def test_frozen_v1_registry_still_shows_the_two_streamcluster_findings():
    """The FROZEN registry (config/final_pair_contracts.json) keeps the old
    source args BY DESIGN: its bytes are hash-bound into the sealed evidence
    package, so it is never regenerated. Audited explicitly, it must still
    show exactly the 2 historical streamcluster findings."""
    proc = run_contracts(
        REPO_ROOT / "specs",
        REPO_ROOT / "config" / "final_pair_contracts.json",
        REPO_ROOT,
    )
    assert proc.returncode == 1, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert _summary_counts(proc.stdout)["mismatches"] == 2
    assert ("MISMATCH rodinia-streamcluster-cuda -> "
            "rodinia-streamcluster-opencl") in proc.stdout
    assert ("MISMATCH rodinia-streamcluster-omp -> "
            "rodinia-streamcluster-opencl") in proc.stdout


def test_frozen_v1_registry_bytes_match_the_freeze_manifest():
    """Byte-identity guard: the frozen registry's sha256 must equal the hash
    recorded by the Task 7 contract freeze in
    config/final_contract_manifest.json. If this fails, someone edited the
    frozen artifact that the sealed evidence package hash-binds."""
    import hashlib
    frozen = REPO_ROOT / "config" / "final_pair_contracts.json"
    manifest = json.loads(
        (REPO_ROOT / "config" / "final_contract_manifest.json").read_text()
    )
    recorded = manifest["pair_registry"]["sha256"]
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == recorded


def test_v2_differs_from_v1_only_in_kernel_only_translated_run():
    """Pin the exact v1 -> v2 delta: same 204-pair set; exactly 22 pairs
    differ; every difference is in translated_run only; every changed pair
    is a kernel-only to-opencl pair whose v2 args equal the TARGET spec's
    correctness args."""
    v1 = {(p["source_spec"], p["target_spec"]): p for p in json.loads(
        (REPO_ROOT / "config" / "final_pair_contracts.json").read_text())["pairs"]}
    v2 = {(p["source_spec"], p["target_spec"]): p for p in json.loads(
        (REPO_ROOT / "config" / "pair_contracts_v2.json").read_text())["pairs"]}
    assert set(v1) == set(v2) and len(v1) == 204
    changed = []
    for key in v1:
        fields = {f for f in set(v1[key]) | set(v2[key])
                  if v1[key].get(f) != v2[key].get(f)}
        if fields:
            assert fields == {"translated_run"}, (key, fields)
            changed.append(key)
    assert len(changed) == 22, sorted(changed)
    for _src, tgt in changed:
        assert tgt.endswith("-opencl"), tgt
        tgt_spec = json.loads((REPO_ROOT / "specs" / f"{tgt}.json").read_text())
        expected = (tgt_spec["run"]["input_configurations"]
                    ["correctness"]["arguments"])
        assert v2[(_src, tgt)]["translated_run"]["args"] == expected
