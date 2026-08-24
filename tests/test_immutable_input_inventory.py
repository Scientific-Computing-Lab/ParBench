"""Tests for scripts/provenance/inventory_immutable_inputs.py and
make_private_archive.py: inventory round-trip, mutation detection, archive
create/check, and the tamper-refusal contract."""

import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
INVENTORY = REPO_ROOT / "scripts" / "provenance" / "inventory_immutable_inputs.py"
ARCHIVE = REPO_ROOT / "scripts" / "provenance" / "make_private_archive.py"


def run(*argv):
    return subprocess.run(
        [sys.executable, *map(str, argv)], capture_output=True, text=True
    )


def make_project(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "proj"
    (root / "specs").mkdir(parents=True)
    (root / "srctree" / "k").mkdir(parents=True)
    (root / "srctree" / "k" / "a.cu").write_text("__global__ void k() {}\n")
    (root / "srctree" / "k" / "data").mkdir()
    (root / "srctree" / "k" / "data" / "in.txt").write_text("1 2 3\n")
    (root / "specs" / "suite-a-cuda.json").write_text(json.dumps({
        "provenance": {"repo_root": "srctree", "source_path": "k"},
        "files": {
            "prompt_payload": ["a.cu"],
            "support_files": ["Makefile", "data"],  # Makefile missing; data is a dir
            "verification_only": [],
        },
    }))
    rec_dir = root / "results" / "evaluation" / "modelA"
    rec_dir.mkdir(parents=True)
    (rec_dir / "suite-a-cuda-to-suite-a-omp-s0.json").write_text(json.dumps({
        "overall_status": "PASS",
        "translated_files": {"a.cpp": "int main() { return 0; }\n"},
    }))
    return root


def make_manifest(root: pathlib.Path) -> pathlib.Path:
    out = root / "results" / "provenance" / "inventory.json"
    res = run(INVENTORY, "--project-root", root, "--output", out)
    assert res.returncode == 0, res.stderr + res.stdout
    return out


def test_inventory_contents_and_verify_roundtrip(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    manifest = json.loads(out.read_text())

    ev = manifest["evaluation_results"]
    assert ev["file_count"] == 1
    rec_key = "results/evaluation/modelA/suite-a-cuda-to-suite-a-omp-s0.json"
    assert rec_key in ev["files"] and len(ev["files"][rec_key]) == 64

    src = manifest["benchmark_sources"]["spec_referenced"]
    assert src["files"]["srctree/k/a.cu"] != "MISSING"
    assert src["files"]["srctree/k/Makefile"] == "MISSING"
    # a directory reference is inventoried file-by-file
    assert src["files"]["srctree/k/data/in.txt"] != "MISSING"
    assert src["hashed_count"] == 2 and src["missing_count"] == 1

    res = run(INVENTORY, "--project-root", root, "--verify-against", out,
              "--scope", "all")
    assert res.returncode == 0, res.stdout + res.stderr


def test_verify_detects_evaluation_result_mutation(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    rec = root / "results" / "evaluation" / "modelA" / "suite-a-cuda-to-suite-a-omp-s0.json"
    rec.write_text(rec.read_text().replace("PASS", "FAIL"))
    res = run(INVENTORY, "--project-root", root, "--verify-against", out,
              "--scope", "evaluation-results")
    assert res.returncode == 1
    assert "changed:" in res.stdout


def test_verify_detects_added_and_removed_results(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    extra = root / "results" / "evaluation" / "modelA" / "extra.json"
    extra.write_text("{}")
    res = run(INVENTORY, "--project-root", root, "--verify-against", out,
              "--scope", "evaluation-results")
    assert res.returncode == 1 and "added:" in res.stdout
    extra.unlink()
    (root / "results" / "evaluation" / "modelA" /
     "suite-a-cuda-to-suite-a-omp-s0.json").unlink()
    res = run(INVENTORY, "--project-root", root, "--verify-against", out,
              "--scope", "evaluation-results")
    assert res.returncode == 1 and "removed:" in res.stdout


def test_verify_detects_benchmark_source_mutation(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    (root / "srctree" / "k" / "a.cu").write_text("__global__ void k2() {}\n")
    res = run(INVENTORY, "--project-root", root, "--verify-against", out,
              "--scope", "benchmark-sources")
    assert res.returncode == 1
    assert "changed: srctree/k/a.cu" in res.stdout
    # the still-missing Makefile must not fail verification
    assert "Makefile" not in "".join(
        line for line in res.stdout.splitlines() if line.startswith("FAIL")
    )


def test_archive_create_check_and_translated_export(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    target = tmp_path / "archives" / "pre_replay_test"
    res = run(ARCHIVE, "--source", root / "results" / "evaluation",
              "--target", target, "--manifest", out)
    assert res.returncode == 0, res.stdout + res.stderr

    copied = target / "evaluation" / "modelA" / "suite-a-cuda-to-suite-a-omp-s0.json"
    assert copied.is_file()
    exported = (target / "translated_files_export" / "modelA" /
                "suite-a-cuda-to-suite-a-omp-s0" / "a.cpp")
    assert exported.read_text() == "int main() { return 0; }\n"

    manifest = json.loads(out.read_text())
    entry = manifest["archives"]["pre_replay_test"]
    assert entry["file_count"] == 2 and len(entry["aggregate_sha256"]) == 64

    res = run(ARCHIVE, "--target", target, "--manifest", out, "--check")
    assert res.returncode == 0, res.stdout + res.stderr

    # idempotent rerun against an identical existing target succeeds, no rewrite
    res = run(ARCHIVE, "--source", root / "results" / "evaluation",
              "--target", target, "--manifest", out)
    assert res.returncode == 0 and "already matches" in res.stdout


def test_archive_tamper_is_refused(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    target = tmp_path / "archives" / "pre_replay_test"
    assert run(ARCHIVE, "--source", root / "results" / "evaluation",
               "--target", target, "--manifest", out).returncode == 0

    victim = target / "evaluation" / "modelA" / "suite-a-cuda-to-suite-a-omp-s0.json"
    original = victim.read_text()
    victim.write_text(original.replace("PASS", "TAMPERED"))

    res = run(ARCHIVE, "--target", target, "--manifest", out, "--check")
    assert res.returncode == 1 and "hash mismatch" in res.stdout

    res = run(ARCHIVE, "--source", root / "results" / "evaluation",
              "--target", target, "--manifest", out)
    assert res.returncode == 2 and "REFUSED" in res.stdout
    # the refusal must not have modified the tampered archive
    assert victim.read_text() == original.replace("PASS", "TAMPERED")


def test_archive_refuses_target_inside_source(tmp_path):
    root = make_project(tmp_path)
    out = make_manifest(root)
    res = run(ARCHIVE, "--source", root / "results" / "evaluation",
              "--target", root / "results" / "evaluation" / "arch",
              "--manifest", out)
    assert res.returncode == 2 and "refusing target inside" in res.stdout
    assert not (root / "results" / "evaluation" / "arch").exists()
