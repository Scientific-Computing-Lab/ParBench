"""Tests for scripts/evaluation/replay_stored_translations.py (Task 5).

Replay rebuilds and re-verifies STORED ``translated_files`` from immutable
input records — it must never make a model or provider call. The tests here
prove that by replacing the provider entry point (``llm_evaluate.call_llm``)
with a function that raises immediately, then showing a fixture replay still
succeeds end-to-end (write stored code -> build with cc -> run -> verify
under the pair contract).

The fixture is a fully synthetic project root under ``tmp_path``: no real
spec, benchmark tree, or ``results/evaluation`` byte is ever touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import llm_evaluate  # noqa: E402
from scripts.evaluation import replay_stored_translations as replay  # noqa: E402

MODEL = "test-model"
SRC_ID = "fix-alpha-cuda"
TGT_ID = "fix-alpha-omp"
RECORD_NAME = f"{SRC_ID}-to-{TGT_ID}.json"

REFERENCE_C = '#include <stdio.h>\nint main(void){printf("REF ONLY\\n");return 0;}\n'
TRANSLATED_C = (
    "#include <stdio.h>\n"
    "int main(int argc, char **argv){\n"
    '    printf("ALPHA OK n=%s\\n", argc > 1 ? argv[1] : "0");\n'
    "    return 0;\n"
    "}\n"
)


def _spec(api: str, sub: str) -> dict:
    return {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": "alpha",
            "parallel_api": api,
            "unique_id": f"fix-alpha-{api}",
            "source_suite": "fix",
        },
        "provenance": {"repo_root": "bench", "source_path": f"alpha/{sub}"},
        "files": {
            "prompt_payload": ["alpha.c"],
            "support_files": [],
            "verification_only": [],
            "translation_targets": ["alpha.c"],
        },
        "build": {
            "working_directory": f"alpha/{sub}",
            "commands": {"build": "cc -O0 -o alpha.out alpha.c", "clean": "rm -f alpha.out"},
            "outputs": {"executable": "./alpha.out"},
        },
        "run": {
            "executable": "./alpha.out",
            "timeout_seconds": 30,
            "input_configurations": {"correctness": {"arguments": ["7"]}},
        },
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "ALPHA OK"},
                {"type": "exit_code", "expected": 0},
            ]
        },
    }


def _contract_entry(oracle_strategies: list[dict] | None = None) -> dict:
    return {
        "source_spec": SRC_ID,
        "target_spec": TGT_ID,
        "shared_input": {"source_args": ["7"], "target_args": ["7"]},
        "translated_run": {"args": ["7"], "environment": {}},
        "target_oracle": {
            "strategies": oracle_strategies
            or [
                {"type": "stdout_pattern", "pattern": "ALPHA OK"},
                {"type": "exit_code", "expected": 0},
            ]
        },
        "negative_checks": [],
        "comparator": {
            "type": "declared-strategy",
            "evidence": "tests/test_replay_stored_translations.py",
        },
        "baseline_witness": {
            "status": "pass",
            "evidence": "tests/test_replay_stored_translations.py",
        },
        "comparability": "eligible",
    }


def make_fixture(
    tmp_path: Path,
    *,
    translated_files: dict[str, str] | None = None,
    parent_status: str = "VERIFY_FAIL",
    oracle_strategies: list[dict] | None = None,
) -> dict[str, Path]:
    """Build a synthetic project root with one stored translation record."""
    root = tmp_path / "proj"
    (root / "specs").mkdir(parents=True)
    for api, sub in (("cuda", "cuda"), ("omp", "omp")):
        (root / "specs" / f"fix-alpha-{api}.json").write_text(
            json.dumps(_spec(api, sub), indent=2)
        )
        d = root / "bench" / "alpha" / sub
        d.mkdir(parents=True)
        (d / "alpha.c").write_text(REFERENCE_C)

    record = {
        "source_spec": SRC_ID,
        "target_spec": TGT_ID,
        "kernel": "alpha",
        "model": MODEL,
        "augment_level": 0,
        "sample_id": 0,
        "overall_status": parent_status,
        "translated_files": (
            translated_files if translated_files is not None else {"alpha.c": TRANSLATED_C}
        ),
    }
    input_root = root / "results" / "evaluation"
    (input_root / MODEL).mkdir(parents=True)
    (input_root / MODEL / RECORD_NAME).write_text(json.dumps(record, indent=2))

    registry = {
        "contract_version": "parbench-final-v1",
        "pairs": [_contract_entry(oracle_strategies)],
    }
    contracts = root / "pair_contracts.json"
    contracts.write_text(json.dumps(registry, indent=2))

    return {
        "root": root,
        "input_root": input_root,
        "output_root": root / "results" / "replay_test" / "ns1",
        "contracts": contracts,
    }


def run_replay(fx: dict[str, Path], *extra: str) -> int:
    return replay.main(
        [
            "--input-root", str(fx["input_root"]),
            "--output-root", str(fx["output_root"]),
            "--pair-contracts", str(fx["contracts"]),
            "--project-root", str(fx["root"]),
            *extra,
        ]
    )


@pytest.fixture(autouse=True)
def forbid_provider_calls(monkeypatch):
    """Every test runs with the provider entry point replaced by a raiser."""

    def _boom(*args, **kwargs):  # pragma: no cover - reaching this is the bug
        raise AssertionError("provider entry point call_llm was invoked during replay")

    monkeypatch.setattr(llm_evaluate, "call_llm", _boom)
    yield


# ---------------------------------------------------------------------------
# Core replay behavior
# ---------------------------------------------------------------------------


def test_fixture_replay_succeeds_without_provider(tmp_path):
    """A stored translation rebuilds, runs, and verifies with no model call."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0

    out_file = fx["output_root"] / MODEL / RECORD_NAME
    assert out_file.is_file()
    out = json.loads(out_file.read_text())
    assert out["overall_status"] == "PASS"
    # Stage statuses use the harness Status enum values (lowercase), matching
    # the live result-JSON schema; overall_status stays uppercase.
    assert out["build_status"] == "pass"
    assert out["run_status"] == "pass"
    assert out["verify_status"] == "pass"
    # Provenance: input record path + SHA-256.
    input_bytes = (fx["input_root"] / MODEL / RECORD_NAME).read_bytes()
    assert out["input_record"]["sha256"] == hashlib.sha256(input_bytes).hexdigest()
    assert out["input_record"]["path"].endswith(f"results/evaluation/{MODEL}/{RECORD_NAME}")
    # Contract version + registry hash.
    assert out["contract"]["version"] == "parbench-final-v1"
    reg_hash = hashlib.sha256(fx["contracts"].read_bytes()).hexdigest()
    assert out["contract"]["registry_sha256"] == reg_hash
    # Spec hashes recorded for both sides.
    for side in ("source", "target"):
        assert len(out["spec_hashes"][side]["sha256"]) == 64
    # Toolchain / platform provenance present.
    assert out["platform"]["platform"]
    assert "cc" in out["toolchain"]
    # Parent verdict is carried for the old-to-new transition.
    assert out["parent"]["overall_status"] == "VERIFY_FAIL"


def test_reference_files_restored_after_replay(tmp_path):
    """Replay writes stored code over the tree, then restores the original."""
    fx = make_fixture(tmp_path)
    target_file = fx["root"] / "bench" / "alpha" / "omp" / "alpha.c"
    assert run_replay(fx) == 0
    assert target_file.read_text() == REFERENCE_C


def test_replay_actually_builds_stored_code(tmp_path):
    """A stored translation that fails the oracle must yield VERIFY_FAIL.

    Guards against the false-positive failure mode where replay builds the
    reference implementation instead of the stored translation.
    """
    bad = TRANSLATED_C.replace("ALPHA OK", "ALPHA WRONG")
    fx = make_fixture(tmp_path, translated_files={"alpha.c": bad})
    assert run_replay(fx) == 0
    out = json.loads((fx["output_root"] / MODEL / RECORD_NAME).read_text())
    assert out["overall_status"] == "VERIFY_FAIL"


def test_broken_stored_code_is_build_fail(tmp_path):
    fx = make_fixture(tmp_path, translated_files={"alpha.c": "int main( {"})
    assert run_replay(fx) == 0
    out = json.loads((fx["output_root"] / MODEL / RECORD_NAME).read_text())
    assert out["overall_status"] == "BUILD_FAIL"


def test_verify_error_is_error_not_verify_fail(tmp_path):
    """A harness-side verification ERROR must surface as ERROR (Task 2 note).

    file_diff with an unreadable reference is a declared check the harness
    cannot execute — infrastructure error, not evidence of a wrong
    translation.
    """
    fx = make_fixture(
        tmp_path,
        oracle_strategies=[
            {"type": "file_diff", "path": "out.txt", "reference_file": "/nonexistent/ref.bin"}
        ],
    )
    assert run_replay(fx) == 0
    out = json.loads((fx["output_root"] / MODEL / RECORD_NAME).read_text())
    assert out["verify_status"] == "error"
    assert out["overall_status"] == "ERROR"
    assert out["overall_status"] != "VERIFY_FAIL"


def test_record_without_translated_files_is_not_replayable(tmp_path):
    """No stored code (e.g. EXTRACTION_FAIL parent) -> NOT_REPLAYABLE, no build."""
    fx = make_fixture(tmp_path, translated_files={}, parent_status="EXTRACTION_FAIL")
    assert run_replay(fx) == 0
    out = json.loads((fx["output_root"] / MODEL / RECORD_NAME).read_text())
    assert out["overall_status"] == "NOT_REPLAYABLE"
    assert out["build_status"] is None


# ---------------------------------------------------------------------------
# Safety: unsafe output roots and provenance mismatches
# ---------------------------------------------------------------------------


def test_output_under_results_evaluation_aborts_before_opening(tmp_path):
    """An output root under results/evaluation aborts before any file opens.

    The input root is deliberately nonexistent: if the unsafe-output check
    ran after input discovery, this call would fail with a different error.
    """
    fx = make_fixture(tmp_path)
    bad_output = fx["root"] / "results" / "evaluation" / "replayed"
    rc = replay.main(
        [
            "--input-root", str(fx["root"] / "does-not-exist"),
            "--output-root", str(bad_output),
            "--pair-contracts", str(fx["root"] / "also-missing.json"),
            "--project-root", str(fx["root"]),
        ]
    )
    assert rc == 2
    assert not bad_output.exists()


def test_output_root_equal_to_results_evaluation_aborts(tmp_path):
    fx = make_fixture(tmp_path)
    rc = replay.main(
        [
            "--input-root", str(fx["input_root"]),
            "--output-root", str(fx["input_root"]),
            "--pair-contracts", str(fx["contracts"]),
            "--project-root", str(fx["root"]),
        ]
    )
    assert rc == 2


def test_resume_skips_existing_matching_output(tmp_path):
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    out_file = fx["output_root"] / MODEL / RECORD_NAME
    before = out_file.read_bytes()
    assert run_replay(fx, "--resume") == 0
    assert out_file.read_bytes() == before


def test_existing_output_without_resume_refuses_overwrite(tmp_path):
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    assert run_replay(fx) == 2


def test_mismatched_provenance_aborts_rather_than_overwrite(tmp_path):
    """A changed input record under --resume is a provenance mismatch: abort."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    out_file = fx["output_root"] / MODEL / RECORD_NAME
    before = out_file.read_bytes()

    # Tamper with the (synthetic) input record — its SHA-256 changes.
    rec_path = fx["input_root"] / MODEL / RECORD_NAME
    rec = json.loads(rec_path.read_text())
    rec["overall_status"] = "PASS"
    rec_path.write_text(json.dumps(rec, indent=2))

    assert run_replay(fx, "--resume") == 2
    assert out_file.read_bytes() == before


# ---------------------------------------------------------------------------
# Companion artifacts
# ---------------------------------------------------------------------------


def test_transitions_companion_artifact(tmp_path):
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    transitions_path = fx["output_root"] / "replay_transitions.json"
    assert transitions_path.is_file()
    transitions = json.loads(transitions_path.read_text())
    assert transitions["counts"]["VERIFY_FAIL->PASS"] == 1
    (entry,) = transitions["records"]
    assert entry["old_status"] == "VERIFY_FAIL"
    assert entry["new_status"] == "PASS"

    manifest_path = fx["output_root"] / "replay_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    rel = f"{MODEL}/{RECORD_NAME}"
    out_bytes = (fx["output_root"] / MODEL / RECORD_NAME).read_bytes()
    assert manifest["files"][rel] == hashlib.sha256(out_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Dimensional transition breakdowns (Task 8)
# ---------------------------------------------------------------------------


def test_transitions_dimensional_breakdowns(tmp_path):
    """Transitions carry per-dimension matrices: model, suite, direction,
    augmentation level, and the old->new failure-stage matrix."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    transitions = json.loads((fx["output_root"] / replay.TRANSITIONS_NAME).read_text())
    assert transitions["by_model"][MODEL]["VERIFY_FAIL->PASS"] == 1
    assert transitions["by_suite"]["fix"]["VERIFY_FAIL->PASS"] == 1
    assert transitions["by_direction"]["cuda-to-omp"]["VERIFY_FAIL->PASS"] == 1
    assert transitions["by_augment_level"]["L0"]["VERIFY_FAIL->PASS"] == 1
    assert transitions["by_failure_stage"]["VERIFY_FAIL"]["PASS"] == 1
    (entry,) = transitions["records"]
    assert entry["augment_level"] == 0
    # Reconciliation invariant: totals equal the record count in every view.
    assert transitions["record_count"] == 1
    assert sum(transitions["counts"].values()) == 1
    for dim in ("by_model", "by_suite", "by_direction", "by_augment_level"):
        assert sum(n for cell in transitions[dim].values() for n in cell.values()) == 1


# ---------------------------------------------------------------------------
# --check-complete (Task 8)
# ---------------------------------------------------------------------------


def run_check_complete(fx: dict[str, Path]) -> int:
    return replay.main(
        [
            "--input-root", str(fx["input_root"]),
            "--output-root", str(fx["output_root"]),
            "--pair-contracts", str(fx["contracts"]),
            "--project-root", str(fx["root"]),
            "--check-complete",
        ]
    )


def test_check_complete_passes_after_full_replay(tmp_path, capsys):
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    assert run_check_complete(fx) == 0
    out = capsys.readouterr().out
    assert "CHECK-COMPLETE OK" in out


def test_check_complete_fails_on_missing_output(tmp_path):
    """An input record without a replay output is an unexplained omission."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    (fx["output_root"] / MODEL / RECORD_NAME).unlink()
    assert run_check_complete(fx) == 1


def test_check_complete_fails_on_orphan_output(tmp_path):
    """An output record with no corresponding input record fails the check."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    (fx["output_root"] / MODEL / "orphan-to-nowhere.json").write_text(
        json.dumps({"overall_status": "PASS"})
    )
    assert run_check_complete(fx) == 1


def test_check_complete_fails_on_registry_provenance_mismatch(tmp_path):
    """Outputs replayed under a different contract registry are stale."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    # Re-serialize with different whitespace: schema-identical content,
    # different bytes, therefore a different registry SHA-256.
    registry = json.loads(fx["contracts"].read_text())
    fx["contracts"].write_text(json.dumps(registry, indent=4))
    assert run_check_complete(fx) == 1


def test_check_complete_writes_nothing(tmp_path):
    """--check-complete is read-only: no file in the namespace changes."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    before = {
        p: p.read_bytes() for p in fx["output_root"].rglob("*") if p.is_file()
    }
    assert run_check_complete(fx) == 0
    after = {
        p: p.read_bytes() for p in fx["output_root"].rglob("*") if p.is_file()
    }
    assert before == after


# ---------------------------------------------------------------------------
# No-provider mode
# ---------------------------------------------------------------------------


def test_no_provider_mode_is_installed_by_main(tmp_path, monkeypatch):
    """replay.main() itself disables call_llm even without the test stub."""
    calls: list[str] = []

    def _spy(*args, **kwargs):
        calls.append("called")
        return {}

    monkeypatch.setattr(llm_evaluate, "call_llm", _spy)
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    assert calls == []
    with pytest.raises(replay.ProviderCallForbidden):
        llm_evaluate.call_llm("m", "s", [])


def test_module_defines_no_provider_guard():
    assert issubclass(replay.ProviderCallForbidden, RuntimeError)


# ---------------------------------------------------------------------------
# 2026-08-12 hardening (Task A findings 4 + 7, replay-contamination root cause)
# ---------------------------------------------------------------------------


def test_partial_file_set_is_not_replayable(tmp_path):
    """A stored set that mismatches translation_targets must never build."""
    fx = make_fixture(tmp_path, translated_files={"other.c": TRANSLATED_C})
    assert run_replay(fx) == 0
    out = json.loads((fx["output_root"] / MODEL / RECORD_NAME).read_text())
    assert out["overall_status"] == "NOT_REPLAYABLE"
    assert "do not match translation_targets" in out["error_message"]
    # The reference tree was never touched.
    ref = fx["root"] / "bench" / "alpha" / "omp" / "alpha.c"
    assert ref.read_text() == REFERENCE_C


def test_resume_aborts_when_spec_changed(tmp_path):
    """A resumed record replayed under a since-changed spec is not current."""
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    spec_path = fx["root"] / "specs" / f"{TGT_ID}.json"
    spec = json.loads(spec_path.read_text())
    spec["run"]["timeout_seconds"] = 31
    spec_path.write_text(json.dumps(spec, indent=2))
    assert run_replay(fx, "--resume") == 2


def _write_manifest(fx: dict[str, Path], name: str = "final_contract_manifest.json",
                    *, note: str | None = None,
                    harness_files: dict[str, str] | None = None,
                    spec_sha_override: str | None = None) -> Path:
    """A realistic frozen-manifest stand-in for the fixture project."""
    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    from harness.constants import (
        CORRECTNESS_INELIGIBLE_SPECS,
        EXCLUDED_SPECS,
        PERFORMANCE_ONLY_SPECS,
    )
    specs_dir = fx["root"] / "specs"
    manifest = {
        "pair_registry": {
            "path": str(fx["contracts"]),
            "sha256": _sha(fx["contracts"]),
        },
        "specs": {"files": {
            p.name: (spec_sha_override or _sha(p))
            for p in sorted(specs_dir.glob("*.json"))
        }},
        "harness": {"files": harness_files or {}},
        "exclusions": {
            "excluded_specs": sorted(EXCLUDED_SPECS),
            "performance_only_specs": sorted(PERFORMANCE_ONLY_SPECS),
            "correctness_ineligible_specs":
                sorted(CORRECTNESS_INELIGIBLE_SPECS),
        },
    }
    if note:
        manifest["note"] = note
    path = fx["root"] / name
    path.write_text(json.dumps(manifest, indent=1))
    return path


def test_contract_manifest_bound_and_checked(tmp_path):
    """--contract-manifest binds the frozen manifest into each record and is
    validated by --check-complete."""
    fx = make_fixture(tmp_path)
    manifest = _write_manifest(fx)
    assert run_replay(fx, "--contract-manifest", str(manifest)) == 0
    out = json.loads((fx["output_root"] / MODEL / RECORD_NAME).read_text())
    assert out["contract_manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    assert run_replay(fx, "--check-complete",
                      "--contract-manifest", str(manifest)) == 0
    # A different (but internally consistent) manifest: binding mismatch.
    other = _write_manifest(fx, "other_manifest.json", note="different bytes")
    assert run_replay(fx, "--check-complete",
                      "--contract-manifest", str(other)) == 1


def test_replay_aborts_on_manifest_spec_drift(tmp_path):
    """A frozen manifest whose spec hashes do not match the checkout is
    rejected before any replay work."""
    fx = make_fixture(tmp_path)
    manifest = _write_manifest(fx, spec_sha_override="0" * 64)
    assert run_replay(fx, "--contract-manifest", str(manifest)) == 2


def test_harness_drift_blocks_replay_but_only_warns_check_complete(tmp_path, capsys):
    """Post-seal harness drift: new replays under the frozen contract must
    refuse; read-only completeness checks warn (sealed bytes unaffected)."""
    fx = make_fixture(tmp_path)
    harness_dir = fx["root"] / "harness"
    harness_dir.mkdir()
    stub = harness_dir / "stub.py"
    stub.write_text("VERSION = 1\n")
    manifest = _write_manifest(fx, harness_files={
        "stub.py": hashlib.sha256(stub.read_bytes()).hexdigest()})
    assert run_replay(fx, "--contract-manifest", str(manifest)) == 0
    stub.write_text("VERSION = 2\n")
    assert run_replay(fx, "--resume", "--contract-manifest", str(manifest)) == 2
    assert run_replay(fx, "--check-complete",
                      "--contract-manifest", str(manifest)) == 0
    assert "harness drift" in capsys.readouterr().err


def test_stale_spec_side_fails_closed_on_missing_hashes():
    """A record without recorded spec hashes is never 'current'."""
    fake = {"parent": {"source_spec": SRC_ID, "target_spec": TGT_ID},
            "spec_hashes": {}}
    assert replay._stale_spec_side(fake, Path("."), {}) == "source"


def test_check_complete_detects_changed_spec(tmp_path):
    fx = make_fixture(tmp_path)
    assert run_replay(fx) == 0
    spec_path = fx["root"] / "specs" / f"{SRC_ID}.json"
    spec = json.loads(spec_path.read_text())
    spec["run"]["timeout_seconds"] = 31
    spec_path.write_text(json.dumps(spec, indent=2))
    assert run_replay(fx, "--check-complete") == 1


def test_surviving_backup_is_never_overwritten(tmp_path):
    """Root cause of the 2026-08-12 benchmark-tree contamination: a backup
    left by a killed run is the only pristine copy; backing up over it
    launders the contamination into the 'pristine' slot."""
    f = tmp_path / "alpha.c"
    f.write_text("STALE TRANSLATION FROM KILLED RUN")
    bak = tmp_path / "alpha.c.parbench_bak"
    bak.write_text("PRISTINE REFERENCE")
    info = llm_evaluate.backup_files([f])
    assert bak.read_text() == "PRISTINE REFERENCE"
    llm_evaluate.restore_files(info)
    assert f.read_text() == "PRISTINE REFERENCE"
    assert not bak.exists()


def test_output_root_symlink_into_results_evaluation_refused(tmp_path):
    """T5 gate (2026-08-12): a symlink alias must not bypass the safety gate."""
    fx = make_fixture(tmp_path)
    alias = fx["root"] / "alias"
    alias.symlink_to(fx["root"] / "results" / "evaluation")
    assert run_replay({**fx, "output_root": alias / "ns"},
                      ) == 2 or replay.main([
        "--input-root", str(fx["input_root"]),
        "--output-root", str(alias / "ns"),
        "--pair-contracts", str(fx["contracts"]),
        "--project-root", str(fx["root"]),
    ]) == 2


def test_replay_refuses_sealed_output_root(tmp_path):
    """T5 gate (2026-08-12): a sealed namespace is never a replay target."""
    fx = make_fixture(tmp_path)
    fx["output_root"].mkdir(parents=True)
    (fx["output_root"] / ".parbench-seal.json").write_text("{}")
    assert run_replay(fx) == 2


def test_manifest_without_declarations_fails_closed(tmp_path):
    """fixcheck2 (2026-08-12): empty spec maps or missing exclusion lists
    must be problems, not silent skips."""
    import copy
    fx = make_fixture(tmp_path)
    manifest = _write_manifest(fx)
    base = json.loads(manifest.read_text())
    for mutate in (
        lambda m: m["specs"].__setitem__("files", {}),
        lambda m: m.__setitem__("exclusions", {}),
    ):
        m = copy.deepcopy(base)
        mutate(m)
        manifest.write_text(json.dumps(m))
        assert run_replay(fx, "--contract-manifest", str(manifest)) == 2


def test_analyze_eval_refuses_sealed_output(tmp_path):
    """fixcheck2 (2026-08-12): defaulting the summary outputs into a sealed
    results dir must refuse, independent of hooks."""
    import subprocess as sp
    ns = tmp_path / "sealedroot" / "model-a"
    ns.mkdir(parents=True)
    (ns / "a-to-b.json").write_text(json.dumps(
        {"overall_status": "PASS", "source_spec": "rodinia-bfs-cuda",
         "target_spec": "rodinia-bfs-omp"}))
    (tmp_path / "sealedroot" / ".parbench-seal.json").write_text("{}")
    r = sp.run([sys.executable,
                str(PROJECT_ROOT / "scripts/evaluation/analyze_eval.py"),
                "--results-dir", str(tmp_path / "sealedroot")],
               capture_output=True, text=True)
    assert r.returncode != 0
    assert "REFUSE" in (r.stdout + r.stderr)
    assert not (tmp_path / "sealedroot" / "eval_summary.json").exists()
