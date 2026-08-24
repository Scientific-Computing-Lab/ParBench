"""Tests for scripts/spec_tools/freeze_final_contract.py (Task 7).

The freeze must validate schema, hashes, and completion status of every
input artifact and REJECT an unresolved pair or missing artifact rather
than merely hashing what it was given. Fixtures build a minimal synthetic
project root; contract evidence paths point at real tracked repository
files because the pair-contract loader resolves evidence against the
repository root by design.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.spec_tools.freeze_final_contract as ffc  # noqa: E402

# A tracked, always-present repo file used as contract evidence in fixtures.
EVIDENCE = "results/analysis/pair_witnesses.json"

PAIR = ("demo-alpha-cuda", "demo-alpha-omp")


def _contract_entry(**overrides):
    entry = {
        "source_spec": PAIR[0],
        "target_spec": PAIR[1],
        "shared_input": {"source_args": ["16"], "target_args": ["16"]},
        "translated_run": {"args": ["16"]},
        "target_oracle": {"strategies": [
            {"type": "stdout_pattern", "pattern": "OK"},
            {"type": "exit_code", "expected": 0},
        ]},
        "comparator": {"type": "declared-strategy", "evidence": EVIDENCE},
        "baseline_witness": {"status": "pass", "evidence": EVIDENCE},
        "comparability": "eligible",
    }
    entry.update(overrides)
    return entry


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A minimal synthetic project root holding every freeze input."""
    root = tmp_path / "proj"
    (root / "specs").mkdir(parents=True)
    (root / "harness").mkdir()
    (root / "specs" / "demo-alpha-cuda.json").write_text("{}")
    (root / "harness" / "constants.py").write_text("# fixture\n")

    sealed_sha = "ab" * 32
    artifacts = {
        "results/analysis/source_complete_host.json": {
            "schema": "source-complete-host-v1", "complete": True,
        },
        "results/analysis/pair_witnesses.json": {
            "schema": "pair-witnesses-v1", "complete": True,
            "specs": {},
            "pairs": [{
                "source_spec": PAIR[0], "target_spec": PAIR[1],
                "witness_status": "pass", "reasons": [],
            }],
        },
        "results/analysis/baseline_determinism.json": {
            "schema": "baseline-determinism-v1", "complete": True,
            "specs": {
                PAIR[0]: {"stable": True},
                PAIR[1]: {"stable": True},
            },
        },
        "results/analysis/thread_provisioning.json": {
            "schema": "thread-provisioning-v1", "complete": True,
            "specs": {},
        },
        "results/provenance/blind_control_prereg_sealed.json": {
            "schema": "blind-control-prereg-v1",
            "count": 1, "sealed_sha256": sealed_sha,
        },
        "results/provenance/blind_control_labels.json": {
            "sealed_sha256": sealed_sha,
            "labels": [{"control_id": "control_01",
                        "oracle_status": "FAIL"}],
        },
        "results/provenance/blind_control_unblinding.json": {
            "sealed_sha256": sealed_sha,
            "controls": [{"control_id": "control_01",
                          "outcome": "detected_by_oracle"}],
        },
        "results/analysis/blind_control_result.json": {
            "sealed_sha256": sealed_sha,
            "summary": {"seeded": 1, "missed": 0},
            "controls": [{"control_id": "control_01",
                          "outcome": "detected_by_oracle"}],
        },
        "results/analysis/oracle_contract_report.json": {
            "summary": {"total_specs": 2, "eligible_specs": 2},
        },
        "results/analysis/pair_contract_dispositions.json": {
            "dispositions": [],
        },
        "config/pair_contract_candidates_populated.json": {
            "contract_version": "parbench-final-v1",
            "pairs": [{"source_spec": PAIR[0], "target_spec": PAIR[1]}],
        },
    }
    for rel, doc in artifacts.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1) + "\n")

    registry = root / "config" / "final_pair_contracts.json"
    registry.write_text(json.dumps({
        "contract_version": "parbench-final-v1",
        "pairs": [_contract_entry()],
    }, indent=1) + "\n")

    # Toolchain probing is host-environment-dependent; pin it.
    monkeypatch.setattr(
        ffc, "toolchain_versions",
        lambda: {"nvcc": "nvcc fixture", "nvc++": "nvc++ fixture",
                 "gcc": "gcc fixture"})
    return root


def _run(root: Path) -> tuple[int, Path]:
    out = root / "config" / "final_contract_manifest.json"
    rc = ffc.main([
        "--project-root", str(root),
        "--pair-contracts", str(root / "config" / "final_pair_contracts.json"),
        "--output", str(out),
    ])
    return rc, out


def _edit(root: Path, rel: str, mutate) -> None:
    path = root / rel
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, indent=1) + "\n")


class TestHappyPath:
    def test_freeze_succeeds_and_writes_manifest(self, project):
        rc, out = _run(project)
        assert rc == 0
        manifest = json.loads(out.read_text())
        assert manifest["manifest_version"] == "parbench-final-contract-v1"
        assert manifest["eligible_task_list"]["count"] == 1
        assert manifest["pair_registry"]["pairs"] == 1

    def test_task_list_hash_matches_recomputation(self, project):
        rc, out = _run(project)
        assert rc == 0
        manifest = json.loads(out.read_text())
        tasks = manifest["eligible_task_list"]["tasks"]
        recomputed = hashlib.sha256(json.dumps(
            tasks, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        assert manifest["eligible_task_list"]["sha256"] == recomputed

    def test_rerun_is_byte_identical(self, project):
        rc, out = _run(project)
        assert rc == 0
        first = out.read_bytes()
        rc2, out2 = _run(project)
        assert rc2 == 0
        assert out2.read_bytes() == first

    def test_exclusions_come_from_harness_constants(self, project):
        from harness.constants import EXCLUDED_SPECS
        rc, out = _run(project)
        assert rc == 0
        manifest = json.loads(out.read_text())
        assert manifest["exclusions"]["excluded_specs"] == sorted(
            EXCLUDED_SPECS)


class TestRejection:
    def test_missing_artifact_rejected(self, project):
        (project / "results" / "analysis"
         / "thread_provisioning.json").unlink()
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_incomplete_witness_campaign_rejected(self, project):
        _edit(project, "results/analysis/pair_witnesses.json",
              lambda d: d.update(complete=False))
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_unresolved_pair_rejected(self, project):
        registry = project / "config" / "final_pair_contracts.json"
        doc = json.loads(registry.read_text())
        doc["pairs"][0]["comparability"] = "unresolved"
        del doc["pairs"][0]["comparator"]
        doc["pairs"][0]["baseline_witness"] = {"status": "unresolved"}
        registry.write_text(json.dumps(doc, indent=1) + "\n")
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_unaccounted_candidate_pair_rejected(self, project):
        _edit(project, "config/pair_contract_candidates_populated.json",
              lambda d: d["pairs"].append(
                  {"source_spec": "demo-beta-cuda",
                   "target_spec": "demo-beta-omp"}))
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_failing_witness_row_rejected(self, project):
        def mutate(d):
            d["pairs"][0]["witness_status"] = "fail"
            d["pairs"][0]["reasons"] = ["target_baseline_fail"]
        _edit(project, "results/analysis/pair_witnesses.json", mutate)
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_nondeterministic_spec_rejected(self, project):
        _edit(project, "results/analysis/baseline_determinism.json",
              lambda d: d["specs"][PAIR[1]].update(stable=False))
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_blind_control_hash_mismatch_rejected(self, project):
        _edit(project, "results/analysis/blind_control_result.json",
              lambda d: d.update(sealed_sha256="cd" * 32))
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_blind_control_missing_outcome_rejected(self, project):
        _edit(project, "results/analysis/blind_control_result.json",
              lambda d: d["controls"][0].pop("outcome"))
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_incomplete_host_rejected(self, project):
        _edit(project, "results/analysis/source_complete_host.json",
              lambda d: d.update(complete=False))
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()

    def test_missing_toolchain_rejected(self, project, monkeypatch):
        monkeypatch.setattr(
            ffc, "toolchain_versions",
            lambda: {"nvcc": None, "nvc++": "x", "gcc": "x"})
        rc, out = _run(project)
        assert rc == 1
        assert not out.exists()


class TestProductionManifest:
    """Pins on the real frozen manifest when it exists in this checkout."""

    MANIFEST = PROJECT_ROOT / "config" / "final_contract_manifest.json"

    @pytest.mark.skipif(not MANIFEST.exists(),
                        reason="production manifest not frozen yet")
    def test_production_manifest_is_resolved_and_hash_consistent(self):
        manifest = json.loads(self.MANIFEST.read_text())
        assert manifest["manifest_version"] == "parbench-final-contract-v1"
        tasks = manifest["eligible_task_list"]["tasks"]
        assert manifest["eligible_task_list"]["count"] == len(tasks) > 0
        recomputed = hashlib.sha256(json.dumps(
            tasks, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        assert manifest["eligible_task_list"]["sha256"] == recomputed
        # No task may involve a correctness-ineligible spec.
        from harness.constants import CORRECTNESS_INELIGIBLE_SPECS
        for task in tasks:
            assert task["source_spec"] not in CORRECTNESS_INELIGIBLE_SPECS
            assert task["target_spec"] not in CORRECTNESS_INELIGIBLE_SPECS
