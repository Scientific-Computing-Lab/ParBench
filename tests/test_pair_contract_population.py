"""Task 4: build_pair_contracts.py --populate-from-specs.

The populated candidate registry carries spec-side data for every enumerated
pair while leaving the Linux witness explicitly unresolved, so the evaluation
runner's require_resolved() keeps rejecting every pair until Task 7 attaches
real witness evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.pair_contracts import (  # noqa: E402
    PairContractError,
    load_pair_contracts,
)
from scripts.spec_tools.build_pair_contracts import (  # noqa: E402
    populate_from_specs,
)

POPULATED = PROJECT_ROOT / "config" / "pair_contract_candidates_populated.json"
CANDIDATES = PROJECT_ROOT / "config" / "pair_contract_candidates.json"


def _mini_spec(uid: str, api: str, args: list[str], strategies: list[dict]) -> dict:
    suite, kernel, _ = uid.split("-")
    return {
        "identity": {
            "unique_id": uid, "kernel_name": kernel,
            "parallel_api": api, "source_suite": suite,
        },
        "run": {
            "input_configurations": {
                "correctness": {"arguments": args},
            },
        },
        "verification": {"method": "self_checking", "strategies": strategies},
    }


@pytest.fixture()
def mini_corpus(tmp_path):
    spec_root = tmp_path / "specs"
    spec_root.mkdir()
    src = _mini_spec(
        "suitea-kern-cuda", "cuda", ["8", "x.txt"],
        [{"type": "stdout_pattern", "pattern": "OK"},
         {"type": "exit_code", "expected": 0}],
    )
    tgt = _mini_spec(
        "suitea-kern-omp", "omp", ["4"],
        [{"type": "numeric_comparison", "expected": 1.5,
          "extract_regex": r"v=([\d.]+)", "tolerance": 0.1,
          "tolerance_type": "absolute"},
         {"type": "stdout_exclude_pattern", "pattern": "(?i)error",
          "description": "negative"}],
    )
    for spec in (src, tgt):
        path = spec_root / f"{spec['identity']['unique_id']}.json"
        path.write_text(json.dumps(spec))
    candidates = {
        "contract_version": "parbench-final-v1",
        "pairs": [
            {"source_spec": "suitea-kern-cuda",
             "target_spec": "suitea-kern-omp",
             "baseline_witness": {"status": "unresolved"},
             "comparability": "unresolved"},
        ],
    }
    return spec_root, candidates


class TestPopulateFromSpecs:
    def test_populates_spec_side_data(self, mini_corpus):
        spec_root, candidates = mini_corpus
        populated = populate_from_specs(candidates, spec_root)
        [pair] = populated["pairs"]
        assert pair["shared_input"] == {
            "source_args": ["8", "x.txt"], "target_args": ["4"],
        }
        # full-program pair (no .cl-only targets): the model keeps SOURCE
        # argument parsing, so translated_run carries source args
        assert pair["translated_run"]["args"] == ["8", "x.txt"]
        types = [s["type"] for s in pair["target_oracle"]["strategies"]]
        assert types == ["numeric_comparison"]
        assert [n["type"] for n in pair["negative_checks"]] == [
            "stdout_exclude_pattern"
        ]

    def test_strips_fields_the_pair_schema_rejects(self, mini_corpus):
        spec_root, candidates = mini_corpus
        populated = populate_from_specs(candidates, spec_root)
        [pair] = populated["pairs"]
        [numeric] = pair["target_oracle"]["strategies"]
        assert "tolerance_type" not in numeric

    def test_witness_left_unresolved(self, mini_corpus):
        spec_root, candidates = mini_corpus
        populated = populate_from_specs(candidates, spec_root)
        [pair] = populated["pairs"]
        assert pair["baseline_witness"] == {"status": "unresolved"}
        assert pair["comparability"] == "unresolved"

    def test_kernel_only_pair_gets_target_args(self, mini_corpus):
        """OD-3 follow-up (owner ruling 2026-08-21): a kernel-only pair (all
        translation_targets .cl) runs the TARGET-native host, so
        translated_run carries the TARGET's correctness args."""
        spec_root, candidates = mini_corpus
        tgt = _mini_spec(
            "suitea-kern-opencl", "opencl", ["-p", "0"],
            [{"type": "exit_code", "expected": 0}],
        )
        tgt["files"] = {"prompt_payload": ["host.cpp", "k.cl"],
                        "translation_targets": ["k.cl"]}
        (spec_root / "suitea-kern-opencl.json").write_text(json.dumps(tgt))
        candidates["pairs"].append({
            "source_spec": "suitea-kern-cuda",
            "target_spec": "suitea-kern-opencl",
            "baseline_witness": {"status": "unresolved"},
            "comparability": "unresolved",
        })
        populated = populate_from_specs(candidates, spec_root)
        by_key = {(p["source_spec"], p["target_spec"]): p
                  for p in populated["pairs"]}
        kernel_only = by_key[("suitea-kern-cuda", "suitea-kern-opencl")]
        assert kernel_only["translated_run"]["args"] == ["-p", "0"]
        # shared_input still records both sides unchanged
        assert kernel_only["shared_input"]["source_args"] == ["8", "x.txt"]
        # the full-program sibling keeps source args
        full = by_key[("suitea-kern-cuda", "suitea-kern-omp")]
        assert full["translated_run"]["args"] == ["8", "x.txt"]

    def test_write_registry_refuses_frozen_paths(self):
        """The frozen v1 artifacts are hash-bound by
        config/final_contract_manifest.json; write_registry must refuse to
        rewrite them (2026-08-21 Codex finding)."""
        from scripts.spec_tools.build_pair_contracts import write_registry
        frozen = PROJECT_ROOT / "config" / "final_pair_contracts.json"
        with pytest.raises(SystemExit, match="hash-bound"):
            write_registry([], frozen)

    def test_regeneration_stamps_current_version(self, mini_corpus):
        """Every regeneration stamps CONTRACT_VERSION_CURRENT, never the
        frozen v1 string, so new registries are distinguishable at run
        time (2026-08-21 Codex finding)."""
        from scripts.evaluation.pair_contracts import CONTRACT_VERSION_CURRENT
        spec_root, candidates = mini_corpus
        populated = populate_from_specs(candidates, spec_root)
        assert populated["contract_version"] == CONTRACT_VERSION_CURRENT


class TestPopulatedRegistryOnDisk:
    def test_exists_and_schema_valid(self):
        jsonschema = pytest.importorskip("jsonschema")
        assert POPULATED.is_file(), "run the Task 4 populate command"
        doc = json.loads(POPULATED.read_text())
        schema = json.loads(
            (PROJECT_ROOT / "schema" / "pair_contract_schema.json").read_text()
        )
        jsonschema.validate(doc, schema)

    def test_covers_every_candidate_pair(self):
        doc = json.loads(POPULATED.read_text())
        cand = json.loads(CANDIDATES.read_text())
        keyed = {(p["source_spec"], p["target_spec"]) for p in doc["pairs"]}
        assert keyed == {
            (p["source_spec"], p["target_spec"]) for p in cand["pairs"]
        }

    def test_runner_rejects_every_pair_until_task7(self):
        registry = load_pair_contracts(POPULATED)
        doc = json.loads(POPULATED.read_text())
        for pair in doc["pairs"]:
            assert pair["baseline_witness"]["status"] == "unresolved"
            assert pair["comparability"] == "unresolved"
        sample = doc["pairs"][0]
        with pytest.raises(PairContractError):
            registry.resolve(sample["source_spec"], sample["target_spec"])
