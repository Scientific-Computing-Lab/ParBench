"""Task 4 (E-04): mixbench performance_only scope — marker, filtering, controls.

The deterministic mixbench decision (plan Task 4):
1. Attempt harness-controlled extraction of a computed value without editing
   mixbench source. (Outcome recorded in the mixbench disposition note.)
2. Retain mixbench in correctness statistics only if the pristine baseline
   passes an independent comparator AND a seeded do-nothing / wrong kernel
   FAILS the declared oracle.
3. Otherwise mark performance_only and remove its pairs from correctness
   denominators, while keeping it in the artifact and benchmark inventory.

These tests pin the marker (spec JSON + harness.constants), the filtering in
every correctness-denominator consumer, and the negative-control evidence
that makes the verdict run-proven rather than asserted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.constants import (  # noqa: E402
    CORRECTNESS_INELIGIBLE_SPECS,
    EXCLUDED_SPECS,
    PERFORMANCE_ONLY_SPECS,
)

SPEC_ROOT = PROJECT_ROOT / "specs"
MIXBENCH_IDS = frozenset({
    "mixbench-mixbench-cuda",
    "mixbench-mixbench-omp",
    "mixbench-mixbench-opencl",
})
NEG_CONTROL_DIR = PROJECT_ROOT / "results" / "analysis" / "negative_controls"
DISPOSITION_FILE = (
    PROJECT_ROOT / "results" / "analysis" / "source_baseline_dispositions"
    / "mixbench.json"
)


def _load_spec(uid: str) -> dict:
    return json.loads((SPEC_ROOT / f"{uid}.json").read_text())


# --------------------------------------------------------------------------- #
# The canonical constant and the spec-side marker stay in lockstep            #
# --------------------------------------------------------------------------- #

class TestPerformanceOnlyConstant:
    def test_constant_is_exactly_the_three_mixbench_specs(self):
        assert PERFORMANCE_ONLY_SPECS == MIXBENCH_IDS

    def test_disjoint_from_known_fail(self):
        assert not (PERFORMANCE_ONLY_SPECS & EXCLUDED_SPECS)

    def test_correctness_ineligible_is_the_union(self):
        assert CORRECTNESS_INELIGIBLE_SPECS == (
            EXCLUDED_SPECS | PERFORMANCE_ONLY_SPECS
        )

    def test_constant_matches_spec_scope_markers(self):
        """No second list: the constant equals the set of specs whose JSON
        declares verification.scope == "performance_only"."""
        marked = set()
        for spec_path in sorted(SPEC_ROOT.glob("*.json")):
            spec = json.loads(spec_path.read_text())
            if spec.get("verification", {}).get("scope") == "performance_only":
                marked.add(spec["identity"]["unique_id"])
        assert marked == set(PERFORMANCE_ONLY_SPECS)

    def test_scope_reason_present_and_honest(self):
        """Each marked spec records why: no computed value is extractable
        without editing mixbench source, so no independent comparator exists."""
        for uid in sorted(MIXBENCH_IDS):
            ver = _load_spec(uid)["verification"]
            reason = ver.get("scope_reason", "")
            assert "computed value" in reason, uid
            assert "performance" in reason, uid

    def test_marked_specs_still_validate_against_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(
            (PROJECT_ROOT / "schema" / "spec_schema.json").read_text()
        )
        for uid in sorted(MIXBENCH_IDS):
            jsonschema.validate(_load_spec(uid), schema)


# --------------------------------------------------------------------------- #
# Correctness-denominator consumers exclude performance_only pairs            #
# --------------------------------------------------------------------------- #

def _fake_record(src: str, tgt: str, **extra) -> dict:
    rec = {
        "source_spec": src,
        "target_spec": tgt,
        "model": "m1",
        "augment_level": 0,
        "overall_status": "PASS",
        "temperature": 0.0,
    }
    rec.update(extra)
    return rec


class TestCorrectnessFiltering:
    def test_generate_paper_data_excludes_performance_only(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
        try:
            import generate_paper_data as gpd
        finally:
            sys.path.pop(0)
        records = [
            _fake_record("mixbench-mixbench-cuda", "mixbench-mixbench-omp"),
            _fake_record("rodinia-bfs-cuda", "rodinia-bfs-omp"),
        ]
        kept = gpd.exclude_known_fail(records)
        assert len(kept) == 1
        assert kept[0]["source_spec"] == "rodinia-bfs-cuda"

    def test_quantitative_findings_excludes_performance_only(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
        try:
            import quantitative_findings as qf
        finally:
            sys.path.pop(0)
        records = [
            _fake_record("mixbench-mixbench-cuda", "mixbench-mixbench-omp"),
            _fake_record("rodinia-bfs-cuda", "rodinia-bfs-omp"),
        ]
        kept = qf.exclude_known_fail(records)
        assert len(kept) == 1
        assert kept[0]["source_spec"] == "rodinia-bfs-cuda"

    def test_analyze_eval_excludes_performance_only(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "evaluation"))
        try:
            import analyze_eval
        finally:
            sys.path.pop(0)
        model_dir = tmp_path / "m1"
        model_dir.mkdir()
        (model_dir / "a.json").write_text(json.dumps(
            _fake_record("mixbench-mixbench-omp", "mixbench-mixbench-opencl")
        ))
        (model_dir / "b.json").write_text(json.dumps(
            _fake_record("rodinia-bfs-cuda", "rodinia-bfs-omp")
        ))
        records = analyze_eval.load_results(tmp_path)
        assert [r["source_spec"] for r in records] == ["rodinia-bfs-cuda"]

    def test_derive_l0_passers_excludes_performance_only(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "evaluation"))
        try:
            import derive_l0_passers as dlp
        finally:
            sys.path.pop(0)
        (tmp_path / "a.json").write_text(json.dumps(
            _fake_record("mixbench-mixbench-cuda", "mixbench-mixbench-omp")
        ))
        (tmp_path / "b.json").write_text(json.dumps(
            _fake_record("rodinia-bfs-cuda", "rodinia-bfs-omp")
        ))
        passers = dlp.derive_passers(tmp_path, "m1")
        cells = {(p["source_spec"], p["target_spec"]) for p in passers}
        assert cells == {("rodinia-bfs-cuda", "rodinia-bfs-omp")}

    def test_token_analysis_excludes_performance_only(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
        try:
            import token_analysis as ta
        finally:
            sys.path.pop(0)
        model_dir = tmp_path / "results" / "evaluation" / "m1"
        model_dir.mkdir(parents=True)
        (model_dir / "a.json").write_text(json.dumps(
            _fake_record("mixbench-mixbench-cuda", "mixbench-mixbench-omp")
        ))
        (model_dir / "b.json").write_text(json.dumps(
            _fake_record("rodinia-bfs-cuda", "rodinia-bfs-omp")
        ))
        results = ta.load_all_results(tmp_path)
        assert [r["source_spec"] for r in results] == ["rodinia-bfs-cuda"]


# --------------------------------------------------------------------------- #
# The negative-control evidence behind the verdict                            #
# --------------------------------------------------------------------------- #

class TestNegativeControlEvidence:
    """The three-step rule's step 2 evidence: a seeded do-nothing / wrong
    kernel must FAIL the declared oracle for mixbench to stay in correctness
    statistics. The recorded controls show it does not (or that no
    independent comparator exists), so performance_only is forced."""

    @pytest.mark.parametrize("uid", sorted(MIXBENCH_IDS))
    def test_do_nothing_control_recorded(self, uid):
        path = NEG_CONTROL_DIR / f"{uid}.do_nothing.json"
        assert path.is_file(), f"missing do-nothing control record: {path}"
        rec = json.loads(path.read_text())
        assert rec["spec_id"] == uid
        assert rec["build_status"] == "PASS"
        assert rec["observed_status"] in {"PASS", "FAIL"}
        assert rec["control_kind"] == "do_nothing_kernel"
        assert rec["host"], "control must record the executing host"
        # evidence is bound to the oracle it tested
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "spec_tools"))
        try:
            import audit_oracle_contracts as aoc
        finally:
            sys.path.pop(0)
        assert rec["oracle_sha256"] == aoc.oracle_sha256(_load_spec(uid))

    def test_retention_rule_applied(self):
        """If any variant's do-nothing control PASSes the declared oracle,
        retention in correctness statistics is prohibited; all three mixbench
        specs must then carry the performance_only scope."""
        observed = {}
        for uid in sorted(MIXBENCH_IDS):
            path = NEG_CONTROL_DIR / f"{uid}.do_nothing.json"
            if path.is_file():
                observed[uid] = json.loads(path.read_text())["observed_status"]
        assert observed, "no do-nothing control records found"
        if any(status == "PASS" for status in observed.values()):
            for uid in MIXBENCH_IDS:
                spec = _load_spec(uid)
                assert (
                    spec["verification"].get("scope") == "performance_only"
                ), f"{uid}: blind oracle proven but spec not performance_only"

    def test_disposition_note_recorded(self):
        assert DISPOSITION_FILE.is_file()
        note = json.loads(DISPOSITION_FILE.read_text())
        assert note["verdict"] == "performance_only"
        assert note["extraction_attempt"]["outcome"] == "no_computed_value"
        # the decision must cite its run evidence
        refs = note["evidence_files"]
        assert any("negative_controls" in r for r in refs)
