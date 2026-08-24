"""Tests for scripts/evaluation/replay_records.promote_parent_metadata.

Replay records (Task 8, replay_kind == "stored_translation_replay") carry
their identity under "parent". The adapter promotes it in memory so analysis
loaders classify sealed records correctly; sealed files are never rewritten.
Regression for the 2026-08-12 Task A review finding 2 (records loaded as
unknown model / unknown direction, escaping eligibility filters).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.replay_records import promote_parent_metadata  # noqa: E402


def _replay_record(**overrides):
    rec = {
        "replay_schema_version": 1,
        "replay_kind": "stored_translation_replay",
        "overall_status": "VERIFY_FAIL",
        "parent": {
            "model": "azure-gpt-5.4",
            "source_spec": "rodinia-streamcluster-cuda",
            "target_spec": "rodinia-streamcluster-omp",
            "augment_level": 0,
            "sample_id": 0,
            "overall_status": "PASS",
        },
    }
    rec.update(overrides)
    return rec


class TestPromoteParentMetadata:
    def test_promotes_identity_fields(self):
        rec = promote_parent_metadata(_replay_record())
        assert rec["model"] == "azure-gpt-5.4"
        assert rec["source_spec"] == "rodinia-streamcluster-cuda"
        assert rec["target_spec"] == "rodinia-streamcluster-omp"
        assert rec["augment_level"] == 0
        assert rec["sample_id"] == 0

    def test_never_promotes_parent_overall_status(self):
        rec = promote_parent_metadata(_replay_record())
        assert rec["overall_status"] == "VERIFY_FAIL"

    def test_existing_top_level_values_win(self):
        rec = promote_parent_metadata(_replay_record(model="other-model"))
        assert rec["model"] == "other-model"

    def test_existing_zero_is_kept(self):
        rec = _replay_record(augment_level=0)
        rec["parent"]["augment_level"] = 3
        promote_parent_metadata(rec)
        assert rec["augment_level"] == 0

    def test_non_replay_record_untouched(self):
        rec = {"model": "m", "source_spec": "a", "target_spec": "b",
               "overall_status": "PASS"}
        before = dict(rec)
        assert promote_parent_metadata(rec) == before

    def test_malformed_parent_ignored(self):
        rec = {"parent": "not-a-dict", "overall_status": "ERROR"}
        assert promote_parent_metadata(rec)["overall_status"] == "ERROR"


class TestLoadersClassifyReplayRecords:
    def _write_replay_dir(self, tmp_path, source, target):
        model_dir = tmp_path / "azure-gpt-5.4"
        model_dir.mkdir()
        rec = _replay_record()
        rec["parent"]["source_spec"] = source
        rec["parent"]["target_spec"] = target
        (model_dir / f"{source}-to-{target}-s0.json").write_text(json.dumps(rec))
        return tmp_path

    def test_analyze_eval_classifies_replay_records(self, tmp_path):
        from scripts.evaluation.analyze_eval import load_results
        root = self._write_replay_dir(
            tmp_path, "rodinia-streamcluster-cuda", "rodinia-streamcluster-omp")
        records = load_results(root)
        assert len(records) == 1
        assert records[0]["model"] == "azure-gpt-5.4"
        assert records[0]["direction"] == "cuda-to-omp"

    def test_analyze_eval_applies_eligibility_to_replay_records(self, tmp_path):
        from scripts.evaluation.analyze_eval import load_results
        root = self._write_replay_dir(
            tmp_path, "mixbench-mixbench-cuda", "mixbench-mixbench-omp")
        assert load_results(root) == []


def test_entry_points_run_without_pythonpath(tmp_path):
    """The documented direct invocations must work with no PYTHONPATH (the
    installed package covers harness/ but not scripts/; fix re-check 2026-08-12)."""
    import os
    import subprocess
    env = {**os.environ, "PYTHONPATH": ""}
    for script in ("scripts/evaluation/analyze_eval.py",
                   "scripts/analysis/token_analysis.py"):
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script), "--help"],
            capture_output=True, text=True, env=env, cwd=tmp_path)
        assert r.returncode == 0, f"{script} --help failed:\n{r.stderr}"
