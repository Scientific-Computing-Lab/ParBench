"""batch-review 2026-08-17 HIGH finding 3: scripts/build_artifact.sh must ship
ONLY the canonical three-model final analysis build (results/analysis/final/*.json)
into the release, never the superseded top-level per-model files
(paper_data_<model>.json / quantitative_findings_<model>.json carry pre-exclusion
denominators and the wrong GPT-5.4 id). It must also default its raw-result
source to the sealed replay, not results/evaluation.

These tests build a real artifact tree with --dry-run (no Docker) into a temp
staging dir and inspect the staged analysis payload. The build's copy step
(step 1) completes before the platform-fragile anonymization sed (step 3), so
the staged payload is populated regardless of the build's final exit code.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_artifact.sh"
SEALED_REPLAY = PROJECT_ROOT / "results" / "evaluation_final" / "parbench-final-v1"
FINAL_ANALYSIS = PROJECT_ROOT / "results" / "analysis" / "final"


@pytest.fixture(scope="module")
def built_tree(tmp_path_factory) -> Path:
    if not SEALED_REPLAY.is_dir() or not FINAL_ANALYSIS.is_dir():
        pytest.skip("sealed replay or final analysis build not present")
    staging = tmp_path_factory.mktemp("stage")
    output = staging / "artifact.tar.gz"
    # No --result-root: exercise the DEFAULT source (must be the sealed replay).
    subprocess.run(
        ["bash", str(BUILDER), "--dry-run",
         "--staging-dir", str(staging), "--output", str(output)],
        capture_output=True, text=True,
    )
    art = staging / "parbench-artifact"
    analysis = art / "results" / "analysis"
    if not analysis.is_dir():
        pytest.skip("build did not reach the analysis-copy step in this env")
    return art


def test_ships_canonical_final_analysis(built_tree):
    final_dir = built_tree / "results" / "analysis" / "final"
    assert final_dir.is_dir(), "release is missing results/analysis/final/"
    # The three-model canonical files must be present, incl. the corrected id.
    for name in ("quantitative_findings.json", "paper_claims.json",
                 "paper_data_azure-gpt-5.4.json"):
        assert (final_dir / name).is_file(), f"final/ missing {name}"


def test_omits_stale_top_level_per_model_files(built_tree):
    analysis = built_tree / "results" / "analysis"
    stale = sorted(analysis.glob("paper_data_*.json")) + \
        sorted(analysis.glob("quantitative_findings_*.json"))
    assert stale == [], (
        f"release shipped superseded top-level per-model files: "
        f"{[p.name for p in stale]}"
    )
    # The wrong-id spelling must never appear at the top level either.
    assert not (analysis / "paper_data_azure_gpt54.json").exists()


def test_does_not_ship_old_evidence_subdir(built_tree):
    assert not (built_tree / "results" / "analysis" / "final" / "old_evidence").exists(), \
        "release shipped the stale final/old_evidence/ subdir"


def test_default_result_root_is_sealed_replay(built_tree):
    # The raw result JSONs come from the sealed replay's per-model namespaces,
    # staged under the tarball's results/evaluation/ path.
    staged_models = sorted(
        p.name for p in (built_tree / "results" / "evaluation").iterdir()
        if p.is_dir()
    )
    assert staged_models == [
        "azure-gpt-5.3-codex", "azure-gpt-5.4", "together-qwen-3.5-397b-a17b",
    ], f"unexpected staged model namespaces: {staged_models}"
