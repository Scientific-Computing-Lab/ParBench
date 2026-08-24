"""Explicit result-root plumbing on the analysis consumers (Task 5).

Each analysis command must read the SUPPLIED root, never silently fall back
to ``results/evaluation``. The falsifiable check: point each command at an
EMPTY (or missing) explicit root. If the command honored the root, it finds
zero records and fails loudly naming that root; if it silently fell back, it
would find the populated real corpus and proceed — which these tests would
catch as an unexpected success.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=300,
    )


def test_statistical_analysis_reads_supplied_root(tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    r = run([
        str(SCRIPTS / "analysis" / "statistical_analysis.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(empty),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert r.returncode != 0
    assert str(empty) in r.stderr


def test_quantitative_findings_reads_supplied_root(tmp_path):
    empty = tmp_path / "empty-root"
    (empty / "some-model").mkdir(parents=True)
    r = run([
        str(SCRIPTS / "analysis" / "quantitative_findings.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(empty),
        "--model-dir", "some-model",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert r.returncode != 0
    assert str(empty / "some-model") in r.stderr


def test_cross_model_comparison_reads_supplied_model_data(tmp_path):
    missing_a = tmp_path / "a.json"
    missing_b = tmp_path / "b.json"
    r = run([
        str(SCRIPTS / "analysis" / "cross_model_comparison.py"),
        "--model-data", str(missing_a),
        "--model-data", str(missing_b),
        "--output", str(tmp_path / "out.json"),
    ])
    assert r.returncode != 0
    assert str(missing_a) in r.stderr


def test_cross_model_comparison_requires_two_model_data(tmp_path):
    only = tmp_path / "a.json"
    only.write_text("{}")
    r = run([
        str(SCRIPTS / "analysis" / "cross_model_comparison.py"),
        "--model-data", str(only),
        "--output", str(tmp_path / "out.json"),
    ])
    assert r.returncode != 0
    assert "at least two" in r.stderr


def test_generate_paper_claims_reads_supplied_analysis_root(tmp_path):
    empty = tmp_path / "empty-analysis"
    empty.mkdir()
    r = run([
        str(SCRIPTS / "analysis" / "generate_paper_claims.py"),
        "--project-root", str(PROJECT_ROOT),
        "--analysis-root", str(empty),
        "--output", str(tmp_path / "claims.json"),
    ])
    assert r.returncode != 0
    assert str(empty) in r.stderr
    assert not (tmp_path / "claims.json").exists()


def test_augmentation_analysis_reads_supplied_root(tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    r = run([
        str(SCRIPTS / "analysis" / "augmentation_analysis.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(empty),
        "--model-dir", "some-model",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert r.returncode != 0
    assert str(empty) in r.stderr


def test_build_error_taxonomy_reads_supplied_root(tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    r = run([
        str(SCRIPTS / "analysis" / "build_error_taxonomy.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(empty),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert r.returncode != 0
    assert str(empty) in r.stderr


def test_token_analysis_reads_supplied_root(tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    r = run([
        str(SCRIPTS / "analysis" / "token_analysis.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(empty),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert r.returncode != 0
    assert str(empty) in r.stderr


def test_generate_paper_figures_reads_supplied_root(tmp_path):
    """An eval-data figure against an empty explicit root must FAIL LOUDLY,
    naming that root — never silently rendered from the real corpus and never
    pruned to a successful-looking empty run (batch-review 2026-08-17 finding 6).
    """
    empty = tmp_path / "empty-root"
    empty.mkdir()
    r = run([
        str(SCRIPTS / "generate_paper_figures.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(empty),
        "--output-dir", str(tmp_path / "figs"),
        "--figure", "F3",
    ])
    assert r.returncode != 0
    assert str(empty) in r.stderr
    assert "no evaluation data" in r.stderr.lower()


def test_generate_paper_data_requires_results_dir(tmp_path):
    """--results-dir stays required (pre-existing contract, preserved)."""
    r = run([
        str(SCRIPTS / "analysis" / "generate_paper_data.py"),
        "--results-dir", str(tmp_path / "missing"),
        "--output", str(tmp_path / "out.json"),
    ])
    assert r.returncode != 0
    assert "not found" in r.stderr
