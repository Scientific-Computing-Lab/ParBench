"""Tests for T1/T3/T4/T5 table generators in generate_paper_figures.py.

Values are pinned to the canonical results/analysis/final/ build (2026-08-13),
which the generators read by default since 2026-08-21; they match the paper's
printed tables (appendix tab:overall-pass, results pass@k prose).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_generator(fn: Callable, filename: str, tmp_path: Path, **kwargs) -> str:
    """Run a table generator and return the LaTeX content after basic checks."""
    fn(PROJECT_ROOT, tmp_path, verbose=True, **kwargs)
    out = tmp_path / filename
    assert out.exists(), f"{filename} not created"
    content = out.read_text()
    assert r"\begin{tabular}" in content
    assert r"\end{tabular}" in content
    return content


def _t1_rows(content: str) -> list[list[int]]:
    """Integer cells (PASS..Extract, Total) of every model row in T1."""
    rows = []
    for line in content.splitlines():
        if "&" in line and line.strip().endswith(r"\\") and "Model" not in line:
            cells = [c.strip() for c in line.split("&")]
            rows.append([int(c) for c in cells[1:7]])
    return rows


class TestT1OverallPass:
    def test_matches_canonical_build_and_paper(self, tmp_path):
        from scripts.generate_paper_figures import generate_t1_overall_pass

        content = _run_generator(generate_t1_overall_pass, "t1_overall_pass.tex", tmp_path)
        assert r"191 & 235 & 111 & 67 & 0 & 604 & 31.6\% [28.0, 35.4]" in content  # Qwen
        assert r"440 & 115 & 95 & 131 & 3 & 784 & 56.1\% [52.6, 59.6]" in content  # GPT-5.4
        assert r"412 & 130 & 99 & 131 & 0 & 772 & 53.4\% [49.8, 56.9]" in content  # Codex

    def test_status_columns_sum_to_total(self, tmp_path):
        """NOT_REPLAYABLE records are folded into the Extract column, so the
        five status columns always sum to Total (2026-08-21 Codex finding)."""
        from scripts.generate_paper_figures import generate_t1_overall_pass

        content = _run_generator(generate_t1_overall_pass, "t1_overall_pass.tex", tmp_path)
        rows = _t1_rows(content)
        assert len(rows) == 3
        for row in rows:
            assert sum(row[:5]) == row[5], row


class TestT3Passk:
    def test_matches_canonical_build_and_paper(self, tmp_path):
        from scripts.generate_paper_figures import generate_t3_passk

        content = _run_generator(generate_t3_passk, "t3_passk.tex", tmp_path)
        assert r"136 & 20.3\% & 27.9\% & 18 & 20 & 98" in content  # Qwen
        assert r"136 & 46.3\% & 52.2\% & 53 & 18 & 65" in content  # GPT-5.4
        assert r"136 & 45.1\% & 49.3\% & 55 & 12 & 69" in content  # Codex


class TestT4Augmentation:
    def test_generates_valid_latex_with_augmentation_rates(self, tmp_path):
        from scripts.generate_paper_figures import generate_t4_augmentation

        content = _run_generator(generate_t4_augmentation, "t4_augmentation.tex", tmp_path)
        assert "L0" in content
        assert "L4" in content
        # one percentage cell per level per model row
        assert len(re.findall(r"\d+\.\d\\%", content)) >= 3 * 5


class TestT5Stats:
    def test_matches_canonical_pairwise_tests(self, tmp_path):
        from scripts.generate_paper_figures import generate_t5_stats

        content = _run_generator(generate_t5_stats, "t5_stats.tex", tmp_path)
        # Codex vs GPT-5.4: OR 0.895 [0.733, 1.093], p=0.855, h=-0.055 (paper prose)
        assert "0.895" in content
        assert "0.855" in content
        assert "-0.055" in content
        assert "2.475" in content  # Codex vs Qwen OR


class TestFindingsDirOverride:
    def test_findings_dir_override_is_honored(self, tmp_path):
        """reproduce.sh passes its own regenerated aggregates via findings_dir;
        a synthetic directory must be read instead of results/analysis/final."""
        import json

        from scripts.generate_paper_figures import generate_t1_overall_pass

        findings = tmp_path / "findings"
        findings.mkdir()
        synthetic = {
            "canonical": {
                "failure_taxonomy": {
                    "status_counts": {"PASS": 7, "BUILD_FAIL": 2, "NOT_REPLAYABLE": 1},
                    "total_records": 10,
                },
                "aggregate_pass_rates": {
                    "overall": {"value": 0.7, "ci_lower": 0.4, "ci_upper": 0.9}
                },
            }
        }
        (findings / "quantitative_findings_azure-gpt-5.4.json").write_text(
            json.dumps(synthetic)
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        content = _run_generator(
            generate_t1_overall_pass, "t1_overall_pass.tex", out_dir,
            findings_dir=findings,
        )
        assert r"7 & 2 & 0 & 0 & 1 & 10 & 70.0\% [40.0, 90.0]" in content
        assert "191" not in content  # canonical build NOT read

    def test_unmapped_status_fails_loudly(self, tmp_path):
        """A status with no T1 column (ERROR, PROVIDER_ERROR, SKIP, ...) must
        raise instead of being silently dropped from a table whose columns
        sum to Total (2026-08-21 Codex finding)."""
        import json

        import pytest

        from scripts.generate_paper_figures import generate_t1_overall_pass

        findings = tmp_path / "findings"
        findings.mkdir()
        synthetic = {
            "canonical": {
                "failure_taxonomy": {
                    "status_counts": {"PASS": 7, "PROVIDER_ERROR": 3},
                    "total_records": 10,
                },
                "aggregate_pass_rates": {
                    "overall": {"value": 0.7, "ci_lower": 0.4, "ci_upper": 0.9}
                },
            }
        }
        (findings / "quantitative_findings_azure-gpt-5.4.json").write_text(
            json.dumps(synthetic)
        )
        with pytest.raises(ValueError, match="PROVIDER_ERROR"):
            generate_t1_overall_pass(PROJECT_ROOT, tmp_path, verbose=False,
                                     findings_dir=findings)
