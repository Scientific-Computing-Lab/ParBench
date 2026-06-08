"""Tests for T1/T3/T4/T5 table generators in generate_paper_figures.py."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_generator(fn: Callable, filename: str, tmp_path: Path) -> str:
    """Run a table generator and return the LaTeX content after basic checks."""
    fn(PROJECT_ROOT, tmp_path, verbose=True)
    out = tmp_path / filename
    assert out.exists(), f"{filename} not created"
    content = out.read_text()
    assert r"\begin{tabular}" in content
    assert r"\end{tabular}" in content
    return content


class TestT1OverallPass:
    def test_generates_valid_latex_with_correct_counts(self, tmp_path):
        from scripts.generate_paper_figures import generate_t1_overall_pass

        content = _run_generator(generate_t1_overall_pass, "t1_overall_pass.tex", tmp_path)
        assert "230" in content  # Qwen PASS count
        assert "621" in content  # GPT-5.4 PASS count
        assert "604" in content  # Codex PASS count
        assert "626" in content  # Qwen total
        assert "822" in content  # GPT-5.4 total
        assert "814" in content  # Codex total


class TestT3Passk:
    def test_generates_valid_latex_with_correct_passk(self, tmp_path):
        from scripts.generate_paper_figures import generate_t3_passk

        content = _run_generator(generate_t3_passk, "t3_passk.tex", tmp_path)
        assert "142" in content  # n_tasks per model
        assert "23.9" in content  # Qwen pass@1
        assert "62.7" in content  # GPT-5.4 pass@1
        assert "35.2" in content  # Qwen pass@3
        assert "69.7" in content  # GPT-5.4 pass@3


class TestT4Augmentation:
    def test_generates_valid_latex_with_augmentation_rates(self, tmp_path):
        from scripts.generate_paper_figures import generate_t4_augmentation

        content = _run_generator(generate_t4_augmentation, "t4_augmentation.tex", tmp_path)
        assert "62.7" in content  # Codex/GPT-5.4 L0 rate
        assert "23.9" in content  # Qwen L0 rate
        assert "L0" in content
        assert "L4" in content


class TestT5Stats:
    def test_generates_valid_latex_with_pairwise_tests(self, tmp_path):
        from scripts.generate_paper_figures import generate_t5_stats

        content = _run_generator(generate_t5_stats, "t5_stats.tex", tmp_path)
        assert "Codex" in content or "codex" in content
        assert "Qwen" in content or "qwen" in content
        assert "0.931" in content  # Codex vs GPT-5.4 OR
        assert "4.952" in content  # Codex vs Qwen OR
