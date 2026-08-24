"""
pytest configuration for the ParBench test suite.

Auto-skips tests marked @pytest.mark.integration when the benchmark source
directories are not present on disk (e.g., macOS dev machine, CI).

On the Linux GPU machine where benchmark sources exist, integration tests
run as normal. Separate them explicitly with:

    pytest tests/ -m "not integration"   # unit only
    pytest tests/ -m integration          # integration only
    pytest tests/                          # all (integration auto-skips if dirs missing)

Also gates @pytest.mark.llm tests behind PARBENCH_RUN_LLM_TESTS=1 (plan 02-07).
LLM-marked tests cost real money — they are SKIPPED unless the env var is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Public export — downstream tests import PROJECT_ROOT from here.
PROJECT_ROOT = Path(__file__).parent.parent

_RODINIA_SRC = PROJECT_ROOT / "rodinia" / "rodinia-src"
_HECBENCH_SRC = PROJECT_ROOT / "HeCBench-master"


def stage_tmp_project_root(tmp_path: Path) -> Path:
    """Symlink PROJECT_ROOT entries into ``tmp_path`` so tests using ``tmp_path``
    as ``--project-root`` find ``manifest.jsonl``, ``specs/``, and benchmark
    source dirs. Returns ``tmp_path`` for chaining.

    Excludes ``results/`` — the ParBench eval pipeline writes per-task result
    JSONs under ``{project_root}/results/evaluation/<model>/``, and the real
    ``results/`` tree is inviolable (CLAUDE.md D-15, memory
    feedback_protect_cuda_omp_results / feedback_protect_qwen_results). Staging
    real entries as symlinks while leaving ``tmp_path/results/`` as a fresh
    local directory keeps real-API tests correct without polluting production
    eval data.
    """
    for entry in PROJECT_ROOT.iterdir():
        if entry.name == "results":
            continue
        (tmp_path / entry.name).symlink_to(entry)
    return tmp_path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests when benchmark source directories are absent.

    Also skips llm-marked tests unless PARBENCH_RUN_LLM_TESTS=1 (plan 02-07).
    """
    # llm gate (plan 02-07) — applies regardless of integration status.
    run_llm = os.environ.get("PARBENCH_RUN_LLM_TESTS") == "1"
    skip_llm = pytest.mark.skip(
        reason="llm tests require PARBENCH_RUN_LLM_TESTS=1 (real API calls cost money)"
    )
    for item in items:
        if item.get_closest_marker("llm") and not run_llm:
            item.add_marker(skip_llm)

    # integration gate — skip when benchmark source dirs are missing.
    benchmark_dirs_present = _RODINIA_SRC.is_dir() and _HECBENCH_SRC.is_dir()
    if benchmark_dirs_present:
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "Integration tests require benchmark source dirs on disk "
            f"(missing: {_RODINIA_SRC} or {_HECBENCH_SRC}). "
            "Run on the Linux GPU machine, or use -m 'not integration' to run unit tests only."
        )
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_marker)
