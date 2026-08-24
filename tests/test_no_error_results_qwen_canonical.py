"""Verify the Qwen 3.5 397B canonical Phase 3 results are clean (no ERROR status).

The 14 ERROR results from the original run were caused by transient Together.ai
503/429 failures. After deleting those JSONs and re-running, all 504 results
must be non-ERROR and the total count must remain 504.
"""
from __future__ import annotations

import json
import os

import pytest

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "evaluation",
    "together-qwen-3.5-397b-a17b",
)


def _load_results() -> list[dict]:
    d = os.path.normpath(RESULTS_DIR)
    if not os.path.isdir(d):
        pytest.skip(f"Results directory not found: {d}")
    results = []
    import re
    for fname in os.listdir(d):
        # Only canonical sample files (-s0, -s1, -s2); excludes ablation (-L1 .. -L4)
        if not fname.endswith(".json") or not re.search(r"-s\d+\.json$", fname):
            continue
        with open(os.path.join(d, fname)) as f:
            results.append(json.load(f))
    return results


def test_no_error_results() -> None:
    """No result file may have overall_status == 'ERROR' after the re-run."""
    results = _load_results()
    errors = [r for r in results if r.get("overall_status") == "ERROR"]
    filenames = [
        os.path.basename(r.get("result_file", r.get("source_spec", "unknown")))
        for r in errors
    ]
    assert errors == [], (
        f"{len(errors)} ERROR result(s) still present:\n" + "\n".join(filenames)
    )


def test_total_result_count() -> None:
    """Canonical run must have exactly 504 results (168 pairs × 3 samples)."""
    results = _load_results()
    assert len(results) == 504, (
        f"Expected 504 canonical results, found {len(results)}"
    )
