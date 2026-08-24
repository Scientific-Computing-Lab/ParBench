"""Tests for scripts/analysis/emit_release_inventory.py.

The record/file counts are RE-DERIVED by export_translations over the immutable
corpus, so this test asserts the emitted inventory matches that canonical export
(2,341 with source / 3 without / 2,938 generated files; with + without == the
2,344 seen corpus). The zip sha256 is recorded-not-recomputed provenance and
must pass through verbatim.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "emit_release_inventory",
    PROJECT_ROOT / "scripts" / "analysis" / "emit_release_inventory.py")
emit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(emit)

_SHA = "a" * 64


def test_release_inventory_counts_match_export_and_provenance():
    inv = emit.build_inventory(PROJECT_ROOT, _SHA, ["c9c1e64e", "bf645a5c"])
    # Counts re-derived from the immutable corpus by export_translations.
    assert inv["records_with_source"] == 2341
    assert inv["records_without_source"] == 3
    assert inv["generated_files"] == 2938
    # with + without accounts for every seen record in the corpus.
    assert inv["records_with_source"] + inv["records_without_source"] == 2344
    # zip sha256 is recorded provenance, passed through verbatim (not recomputed).
    assert inv["zip_sha256_recorded"] == _SHA
    assert inv["source_commits"] == ["c9c1e64e", "bf645a5c"]
    assert "not recomputed" in inv["note"].lower()
    assert inv["source"]["translations_namespace"] == "results/evaluation"
