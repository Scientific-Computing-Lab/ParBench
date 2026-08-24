"""Tests for the figure-fragment values-emitter (ruling D14).

The emitter's correctness proof of record is impl-fig4's independent comparator
(scripts/verify_figure_fragments.py); these tests guard the emitter's own derivation helpers and
assert the committed fragments are idempotent under a re-emit (i.e. they already carry the
canonical values the emitter produces). Data-dependent tests skip when the sealed replay is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import emit_figure_fragments as e  # noqa: E402

_HAS_REPLAY = e.REPLAY_ROOT.exists()
needs_replay = pytest.mark.skipif(not _HAS_REPLAY, reason="sealed replay namespace not present")


# --- pure derivation helpers ------------------------------------------------------------------

def test_api_and_kernel_extraction():
    assert e._api_of("rodinia-bfs-cuda") == "cuda"
    assert e._api_of("hecbench-scan-omp_target") == "omp_target"
    assert e._kernel_of("rodinia-bfs-cuda") == "bfs"
    assert e._kernel_of("hecbench-page-rank-omp") == "page-rank"


def test_status_code_maps_only_the_five():
    assert e._status_code("PASS") == "P"
    assert e._status_code("BUILD_FAIL") == "BF"
    assert e._status_code("EXTRACTION_FAIL") == "EF"
    # unmapped verdicts and missing records collapse to NA
    assert e._status_code("ERROR") == e.NA
    assert e._status_code(None) == e.NA


def test_promote_identity_fills_only_missing():
    rec = {"model": None, "source_spec": "", "overall_status": "PASS", "sample_id": 0,
           "parent": {"model": "m", "source_spec": "s-k-cuda", "sample_id": 7}}
    e._promote_identity(rec)
    assert rec["model"] == "m"
    assert rec["source_spec"] == "s-k-cuda"
    assert rec["sample_id"] == 0  # a real 0 is kept, not overwritten by the parent's 7
    assert rec["overall_status"] == "PASS"  # never promoted from parent


def test_wilson_known_value():
    lo, hi = e._wilson(15, 110)
    assert 0.084 < lo < 0.085 and 0.212 < hi < 0.213  # standard 95% Wilson for 15/110
    assert e._wilson(0, 0) == (0.0, 0.0)


def test_fmt_like_preserves_decimals():
    assert e._fmt_like("40.3", 40.34) == "40.3"
    assert e._fmt_like("0.0", 0.0) == "0.0"
    assert e._fmt_like("12", 12.6) == "13"


# --- idempotency: the committed fragments already carry the canonical values -------------------

@needs_replay
@pytest.mark.parametrize("fragment", ["f3", "f4", "f5", "f6", "f7", "aug_heatmap"])
def test_emitter_idempotent(fragment):
    all_records = e.load_all_records()
    l0 = e.load_l0_records()
    grid = {(r["model"], r["kernel"], r["direction"]): r["overall_status"] for r in l0}
    if fragment == "f3":
        r = e.emit_f3(grid, apply=False)
    elif fragment == "f4":
        r = e.emit_f4(l0, apply=False)
    elif fragment == "f5":
        r = e.emit_f5(all_records, apply=False)
    elif fragment == "f6":
        r = e.emit_f6(l0, apply=False)
    elif fragment == "f7":
        r = e.emit_f7(all_records, l0, apply=False)
    else:
        r = e.emit_aug_heatmap(all_records, l0, apply=False)
    assert r["changed"] == 0, f"{fragment} committed fragment diverges from canonical by {r['changed']}"


@needs_replay
def test_canonical_l0_is_s0_only():
    # the sealed replay carries no base files, so canonical L0 must come from the s0 samples
    recs = e.load_all_records()
    assert recs, "replay present but no records loaded"
    assert all(r["is_sample"] for r in e.load_l0_records())
