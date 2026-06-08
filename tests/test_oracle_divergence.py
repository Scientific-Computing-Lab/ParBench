"""S7b strong-oracle divergence probes.

Each test captures the audit finding that supported the 2026-04-19 S7b oracle downgrade.
Pair tests compare spec-embedded `expected_sha256` (or baseline Distance value) across
source/target APIs; singleton tests document structural (not hash) divergence.

Rationale: the harness `file_hash` strategy was already confirmed deterministic by the S6
spec-verify sweep (88/88 PASS). The only novel question for S7b is whether cross-API
outputs are byte-identical — that is, whether a faithful LLM translation can satisfy
both source and target file_hash references.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _file_hash_for(spec_name: str) -> str | None:
    data = json.loads((SPECS / f"{spec_name}.json").read_text())
    for strat in data["verification"]["strategies"]:
        if strat.get("type") == "file_hash":
            return strat.get("expected_sha256")
    return None


def _oracle_strength(spec_name: str) -> str:
    data = json.loads((SPECS / f"{spec_name}.json").read_text())
    return data["verification"].get("oracle_strength", "unknown")


# ---------------------------------------------------------------------------
# CLEAN (convergent) — expected PASS: same hash across APIs means strong oracle
# is safe for cross-API translation.
# ---------------------------------------------------------------------------

def test_bptree_cuda_omp_hashes_converge():
    """bptree CUDA and OMP produce byte-identical output.txt (no FP reduction,
    deterministic I/O sort). Strong oracle is safe; bptree keeps file_hash."""
    cuda = _file_hash_for("rodinia-bptree-cuda")
    omp = _file_hash_for("rodinia-bptree-omp")
    assert cuda is not None, "bptree-cuda should still carry file_hash"
    assert omp is not None, "bptree-omp should still carry file_hash"
    assert cuda == omp, f"bptree cross-API hash should match (cuda={cuda}, omp={omp})"
    assert _oracle_strength("rodinia-bptree-cuda") == "strong"
    assert _oracle_strength("rodinia-bptree-omp") == "strong"


# ---------------------------------------------------------------------------
# DIVERGENT (FP reduction-order) — downgraded to weak oracle.
# These tests document the divergence finding. After S7b downgrade the specs
# no longer carry file_hash, so we assert oracle_strength=weak as evidence the
# downgrade landed, and capture the known-divergent hashes as documentation.
# ---------------------------------------------------------------------------

_DIVERGENT_HASHES = {
    "cfd": ("ab07aa0cb89cd5737eb61e9ebdc07cf5e9893b12299a2a72a7598f6067d8278d",
            "4283a12d27bd594258fa00b49ecfb053d7d90ae58ce0ecb1dd3ad52772486329"),
    "hotspot": ("06b21039f105501b2e297e43809f8db2faa3bf1253b11fad6f442e09a89c6fe2",
                "f90474ff7bee2b6cf5f2632f75da25d2ac1b671541c15592a53c6ed80dd44af4"),
    "myocyte": ("1e329f029af6524ec9cf589852b560c39c03f9d3badebab7f146fc9b9d7ead05",
                "59efddcb7914a312d4de57cf1ee30cfaad9ef9457afac70f099fb3d60457e40c"),
}


@pytest.mark.parametrize("kernel,apis", [
    ("cfd", ("cuda", "omp")),
    ("hotspot", ("cuda", "omp")),
    ("myocyte", ("cuda", "opencl")),
])
def test_fp_divergent_pair_downgraded(kernel, apis):
    """CFD / hotspot / myocyte diverge bit-exact across APIs due to floating-point
    non-associativity in parallel reductions. S7b downgraded to weak oracle."""
    a, b = apis
    assert _oracle_strength(f"rodinia-{kernel}-{a}") == "weak", (
        f"{kernel}-{a} must be weak after S7b downgrade")
    assert _oracle_strength(f"rodinia-{kernel}-{b}") == "weak", (
        f"{kernel}-{b} must be weak after S7b downgrade")
    assert _file_hash_for(f"rodinia-{kernel}-{a}") is None, (
        f"{kernel}-{a} must not carry file_hash after S7b downgrade")
    assert _file_hash_for(f"rodinia-{kernel}-{b}") is None, (
        f"{kernel}-{b} must not carry file_hash after S7b downgrade")
    # Audit record: the pre-downgrade hashes were divergent.
    h_a, h_b = _DIVERGENT_HASHES[kernel]
    assert h_a != h_b, f"{kernel} pre-downgrade hashes should differ (audit evidence)"


# ---------------------------------------------------------------------------
# STRUCTURAL ASYMMETRY (synthesis-unfair) — downgraded to weak oracle.
# ---------------------------------------------------------------------------

def test_nw_omp_downgraded_traceback_asymmetry():
    """CUDA nw/needle.cu:194 comments out `#define TRACEBACK` — no result.txt.
    OMP nw/needle.cpp:314 has TRACEBACK active — writes result.txt. Faithful
    CUDA->OMP translation cannot produce result.txt without synthesis; OMP->CUDA
    translation correctly drops TRACEBACK and lacks result.txt entirely. file_hash
    punishes faithful translation in both directions. S7b downgraded to weak."""
    assert _oracle_strength("rodinia-nw-omp") == "weak"
    assert _file_hash_for("rodinia-nw-omp") is None


def test_nn_cuda_downgraded_distance_asymmetry():
    """CUDA nn_cuda.cu:185 emits `printf("%s --> Distance=%f\\n", ...)`.
    OMP nn_openmp.c emits no `Distance=` lines. numeric_comparison on Distance=
    punishes a faithful OMP->CUDA translation that does not synthesize the
    cuda-only print. S7b downgraded to weak."""
    spec = json.loads((SPECS / "rodinia-nn-cuda.json").read_text())
    types = [s.get("type") for s in spec["verification"]["strategies"]]
    assert _oracle_strength("rodinia-nn-cuda") == "weak"
    assert "numeric_comparison" not in types, (
        "nn-cuda must drop numeric_comparison after S7b downgrade")
