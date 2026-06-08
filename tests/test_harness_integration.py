"""Integration smoke tests for the full build-run-verify pipeline.

Loads 1 spec per benchmark suite (Rodinia, XSBench, RSBench, mixbench, HeCBench),
runs the complete build -> run -> verify pipeline, and asserts PASS at each stage.

Run:  python3 -m pytest tests/test_harness_integration.py -v -m integration
Unit: python3 -m pytest tests/test_harness_integration.py -v -m "not integration"

Requires: benchmark source dirs on disk + GPU (integration tests auto-skip otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.builder import build_spec
from harness.constants import EXCLUDED_SPECS
from harness.models import Status
from harness.runner import run_spec
from harness.spec_loader import load_spec
from harness.verifier import verify_run


PROJECT_ROOT = Path(".")
SPECS_DIR = PROJECT_ROOT / "specs"

# One known-PASS spec per suite -- avoids KNOWN_FAIL specs.
SUITE_SPECS = {
    "rodinia": SPECS_DIR / "rodinia-bfs-cuda.json",
    "xsbench": SPECS_DIR / "xsbench-xsbench-cuda.json",
    "rsbench": SPECS_DIR / "rsbench-rsbench-cuda.json",
    "mixbench": SPECS_DIR / "mixbench-mixbench-cuda.json",
    "hecbench": SPECS_DIR / "hecbench-bezier-surface-cuda.json",
}

# The 9 KNOWN_FAIL spec IDs -- must match harness/constants.py exactly.
_EXPECTED_EXCLUDED = frozenset({
    "rodinia-kmeans-cuda",
    "rodinia-mummergpu-cuda",
    "rodinia-mummergpu-omp",
    "rodinia-hybridsort-cuda",
    "rodinia-nn-opencl",
    "rodinia-kmeans-opencl",
    "rodinia-backprop-opencl",
    "hecbench-stencil1d-omp_target",
    "hecbench-scan-omp_target",
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(SUITE_SPECS.keys()), ids=list(SUITE_SPECS.keys()))
def suite_spec(request: pytest.FixtureRequest) -> tuple[str, dict[str, Any]]:
    """Parametrized fixture: returns (suite_name, loaded_spec) for each suite."""
    suite = request.param
    spec_path = SUITE_SPECS[suite]
    spec = load_spec(str(spec_path))
    return suite, spec


# ---------------------------------------------------------------------------
# Integration tests (require benchmark dirs + GPU)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline(suite_spec: tuple[str, dict[str, Any]]) -> None:
    """Build, run, and verify 1 spec per suite -- full pipeline smoke test."""
    suite, spec = suite_spec
    spec_id = spec["identity"]["unique_id"]

    if spec_id in EXCLUDED_SPECS:
        pytest.skip(f"{spec_id} is KNOWN_FAIL -- see known-issues.md")

    # Build
    build_result = build_spec(spec, PROJECT_ROOT, verbose=True)
    assert build_result.status == Status.PASS, (
        f"Build FAILED for {spec_id}: {build_result.stderr[:500]}"
    )

    # Run
    run_result = run_spec(
        spec, PROJECT_ROOT, configuration="correctness", verbose=True,
    )
    assert run_result.status == Status.PASS, (
        f"Run FAILED for {spec_id} (exit={run_result.exit_code}): "
        f"{run_result.stderr[:500]}"
    )

    # Verify
    ver_result = verify_run(spec, run_result)
    assert ver_result.status == Status.PASS, (
        f"Verify FAILED for {spec_id}: {ver_result.details}"
    )


# ---------------------------------------------------------------------------
# Unit tests (no GPU/benchmark dirs required)
# ---------------------------------------------------------------------------


def test_excluded_specs_count() -> None:
    """EXCLUDED_SPECS has exactly 9 entries matching known-issues.md."""
    assert len(EXCLUDED_SPECS) == 9, (
        f"Expected 9 EXCLUDED_SPECS, got {len(EXCLUDED_SPECS)}"
    )
    for spec_id in _EXPECTED_EXCLUDED:
        assert spec_id in EXCLUDED_SPECS, f"{spec_id} missing from EXCLUDED_SPECS"


def test_excluded_specs_are_frozen() -> None:
    """EXCLUDED_SPECS is a frozenset (immutable)."""
    assert isinstance(EXCLUDED_SPECS, frozenset)
