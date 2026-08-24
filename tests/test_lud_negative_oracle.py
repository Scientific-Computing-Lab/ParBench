"""Required unit tests (plan §4.3) (rebuttal P0.2, 2026-07-24; Task 2, 2026-08-10).

Locks in: (a) the negative mismatch oracle on the hecbench-lud specs, and
(b) that the lud mismatch negative check declared in a pair contract's
negative_checks reaches cross-API verification in both directions.
(Originally these tests pinned union-oracle propagation of source+target
excludes; Task 2 replaced the union with the declared pair contract, so the
negative check now rides the contract, not the source spec.)

Context: the hecbench-lud-omp baseline reads partly uninitialized team-shared arrays
when thread_limit (an upper bound) under-provisions its teams - undefined behavior,
corrected 2026-07-27 from "data race" (5,024/65,536 cells mismatched, errors to 3.2e36;
see results/response_2026/lud/CORRECTION_2026-07-27.md and characterization.md)
yet PASSed its original exit_code+banner oracle. These tests make that false
pass, and its cross-API variants, impossible to reintroduce.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.models import RunResult, Status  # noqa: E402
from harness.verifier import verify_run  # noqa: E402
from scripts.evaluation.llm_evaluate import _build_cross_api_verify_spec  # noqa: E402
from scripts.evaluation.pair_contracts import PairContract  # noqa: E402

OMP_SPEC = json.loads((PROJECT_ROOT / "specs/hecbench-lud-omp.json").read_text())
CUDA_SPEC = json.loads((PROJECT_ROOT / "specs/hecbench-lud-cuda.json").read_text())


def _correctness_args(spec: dict) -> list[str]:
    return list(
        ((spec.get("run") or {}).get("input_configurations", {}).get("correctness") or {})
        .get("arguments", [])
    )


def lud_contract(source: dict, target: dict) -> PairContract:
    """A resolved pair contract carrying the lud mismatch negative check."""
    strategies = (target.get("verification") or {}).get("strategies", [])
    entry = {
        "source_spec": source["identity"]["unique_id"],
        "target_spec": target["identity"]["unique_id"],
        "shared_input": {
            "source_args": _correctness_args(source),
            "target_args": _correctness_args(target),
        },
        "translated_run": {"args": _correctness_args(target), "environment": {}},
        "target_oracle": {
            "strategies": [
                dict(s) for s in strategies if s["type"] != "stdout_exclude_pattern"
            ],
        },
        "negative_checks": [
            {"type": "stdout_exclude_pattern", "pattern": "mismatch at \\("},
        ],
        "comparator": {
            "type": "declared-strategy",
            "evidence": "tests/test_lud_negative_oracle.py",
        },
        "baseline_witness": {
            "status": "pass",
            "evidence": "tests/test_lud_negative_oracle.py",
        },
        "comparability": "eligible",
    }
    return PairContract(entry["source_spec"], entry["target_spec"], entry)

CLEAN_STDOUT = (
    "Generate input matrix internally, size=256\n"
    "Creating matrix internally size=256\n"
    "Before LUD\n"
    "WG size of kernel = 16 X 16\n"
    "Total kernel execution time : 0.000407 (s)\n"
    "Device offloading time (s): 0.092836\n"
    "After LUD\n"
    ">>>Verify<<<<\n"
)
MISMATCH_STDOUT = CLEAN_STDOUT + (
    "mismatch at (9, 16): (o)9.930244 (n)9.950047\n"
    "mismatch at (255, 16): (o)7.874148 (n)-7866187505467392.000000\n"
)


def run_result(stdout: str, exit_code: int = 0) -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


def strategy_types(spec: dict) -> list[str]:
    return [s["type"] for s in spec["verification"]["strategies"]]


def test_1_clean_lud_output_passes():
    result = verify_run(OMP_SPEC, run_result(CLEAN_STDOUT))
    assert result.status == Status.PASS, result.details


def test_2_mismatch_output_fails():
    result = verify_run(OMP_SPEC, run_result(MISMATCH_STDOUT))
    assert result.status == Status.FAIL, result.details


def test_3_cuda_to_omp_contract_mismatch_check_fails():
    verify_spec = _build_cross_api_verify_spec(
        OMP_SPEC, CUDA_SPEC, lud_contract(CUDA_SPEC, OMP_SPEC)
    )
    assert "stdout_exclude_pattern" in strategy_types(verify_spec)
    result = verify_run(verify_spec, run_result(MISMATCH_STDOUT))
    assert result.status == Status.FAIL, result.details


def test_4_omp_to_cuda_contract_mismatch_check_fails():
    # Strip the target spec's own exclude to prove the CONTRACT's negative
    # check alone rejects the mismatch (the spec no longer has to carry it).
    bare_cuda = copy.deepcopy(CUDA_SPEC)
    bare_cuda["verification"]["strategies"] = [
        s for s in bare_cuda["verification"]["strategies"]
        if s["type"] != "stdout_exclude_pattern"
    ]
    verify_spec = _build_cross_api_verify_spec(
        bare_cuda, OMP_SPEC, lud_contract(OMP_SPEC, bare_cuda)
    )
    assert "stdout_exclude_pattern" in strategy_types(verify_spec)
    result = verify_run(verify_spec, run_result(MISMATCH_STDOUT))
    assert result.status == Status.FAIL, result.details


def test_5_timing_output_plus_mismatch_still_fails():
    # Positive banner present AND mismatch present: conjunction must FAIL.
    assert "Total kernel execution time" in MISMATCH_STDOUT
    result = verify_run(OMP_SPEC, run_result(MISMATCH_STDOUT))
    assert result.status == Status.FAIL, result.details


def test_6_absent_positive_pattern_fails():
    no_banner = ">>>Verify<<<<\nAfter LUD\n"
    result = verify_run(OMP_SPEC, run_result(no_banner))
    assert result.status == Status.FAIL, result.details
