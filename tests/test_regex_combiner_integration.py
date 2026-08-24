"""Integration tests: cross-API verification is pair-contract-driven (Task 2, E-02).

This file previously locked in the union-oracle regex combiner: cross-API
verification accepted the SOURCE spec's success pattern as an alternative to
the TARGET's, which let a translation pass through a source-side success
string the target program never prints (July 2026 audit). Its assertions also
read Qwen result records purged in the sanctioned 2026-04-20 decommission.

Task 2 replaces that union with a declared pair-specific contract. These
tests use the same real spec pairs that exercised the old combiner
(rodinia-nn-opencl -> rodinia-nn-{cuda,omp}) and prove:

- the verify spec contains ONLY the contract's target oracle and negative
  checks — no source-pattern alternation;
- wrong-side (source) success text fails verification;
- a missing contract fails closed before any spec is built;
- the alternation helper `_wrap_pattern` is gone from the pipeline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.evaluation.llm_evaluate as llm_evaluate  # noqa: E402
from harness.models import RunResult, Status  # noqa: E402
from harness.verifier import verify_run  # noqa: E402
from scripts.evaluation.llm_evaluate import (  # noqa: E402
    _build_cross_api_run_spec,
    _build_cross_api_verify_spec,
)
from scripts.evaluation.pair_contracts import (  # noqa: E402
    PairContract,
    PairContractError,
)

SPECS_DIR = PROJECT_ROOT / "specs"


def _load_spec(spec_id: str) -> dict:
    return json.loads((SPECS_DIR / f"{spec_id}.json").read_text())


def _correctness_args(spec: dict) -> list[str]:
    return list(
        ((spec.get("run") or {}).get("input_configurations", {}).get("correctness") or {})
        .get("arguments", [])
    )


def _target_native_oracle(spec: dict) -> list[dict]:
    return [dict(s) for s in (spec.get("verification") or {}).get("strategies", [])]


def make_contract(source: dict, target: dict) -> PairContract:
    """A resolved contract for (source -> target) declaring the TARGET-native
    run args and oracle — the shape Task 7's production registry will carry."""
    entry = {
        "source_spec": source["identity"]["unique_id"],
        "target_spec": target["identity"]["unique_id"],
        "shared_input": {
            "source_args": _correctness_args(source),
            "target_args": _correctness_args(target),
        },
        "translated_run": {"args": _correctness_args(target), "environment": {}},
        "target_oracle": {"strategies": _target_native_oracle(target)},
        "negative_checks": [
            s for s in _target_native_oracle(target)
            if s.get("type") == "stdout_exclude_pattern"
        ],
        "comparator": {
            "type": "declared-strategy",
            "evidence": "tests/test_regex_combiner_integration.py",
        },
        "baseline_witness": {
            "status": "pass",
            "evidence": "tests/test_regex_combiner_integration.py",
        },
        "comparability": "eligible",
    }
    # target_oracle keeps only positive checks; negatives ride negative_checks.
    entry["target_oracle"]["strategies"] = [
        s for s in entry["target_oracle"]["strategies"]
        if s.get("type") != "stdout_exclude_pattern"
    ]
    return PairContract(entry["source_spec"], entry["target_spec"], entry)


def run_result(stdout: str, exit_code: int = 0) -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


# The real pairs whose union oracle produced the audited wrong-side passes.
PAIRS = [
    ("rodinia-nn-opencl", "rodinia-nn-cuda"),
    ("rodinia-nn-opencl", "rodinia-nn-omp"),
]


class TestUnionOracleIsGone:
    @pytest.mark.parametrize("source_id,target_id", PAIRS)
    def test_verify_strategies_are_contract_only(self, source_id, target_id):
        source, target = _load_spec(source_id), _load_spec(target_id)
        contract = make_contract(source, target)
        verify_spec = _build_cross_api_verify_spec(target, source, contract)
        assert verify_spec["verification"]["strategies"] == (
            contract.target_oracle["strategies"] + contract.negative_checks
        )

    @pytest.mark.parametrize("source_id,target_id", PAIRS)
    def test_no_source_pattern_and_no_alternation(self, source_id, target_id):
        source, target = _load_spec(source_id), _load_spec(target_id)
        verify_spec = _build_cross_api_verify_spec(
            target, source, make_contract(source, target)
        )
        source_patterns = [
            s["pattern"]
            for s in (source.get("verification") or {}).get("strategies", [])
            if s.get("type") == "stdout_pattern" and s.get("pattern")
        ]
        assert source_patterns, f"{source_id} should declare a stdout_pattern"
        for strategy in verify_spec["verification"]["strategies"]:
            pattern = strategy.get("pattern", "")
            for src_pattern in source_patterns:
                assert src_pattern not in pattern, (
                    f"source pattern {src_pattern!r} leaked into {pattern!r}"
                )
            assert "(?:" not in pattern and "(?i:" not in pattern, (
                f"alternation-wrapped pattern survived: {pattern!r}"
            )

    def test_wrap_pattern_helper_removed(self):
        assert not hasattr(llm_evaluate, "_wrap_pattern")


class TestWrongSideSuccessTextFailsClosed:
    def test_source_side_success_text_fails_nn_cuda(self):
        # nn-opencl's oracle is "(?i)(pass|correct|match|verified)"; nn-cuda
        # prints "Distance=". Under the union oracle, this stdout PASSed —
        # exactly the audited wrong-side flip. It must now FAIL.
        source, target = _load_spec("rodinia-nn-opencl"), _load_spec("rodinia-nn-cuda")
        verify_spec = _build_cross_api_verify_spec(
            target, source, make_contract(source, target)
        )
        result = verify_run(verify_spec, run_result("Verification PASSED. correct.\n"))
        assert result.status == Status.FAIL, result.details

    def test_target_native_output_passes_nn_cuda(self):
        source, target = _load_spec("rodinia-nn-opencl"), _load_spec("rodinia-nn-cuda")
        verify_spec = _build_cross_api_verify_spec(
            target, source, make_contract(source, target)
        )
        result = verify_run(verify_spec, run_result("x --> Distance=0.123456\n"))
        assert result.status == Status.PASS, result.details

    def test_target_pattern_stays_case_sensitive(self):
        # Under the union oracle the source's (?i) flag had to be scoped to
        # avoid leaking; now the source pattern is absent entirely.
        source, target = _load_spec("rodinia-nn-opencl"), _load_spec("rodinia-nn-cuda")
        verify_spec = _build_cross_api_verify_spec(
            target, source, make_contract(source, target)
        )
        patterns = [
            s["pattern"] for s in verify_spec["verification"]["strategies"]
            if s.get("type") == "stdout_pattern"
        ]
        assert patterns
        for pattern in patterns:
            assert not re.search(pattern, "distance=0.5"), (
                "case-insensitivity leaked into the target oracle"
            )


class TestFailClosedWithoutContract:
    @pytest.mark.parametrize("source_id,target_id", PAIRS)
    def test_verify_builder_requires_contract(self, source_id, target_id):
        with pytest.raises(PairContractError):
            _build_cross_api_verify_spec(
                _load_spec(target_id), _load_spec(source_id), None
            )

    @pytest.mark.parametrize("source_id,target_id", PAIRS)
    def test_run_builder_requires_contract(self, source_id, target_id):
        with pytest.raises(PairContractError):
            _build_cross_api_run_spec(
                _load_spec(target_id), _load_spec(source_id), None
            )


class TestRunArgsComeFromContract:
    @pytest.mark.parametrize("source_id,target_id", PAIRS)
    def test_translated_run_args_used(self, source_id, target_id):
        source, target = _load_spec(source_id), _load_spec(target_id)
        contract = make_contract(source, target)
        run_spec = _build_cross_api_run_spec(target, source, contract)
        args = run_spec["run"]["input_configurations"]["correctness"]["arguments"]
        assert args == contract.translated_run["args"]

    def test_mismatched_declared_input_fails_closed(self):
        source, target = _load_spec("rodinia-nn-opencl"), _load_spec("rodinia-nn-cuda")
        contract = make_contract(source, target)
        contract.data["shared_input"]["source_args"] = ["not", "the", "real", "args"]
        with pytest.raises(PairContractError, match="source_args"):
            _build_cross_api_run_spec(target, source, contract)
