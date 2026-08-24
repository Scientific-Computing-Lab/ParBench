"""Unit tests: the regex pattern combiner is gone (Task 2, gate E-02).

This file previously covered _wrap_pattern() and the union-oracle pattern
alternation in _build_cross_api_verify_spec(). Task 2 removed both: cross-API
verification now uses ONLY the declared pair contract's target oracle and
negative checks, so there is no pattern combining left to scope inline flags
for. These unit tests pin that removal on synthetic specs; the real-spec
integration coverage lives in tests/test_regex_combiner_integration.py and
the loader/consumer coverage in tests/test_pair_contracts.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scripts.evaluation.llm_evaluate as llm_evaluate
from scripts.evaluation.llm_evaluate import _build_cross_api_verify_spec
from scripts.evaluation.pair_contracts import PairContract, PairContractError


def make_specs() -> tuple[dict, dict]:
    source_spec = {
        "identity": {"unique_id": "demo-k-cuda", "parallel_api": "cuda"},
        "run": {"input_configurations": {"correctness": {"arguments": []}}},
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "(?i)(pass|correct)"}
            ]
        },
    }
    target_spec = {
        "identity": {"unique_id": "demo-k-omp", "parallel_api": "omp"},
        "run": {"input_configurations": {"correctness": {"arguments": []}}},
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "Distance="},
                {"type": "exit_code", "expected": 0},
            ]
        },
        "files": {"translation_targets": ["main.cpp"]},
    }
    return source_spec, target_spec


def make_contract() -> PairContract:
    entry = {
        "source_spec": "demo-k-cuda",
        "target_spec": "demo-k-omp",
        "shared_input": {"source_args": [], "target_args": []},
        "translated_run": {"args": [], "environment": {}},
        "target_oracle": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "Distance="},
                {"type": "exit_code", "expected": 0},
            ]
        },
        "negative_checks": [],
        "comparator": {"type": "declared-strategy", "evidence": "tests/test_regex_combiner.py"},
        "baseline_witness": {"status": "pass", "evidence": "tests/test_regex_combiner.py"},
        "comparability": "eligible",
    }
    return PairContract(entry["source_spec"], entry["target_spec"], entry)


def test_wrap_pattern_removed():
    """The alternation helper has no reason to exist without the union oracle."""
    assert not hasattr(llm_evaluate, "_wrap_pattern")


def test_no_pattern_combining():
    """The verify spec carries the contract's patterns verbatim — no (?:...)
    wrapping, no '|' alternation with the source pattern."""
    source_spec, target_spec = make_specs()
    result = _build_cross_api_verify_spec(target_spec, source_spec, make_contract())
    patterns = [
        s["pattern"] for s in result["verification"]["strategies"]
        if s["type"] == "stdout_pattern"
    ]
    assert patterns == ["Distance="]


def test_source_pattern_absent():
    """The source spec's success pattern must not reach the verify spec."""
    source_spec, target_spec = make_specs()
    result = _build_cross_api_verify_spec(target_spec, source_spec, make_contract())
    for strategy in result["verification"]["strategies"]:
        assert "(?i)(pass|correct)" not in strategy.get("pattern", "")


def test_kernel_only_target_also_requires_contract():
    """The pre-Task-2 kernel-only bypass is gone: every cross-API pair —
    including OpenCL kernel-only targets — needs a resolved contract."""
    source_spec, target_spec = make_specs()
    target_spec["files"]["translation_targets"] = ["kernel.cl"]
    with pytest.raises(PairContractError):
        _build_cross_api_verify_spec(target_spec, source_spec, None)
