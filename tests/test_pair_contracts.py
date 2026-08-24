"""Task 2 (gate E-02): pair-contract schema, loader, and fail-closed consumers.

Covers the acceptance criteria of Task 2 in
docs/superpowers/plans/2026-08-10-parbench-final-engineering.md:
wrong-side success text, mismatched inputs, unresolved or absent witnesses,
duplicate pair keys, and missing registry entries all fail closed, and the
candidate-pair generator deterministically enumerates directed-pair keys
without certifying production eligibility.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.models import RunResult, Status  # noqa: E402
from harness.verifier import verify_run  # noqa: E402
from scripts.evaluation.llm_evaluate import (  # noqa: E402
    _build_cross_api_run_spec,
    _build_cross_api_verify_spec,
)
from scripts.evaluation.pair_contracts import (  # noqa: E402
    CONTRACT_VERSION,
    PairContract,
    PairContractError,
    load_pair_contracts,
    required_pair_keys,
    validate_registry_completeness,
)

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "pair_contracts"
COMPLETE = FIXTURES / "complete.json"
UNRESOLVED = FIXTURES / "unresolved.json"


# --------------------------------------------------------------------------- #
# Synthetic specs and contract entries                                        #
# --------------------------------------------------------------------------- #

def make_source_spec() -> dict:
    return {
        "identity": {
            "kernel_name": "alpha",
            "parallel_api": "cuda",
            "unique_id": "demo-alpha-cuda",
            "source_suite": "demo",
        },
        "run": {
            "input_configurations": {
                "correctness": {"arguments": ["16", "alpha.dat"]},
            },
        },
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "CUDA OK"},
                {"type": "exit_code", "expected": 0},
            ],
        },
    }


def make_target_spec() -> dict:
    return {
        "identity": {
            "kernel_name": "alpha",
            "parallel_api": "omp",
            "unique_id": "demo-alpha-omp",
            "source_suite": "demo",
        },
        "run": {
            "input_configurations": {
                "correctness": {"arguments": ["16", "alpha.dat", "4"]},
            },
        },
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "OMP OK"},
                {"type": "exit_code", "expected": 0},
            ],
        },
    }


def make_entry(**overrides) -> dict:
    """A fully resolved contract entry for demo-alpha-cuda -> demo-alpha-omp."""
    entry = {
        "source_spec": "demo-alpha-cuda",
        "target_spec": "demo-alpha-omp",
        "shared_input": {
            "source_args": ["16", "alpha.dat"],
            "target_args": ["16", "alpha.dat", "4"],
        },
        "translated_run": {
            "args": ["16", "alpha.dat", "4"],
            "environment": {"OMP_NUM_THREADS": "4"},
        },
        "target_oracle": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": "OMP OK"},
                {"type": "exit_code", "expected": 0},
            ],
        },
        "negative_checks": [
            {"type": "stdout_exclude_pattern", "pattern": "(?i)mismatch at"},
        ],
        "comparator": {
            "type": "declared-strategy",
            "evidence": "tests/fixtures/pair_contracts/complete.json",
        },
        "baseline_witness": {
            "status": "pass",
            "evidence": "tests/fixtures/pair_contracts/complete.json",
        },
        "comparability": "eligible",
    }
    entry.update(overrides)
    return entry


def make_contract(**overrides) -> PairContract:
    entry = make_entry(**overrides)
    return PairContract(entry["source_spec"], entry["target_spec"], entry)


def write_registry(tmp_path: Path, pairs: list[dict], version: str = CONTRACT_VERSION) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"contract_version": version, "pairs": pairs}, indent=2))
    return path


def run_result(stdout: str, exit_code: int = 0) -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


# --------------------------------------------------------------------------- #
# Loader                                                                      #
# --------------------------------------------------------------------------- #

class TestLoader:
    def test_loads_complete_fixture(self):
        registry = load_pair_contracts(COMPLETE)
        assert registry.contract_version == CONTRACT_VERSION
        assert len(registry) == 2
        contract = registry.get("demo-alpha-cuda", "demo-alpha-omp")
        assert contract.target_oracle["strategies"][0]["pattern"] == "OMP OK"

    def test_missing_file_fails_closed(self, tmp_path):
        with pytest.raises(PairContractError, match="not found"):
            load_pair_contracts(tmp_path / "nope.json")

    def test_invalid_json_fails_closed(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(PairContractError, match="not valid JSON"):
            load_pair_contracts(path)

    def test_schema_violation_fails_closed(self, tmp_path):
        entry = make_entry()
        del entry["target_spec"]
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_unknown_entry_key_fails_closed(self, tmp_path):
        path = write_registry(tmp_path, [make_entry(surprise=1)])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_missing_contract_version_fails_closed(self, tmp_path):
        path = tmp_path / "noversion.json"
        path.write_text(json.dumps({"pairs": [make_entry()]}))
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_duplicate_pair_key_fails_closed(self, tmp_path):
        path = write_registry(tmp_path, [make_entry(), make_entry()])
        with pytest.raises(PairContractError, match="duplicate"):
            load_pair_contracts(path)

    def test_absent_witness_fails_closed_at_load(self, tmp_path):
        entry = make_entry()
        del entry["baseline_witness"]
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)


# --------------------------------------------------------------------------- #
# Per-pair resolution (fail closed)                                           #
# --------------------------------------------------------------------------- #

class TestResolution:
    def _registry(self, tmp_path, entry):
        return load_pair_contracts(write_registry(tmp_path, [entry]))

    @pytest.mark.parametrize(
        "overrides,reason",
        [
            ({"comparability": "unresolved"}, "comparability"),
            ({"comparability": "incomparable"}, "comparability"),
            ({"baseline_witness": {"status": "unresolved"}}, "witness"),
            ({"baseline_witness": {"status": "fail", "evidence": "x.json"}}, "witness"),
            ({"baseline_witness": {"status": "pass"}}, "witness"),  # no evidence
        ],
        ids=[
            "comparability-unresolved",
            "comparability-incomparable",
            "witness-unresolved",
            "witness-failed",
            "witness-no-evidence",
        ],
    )
    def test_unresolved_entry_fails_closed(self, tmp_path, overrides, reason):
        registry = self._registry(tmp_path, make_entry(**overrides))
        with pytest.raises(PairContractError, match=reason):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")

    @pytest.mark.parametrize(
        "missing_field,reason",
        [
            ("comparator", "comparator"),
            ("shared_input", "shared_input"),
            ("translated_run", "translated_run"),
            ("target_oracle", "target_oracle"),
        ],
    )
    def test_absent_section_fails_closed(self, tmp_path, missing_field, reason):
        entry = make_entry()
        del entry[missing_field]
        registry = self._registry(tmp_path, entry)
        with pytest.raises(PairContractError, match=reason):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")

    def test_empty_oracle_fails_closed(self, tmp_path):
        registry = self._registry(
            tmp_path, make_entry(target_oracle={"strategies": []})
        )
        with pytest.raises(PairContractError, match="target_oracle"):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")

    def test_missing_pair_fails_closed(self, tmp_path):
        registry = self._registry(tmp_path, make_entry())
        with pytest.raises(PairContractError, match="demo-alpha-opencl"):
            registry.resolve("demo-alpha-opencl", "demo-alpha-omp")

    def test_resolved_entry_resolves(self, tmp_path):
        registry = self._registry(tmp_path, make_entry())
        contract = registry.resolve("demo-alpha-cuda", "demo-alpha-omp")
        assert contract.comparability == "eligible"


# --------------------------------------------------------------------------- #
# Codex gate pass-1 regressions: schema strictness, evidence paths,           #
# pre-model spec cross-validation                                             #
# --------------------------------------------------------------------------- #

class TestSchemaStrictness:
    def test_wrong_contract_version_fails_closed(self, tmp_path):
        path = write_registry(tmp_path, [make_entry()], version="wrong-v9")
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_unknown_oracle_strategy_type_fails_closed(self, tmp_path):
        entry = make_entry(target_oracle={"strategies": [{"type": "not-a-verifier"}]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_exit_code_without_expected_fails_closed(self, tmp_path):
        entry = make_entry(target_oracle={"strategies": [{"type": "exit_code"}]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_stdout_pattern_without_pattern_fails_closed(self, tmp_path):
        entry = make_entry(target_oracle={"strategies": [{"type": "stdout_pattern"}]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_negative_check_must_be_exclude_pattern(self, tmp_path):
        entry = make_entry(negative_checks=[{"type": "stdout_pattern", "pattern": "x"}])
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_exclude_pattern_not_allowed_in_target_oracle(self, tmp_path):
        entry = make_entry(target_oracle={"strategies": [
            {"type": "stdout_exclude_pattern", "pattern": "mismatch"},
        ]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_non_numeric_tolerance_fails_closed(self, tmp_path):
        # Codex gate pass 2, finding 1: tolerance "inf" (a string) previously
        # passed schema validation and could turn any value into a PASS.
        entry = make_entry(target_oracle={"strategies": [{
            "type": "numeric_comparison", "extract_regex": r"(\d+)",
            "expected": 1, "tolerance": "inf",
        }]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    @pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN", "1e999", "-1e999"])
    def test_nonfinite_json_token_fails_closed(self, tmp_path, token):
        # Codex gate pass 3: json.loads accepts these non-standard tokens by
        # default, which would bypass the schema's numeric tolerance bound.
        entry = make_entry(target_oracle={"strategies": [{
            "type": "numeric_comparison", "extract_regex": r"(\d+)",
            "expected": 1, "tolerance": 0,
        }]})
        path = tmp_path / "nonfinite.json"
        text = json.dumps(
            {"contract_version": CONTRACT_VERSION, "pairs": [entry]}
        ).replace('"tolerance": 0', f'"tolerance": {token}')
        assert f"tolerance\": {token}" in text
        path.write_text(text)
        with pytest.raises(PairContractError, match="not valid JSON"):
            load_pair_contracts(path)

    @pytest.mark.parametrize(
        "field,token",
        [
            ("expected", "1e-999"),
            ("expected", "-1e-999"),
            ("tolerance", "1e-999"),
            ("tolerance", "-1e-999"),
            ("expected", "1.7976931348623158e308"),
            ("expected", "-1.7976931348623158e308"),
            ("tolerance", "1.7976931348623158e308"),
        ],
        ids=["expected-underflow", "expected-negative-underflow",
             "tolerance-underflow", "tolerance-negative-underflow",
             "expected-rounds-to-dblmax", "expected-negative-rounds-to-dblmax",
             "tolerance-rounds-to-dblmax"],
    )
    def test_raw_token_representability_fails_closed(
            self, tmp_path, field, token):
        # Task 2 residual, ruled Task 7 work: 1e-999 underflows to 0.0 and
        # 1.7976931348623158e308 rounds DOWN to DBL_MAX before the schema
        # bound applies. The loader now inspects the RAW token via
        # decimal.Decimal and refuses both escapes for both numeric fields.
        strategy = {
            "type": "numeric_comparison", "extract_regex": r"(\d+)",
            "expected": 1, "tolerance": 0,
        }
        entry = make_entry(target_oracle={"strategies": [strategy]})
        path = tmp_path / "representability.json"
        text = json.dumps(
            {"contract_version": CONTRACT_VERSION, "pairs": [entry]}
        ).replace(f'"{field}": {json.dumps(strategy[field])}',
                  f'"{field}": {token}')
        assert f'"{field}": {token}' in text
        path.write_text(text)
        with pytest.raises(PairContractError, match="not valid JSON"):
            load_pair_contracts(path)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("expected", 10**309),
            ("expected", -(10**309)),
            ("tolerance", 10**309),
        ],
        ids=["expected-overflow", "expected-negative-overflow", "tolerance-overflow"],
    )
    def test_oversized_integer_fails_closed(self, tmp_path, field, value):
        # Codex gate pass 5: integer tokens bypass parse_float, and a value
        # beyond the finite binary64 range overflows the verifier's float
        # coercion after model invocation. The schema bounds must reject it.
        strategy = {
            "type": "numeric_comparison", "extract_regex": r"(\d+)",
            "expected": 1, "tolerance": 0,
        }
        strategy[field] = value
        entry = make_entry(target_oracle={"strategies": [strategy]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_duplicate_json_member_fails_closed(self, tmp_path):
        # Codex gate pass 6: json.loads keeps the LAST duplicate member, so
        # "tolerance": 0, "tolerance": 1e308 would validate as one value and
        # verify as another. Raw-JSON regression — json.dumps cannot emit
        # duplicates, so the text is spliced by hand.
        entry = make_entry(target_oracle={"strategies": [{
            "type": "numeric_comparison", "extract_regex": r"(\d+)",
            "expected": 1, "tolerance": 0,
        }]})
        text = json.dumps({"contract_version": CONTRACT_VERSION, "pairs": [entry]})
        text = text.replace(
            '"tolerance": 0',
            '"tolerance": 0, "tolerance": 1.7976931348623157e+308',
        )
        assert text.count('"tolerance"') == 2
        path = tmp_path / "duplicate_member.json"
        path.write_text(text)
        with pytest.raises(PairContractError, match="duplicate JSON member"):
            load_pair_contracts(path)

    def test_negative_tolerance_fails_closed(self, tmp_path):
        entry = make_entry(target_oracle={"strategies": [{
            "type": "numeric_comparison", "extract_regex": r"(\d+)",
            "expected": 1, "tolerance": -1,
        }]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_undeclared_strategy_property_fails_closed(self, tmp_path):
        entry = make_entry(target_oracle={"strategies": [{
            "type": "stdout_pattern", "pattern": "OK", "surprise": 1,
        }]})
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)

    def test_unknown_comparator_type_fails_closed(self, tmp_path):
        # Codex gate pass 2, finding 2: an unimplemented comparator type must
        # not validate, let alone certify eligibility.
        entry = make_entry(comparator={
            "type": "not-implemented",
            "evidence": "tests/fixtures/pair_contracts/complete.json",
        })
        path = write_registry(tmp_path, [entry])
        with pytest.raises(PairContractError, match="schema"):
            load_pair_contracts(path)


class TestEvidencePaths:
    @pytest.mark.parametrize(
        "evidence,reason",
        [
            ("tests/no/such/evidence.json", "regular file"),
            ("/etc/passwd", "absolute"),
            ("../../../etc/passwd", "escapes"),
        ],
        ids=["nonexistent", "absolute", "escaping"],
    )
    def test_bad_witness_evidence_fails_closed(self, tmp_path, evidence, reason):
        entry = make_entry(baseline_witness={"status": "pass", "evidence": evidence})
        registry = load_pair_contracts(write_registry(tmp_path, [entry]))
        with pytest.raises(PairContractError, match=reason):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")

    def test_bad_comparator_evidence_fails_closed(self, tmp_path):
        entry = make_entry(comparator={
            "type": "declared-strategy", "evidence": "tests/no/such/evidence.json",
        })
        registry = load_pair_contracts(write_registry(tmp_path, [entry]))
        with pytest.raises(PairContractError, match="regular file"):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")

    def test_directory_evidence_fails_closed(self, tmp_path):
        # Codex gate pass 2, finding 2: a directory is not reviewable evidence.
        entry = make_entry(baseline_witness={"status": "pass", "evidence": "tests"})
        registry = load_pair_contracts(write_registry(tmp_path, [entry]))
        with pytest.raises(PairContractError, match="regular file"):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")

    def test_unreadable_evidence_fails_closed(self, tmp_path):
        import os

        import scripts.evaluation.pair_contracts as pc
        unreadable = PROJECT_ROOT / "tests" / "fixtures" / "pair_contracts" / ".unreadable_evidence"
        unreadable.write_text("x")
        try:
            unreadable.chmod(0o000)
            if os.access(unreadable, os.R_OK):
                pytest.skip("host cannot make a file unreadable (e.g. running as root)")
            entry = make_entry(baseline_witness={
                "status": "pass",
                "evidence": "tests/fixtures/pair_contracts/.unreadable_evidence",
            })
            registry = load_pair_contracts(write_registry(tmp_path, [entry]))
            with pytest.raises(pc.PairContractError, match="not readable"):
                registry.resolve("demo-alpha-cuda", "demo-alpha-omp")
        finally:
            unreadable.chmod(0o644)
            unreadable.unlink()


class TestSpecCrossValidation:
    def test_matching_contract_validates(self):
        from scripts.evaluation.pair_contracts import validate_contract_against_specs
        validate_contract_against_specs(
            make_contract(), make_source_spec(), make_target_spec()
        )

    def test_mismatched_args_fail_before_model(self):
        from scripts.evaluation.pair_contracts import validate_contract_against_specs
        contract = make_contract(shared_input={
            "source_args": ["999"], "target_args": ["16", "alpha.dat", "4"],
        })
        with pytest.raises(PairContractError, match="source_args"):
            validate_contract_against_specs(
                contract, make_source_spec(), make_target_spec()
            )

    def test_batch_gate_rejects_mismatched_contract(self, tmp_path):
        """The batch-level gate must catch a resolved-but-mismatched contract
        before run_batch() — Codex gate pass 1, finding 1."""
        from scripts.evaluation.run_eval_batch import validate_batch_pair_contracts
        src = tmp_path / "demo-alpha-cuda.json"
        tgt = tmp_path / "demo-alpha-omp.json"
        src.write_text(json.dumps(make_source_spec()))
        tgt.write_text(json.dumps(make_target_spec()))
        tasks = [{
            "src_id": "demo-alpha-cuda", "tgt_id": "demo-alpha-omp",
            "src_spec": src, "tgt_spec": tgt,
        }]
        good = load_pair_contracts(COMPLETE)
        assert validate_batch_pair_contracts(good, tasks) is None
        bad_entry = make_entry(shared_input={
            "source_args": ["999"], "target_args": ["16", "alpha.dat", "4"],
        })
        bad = load_pair_contracts(write_registry(tmp_path, [bad_entry]))
        with pytest.raises(PairContractError, match="source_args"):
            validate_batch_pair_contracts(bad, tasks)


# --------------------------------------------------------------------------- #
# Registry completeness against a task list                                   #
# --------------------------------------------------------------------------- #

class TestCompleteness:
    def test_required_pair_keys_source_target_style(self):
        tasks = [
            {"source_spec": "a-cuda", "target_spec": "a-omp"},
            {"source_spec": "a-omp", "target_spec": "a-cuda"},
            {"source_spec": "a-cuda", "target_spec": "a-omp"},  # dedupe
        ]
        assert required_pair_keys(tasks) == {("a-cuda", "a-omp"), ("a-omp", "a-cuda")}

    def test_required_pair_keys_batch_task_style(self):
        tasks = [{"src_id": "a-cuda", "tgt_id": "a-omp", "model": "m"}]
        assert required_pair_keys(tasks) == {("a-cuda", "a-omp")}

    def test_required_pair_keys_skips_same_spec(self):
        # Same-API (augmentation-only) tasks translate a spec onto itself and
        # do not require a pair contract.
        assert required_pair_keys([{"source_spec": "a-omp", "target_spec": "a-omp"}]) == set()

    def test_required_pair_keys_rejects_unidentifiable_task(self):
        with pytest.raises(PairContractError, match="source/target"):
            required_pair_keys([{"model": "m"}])

    def test_completeness_passes_when_covered(self):
        registry = load_pair_contracts(COMPLETE)
        tasks = [
            {"source_spec": "demo-alpha-cuda", "target_spec": "demo-alpha-omp"},
            {"source_spec": "demo-alpha-omp", "target_spec": "demo-alpha-cuda"},
        ]
        assert validate_registry_completeness(registry, tasks) is None

    def test_missing_registry_entry_fails_closed(self):
        registry = load_pair_contracts(COMPLETE)
        tasks = [{"source_spec": "demo-alpha-cuda", "target_spec": "demo-alpha-opencl"}]
        with pytest.raises(PairContractError, match="demo-alpha-opencl"):
            validate_registry_completeness(registry, tasks)

    def test_unresolved_registry_entry_fails_closed(self):
        registry = load_pair_contracts(UNRESOLVED)
        tasks = [{"source_spec": "demo-alpha-cuda", "target_spec": "demo-alpha-omp"}]
        with pytest.raises(PairContractError):
            validate_registry_completeness(registry, tasks)


# --------------------------------------------------------------------------- #
# Fail-closed consumers: the cross-API run/verify spec builders               #
# --------------------------------------------------------------------------- #

class TestRunSpecBuilder:
    def test_args_come_from_contract_not_source(self):
        run_spec = _build_cross_api_run_spec(
            make_target_spec(), make_source_spec(), make_contract()
        )
        args = run_spec["run"]["input_configurations"]["correctness"]["arguments"]
        assert args == ["16", "alpha.dat", "4"]

    def test_environment_merged_into_run_block(self):
        target = make_target_spec()
        target["run"]["environment_variables"] = {"KEEP": "1"}
        run_spec = _build_cross_api_run_spec(target, make_source_spec(), make_contract())
        assert run_spec["run"]["environment_variables"] == {
            "KEEP": "1",
            "OMP_NUM_THREADS": "4",
        }

    def test_missing_contract_fails_closed(self):
        with pytest.raises(PairContractError, match="pair contract"):
            _build_cross_api_run_spec(make_target_spec(), make_source_spec(), None)

    def test_contract_for_different_pair_fails_closed(self):
        contract = make_contract(target_spec="demo-alpha-opencl")
        with pytest.raises(PairContractError, match="demo-alpha-opencl"):
            _build_cross_api_run_spec(make_target_spec(), make_source_spec(), contract)

    def test_mismatched_source_input_fails_closed(self):
        contract = make_contract(
            shared_input={
                "source_args": ["999"],
                "target_args": ["16", "alpha.dat", "4"],
            }
        )
        with pytest.raises(PairContractError, match="source_args"):
            _build_cross_api_run_spec(make_target_spec(), make_source_spec(), contract)

    def test_mismatched_target_input_fails_closed(self):
        contract = make_contract(
            shared_input={
                "source_args": ["16", "alpha.dat"],
                "target_args": ["999"],
            }
        )
        with pytest.raises(PairContractError, match="target_args"):
            _build_cross_api_run_spec(make_target_spec(), make_source_spec(), contract)

    def test_unresolved_contract_fails_closed(self):
        contract = make_contract(baseline_witness={"status": "unresolved"})
        with pytest.raises(PairContractError, match="witness"):
            _build_cross_api_run_spec(make_target_spec(), make_source_spec(), contract)


class TestVerifySpecBuilder:
    def test_strategies_are_exactly_oracle_plus_negative_checks(self):
        verify_spec = _build_cross_api_verify_spec(
            make_target_spec(), make_source_spec(), make_contract()
        )
        contract = make_contract()
        assert verify_spec["verification"]["strategies"] == (
            contract.target_oracle["strategies"] + contract.negative_checks
        )

    def test_source_success_pattern_never_enters_verify_spec(self):
        verify_spec = _build_cross_api_verify_spec(
            make_target_spec(), make_source_spec(), make_contract()
        )
        blob = json.dumps(verify_spec["verification"]["strategies"])
        assert "CUDA OK" not in blob

    def test_wrong_side_success_text_fails_closed(self):
        # The July audit failure mode: SOURCE-side success text alone must not
        # pass TARGET-side verification.
        verify_spec = _build_cross_api_verify_spec(
            make_target_spec(), make_source_spec(), make_contract()
        )
        assert verify_run(verify_spec, run_result("CUDA OK\n")).status == Status.FAIL

    def test_target_success_text_passes(self):
        verify_spec = _build_cross_api_verify_spec(
            make_target_spec(), make_source_spec(), make_contract()
        )
        assert verify_run(verify_spec, run_result("OMP OK\n")).status == Status.PASS

    def test_negative_check_is_conjunctive(self):
        verify_spec = _build_cross_api_verify_spec(
            make_target_spec(), make_source_spec(), make_contract()
        )
        stdout = "OMP OK\nmismatch at (1, 2): (o)1.0 (n)9.0\n"
        assert verify_run(verify_spec, run_result(stdout)).status == Status.FAIL

    def test_missing_contract_fails_closed(self):
        with pytest.raises(PairContractError, match="pair contract"):
            _build_cross_api_verify_spec(make_target_spec(), make_source_spec(), None)

    def test_unresolved_contract_fails_closed(self):
        contract = make_contract(comparability="unresolved")
        with pytest.raises(PairContractError, match="comparability"):
            _build_cross_api_verify_spec(make_target_spec(), make_source_spec(), contract)


# --------------------------------------------------------------------------- #
# Registry CLI (--check-completeness)                                         #
# --------------------------------------------------------------------------- #

class TestRegistryCli:
    def _run(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "scripts.evaluation.pair_contracts", *argv],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    def test_complete_registry_exits_zero(self):
        proc = self._run("--registry", str(COMPLETE), "--check-completeness")
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_unresolved_registry_exits_nonzero(self):
        proc = self._run("--registry", str(UNRESOLVED), "--check-completeness")
        assert proc.returncode != 0
        assert "demo-alpha-cuda" in proc.stdout + proc.stderr

    def test_missing_registry_exits_nonzero(self):
        proc = self._run("--registry", "no/such/registry.json", "--check-completeness")
        assert proc.returncode != 0


# --------------------------------------------------------------------------- #
# Candidate-pair enumeration                                                  #
# --------------------------------------------------------------------------- #

class TestCandidateEnumeration:
    @pytest.fixture()
    def spec_root(self, tmp_path):
        root = tmp_path / "specs"
        root.mkdir()
        for uid, kernel, api in [
            ("demo-alpha-cuda", "alpha", "cuda"),
            ("demo-alpha-omp", "alpha", "omp"),
            ("demo-beta-cuda", "beta", "cuda"),
        ]:
            (root / f"{uid}.json").write_text(json.dumps({
                "identity": {
                    "kernel_name": kernel,
                    "parallel_api": api,
                    "unique_id": uid,
                    "source_suite": "demo",
                },
            }))
        return root

    def _enumerate(self, spec_root: Path, output: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, "scripts/spec_tools/build_pair_contracts.py",
                "--enumerate-candidates",
                "--spec-root", str(spec_root),
                "--output", str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    def test_enumerates_directed_pairs_only_within_kernel(self, spec_root, tmp_path):
        out = tmp_path / "candidates.json"
        proc = self._enumerate(spec_root, out)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        data = json.loads(out.read_text())
        keys = [(p["source_spec"], p["target_spec"]) for p in data["pairs"]]
        assert keys == [
            ("demo-alpha-cuda", "demo-alpha-omp"),
            ("demo-alpha-omp", "demo-alpha-cuda"),
        ]

    def test_enumeration_is_deterministic(self, spec_root, tmp_path):
        out1 = tmp_path / "c1.json"
        out2 = tmp_path / "c2.json"
        assert self._enumerate(spec_root, out1).returncode == 0
        assert self._enumerate(spec_root, out2).returncode == 0
        assert out1.read_bytes() == out2.read_bytes()

    def test_candidates_load_but_do_not_certify_eligibility(self, spec_root, tmp_path):
        out = tmp_path / "candidates.json"
        assert self._enumerate(spec_root, out).returncode == 0
        registry = load_pair_contracts(out)
        assert len(registry) == 2
        with pytest.raises(PairContractError):
            registry.resolve("demo-alpha-cuda", "demo-alpha-omp")
