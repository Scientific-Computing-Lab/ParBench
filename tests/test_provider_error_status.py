"""Regression tests for the PROVIDER_ERROR overall_status (OD-4, 2026-08-20).

Before OD-4, a provider/API exception on a *later* attempt of
``evaluate_translation`` left ``final_status`` at whatever the previous attempt
had set (e.g. ``BUILD_FAIL``), so a transient outage masqueraded as a
deterministic failure that ``run_eval_batch.py --resume`` would never retry.

These tests pin the fixed behaviour:

1. A provider exception on attempt 2 (after attempt 1 reached BUILD_FAIL) makes
   the written result's ``overall_status`` equal ``"PROVIDER_ERROR"``, not the
   attempt-1 status.
2. The ``--resume`` retry predicate in ``run_eval_batch.py`` treats
   ``PROVIDER_ERROR`` as retryable.

Hermetic: no network, no GPU, no benchmark build. ``call_llm`` and
``build_spec`` are monkeypatched; specs are synthetic files in ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.evaluation.llm_evaluate as le
from harness.models import BuildResult, Status
from scripts.evaluation.llm_evaluate import evaluate_translation


def _write_spec(
    path: Path,
    *,
    unique_id: str,
    kernel_name: str,
    parallel_api: str,
    translation_targets: list[str],
    prompt_payload: list[str],
) -> None:
    spec: dict[str, Any] = {
        "identity": {
            "unique_id": unique_id,
            "kernel_name": kernel_name,
            "parallel_api": parallel_api,
        },
        "provenance": {"repo_root": "", "source_path": ""},
        "files": {
            "prompt_payload": prompt_payload,
            "support_files": [],
            "verification_only": [],
            "translation_targets": translation_targets,
        },
        "build": {"commands": {"build": "make foo"}, "working_directory": ""},
        "run": {"input_configurations": {}},
        "verification": {"strategies": []},
    }
    path.write_text(json.dumps(spec), encoding="utf-8")


def _llm_response_for_single_target() -> dict[str, Any]:
    """A well-formed LLM result whose code fence extracts to translated_0.cpp."""
    return {
        "response_text": (
            "```cpp filename=translated_0.cpp\n"
            "int main(){ return 0; }\n"
            "```\n"
        ),
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "duration_seconds": 0.01,
        "finish_reason": "stop",
    }


def test_provider_error_overrides_prior_build_fail(tmp_path, monkeypatch):
    """attempt1 -> BUILD_FAIL, attempt2 -> provider exception ==> PROVIDER_ERROR."""
    # Synthetic same-API pair (source_api == target_api) so no pair contract is
    # required before the (mocked) provider call.
    source_path = tmp_path / "src.json"
    target_path = tmp_path / "tgt.json"
    _write_spec(
        source_path,
        unique_id="testsuite-foo-cuda",
        kernel_name="foo",
        parallel_api="cuda",
        translation_targets=["translated_out.cpp"],
        prompt_payload=["src_kernel.cu"],
    )
    _write_spec(
        target_path,
        unique_id="testsuite-foo2-cuda",
        kernel_name="foo",
        parallel_api="cuda",
        translation_targets=["translated_out.cpp"],
        prompt_payload=["translated_out.cpp"],
    )
    # Give the source payload a real file (realism; not strictly required).
    (tmp_path / "src_kernel.cu").write_text("// source kernel\n", encoding="utf-8")

    # attempt 1: valid response (extraction succeeds); attempt 2: provider raises.
    call_count = {"n": 0}

    def _fake_call_llm(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _llm_response_for_single_target()
        raise RuntimeError("simulated provider outage (HTTP 503)")

    monkeypatch.setattr(le, "call_llm", _fake_call_llm)

    # attempt 1 build deterministically fails.
    def _fake_build_spec(*_args, **_kwargs):
        return BuildResult(
            status=Status.FAIL,
            duration_seconds=0.02,
            stdout="",
            stderr="error: undefined reference to `foo'",
        )

    monkeypatch.setattr(le, "build_spec", _fake_build_spec)

    result = evaluate_translation(
        source_path=source_path,
        target_path=target_path,
        model="test-model",
        project_root=tmp_path,
        max_retries=2,
        save_to_disk=False,
    )

    # Scenario sanity: two attempts, attempt 1 hit BUILD_FAIL, attempt 2 errored.
    assert call_count["n"] == 2, result
    assert result["total_attempts"] == 2
    assert result["attempts"][0]["build_status"] == "fail"
    assert "error" in result["attempts"][1]

    # The regression: overall_status must reflect the provider error, NOT the
    # prior attempt's deterministic BUILD_FAIL.
    assert result["overall_status"] == "PROVIDER_ERROR", result["overall_status"]
    assert result["error_message"] and "LLM call failed" in result["error_message"]


def test_provider_error_first_attempt(tmp_path, monkeypatch):
    """A provider exception on the very first attempt is PROVIDER_ERROR too.

    Pre-OD-4 this was the generic "ERROR"; OD-4 makes it the more precise
    PROVIDER_ERROR (still retryable by --resume). Documents the intentional
    first-attempt status change.
    """
    source_path = tmp_path / "src.json"
    target_path = tmp_path / "tgt.json"
    _write_spec(
        source_path,
        unique_id="testsuite-bar-cuda",
        kernel_name="bar",
        parallel_api="cuda",
        translation_targets=["translated_out.cpp"],
        prompt_payload=["src_kernel.cu"],
    )
    _write_spec(
        target_path,
        unique_id="testsuite-bar2-cuda",
        kernel_name="bar",
        parallel_api="cuda",
        translation_targets=["translated_out.cpp"],
        prompt_payload=["translated_out.cpp"],
    )

    def _always_raise(*_args, **_kwargs):
        raise RuntimeError("simulated provider outage on first call")

    monkeypatch.setattr(le, "call_llm", _always_raise)

    result = evaluate_translation(
        source_path=source_path,
        target_path=target_path,
        model="test-model",
        project_root=tmp_path,
        max_retries=1,
        save_to_disk=False,
    )

    assert result["overall_status"] == "PROVIDER_ERROR", result["overall_status"]


def test_resume_retry_predicate_includes_provider_error():
    """run_eval_batch --resume must re-run PROVIDER_ERROR records."""
    from scripts.evaluation.run_eval_batch import RESUME_RETRY_STATUSES

    assert "PROVIDER_ERROR" in RESUME_RETRY_STATUSES
    # The pre-existing transient statuses stay retryable.
    assert "ERROR" in RESUME_RETRY_STATUSES
    assert "EXTRACTION_FAIL" in RESUME_RETRY_STATUSES
    # Deterministic verdicts must remain non-retryable.
    for deterministic in ("PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "SKIP"):
        assert deterministic not in RESUME_RETRY_STATUSES


class _FakeRegistry:
    def __init__(self, sha256):
        self.sha256 = sha256


def test_resume_disposition_is_registry_aware():
    """--resume retries a record STAMPED with a different contract-registry
    hash than the batch registry; unstamped (historical) records keep the
    plain status rule (2026-08-21 Codex finding)."""
    from scripts.evaluation.run_eval_batch import resume_disposition

    stamped_pass = {"overall_status": "PASS",
                    "pair_contract_registry": {"sha256": "aaa"}}
    historical_pass = {"overall_status": "PASS"}
    stamped_error = {"overall_status": "PROVIDER_ERROR",
                     "pair_contract_registry": {"sha256": "bbb"}}

    # Different registry: retry even a PASS - keeping it would mix generations.
    assert resume_disposition(stamped_pass, _FakeRegistry("bbb")) == "retry-registry"
    # Same registry: normal rules apply.
    assert resume_disposition(stamped_pass, _FakeRegistry("aaa")) == "skip"
    assert resume_disposition(stamped_error, _FakeRegistry("bbb")) == "retry-status"
    # Historical records without the stamp: status rule only.
    assert resume_disposition(historical_pass, _FakeRegistry("bbb")) == "skip"
    # No registry in this batch (same-API-only batch): status rule only.
    assert resume_disposition(stamped_pass, None) == "skip"


def test_provider_error_constant_reuses_stop_reason_literal():
    """The written status reuses the existing module-level constant literal."""
    assert le.STOP_REASON_PROVIDER_ERROR == "PROVIDER_ERROR"


@pytest.mark.parametrize("fault", [
    ValueError("Unknown model provider for 'test-model'."),
    ImportError("openai package not installed."),
])
def test_local_config_fault_stays_generic_error(tmp_path, monkeypatch, fault):
    """ImportError/ValueError out of call_llm are LOCAL configuration faults
    (missing SDK, unknown provider), not outages - they keep the generic
    ERROR verdict (2026-08-21 Codex review narrowing of OD-4)."""
    source_path = tmp_path / "src.json"
    target_path = tmp_path / "tgt.json"
    _write_spec(
        source_path,
        unique_id="testsuite-baz-cuda",
        kernel_name="baz",
        parallel_api="cuda",
        translation_targets=["translated_out.cpp"],
        prompt_payload=["src_kernel.cu"],
    )
    _write_spec(
        target_path,
        unique_id="testsuite-baz2-cuda",
        kernel_name="baz",
        parallel_api="cuda",
        translation_targets=["translated_out.cpp"],
        prompt_payload=["translated_out.cpp"],
    )

    def _raise_config_fault(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(le, "call_llm", _raise_config_fault)

    result = evaluate_translation(
        source_path=source_path,
        target_path=target_path,
        model="test-model",
        project_root=tmp_path,
        max_retries=1,
        save_to_disk=False,
    )

    assert result["overall_status"] == "ERROR", result["overall_status"]
