"""Gate finding 1 (2026-08-12): the bounded agentic-repair protocol is wired
into the provider-calling path of evaluate_translation().

Tested with a MOCKED provider (llm_evaluate.call_llm is monkeypatched) — no
model or provider call is ever made. Covers: pre-attempt budget reservation,
the dynamic completion cap flowing into the provider call, normalized usage
recording, the attempt bound overriding max_retries, visibility enforcement
before any call, and the unchanged live (non-protocol) path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import llm_evaluate  # noqa: E402
from scripts.evaluation.llm_evaluate import (  # noqa: E402
    STOP_REASON_ATTEMPT_LIMIT,
    STOP_REASON_TOKEN_BUDGET_EXHAUSTED,
    AgenticVisibilityError,
    evaluate_translation,
)

ORACLE_SENTINEL = "SENTINEL_ORCH_ORACLE_PATTERN_9c41d2"


def _protocol(**budget_overrides) -> dict:
    budget = {
        "trajectory_total_tokens": 200_000,
        "per_call_response_limit_tokens": 4_096,
        "protocol_minimum_completion_tokens": 256,
        "prompt_estimator": {
            "method": "conservative_byte_upper_bound",
            "bytes_per_token": 1.0,
            "per_message_overhead_tokens": 8,
        },
    }
    budget.update(budget_overrides)
    return {
        "protocol_name": "test-orchestration",
        "protocol_version": "1.0.0",
        "attempts": {
            "original_attempts": 1,
            "max_repair_attempts": 2,
            "max_total_attempts": 3,
        },
        "token_budget": budget,
    }


@pytest.fixture()
def spec_pair(tmp_path):
    """Same-API synthetic spec pair on disk (no pair contract required).

    The target has NONEMPTY infrastructure (a prompt_payload file that is not
    a translation target), exercising the full prompt path the gate review
    flagged.
    """
    src_dir = tmp_path / "src_omp"
    tgt_dir = tmp_path / "tgt_omp"
    src_dir.mkdir()
    tgt_dir.mkdir()

    (src_dir / "kernel_a.cpp").write_text(
        "// source implementation\nvoid kernel_a() { /* work */ }\n", encoding="utf-8"
    )
    (tgt_dir / "kernel_b.cpp").write_text(
        "// target reference implementation body\nvoid kernel_b() {}\n",
        encoding="utf-8",
    )
    (tgt_dir / "infra.cpp").write_text(
        "int infra_helper(int v) { return v + 1; } // infra\n", encoding="utf-8"
    )

    source_spec = {
        "identity": {
            "unique_id": "synthetic-orch-a-omp",
            "kernel_name": "orch",
            "parallel_api": "omp",
        },
        "provenance": {"repo_root": "", "source_path": "src_omp"},
        "files": {
            "prompt_payload": ["kernel_a.cpp"],
            "translation_targets": ["kernel_a.cpp"],
            "support_files": [],
            "verification_only": [],
        },
        "build": {"commands": {"build": "true"}},
    }
    target_spec = {
        "identity": {
            "unique_id": "synthetic-orch-b-omp",
            "kernel_name": "orch",
            "parallel_api": "omp",
        },
        "provenance": {"repo_root": "", "source_path": "tgt_omp"},
        "files": {
            "prompt_payload": ["kernel_b.cpp", "infra.cpp"],
            "translation_targets": ["kernel_b.cpp"],
            "support_files": [],
            "verification_only": [],
        },
        "build": {"commands": {"build": "true"}},
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": ORACLE_SENTINEL},
            ],
        },
    }
    source_path = tmp_path / "source_spec.json"
    target_path = tmp_path / "target_spec.json"
    source_path.write_text(json.dumps(source_spec), encoding="utf-8")
    target_path.write_text(json.dumps(target_spec), encoding="utf-8")
    return source_path, target_path


class FakeProvider:
    """call_llm stand-in that records every request and returns canned output."""

    def __init__(self, response_text: str = "no code fences here"):
        self.calls: list[dict] = []
        self.response_text = response_text

    def __call__(self, model, system_msg, messages, verbose=False, temperature=0.0,
                 thinking_enabled=False, seed=None, top_p=1.0,
                 max_completion_tokens=None):
        self.calls.append({
            "n_messages": len(messages),
            "max_completion_tokens": max_completion_tokens,
        })
        return {
            "response_text": self.response_text,
            "prompt_tokens": 1_000,
            "completion_tokens": 500,
            "duration_seconds": 0.01,
            "finish_reason": "stop",
        }


def _evaluate(spec_pair, tmp_path, monkeypatch, fake, **kwargs):
    monkeypatch.setattr(llm_evaluate, "call_llm", fake)
    source_path, target_path = spec_pair
    return evaluate_translation(
        source_path,
        target_path,
        model="mock-model",
        project_root=tmp_path,
        save_to_disk=False,
        **kwargs,
    )


def test_budget_refusal_stops_before_any_provider_call(
    spec_pair, tmp_path, monkeypatch
):
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        agentic_protocol=_protocol(trajectory_total_tokens=100),
    )
    assert fake.calls == []  # refused before the first provider call
    protocol_block = result["agentic_protocol"]
    assert protocol_block["stop_reason"] == STOP_REASON_TOKEN_BUDGET_EXHAUSTED
    assert protocol_block["attempts_started"] == 0
    assert protocol_block["refused_attempt"]["allowed"] is False
    assert result["overall_status"] == "ERROR"


def test_attempt_bound_and_dynamic_cap_flow_through_provider_calls(
    spec_pair, tmp_path, monkeypatch
):
    # The fake response has no code fences, so every attempt is an
    # EXTRACTION_FAIL retry. max_retries=5 must still stop at the protocol's
    # 3 attempts, and every provider call must carry the dynamic cap.
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        max_retries=5,
        agentic_protocol=_protocol(),
    )
    assert len(fake.calls) == 3  # a fourth attempt is impossible
    for call in fake.calls:
        assert call["max_completion_tokens"] == 4_096  # ample budget: per-call limit
    protocol_block = result["agentic_protocol"]
    assert protocol_block["stop_reason"] == STOP_REASON_ATTEMPT_LIMIT
    assert protocol_block["attempts_started"] == 3
    # Normalized usage: 3 attempts * (1000 prompt + 500 completion).
    assert protocol_block["tokens_used_normalized"] == 4_500
    assert protocol_block["trajectory_budget_total"] == 200_000


def test_dynamic_cap_shrinks_below_per_call_limit(spec_pair, tmp_path, monkeypatch):
    # Probe run: learn attempt 1's byte-bound prompt estimate, then size the
    # budget so attempt 1's cap is remaining - prompt < per-call limit.
    probe = _evaluate(
        spec_pair, tmp_path, monkeypatch, FakeProvider(),
        max_retries=1,
        agentic_protocol=_protocol(),
    )
    estimate = probe["attempts"][0]["agentic_prompt_tokens_estimated"]
    budget_total = estimate + 2_000
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        max_retries=1,
        agentic_protocol=_protocol(trajectory_total_tokens=budget_total),
    )
    assert len(fake.calls) == 1
    cap = fake.calls[0]["max_completion_tokens"]
    assert result["attempts"][0]["agentic_prompt_tokens_estimated"] == estimate
    assert cap == budget_total - estimate == 2_000
    assert 0 < cap < 4_096


def test_visibility_violation_refuses_before_provider_call(
    spec_pair, tmp_path, monkeypatch
):
    # A TRUE leak: the prompt-construction path injects the target's oracle
    # pattern even though it occurs nowhere in the agent's permitted inputs.
    # The composed request must be refused before any provider call.
    # (An oracle-matching string sitting in the SOURCE payload is the
    # opposite case: permitted content the agent holds anyway, exempt under
    # the allowed-baseline rule - covered below.)
    real_build = llm_evaluate.build_translation_prompt

    def _leaky_build(*args, **kwargs):
        system_msg, user_msg, anon_map = real_build(*args, **kwargs)
        return system_msg + f"\nHINT: expected output matches {ORACLE_SENTINEL}", \
            user_msg, anon_map

    monkeypatch.setattr(llm_evaluate, "build_translation_prompt", _leaky_build)
    fake = FakeProvider()
    with pytest.raises(AgenticVisibilityError):
        _evaluate(
            spec_pair, tmp_path, monkeypatch, fake,
            agentic_protocol=_protocol(),
        )
    assert fake.calls == []


def test_oracle_string_inherent_to_source_is_permitted(
    spec_pair, tmp_path, monkeypatch
):
    # Allowed-baseline rule (gate 4 re-check, 2026-08-12): an oracle-matching
    # string that the agent's own source payload already contains carries no
    # hidden information and must NOT refuse the attempt.
    (tmp_path / "src_omp" / "kernel_a.cpp").write_text(
        f'// prints its own banner\nconst char *p = "{ORACLE_SENTINEL}";\n'
        "void kernel_a() {}\n",
        encoding="utf-8",
    )
    fake = FakeProvider()
    _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        agentic_protocol=_protocol(),
    )
    assert len(fake.calls) >= 1


def test_live_path_without_protocol_is_unchanged(spec_pair, tmp_path, monkeypatch):
    fake = FakeProvider()
    result = _evaluate(spec_pair, tmp_path, monkeypatch, fake, max_retries=1)
    assert len(fake.calls) == 1
    assert fake.calls[0]["max_completion_tokens"] is None  # provider default cap
    assert "agentic_protocol" not in result


# ---------------------------------------------------------------------------
# Gate 2 findings (2026-08-12): explicit caller-limit stop reason + feedback
# delivery semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retries", [1, 2])
def test_caller_attempt_limit_gets_explicit_stop_reason(
    spec_pair, tmp_path, monkeypatch, retries
):
    # Gate 2 finding 2 (2026-08-12): when the evaluation loop ends because
    # max_retries is below the protocol's 3-attempt cap, stop_reason must not
    # stay null. The default max_retries=1 reproduces this after one failure.
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        max_retries=retries,
        agentic_protocol=_protocol(),
    )
    protocol_block = result["agentic_protocol"]
    assert protocol_block["attempts_started"] == retries
    assert protocol_block["stop_reason"] == llm_evaluate.STOP_REASON_CALLER_ATTEMPT_LIMIT


def test_default_max_retries_gets_explicit_stop_reason(
    spec_pair, tmp_path, monkeypatch
):
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        agentic_protocol=_protocol(),
    )
    assert (
        result["agentic_protocol"]["stop_reason"]
        == llm_evaluate.STOP_REASON_CALLER_ATTEMPT_LIMIT
    )


def test_protocol_attempt_cap_still_wins_at_three_attempts(
    spec_pair, tmp_path, monkeypatch
):
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        max_retries=3,
        agentic_protocol=_protocol(),
    )
    assert result["agentic_protocol"]["stop_reason"] == STOP_REASON_ATTEMPT_LIMIT


def test_feedback_marked_sent_only_after_next_attempt_allowed(
    spec_pair, tmp_path, monkeypatch
):
    # Gate 2 finding 3 (2026-08-12): feedback is PREPARED when an attempt
    # fails, but it was only DELIVERED if the next attempt passed its budget
    # gate. With ample budget, attempt 1's feedback is delivered and
    # attempt 2's request provably carries it.
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        max_retries=2,
        agentic_protocol=_protocol(),
    )
    assert len(fake.calls) == 2
    first, second = result["attempts"]
    assert first["error_feedback_prepared"] is not None
    assert first["error_feedback_sent"] == first["error_feedback_prepared"]
    assert first["diagnostics_exposed_to_model"] == []
    assert len(second["diagnostics_exposed_to_model"]) == 1
    assert "REDACTED" in second["diagnostics_exposed_to_model"][0]


def test_refused_next_attempt_leaves_feedback_unsent(
    spec_pair, tmp_path, monkeypatch
):
    # Gate 2 finding 3 (2026-08-12): when the next attempt is refused by the
    # budget gate, the failed attempt's record must NOT claim its feedback
    # was sent to a provider.
    # Probe run: learn attempt 1's prompt estimate for this fixture.
    probe = _evaluate(
        spec_pair, tmp_path, monkeypatch, FakeProvider(),
        max_retries=1,
        agentic_protocol=_protocol(),
    )
    estimate_1 = probe["attempts"][0]["agentic_prompt_tokens_estimated"]
    # Budget funds attempt 1 (cap 1,800 >= protocol minimum 256) but leaves
    # estimate_1 + 300 remaining after 1,500 tokens of usage — attempt 2's
    # larger prompt cannot reserve the 256-token minimum completion.
    fake = FakeProvider()
    result = _evaluate(
        spec_pair, tmp_path, monkeypatch, fake,
        max_retries=2,
        agentic_protocol=_protocol(trajectory_total_tokens=estimate_1 + 1_800),
    )
    assert len(fake.calls) == 1
    protocol_block = result["agentic_protocol"]
    assert protocol_block["stop_reason"] == STOP_REASON_TOKEN_BUDGET_EXHAUSTED
    (first,) = result["attempts"]
    assert first["error_feedback_prepared"] is not None
    assert first["error_feedback_sent"] is None
