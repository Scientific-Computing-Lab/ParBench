"""Task 10 (gate E-08): bounded agentic-repair protocol budget enforcement.

Covers the acceptance criteria of Task 10 in
docs/superpowers/plans/2026-08-10-parbench-final-engineering.md:
attempts cannot exceed the cumulative trajectory budget, provider accounting
fixtures normalize to the same fields, the dynamic output cap and its
TOKEN_BUDGET_EXHAUSTED stop reason fire, a fourth attempt is impossible, and
the shipped versioned config validates against its schema.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.llm_evaluate import (  # noqa: E402
    AGENTIC_PROTOCOL_CONFIG_PATH,
    AGENTIC_PROTOCOL_SCHEMA_PATH,
    STOP_REASON_ATTEMPT_LIMIT,
    STOP_REASON_TOKEN_BUDGET_EXHAUSTED,
    AgenticProtocolError,
    AgenticTrajectoryBudget,
    dynamic_completion_cap,
    estimate_prompt_tokens_conservative,
    load_agentic_protocol,
    may_start_attempt,
    normalize_provider_usage,
    remaining_trajectory_tokens,
)

DRY_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "dry_run_agentic_protocol.py"


def _protocol(**overrides) -> dict:
    """Small in-memory protocol for controller tests."""
    protocol = {
        "protocol_name": "test",
        "protocol_version": "1.0.0",
        "attempts": {
            "original_attempts": 1,
            "max_repair_attempts": 2,
            "max_total_attempts": 3,
        },
        "token_budget": {
            "trajectory_total_tokens": 10_000,
            "per_call_response_limit_tokens": 2_000,
            "protocol_minimum_completion_tokens": 100,
            "prompt_estimator": {
                "method": "conservative_byte_upper_bound",
                "bytes_per_token": 1.0,
                "per_message_overhead_tokens": 8,
            },
        },
    }
    protocol["token_budget"].update(overrides)
    return protocol


# ---------------------------------------------------------------------------
# Plan-signature functions
# ---------------------------------------------------------------------------


def test_remaining_trajectory_tokens_basic():
    assert remaining_trajectory_tokens(1000, 0) == 1000
    assert remaining_trajectory_tokens(1000, 400) == 600
    assert remaining_trajectory_tokens(1000, 1000) == 0


def test_remaining_trajectory_tokens_never_negative():
    assert remaining_trajectory_tokens(1000, 5000) == 0


def test_remaining_trajectory_tokens_rejects_negative_inputs():
    with pytest.raises(AgenticProtocolError):
        remaining_trajectory_tokens(-1, 0)
    with pytest.raises(AgenticProtocolError):
        remaining_trajectory_tokens(1000, -1)


def test_may_start_attempt_reservation_boundary():
    # prompt + max completion exactly equal to remaining: allowed.
    assert may_start_attempt(1000, 600, 400) is True
    # One token over: refused.
    assert may_start_attempt(1000, 600, 401) is False
    assert may_start_attempt(0, 0, 1) is False


def test_dynamic_completion_cap():
    # Cap is min(per-call limit, remaining - prompt), floored at 0.
    assert dynamic_completion_cap(10_000, 4_000, 2_000) == 2_000
    assert dynamic_completion_cap(5_000, 4_500, 2_000) == 500
    assert dynamic_completion_cap(4_000, 4_500, 2_000) == 0


# ---------------------------------------------------------------------------
# Normalized provider accounting
# ---------------------------------------------------------------------------

PROVIDER_USAGE_FIXTURES = [
    # (fixture name, raw usage payload as each provider reports it)
    ("openai_compatible", {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}),
    ("anthropic", {"input_tokens": 120, "output_tokens": 30}),
    ("google_native", {"promptTokenCount": 120, "candidatesTokenCount": 30}),
]


@pytest.mark.parametrize("name,raw", PROVIDER_USAGE_FIXTURES)
def test_provider_accounting_normalizes_to_same_fields(name, raw):
    usage = normalize_provider_usage(raw)
    assert usage == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }, f"provider fixture {name} did not normalize to the common fields"


def test_provider_accounting_rejects_unknown_shape():
    with pytest.raises(AgenticProtocolError):
        normalize_provider_usage({"tokens": 100})


def test_provider_accounting_rejects_negative_counts():
    with pytest.raises(AgenticProtocolError):
        normalize_provider_usage({"prompt_tokens": -1, "completion_tokens": 5})


def test_prompt_estimator_accepts_fractional_bytes_per_token():
    # Gate finding 6 (2026-08-12): the schema accepts fractions in (0, 1];
    # 0.5 used to be truncated to int 0 and raise ZeroDivisionError. The
    # estimator must compute a positive ceiling.
    estimate = estimate_prompt_tokens_conservative(
        "sys", [{"role": "user", "content": "abc"}], bytes_per_token=0.5
    )
    # 6 bytes at 0.5 bytes/token = 12 tokens + 2 messages * 8 overhead.
    assert estimate == 12 + 16
    with pytest.raises(AgenticProtocolError):
        estimate_prompt_tokens_conservative("sys", [], bytes_per_token=0)


def test_prompt_estimator_is_conservative_and_monotonic():
    short = estimate_prompt_tokens_conservative("sys", [{"role": "user", "content": "x" * 300}])
    longer = estimate_prompt_tokens_conservative("sys", [{"role": "user", "content": "x" * 600}])
    assert longer > short
    # 303 bytes at 1 byte/token = 303 tokens minimum before overhead.
    assert short >= 303


# ---------------------------------------------------------------------------
# Trajectory controller: cumulative budget and attempt cap
# ---------------------------------------------------------------------------


def test_attempts_cannot_exceed_cumulative_budget():
    tracker = AgenticTrajectoryBudget(_protocol())
    total = tracker.total_budget
    # Attempt 1 funded and charged.
    decision = tracker.begin_attempt(3_000)
    assert decision["allowed"] is True
    tracker.record_attempt_usage({"prompt_tokens": 3_000, "completion_tokens": 1_800})
    assert tracker.normalized_used <= total
    # Attempt 2 funded and charged; usage stays within the budget.
    decision = tracker.begin_attempt(3_000)
    assert decision["allowed"] is True
    tracker.record_attempt_usage({"input_tokens": 3_000, "output_tokens": 1_900})
    assert tracker.normalized_used <= total
    # Attempt 3: remaining is 100, prompt alone exceeds it -> refused, and
    # nothing further is ever charged.
    used_before = tracker.normalized_used
    decision = tracker.begin_attempt(3_000)
    assert decision["allowed"] is False
    assert decision["stop_reason"] == STOP_REASON_TOKEN_BUDGET_EXHAUSTED
    assert tracker.normalized_used == used_before <= total


def test_reservation_refuses_before_attempt_starts():
    # Pre-attempt reservation applies to attempt ONE as well.
    tracker = AgenticTrajectoryBudget(_protocol(trajectory_total_tokens=500))
    decision = tracker.begin_attempt(450)
    # cap = min(2000, 500-450) = 50 < protocol minimum 100 -> refused.
    assert decision["allowed"] is False
    assert decision["stop_reason"] == STOP_REASON_TOKEN_BUDGET_EXHAUSTED
    assert tracker.attempts_started == 0


def test_dynamic_cap_below_protocol_minimum_stops_with_exhausted():
    tracker = AgenticTrajectoryBudget(_protocol())
    # Fund and spend most of the budget through a legitimate reservation
    # (record_attempt_usage now fails closed without one).
    decision = tracker.begin_attempt(8_000)
    assert decision["allowed"] is True
    tracker.record_attempt_usage({"prompt_tokens": 8_000, "completion_tokens": 1_950})
    decision = tracker.begin_attempt(10)
    # remaining = 50; cap = 40 < minimum 100.
    assert decision["allowed"] is False
    assert decision["stop_reason"] == STOP_REASON_TOKEN_BUDGET_EXHAUSTED


def test_usage_exceeding_reservation_fails_closed():
    # Gate finding 2 (2026-08-12): an allowed 400-prompt/500-cap reservation
    # against a 1,000-token budget must not accept actual usage of 1,100.
    tracker = AgenticTrajectoryBudget(
        _protocol(
            trajectory_total_tokens=1_000,
            per_call_response_limit_tokens=500,
            protocol_minimum_completion_tokens=100,
        )
    )
    decision = tracker.begin_attempt(400)
    assert decision["allowed"] is True
    used_before = tracker.normalized_used
    with pytest.raises(AgenticProtocolError):
        tracker.record_attempt_usage({"prompt_tokens": 600, "completion_tokens": 500})
    # The over-reservation usage is never charged as accepted spend.
    assert tracker.normalized_used == used_before


def test_record_usage_without_reservation_fails_closed():
    tracker = AgenticTrajectoryBudget(_protocol())
    with pytest.raises(AgenticProtocolError):
        tracker.record_attempt_usage({"prompt_tokens": 10, "completion_tokens": 10})


def test_double_record_for_one_reservation_fails_closed():
    tracker = AgenticTrajectoryBudget(_protocol())
    decision = tracker.begin_attempt(1_000)
    assert decision["allowed"] is True
    tracker.record_attempt_usage({"prompt_tokens": 1_000, "completion_tokens": 500})
    with pytest.raises(AgenticProtocolError):
        tracker.record_attempt_usage({"prompt_tokens": 1_000, "completion_tokens": 500})


def test_fourth_attempt_is_impossible_even_with_ample_budget():
    tracker = AgenticTrajectoryBudget(_protocol(trajectory_total_tokens=1_000_000))
    for _ in range(3):
        decision = tracker.begin_attempt(1_000)
        assert decision["allowed"] is True
        tracker.record_attempt_usage({"prompt_tokens": 1_000, "completion_tokens": 500})
    decision = tracker.begin_attempt(1_000)
    assert decision["allowed"] is False
    assert decision["stop_reason"] == STOP_REASON_ATTEMPT_LIMIT
    assert tracker.attempts_started == 3


def test_max_total_attempts_is_one_original_plus_two_repairs():
    protocol = load_agentic_protocol()
    assert protocol["attempts"]["original_attempts"] == 1
    assert protocol["attempts"]["max_repair_attempts"] == 2
    assert protocol["attempts"]["max_total_attempts"] == 3


# ---------------------------------------------------------------------------
# Versioned config, schema, and the distinct-budget declaration
# ---------------------------------------------------------------------------


def test_shipped_config_validates_against_shipped_schema():
    protocol = load_agentic_protocol(
        AGENTIC_PROTOCOL_CONFIG_PATH, AGENTIC_PROTOCOL_SCHEMA_PATH
    )
    assert protocol["protocol_version"].count(".") == 2


def test_trajectory_budget_distinct_from_per_call_response_limit():
    protocol = load_agentic_protocol()
    budget = protocol["token_budget"]
    assert budget["trajectory_total_tokens"] > budget["per_call_response_limit_tokens"]


def test_schema_rejects_more_than_three_attempts(tmp_path):
    protocol = json.loads(AGENTIC_PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))
    protocol["attempts"] = {
        "original_attempts": 1,
        "max_repair_attempts": 3,
        "max_total_attempts": 4,
    }
    bad = tmp_path / "bad_protocol.json"
    bad.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(AgenticProtocolError):
        load_agentic_protocol(bad, AGENTIC_PROTOCOL_SCHEMA_PATH)


def test_loader_rejects_inconsistent_attempt_arithmetic(tmp_path):
    protocol = json.loads(AGENTIC_PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))
    protocol["attempts"] = {
        "original_attempts": 1,
        "max_repair_attempts": 1,
        "max_total_attempts": 3,
    }
    bad = tmp_path / "bad_protocol.json"
    bad.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(AgenticProtocolError):
        load_agentic_protocol(bad, AGENTIC_PROTOCOL_SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Dry-run script end to end (synthetic, provider-free)
# ---------------------------------------------------------------------------


def test_dry_run_script_writes_bounded_redacted_trajectory(tmp_path):
    out = tmp_path / "agentic_protocol_dry_run.json"
    for extra in ([], ["--check-complete"]):
        proc = subprocess.run(
            [
                sys.executable,
                str(DRY_RUN_SCRIPT),
                "--protocol",
                str(AGENTIC_PROTOCOL_CONFIG_PATH),
                "--output",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
    artifact = json.loads(out.read_text(encoding="utf-8"))
    trajectory = artifact["trajectory"]
    assert artifact["provider_calls_made"] == 0
    assert len(trajectory["attempts"]) == 3
    assert trajectory["stop_reason"] == STOP_REASON_TOKEN_BUDGET_EXHAUSTED
    assert trajectory["tokens_used_normalized"] <= trajectory["trajectory_budget_total"]
    assert trajectory["attempts_started"] <= trajectory["max_total_attempts"]
    # Determinism: regeneration is byte-identical.
    first_bytes = out.read_bytes()
    proc = subprocess.run(
        [
            sys.executable,
            str(DRY_RUN_SCRIPT),
            "--protocol",
            str(AGENTIC_PROTOCOL_CONFIG_PATH),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.read_bytes() == first_bytes


def test_dry_run_diagnostics_are_derived_from_pre_call_messages(tmp_path):
    # Gate finding 5 (2026-08-12): "diagnostics exposed to the model" must
    # describe what the attempt's own request contained, derived from the
    # pre-call message list — not the diagnostic the attempt produced.
    # Attempt 1 (original) saw none; attempt 3's prompt carries both prior
    # failure diagnostics (1252 chars build, 950 chars run).
    out = tmp_path / "agentic_protocol_dry_run.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(DRY_RUN_SCRIPT),
            "--protocol",
            str(AGENTIC_PROTOCOL_CONFIG_PATH),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    attempts = json.loads(out.read_text(encoding="utf-8"))["trajectory"]["attempts"]
    exposed = [a["diagnostics_exposed_to_model"] for a in attempts]
    assert exposed[0] == []
    assert exposed[1] == ["[REDACTED: 1252 chars of failure diagnostics]"]
    assert exposed[2] == [
        "[REDACTED: 1252 chars of failure diagnostics]",
        "[REDACTED: 950 chars of failure diagnostics]",
    ]


# ---------------------------------------------------------------------------
# Gate 2 findings (2026-08-12): provable estimator bound + deep artifact check
# ---------------------------------------------------------------------------


def _run_dry_run(out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(DRY_RUN_SCRIPT),
            "--protocol",
            str(AGENTIC_PROTOCOL_CONFIG_PATH),
            "--output",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def test_estimator_is_a_byte_upper_bound():
    # Gate 2 finding 1 (2026-08-12): the estimate must be a PROVABLE upper
    # bound on tokens. Byte-level BPE tokenizers emit tokens covering >= 1
    # byte each, so token count <= UTF-8 byte count. The estimator therefore
    # counts BYTES (a 2-byte char counts twice) at bytes_per_token <= 1.
    ascii_estimate = estimate_prompt_tokens_conservative(
        "s", [{"role": "user", "content": "x" * 30}], bytes_per_token=1.0
    )
    assert ascii_estimate == 31 + 2 * 8
    multibyte_estimate = estimate_prompt_tokens_conservative(
        "s", [{"role": "user", "content": "é" * 30}], bytes_per_token=1.0
    )
    assert multibyte_estimate == 1 + 60 + 2 * 8  # e-acute is 2 UTF-8 bytes


def test_estimator_rejects_bytes_per_token_above_one():
    # bytes_per_token > 1 would divide the byte count and is NOT a provable
    # upper bound for byte-level tokenizers; the estimator must refuse it.
    with pytest.raises(AgenticProtocolError):
        estimate_prompt_tokens_conservative(
            "s", [{"role": "user", "content": "abc"}], bytes_per_token=3.0
        )
    with pytest.raises(AgenticProtocolError):
        estimate_prompt_tokens_conservative("s", [], bytes_per_token=0)


def test_usage_prompt_above_estimate_fails_closed():
    # Gate 2 finding 1 (2026-08-12): the bound only holds if actual prompt
    # usage never exceeds the pre-call estimate. A provider reporting more
    # prompt tokens than the reservation estimated is a protocol violation.
    tracker = AgenticTrajectoryBudget(_protocol())
    decision = tracker.begin_attempt(1_000)
    assert decision["allowed"] is True
    used_before = tracker.normalized_used
    with pytest.raises(AgenticProtocolError):
        tracker.record_attempt_usage(
            {"prompt_tokens": 1_001, "completion_tokens": 100}
        )
    assert tracker.normalized_used == used_before


def test_dry_run_estimates_never_below_normalized_prompt_usage(tmp_path):
    # Gate 2 finding 1 (2026-08-12): in the shipped artifact, every attempt's
    # estimate must dominate its normalized prompt usage, so that
    # prompt + dynamic cap can never exceed the remaining budget.
    out = tmp_path / "agentic_protocol_dry_run.json"
    proc = _run_dry_run(out)
    assert proc.returncode == 0, proc.stderr
    attempts = json.loads(out.read_text(encoding="utf-8"))["trajectory"]["attempts"]
    assert len(attempts) == 3
    for attempt in attempts:
        assert (
            attempt["prompt_tokens_estimated"]
            >= attempt["normalized_usage"]["prompt_tokens"]
        ), f"attempt {attempt['attempt']}: estimate below actual prompt usage"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda a: a.__setitem__("leaked_reference_implementation", "void k() {}"),
        lambda a: a["trajectory"].__setitem__(
            "tokens_used_normalized", a["trajectory"]["tokens_used_normalized"] - 1
        ),
        lambda a: a["trajectory"]["attempts"][0]["normalized_usage"].__setitem__(
            "prompt_tokens", 1
        ),
    ],
    ids=["unexpected-field", "altered-total-accounting", "altered-attempt-usage"],
)
def test_check_complete_rejects_tampered_artifact(tmp_path, tamper):
    # Gate 2 finding 4 (2026-08-12): --check-complete must reject ANY
    # deviation from deterministic regeneration, not just a fixed field list.
    out = tmp_path / "agentic_protocol_dry_run.json"
    proc = _run_dry_run(out)
    assert proc.returncode == 0, proc.stderr
    artifact = json.loads(out.read_text(encoding="utf-8"))
    tamper(artifact)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    proc = _run_dry_run(out, "--check-complete")
    assert proc.returncode == 1, (
        "tampered artifact passed --check-complete\n" + proc.stdout + proc.stderr
    )


def test_every_runtime_stop_reason_is_declared_in_config():
    """Gate 7 finding 2 (2026-08-12): the versioned protocol must declare
    every stop reason the runtime can emit."""
    import json as _json

    from scripts.evaluation import llm_evaluate as le
    declared = set(_json.loads(
        le.AGENTIC_PROTOCOL_CONFIG_PATH.read_text())["stop_reasons"])
    runtime = {
        v for k, v in vars(le).items() if k.startswith("STOP_REASON_")
    }
    assert runtime <= declared, (
        f"runtime stop reasons missing from config: {sorted(runtime - declared)}"
    )
