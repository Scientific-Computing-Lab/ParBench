#!/usr/bin/env python3
"""Synthetic, provider-free dry run of the bounded agentic-repair protocol.

Plan Task 10 (gate E-08). This script exercises the enforcement code in
``scripts/evaluation/llm_evaluate.py`` against a fully synthetic trajectory
and writes the redacted artifact ``results/analysis/agentic_protocol_dry_run.json``.

Safety contract:
  - NO model or provider call is possible: on startup ``llm_evaluate.call_llm``
    is replaced with a function that raises ``ProviderCallForbidden``
    (the ``replay_stored_translations.py`` pattern), and the guard is
    self-tested before the trajectory runs.
  - The output is REDACTED: it contains no source code, no diagnostics text,
    no oracle patterns, no expected outputs — only sizes, token counts, and
    protocol decisions.
  - The output is deterministic: wall-clock durations are synthetic constants
    and no real timestamps are recorded, so regeneration is byte-identical.

Command contract:
    python3 scripts/evaluation/dry_run_agentic_protocol.py \\
      --protocol config/agentic_repair_protocol.json \\
      --output results/analysis/agentic_protocol_dry_run.json
    # validate an existing artifact without regenerating:
    python3 scripts/evaluation/dry_run_agentic_protocol.py ... --check-complete
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation import llm_evaluate  # noqa: E402
from scripts.evaluation.llm_evaluate import (  # noqa: E402
    STOP_REASON_TOKEN_BUDGET_EXHAUSTED,
    AgenticTrajectoryBudget,
    check_agentic_visibility,
    estimate_prompt_tokens_conservative,
    load_agentic_protocol,
)

ARTIFACT_VERSION = "1.0.0"


class ProviderCallForbidden(RuntimeError):
    """Raised when any provider entry point is invoked during the dry run."""


def _forbidden_provider_call(*args: Any, **kwargs: Any) -> Any:
    raise ProviderCallForbidden(
        "dry run is provider-free: llm_evaluate.call_llm must never be invoked"
    )


def install_no_provider_mode() -> None:
    """Make any attempted provider call fail immediately (design contract)."""
    llm_evaluate.call_llm = _forbidden_provider_call


def _self_test_no_provider_guard() -> None:
    try:
        llm_evaluate.call_llm("model", "sys", [])
    except ProviderCallForbidden:
        return
    raise RuntimeError("no-provider guard is NOT installed; refusing to continue")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic trajectory
# ---------------------------------------------------------------------------

# Synthetic prompt payloads. Sizes are chosen so that, under the provable
# byte-upper-bound estimator (gate 2 finding 1, 2026-08-12), three attempts
# fit the shipped 180,000-token trajectory budget but a fourth cannot: the
# fourth attempt's prompt bound alone exceeds the remaining tokens, so its
# reservation fails with TOKEN_BUDGET_EXHAUSTED (and the attempt cap is
# simultaneously reached). Every synthetic usage keeps prompt tokens at or
# below the byte bound — the invariant the estimator now guarantees.
_SYNTHETIC_SOURCE_CODE = "// synthetic CUDA kernel body\n" + ("x" * 30_000)
_SYNTHETIC_DIAGNOSTICS = {
    1: "synthetic build stderr tail (redacted in artifact): " + ("e" * 1_200),
    2: "synthetic run stderr tail (redacted in artifact): " + ("r" * 900),
}
# Forbidden-content sentinels the visibility check screens each request for.
_SYNTHETIC_FORBIDDEN = {
    "reference_implementations": ["SENTINEL_TARGET_REFERENCE_IMPLEMENTATION_BODY_0xA1"],
    "target_implementations": ["SENTINEL_TARGET_REFERENCE_IMPLEMENTATION_BODY_0xA1"],
    "verification_only_files": ["SENTINEL_VERIFICATION_ONLY_FILE_BODY_0xB2"],
    "hidden_expected_outputs": ["SENTINEL_HIDDEN_EXPECTED_OUTPUT_0xC3"],
    "oracle_contents": ["SENTINEL_ORACLE_STDOUT_PATTERN_0xD4"],
}
# Per-attempt synthetic provider usage, one shape per provider family, to
# demonstrate normalized accounting across providers without any call.
# Prompt counts sit just below each attempt's byte-bound estimate (the
# protocol fails closed if they ever exceed it).
_SYNTHETIC_RAW_USAGE = {
    1: {"prompt_tokens": 30_000, "completion_tokens": 10_000},          # OpenAI-compatible
    2: {"input_tokens": 41_000, "output_tokens": 20_000},               # Anthropic
    3: {"promptTokenCount": 52_000, "candidatesTokenCount": 3_500},     # Google native
}
_SYNTHETIC_OUTCOMES = {1: "BUILD_FAIL", 2: "RUN_FAIL", 3: "VERIFY_FAIL"}
_SYNTHETIC_WALL_CLOCK_SECONDS = {1: 41.0, 2: 38.5, 3: 44.25}
# Deterministic synthetic model responses, so the trajectory can publish
# per-attempt response RECORDS (chars + sha256) without any provider call
# (gate 4 finding 3, 2026-08-12).
_SYNTHETIC_RESPONSES = {
    1: "// synthetic translation attempt 1\n" + ("a" * 8_000),
    2: "// synthetic repair attempt 2\n" + ("b" * 6_000),
    3: "// synthetic repair attempt 3\n" + ("c" * 2_000),
}


def _redacted_text_record(text: str) -> dict[str, Any]:
    """Content-free record of a text: length and SHA-256 only."""
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _build_synthetic_messages(attempt_number: int) -> tuple[str, list[dict[str, str]]]:
    """Agent-visible request for *attempt_number* (grows with each repair)."""
    system_msg = (
        "You are a parallel programming expert specializing in CUDA to OpenMP "
        "translation. (synthetic dry-run system message)"
    )
    messages = [{"role": "user", "content": _SYNTHETIC_SOURCE_CODE}]
    for prior in range(1, attempt_number):
        messages.append(
            {"role": "assistant", "content": "// synthetic translated code " + "y" * 10_000}
        )
        messages.append({"role": "user", "content": _SYNTHETIC_DIAGNOSTICS.get(prior, "")})
    return system_msg, messages


def run_synthetic_trajectory(protocol: dict[str, Any]) -> dict[str, Any]:
    """Drive AgenticTrajectoryBudget through a redacted three-attempt trajectory."""
    estimator = protocol["token_budget"]["prompt_estimator"]
    tracker = AgenticTrajectoryBudget(protocol)

    attempts: list[dict[str, Any]] = []
    visibility_checked = 0
    stop_reason: str | None = None
    refusal: dict[str, Any] | None = None

    attempt_number = 1
    while True:
        system_msg, messages = _build_synthetic_messages(attempt_number)
        prompt_estimate = estimate_prompt_tokens_conservative(
            system_msg,
            messages,
            bytes_per_token=estimator["bytes_per_token"],
            per_message_overhead_tokens=estimator["per_message_overhead_tokens"],
        )

        # Visibility rule: screen the complete agent-visible request against
        # every forbidden-content category BEFORE the attempt may start.
        violations = check_agentic_visibility(
            [{"role": "system", "content": system_msg}, *messages],
            _SYNTHETIC_FORBIDDEN,
        )
        if violations:
            raise RuntimeError(f"visibility violation in synthetic request: {violations}")
        visibility_checked += 1

        decision = tracker.begin_attempt(prompt_estimate)
        if not decision["allowed"]:
            stop_reason = decision["stop_reason"]
            refusal = decision
            break

        raw_usage = _SYNTHETIC_RAW_USAGE[attempt_number]
        usage = tracker.record_attempt_usage(raw_usage)
        # Diagnostics exposed to THIS attempt = the prior-failure diagnostics
        # actually present in its pre-call message list (gate finding 5,
        # 2026-08-12) — never the diagnostic this attempt produced. The
        # original attempt therefore reports an empty list.
        message_contents = {m["content"] for m in messages}
        diagnostics_exposed = [
            f"[REDACTED: {len(diag)} chars of failure diagnostics]"
            for prior, diag in sorted(_SYNTHETIC_DIAGNOSTICS.items())
            if diag in message_contents
        ]
        attempts.append(
            {
                "attempt": attempt_number,
                "role": "original" if attempt_number == 1 else "repair",
                "prompt_tokens_estimated": prompt_estimate,
                "completion_cap_tokens": decision["completion_cap"],
                "provider_usage_shape": sorted(raw_usage),
                "normalized_usage": usage,
                "outcome": _SYNTHETIC_OUTCOMES[attempt_number],
                "diagnostics_exposed_to_model": diagnostics_exposed,
                # Redacted per-attempt prompt/response records (gate 4
                # finding 3, 2026-08-12): role structure, lengths, and
                # hashes - never the content itself.
                "prompt_record_redacted": [
                    {"role": "system", **_redacted_text_record(system_msg)},
                    *[
                        {"role": m["role"], **_redacted_text_record(m["content"])}
                        for m in messages
                    ],
                ],
                "response_record_redacted": _redacted_text_record(
                    _SYNTHETIC_RESPONSES[attempt_number]
                ),
                "wall_clock_seconds_synthetic": _SYNTHETIC_WALL_CLOCK_SECONDS[
                    attempt_number
                ],
                "remaining_tokens_after": tracker.remaining,
            }
        )
        attempt_number += 1

    return {
        "attempts": attempts,
        "stop_reason": stop_reason,
        "refused_next_attempt": refusal,
        "visibility_requests_checked": visibility_checked,
        "tokens_used_normalized": tracker.normalized_used,
        "trajectory_budget_total": tracker.total_budget,
        "tokens_remaining": tracker.remaining,
        "attempts_started": tracker.attempts_started,
        "max_total_attempts": tracker.max_total_attempts,
    }


def build_artifact(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    trajectory = run_synthetic_trajectory(protocol)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "generator": "scripts/evaluation/dry_run_agentic_protocol.py",
        "synthetic": True,
        "provider_calls_made": 0,
        "no_provider_guard": (
            "llm_evaluate.call_llm replaced with a ProviderCallForbidden raiser "
            "and self-tested before the trajectory ran"
        ),
        "determinism": (
            "wall-clock durations are synthetic constants; no real timestamps; "
            "regeneration is byte-identical"
        ),
        # Canonical repo-relative location, NOT the --protocol argument: the
        # artifact must regenerate byte-identically from a staged copy of the
        # config too. The sha256 below binds the config CONTENT.
        "protocol_config_path": "config/agentic_repair_protocol.json",
        "protocol_config_sha256": _sha256_file(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "redaction": {
            "policy": (
                "no source code, no diagnostics text, no oracle patterns, no "
                "expected outputs; only sizes, token counts, and protocol decisions"
            ),
            "forbidden_to_agent": protocol["visibility"]["forbidden_to_agent"],
        },
        "task": {
            "source_api": "cuda",
            "target_api": "omp",
            "kernel": "synthetic-kernel",
            "note": "synthetic task; no benchmark source was read",
        },
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Completeness check (--check-complete)
# ---------------------------------------------------------------------------


def check_complete(
    artifact_path: Path, protocol_path: Path, protocol: dict[str, Any]
) -> list[str]:
    """Validate an existing artifact. Returns a list of problems (empty = OK).

    Runs targeted field checks first (for readable diagnostics), then the
    decisive check: the artifact must be byte-for-byte identical to a fresh
    deterministic regeneration (gate 2 finding 4, 2026-08-12). ANY deviation
    — an added field, altered accounting, reordered attempts — fails.
    """
    problems: list[str] = []
    try:
        artifact_text = artifact_path.read_text(encoding="utf-8")
        artifact = json.loads(artifact_text)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read artifact {artifact_path}: {exc}"]

    if artifact.get("provider_calls_made") != 0:
        problems.append("provider_calls_made must be 0")
    if artifact.get("synthetic") is not True:
        problems.append("artifact must declare synthetic: true")
    if artifact.get("protocol_config_sha256") != _sha256_file(protocol_path):
        problems.append("protocol_config_sha256 does not match the current config file")

    trajectory = artifact.get("trajectory") or {}
    attempts = trajectory.get("attempts") or []
    if len(attempts) != 3:
        problems.append(f"expected a complete three-attempt trajectory, got {len(attempts)}")
    if trajectory.get("stop_reason") != STOP_REASON_TOKEN_BUDGET_EXHAUSTED:
        problems.append(
            f"expected explicit stop reason {STOP_REASON_TOKEN_BUDGET_EXHAUSTED}, "
            f"got {trajectory.get('stop_reason')!r}"
        )
    used = trajectory.get("tokens_used_normalized", -1)
    total = trajectory.get("trajectory_budget_total", 0)
    if not (0 <= used <= total):
        problems.append(f"normalized usage {used} exceeds trajectory budget {total}")
    if trajectory.get("attempts_started", 99) > trajectory.get("max_total_attempts", 0):
        problems.append("more attempts started than the protocol maximum")
    # Every attempt must publish redacted prompt/response records and its
    # wall-clock duration (gate 4 finding 3, 2026-08-12).
    for entry in attempts:
        n = entry.get("attempt")
        prompt_rec = entry.get("prompt_record_redacted") or []
        if not prompt_rec or not all(
            {"role", "chars", "sha256"} <= set(m) for m in prompt_rec
        ):
            problems.append(f"attempt {n}: missing/incomplete redacted prompt record")
        resp = entry.get("response_record_redacted") or {}
        if not {"chars", "sha256"} <= set(resp):
            problems.append(f"attempt {n}: missing/incomplete redacted response record")
        if not isinstance(entry.get("wall_clock_seconds_synthetic"), (int, float)):
            problems.append(f"attempt {n}: missing wall-clock duration")

    text = json.dumps(artifact)
    for sentinels in _SYNTHETIC_FORBIDDEN.values():
        for sentinel in sentinels:
            if sentinel in text:
                problems.append(f"forbidden sentinel leaked into artifact: {sentinel}")
    for marker in ("SENTINEL_", "synthetic build stderr tail", "synthetic run stderr tail"):
        if marker in text:
            problems.append(f"unredacted content marker present: {marker!r}")

    # Decisive tamper check: byte-for-byte equality with deterministic
    # regeneration (the artifact is written as json.dumps(..., indent=2)+"\n").
    expected_text = json.dumps(build_artifact(protocol_path, protocol), indent=2) + "\n"
    if artifact_text != expected_text:
        problems.append(
            "artifact is not byte-identical to deterministic regeneration "
            "(tampered, stale, or written by a different generator version)"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True,
                        help="Path to config/agentic_repair_protocol.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Artifact path (results/analysis/agentic_protocol_dry_run.json)")
    parser.add_argument("--schema", type=Path, default=None,
                        help="Protocol JSON schema (default: schema/agentic_repair_protocol_schema.json)")
    parser.add_argument("--check-complete", action="store_true",
                        help="Validate the existing --output artifact; write nothing")
    args = parser.parse_args()

    install_no_provider_mode()
    _self_test_no_provider_guard()

    protocol = load_agentic_protocol(args.protocol, args.schema)

    if args.check_complete:
        problems = check_complete(args.output, args.protocol, protocol)
        if problems:
            for problem in problems:
                print(f"INCOMPLETE: {problem}", file=sys.stderr)
            return 1
        print(f"COMPLETE: {args.output} is a valid redacted three-attempt dry run")
        return 0

    artifact = build_artifact(args.protocol, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    trajectory = artifact["trajectory"]
    print(
        f"wrote {args.output}: {len(trajectory['attempts'])} attempts, "
        f"stop_reason={trajectory['stop_reason']}, "
        f"tokens {trajectory['tokens_used_normalized']}/{trajectory['trajectory_budget_total']}, "
        f"provider_calls_made=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
