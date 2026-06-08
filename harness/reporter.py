"""harness.reporter — Format and display pipeline results."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from harness.models import MetricResult, SpecResult, Status


def format_result(result: SpecResult) -> str:
    """One-line human-readable summary of a :class:`SpecResult`.

    Example::

        [hecbench-nn-cuda] BUILD: PASS (2.3s) | RUN: PASS (0.5s) | VERIFY: PASS (stdout_pattern)
    """
    parts: list[str] = [f"[{result.spec_id}]"]

    # Build
    b = result.build
    parts.append(f"BUILD: {b.status.value.upper()} ({b.duration_seconds}s)")

    # Run(s)
    if result.runs:
        for config_name, r in result.runs.items():
            parts.append(
                f"RUN({config_name}): {r.status.value.upper()} ({r.duration_seconds}s)"
            )
    else:
        parts.append("RUN: —")

    # Verification
    if result.verification:
        v = result.verification
        parts.append(f"VERIFY: {v.status.value.upper()} ({v.strategy_used})")
    else:
        parts.append("VERIFY: —")

    # Metrics
    if result.metrics:
        metric_strs = [f"{m.name}={m.value}{m.unit}" for m in result.metrics]
        parts.append("METRICS: " + ", ".join(metric_strs))

    return " | ".join(parts)


def format_json(result: SpecResult) -> dict[str, Any]:
    """Machine-readable dict representation of a :class:`SpecResult`.

    Enum values are serialised as plain strings so the output is
    directly ``json.dumps``-able.
    """

    def _serialise(obj: Any) -> Any:
        if isinstance(obj, Status):
            return obj.value
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialise(v) for v in obj]
        return obj

    return _serialise(asdict(result))


def print_spec_info(spec: dict[str, Any]) -> None:
    """Pretty-print a short summary of a spec (no build/run)."""
    identity = spec.get("identity", {})
    provenance = spec.get("provenance", {})
    build = spec.get("build", {})
    files = spec.get("files", {})
    meta = spec.get("metadata", {})

    print(f"Spec:       {identity.get('unique_id', '?')}")
    print(f"Kernel:     {identity.get('kernel_name', '?')}")
    print(f"API:        {identity.get('parallel_api', '?')}")
    print(f"Suite:      {identity.get('source_suite', '?')}")
    print(f"Language:   {spec.get('implementation', {}).get('language', '?')}")
    print(f"Repo root:  {provenance.get('repo_root', '?')}")
    print(f"Source:     {provenance.get('source_path', '?')}")

    print(f"\nPrompt payload ({len(files.get('prompt_payload', []))} files):")
    for f in files.get("prompt_payload", []):
        print(f"  • {f}")

    support = files.get("support_files", [])
    if support:
        print(f"Support files ({len(support)}):")
        for f in support:
            print(f"  • {f}")

    verif = files.get("verification_only", [])
    if verif:
        print(f"Verification-only ({len(verif)}):")
        for f in verif:
            print(f"  • {f}")

    print(
        f"\nBuild:      {build.get('build_system', '?')} — "
        f"'{build.get('commands', {}).get('build', '?')}'"
    )
    print(f"Executable: {build.get('outputs', {}).get('executable', '?')}")

    configs = spec.get("run", {}).get("input_configurations", {})
    if configs:
        print(f"\nRun configurations:")
        for name, cfg in configs.items():
            args = " ".join(cfg.get("arguments", []))
            print(f"  {name}: {cfg.get('description', '')}  [{args}]")

    if meta.get("description"):
        print(f"\nDescription: {meta['description']}")


def print_prompt_payload(
    spec_id: str,
    api: str,
    payload: dict[str, str],
) -> None:
    """Print the files an LLM would receive for translation."""
    print(f"═══ Prompt Payload for {spec_id} (API: {api}) ═══\n")
    for fname, content in payload.items():
        print(f"── {fname} {'─' * (60 - len(fname) - 3)}")
        print(content)
        print()


def print_translation_pairs(
    pairs: list[tuple[str, str, str, str]],
) -> None:
    """Print all translation pairs in a readable table."""
    print(f"{'Suite':<12} {'Kernel':<30} {'Source':>8} → {'Target':<8}")
    print("─" * 64)
    for suite, kernel, src, tgt in pairs:
        print(f"{suite:<12} {kernel:<30} {src:>8} → {tgt:<8}")
    print(f"\nTotal: {len(pairs)} translation pairs")
