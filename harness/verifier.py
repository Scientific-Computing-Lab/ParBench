"""harness.verifier — Verify kernel output against spec strategies."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from harness.models import MetricResult, RunResult, Status, VerificationResult

log = logging.getLogger(__name__)


def verify_run(
    spec: dict[str, Any],
    run_result: RunResult,
    *,
    working_dir: Path | None = None,
) -> VerificationResult:
    """Apply ALL verification strategies; every non-SKIP strategy must PASS.

    Returns PASS only when every implemented strategy passes.  Returns FAIL
    on the first strategy that fails (with that strategy's details).  SKIP
    strategies (unimplemented types like file_diff) are ignored —
    they neither block a PASS nor cause a FAIL.

    Supported strategy types
    ------------------------
    * **exit_code** — check ``run_result.exit_code == strategy["expected"]``
    * **stdout_pattern** — ``re.search(pattern, stdout)``
    * **stdout_exclude_pattern** — FAIL if ``re.search(pattern, stdout)`` matches
    * **numeric_comparison** — regex-extract a float from stdout and compare to
      ``strategy["expected"]`` within ``strategy.get("tolerance", 0.0)``.
    * **file_hash** — SHA-256 of ``working_dir / strategy["path"]`` must equal
      ``strategy["expected_sha256"]``. Requires ``working_dir`` kwarg.
    * **file_diff** — *TODO* (returns SKIP)
    * **custom_script** — *TODO* (returns SKIP)

    Parameters
    ----------
    spec:
        Parsed spec dict.
    run_result:
        The :class:`RunResult` to verify.
    working_dir:
        Directory in which the binary ran; required for file-based strategies
        (``file_hash``). ``None`` triggers ERROR for those strategies.

    Returns
    -------
    VerificationResult
    """
    verification = spec.get("verification", {})
    strategies: list[dict[str, Any]] = verification.get("strategies", [])

    if not strategies:
        return VerificationResult(
            status=Status.SKIP,
            strategy_used="(none)",
            details="No verification strategies defined in spec",
        )

    passed_strategies: list[str] = []

    for strategy in strategies:
        stype = strategy.get("type", "")

        if stype == "exit_code":
            result = _check_exit_code(strategy, run_result)
        elif stype == "stdout_pattern":
            result = _check_stdout_pattern(strategy, run_result)
        elif stype == "numeric_comparison":
            result = _verify_numeric_comparison(strategy, run_result)
        elif stype == "stdout_exclude_pattern":
            result = _check_stdout_exclude_pattern(strategy, run_result)
        elif stype == "file_hash":
            result = _verify_file_hash(strategy, working_dir)
        elif stype == "file_diff":
            result = _stub_strategy(stype)
        elif stype == "custom_script":
            result = _stub_strategy(stype)
        else:
            log.warning("Unknown verification strategy type: %s", stype)
            result = _stub_strategy(stype)

        if result.status in (Status.FAIL, Status.ERROR):
            return result
        if result.status == Status.PASS:
            passed_strategies.append(stype)
        # SKIP: ignore and continue to next strategy

    if passed_strategies:
        return VerificationResult(
            status=Status.PASS,
            strategy_used="+".join(passed_strategies),
            details=f"All {len(passed_strategies)} strategies passed: {', '.join(passed_strategies)}",
        )

    # If we exhausted all strategies without a PASS/FAIL, report overall
    return VerificationResult(
        status=Status.SKIP,
        strategy_used="(all skipped)",
        details="No strategy produced a definitive PASS or FAIL",
    )


def extract_metrics(
    spec: dict[str, Any],
    run_result: RunResult,
) -> list[MetricResult]:
    """Extract performance metrics from run output using spec definitions.

    Parameters
    ----------
    spec:
        Parsed spec dict.
    run_result:
        The :class:`RunResult` whose stdout is scanned.

    Returns
    -------
    list[MetricResult]
    """
    perf = spec.get("performance") or {}
    metric_defs: list[dict[str, Any]] = perf.get("metrics", [])

    results: list[MetricResult] = []
    for mdef in metric_defs:
        name = mdef.get("name", "unknown")
        unit = mdef.get("unit", "")
        extraction = mdef.get("extraction", {})

        if extraction.get("type") != "regex":
            log.debug("Skipping non-regex metric extraction: %s", name)
            continue

        pattern = extraction.get("pattern", "")
        capture_group = extraction.get("capture_group", 1)

        try:
            match = re.search(pattern, run_result.stdout)
            if match:
                value = float(match.group(capture_group))
                results.append(MetricResult(name=name, value=value, unit=unit))
            else:
                log.debug("Metric '%s' pattern did not match stdout", name)
        except (ValueError, IndexError, re.error) as exc:
            log.warning("Failed to extract metric '%s': %s", name, exc)

    return results


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _check_exit_code(
    strategy: dict[str, Any],
    run_result: RunResult,
) -> VerificationResult:
    """Verify ``exit_code == expected``."""
    expected = strategy.get("expected", 0)
    desc = strategy.get("description", "")

    if run_result.exit_code == expected:
        return VerificationResult(
            status=Status.PASS,
            strategy_used="exit_code",
            details=f"exit_code={run_result.exit_code} == expected {expected}. {desc}",
        )
    return VerificationResult(
        status=Status.FAIL,
        strategy_used="exit_code",
        details=(f"exit_code={run_result.exit_code} != expected {expected}. {desc}"),
    )


def _check_stdout_pattern(
    strategy: dict[str, Any],
    run_result: RunResult,
) -> VerificationResult:
    """Verify stdout matches a regex pattern."""
    pattern = strategy.get("pattern", "")
    desc = strategy.get("description", "")

    if not pattern:
        return VerificationResult(
            status=Status.SKIP,
            strategy_used="stdout_pattern",
            details="Empty pattern",
        )

    flags = re.MULTILINE if strategy.get("multiline") else 0

    try:
        if re.search(pattern, run_result.stdout, flags):
            return VerificationResult(
                status=Status.PASS,
                strategy_used="stdout_pattern",
                details=f"Pattern '{pattern}' matched. {desc}",
            )
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="stdout_pattern",
            details=f"Pattern '{pattern}' NOT found in stdout. {desc}",
        )
    except re.error as exc:
        return VerificationResult(
            status=Status.ERROR,
            strategy_used="stdout_pattern",
            details=f"Invalid regex '{pattern}': {exc}",
        )


def _check_stdout_exclude_pattern(
    strategy: dict[str, Any],
    run_result: RunResult,
) -> VerificationResult:
    """FAIL if a regex pattern IS found in stdout (inverse of stdout_pattern)."""
    pattern = strategy.get("pattern", "")
    desc = strategy.get("description", "")

    if not pattern:
        return VerificationResult(
            status=Status.SKIP,
            strategy_used="stdout_exclude_pattern",
            details="Empty pattern",
        )

    flags = re.MULTILINE if strategy.get("multiline") else 0

    try:
        if re.search(pattern, run_result.stdout, flags):
            return VerificationResult(
                status=Status.FAIL,
                strategy_used="stdout_exclude_pattern",
                details=f"Excluded pattern '{pattern}' found in stdout. {desc}",
            )
        return VerificationResult(
            status=Status.PASS,
            strategy_used="stdout_exclude_pattern",
            details=f"Excluded pattern '{pattern}' not found. {desc}",
        )
    except re.error as exc:
        return VerificationResult(
            status=Status.ERROR,
            strategy_used="stdout_exclude_pattern",
            details=f"Invalid regex '{pattern}': {exc}",
        )


def _verify_numeric_comparison(
    strategy: dict[str, Any],
    run_result: RunResult,
) -> VerificationResult:
    """Regex-extract a float from stdout and compare to expected within tolerance."""
    extract_regex = strategy.get("extract_regex")
    if not extract_regex:
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="numeric_comparison",
            details="Strategy missing required 'extract_regex' field",
        )

    expected = strategy.get("expected")
    if expected is None:
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="numeric_comparison",
            details="Strategy missing required 'expected' field",
        )

    try:
        expected_f = float(expected)
    except (TypeError, ValueError):
        return VerificationResult(
            status=Status.ERROR,
            strategy_used="numeric_comparison",
            details=f"Strategy 'expected' ({expected!r}) is not a number",
        )

    try:
        tolerance = float(strategy.get("tolerance", 0.0))
    except (TypeError, ValueError):
        return VerificationResult(
            status=Status.ERROR,
            strategy_used="numeric_comparison",
            details=f"Strategy 'tolerance' ({strategy.get('tolerance')!r}) is not a number",
        )

    try:
        match = re.search(extract_regex, run_result.stdout)
    except re.error as exc:
        return VerificationResult(
            status=Status.ERROR,
            strategy_used="numeric_comparison",
            details=f"Invalid regex '{extract_regex}': {exc}",
        )

    if not match:
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="numeric_comparison",
            details=f"Pattern '{extract_regex}' did not match stdout",
        )

    capture = match.group(1) if match.groups() else match.group(0)
    try:
        actual = float(capture)
    except (TypeError, ValueError):
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="numeric_comparison",
            details=f"Captured value {capture!r} is not parseable as float",
        )

    if abs(actual - expected_f) <= tolerance:
        return VerificationResult(
            status=Status.PASS,
            strategy_used="numeric_comparison",
            details=f"actual={actual} within tolerance {tolerance} of expected={expected_f}",
        )
    return VerificationResult(
        status=Status.FAIL,
        strategy_used="numeric_comparison",
        details=f"actual={actual} differs from expected={expected_f} by more than tolerance {tolerance}",
    )


def _verify_file_hash(
    strategy: dict[str, Any],
    working_dir: Path | None,
) -> VerificationResult:
    """SHA-256 of `working_dir / strategy['path']` must equal `expected_sha256`."""
    if working_dir is None:
        return VerificationResult(
            status=Status.ERROR,
            strategy_used="file_hash",
            details="file_hash strategy requires working_dir; none provided",
        )

    rel_path = strategy.get("path")
    expected = strategy.get("expected_sha256")
    if not rel_path or not expected:
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="file_hash",
            details="Strategy missing required 'path' or 'expected_sha256' field",
        )

    base = working_dir.resolve()
    try:
        target = (base / rel_path).resolve()
        target.relative_to(base)
    except (OSError, ValueError):
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="file_hash",
            details=f"Path '{rel_path}' escapes working_dir",
        )

    if not target.is_file():
        return VerificationResult(
            status=Status.FAIL,
            strategy_used="file_hash",
            details=f"Output file not found at '{target}'",
        )

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest == expected.lower():
        return VerificationResult(
            status=Status.PASS,
            strategy_used="file_hash",
            details=f"SHA-256 of '{rel_path}' matches expected",
        )
    return VerificationResult(
        status=Status.FAIL,
        strategy_used="file_hash",
        details=f"SHA-256 of '{rel_path}' = {digest}; expected {expected}",
    )


def _stub_strategy(stype: str) -> VerificationResult:
    """Placeholder for unimplemented strategy types."""
    return VerificationResult(
        status=Status.SKIP,
        strategy_used=stype,
        details=f"Strategy '{stype}' is not yet implemented (TODO)",
    )
