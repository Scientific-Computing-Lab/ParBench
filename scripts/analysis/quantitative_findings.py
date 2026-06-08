#!/usr/bin/env python3
"""Quantitative Findings — NeurIPS 2026 ParBench paper.

Computes all 14 quantitative dimensions from the Phase 3 canonical+augmentation corpus:
  D1: Aggregate pass rates          D8: Per-kernel difficulty tiers
  D2: Per-direction rates            D9: Translation complexity correlation
  D3: Direction asymmetry           D10: Cross-suite comparison
  D4: Augmentation trends           D11: Token cost analysis
  D5: Failure taxonomy              D12: SLoC correlation
  D6: Self-repair effectiveness     D13: OpenCL kernel-only effect
  D7: pass@k estimates (C2 only)   D14: Provenance + paper claims mapping

Produces:
  results/analysis/quantitative_findings.json
  results/analysis/quantitative_findings.md

Usage:
    python3 scripts/analysis/quantitative_findings.py \\
        --project-root . -v
    python3 scripts/analysis/quantitative_findings.py \\
        --project-root . --validate
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from harness.constants import EXCLUDED_SPECS

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_DIRECTIONS: list[str] = [
    "cuda-to-omp", "omp-to-cuda",
    "cuda-to-opencl", "opencl-to-cuda",
    "omp-to-opencl", "opencl-to-omp",
]

CASE_STUDY_DIRECTIONS: list[str] = [
    "cuda-to-omp_target", "omp_target-to-cuda",
]

# Together AI pricing (Qwen 3.5 397B)
TOGETHER_INPUT_PRICE_PER_M = 0.60
TOGETHER_OUTPUT_PRICE_PER_M = 3.60

ALPHA = 0.05

# Valid overall_status values
STATUS_VALUES = ("PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL")


# ---------------------------------------------------------------------------
# Provenance helper (Dimension 14)
# ---------------------------------------------------------------------------

def make_finding(
    value,
    source: str,
    files_matched: int,
    derivation: str,
    ci_lower=None,
    ci_upper=None,
    ci_level: float = 0.95,
    n=None,
) -> dict:
    """Wrap a computed value with provenance metadata.

    All numeric rates stored as decimals 0-1.
    """
    finding: dict = {
        "value": value,
        "source": source,
        "files_matched": files_matched,
        "derivation": derivation,
    }
    if ci_lower is not None:
        finding["ci_lower"] = ci_lower
    if ci_upper is not None:
        finding["ci_upper"] = ci_upper
    if ci_lower is not None or ci_upper is not None:
        finding["ci_level"] = ci_level
    if n is not None:
        finding["n"] = n
    return finding


# ---------------------------------------------------------------------------
# Wilson score CI
# ---------------------------------------------------------------------------

def wilson_ci(passes: int, total: int, alpha: float = ALPHA) -> dict:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return {"value": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0, "ci_level": 0.95}

    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = passes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom

    return {
        "value": round(p_hat, 4),
        "ci_lower": round(max(0.0, center - spread), 4),
        "ci_upper": round(min(1.0, center + spread), 4),
        "n": total,
        "ci_level": 0.95,
    }


# ---------------------------------------------------------------------------
# Cohen's h effect size
# ---------------------------------------------------------------------------

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - \
           2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))


# ---------------------------------------------------------------------------
# Cochran-Armitage trend test
# ---------------------------------------------------------------------------

def cochran_armitage_trend(
    pass_counts: list[int],
    total_counts: list[int],
    scores: list[int] | None = None,
) -> dict:
    """Cochran-Armitage test for trend in binomial proportions."""
    k = len(pass_counts)
    if scores is None:
        scores = list(range(k))
    n = np.array(total_counts, dtype=float)
    r = np.array(pass_counts, dtype=float)
    s = np.array(scores, dtype=float)
    N = n.sum()
    R = r.sum()

    if N == 0:
        return {"z": 0.0, "p_value": 1.0, "trend_direction": "none"}

    p_bar = R / N
    t_bar = np.sum(n * s) / N

    numerator = np.sum(s * (r - n * p_bar))
    denominator_sq = p_bar * (1 - p_bar) * (np.sum(n * s**2) - N * t_bar**2)

    if denominator_sq <= 0:
        return {"z": 0.0, "p_value": 1.0, "trend_direction": "none"}

    denominator = math.sqrt(float(denominator_sq))
    z = float(numerator / denominator)
    p_value = float(2 * sp_stats.norm.sf(abs(z)))

    return {
        "z": round(z, 4),
        "p_value": round(p_value, 6),
        "trend_direction": "decreasing" if z < 0 else "increasing" if z > 0 else "none",
    }


# ---------------------------------------------------------------------------
# McNemar exact test
# ---------------------------------------------------------------------------

def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    """McNemar's exact test on paired boolean observations.

    pairs: list of (forward_pass, reverse_pass) tuples.
    """
    a = sum(1 for (f, r) in pairs if f and r)       # both pass
    b = sum(1 for (f, r) in pairs if not f and r)    # reverse only
    c = sum(1 for (f, r) in pairs if f and not r)    # forward only
    d = sum(1 for (f, r) in pairs if not f and not r)  # both fail
    discordant = b + c

    if discordant == 0:
        p_value = 1.0
        method = "exact"
    elif discordant < 25:
        binom_result = sp_stats.binomtest(b, discordant, 0.5, alternative="two-sided")
        p_value = float(binom_result.pvalue)
        method = "exact"
    else:
        chi2_val = (abs(b - c) - 1) ** 2 / discordant if discordant > 0 else 0
        p_value = float(1 - sp_stats.chi2.cdf(chi2_val, df=1))
        method = "chi-squared"

    n_p = len(pairs)
    fwd_pass_count = a + c
    rev_pass_count = a + b
    fwd_rate = fwd_pass_count / n_p if n_p > 0 else 0
    rev_rate = rev_pass_count / n_p if n_p > 0 else 0

    h = cohens_h(fwd_rate, rev_rate)
    abs_h = abs(h)
    effect = (
        "negligible" if abs_h < 0.20
        else "small" if abs_h < 0.50
        else "medium" if abs_h < 0.80
        else "large"
    )

    return {
        "n_paired": n_p,
        "forward_pass_rate": round(fwd_rate, 4),
        "reverse_pass_rate": round(rev_rate, 4),
        "table": {"both_pass": a, "reverse_only": b, "forward_only": c, "both_fail": d},
        "discordant": discordant,
        "method": method,
        "p_value": round(p_value, 6),
        "cohens_h": round(h, 4),
        "effect_size": effect,
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _augment_level_from_filename(stem: str) -> int:
    """Extract augmentation level from result file stem.

    Convention: {src_id}-to-{tgt_id}-L{N}.json       -> N
                {src_id}-to-{tgt_id}-L{N}-s{M}.json  -> N
                {src_id}-to-{tgt_id}.json             -> 0
                {src_id}-to-{tgt_id}-s{M}.json        -> 0
    """
    m = re.search(r"-L(\d+)(?:-s\d+)?$", stem)
    return int(m.group(1)) if m else 0


def _sample_id_from_filename(stem: str) -> int | None:
    """Extract sample_id from result file stem (-s{N} suffix), or None."""
    m = re.search(r"-s(\d+)$", stem)
    return int(m.group(1)) if m else None


def _direction_from_data(data: dict) -> str:
    """Infer translation direction from source_spec/target_spec fields."""
    src = data.get("source_spec", "")
    tgt = data.get("target_spec", "")
    src_api = src.rsplit("-", 1)[-1] if src else "unknown"
    tgt_api = tgt.rsplit("-", 1)[-1] if tgt else "unknown"
    return f"{src_api}-to-{tgt_api}"


def _kernel_from_spec(spec_name: str) -> str:
    """Extract kernel name from spec ID like 'rodinia-bfs-cuda' -> 'bfs'."""
    parts = spec_name.split("-")
    if len(parts) < 3:
        return spec_name
    return "-".join(parts[1:-1])


def _suite_from_spec(spec_name: str) -> str:
    """Extract suite from spec ID like 'rodinia-bfs-cuda' -> 'rodinia'."""
    parts = spec_name.split("-")
    return parts[0] if parts else spec_name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, verbose: bool = False) -> list[dict]:
    """Load all result JSONs from a single model directory."""
    records = []
    skipped = 0

    for json_file in sorted(results_dir.glob("*.json")):
        basename = json_file.name
        # Skip non-result files
        if basename.startswith("batch_") or basename == "eval_summary.json":
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if verbose:
                print(f"  WARNING: Could not read {json_file}: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Enrich with parsed metadata
        stem = json_file.stem
        data["_filename"] = basename
        data["_stem"] = stem

        # Ensure augment_level is set (from file or filename)
        if "augment_level" not in data or data["augment_level"] is None:
            data["augment_level"] = _augment_level_from_filename(stem)

        # Ensure sample_id is set
        if "sample_id" not in data or data.get("sample_id") is None:
            data["sample_id"] = _sample_id_from_filename(stem)

        # Ensure direction is set
        if "direction" not in data or not data.get("direction"):
            data["direction"] = _direction_from_data(data)

        # Ensure kernel is set
        if "kernel" not in data or not data.get("kernel"):
            data["kernel"] = _kernel_from_spec(data.get("source_spec", ""))

        # Ensure _suite is set
        data["_suite"] = _suite_from_spec(data.get("source_spec", ""))

        # Ensure temperature is set
        if "temperature" not in data or data["temperature"] is None:
            data["temperature"] = 0.0

        records.append(data)

    if verbose:
        print(f"  Loaded {len(records)} result files ({skipped} skipped)")

    return records


def exclude_known_fail(records: list[dict]) -> list[dict]:
    """Remove records involving KNOWN_FAIL specs (all 9)."""
    return [
        r for r in records
        if r.get("source_spec") not in EXCLUDED_SPECS
        and r.get("target_spec") not in EXCLUDED_SPECS
    ]


def split_campaigns(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split records into legacy (temp=0.0) and canonical (temp>0).

    Legacy: temp=0.0 (never run — always empty)
    Canonical: temp=0.7 (pass@3 stochastic sampling)
    """
    c1 = []
    c2 = []
    for r in records:
        temp = r.get("temperature", 0.0) or 0.0
        if temp == 0.0:
            c1.append(r)
        else:
            c2.append(r)
    return c1, c2


# ---------------------------------------------------------------------------
# BUILD_FAIL regex patterns (priority order, first match wins)
# ---------------------------------------------------------------------------

_BUILD_FAIL_PATTERNS: list[tuple[str, re.Pattern, object]] = [
    (
        "retained_cuda_api",
        re.compile(
            r"cudaMalloc|cudaFree|cudaMemcpy|cudaMemset|cudaDeviceSynchronize"
            r"|__global__|__device__|__shared__"
            r"|<<<|>>>"
            r"|cudaGetLastError|cudaSetDevice|cudaGetDevice"
            r"|cuda_runtime\.h|cudaError_t|cudaSuccess"
            r"|blockIdx|threadIdx|blockDim|gridDim"
            r"|__syncthreads|atomicAdd\b|atomicSub\b|atomicExch\b"
            r"|atomicMin\b|atomicMax\b|atomicCAS\b"
            r"|calling a __device__ function.*from a __host__",
            re.IGNORECASE,
        ),
        lambda direction: not direction.endswith("-to-cuda"),
    ),
    (
        "retained_cuda_types",
        re.compile(
            r"\bfloat3\b|\bfloat4\b|\bint3\b|\bint4\b|\bdouble3\b|\bdouble4\b"
            r"|\bdim3\b|cudaStream_t|cudaEvent_t"
            r"|texture<|surf<|__constant__",
            re.IGNORECASE,
        ),
        lambda direction: not direction.endswith("-to-cuda"),
    ),
    (
        "retained_opencl_api",
        re.compile(
            r"clCreateBuffer|clEnqueueWriteBuffer|clEnqueueReadBuffer"
            r"|clBuildProgram|clCreateKernel|clSetKernelArg"
            r"|clEnqueueNDRangeKernel|clReleaseMemObject"
            r"|get_global_id|get_local_id|__kernel\b|cl_mem\b",
            re.IGNORECASE,
        ),
        lambda direction: not direction.endswith("-to-opencl"),
    ),
    (
        "missing_header",
        re.compile(
            r"No such file or directory"
            r"|fatal error:.*not found"
            r"|cannot find.*include",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "linker_error",
        re.compile(
            r"undefined reference to"
            r"|multiple definition of"
            r"|collect2:.*error"
            r"|ld returned \d+ exit",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "undeclared_identifier",
        re.compile(
            r"\bundeclared\b"
            r"|was not declared in this scope"
            r"|identifier.*is undefined"
            r"|does not name a type"
            r"|unknown type name"
            r"|has no member named",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "type_mismatch",
        re.compile(
            r"cannot convert"
            r"|incompatible type"
            r"|no matching function"
            r"|conflicting types"
            r"|array subscript is not an integer"
            r"|no viable conversion"
            r"|no instance of overloaded"
            r"|invalid.*operands? to binary"
            r"|cannot be used to initialize",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "syntax_error",
        re.compile(
            r"expected.*before"
            r"|expected.*token"
            r"|expected a [\"'(]|expected a declaration"
            r"|stray.*in program"
            r"|parse error"
            r"|unterminated"
            r"|invalid.*form.*pragma"
            r"|is not valid for.*#pragma"
            r"|must be closely nested"
            r"|too few arguments to function"
            r"|too many arguments (?:to function|in function call)"
            r"|no storage class or type specifier"
            r"|missing closing quote",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "implicit_declaration",
        re.compile(r"implicit declaration of function", re.IGNORECASE),
        None,
    ),
    (
        "redefinition",
        re.compile(
            r"redefinition of"
            r"|redeclaration of"
            r"|duplicate member"
            r"|previously declared here",
            re.IGNORECASE,
        ),
        None,
    ),
]

# RUN_FAIL classification checks
_RUN_FAIL_CHECKS: list[tuple[str, object]] = [
    (
        "wrong_checksum",
        lambda ec, se, so: bool(
            re.search(r"INAVALID CHECKSUM|INVALID CHECKSUM", so or "", re.IGNORECASE)
        ),
    ),
    (
        "wrong_args",
        lambda ec, se, so: bool(
            re.search(r"[Uu]sage:", se or "")
            or re.search(r"[Uu]sage:", so or "")
            or re.search(r"<grid_rows|<sim_time|<temp_file|<power_file|<penalty>", se or "")
            or re.search(r"<inputfile>|<num of frames>", so or "")
        ),
    ),
    (
        "opencl_jit_error",
        lambda ec, se, so: bool(
            re.search(r"\d+\s+errors? generated", se or "")
        ),
    ),
    (
        "simulation_mode_unsupported",
        lambda ec, se, so: bool(
            re.search(
                r"not implemented.*OpenMP|History-based simulation not implemented",
                so or "",
            )
        ),
    ),
    (
        "gpu_memory_error",
        lambda ec, se, so: bool(
            re.search(r"illegal memory access|GPUassert", se or "")
            or re.search(r"illegal memory access|GPUassert", so or "")
        ),
    ),
    (
        "segfault",
        lambda ec, se, so: ec == -11
        or bool(re.search(r"Segmentation fault|SIGSEGV|signal 11", se or "")),
    ),
    (
        "abort",
        lambda ec, se, so: ec == -6
        or bool(re.search(r"stack smashing|SIGABRT|Aborted|signal 6", se or "")),
    ),
    (
        "timeout",
        lambda ec, se, so: bool(
            re.search(r"TIMEOUT|exceeded timeout|killed.*timeout", se or "")
            or re.search(r"TIMEOUT", so or "")
        ),
    ),
    (
        "data_file_missing",
        lambda ec, se, so: bool(
            re.search(r"error opening|file not found|cannot open", so or "")
            or re.search(r"error opening|file not found|cannot open", se or "")
        ),
    ),
    (
        "wrong_exit_code",
        lambda ec, se, so: ec is not None and ec != 0,
    ),
]

# EXTRACTION_FAIL patterns
_EXTRACTION_FAIL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("no_code_blocks", re.compile(r"did not contain parseable code", re.IGNORECASE)),
    ("missing_files", re.compile(
        r"expected file|not found in output|missing.*file|could not find|no .* file",
        re.IGNORECASE,
    )),
]


def _classify_build_fail(data: dict) -> str:
    """Return primary BUILD_FAIL subcategory."""
    snippet = data.get("build_error_snippet") or ""
    direction = data.get("direction") or _direction_from_data(data)

    # Use final attempt's snippet if available
    attempts = data.get("attempts") or []
    if attempts:
        last_snippet = attempts[-1].get("build_error_snippet")
        if last_snippet:
            snippet = last_snippet

    for cat_name, pattern, direction_filter in _BUILD_FAIL_PATTERNS:
        if direction_filter is not None and not direction_filter(direction):
            continue
        if pattern.search(snippet):
            return cat_name

    return "other_build"


def _classify_run_fail(data: dict) -> str:
    """Return primary RUN_FAIL subcategory."""
    exit_code = data.get("run_exit_code")
    stderr = data.get("run_stderr_snippet") or ""
    stdout = data.get("run_stdout_snippet") or ""

    for cat_name, check_fn in _RUN_FAIL_CHECKS:
        if check_fn(exit_code, stderr, stdout):
            return cat_name

    return "other_runtime"


def _classify_extraction_fail(data: dict) -> str:
    """Return primary EXTRACTION_FAIL subcategory."""
    error_msg = data.get("error_message") or ""

    for cat_name, pattern in _EXTRACTION_FAIL_PATTERNS:
        if pattern.search(error_msg):
            return cat_name

    return "malformed_output"


def _classify_verify_fail(data: dict) -> str:
    """Return primary VERIFY_FAIL subcategory."""
    verify_status = data.get("verify_status") or ""
    stdout = data.get("run_stdout_snippet") or ""

    if verify_status == "pass":
        return "pass_overall_mislabel"
    elif verify_status == "error":
        return "verification_error"
    elif verify_status == "fail":
        if not stdout.strip():
            return "missing_output"
        return "wrong_numerical_output"
    return "other_verify"


# ---------------------------------------------------------------------------
# Dimension 1: Aggregate pass rates
# ---------------------------------------------------------------------------

def compute_aggregate_pass_rates(records: list[dict]) -> dict:
    """Compute aggregate pass rates by suite and overall.

    Returns dict with 'overall' and 'per_suite' keys.
    """
    total = len(records)
    passes = sum(1 for r in records if r.get("overall_status") == "PASS")
    ci = wilson_ci(passes, total)

    overall = make_finding(
        value=ci["value"],
        source="computed",
        files_matched=total,
        derivation="wilson_ci(PASS count, total valid records)",
        ci_lower=ci["ci_lower"],
        ci_upper=ci["ci_upper"],
        n=total,
    )

    # Per suite
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_suite[r.get("_suite", "unknown")].append(r)

    per_suite = {}
    for suite_name, recs in sorted(by_suite.items()):
        s_total = len(recs)
        s_passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        s_ci = wilson_ci(s_passes, s_total)
        per_suite[suite_name] = make_finding(
            value=s_ci["value"],
            source="computed",
            files_matched=s_total,
            derivation=f"wilson_ci(PASS count, total) for suite={suite_name}",
            ci_lower=s_ci["ci_lower"],
            ci_upper=s_ci["ci_upper"],
            n=s_total,
        )

    return {"overall": overall, "per_suite": per_suite}


# ---------------------------------------------------------------------------
# Dimension 2: Per-direction pass rates
# ---------------------------------------------------------------------------

def compute_direction_pass_rates(records: list[dict]) -> dict:
    """Compute pass rates per direction (L0 only).

    Separates standard vs case_study directions.
    """
    # L0 records only
    l0 = [r for r in records if r.get("augment_level", 0) == 0]

    by_dir: dict[str, list[dict]] = defaultdict(list)
    for r in l0:
        by_dir[r.get("direction", "unknown")].append(r)

    standard = {}
    case_study = {}

    for d, recs in sorted(by_dir.items()):
        d_total = len(recs)
        d_passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        d_ci = wilson_ci(d_passes, d_total)
        finding = make_finding(
            value=d_ci["value"],
            source="computed",
            files_matched=d_total,
            derivation=f"wilson_ci(PASS count, total) for direction={d}, L0 only",
            ci_lower=d_ci["ci_lower"],
            ci_upper=d_ci["ci_upper"],
            n=d_total,
        )

        if d in STANDARD_DIRECTIONS:
            standard[d] = finding
        elif d in CASE_STUDY_DIRECTIONS:
            case_study[d] = finding
        else:
            # Unknown direction, add to standard
            standard[d] = finding

    return {"standard": standard, "case_study": case_study}


# ---------------------------------------------------------------------------
# Dimension 3: Direction asymmetry (McNemar at L0)
# ---------------------------------------------------------------------------

def compute_direction_asymmetry(records: list[dict]) -> dict:
    """McNemar exact test for direction asymmetry at L0 only.

    Pairs on (suite, kernel) to avoid cross-suite false pairs.
    """
    # Only L0 records
    l0 = [r for r in records if r.get("augment_level", 0) == 0]

    # Build lookup: (suite, kernel, direction) -> passed
    lookup: dict[tuple[str, str, str], bool] = {}
    for r in l0:
        suite = r.get("_suite", "unknown")
        kernel = r.get("kernel", "?")
        direction = r.get("direction", "unknown")
        passed = r.get("overall_status") == "PASS"
        lookup[(suite, kernel, direction)] = passed

    # Identify direction pairs
    directions = {r.get("direction", "") for r in l0 if r.get("direction")}
    pair_map: dict[tuple[str, str], tuple[str, str]] = {}
    for d in directions:
        parts = d.split("-to-")
        if len(parts) == 2:
            reverse = f"{parts[1]}-to-{parts[0]}"
            if reverse in directions:
                key = tuple(sorted([d, reverse]))
                if key not in pair_map:
                    pair_map[key] = (d, reverse)

    n_tests = max(len(pair_map), 1)
    alpha_corrected = ALPHA / n_tests

    results = {}
    for (d_fwd, d_rev) in pair_map.values():
        pairs: list[tuple[bool, bool]] = []

        # Find (suite, kernel) combos present in forward direction
        fwd_keys = {(s, k) for (s, k, d) in lookup if d == d_fwd}
        for (suite, kernel) in fwd_keys:
            fwd_pass = lookup.get((suite, kernel, d_fwd))
            rev_pass = lookup.get((suite, kernel, d_rev))
            if fwd_pass is not None and rev_pass is not None:
                pairs.append((fwd_pass, rev_pass))

        if not pairs:
            continue

        test = mcnemar_exact(pairs)
        test["alpha_corrected"] = round(alpha_corrected, 6)
        test["significant"] = test["p_value"] < alpha_corrected

        finding = make_finding(
            value={
                "forward": d_fwd,
                "reverse": d_rev,
                "test_result": test,
            },
            source="computed",
            files_matched=test["n_paired"] * 2,
            derivation=f"mcnemar_exact paired on (suite, kernel) for {d_fwd} vs {d_rev}, L0 only",
        )
        results[f"{d_fwd} vs {d_rev}"] = finding

    return results


# ---------------------------------------------------------------------------
# Dimension 4: Augmentation trends
# ---------------------------------------------------------------------------

def compute_augmentation_trends(records: list[dict]) -> dict:
    """Cochran-Armitage trend test per-direction and aggregate.

    Includes Cohen's h between adjacent levels.
    """
    result = {}

    # --- Aggregate ---
    agg = _compute_augmentation_for_subset(records, "all records")
    result["aggregate"] = agg

    # --- Per direction ---
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_dir[r.get("direction", "unknown")].append(r)

    per_direction = {}
    for d, recs in sorted(by_dir.items()):
        per_direction[d] = _compute_augmentation_for_subset(recs, f"direction={d}")

    result["per_direction"] = per_direction

    return result


def _compute_augmentation_for_subset(records: list[dict], label: str) -> dict:
    """Compute augmentation trend for a subset of records."""
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_level[r.get("augment_level", 0)].append(r)

    levels_sorted = sorted(by_level.keys())
    if not levels_sorted:
        return {"error": f"No records for {label}"}

    # Per-level pass rates
    per_level = {}
    for lv in levels_sorted:
        recs = by_level[lv]
        passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        ci = wilson_ci(passes, len(recs))
        per_level[f"L{lv}"] = ci

    # Adequate sample size check (n >= 10 per level)
    adequate = all(len(by_level[lv]) >= 10 for lv in levels_sorted)

    # Cochran-Armitage trend test
    pass_counts = [
        sum(1 for r in by_level[lv] if r.get("overall_status") == "PASS")
        for lv in levels_sorted
    ]
    total_counts = [len(by_level[lv]) for lv in levels_sorted]

    ca = cochran_armitage_trend(pass_counts, total_counts, scores=levels_sorted)
    ca["significant"] = ca["p_value"] < ALPHA
    ca["levels"] = levels_sorted
    ca["pass_counts"] = pass_counts
    ca["total_counts"] = total_counts

    # Cohen's h between adjacent levels
    rates = [p / t if t > 0 else 0 for p, t in zip(pass_counts, total_counts)]
    cohens_h_adjacent = {}
    for i in range(len(levels_sorted) - 1):
        lv_a = levels_sorted[i]
        lv_b = levels_sorted[i + 1]
        h = cohens_h(rates[i + 1], rates[i])
        cohens_h_adjacent[f"L{lv_a}_to_L{lv_b}"] = round(h, 4)

    return {
        "per_level": per_level,
        "adequate_sample_size": adequate,
        "cochran_armitage": ca,
        "cohens_h_adjacent": cohens_h_adjacent,
    }


# ---------------------------------------------------------------------------
# Dimension 5: Failure taxonomy
# ---------------------------------------------------------------------------

def compute_failure_taxonomy(records: list[dict]) -> dict:
    """Classify failure types with subcategories.

    Groups by overall_status, then subcategorizes BUILD_FAIL, RUN_FAIL,
    VERIFY_FAIL, and EXTRACTION_FAIL.
    """
    total = len(records)
    failures = [r for r in records if r.get("overall_status") != "PASS"]
    total_failures = len(failures)

    # Count by status
    status_counts: dict[str, int] = Counter()
    for r in records:
        status_counts[r.get("overall_status", "UNKNOWN")] += 1

    # Classify each failure type
    by_status: dict[str, dict] = {}

    # BUILD_FAIL
    build_fails = [r for r in records if r.get("overall_status") == "BUILD_FAIL"]
    bf_subcats: dict[str, int] = Counter()
    for r in build_fails:
        bf_subcats[_classify_build_fail(r)] += 1
    by_status["BUILD_FAIL"] = {
        "count": len(build_fails),
        "subcategories": dict(bf_subcats),
    }

    # RUN_FAIL
    run_fails = [r for r in records if r.get("overall_status") == "RUN_FAIL"]
    rf_subcats: dict[str, int] = Counter()
    for r in run_fails:
        rf_subcats[_classify_run_fail(r)] += 1
    by_status["RUN_FAIL"] = {
        "count": len(run_fails),
        "subcategories": dict(rf_subcats),
    }

    # VERIFY_FAIL
    verify_fails = [r for r in records if r.get("overall_status") == "VERIFY_FAIL"]
    vf_subcats: dict[str, int] = Counter()
    for r in verify_fails:
        vf_subcats[_classify_verify_fail(r)] += 1
    by_status["VERIFY_FAIL"] = {
        "count": len(verify_fails),
        "subcategories": dict(vf_subcats),
    }

    # EXTRACTION_FAIL
    extraction_fails = [r for r in records if r.get("overall_status") == "EXTRACTION_FAIL"]
    ef_subcats: dict[str, int] = Counter()
    for r in extraction_fails:
        ef_subcats[_classify_extraction_fail(r)] += 1
    by_status["EXTRACTION_FAIL"] = {
        "count": len(extraction_fails),
        "subcategories": dict(ef_subcats),
    }

    # Per-suite failure taxonomy
    per_suite: dict[str, dict] = {}
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_suite[r.get("_suite", "unknown")].append(r)
    for suite_name, recs in sorted(by_suite.items()):
        suite_counts: dict[str, int] = Counter()
        for r in recs:
            suite_counts[r.get("overall_status", "UNKNOWN")] += 1
        per_suite[suite_name] = {
            "total": len(recs),
            "status_counts": dict(suite_counts),
        }

    # Top-3 build error subcategories
    top_3_build = sorted(bf_subcats.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "total_records": total,
        "total_failures": total_failures,
        "status_counts": dict(status_counts),
        "by_status": by_status,
        "per_suite": per_suite,
        "top_3_build_subcategories": [
            {"subcategory": name, "count": count} for name, count in top_3_build
        ],
    }


# ---------------------------------------------------------------------------
# Dimension 7: pass@k estimates (canonical only)
# ---------------------------------------------------------------------------

def compute_pass_at_k(records: list[dict]) -> dict:
    """Compute pass@k using Chen et al. (2021) unbiased estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Canonical ONLY. Groups by task = (source_spec, target_spec).
    pass@1 = expected single-sample success rate (c/n per task, macro-averaged).
    pass@3 = probability at least 1 of 3 samples passes.
    Monotonicity: pass@1 <= pass@3 always.
    """
    # Only canonical (augment_level=0) records are seeds;
    # L1-L4 are different augmentation levels, not samples.
    canonical = [r for r in records if r.get("augment_level", 0) == 0]

    # Group by task
    by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in canonical:
        key = (r.get("source_spec", ""), r.get("target_spec", ""))
        by_task[key].append(r)

    total_tasks = len(by_task)
    warnings: list[str] = []

    # Check seed counts
    for task_key, recs in by_task.items():
        if len(recs) != 3:
            warnings.append(
                f"Task {task_key[0]}-to-{task_key[1]} has {len(recs)} seeds (expected 3)"
            )

    def _chen_passk(n: int, c: int, k: int) -> float:
        """Chen et al. (2021) unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
        den = math.comb(n, k)
        if den == 0:
            return 0.0
        return 1.0 - math.comb(n - c, k) / den

    # Per-task pass@k values for macro-averaging
    passk1_values: list[float] = []
    passk3_values: list[float] = []

    # Per-direction and per-suite accumulators
    dir_passk1: dict[str, list[float]] = defaultdict(list)
    dir_passk3: dict[str, list[float]] = defaultdict(list)
    suite_passk1: dict[str, list[float]] = defaultdict(list)
    suite_passk3: dict[str, list[float]] = defaultdict(list)

    # Classify tasks: always_pass, hard_fail, noisy_fail
    always_pass = 0
    hard_fail = 0
    noisy_fail = 0

    for task_key, recs in sorted(by_task.items()):
        passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        n_seeds = len(recs)

        pk1 = _chen_passk(n_seeds, passes, 1)
        pk3 = _chen_passk(n_seeds, passes, 3) if n_seeds >= 3 else pk1
        passk1_values.append(pk1)
        passk3_values.append(pk3)

        # Classify
        if passes == n_seeds:
            always_pass += 1
        elif passes == 0:
            hard_fail += 1
        else:
            noisy_fail += 1

        # Direction and suite breakdown
        direction = recs[0].get("direction") or _direction_from_data(recs[0])
        suite = _suite_from_spec(recs[0].get("source_spec", ""))

        dir_passk1[direction].append(pk1)
        dir_passk3[direction].append(pk3)
        suite_passk1[suite].append(pk1)
        suite_passk3[suite].append(pk3)

    # Macro-average across tasks with normal-approximation CI
    # (pass@k values are continuous per-task floats, not binomial counts)
    def _macro_avg_ci(values: list[float], alpha: float = 0.05) -> dict:
        n = len(values)
        if n == 0:
            return {"value": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_level": 1 - alpha, "n": 0}
        mean = sum(values) / n
        if n == 1:
            return {"value": round(mean, 4), "ci_lower": round(mean, 4), "ci_upper": round(mean, 4), "ci_level": 1 - alpha, "n": 1}
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        se = math.sqrt(variance / n)
        z = 1.96 if alpha == 0.05 else 1.645
        lo = max(0.0, mean - z * se)
        hi = min(1.0, mean + z * se)
        return {"value": round(mean, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4), "ci_level": 1 - alpha, "n": n}

    p1_ci = _macro_avg_ci(passk1_values)
    p3_ci = _macro_avg_ci(passk3_values)

    # Per-direction rates (macro-avg within direction)
    per_direction_pass1 = {}
    per_direction_pass3 = {}
    for d in sorted(set(list(dir_passk1.keys()) + list(dir_passk3.keys()))):
        per_direction_pass1[d] = _macro_avg_ci(dir_passk1[d])
        per_direction_pass3[d] = _macro_avg_ci(dir_passk3[d])

    # Per-suite rates (macro-avg within suite)
    per_suite_pass1 = {}
    per_suite_pass3 = {}
    for s in sorted(set(list(suite_passk1.keys()) + list(suite_passk3.keys()))):
        per_suite_pass1[s] = _macro_avg_ci(suite_passk1[s])
        per_suite_pass3[s] = _macro_avg_ci(suite_passk3[s])

    return {
        "total_tasks": make_finding(
            total_tasks, "computed", total_tasks * 3,
            "count of unique (source_spec, target_spec) tasks in canonical evaluation",
        ),
        "pass_at_1": make_finding(
            p1_ci["value"], "computed", total_tasks,
            "expected single-sample success rate (Chen et al. 2021: 1-C(n-c,1)/C(n,1) = c/n)",
            ci_lower=p1_ci["ci_lower"], ci_upper=p1_ci["ci_upper"],
            n=total_tasks,
        ),
        "pass_at_3": make_finding(
            p3_ci["value"], "computed", total_tasks,
            "probability at least 1 of 3 passes (Chen et al. 2021: 1-C(n-c,3)/C(n,3))",
            ci_lower=p3_ci["ci_lower"], ci_upper=p3_ci["ci_upper"],
            n=total_tasks,
        ),
        "task_classification": {
            "always_pass": always_pass,
            "hard_fail": hard_fail,
            "noisy_fail": noisy_fail,
        },
        "per_direction": {
            "pass_at_1": per_direction_pass1,
            "pass_at_3": per_direction_pass3,
        },
        "per_suite": {
            "pass_at_1": per_suite_pass1,
            "pass_at_3": per_suite_pass3,
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Dimension 8: Per-kernel difficulty tiers (L0 legacy)
# ---------------------------------------------------------------------------

def compute_per_kernel_tiers(records: list[dict]) -> dict:
    """Per-kernel difficulty tiers from L0 legacy records.

    Ranks kernels by pass rate, assigns quartile tiers, identifies
    top-5 easiest and top-5 hardest kernels.
    """
    # L0 only
    l0 = [r for r in records if r.get("augment_level", 0) == 0]

    # Group by kernel
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in l0:
        kernel = r.get("kernel", "unknown")
        by_kernel[kernel].append(r)

    # Compute per-kernel pass rates
    kernel_stats: list[dict] = []
    for kernel_name, recs in sorted(by_kernel.items()):
        k_total = len(recs)
        k_passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        k_ci = wilson_ci(k_passes, k_total)

        # Per-direction breakdown
        dir_breakdown: dict[str, dict] = {}
        by_dir: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_dir[r.get("direction", "unknown")].append(r)
        for d, d_recs in sorted(by_dir.items()):
            d_passes = sum(1 for r in d_recs if r.get("overall_status") == "PASS")
            dir_breakdown[d] = {
                "passes": d_passes,
                "total": len(d_recs),
                "rate": round(d_passes / len(d_recs), 4) if d_recs else 0.0,
            }

        suite = recs[0].get("_suite", "unknown") if recs else "unknown"

        kernel_stats.append({
            "kernel": kernel_name,
            "suite": suite,
            "pass_rate": k_ci["value"],
            "ci_lower": k_ci["ci_lower"],
            "ci_upper": k_ci["ci_upper"],
            "passes": k_passes,
            "total": k_total,
            "per_direction": dir_breakdown,
        })

    # Sort by pass rate descending (easiest first)
    kernel_stats.sort(key=lambda x: x["pass_rate"], reverse=True)

    # Quartile boundaries
    n_kernels = len(kernel_stats)
    q1_boundary = n_kernels // 4
    q2_boundary = n_kernels // 2
    q3_boundary = (3 * n_kernels) // 4

    for i, ks in enumerate(kernel_stats):
        if i < q1_boundary:
            ks["tier"] = "Q1_easiest"
        elif i < q2_boundary:
            ks["tier"] = "Q2"
        elif i < q3_boundary:
            ks["tier"] = "Q3"
        else:
            ks["tier"] = "Q4_hardest"

    # Top-5 / bottom-5
    top_5 = kernel_stats[:5] if len(kernel_stats) >= 5 else kernel_stats
    bottom_5 = kernel_stats[-5:] if len(kernel_stats) >= 5 else kernel_stats

    # Flag anomalous per-direction patterns (>50pp difference between directions)
    anomalies: list[dict] = []
    for ks in kernel_stats:
        dir_rates = [
            (d, info["rate"])
            for d, info in ks["per_direction"].items()
            if info["total"] > 0
        ]
        if len(dir_rates) >= 2:
            rates = [r for _, r in dir_rates]
            max_rate = max(rates)
            min_rate = min(rates)
            if max_rate - min_rate > 0.50:
                best_dir = max(dir_rates, key=lambda x: x[1])
                worst_dir = min(dir_rates, key=lambda x: x[1])
                anomalies.append({
                    "kernel": ks["kernel"],
                    "best_direction": best_dir[0],
                    "best_rate": best_dir[1],
                    "worst_direction": worst_dir[0],
                    "worst_rate": worst_dir[1],
                    "gap_pp": round((best_dir[1] - worst_dir[1]) * 100, 1),
                })

    return {
        "n_kernels": n_kernels,
        "kernels": kernel_stats,
        "top_5_easiest": [
            {"kernel": k["kernel"], "pass_rate": k["pass_rate"], "suite": k["suite"]}
            for k in top_5
        ],
        "top_5_hardest": [
            {"kernel": k["kernel"], "pass_rate": k["pass_rate"], "suite": k["suite"]}
            for k in bottom_5
        ],
        "quartile_boundaries": {
            "Q1_threshold": kernel_stats[q1_boundary]["pass_rate"] if q1_boundary < n_kernels else None,
            "Q2_threshold": kernel_stats[q2_boundary]["pass_rate"] if q2_boundary < n_kernels else None,
            "Q3_threshold": kernel_stats[q3_boundary]["pass_rate"] if q3_boundary < n_kernels else None,
        },
        "direction_anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# Dimension 9: Translation complexity correlation
# ---------------------------------------------------------------------------

def compute_complexity_correlation(records: list[dict], project_root: Path) -> dict:
    """Compute correlation between translation complexity and pass rate.

    Complexity classes: single_file, multi_to_single, single_to_multi, multi_to_multi.
    Uses Chi-squared or Fisher's exact test.
    """
    specs_dir = project_root / "specs"

    # Cache spec file data
    spec_cache: dict[str, dict] = {}

    def _load_spec(spec_id: str) -> dict | None:
        if spec_id in spec_cache:
            return spec_cache[spec_id]
        path = specs_dir / f"{spec_id}.json"
        if not path.exists():
            spec_cache[spec_id] = None  # type: ignore[assignment]
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            spec_cache[spec_id] = data
            return data
        except (json.JSONDecodeError, OSError):
            spec_cache[spec_id] = None  # type: ignore[assignment]
            return None

    def _tt_count(spec: dict) -> int:
        """Count translation target files."""
        files = spec.get("files") or {}
        tt = files.get("translation_targets")
        if tt is not None:
            return len(tt)
        return len(files.get("prompt_payload", []))

    def _classify_pair(src_count: int, tgt_count: int) -> str:
        if src_count <= 1 and tgt_count <= 1:
            return "single_file"
        elif src_count > 1 and tgt_count <= 1:
            return "multi_to_single"
        elif src_count <= 1 and tgt_count > 1:
            return "single_to_multi"
        else:
            return "multi_to_multi"

    # Classify each record
    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    classified = 0
    unclassified = 0

    for r in records:
        src_spec = _load_spec(r.get("source_spec", ""))
        tgt_spec = _load_spec(r.get("target_spec", ""))
        if src_spec is None or tgt_spec is None:
            unclassified += 1
            continue

        src_count = _tt_count(src_spec)
        tgt_count = _tt_count(tgt_spec)
        complexity = _classify_pair(src_count, tgt_count)

        passed = r.get("overall_status") == "PASS"
        if passed:
            class_counts[complexity]["pass"] += 1
        else:
            class_counts[complexity]["fail"] += 1
        classified += 1

    # Per-class pass rates
    per_class: dict[str, dict] = {}
    for cls_name in ["single_file", "multi_to_single", "single_to_multi", "multi_to_multi"]:
        data = class_counts[cls_name]
        total = data["pass"] + data["fail"]
        ci = wilson_ci(data["pass"], total)
        per_class[cls_name] = {
            **ci,
            "passes": data["pass"],
            "fails": data["fail"],
            "total": total,
        }

    # Contingency table for statistical test
    # Rows: complexity classes that exist (have > 0 observations)
    # Columns: [pass, fail]
    active_classes = [c for c in ["single_file", "multi_to_single", "single_to_multi", "multi_to_multi"]
                      if class_counts[c]["pass"] + class_counts[c]["fail"] > 0]

    contingency = np.array([
        [class_counts[c]["pass"], class_counts[c]["fail"]]
        for c in active_classes
    ])

    # Check expected cell counts for Chi-squared validity
    test_result: dict = {}
    if contingency.shape[0] >= 2 and contingency.sum() > 0:
        row_sums = contingency.sum(axis=1, keepdims=True)
        col_sums = contingency.sum(axis=0, keepdims=True)
        grand_total = contingency.sum()
        expected = row_sums * col_sums / grand_total

        min_expected = float(expected.min()) if expected.size > 0 else 0

        if min_expected < 5:
            # Fisher's exact test (for 2x2) or simulated p-value
            if contingency.shape[0] == 2 and contingency.shape[1] == 2:
                odds_ratio, p_val = sp_stats.fisher_exact(contingency)
                test_result = {
                    "method": "fisher_exact",
                    "p_value": round(float(p_val), 6),
                    "odds_ratio": round(float(odds_ratio), 4),
                    "min_expected_cell": round(min_expected, 2),
                }
            else:
                # Use Chi-squared anyway but flag it
                chi2_val, p_val, dof, _ = sp_stats.chi2_contingency(contingency)
                test_result = {
                    "method": "chi_squared_warning_small_cells",
                    "chi2": round(float(chi2_val), 4),
                    "p_value": round(float(p_val), 6),
                    "dof": int(dof),
                    "min_expected_cell": round(min_expected, 2),
                }
        else:
            chi2_val, p_val, dof, _ = sp_stats.chi2_contingency(contingency)
            test_result = {
                "method": "chi_squared",
                "chi2": round(float(chi2_val), 4),
                "p_value": round(float(p_val), 6),
                "dof": int(dof),
                "min_expected_cell": round(min_expected, 2),
            }
        test_result["significant"] = test_result["p_value"] < ALPHA
    else:
        test_result = {"method": "insufficient_data", "p_value": None, "significant": False}

    return {
        "classified": classified,
        "unclassified": unclassified,
        "per_class": per_class,
        "contingency_table": {
            "classes": active_classes,
            "data": contingency.tolist() if contingency.size > 0 else [],
        },
        "test_result": test_result,
    }


# ---------------------------------------------------------------------------
# Dimension 10: Cross-suite comparison
# ---------------------------------------------------------------------------

def compute_cross_suite(records: list[dict], project_root: Path) -> dict:
    """Cross-suite aggregate comparison with SLoC and multi-file characteristics."""
    # Per-suite pass rates (L0 only for fair comparison)
    l0 = [r for r in records if r.get("augment_level", 0) == 0]

    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in l0:
        by_suite[r.get("_suite", "unknown")].append(r)

    per_suite_rates: dict[str, dict] = {}
    for suite_name, recs in sorted(by_suite.items()):
        s_passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        ci = wilson_ci(s_passes, len(recs))
        per_suite_rates[suite_name] = make_finding(
            ci["value"], "computed", len(recs),
            f"wilson_ci(PASS, total) for suite={suite_name}, L0 only",
            ci_lower=ci["ci_lower"], ci_upper=ci["ci_upper"], n=len(recs),
        )

    # SLoC characteristics per suite from sloc_analysis.json
    sloc_path = project_root / "results" / "analysis" / "sloc_analysis.json"
    suite_sloc: dict[str, dict] = {}
    if sloc_path.exists():
        try:
            sloc_data = json.loads(sloc_path.read_text(encoding="utf-8"))
            kernels = sloc_data.get("kernels", {})
            # Group SLoC by suite
            suite_slocs: dict[str, list[int]] = defaultdict(list)
            for kname, kdata in kernels.items():
                s = kdata.get("suite", "unknown")
                sloc_val = kdata.get("physical_sloc", 0)
                if sloc_val > 0:
                    suite_slocs[s].append(sloc_val)
            for s, vals in sorted(suite_slocs.items()):
                vals_sorted = sorted(vals)
                suite_sloc[s] = {
                    "n_kernels": len(vals),
                    "mean_sloc": round(sum(vals) / len(vals), 1) if vals else 0,
                    "median_sloc": float(vals_sorted[len(vals) // 2]) if vals else 0,
                    "min_sloc": min(vals) if vals else 0,
                    "max_sloc": max(vals) if vals else 0,
                }
        except (json.JSONDecodeError, OSError):
            pass

    # Multi-file fraction per suite from spec files
    specs_dir = project_root / "specs"
    suite_multifile: dict[str, dict[str, int]] = defaultdict(lambda: {"multi": 0, "total": 0})
    for spec_file in sorted(specs_dir.glob("*.json")):
        try:
            spec = json.loads(spec_file.read_text(encoding="utf-8"))
            suite = (spec.get("identity") or {}).get("source_suite", "unknown")
            files = spec.get("files") or {}
            tt = files.get("translation_targets")
            if tt is None:
                tt = files.get("prompt_payload", [])
            suite_multifile[suite]["total"] += 1
            if len(tt) > 1:
                suite_multifile[suite]["multi"] += 1
        except (json.JSONDecodeError, OSError):
            continue

    multifile_fractions: dict[str, dict] = {}
    for suite_name in sorted(suite_multifile.keys()):
        md = suite_multifile[suite_name]
        frac = md["multi"] / md["total"] if md["total"] > 0 else 0.0
        multifile_fractions[suite_name] = {
            "multi_file_specs": md["multi"],
            "total_specs": md["total"],
            "fraction": round(frac, 4),
        }

    return {
        "per_suite_pass_rate_l0": per_suite_rates,
        "sloc_characteristics": suite_sloc,
        "multi_file_fraction": multifile_fractions,
    }


# ---------------------------------------------------------------------------
# Dimension 11: Token cost analysis
# ---------------------------------------------------------------------------

def compute_token_cost(records: list[dict]) -> dict:
    """Token cost analysis using Together AI pricing.

    Uses top-level prompt_tokens and completion_tokens fields.
    """
    total_input = 0
    total_output = 0
    tasks_with_tokens = 0
    pass_count = 0

    by_suite: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"input": 0, "output": 0, "tasks": 0, "passes": 0}
    )

    for r in records:
        pt = r.get("prompt_tokens") or 0
        ct = r.get("completion_tokens") or 0

        if pt > 0 or ct > 0:
            tasks_with_tokens += 1
            total_input += pt
            total_output += ct

            suite = r.get("_suite", "unknown")
            by_suite[suite]["input"] += pt
            by_suite[suite]["output"] += ct
            by_suite[suite]["tasks"] += 1

            if r.get("overall_status") == "PASS":
                pass_count += 1
                by_suite[suite]["passes"] += 1

    total_cost = (
        total_input * TOGETHER_INPUT_PRICE_PER_M / 1_000_000
        + total_output * TOGETHER_OUTPUT_PRICE_PER_M / 1_000_000
    )
    cost_per_task = total_cost / tasks_with_tokens if tasks_with_tokens > 0 else 0.0
    cost_per_pass = total_cost / pass_count if pass_count > 0 else 0.0

    # Per-suite
    per_suite: dict[str, dict] = {}
    for suite_name in sorted(by_suite.keys()):
        sd = by_suite[suite_name]
        s_cost = (
            sd["input"] * TOGETHER_INPUT_PRICE_PER_M / 1_000_000
            + sd["output"] * TOGETHER_OUTPUT_PRICE_PER_M / 1_000_000
        )
        s_tasks = sd["tasks"]
        s_passes = sd["passes"]
        per_suite[suite_name] = {
            "input_tokens": sd["input"],
            "output_tokens": sd["output"],
            "total_cost": round(s_cost, 4),
            "tasks": s_tasks,
            "passes": s_passes,
            "cost_per_task": round(s_cost / s_tasks, 4) if s_tasks > 0 else 0.0,
            "cost_per_pass": round(s_cost / s_passes, 4) if s_passes > 0 else 0.0,
        }

    return {
        "total_input_tokens": make_finding(
            total_input, "computed", tasks_with_tokens,
            "sum of prompt_tokens across legacy temp=0.0 records",
        ),
        "total_output_tokens": make_finding(
            total_output, "computed", tasks_with_tokens,
            "sum of completion_tokens across legacy temp=0.0 records",
        ),
        "total_cost": make_finding(
            round(total_cost, 4), "computed", tasks_with_tokens,
            f"input*${TOGETHER_INPUT_PRICE_PER_M}/1M + output*${TOGETHER_OUTPUT_PRICE_PER_M}/1M",
        ),
        "tasks_with_tokens": tasks_with_tokens,
        "pass_count": pass_count,
        "cost_per_task": make_finding(
            round(cost_per_task, 4), "computed", tasks_with_tokens,
            "total_cost / tasks_with_tokens",
        ),
        "cost_per_pass": make_finding(
            round(cost_per_pass, 4), "computed", pass_count,
            "total_cost / pass_count",
        ),
        "per_suite": per_suite,
    }


# ---------------------------------------------------------------------------
# Dimension 12: SLoC correlation
# ---------------------------------------------------------------------------

def compute_sloc_correlation(records: list[dict], project_root: Path) -> dict:
    """Compute Spearman and Pearson correlation between SLoC and pass rate.

    Uses per-kernel L0 pass rates from legacy (temp=0.0) and SLoC from sloc_analysis.json.
    """
    sloc_path = project_root / "results" / "analysis" / "sloc_analysis.json"
    if not sloc_path.exists():
        return {"error": "sloc_analysis.json not found"}

    try:
        sloc_data = json.loads(sloc_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"Could not read sloc_analysis.json: {e}"}

    sloc_kernels = sloc_data.get("kernels", {})

    # Build per-kernel pass rates at L0
    l0 = [r for r in records if r.get("augment_level", 0) == 0]
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in l0:
        by_kernel[r.get("kernel", "unknown")].append(r)

    # Pair SLoC with pass rate
    paired: list[dict] = []
    for kernel_name, recs in sorted(by_kernel.items()):
        if kernel_name not in sloc_kernels:
            continue
        sloc_val = sloc_kernels[kernel_name].get("physical_sloc", 0)
        if sloc_val <= 0:
            continue
        k_passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        k_rate = k_passes / len(recs) if recs else 0.0
        paired.append({
            "kernel": kernel_name,
            "sloc": sloc_val,
            "pass_rate": round(k_rate, 4),
            "n_tasks": len(recs),
        })

    n_paired = len(paired)
    if n_paired < 3:
        return {
            "n_kernels": n_paired,
            "error": "Insufficient paired data (need >= 3 kernels)",
            "paired_data": paired,
        }

    sloc_values = np.array([p["sloc"] for p in paired], dtype=float)
    rate_values = np.array([p["pass_rate"] for p in paired], dtype=float)

    # Spearman
    sp_rho, sp_p = sp_stats.spearmanr(sloc_values, rate_values)
    # Pearson
    pe_r, pe_p = sp_stats.pearsonr(sloc_values, rate_values)

    return {
        "n_kernels": n_paired,
        "spearman": make_finding(
            {"rho": round(float(sp_rho), 4), "p_value": round(float(sp_p), 6)},
            "computed", n_paired,
            "scipy.stats.spearmanr(sloc, pass_rate) at L0",
        ),
        "pearson": make_finding(
            {"r": round(float(pe_r), 4), "p_value": round(float(pe_p), 6)},
            "computed", n_paired,
            "scipy.stats.pearsonr(sloc, pass_rate) at L0",
        ),
        "significant_spearman": float(sp_p) < ALPHA,
        "significant_pearson": float(pe_p) < ALPHA,
        "interpretation": (
            "significant_negative" if float(sp_p) < ALPHA and float(sp_rho) < 0
            else "significant_positive" if float(sp_p) < ALPHA and float(sp_rho) > 0
            else "not_significant"
        ),
        "paired_data": paired,
    }


# ---------------------------------------------------------------------------
# Dimension 13: OpenCL kernel-only effect
# ---------------------------------------------------------------------------

def compute_opencl_kernel_only_effect(records: list[dict]) -> dict:
    """Compare X-to-opencl (kernel-only) vs X-to-omp (full program) pass rates.

    L0 legacy temp=0.0 only. Uses Fisher's exact test on 2x2 contingency table.
    """
    l0 = [r for r in records if r.get("augment_level", 0) == 0]

    # Group 1: All X-to-opencl directions (kernel-only translation)
    opencl_targets = [
        r for r in l0
        if (r.get("direction") or "").endswith("-to-opencl")
    ]
    # Group 2: All X-to-omp directions (full program translation)
    omp_targets = [
        r for r in l0
        if (r.get("direction") or "").endswith("-to-omp")
    ]

    ocl_passes = sum(1 for r in opencl_targets if r.get("overall_status") == "PASS")
    ocl_total = len(opencl_targets)
    omp_passes = sum(1 for r in omp_targets if r.get("overall_status") == "PASS")
    omp_total = len(omp_targets)

    ocl_ci = wilson_ci(ocl_passes, ocl_total)
    omp_ci = wilson_ci(omp_passes, omp_total)

    # Fisher's exact test (2x2)
    contingency = np.array([
        [ocl_passes, ocl_total - ocl_passes],
        [omp_passes, omp_total - omp_passes],
    ])

    test_result: dict = {}
    if ocl_total > 0 and omp_total > 0:
        odds_ratio, p_val = sp_stats.fisher_exact(contingency)
        h = cohens_h(
            ocl_passes / ocl_total if ocl_total > 0 else 0,
            omp_passes / omp_total if omp_total > 0 else 0,
        )
        test_result = {
            "method": "fisher_exact",
            "odds_ratio": round(float(odds_ratio), 4),
            "p_value": round(float(p_val), 6),
            "cohens_h": round(h, 4),
            "significant": float(p_val) < ALPHA,
        }
    else:
        test_result = {"method": "insufficient_data", "p_value": None, "significant": False}

    return {
        "x_to_opencl": make_finding(
            ocl_ci["value"], "computed", ocl_total,
            "wilson_ci for all X-to-opencl directions, L0 only",
            ci_lower=ocl_ci["ci_lower"], ci_upper=ocl_ci["ci_upper"], n=ocl_total,
        ),
        "x_to_omp": make_finding(
            omp_ci["value"], "computed", omp_total,
            "wilson_ci for all X-to-omp directions, L0 only",
            ci_lower=omp_ci["ci_lower"], ci_upper=omp_ci["ci_upper"], n=omp_total,
        ),
        "test_result": test_result,
    }


# ---------------------------------------------------------------------------
# Rodinia subset helper
# ---------------------------------------------------------------------------

def compute_rodinia_subset(records: list[dict]) -> dict:
    """Compute key metrics for Rodinia-only subset (for paper_claims scope matching)."""
    rodinia = [r for r in records if r.get("_suite") == "rodinia"]
    total = len(rodinia)
    passes = sum(1 for r in rodinia if r.get("overall_status") == "PASS")
    ci = wilson_ci(passes, total)

    return {
        "total_tasks": total,
        "passes": passes,
        "pass_rate": make_finding(
            ci["value"], "computed", total,
            "wilson_ci(PASS, total) for suite=rodinia only",
            ci_lower=ci["ci_lower"], ci_upper=ci["ci_upper"], n=total,
        ),
    }


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_metadata(
    all_records: list[dict],
    valid_records: list[dict],
    c1: list[dict],
    c2: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Build metadata section with file counts, timestamps, git hash."""
    # Git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(args.project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "project_root": str(args.project_root),
        "model": args.model_dir,
        "excluded_specs": sorted(EXCLUDED_SPECS),
        "excluded_specs_count": len(EXCLUDED_SPECS),
        "file_counts": {
            "total_on_disk": len(all_records),
            "excluded_known_fail": len(all_records) - len(valid_records),
            "valid_after_exclusion": len(valid_records),
            "canonical_valid": len(c2),
        },
        "canonical_description": "Canonical evaluation (temperature=0.7, stochastic sampling, pass@3)",
    }


def build_paper_claims(c1_results: dict, c2_results: dict, project_root: Path) -> list[dict]:
    """Map paper.tex claims to JSON field paths with scope annotations.

    Each claim gets: claim_id, paper_location, json_path, scope, current_value, display_value.
    """
    claims: list[dict] = []

    def _safe_get(d: dict, path: str, default=None):
        """Navigate dot-separated path safely."""
        parts = path.split(".")
        current = d
        for p in parts:
            if isinstance(current, dict):
                current = current.get(p, default)
            else:
                return default
        return current

    # Helper: extract scalar from provenance-wrapped finding
    def _val(finding, key="value"):
        if isinstance(finding, dict):
            return finding.get(key)
        return finding

    # --- Claim 1: Overall pass rate (Rodinia only, 480 tasks) ---
    rod_sub = c1_results.get("rodinia_subset", {})
    rod_rate = _val(rod_sub.get("pass_rate", {}))
    rod_ci_lo = _val(rod_sub.get("pass_rate", {}), "ci_lower")
    rod_ci_hi = _val(rod_sub.get("pass_rate", {}), "ci_upper")
    claims.append({
        "claim_id": "overall_pass_rate_rodinia",
        "paper_location": "abstract/line~71, S6.1/line~707",
        "json_path": "canonical.rodinia_subset.pass_rate.value",
        "scope": "rodinia_only",
        "current_value": rod_rate,
        "display_value": f"{_pct(rod_rate)} [{_pct(rod_ci_lo)}, {_pct(rod_ci_hi)}]" if rod_rate is not None else "N/A",
    })

    # --- Claim 2: Primary campaign tasks count (480 Rodinia, 700 all-suite) ---
    rod_total = rod_sub.get("total_tasks", 0)
    agg = c1_results.get("aggregate_pass_rates", {})
    overall_n = _val(agg.get("overall", {}), "n")
    claims.append({
        "claim_id": "primary_campaign_task_counts",
        "paper_location": "abstract/line~61, S5.2/line~630",
        "json_path": "canonical.aggregate_pass_rates.overall.n",
        "scope": "all_suite",
        "current_value": {"rodinia_tasks": rod_total, "all_suite_tasks": overall_n},
        "display_value": f"{rod_total} Rodinia, {overall_n} all-suite",
    })

    # --- Claim 3: pass@k tasks count ---
    c2_passk = c2_results.get("pass_at_k", {})
    passk_total = _val(c2_passk.get("total_tasks", {}))
    claims.append({
        "claim_id": "passk_task_count",
        "paper_location": "S1/line~106, S5.5/line~689",
        "json_path": "canonical.pass_at_k.total_tasks.value",
        "scope": "all_suite",
        "current_value": passk_total,
        "display_value": f"{passk_total} pass@k tasks",
    })

    # --- Claim 4: BUILD_FAIL percentage ---
    tax = c1_results.get("failure_taxonomy", {})
    sc = tax.get("status_counts", {})
    bf_count = sc.get("BUILD_FAIL", 0)
    total_recs = tax.get("total_records", 1)
    bf_pct = bf_count / total_recs if total_recs > 0 else 0
    claims.append({
        "claim_id": "build_fail_percentage",
        "paper_location": "abstract/line~66, S1/line~107, S6.2/line~714",
        "json_path": "canonical.failure_taxonomy.status_counts.BUILD_FAIL",
        "scope": "all_suite",
        "current_value": {"count": bf_count, "percentage": round(bf_pct, 4)},
        "display_value": f"{bf_count}/{total_recs} = {bf_pct*100:.1f}%",
    })

    # --- Claim 5: VERIFY_FAIL percentage ---
    vf_count = sc.get("VERIFY_FAIL", 0)
    vf_pct = vf_count / total_recs if total_recs > 0 else 0
    claims.append({
        "claim_id": "verify_fail_percentage",
        "paper_location": "abstract/line~66, S6.2/line~714",
        "json_path": "canonical.failure_taxonomy.status_counts.VERIFY_FAIL",
        "scope": "all_suite",
        "current_value": {"count": vf_count, "percentage": round(vf_pct, 4)},
        "display_value": f"{vf_count}/{total_recs} = {vf_pct*100:.1f}%",
    })

    # --- Claim 6: CUDA-to-OpenMP pass rate ---
    dir_rates = c1_results.get("direction_pass_rates", {})
    std_dirs = dir_rates.get("standard", {})
    cuda_omp = std_dirs.get("cuda-to-omp", {})
    claims.append({
        "claim_id": "cuda_to_omp_pass_rate",
        "paper_location": "S6.1/line~909, S7.1/line~1041",
        "json_path": "canonical.direction_pass_rates.standard.cuda-to-omp.value",
        "scope": "all_suite",
        "current_value": _val(cuda_omp),
        "display_value": _pct(_val(cuda_omp)),
    })

    # --- Claim 7: L0 pass rate ---
    aug = c1_results.get("augmentation_trends", {})
    agg_trend = aug.get("aggregate", {})
    per_level = agg_trend.get("per_level", {})
    l0_data = per_level.get("L0", {})
    claims.append({
        "claim_id": "l0_pass_rate",
        "paper_location": "S6.4/line~899",
        "json_path": "canonical.augmentation_trends.aggregate.per_level.L0.value",
        "scope": "all_suite",
        "current_value": l0_data.get("value"),
        "display_value": _pct(l0_data.get("value")),
    })

    # --- Claim 8: Cochran-Armitage z and p ---
    ca = agg_trend.get("cochran_armitage", {})
    claims.append({
        "claim_id": "cochran_armitage_trend",
        "paper_location": "abstract/line~71, S6.4/line~899, S6.8/line~1003",
        "json_path": "canonical.augmentation_trends.aggregate.cochran_armitage",
        "scope": "all_suite",
        "current_value": {"z": ca.get("z"), "p_value": ca.get("p_value")},
        "display_value": f"z={ca.get('z')}, p={ca.get('p_value')}",
    })

    # --- Claim 12: Cohen's h range ---
    cohens_adj = agg_trend.get("cohens_h_adjacent", {})
    if cohens_adj:
        h_values = list(cohens_adj.values())
        h_min = min(h_values) if h_values else 0
        h_max = max(h_values) if h_values else 0
    else:
        h_min = h_max = 0
    claims.append({
        "claim_id": "cohens_h_range",
        "paper_location": "S6.4/implied",
        "json_path": "canonical.augmentation_trends.aggregate.cohens_h_adjacent",
        "scope": "all_suite",
        "current_value": {"min": h_min, "max": h_max},
        "display_value": f"h range [{h_min:.4f}, {h_max:.4f}]",
    })

    # --- Claim 13: Spec count (96) ---
    claims.append({
        "claim_id": "spec_count",
        "paper_location": "abstract/line~60, S3.2/line~297, S4.3/line~511",
        "json_path": "metadata.note",
        "scope": "all_suite",
        "current_value": 96,
        "display_value": "96 specs (60+25+4+4+3)",
    })

    # --- Claim 14: OpenCL-to-CUDA pass rate ---
    ocl_cuda = std_dirs.get("opencl-to-cuda", {})
    claims.append({
        "claim_id": "opencl_to_cuda_pass_rate",
        "paper_location": "S6.1/line~941",
        "json_path": "canonical.direction_pass_rates.standard.opencl-to-cuda.value",
        "scope": "all_suite",
        "current_value": _val(ocl_cuda),
        "display_value": _pct(_val(ocl_cuda)),
    })

    # --- Claim 15: Multi-file percentage ---
    cross = c1_results.get("cross_suite", {})
    mf = cross.get("multi_file_fraction", {})
    total_multi = sum(v.get("multi_file_specs", 0) for v in mf.values())
    total_specs = sum(v.get("total_specs", 0) for v in mf.values())
    mf_pct = total_multi / total_specs if total_specs > 0 else 0
    claims.append({
        "claim_id": "multi_file_percentage",
        "paper_location": "S1/implied, S4/implied",
        "json_path": "canonical.cross_suite.multi_file_fraction",
        "scope": "all_suite",
        "current_value": {"multi_file_specs": total_multi, "total_specs": total_specs, "fraction": round(mf_pct, 4)},
        "display_value": f"{total_multi}/{total_specs} = {mf_pct*100:.1f}%",
    })

    # --- Claim 16: Overall pass rate (all-suite) ---
    all_rate = _val(agg.get("overall", {}))
    all_ci_lo = _val(agg.get("overall", {}), "ci_lower")
    all_ci_hi = _val(agg.get("overall", {}), "ci_upper")
    claims.append({
        "claim_id": "overall_pass_rate_all_suite",
        "paper_location": "S6.1 (all-suite scope)",
        "json_path": "canonical.aggregate_pass_rates.overall.value",
        "scope": "all_suite",
        "current_value": all_rate,
        "display_value": f"{_pct(all_rate)} [{_pct(all_ci_lo)}, {_pct(all_ci_hi)}]" if all_rate is not None else "N/A",
    })

    # --- Claim 18: pass@1 and pass@3 ---
    p1 = _val(c2_passk.get("pass_at_1", {}))
    p3 = _val(c2_passk.get("pass_at_3", {}))
    claims.append({
        "claim_id": "pass_at_k_rates",
        "paper_location": "S6.5/line~955",
        "json_path": "canonical.pass_at_k.pass_at_1.value",
        "scope": "all_suite",
        "current_value": {"pass_at_1": p1, "pass_at_3": p3},
        "display_value": f"pass@1={_pct(p1)}, pass@3={_pct(p3)}",
    })

    # --- Claim 19: Token cost ---
    tc = c1_results.get("token_cost", {})
    total_cost = _val(tc.get("total_cost", {}))
    claims.append({
        "claim_id": "token_cost",
        "paper_location": "S5.2/implied",
        "json_path": "canonical.token_cost.total_cost.value",
        "scope": "all_suite",
        "current_value": total_cost,
        "display_value": f"${total_cost:.2f}" if total_cost is not None else "N/A",
    })

    # --- Claim 20: SLoC correlation ---
    sloc_corr = c1_results.get("sloc_correlation", {})
    sp_finding = sloc_corr.get("spearman", {})
    sp_val = _val(sp_finding)
    claims.append({
        "claim_id": "sloc_correlation",
        "paper_location": "S7/implied",
        "json_path": "canonical.sloc_correlation.spearman.value",
        "scope": "all_suite",
        "current_value": sp_val,
        "display_value": (
            f"rho={sp_val.get('rho')}, p={sp_val.get('p_value')}"
            if isinstance(sp_val, dict) else str(sp_val)
        ),
    })

    return claims


# ---------------------------------------------------------------------------
# Cross-check against existing analysis files
# ---------------------------------------------------------------------------

def cross_check(
    c1_results: dict,
    c2_results: dict,
    project_root: Path,
    verbose: bool,
) -> dict:
    """Cross-check computed values against existing analysis JSONs.

    Compares against paper_data.json, statistical_analysis.json, error_taxonomy.json.
    Returns dict with warnings list and pass/fail status.
    """
    warnings = []
    checks_run = 0

    analysis_dir = project_root / "results" / "analysis"

    # --- Check against paper_data.json ---
    paper_data_path = analysis_dir / "paper_data.json"
    if paper_data_path.exists():
        try:
            pd = json.loads(paper_data_path.read_text())
            checks_run += 1

            # File count check (note: paper_data.json uses 6 excluded, we use 8)
            pd_total = (pd.get("file_counts") or {}).get("total_on_disk", 0)
            our_total = c1_results.get("_metadata", {}).get("total_on_disk", 0)
            if pd_total != our_total and pd_total > 0 and our_total > 0:
                warnings.append(
                    f"INFO: paper_data.json total_on_disk={pd_total}, "
                    f"our total={our_total} (expected match)"
                )

            # Legacy overall pass rate (paper_data.json uses 6 exclusions = 710 records)
            pd_c1_rate = (pd.get("primary_campaign") or {}).get("overall", {}).get("pass_rate")
            our_c1_overall = c1_results.get("aggregate_pass_rates", {}).get("overall", {})
            our_c1_rate = our_c1_overall.get("value")

            if pd_c1_rate is not None and our_c1_rate is not None:
                checks_run += 1
                diff = abs(pd_c1_rate - our_c1_rate)
                if diff > 0.05:
                    warnings.append(
                        f"WARNING: C1 overall pass rate mismatch: paper_data={pd_c1_rate}, "
                        f"ours={our_c1_rate} (diff={diff:.4f}, >5%). "
                        f"Note: different KNOWN_FAIL exclusion count (6 vs 8)."
                    )
                elif diff > 0.001:
                    warnings.append(
                        f"INFO: C1 overall pass rate minor diff: paper_data={pd_c1_rate}, "
                        f"ours={our_c1_rate} (diff={diff:.4f}). "
                        f"Expected due to 8 vs 6 KNOWN_FAIL exclusions."
                    )

        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"WARNING: Could not read paper_data.json: {e}")

    # --- Check against error_taxonomy.json ---
    taxonomy_path = analysis_dir / "error_taxonomy.json"
    if taxonomy_path.exists():
        try:
            et = json.loads(taxonomy_path.read_text())
            checks_run += 1

            # Total results check (error_taxonomy.json includes KNOWN_FAIL)
            et_total = et.get("total_results", 0)
            if verbose:
                print(f"  Cross-check: error_taxonomy.json total_results={et_total}")

        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"WARNING: Could not read error_taxonomy.json: {e}")

    # --- Check against statistical_analysis.json ---
    stat_path = analysis_dir / "statistical_analysis.json"
    if stat_path.exists():
        try:
            sa = json.loads(stat_path.read_text())
            checks_run += 1

            if verbose:
                print(f"  Cross-check: statistical_analysis.json total_records={sa.get('total_records')}")

        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"WARNING: Could not read statistical_analysis.json: {e}")

    # --- Check against token_analysis.json ---
    token_path = analysis_dir / "token_analysis.json"
    if token_path.exists():
        try:
            ta = json.loads(token_path.read_text())
            checks_run += 1

            # Compare total tokens
            ta_prompt = ta.get("grand_prompt_tokens", 0)
            ta_completion = ta.get("grand_completion_tokens", 0)
            ta_cost = ta.get("grand_total_cost_usd", 0)

            our_tc = c1_results.get("token_cost", {})
            our_input = (our_tc.get("total_input_tokens") or {}).get("value", 0)
            our_output = (our_tc.get("total_output_tokens") or {}).get("value", 0)
            our_cost = (our_tc.get("total_cost") or {}).get("value", 0)

            if ta_prompt > 0 and our_input > 0:
                checks_run += 1
                # token_analysis.json may have different scope (different exclusion set)
                ratio = our_input / ta_prompt if ta_prompt > 0 else 0
                if ratio < 0.5 or ratio > 2.0:
                    warnings.append(
                        f"WARNING: Token count large discrepancy: token_analysis prompt={ta_prompt}, "
                        f"ours={our_input} (ratio={ratio:.2f}). "
                        f"Note: different KNOWN_FAIL exclusion scope."
                    )
                elif abs(ratio - 1.0) > 0.1:
                    warnings.append(
                        f"INFO: Token count minor diff: token_analysis prompt={ta_prompt}, "
                        f"ours={our_input} (ratio={ratio:.2f}). "
                        f"Expected due to different KNOWN_FAIL exclusion scope."
                    )
            if ta_completion > 0 and our_output > 0:
                checks_run += 1
                ratio_out = our_output / ta_completion
                if abs(ratio_out - 1.0) > 0.1:
                    warnings.append(
                        f"INFO: Output token minor diff: token_analysis completion={ta_completion}, "
                        f"ours={our_output} (ratio={ratio_out:.2f}). "
                        f"Expected due to different KNOWN_FAIL exclusion scope."
                    )
            if ta_cost > 0 and our_cost > 0:
                checks_run += 1
                ratio_cost = our_cost / ta_cost
                if abs(ratio_cost - 1.0) > 0.1:
                    warnings.append(
                        f"INFO: Cost minor diff: token_analysis cost=${ta_cost:.2f}, "
                        f"ours=${our_cost:.2f} (ratio={ratio_cost:.2f}). "
                        f"Expected due to different KNOWN_FAIL exclusion scope."
                    )

            if verbose:
                print(
                    f"  Cross-check: token_analysis.json prompt={ta_prompt}, "
                    f"completion={ta_completion}, cost=${ta_cost:.2f}"
                )

        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"WARNING: Could not read token_analysis.json: {e}")

    return {
        "checks_run": checks_run,
        "warnings": warnings,
        "status": "pass" if not any(w.startswith("WARNING") for w in warnings) else "warn",
    }


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def write_markdown(output: dict, path: Path) -> None:
    """Write quantitative findings as markdown."""
    lines = []
    lines.append("# Quantitative Findings — NeurIPS 2026 ParBench")
    lines.append("")
    lines.append(f"Generated: {output.get('metadata', {}).get('generated_at', 'unknown')}")
    lines.append(f"Git hash: {output.get('metadata', {}).get('git_hash', 'unknown')}")
    lines.append("")

    meta = output.get("metadata", {})
    fc = meta.get("file_counts", {})
    lines.append("## File Counts")
    lines.append("")
    lines.append(f"- Total on disk: {fc.get('total_on_disk', '?')}")
    lines.append(f"- Excluded (KNOWN_FAIL, {len(EXCLUDED_SPECS)} specs): {fc.get('excluded_known_fail', '?')}")
    lines.append(f"- Valid after exclusion: {fc.get('valid_after_exclusion', '?')}")
    lines.append(f"- Canonical (temp=0.7): {fc.get('canonical_valid', '?')}")
    lines.append("")

    # Legacy empty section (temp=0.0 was never run)
    c1 = {}
    lines.append("---")
    lines.append("")
    lines.append("## Legacy (temperature=0.0) — No Data")
    lines.append("")

    # Dimension 1: Aggregate pass rates
    agg = c1.get("aggregate_pass_rates", {})
    lines.append("### Dimension 1: Aggregate Pass Rates")
    lines.append("")
    overall = agg.get("overall", {})
    lines.append(f"**Overall:** {_pct(overall.get('value'))} "
                 f"[{_pct(overall.get('ci_lower'))}, {_pct(overall.get('ci_upper'))}] "
                 f"(n={overall.get('n', '?')})")
    lines.append("")

    per_suite = agg.get("per_suite", {})
    if per_suite:
        lines.append("| Suite | Pass Rate | 95% CI | n |")
        lines.append("|-------|-----------|--------|---|")
        for suite, data in sorted(per_suite.items()):
            lines.append(f"| {suite} | {_pct(data.get('value'))} | "
                         f"[{_pct(data.get('ci_lower'))}, {_pct(data.get('ci_upper'))}] | "
                         f"{data.get('n', '?')} |")
        lines.append("")

    # Dimension 2: Per-direction pass rates
    dir_rates = c1.get("direction_pass_rates", {})
    lines.append("### Dimension 2: Per-Direction Pass Rates (L0 only)")
    lines.append("")

    std = dir_rates.get("standard", {})
    if std:
        lines.append("**Standard directions:**")
        lines.append("")
        lines.append("| Direction | Pass Rate | 95% CI | n |")
        lines.append("|-----------|-----------|--------|---|")
        for d, data in sorted(std.items()):
            lines.append(f"| {d} | {_pct(data.get('value'))} | "
                         f"[{_pct(data.get('ci_lower'))}, {_pct(data.get('ci_upper'))}] | "
                         f"{data.get('n', '?')} |")
        lines.append("")

    cs = dir_rates.get("case_study", {})
    if cs:
        lines.append("**Case study directions (omp_target):**")
        lines.append("")
        lines.append("| Direction | Pass Rate | 95% CI | n |")
        lines.append("|-----------|-----------|--------|---|")
        for d, data in sorted(cs.items()):
            lines.append(f"| {d} | {_pct(data.get('value'))} | "
                         f"[{_pct(data.get('ci_lower'))}, {_pct(data.get('ci_upper'))}] | "
                         f"{data.get('n', '?')} |")
        lines.append("")

    # Observation
    if std:
        rates = [(d, data.get("value", 0)) for d, data in std.items()]
        rates.sort(key=lambda x: x[1], reverse=True)
        if len(rates) >= 2:
            best_d, best_r = rates[0]
            worst_d, worst_r = rates[-1]
            ratio = best_r / worst_r if worst_r > 0 else float("inf")
            lines.append(f"*Observation: {best_d} has {ratio:.1f}x higher pass rate than {worst_d}.*")
            lines.append("")

    # Dimension 3: Direction asymmetry
    asym = c1.get("direction_asymmetry", {})
    lines.append("### Dimension 3: Direction Asymmetry (McNemar, L0)")
    lines.append("")
    if asym:
        lines.append("| Pair | Fwd Rate | Rev Rate | p-value | Cohen's h | Effect | Sig? |")
        lines.append("|------|----------|----------|---------|-----------|--------|------|")
        for pair_name, finding in sorted(asym.items()):
            tr = finding.get("value", {}).get("test_result", {}) if isinstance(finding.get("value"), dict) else {}
            lines.append(
                f"| {pair_name} | {_pct(tr.get('forward_pass_rate'))} | "
                f"{_pct(tr.get('reverse_pass_rate'))} | "
                f"{tr.get('p_value', '?')} | {tr.get('cohens_h', '?')} | "
                f"{tr.get('effect_size', '?')} | "
                f"{'Yes' if tr.get('significant') else 'No'} |"
            )
        lines.append("")
    else:
        lines.append("No direction pairs found for McNemar test.")
        lines.append("")

    # Dimension 4: Augmentation trends
    aug = c1.get("augmentation_trends", {})
    lines.append("### Dimension 4: Augmentation Trends")
    lines.append("")

    agg_trend = aug.get("aggregate", {})
    if agg_trend:
        ca = agg_trend.get("cochran_armitage", {})
        lines.append(f"**Aggregate Cochran-Armitage:** z={ca.get('z', '?')}, "
                     f"p={ca.get('p_value', '?')}, "
                     f"trend={ca.get('trend_direction', '?')}, "
                     f"significant={'Yes' if ca.get('significant') else 'No'}")
        lines.append("")

        per_level = agg_trend.get("per_level", {})
        if per_level:
            lines.append("| Level | Pass Rate | 95% CI | n |")
            lines.append("|-------|-----------|--------|---|")
            for lv_key in sorted(per_level.keys()):
                data = per_level[lv_key]
                lines.append(f"| {lv_key} | {_pct(data.get('value'))} | "
                             f"[{_pct(data.get('ci_lower'))}, {_pct(data.get('ci_upper'))}] | "
                             f"{data.get('n', '?')} |")
            lines.append("")

        cohens_adj = agg_trend.get("cohens_h_adjacent", {})
        if cohens_adj:
            lines.append("**Cohen's h (adjacent levels):**")
            for pair, h_val in sorted(cohens_adj.items()):
                lines.append(f"- {pair}: h={h_val}")
            lines.append("")

    # Dimension 5: Failure taxonomy
    tax = c1.get("failure_taxonomy", {})
    lines.append("### Dimension 5: Failure Taxonomy")
    lines.append("")

    sc = tax.get("status_counts", {})
    if sc:
        lines.append("| Status | Count | % |")
        lines.append("|--------|-------|---|")
        total_recs = tax.get("total_records", 1)
        for status in ["PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL", "ERROR"]:
            count = sc.get(status, 0)
            pct = count / total_recs * 100 if total_recs > 0 else 0
            lines.append(f"| {status} | {count} | {pct:.1f}% |")
        lines.append("")

    top3 = tax.get("top_3_build_subcategories", [])
    if top3:
        lines.append("**Top-3 BUILD_FAIL subcategories:**")
        lines.append("")
        lines.append("| Subcategory | Count |")
        lines.append("|-------------|-------|")
        for entry in top3:
            lines.append(f"| {entry['subcategory']} | {entry['count']} |")
        lines.append("")

    # Dimension 8: Per-kernel difficulty tiers
    tiers = c1.get("per_kernel_tiers", {})
    lines.append("### Dimension 8: Per-Kernel Difficulty Tiers (L0)")
    lines.append("")
    lines.append(f"**Total kernels:** {tiers.get('n_kernels', '?')}")
    lines.append("")

    kernel_list = tiers.get("kernels", [])
    if kernel_list:
        lines.append("| Rank | Kernel | Suite | Pass Rate | 95% CI | Passes/Total | Tier |")
        lines.append("|------|--------|-------|-----------|--------|--------------|------|")
        for i, ks in enumerate(kernel_list, 1):
            lines.append(
                f"| {i} | {ks['kernel']} | {ks['suite']} | "
                f"{_pct(ks.get('pass_rate'))} | "
                f"[{_pct(ks.get('ci_lower'))}, {_pct(ks.get('ci_upper'))}] | "
                f"{ks.get('passes', 0)}/{ks.get('total', 0)} | {ks.get('tier', '?')} |"
            )
        lines.append("")

    top5 = tiers.get("top_5_easiest", [])
    bot5 = tiers.get("top_5_hardest", [])
    if top5:
        top5_str = ", ".join(f"{k['kernel']} ({_pct(k['pass_rate'])})" for k in top5)
        lines.append(f"**Top-5 easiest:** {top5_str}")
    if bot5:
        bot5_str = ", ".join(f"{k['kernel']} ({_pct(k['pass_rate'])})" for k in bot5)
        lines.append(f"**Top-5 hardest:** {bot5_str}")
    lines.append("")

    anomalies = tiers.get("direction_anomalies", [])
    if anomalies:
        lines.append("**Direction anomalies (>50pp gap):**")
        lines.append("")
        for a in anomalies:
            lines.append(
                f"- {a['kernel']}: {a['best_direction']}={_pct(a['best_rate'])} vs "
                f"{a['worst_direction']}={_pct(a['worst_rate'])} ({a['gap_pp']}pp gap)"
            )
        lines.append("")

    # Dimension 9: Complexity correlation
    cc = c1.get("complexity_correlation", {})
    lines.append("### Dimension 9: Translation Complexity Correlation")
    lines.append("")
    pc = cc.get("per_class", {})
    if pc:
        lines.append("| Complexity Class | Pass Rate | 95% CI | Passes/Total |")
        lines.append("|------------------|-----------|--------|--------------|")
        for cls_name in ["single_file", "multi_to_single", "single_to_multi", "multi_to_multi"]:
            data = pc.get(cls_name, {})
            if data.get("total", 0) > 0:
                lines.append(
                    f"| {cls_name} | {_pct(data.get('value'))} | "
                    f"[{_pct(data.get('ci_lower'))}, {_pct(data.get('ci_upper'))}] | "
                    f"{data.get('passes', 0)}/{data.get('total', 0)} |"
                )
        lines.append("")

    tr = cc.get("test_result", {})
    if tr:
        method = tr.get("method", "?")
        p_val = tr.get("p_value", "?")
        sig = "Yes" if tr.get("significant") else "No"
        lines.append(f"**Statistical test:** {method}, p={p_val}, significant={sig}")
        lines.append("")

    # Dimension 10: Cross-suite comparison
    cs_data = c1.get("cross_suite", {})
    lines.append("### Dimension 10: Cross-Suite Comparison (L0)")
    lines.append("")

    cs_rates = cs_data.get("per_suite_pass_rate_l0", {})
    cs_sloc = cs_data.get("sloc_characteristics", {})
    cs_mf = cs_data.get("multi_file_fraction", {})
    if cs_rates:
        lines.append("| Suite | Pass Rate (L0) | 95% CI | n | Mean SLoC | Multi-File % |")
        lines.append("|-------|----------------|--------|---|-----------|-------------|")
        for suite_name, rate_data in sorted(cs_rates.items()):
            sloc_info = cs_sloc.get(suite_name, {})
            mf_info = cs_mf.get(suite_name, {})
            lines.append(
                f"| {suite_name} | {_pct(rate_data.get('value'))} | "
                f"[{_pct(rate_data.get('ci_lower'))}, {_pct(rate_data.get('ci_upper'))}] | "
                f"{rate_data.get('n', '?')} | "
                f"{sloc_info.get('mean_sloc', '?')} | "
                f"{_pct(mf_info.get('fraction'))} |"
            )
        lines.append("")

    # Dimension 11: Token cost
    tc = c1.get("token_cost", {})
    lines.append("### Dimension 11: Token Cost Analysis")
    lines.append("")
    tc_total = tc.get("total_cost", {})
    tc_per_task = tc.get("cost_per_task", {})
    tc_per_pass = tc.get("cost_per_pass", {})
    total_cost_val = tc_total.get("value") if isinstance(tc_total, dict) else tc_total
    per_task_val = tc_per_task.get("value") if isinstance(tc_per_task, dict) else tc_per_task
    per_pass_val = tc_per_pass.get("value") if isinstance(tc_per_pass, dict) else tc_per_pass
    lines.append(f"- **Total cost:** ${total_cost_val:.2f}" if total_cost_val else "- **Total cost:** ?")
    lines.append(f"- **Cost per task:** ${per_task_val:.4f}" if per_task_val else "- **Cost per task:** ?")
    lines.append(f"- **Cost per PASS:** ${per_pass_val:.4f}" if per_pass_val else "- **Cost per PASS:** ?")
    lines.append(f"- **Tasks with tokens:** {tc.get('tasks_with_tokens', '?')}")
    lines.append("")

    tc_suite = tc.get("per_suite", {})
    if tc_suite:
        lines.append("| Suite | Input Tokens | Output Tokens | Cost | Tasks | Cost/Task |")
        lines.append("|-------|-------------|---------------|------|-------|-----------|")
        for s, sd in sorted(tc_suite.items()):
            lines.append(
                f"| {s} | {sd.get('input_tokens', 0):,} | {sd.get('output_tokens', 0):,} | "
                f"${sd.get('total_cost', 0):.2f} | {sd.get('tasks', 0)} | "
                f"${sd.get('cost_per_task', 0):.4f} |"
            )
        lines.append("")

    # Dimension 12: SLoC correlation
    sloc_c = c1.get("sloc_correlation", {})
    lines.append("### Dimension 12: SLoC Correlation")
    lines.append("")
    sp_finding = sloc_c.get("spearman", {})
    pe_finding = sloc_c.get("pearson", {})
    sp_val = sp_finding.get("value", {}) if isinstance(sp_finding, dict) else {}
    pe_val = pe_finding.get("value", {}) if isinstance(pe_finding, dict) else {}
    if isinstance(sp_val, dict):
        lines.append(f"- **Spearman:** rho={sp_val.get('rho', '?')}, p={sp_val.get('p_value', '?')} "
                     f"({'significant' if sloc_c.get('significant_spearman') else 'not significant'})")
    if isinstance(pe_val, dict):
        lines.append(f"- **Pearson:** r={pe_val.get('r', '?')}, p={pe_val.get('p_value', '?')} "
                     f"({'significant' if sloc_c.get('significant_pearson') else 'not significant'})")
    lines.append(f"- **Interpretation:** {sloc_c.get('interpretation', '?')}")
    lines.append(f"- **n kernels:** {sloc_c.get('n_kernels', '?')}")
    lines.append("")

    # Dimension 13: OpenCL kernel-only effect
    ocl = c1.get("opencl_kernel_only_effect", {})
    lines.append("### Dimension 13: OpenCL Kernel-Only Effect (L0)")
    lines.append("")
    ocl_rate = ocl.get("x_to_opencl", {})
    omp_rate = ocl.get("x_to_omp", {})
    lines.append(
        f"- **X-to-OpenCL (kernel-only):** {_pct(ocl_rate.get('value'))} "
        f"[{_pct(ocl_rate.get('ci_lower'))}, {_pct(ocl_rate.get('ci_upper'))}] "
        f"(n={ocl_rate.get('n', '?')})"
    )
    lines.append(
        f"- **X-to-OMP (full program):** {_pct(omp_rate.get('value'))} "
        f"[{_pct(omp_rate.get('ci_lower'))}, {_pct(omp_rate.get('ci_upper'))}] "
        f"(n={omp_rate.get('n', '?')})"
    )
    ocl_test = ocl.get("test_result", {})
    if ocl_test:
        lines.append(
            f"- **Fisher's exact:** p={ocl_test.get('p_value', '?')}, "
            f"OR={ocl_test.get('odds_ratio', '?')}, "
            f"Cohen's h={ocl_test.get('cohens_h', '?')}, "
            f"significant={'Yes' if ocl_test.get('significant') else 'No'}"
        )
    lines.append("")

    # ---- Canonical Evaluation ----
    c2 = output.get("canonical", {})
    lines.append("---")
    lines.append("")
    lines.append("## Canonical Evaluation (temperature=0.7)")
    lines.append("")
    c2_agg = c2.get("aggregate_pass_rates", {})
    c2_overall = c2_agg.get("overall", {})
    lines.append(f"**Overall:** {_pct(c2_overall.get('value'))} "
                 f"[{_pct(c2_overall.get('ci_lower'))}, {_pct(c2_overall.get('ci_upper'))}] "
                 f"(n={c2_overall.get('n', '?')})")
    lines.append("")

    # Dimension 7: pass@k
    passk = c2.get("pass_at_k", {})
    lines.append("### Dimension 7: pass@k Estimates")
    lines.append("")
    p1_data = passk.get("pass_at_1", {})
    p3_data = passk.get("pass_at_3", {})
    total_tasks_finding = passk.get("total_tasks", {})
    total_tasks_val = total_tasks_finding.get("value") if isinstance(total_tasks_finding, dict) else total_tasks_finding
    lines.append(f"**Total tasks:** {total_tasks_val}")
    lines.append(
        f"- **pass@1** (single-sample success rate): {_pct(p1_data.get('value'))} "
        f"[{_pct(p1_data.get('ci_lower'))}, {_pct(p1_data.get('ci_upper'))}]"
    )
    lines.append(
        f"- **pass@3** (at least 1 of 3 passes): {_pct(p3_data.get('value'))} "
        f"[{_pct(p3_data.get('ci_lower'))}, {_pct(p3_data.get('ci_upper'))}]"
    )
    lines.append("")

    tc_class = passk.get("task_classification", {})
    if tc_class:
        lines.append(
            f"**Task classification:** {tc_class.get('always_pass', 0)} always pass, "
            f"{tc_class.get('noisy_fail', 0)} noisy fail, "
            f"{tc_class.get('hard_fail', 0)} hard fail"
        )
        lines.append("")

    # Per-direction pass@k
    pd_passk = passk.get("per_direction", {})
    pd_p1 = pd_passk.get("pass_at_1", {})
    pd_p3 = pd_passk.get("pass_at_3", {})
    if pd_p1:
        lines.append("**Per-direction pass@k:**")
        lines.append("")
        lines.append("| Direction | pass@1 | pass@3 | n |")
        lines.append("|-----------|--------|--------|---|")
        for d in sorted(pd_p1.keys()):
            p1d = pd_p1.get(d, {})
            p3d = pd_p3.get(d, {})
            lines.append(
                f"| {d} | {_pct(p1d.get('value'))} | {_pct(p3d.get('value'))} | "
                f"{p1d.get('n', '?')} |"
            )
        lines.append("")

    # Per-suite pass@k
    ps_passk = passk.get("per_suite", {})
    ps_p1 = ps_passk.get("pass_at_1", {})
    ps_p3 = ps_passk.get("pass_at_3", {})
    if ps_p1:
        lines.append("**Per-suite pass@k:**")
        lines.append("")
        lines.append("| Suite | pass@1 | pass@3 | n |")
        lines.append("|-------|--------|--------|---|")
        for s in sorted(ps_p1.keys()):
            p1s = ps_p1.get(s, {})
            p3s = ps_p3.get(s, {})
            lines.append(
                f"| {s} | {_pct(p1s.get('value'))} | {_pct(p3s.get('value'))} | "
                f"{p1s.get('n', '?')} |"
            )
        lines.append("")

    # ---- Paper Claims ----
    pc_list = output.get("paper_claims", [])
    if pc_list:
        lines.append("---")
        lines.append("")
        lines.append("## Paper Claims Mapping")
        lines.append("")
        lines.append("| # | Claim ID | Scope | Display Value | Paper Location |")
        lines.append("|---|----------|-------|---------------|----------------|")
        for i, claim in enumerate(pc_list, 1):
            lines.append(
                f"| {i} | {claim.get('claim_id', '?')} | {claim.get('scope', '?')} | "
                f"{claim.get('display_value', '?')} | {claim.get('paper_location', '?')} |"
            )
        lines.append("")

    # Cross-check summary
    xc = output.get("cross_check", {})
    if xc:
        lines.append("---")
        lines.append("")
        lines.append("## Cross-Check Results")
        lines.append("")
        lines.append(f"Checks run: {xc.get('checks_run', 0)}")
        lines.append(f"Status: {xc.get('status', 'unknown')}")
        xc_warnings = xc.get("warnings", [])
        if xc_warnings:
            lines.append("")
            for w in xc_warnings:
                lines.append(f"- {w}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(val) -> str:
    """Format a decimal as percentage string."""
    if val is None:
        return "?"
    return f"{val * 100:.1f}%"


# ---------------------------------------------------------------------------
# Validation (--validate)
# ---------------------------------------------------------------------------


def run_validation(output: dict, project_root: Path, verbose: bool, model_dir: str = "together-qwen-3.5-397b-a17b") -> dict:
    """Validate computed findings via spot-checks, cross-checks, consistency, and paper claims.

    Uses INDEPENDENT code paths (not the main computation functions) to verify
    that computed values match what raw files on disk actually contain.

    Returns a validation report dict and writes it to
    results/analysis/quantitative_findings_validation.json.
    """
    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spot_checks": [],
        "cross_checks": [],
        "consistency_checks": [],
        "paper_claims_audit": [],
        "summary": {"total": 0, "passed": 0, "failed": 0, "warnings": 0},
    }

    results_dir = project_root / "results" / "evaluation" / model_dir
    analysis_dir = project_root / "results" / "analysis"
    metadata = output.get("metadata", {})
    file_counts = metadata.get("file_counts", {})
    c1 = {}
    c2 = output.get("canonical", {})

    # -----------------------------------------------------------------------
    # Category 1: Spot-checks (independent file counting)
    # -----------------------------------------------------------------------

    def _spot(name: str, expected, actual, tolerance: float = 0.0) -> dict:
        if tolerance > 0 and isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            ok = abs(expected - actual) <= tolerance
        else:
            ok = expected == actual
        entry = {
            "check": name,
            "expected": expected,
            "actual": actual,
            "status": "PASS" if ok else "FAIL",
        }
        if tolerance > 0:
            entry["tolerance"] = tolerance
        return entry

    # --- Spot-check 1: Total file count ---
    # Independently count *.json minus non-result files
    all_jsons = list(results_dir.glob("*.json"))
    raw_count = 0
    for jp in all_jsons:
        bn = jp.name
        if bn.startswith("batch_") or bn == "eval_summary.json":
            continue
        raw_count += 1
    results["spot_checks"].append(
        _spot("total_file_count", file_counts.get("total_on_disk"), raw_count)
    )

    # --- Spot-check 2: Legacy count ---
    # Independently load each file, check temperature and KNOWN_FAIL exclusion
    c1_count = 0
    c2_count = 0
    kf_excluded_count = 0
    pass_count_independent = 0
    suite_c1_counts: dict[str, dict[str, int]] = {}  # {suite: {"total": N, "pass": N}}
    status_counter: dict[str, int] = {}
    level_counter: dict[str, int] = {}
    direction_set: set[str] = set()
    bf_count_independent = 0

    for jp in sorted(results_dir.glob("*.json")):
        bn = jp.name
        if bn.startswith("batch_") or bn == "eval_summary.json":
            continue
        try:
            with open(jp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        src_spec = data.get("source_spec", "")
        tgt_spec = data.get("target_spec", "")
        if src_spec in EXCLUDED_SPECS or tgt_spec in EXCLUDED_SPECS:
            kf_excluded_count += 1
            continue

        temp = data.get("temperature")
        if temp is None:
            temp = 0.0

        overall_status = data.get("overall_status", "")
        stem = jp.stem
        aug_level = 0
        m = re.search(r"-L(\d+)(?:-s\d+)?$", stem)
        if m:
            aug_level = int(m.group(1))

        # Determine suite independently
        suite = src_spec.split("-")[0] if src_spec else "unknown"

        # Determine direction independently
        src_api = src_spec.rsplit("-", 1)[-1] if src_spec else "unknown"
        tgt_api = tgt_spec.rsplit("-", 1)[-1] if tgt_spec else "unknown"
        direction = f"{src_api}-to-{tgt_api}"

        if temp == 0.0:
            c1_count += 1
            level_key = f"L{aug_level}"
            level_counter[level_key] = level_counter.get(level_key, 0) + 1

            if aug_level == 0:
                direction_set.add(direction)

            # Status counting
            status_counter[overall_status] = status_counter.get(overall_status, 0) + 1

            if overall_status == "PASS":
                pass_count_independent += 1
            if overall_status == "BUILD_FAIL":
                bf_count_independent += 1

            if suite not in suite_c1_counts:
                suite_c1_counts[suite] = {"total": 0, "pass": 0}
            suite_c1_counts[suite]["total"] += 1
            if overall_status == "PASS":
                suite_c1_counts[suite]["pass"] += 1
        else:
            c2_count += 1

    # campaign_1 (temp=0.0) was never run — skip its spot-check

    # --- Spot-check 3: Canonical count ---
    results["spot_checks"].append(
        _spot("canonical_count", file_counts.get("canonical_valid"), c2_count)
    )

    # --- Spot-check 4: KNOWN_FAIL exclusion count ---
    results["spot_checks"].append(
        _spot("known_fail_exclusion_count", file_counts.get("excluded_known_fail"), kf_excluded_count)
    )

    # --- Spot-check 5: Rodinia C1 pass count ---
    rod_sub = c1.get("rodinia_subset", {})
    rod_passes_expected = rod_sub.get("passes", 0)
    rodinia_c1_pass = suite_c1_counts.get("rodinia", {}).get("pass", 0)
    results["spot_checks"].append(
        _spot("rodinia_c1_pass_count", rod_passes_expected, rodinia_c1_pass)
    )

    # --- Spot-check 6: Overall C1 pass rate ---
    agg = c1.get("aggregate_pass_rates", {}).get("overall", {})
    expected_rate = agg.get("value", 0)
    actual_rate = pass_count_independent / c1_count if c1_count > 0 else 0
    results["spot_checks"].append(
        _spot("overall_c1_pass_rate", expected_rate, round(actual_rate, 4), tolerance=0.001)
    )

    # --- Spot-check 7: Direction count check ---
    # L0 standard + case_study directions
    dp = c1.get("direction_pass_rates", {})
    std_count = len(dp.get("standard", {}))
    cs_count = len(dp.get("case_study", {}))
    expected_dir_count = std_count + cs_count
    actual_dir_count = len(direction_set)
    results["spot_checks"].append(
        _spot("direction_count", expected_dir_count, actual_dir_count)
    )

    # --- Spot-check 8: Per-suite file count sum ---
    ps = c1.get("aggregate_pass_rates", {}).get("per_suite", {})
    expected_suite_sum = sum(v.get("n", 0) for v in ps.values())
    actual_suite_sum = sum(suite_c1_counts.values())
    results["spot_checks"].append(
        _spot("per_suite_count_sum", expected_suite_sum, actual_suite_sum)
    )

    # --- Spot-check 9: Augmentation level distribution ---
    expected_level_total = 0  # temp=0.0 was never run
    actual_level_total = sum(level_counter.values())
    results["spot_checks"].append(
        _spot("augmentation_level_sum", expected_level_total, actual_level_total)
    )

    # --- Spot-check 10: Canonical seed count ---
    # Group C2 files by (source_spec, target_spec) and check each has exactly 3 seeds
    c2_groups: dict[str, int] = defaultdict(int)
    for jp in sorted(results_dir.glob("*.json")):
        bn = jp.name
        if bn.startswith("batch_") or bn == "eval_summary.json":
            continue
        try:
            with open(jp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        src_spec = data.get("source_spec", "")
        tgt_spec = data.get("target_spec", "")
        if src_spec in EXCLUDED_SPECS or tgt_spec in EXCLUDED_SPECS:
            continue
        temp = data.get("temperature")
        if temp is None:
            temp = 0.0
        if temp > 0:
            key = f"{src_spec}|{tgt_spec}"
            c2_groups[key] += 1

    all_have_3 = all(v == 3 for v in c2_groups.values())
    c2_task_count = len(c2_groups)
    pk = c2.get("pass_at_k", {})
    pk_total_tasks = pk.get("total_tasks", {})
    pk_expected = pk_total_tasks.get("value") if isinstance(pk_total_tasks, dict) else pk_total_tasks
    results["spot_checks"].append({
        "check": "canonical_seed_count",
        "expected": f"{pk_expected} tasks, all with 3 seeds",
        "actual": f"{c2_task_count} tasks, all_have_3={all_have_3}",
        "status": "PASS" if (c2_task_count == pk_expected and all_have_3) else "FAIL",
    })

    # --- Spot-check 11: BUILD_FAIL count ---
    ft = c1.get("failure_taxonomy", {})
    ft_sc = ft.get("status_counts", {})
    expected_bf = ft_sc.get("BUILD_FAIL", 0)
    results["spot_checks"].append(
        _spot("build_fail_count", expected_bf, bf_count_independent)
    )

    if verbose:
        print(f"  {len(results['spot_checks'])} spot-checks completed")

    # -----------------------------------------------------------------------
    # Category 2: Cross-checks against existing analysis files
    # -----------------------------------------------------------------------

    def _xcheck(
        name: str, our_value, their_value, scope_note: str = "", tolerance: float = 0.001,
    ) -> dict:
        """Create a cross-check entry. Uses DIFFERENT_SCOPE for expected differences."""
        if our_value is None or their_value is None:
            return {
                "check": name,
                "our_value": our_value,
                "their_value": their_value,
                "discrepancy_pct": None,
                "status": "WARNING",
                "note": f"One or both values are None. {scope_note}",
            }
        if isinstance(our_value, (int, float)) and isinstance(their_value, (int, float)):
            base = max(abs(their_value), 1e-9)
            disc_pct = abs(our_value - their_value) / base * 100
        else:
            disc_pct = 0.0 if our_value == their_value else 100.0

        if scope_note:
            status = "DIFFERENT_SCOPE"
        elif disc_pct <= tolerance * 100:
            status = "MATCH"
        else:
            status = "WARNING"

        return {
            "check": name,
            "our_value": our_value,
            "their_value": their_value,
            "discrepancy_pct": round(disc_pct, 4),
            "status": status,
            "note": scope_note if scope_note else ("exact match" if disc_pct == 0 else f"diff {disc_pct:.4f}%"),
        }

    # --- Cross-check vs paper_data.json ---
    pd_path = analysis_dir / "paper_data.json"
    if pd_path.exists():
        try:
            pd = json.loads(pd_path.read_text())
            # paper_data.json uses 6 excluded specs, we use 8 -> DIFFERENT_SCOPE
            pd_rate = (pd.get("primary_campaign") or {}).get("overall", {}).get("pass_rate")
            our_rate = agg.get("value")
            results["cross_checks"].append(
                _xcheck(
                    "vs_paper_data_overall_rate",
                    our_rate, pd_rate,
                    scope_note="paper_data uses 6 KNOWN_FAIL exclusions, we use 8",
                )
            )

            # Per-direction comparison
            pd_dirs = (pd.get("primary_campaign") or {}).get("by_direction", {})
            for dname, ddata in pd_dirs.items():
                pd_dir_rate = ddata.get("pass_rate")
                our_dir_data = dp.get("standard", {}).get(dname) or dp.get("case_study", {}).get(dname)
                our_dir_rate = our_dir_data.get("value") if isinstance(our_dir_data, dict) else None
                if pd_dir_rate is not None and our_dir_rate is not None:
                    results["cross_checks"].append(
                        _xcheck(
                            f"vs_paper_data_dir_{dname}",
                            our_dir_rate, pd_dir_rate,
                            scope_note="paper_data uses 6 KNOWN_FAIL exclusions, we use 8",
                        )
                    )
        except (json.JSONDecodeError, OSError):
            results["cross_checks"].append({
                "check": "vs_paper_data_json",
                "status": "WARNING",
                "note": "Could not read paper_data.json",
            })

    # --- Cross-check vs paper_data_rodinia.json ---
    pdr_path = analysis_dir / "paper_data_rodinia.json"
    if pdr_path.exists():
        try:
            pdr = json.loads(pdr_path.read_text())
            pdr_rate = (pdr.get("primary_campaign") or {}).get("overall", {}).get("pass_rate")
            rod_rate_val = rod_sub.get("pass_rate", {}).get("value") if isinstance(rod_sub.get("pass_rate"), dict) else None
            results["cross_checks"].append(
                _xcheck(
                    "vs_paper_data_rodinia_rate",
                    rod_rate_val, pdr_rate,
                    scope_note="paper_data_rodinia uses 6 Rodinia KNOWN_FAIL exclusions, we use 6 Rodinia + 2 HeCBench (Rodinia subset matches)",
                )
            )
        except (json.JSONDecodeError, OSError):
            results["cross_checks"].append({
                "check": "vs_paper_data_rodinia_json",
                "status": "WARNING",
                "note": "Could not read paper_data_rodinia.json",
            })

    # --- Cross-check vs statistical_analysis.json ---
    stat_path = analysis_dir / "statistical_analysis.json"
    if stat_path.exists():
        try:
            sa = json.loads(stat_path.read_text())
            # Compare Cochran-Armitage z and p values
            ca_data = c1.get("augmentation_trends", {}).get("aggregate", {}).get("cochran_armitage", {})
            sa_ca = sa.get("cochran_armitage", {})
            if ca_data and sa_ca:
                results["cross_checks"].append(
                    _xcheck(
                        "vs_stat_cochran_z",
                        ca_data.get("z"), sa_ca.get("z"),
                        scope_note="statistical_analysis uses 6 KNOWN_FAIL exclusions, we use 8",
                    )
                )
                results["cross_checks"].append(
                    _xcheck(
                        "vs_stat_cochran_p",
                        ca_data.get("p_value"), sa_ca.get("p_value"),
                        scope_note="statistical_analysis uses 6 KNOWN_FAIL exclusions, we use 8",
                    )
                )

            # Wilson CIs comparison
            sa_wc = sa.get("wilson_cis", {}).get("by_model", {})
            for model_name, wc_data in sa_wc.items():
                sa_overall_rate = wc_data.get("rate")
                results["cross_checks"].append(
                    _xcheck(
                        "vs_stat_wilson_overall_rate",
                        agg.get("value"), sa_overall_rate,
                        scope_note="statistical_analysis uses 6 KNOWN_FAIL exclusions, we use 8",
                    )
                )
                break  # Only one model

        except (json.JSONDecodeError, OSError):
            results["cross_checks"].append({
                "check": "vs_statistical_analysis_json",
                "status": "WARNING",
                "note": "Could not read statistical_analysis.json",
            })

    # --- Cross-check vs error_taxonomy.json ---
    et_path = analysis_dir / "error_taxonomy.json"
    if et_path.exists():
        try:
            et = json.loads(et_path.read_text())
            # error_taxonomy includes all records (no KF exclusion)
            et_bf = et.get("status_counts", {}).get("BUILD_FAIL", 0)
            our_bf = ft_sc.get("BUILD_FAIL", 0)
            results["cross_checks"].append(
                _xcheck(
                    "vs_error_taxonomy_build_fail",
                    our_bf, et_bf,
                    scope_note=f"error_taxonomy includes ALL records from the model dir; we exclude {len(EXCLUDED_SPECS)} KF specs",
                )
            )
        except (json.JSONDecodeError, OSError):
            results["cross_checks"].append({
                "check": "vs_error_taxonomy_json",
                "status": "WARNING",
                "note": "Could not read error_taxonomy.json",
            })

    # --- Cross-check vs token_analysis.json ---
    tk_path = analysis_dir / "token_analysis.json"
    if tk_path.exists():
        try:
            ta = json.loads(tk_path.read_text())
            ta_cost = ta.get("grand_total_cost_usd", 0)
            tc = c1.get("token_cost", {})
            our_cost = (tc.get("total_cost") or {}).get("value", 0)
            results["cross_checks"].append(
                _xcheck(
                    "vs_token_analysis_cost",
                    our_cost, ta_cost,
                    scope_note="token_analysis includes ALL records (no KF exclusion + both campaigns), we use C1-only with 8 KF exclusions",
                )
            )
        except (json.JSONDecodeError, OSError):
            results["cross_checks"].append({
                "check": "vs_token_analysis_json",
                "status": "WARNING",
                "note": "Could not read token_analysis.json",
            })

    if verbose:
        print(f"  {len(results['cross_checks'])} cross-checks completed")

    # -----------------------------------------------------------------------
    # Category 3: Internal consistency checks
    # -----------------------------------------------------------------------

    def _consist(name: str, ok: bool, detail: str = "") -> dict:
        return {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}

    # 1. Wilson CI bounds: ci_lower <= value <= ci_upper
    def _check_wilson_bounds(data: dict, path: str = "") -> list[dict]:
        checks = []
        if isinstance(data, dict):
            if "ci_lower" in data and "ci_upper" in data and "value" in data:
                v = data["value"]
                lo = data["ci_lower"]
                hi = data["ci_upper"]
                if isinstance(v, (int, float)) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    ok = lo <= v + 1e-9 and v <= hi + 1e-9
                    checks.append(_consist(
                        f"wilson_ci_bounds_{path}" if path else "wilson_ci_bounds",
                        ok,
                        f"ci_lower={lo} <= value={v} <= ci_upper={hi}" if ok
                        else f"VIOLATION: ci_lower={lo}, value={v}, ci_upper={hi}",
                    ))
            for k, sub in data.items():
                if isinstance(sub, dict):
                    checks.extend(_check_wilson_bounds(sub, f"{path}.{k}" if path else k))
        return checks

    wilson_checks = _check_wilson_bounds(c1)
    wilson_checks.extend(_check_wilson_bounds(c2))
    # Only include first few + any failures (avoid excessive entries)
    wilson_fails = [c for c in wilson_checks if c["status"] == "FAIL"]
    wilson_passes = [c for c in wilson_checks if c["status"] == "PASS"]
    results["consistency_checks"].append(
        _consist(
            "wilson_ci_bounds_all",
            len(wilson_fails) == 0,
            f"{len(wilson_passes)} CIs checked, {len(wilson_fails)} violations"
            + (f": {wilson_fails[0]['detail']}" if wilson_fails else ""),
        )
    )

    # 2. Per-suite counts sum to aggregate total
    suite_sum = sum(v.get("n", 0) for v in ps.values())
    overall_n = agg.get("n", 0)
    results["consistency_checks"].append(
        _consist(
            "per_suite_sum_matches_overall",
            suite_sum == overall_n,
            f"sum(per_suite.n)={suite_sum}, overall.n={overall_n}",
        )
    )

    # 3. No NaN or null in required numeric fields
    def _check_no_nan(data, path: str = "") -> list[str]:
        problems = []
        if isinstance(data, dict):
            for k, v in data.items():
                fp = f"{path}.{k}" if path else k
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    problems.append(fp)
                elif isinstance(v, dict):
                    problems.extend(_check_no_nan(v, fp))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        problems.extend(_check_no_nan(item, f"{fp}[{i}]"))
        return problems

    nan_problems = _check_no_nan(output)
    results["consistency_checks"].append(
        _consist(
            "no_nan_or_inf",
            len(nan_problems) == 0,
            f"{len(nan_problems)} NaN/Inf fields"
            + (f": {nan_problems[:3]}" if nan_problems else ""),
        )
    )

    # 4. Augmentation level counts: L0+L1+L2+L3+L4 match C1 total
    at_agg = c1.get("augmentation_trends", {}).get("aggregate", {}).get("per_level", {})
    level_sum = sum(v.get("n", 0) for v in at_agg.values())
    c1_total = 0  # temp=0.0 was never run
    results["consistency_checks"].append(
        _consist(
            "augmentation_level_counts_sum",
            level_sum == c1_total,
            f"sum(per_level.n)={level_sum}, C1_total={c1_total}",
        )
    )

    # 5. pass@k: pass_at_1 <= pass_at_3 (Chen et al. monotonicity)
    p1 = (pk.get("pass_at_1") or {}).get("value")
    p3 = (pk.get("pass_at_3") or {}).get("value")
    if p1 is not None and p3 is not None:
        results["consistency_checks"].append(
            _consist(
                "pass_at_1_lte_pass_at_3",
                p1 <= p3 + 1e-9,
                f"pass@1={p1}, pass@3={p3}",
            )
        )

    # 6. Token cost: total_cost == input_cost + output_cost (within rounding)
    tc = c1.get("token_cost", {})
    tc_total = (tc.get("total_cost") or {}).get("value", 0) or 0
    tc_input_tokens = (tc.get("total_input_tokens") or {}).get("value", 0) or 0
    tc_output_tokens = (tc.get("total_output_tokens") or {}).get("value", 0) or 0
    recomputed_cost = round(
        tc_input_tokens * TOGETHER_INPUT_PRICE_PER_M / 1_000_000
        + tc_output_tokens * TOGETHER_OUTPUT_PRICE_PER_M / 1_000_000,
        4,
    )
    results["consistency_checks"].append(
        _consist(
            "token_cost_sum",
            abs(tc_total - recomputed_cost) < 0.01,
            f"total_cost={tc_total}, recomputed={recomputed_cost}",
        )
    )

    # 7. Failure taxonomy: sum of failure type counts + PASS count == total C1
    ft_total = ft.get("total_records", 0)
    ft_status_sum = sum(ft_sc.values())
    results["consistency_checks"].append(
        _consist(
            "failure_taxonomy_sum",
            ft_status_sum == ft_total,
            f"sum(status_counts)={ft_status_sum}, total_records={ft_total}",
        )
    )

    if verbose:
        print(f"  {len(results['consistency_checks'])} consistency checks completed")

    # -----------------------------------------------------------------------
    # Category 4: Paper claims pre-audit
    # -----------------------------------------------------------------------

    paper_tex_path = project_root / "docs" / "paper" / "latex" / "paper.tex"
    paper_text = ""
    if paper_tex_path.exists():
        try:
            paper_text = paper_tex_path.read_text(encoding="utf-8")
        except OSError:
            pass

    paper_claims = output.get("paper_claims", [])

    for claim in paper_claims:
        claim_id = claim.get("claim_id", "unknown")
        display_value = claim.get("display_value", "")
        current_value = claim.get("current_value")

        if not paper_text:
            results["paper_claims_audit"].append({
                "claim_id": claim_id,
                "paper_value": None,
                "findings_value": display_value,
                "status": "NOT_FOUND",
                "paper_location": claim.get("paper_location", ""),
            })
            continue

        # Build search patterns from the display_value and current_value
        search_patterns = _build_claim_search_patterns(claim_id, current_value, display_value)

        found = False
        paper_location = ""
        paper_value_found = ""
        for pat in search_patterns:
            matches = list(re.finditer(pat, paper_text))
            if matches:
                found = True
                # Find line number
                pos = matches[0].start()
                line_num = paper_text[:pos].count("\n") + 1
                paper_value_found = matches[0].group(0)
                paper_location = f"line ~{line_num}"
                break

        status = "MATCH" if found else "NOT_FOUND"
        results["paper_claims_audit"].append({
            "claim_id": claim_id,
            "paper_value": paper_value_found if found else None,
            "findings_value": display_value,
            "status": status,
            "paper_location": paper_location if found else claim.get("paper_location", ""),
        })

    if verbose:
        print(f"  {len(results['paper_claims_audit'])} paper claims audited")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    all_checks = (
        results["spot_checks"]
        + results["cross_checks"]
        + results["consistency_checks"]
        + results["paper_claims_audit"]
    )
    results["summary"]["total"] = len(all_checks)
    results["summary"]["passed"] = sum(
        1 for c in all_checks if c.get("status") in ("PASS", "MATCH")
    )
    results["summary"]["failed"] = sum(
        1 for c in all_checks if c.get("status") == "FAIL"
    )
    results["summary"]["warnings"] = sum(
        1 for c in all_checks
        if c.get("status") in ("WARNING", "STALE", "DIFFERENT_SCOPE", "NOT_FOUND")
    )

    # Print summary
    s = results["summary"]
    print(
        f"Validation: {s['total']} checks, {s['passed']} PASS, "
        f"{s['failed']} FAIL, {s['warnings']} warnings"
    )

    # Print any FAILs
    fails = [c for c in all_checks if c.get("status") == "FAIL"]
    if fails:
        print("\nFAILED checks:")
        for fc in fails:
            print(f"  - {fc.get('check', 'unknown')}: expected={fc.get('expected')}, actual={fc.get('actual')}")

    # Write validation JSON
    val_path = analysis_dir / "quantitative_findings_validation.json"
    val_path.write_text(
        json.dumps(results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"\nWrote: {val_path}")

    return results


def _build_claim_search_patterns(claim_id: str, current_value, display_value: str) -> list[str]:
    """Build regex patterns to search for a claim value in paper.tex.

    Returns a list of patterns to try, from most specific to least.
    """
    patterns: list[str] = []

    if current_value is None:
        return patterns

    # For numeric values, search for the number in common paper formats
    if isinstance(current_value, (int, float)):
        if isinstance(current_value, float):
            # Try percentage format: "36.2%" or "36.2\\%"
            pct_str = f"{current_value * 100:.1f}"
            patterns.append(re.escape(pct_str) + r"\\?%")
            # Also try decimal
            patterns.append(re.escape(f"{current_value:.4f}"))
        elif isinstance(current_value, int):
            patterns.append(r"\b" + re.escape(str(current_value)) + r"\b")

    # For dict values with meaningful fields
    elif isinstance(current_value, dict):
        if "count" in current_value:
            patterns.append(r"\b" + re.escape(str(current_value["count"])) + r"\b")
        if "percentage" in current_value:
            pct_str = f"{current_value['percentage'] * 100:.1f}"
            patterns.append(re.escape(pct_str) + r"\\?%")
        if "rho" in current_value:
            # Spearman rho, might appear differently
            rho_val = current_value["rho"]
            if isinstance(rho_val, (int, float)):
                patterns.append(re.escape(f"{rho_val:.3f}"))
                patterns.append(re.escape(f"{rho_val:.2f}"))
        if "z" in current_value:
            z_val = current_value["z"]
            if isinstance(z_val, (int, float)):
                patterns.append(re.escape(f"{z_val:.2f}"))
        if "pass_at_1" in current_value:
            p1 = current_value["pass_at_1"]
            if isinstance(p1, (int, float)):
                pct_str = f"{p1 * 100:.1f}"
                patterns.append(re.escape(pct_str) + r"\\?%")
        if "full_repairs" in current_value:
            patterns.append(r"\b" + re.escape(str(current_value["full_repairs"])) + r"\b")
        if "rodinia_tasks" in current_value:
            patterns.append(r"\b" + re.escape(str(current_value["rodinia_tasks"])) + r"\b")
        if "fraction" in current_value:
            frac = current_value["fraction"]
            if isinstance(frac, (int, float)):
                pct_str = f"{frac * 100:.1f}"
                patterns.append(re.escape(pct_str) + r"\\?%")
        if "min" in current_value and "max" in current_value:
            # Cohen's h range
            h_min = current_value["min"]
            if isinstance(h_min, (int, float)):
                patterns.append(re.escape(f"{h_min:.4f}"))

    # Fallback: extract numbers from display_value
    if not patterns and display_value:
        nums = re.findall(r"\d+\.?\d*", display_value)
        for n in nums[:3]:  # Limit to first 3 numbers
            if len(n) >= 2:  # Skip single digits
                patterns.append(re.escape(n))

    return patterns


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantitative findings for NeurIPS 2026 ParBench paper (Phase 3 canonical+augmentation corpus)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="together-qwen-3.5-397b-a17b",
        help="Model results subdirectory under results/evaluation/",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation spot-checks, cross-checks, consistency checks, and paper claims audit",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_dir = project_root / "results" / "evaluation" / args.model_dir
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    if args.verbose:
        print("Loading result files...")
    all_records = load_results(results_dir, verbose=args.verbose)

    # --- Exclude KNOWN_FAIL ---
    if args.verbose:
        print(f"Excluding {len(EXCLUDED_SPECS)} KNOWN_FAIL specs...")
    valid = exclude_known_fail(all_records)
    excluded_count = len(all_records) - len(valid)
    if args.verbose:
        print(f"  Excluded {excluded_count} records ({len(valid)} remaining)")

    # --- Split campaigns ---
    c1, c2 = split_campaigns(valid)
    if args.verbose:
        print(f"Legacy (temp=0.0): {len(c1)} records")
        print(f"Canonical (temp>0):   {len(c2)} records")

    # --- Compute dimensions 1-5 for legacy (temp=0.0) ---
    if args.verbose:
        print("\nComputing dimensions 1-5 for legacy (temp=0.0)...")

    c1_results: dict = {}
    c1_results["_metadata"] = {"total_on_disk": len(all_records)}

    if args.verbose:
        print("  Dimension 1: Aggregate pass rates...")
    c1_results["aggregate_pass_rates"] = compute_aggregate_pass_rates(c1)

    if args.verbose:
        print("  Dimension 2: Per-direction pass rates...")
    c1_results["direction_pass_rates"] = compute_direction_pass_rates(c1)

    if args.verbose:
        print("  Dimension 3: Direction asymmetry...")
    c1_results["direction_asymmetry"] = compute_direction_asymmetry(c1)

    if args.verbose:
        print("  Dimension 4: Augmentation trends...")
    c1_results["augmentation_trends"] = compute_augmentation_trends(c1)

    if args.verbose:
        print("  Dimension 5: Failure taxonomy...")
    c1_results["failure_taxonomy"] = compute_failure_taxonomy(c1)

    # --- Compute dimensions 6-13 for legacy (temp=0.0) ---
    if args.verbose:
        print("\nComputing dimensions 8-13 for legacy (temp=0.0)...")

    if args.verbose:
        print("  Dimension 8: Per-kernel difficulty tiers...")
    c1_results["per_kernel_tiers"] = compute_per_kernel_tiers(c1)

    if args.verbose:
        print("  Dimension 9: Translation complexity correlation...")
    c1_results["complexity_correlation"] = compute_complexity_correlation(c1, project_root)

    if args.verbose:
        print("  Dimension 10: Cross-suite comparison...")
    c1_results["cross_suite"] = compute_cross_suite(c1, project_root)

    if args.verbose:
        print("  Dimension 11: Token cost analysis...")
    c1_results["token_cost"] = compute_token_cost(c1)

    if args.verbose:
        print("  Dimension 12: SLoC correlation...")
    c1_results["sloc_correlation"] = compute_sloc_correlation(c1, project_root)

    if args.verbose:
        print("  Dimension 13: OpenCL kernel-only effect...")
    c1_results["opencl_kernel_only_effect"] = compute_opencl_kernel_only_effect(c1)

    # --- Rodinia subset for paper claims scope matching ---
    if args.verbose:
        print("  Computing Rodinia subset...")
    c1_results["rodinia_subset"] = compute_rodinia_subset(c1)

    # --- Compute dimensions 1-5 for canonical ---
    if args.verbose:
        print("\nComputing dimensions 1-5 for canonical...")

    c2_results: dict = {}

    c2_results["aggregate_pass_rates"] = compute_aggregate_pass_rates(c2)
    c2_results["direction_pass_rates"] = compute_direction_pass_rates(c2)
    c2_results["direction_asymmetry"] = compute_direction_asymmetry(c2)
    c2_results["augmentation_trends"] = compute_augmentation_trends(c2)
    c2_results["failure_taxonomy"] = compute_failure_taxonomy(c2)

    # --- Dimension 7: pass@k (canonical only) ---
    if args.verbose:
        print("\nComputing dimension 7: pass@k for canonical...")
    c2_results["pass_at_k"] = compute_pass_at_k(c2)

    # --- Cross-check ---
    if args.verbose:
        print("\nRunning cross-checks...")
    xc = cross_check(c1_results, c2_results, project_root, args.verbose)
    if args.verbose:
        for w in xc.get("warnings", []):
            print(f"  {w}")

    # --- Build output ---
    metadata = build_metadata(all_records, valid, c1, c2, args)
    paper_claims = build_paper_claims(c1_results, c2_results, project_root)

    output = {
        "metadata": metadata,
        "canonical": c2_results,
        "cross_check": xc,
        "paper_claims": paper_claims,
    }

    # c1_results is always empty (temp=0.0 was never run) — omitted from output

    # --- Validate mode ---
    if args.validate:
        if args.verbose:
            print("\n--- Running validation ---")
        val_results = run_validation(output, project_root, args.verbose, model_dir=args.model_dir)
        if val_results["summary"]["failed"] > 0:
            return 1
        return 0

    # --- Write JSON ---
    model_slug = args.model_dir.replace("/", "_")
    json_path = output_dir / f"quantitative_findings_{model_slug}.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if args.verbose:
        print(f"\nWrote: {json_path}")

    # --- Write markdown ---
    md_path = output_dir / f"quantitative_findings_{model_slug}.md"
    write_markdown(output, md_path)
    if args.verbose:
        print(f"Wrote: {md_path}")

    # --- Summary ---
    c1_overall = c1_results.get("aggregate_pass_rates", {}).get("overall", {})
    print(
        f"Quantitative findings: {len(all_records)} total files, "
        f"{len(valid)} valid ({len(c1)} C1 + {len(c2)} C2), "
        f"C1 pass rate = {_pct(c1_overall.get('value'))} "
        f"[{_pct(c1_overall.get('ci_lower'))}, {_pct(c1_overall.get('ci_upper'))}]"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
