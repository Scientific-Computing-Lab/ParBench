#!/usr/bin/env python3
"""Generate paper_data.json — clean, single-source data for NeurIPS 2026 paper tables.

Reads all result JSONs from results/evaluation/{model}/, separates primary
(temp=0.0) and pass@k (temp=0.7) campaigns, computes statistics with Wilson
CIs, error taxonomy, self-repair analysis, and augmentation trend tests.

Produces: results/analysis/paper_data.json

--results-dir is required (no default) to force explicit model selection and
prevent accidentally analyzing the wrong corpus.

Usage:
    python3 scripts/analysis/generate_paper_data.py \\
        --results-dir results/evaluation/together-qwen-3.5-397b-a17b \\
        --output results/analysis/paper_data.json -v
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from harness.constants import EXCLUDED_SPECS

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Sample-size and statistical thresholds
MIN_L0_SAMPLE_SIZE: int = 10
MIN_PASSK_TASK_SAMPLES: int = 3
MIN_BALANCED_KERNELS: int = 20
ALPHA_SIGNIFICANCE: float = 0.05
COHEN_H_SMALL_THRESHOLD: float = 0.20
COHEN_H_MEDIUM_THRESHOLD: float = 0.80
PASSK_K_VALUES: list[int] = [1, 3]

# Valid overall_status values
STATUS_VALUES = ("PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL")


# --------------------------------------------------------------------------- #
# Parsing helpers (copied from analyze_eval.py to keep self-contained)         #
# --------------------------------------------------------------------------- #

def _kernel_from_spec(spec_id: str) -> str:
    """Extract kernel name from spec ID like 'rodinia-bfs-cuda' -> 'bfs'."""
    parts = spec_id.split("-")
    if len(parts) < 3:
        return spec_id
    return "-".join(parts[1:-1])


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


# --------------------------------------------------------------------------- #
# Wilson score CI (copied from statistical_analysis.py)                        #
# --------------------------------------------------------------------------- #

def wilson_ci(passes: int, total: int, alpha: float = ALPHA_SIGNIFICANCE) -> dict:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return {"rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    from scipy import stats as sp_stats
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = passes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom

    return {
        "rate": round(p_hat, 4),
        "ci_lower": round(max(0.0, center - spread), 4),
        "ci_upper": round(min(1.0, center + spread), 4),
        "n": total,
    }


# --------------------------------------------------------------------------- #
# Cochran-Armitage trend test (copied from statistical_analysis.py)            #
# --------------------------------------------------------------------------- #

def cochran_armitage_trend(
    pass_counts: list[int],
    total_counts: list[int],
    scores: list[int] | None = None,
) -> dict:
    """Cochran-Armitage test for trend in binomial proportions."""
    import numpy as np
    from scipy import stats as sp_stats

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


# --------------------------------------------------------------------------- #
# McNemar exact test (copied from statistical_analysis.py)                     #
# --------------------------------------------------------------------------- #

def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    """McNemar's exact test on paired boolean observations.

    pairs: list of (forward_pass, reverse_pass) tuples.
    Returns dict with 2x2 table, p-value, method.
    """
    from scipy import stats as sp_stats

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

    # Cohen's h effect size
    h = 2 * math.asin(math.sqrt(max(0.0, min(1.0, fwd_rate)))) - \
        2 * math.asin(math.sqrt(max(0.0, min(1.0, rev_rate))))
    abs_h = abs(h)
    effect = "small" if abs_h < COHEN_H_SMALL_THRESHOLD else "medium" if abs_h < COHEN_H_MEDIUM_THRESHOLD else "large"

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


# --------------------------------------------------------------------------- #
# Error taxonomy classification (adapted from build_error_taxonomy.py)         #
# --------------------------------------------------------------------------- #

# BUILD_FAIL regex patterns (priority order, first match wins)
_BUILD_FAIL_PATTERNS: list[tuple[str, re.Pattern, callable | None]] = [
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

# RUN_FAIL classification lambdas (priority order)
_RUN_FAIL_CHECKS: list[tuple[str, callable]] = [
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

# EXTRACTION_FAIL regex patterns
_EXTRACTION_FAIL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("no_code_blocks", re.compile(r"did not contain parseable code", re.IGNORECASE)),
    ("missing_files", re.compile(
        r"expected file|not found in output|missing.*file|could not find|no .* file",
        re.IGNORECASE,
    )),
]


def classify_build_fail(data: dict) -> str:
    """Return primary BUILD_FAIL subcategory."""
    snippet = data.get("build_error_snippet") or ""
    direction = _direction_from_data(data)

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


def classify_run_fail(data: dict) -> str:
    """Return primary RUN_FAIL subcategory."""
    exit_code = data.get("run_exit_code")
    stderr = data.get("run_stderr_snippet") or ""
    stdout = data.get("run_stdout_snippet") or ""

    for cat_name, check_fn in _RUN_FAIL_CHECKS:
        if check_fn(exit_code, stderr, stdout):
            return cat_name

    return "other_runtime"


def classify_extraction_fail(data: dict) -> str:
    """Return primary EXTRACTION_FAIL subcategory."""
    error_msg = data.get("error_message") or ""

    for cat_name, pattern in _EXTRACTION_FAIL_PATTERNS:
        if pattern.search(error_msg):
            return cat_name

    return "malformed_output"


def classify_verify_fail(data: dict) -> str:
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


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

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

        # Ensure direction is set
        if "direction" not in data or not data.get("direction"):
            data["direction"] = _direction_from_data(data)

        # Ensure kernel is set
        if "kernel" not in data or not data.get("kernel"):
            data["kernel"] = _kernel_from_spec(data.get("source_spec", ""))

        # Ensure temperature is set
        if "temperature" not in data or data["temperature"] is None:
            data["temperature"] = 0.0

        records.append(data)

    if verbose:
        print(f"  Loaded {len(records)} result files ({skipped} skipped)")

    return records


def exclude_known_fail(records: list[dict]) -> list[dict]:
    """Remove records involving KNOWN_FAIL specs."""
    return [
        r for r in records
        if r.get("source_spec") not in EXCLUDED_SPECS
        and r.get("target_spec") not in EXCLUDED_SPECS
    ]


def split_campaigns(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split records into primary (temp=0.0) and pass@k (temp=0.7) campaigns."""
    primary = []
    passk = []
    for r in records:
        temp = r.get("temperature", 0.0) or 0.0
        if temp == 0.0:
            primary.append(r)
        else:
            passk.append(r)
    return primary, passk


# --------------------------------------------------------------------------- #
# Status breakdown helper                                                      #
# --------------------------------------------------------------------------- #

def status_breakdown(records: list[dict]) -> dict:
    """Compute pass/fail counts with Wilson CI."""
    total = len(records)
    by_status: dict[str, int] = defaultdict(int)
    for r in records:
        status = r.get("overall_status", "UNKNOWN")
        by_status[status] += 1

    passes = by_status.get("PASS", 0)
    ci = wilson_ci(passes, total)

    return {
        "total": total,
        "pass": passes,
        "pass_rate": ci["rate"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "by_status": dict(by_status),
    }


# --------------------------------------------------------------------------- #
# Primary campaign analysis                                                    #
# --------------------------------------------------------------------------- #

def analyze_primary(records: list[dict], verbose: bool = False) -> dict:
    """Full analysis of the primary (temp=0.0) campaign."""
    result: dict = {}

    # Overall
    result["total"] = len(records)
    result["overall"] = status_breakdown(records)

    # By direction
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_dir[r.get("direction", "unknown")].append(r)
    result["by_direction"] = {d: status_breakdown(recs) for d, recs in sorted(by_dir.items())}

    # By kernel
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_kernel[r.get("kernel", "unknown")].append(r)
    result["by_kernel"] = {k: status_breakdown(recs) for k, recs in sorted(by_kernel.items())}

    # By level
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_level[r.get("augment_level", 0)].append(r)
    result["by_level"] = {
        f"L{lv}": status_breakdown(recs)
        for lv, recs in sorted(by_level.items())
    }

    # Augmentation analysis
    result["augmentation"] = _analyze_augmentation(records, verbose)

    # Direction asymmetry (L0 primary only, McNemar)
    result["direction_asymmetry"] = _analyze_direction_asymmetry(records)

    # Error taxonomy subcategories
    result["build_fail_subcategories"] = _classify_failures(records, "BUILD_FAIL", classify_build_fail)
    result["run_fail_subcategories"] = _classify_failures(records, "RUN_FAIL", classify_run_fail)
    result["verify_fail_subcategories"] = _classify_failures(records, "VERIFY_FAIL", classify_verify_fail)
    result["extraction_fail_subcategories"] = _classify_failures(records, "EXTRACTION_FAIL", classify_extraction_fail)

    # Token and cost metrics
    result["token_metrics"] = _analyze_tokens(records)

    return result


def _classify_failures(records: list[dict], status: str, classifier) -> dict:
    """Classify failures of a given status using the provided classifier."""
    failures = [r for r in records if r.get("overall_status") == status]
    counts: dict[str, int] = defaultdict(int)
    for r in failures:
        cat = classifier(r)
        counts[cat] += 1
    return {"total": len(failures), "subcategories": dict(counts)}


# --------------------------------------------------------------------------- #
# Augmentation analysis                                                        #
# --------------------------------------------------------------------------- #

def _analyze_augmentation(records: list[dict], verbose: bool = False) -> dict:
    """Augmentation curve and Cochran-Armitage trend test."""
    result: dict = {}

    # All directions: pass rate per level
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_level[r.get("augment_level", 0)].append(r)

    all_dir_by_level = {}
    for lv in sorted(by_level.keys()):
        recs = by_level[lv]
        passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        all_dir_by_level[f"L{lv}"] = wilson_ci(passes, len(recs))
    result["all_directions"] = all_dir_by_level

    # Balanced cuda-to-omp subset for Cochran-Armitage
    c2o = [r for r in records if r.get("direction") == "cuda-to-omp"]

    # Find balanced subset: kernels present at ALL levels
    kernel_levels: dict[str, set[int]] = defaultdict(set)
    for r in c2o:
        kernel_levels[r.get("kernel", "?")].add(r.get("augment_level", 0))

    all_levels_present = set()
    for r in c2o:
        all_levels_present.add(r.get("augment_level", 0))

    balanced_kernels = {k for k, levels in kernel_levels.items()
                        if levels == all_levels_present}

    balanced = [r for r in c2o if r.get("kernel", "?") in balanced_kernels]

    # Balanced pass rates per level
    bal_by_level: dict[int, list[dict]] = defaultdict(list)
    for r in balanced:
        bal_by_level[r.get("augment_level", 0)].append(r)

    cuda_to_omp_balanced = {}
    for lv in sorted(bal_by_level.keys()):
        recs = bal_by_level[lv]
        passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
        cuda_to_omp_balanced[f"L{lv}"] = wilson_ci(passes, len(recs))
    result["cuda_to_omp_balanced"] = cuda_to_omp_balanced

    if verbose:
        print(f"  Augmentation: balanced cuda-to-omp subset = {len(balanced_kernels)} kernels "
              f"x {len(all_levels_present)} levels = {len(balanced)} tasks")

    # Cochran-Armitage trend test on balanced subset
    if balanced:
        levels_sorted = sorted(bal_by_level.keys())
        pass_counts = [sum(1 for r in bal_by_level[lv] if r.get("overall_status") == "PASS")
                       for lv in levels_sorted]
        total_counts = [len(bal_by_level[lv]) for lv in levels_sorted]

        ca = cochran_armitage_trend(pass_counts, total_counts)
        ca["significant"] = ca["p_value"] < ALPHA_SIGNIFICANCE
        ca["levels"] = levels_sorted
        ca["pass_counts"] = pass_counts
        ca["total_counts"] = total_counts
        ca["n_kernels"] = len(balanced_kernels)
        result["cochran_armitage"] = ca
    else:
        result["cochran_armitage"] = {"error": "No balanced cuda-to-omp subset found"}

    # Per-direction breakdown by level
    per_dir_level: dict[str, dict[str, dict]] = {}
    for r in records:
        d = r.get("direction", "unknown")
        lv = r.get("augment_level", 0)
        if d not in per_dir_level:
            per_dir_level[d] = {}
        lv_key = f"L{lv}"
        if lv_key not in per_dir_level[d]:
            per_dir_level[d][lv_key] = {"pass": 0, "total": 0}
        per_dir_level[d][lv_key]["total"] += 1
        if r.get("overall_status") == "PASS":
            per_dir_level[d][lv_key]["pass"] += 1

    # Add Wilson CIs to per-direction-level data
    for d in per_dir_level:
        for lv_key in per_dir_level[d]:
            entry = per_dir_level[d][lv_key]
            ci = wilson_ci(entry["pass"], entry["total"])
            entry.update(ci)
    result["per_direction_by_level"] = per_dir_level

    return result


# --------------------------------------------------------------------------- #
# Direction asymmetry (McNemar at L0)                                          #
# --------------------------------------------------------------------------- #

def _analyze_direction_asymmetry(records: list[dict]) -> dict:
    """McNemar's exact test for direction asymmetry at L0 only."""
    # Only L0 records to avoid augmentation confound
    l0 = [r for r in records if r.get("augment_level", 0) == 0]

    # Build lookup: (kernel, direction) -> passed
    lookup: dict[tuple[str, str], bool] = {}
    for r in l0:
        kernel = r.get("kernel", "?")
        direction = r.get("direction", "unknown")
        passed = r.get("overall_status") == "PASS"
        lookup[(kernel, direction)] = passed

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
    alpha_corrected = ALPHA_SIGNIFICANCE / n_tests

    results = {}
    for (d_fwd, d_rev) in pair_map.values():
        pairs: list[tuple[bool, bool]] = []
        # Find kernels in forward direction
        fwd_kernels = {k for (k, d) in lookup if d == d_fwd}
        for kernel in fwd_kernels:
            fwd_pass = lookup.get((kernel, d_fwd))
            rev_pass = lookup.get((kernel, d_rev))
            if fwd_pass is not None and rev_pass is not None:
                pairs.append((fwd_pass, rev_pass))

        if not pairs:
            continue

        test = mcnemar_exact(pairs)
        test["alpha_corrected"] = round(alpha_corrected, 6)
        test["significant"] = test["p_value"] < alpha_corrected
        results[f"{d_fwd} vs {d_rev}"] = test

    return results


# --------------------------------------------------------------------------- #
# Token & cost metrics                                                         #
# --------------------------------------------------------------------------- #

def _analyze_tokens(records: list[dict]) -> dict:
    """Aggregate token usage and timing metrics."""
    prompt_tokens = []
    completion_tokens = []
    response_times = []
    extraction_stats = {"expected": 0, "extracted": 0, "mismatch_count": 0}

    # Per-direction token aggregation
    dir_tokens: dict[str, dict[str, list]] = defaultdict(lambda: {"prompt": [], "completion": []})

    for r in records:
        pt = r.get("prompt_tokens")
        ct = r.get("completion_tokens")
        rt = r.get("llm_response_time_seconds")

        if pt is not None:
            prompt_tokens.append(pt)
            dir_tokens[r.get("direction", "unknown")]["prompt"].append(pt)
        if ct is not None:
            completion_tokens.append(ct)
            dir_tokens[r.get("direction", "unknown")]["completion"].append(ct)
        if rt is not None:
            response_times.append(rt)

        # Extraction reliability
        expected = r.get("target_files_expected") or []
        extracted = r.get("target_files_extracted") or []
        extraction_stats["expected"] += len(expected)
        extraction_stats["extracted"] += len(extracted)
        if len(expected) != len(extracted):
            extraction_stats["mismatch_count"] += 1

    def _stats(vals: list) -> dict:
        if not vals:
            return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "total": 0}
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        return {
            "mean": round(sum(vals) / n, 1),
            "median": round(sorted_vals[n // 2], 1),
            "p95": round(sorted_vals[int(n * 0.95)], 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "total": round(sum(vals), 1),
        }

    # Per-direction summary
    per_direction_tokens = {}
    for d, toks in sorted(dir_tokens.items()):
        per_direction_tokens[d] = {
            "prompt": _stats(toks["prompt"]),
            "completion": _stats(toks["completion"]),
            "n": len(toks["prompt"]),
        }

    return {
        "prompt_tokens": _stats(prompt_tokens),
        "completion_tokens": _stats(completion_tokens),
        "llm_response_time_seconds": _stats(response_times),
        "per_direction": per_direction_tokens,
        "extraction_reliability": extraction_stats,
    }


# --------------------------------------------------------------------------- #
# Pass@k analysis                                                              #
# --------------------------------------------------------------------------- #

def analyze_passk(records: list[dict], verbose: bool = False) -> dict:
    """Analyze pass@k campaign (temp=0.7 samples).

    Filters to L0-only records: L1-L4 records use augmented source variants,
    not i.i.d. samples, so they must not inflate n for pass@k.
    """
    records = [r for r in records if r.get("augment_level", 0) == 0]
    result: dict = {}
    result["total"] = len(records)
    result["overall"] = status_breakdown(records)

    # By direction
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_dir[r.get("direction", "unknown")].append(r)
    result["by_direction"] = {d: status_breakdown(recs) for d, recs in sorted(by_dir.items())}

    # By kernel
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_kernel[r.get("kernel", "unknown")].append(r)
    result["by_kernel"] = {k: status_breakdown(recs) for k, recs in sorted(by_kernel.items())}

    # pass@k estimates grouped by (kernel, direction)
    # Group by task identity (kernel + direction) and collect samples
    task_samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r.get("kernel", "?"), r.get("direction", "?"))
        task_samples[key].append(r)

    passk_estimates: dict[str, dict] = {}
    for (kernel, direction), samples in sorted(task_samples.items()):
        n = len(samples)
        c = sum(1 for s in samples if s.get("overall_status") == "PASS")

        # pass@k formula: pass@k = 1 - C(n-c, k) / C(n, k)
        estimates = {}
        for k_val in PASSK_K_VALUES:
            if k_val > n:
                estimates[f"pass@{k_val}"] = None
                continue
            numerator = math.comb(n - c, k_val)
            denominator = math.comb(n, k_val)
            passk_val = 1.0 - (numerator / denominator) if denominator > 0 else 0.0
            estimates[f"pass@{k_val}"] = round(passk_val, 4)

        passk_estimates[f"{kernel}:{direction}"] = {
            "n": n,
            "c": c,
            **estimates,
        }

    result["passk_estimates"] = passk_estimates

    # Aggregate pass@k across all tasks
    all_n = []
    all_c = []
    for (kernel, direction), samples in task_samples.items():
        n = len(samples)
        c = sum(1 for s in samples if s.get("overall_status") == "PASS")
        all_n.append(n)
        all_c.append(c)

    if all_n:
        # Macro-average pass@k across tasks
        passk1_values = []
        passk3_values = []
        for n, c in zip(all_n, all_c):
            for k_val, values_list in [(1, passk1_values), (3, passk3_values)]:
                if k_val <= n:
                    num = math.comb(n - c, k_val)
                    den = math.comb(n, k_val)
                    values_list.append(1.0 - num / den if den > 0 else 0.0)

        result["aggregate_passk"] = {
            "pass@1_macro_avg": round(sum(passk1_values) / len(passk1_values), 4) if passk1_values else None,
            "pass@3_macro_avg": round(sum(passk3_values) / len(passk3_values), 4) if passk3_values else None,
            "n_tasks": len(all_n),
            "total_samples": sum(all_n),
        }

    return result


# --------------------------------------------------------------------------- #
# Sample size flags                                                            #
# --------------------------------------------------------------------------- #

def compute_sample_size_flags(primary: list[dict], passk: list[dict]) -> list[str]:
    """Flag any subsets with sample sizes too small for reliable inference."""
    flags = []

    # Check per-direction sample sizes at L0
    l0_primary = [r for r in primary if r.get("augment_level", 0) == 0]
    dir_counts: dict[str, int] = defaultdict(int)
    for r in l0_primary:
        dir_counts[r.get("direction", "unknown")] += 1

    for d, n in dir_counts.items():
        if n < MIN_L0_SAMPLE_SIZE:
            flags.append(f"L0 {d}: n={n} (< {MIN_L0_SAMPLE_SIZE} — wide CIs, interpret with caution)")

    # pass@k sample size per task
    task_counts: dict[str, int] = defaultdict(int)
    for r in passk:
        key = f"{r.get('kernel', '?')}:{r.get('direction', '?')}"
        task_counts[key] += 1

    low_sample_tasks = [(t, n) for t, n in task_counts.items() if n < MIN_PASSK_TASK_SAMPLES]
    if low_sample_tasks:
        flags.append(f"pass@k: {len(low_sample_tasks)} tasks with n < {MIN_PASSK_TASK_SAMPLES} samples")

    # Augmentation balanced subset size
    c2o = [r for r in primary if r.get("direction") == "cuda-to-omp"]
    kernel_levels: dict[str, set[int]] = defaultdict(set)
    for r in c2o:
        kernel_levels[r.get("kernel", "?")].add(r.get("augment_level", 0))
    all_levels = set()
    for r in c2o:
        all_levels.add(r.get("augment_level", 0))
    balanced = {k for k, levels in kernel_levels.items() if levels == all_levels}
    if len(balanced) < MIN_BALANCED_KERNELS:
        flags.append(
            f"Cochran-Armitage balanced subset: n={len(balanced)} kernels "
            f"(< {MIN_BALANCED_KERNELS} — limited statistical power)"
        )

    # Single model flag
    flags.append("Single model evaluation — cannot generalize across models")

    return flags


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper_data.json for NeurIPS 2026 paper tables.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to model results directory (required — no default to prevent wrong-corpus analysis)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (include model name, e.g. paper_data_azure_gpt54.json)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Filter to a specific suite (e.g. 'rodinia'). "
             "Only results whose source_spec starts with this prefix are included.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    if not results_dir.is_dir():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loading results from: {results_dir}")

    # Step 1: Load and filter
    all_records = load_results(results_dir, args.verbose)
    filtered = exclude_known_fail(all_records)

    primary, passk = split_campaigns(filtered)

    # Suite filter: limit PRIMARY campaign to a specific suite prefix (e.g. "rodinia").
    # Pass@k campaign is intentionally NOT filtered — it covers all suites for broader
    # coverage (426 tasks vs 288 if restricted to Rodinia).
    if args.suite:
        suite_prefix = args.suite + "-"
        before = len(primary)
        primary = [
            r for r in primary
            if r.get("source_spec", "").startswith(suite_prefix)
        ]
        if args.verbose:
            print(f"  Suite filter '{args.suite}' (primary only): {before} -> {len(primary)} records")

    if args.verbose:
        print(f"  Total loaded: {len(all_records)}")
        print(f"  After KNOWN_FAIL exclusion: {len(filtered)}")
        print(f"  Primary campaign (temp=0.0): {len(primary)}")
        print(f"  Pass@k campaign (temp=0.7, all suites): {len(passk)}")

    # Step 2: Analyze
    if args.verbose:
        print("Analyzing primary campaign...")
    primary_analysis = analyze_primary(primary, args.verbose)

    if args.verbose:
        print("Analyzing pass@k campaign...")
    passk_analysis = analyze_passk(passk, args.verbose)

    # Step 3: Sample size flags
    flags = compute_sample_size_flags(primary, passk)

    # Step 4: Assemble output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": results_dir.name,
        "results_dir": str(results_dir),
        "suite_filter": args.suite,
        "excluded_specs": sorted(EXCLUDED_SPECS),
        "file_counts": {
            "total_on_disk": len(all_records),
            "excluded_known_fail": len(all_records) - len(filtered),
            "primary_campaign": len(primary),
            "passk_campaign": len(passk),
        },
        "primary_campaign": primary_analysis,
        "passk_campaign": passk_analysis,
        "sample_size_flags": flags,
    }

    # Step 5: Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if args.verbose:
        print(f"\nWrote: {output_path}")

    # Summary to stdout
    p = primary_analysis["overall"]
    print(f"\n=== Paper Data Summary ===")
    print(f"Primary campaign: {p['total']} tasks, {p['pass']}/{p['total']} PASS "
          f"({p['pass_rate']:.1%}) [{p['ci_lower']:.1%}, {p['ci_upper']:.1%}]")

    pk = passk_analysis["overall"]
    print(f"Pass@k campaign:  {pk['total']} tasks, {pk['pass']}/{pk['total']} PASS "
          f"({pk['pass_rate']:.1%}) [{pk['ci_lower']:.1%}, {pk['ci_upper']:.1%}]")

    print(f"\nPer-direction pass rates (primary L0):")
    l0_records = [r for r in primary if r.get("augment_level", 0) == 0]
    dir_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in l0_records:
        d = r.get("direction", "unknown")
        dir_counts[d]["total"] += 1
        if r.get("overall_status") == "PASS":
            dir_counts[d]["pass"] += 1
    for d in sorted(dir_counts):
        c = dir_counts[d]
        ci = wilson_ci(c["pass"], c["total"])
        print(f"  {d}: {c['pass']}/{c['total']} ({ci['rate']:.1%}) "
              f"[{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    # Augmentation summary
    aug = primary_analysis["augmentation"]
    ca = aug.get("cochran_armitage", {})
    if "z" in ca:
        print(f"\nCochran-Armitage: z={ca['z']}, p={ca['p_value']}, "
              f"significant={ca.get('significant', 'N/A')}")

    if flags:
        print(f"\nSample size warnings:")
        for flag in flags:
            print(f"  - {flag}")


if __name__ == "__main__":
    main()
