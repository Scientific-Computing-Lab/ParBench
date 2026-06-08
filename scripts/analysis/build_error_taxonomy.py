#!/usr/bin/env python3
"""Build a structured error taxonomy from all LLM evaluation result JSONs.

Reads every result JSON in results/evaluation/{model}/, classifies each
non-PASS result by root-cause category, and produces:
  - results/analysis/error_taxonomy.json  (structured data)
  - results/analysis/error_taxonomy.md    (publication-ready markdown tables)

Each failure gets exactly ONE primary category (first match in priority order).
Multi-cause failures also get secondary_categories for compound analysis.

Usage:
    python3 scripts/analysis/build_error_taxonomy.py --project-root .
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def safe_percentage(numerator: int | float, denominator: int | float) -> float:
    """Return numerator/denominator * 100, or 0.0 if denominator is zero."""
    return numerator / denominator * 100 if denominator else 0.0


MAX_EXAMPLES_PER_CATEGORY: int = 3
OTHER_BUCKET_REFINEMENT_THRESHOLD: float = 20.0


# ============================================================
# BUILD_FAIL categories — checked in priority order on FULL snippet
# First match wins as primary category.
# ============================================================

BUILD_FAIL_CATEGORIES = [
    (
        "retained_cuda_api",
        "LLM retained CUDA API calls/keywords in non-CUDA target code",
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
        "LLM retained CUDA-specific types (float3, dim3, etc.) in non-CUDA target",
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
        "LLM retained OpenCL API calls in non-OpenCL target code",
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
        "missing_target_api",
        "LLM translation contains no target API constructs (no pragmas/kernels for target language)",
        None,  # Special: checked via translated_files in classify_build_fail, not via snippet regex
        None,
    ),
    (
        "missing_header",
        "Missing or wrong #include directive — file not found",
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
        "Compilation succeeded but linking failed (undefined references, etc.)",
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
        "Function, variable, or type not declared/defined in scope",
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
        "Type mismatches, wrong function signatures, incompatible conversions",
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
        "Parse errors, malformed code, unexpected tokens",
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
        "Missing function prototypes — implicit declaration warnings promoted to errors",
        re.compile(
            r"implicit declaration of function",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "redefinition",
        "Duplicate or conflicting definitions of types/functions/variables",
        re.compile(
            r"redefinition of"
            r"|redeclaration of"
            r"|duplicate member"
            r"|previously declared here",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "other_build",
        "Build failures not matching any specific pattern",
        re.compile(r".*"),  # catch-all
        None,
    ),
]

# ============================================================
# RUN_FAIL categories — checked in priority order
# Uses exit_code, stderr, AND stdout (critical for XSBench)
# ============================================================

RUN_FAIL_CATEGORIES = [
    (
        "wrong_checksum",
        "Program ran to completion but output checksum was incorrect",
        lambda exit_code, stderr, stdout: bool(
            re.search(r"INAVALID CHECKSUM|INVALID CHECKSUM", stdout or "", re.IGNORECASE)
        ),
    ),
    (
        "wrong_args",
        "Translated binary expects different arguments than spec provides",
        lambda exit_code, stderr, stdout: bool(
            re.search(r"[Uu]sage:", stderr or "")
            or re.search(r"[Uu]sage:", stdout or "")
            or re.search(r"<grid_rows|<sim_time|<temp_file|<power_file|<penalty>", stderr or "")
            or re.search(r"<inputfile>|<num of frames>", stdout or "")
        ),
    ),
    (
        "opencl_jit_error",
        "OpenCL runtime kernel compilation failed (clBuildProgram errors)",
        lambda exit_code, stderr, stdout: bool(
            re.search(r"\d+\s+errors? generated", stderr or "")
        ),
    ),
    (
        "simulation_mode_unsupported",
        "XSBench translated code used wrong simulation mode (history vs event)",
        lambda exit_code, stderr, stdout: bool(
            re.search(
                r"not implemented.*OpenMP|History-based simulation not implemented",
                stdout or "",
            )
        ),
    ),
    (
        "gpu_memory_error",
        "CUDA illegal memory access during kernel execution",
        lambda exit_code, stderr, stdout: bool(
            re.search(r"illegal memory access|GPUassert", stderr or "")
            or re.search(r"illegal memory access|GPUassert", stdout or "")
        ),
    ),
    (
        "segfault",
        "Process killed by SIGSEGV (signal 11) — null pointer or buffer overflow",
        lambda exit_code, stderr, stdout: exit_code == -11
        or bool(re.search(r"Segmentation fault|SIGSEGV|signal 11", stderr or "")),
    ),
    (
        "abort",
        "Process killed by SIGABRT — stack smashing, assertion failure, or abort()",
        lambda exit_code, stderr, stdout: exit_code == -6
        or bool(
            re.search(r"stack smashing|SIGABRT|Aborted|signal 6", stderr or "")
        ),
    ),
    (
        "timeout",
        "Process exceeded timeout limit",
        lambda exit_code, stderr, stdout: bool(
            re.search(r"TIMEOUT|exceeded timeout|killed.*timeout", stderr or "")
            or re.search(r"TIMEOUT", stdout or "")
        ),
    ),
    (
        "data_file_missing",
        "Required input data file not found at expected path",
        lambda exit_code, stderr, stdout: bool(
            re.search(r"error opening|file not found|cannot open", stdout or "")
            or re.search(r"error opening|file not found|cannot open", stderr or "")
        ),
    ),
    (
        "wrong_exit_code",
        "Non-zero exit code without crash signal or other specific error pattern",
        lambda exit_code, stderr, stdout: exit_code is not None and exit_code != 0,
    ),
    (
        "other_runtime",
        "Runtime failures not matching any specific pattern",
        lambda exit_code, stderr, stdout: True,  # catch-all
    ),
]

# ============================================================
# EXTRACTION_FAIL categories
# ============================================================

EXTRACTION_FAIL_CATEGORIES = [
    (
        "no_code_blocks",
        "LLM response did not contain parseable code for expected target files",
        re.compile(r"did not contain parseable code", re.IGNORECASE),
    ),
    (
        "missing_files",
        "Expected output files not found in LLM response",
        re.compile(
            r"expected file|not found in output|missing.*file|could not find|no .* file",
            re.IGNORECASE,
        ),
    ),
    (
        "malformed_output",
        "LLM response was malformed or could not be parsed",
        re.compile(r".*"),  # catch-all
    ),
]

# ============================================================
# VERIFY_FAIL categories — checked by verify_status + heuristics
# ============================================================

VERIFY_FAIL_CATEGORIES = [
    (
        "wrong_numerical_output",
        "Program produced output that did not match the expected stdout pattern",
    ),
    (
        "missing_output",
        "Program produced no stdout output (empty or whitespace-only)",
    ),
    (
        "verification_error",
        "Verification process itself errored (regex failure, harness issue)",
    ),
    (
        "pass_overall_mislabel",
        "Verify passed but overall_status is VERIFY_FAIL (pipeline labeling bug)",
    ),
    (
        "other_verify",
        "Verification failure not matching any specific pattern",
    ),
]


def extract_direction(data):
    """Infer translation direction from source_spec and target_spec."""
    src = data.get("source_spec", "")
    tgt = data.get("target_spec", "")
    # Extract API from spec id: last segment after the last hyphen
    # e.g., "rodinia-backprop-cuda" -> "cuda"
    src_api = src.rsplit("-", 1)[-1] if src else "unknown"
    tgt_api = tgt.rsplit("-", 1)[-1] if tgt else "unknown"
    return f"{src_api}-to-{tgt_api}"


def _check_missing_target_api(translated_files, direction):
    """Check if translated code is missing target API constructs entirely."""
    if not translated_files:
        return False
    target_api = direction.split("-to-")[-1] if "-to-" in direction else ""
    all_content = "\n".join(translated_files.values())
    if target_api == "omp":
        return not re.search(r"#pragma\s+omp\s+parallel|#pragma\s+omp\s+for", all_content)
    elif target_api == "opencl":
        return not re.search(r"__kernel\b|get_global_id", all_content)
    elif target_api == "cuda":
        return not re.search(r"__global__|<<<", all_content)
    elif target_api == "omp_target":
        return not re.search(r"#pragma\s+omp\s+target", all_content)
    return False


def classify_build_fail(data):
    """Classify a BUILD_FAIL result. Returns (primary, [secondaries])."""
    snippet = data.get("build_error_snippet") or ""
    direction = extract_direction(data)
    translated_files = data.get("translated_files") or {}

    primary = None
    secondaries = []

    for cat_name, _desc, pattern, direction_filter in BUILD_FAIL_CATEGORIES:
        if direction_filter is not None and not direction_filter(direction):
            continue
        if cat_name == "other_build":
            if primary is None:
                primary = cat_name
            continue
        if cat_name == "missing_target_api":
            if _check_missing_target_api(translated_files, direction):
                if primary is None:
                    primary = cat_name
                else:
                    secondaries.append(cat_name)
            continue
        if pattern.search(snippet):
            if primary is None:
                primary = cat_name
            else:
                secondaries.append(cat_name)

    return primary, secondaries


def classify_run_fail(data):
    """Classify a RUN_FAIL result. Returns (primary, [secondaries])."""
    exit_code = data.get("run_exit_code")
    stderr = data.get("run_stderr_snippet") or ""
    stdout = data.get("run_stdout_snippet") or ""

    primary = None
    secondaries = []

    for cat_name, _desc, check_fn in RUN_FAIL_CATEGORIES:
        if cat_name == "other_runtime":
            if primary is None:
                primary = cat_name
            continue
        if check_fn(exit_code, stderr, stdout):
            if primary is None:
                primary = cat_name
            elif cat_name != "wrong_exit_code":
                secondaries.append(cat_name)

    return primary, secondaries


def classify_extraction_fail(data):
    """Classify an EXTRACTION_FAIL result. Returns (primary, [secondaries])."""
    error_msg = data.get("error_message") or ""

    for cat_name, _desc, pattern in EXTRACTION_FAIL_CATEGORIES:
        if pattern.search(error_msg):
            return cat_name, []

    return "malformed_output", []


def classify_verify_fail(data):
    """Classify a VERIFY_FAIL result. Returns (primary, [secondaries])."""
    verify_status = data.get("verify_status") or ""
    stdout = data.get("run_stdout_snippet") or ""

    secondaries = []

    # Check for data file errors in stdout (secondary signal)
    if re.search(r"error opening|file not found|cannot open", stdout, re.IGNORECASE):
        secondaries.append("data_file_error")

    if verify_status == "pass":
        # verify_status says pass but overall_status is VERIFY_FAIL — pipeline labeling bug
        return "pass_overall_mislabel", secondaries
    elif verify_status == "error":
        # Verification process itself failed (harness/regex issue)
        return "verification_error", secondaries
    elif verify_status == "fail":
        # Pattern didn't match — distinguish empty vs wrong output
        if not stdout.strip():
            return "missing_output", secondaries
        return "wrong_numerical_output", secondaries
    else:
        return "other_verify", secondaries


def load_results(project_root):
    """Load all result JSONs from results/evaluation/{model}/."""
    eval_dir = Path(project_root) / "results" / "evaluation"
    results = []

    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for json_file in sorted(model_dir.glob("*.json")):
            basename = json_file.name
            # Skip non-result files
            if basename.startswith("batch_") or basename == "eval_summary.json":
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                data["_file_path"] = str(json_file)
                results.append(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"WARNING: Could not read {json_file}: {e}", file=sys.stderr)

    return results


def build_taxonomy(results):
    """Classify all results and build the taxonomy data structure."""
    taxonomy = {
        "total_results": len(results),
        "total_pass": 0,
        "total_failures": 0,
        "status_counts": Counter(),
        "build_fail_categories": defaultdict(lambda: {"count": 0, "examples": [], "by_model": Counter(), "by_direction": Counter(), "by_kernel": Counter()}),
        "run_fail_categories": defaultdict(lambda: {"count": 0, "examples": [], "by_model": Counter(), "by_direction": Counter(), "by_kernel": Counter()}),
        "extraction_fail_categories": defaultdict(lambda: {"count": 0, "examples": [], "by_model": Counter(), "by_direction": Counter(), "by_kernel": Counter()}),
        "verify_fail_categories": defaultdict(lambda: {"count": 0, "examples": [], "by_model": Counter(), "by_direction": Counter(), "by_kernel": Counter()}),
        "error_count": 0,
        "error_examples": [],
        "per_kernel": defaultdict(lambda: {"total": 0, "pass": 0, "build_fail": Counter(), "run_fail": Counter(), "extraction_fail": Counter(), "verify_fail": Counter(), "error": 0}),
        "per_model": defaultdict(lambda: {"total": 0, "pass": 0, "build_fail": Counter(), "run_fail": Counter(), "extraction_fail": Counter(), "verify_fail": Counter(), "error": 0}),
        "per_direction": defaultdict(lambda: {"total": 0, "pass": 0, "build_fail": Counter(), "run_fail": Counter(), "extraction_fail": Counter(), "verify_fail": Counter(), "error": 0}),
        "classified_results": [],
    }

    for data in results:
        status = data.get("overall_status", "UNKNOWN")
        model = data.get("model", "unknown")
        kernel = data.get("kernel", "unknown")
        direction = extract_direction(data)
        aug_level = data.get("augment_level", 0)

        taxonomy["status_counts"][status] += 1
        taxonomy["per_kernel"][kernel]["total"] += 1
        taxonomy["per_model"][model]["total"] += 1
        taxonomy["per_direction"][direction]["total"] += 1

        if status == "PASS":
            taxonomy["total_pass"] += 1
            taxonomy["per_kernel"][kernel]["pass"] += 1
            taxonomy["per_model"][model]["pass"] += 1
            taxonomy["per_direction"][direction]["pass"] += 1
            continue

        taxonomy["total_failures"] += 1

        record = {
            "file": data.get("_file_path", ""),
            "model": model,
            "kernel": kernel,
            "direction": direction,
            "augment_level": aug_level,
            "overall_status": status,
            "primary_category": None,
            "secondary_categories": [],
        }

        if status == "BUILD_FAIL":
            primary, secondaries = classify_build_fail(data)
            record["primary_category"] = primary
            record["secondary_categories"] = secondaries

            cat_data = taxonomy["build_fail_categories"][primary]
            cat_data["count"] += 1
            cat_data["by_model"][model] += 1
            cat_data["by_direction"][direction] += 1
            cat_data["by_kernel"][kernel] += 1
            if len(cat_data["examples"]) < MAX_EXAMPLES_PER_CATEGORY:
                snippet = data.get("build_error_snippet") or ""
                cat_data["examples"].append({
                    "file": os.path.basename(data.get("_file_path", "")),
                    "model": model,
                    "kernel": kernel,
                    "direction": direction,
                    "snippet_preview": snippet[:300] if snippet else "(empty)",
                })

            taxonomy["per_kernel"][kernel]["build_fail"][primary] += 1
            taxonomy["per_model"][model]["build_fail"][primary] += 1
            taxonomy["per_direction"][direction]["build_fail"][primary] += 1

        elif status == "RUN_FAIL":
            primary, secondaries = classify_run_fail(data)
            record["primary_category"] = primary
            record["secondary_categories"] = secondaries

            cat_data = taxonomy["run_fail_categories"][primary]
            cat_data["count"] += 1
            cat_data["by_model"][model] += 1
            cat_data["by_direction"][direction] += 1
            cat_data["by_kernel"][kernel] += 1
            if len(cat_data["examples"]) < MAX_EXAMPLES_PER_CATEGORY:
                stderr = data.get("run_stderr_snippet") or ""
                stdout = data.get("run_stdout_snippet") or ""
                cat_data["examples"].append({
                    "file": os.path.basename(data.get("_file_path", "")),
                    "model": model,
                    "kernel": kernel,
                    "direction": direction,
                    "stderr_preview": stderr[:200] if stderr else "(empty)",
                    "stdout_preview": stdout[:200] if stdout else "(empty)",
                    "exit_code": data.get("run_exit_code"),
                })

            taxonomy["per_kernel"][kernel]["run_fail"][primary] += 1
            taxonomy["per_model"][model]["run_fail"][primary] += 1
            taxonomy["per_direction"][direction]["run_fail"][primary] += 1

        elif status == "EXTRACTION_FAIL":
            primary, secondaries = classify_extraction_fail(data)
            record["primary_category"] = primary
            record["secondary_categories"] = secondaries

            cat_data = taxonomy["extraction_fail_categories"][primary]
            cat_data["count"] += 1
            cat_data["by_model"][model] += 1
            cat_data["by_direction"][direction] += 1
            cat_data["by_kernel"][kernel] += 1
            if len(cat_data["examples"]) < MAX_EXAMPLES_PER_CATEGORY:
                cat_data["examples"].append({
                    "file": os.path.basename(data.get("_file_path", "")),
                    "model": model,
                    "kernel": kernel,
                    "direction": direction,
                    "error_message": (data.get("error_message") or "")[:200],
                })

            taxonomy["per_kernel"][kernel]["extraction_fail"][primary] += 1
            taxonomy["per_model"][model]["extraction_fail"][primary] += 1
            taxonomy["per_direction"][direction]["extraction_fail"][primary] += 1

        elif status == "VERIFY_FAIL":
            primary, secondaries = classify_verify_fail(data)
            record["primary_category"] = primary
            record["secondary_categories"] = secondaries

            cat_data = taxonomy["verify_fail_categories"][primary]
            cat_data["count"] += 1
            cat_data["by_model"][model] += 1
            cat_data["by_direction"][direction] += 1
            cat_data["by_kernel"][kernel] += 1
            if len(cat_data["examples"]) < MAX_EXAMPLES_PER_CATEGORY:
                stdout = data.get("run_stdout_snippet") or ""
                cat_data["examples"].append({
                    "file": os.path.basename(data.get("_file_path", "")),
                    "model": model,
                    "kernel": kernel,
                    "direction": direction,
                    "verify_status": data.get("verify_status", ""),
                    "stdout_preview": stdout[:200] if stdout else "(empty)",
                    "exit_code": data.get("run_exit_code"),
                })

            taxonomy["per_kernel"][kernel]["verify_fail"][primary] += 1
            taxonomy["per_model"][model]["verify_fail"][primary] += 1
            taxonomy["per_direction"][direction]["verify_fail"][primary] += 1

        elif status == "ERROR":
            record["primary_category"] = "api_error"
            record["secondary_categories"] = []
            taxonomy["error_count"] += 1
            taxonomy["error_examples"].append({
                "file": os.path.basename(data.get("_file_path", "")),
                "model": model,
                "kernel": kernel,
                "direction": direction,
                "error_message": (data.get("error_message") or "")[:200],
            })
            taxonomy["per_kernel"][kernel]["error"] += 1
            taxonomy["per_model"][model]["error"] += 1
            taxonomy["per_direction"][direction]["error"] += 1

        taxonomy["classified_results"].append(record)

    return taxonomy


def serialize_taxonomy(taxonomy):
    """Convert taxonomy to JSON-serializable dict."""
    out = {
        "total_results": taxonomy["total_results"],
        "total_pass": taxonomy["total_pass"],
        "total_failures": taxonomy["total_failures"],
        "status_counts": dict(taxonomy["status_counts"]),
        "build_fail_categories": {},
        "run_fail_categories": {},
        "extraction_fail_categories": {},
        "verify_fail_categories": {},
        "error_count": taxonomy["error_count"],
        "error_examples": taxonomy["error_examples"],
        "per_kernel": {},
        "per_model": {},
        "per_direction": {},
        "classified_results": taxonomy["classified_results"],
    }

    # Convert category dicts
    for cat_type in ("build_fail_categories", "run_fail_categories", "extraction_fail_categories", "verify_fail_categories"):
        for cat_name, cat_data in taxonomy[cat_type].items():
            out[cat_type][cat_name] = {
                "count": cat_data["count"],
                "examples": cat_data["examples"],
                "by_model": dict(cat_data["by_model"]),
                "by_direction": dict(cat_data["by_direction"]),
                "by_kernel": dict(cat_data["by_kernel"]),
            }

    # Convert per-* dicts
    for group_type in ("per_kernel", "per_model", "per_direction"):
        for key, val in taxonomy[group_type].items():
            out[group_type][key] = {
                "total": val["total"],
                "pass": val["pass"],
                "build_fail": dict(val["build_fail"]),
                "run_fail": dict(val["run_fail"]),
                "extraction_fail": dict(val["extraction_fail"]),
                "verify_fail": dict(val["verify_fail"]),
                "error": val["error"],
            }

    return out


def generate_markdown(taxonomy, output_path):
    """Generate publication-ready markdown tables."""
    lines = []
    lines.append("# Error Taxonomy — ParBench LLM Evaluation Results")
    lines.append("")
    lines.append(f"**Total results:** {taxonomy['total_results']}  ")
    lines.append(f"**PASS:** {taxonomy['total_pass']} ({safe_percentage(taxonomy['total_pass'], taxonomy['total_results']):.1f}%)  ")
    lines.append(f"**Failures:** {taxonomy['total_failures']} ({safe_percentage(taxonomy['total_failures'], taxonomy['total_results']):.1f}%)")
    lines.append("")

    # Table 1: Overall status distribution
    lines.append("## Table 1: Overall Status Distribution")
    lines.append("")
    lines.append("| Status | Count | % of Total |")
    lines.append("|--------|------:|----------:|")
    for status in ["PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL", "ERROR"]:
        count = taxonomy["status_counts"].get(status, 0)
        pct = safe_percentage(count, taxonomy["total_results"])
        lines.append(f"| {status} | {count} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{taxonomy['total_results']}** | **100.0%** |")
    lines.append("")

    # Table 2: BUILD_FAIL categories
    lines.append("## Table 2: BUILD_FAIL Root Cause Categories")
    lines.append("")
    total_bf = sum(c["count"] for c in taxonomy["build_fail_categories"].values())
    n_bf_cats = len(taxonomy['build_fail_categories'])
    lines.append(f"*{total_bf} total BUILD_FAIL results classified into {n_bf_cats} {'category' if n_bf_cats == 1 else 'categories'}.*")
    lines.append("")
    lines.append("| # | Category | Count | % of BUILD_FAIL | Description |")
    lines.append("|--:|----------|------:|----------------:|-------------|")
    # Get category descriptions
    cat_descs = {name: desc for name, desc, _, _ in BUILD_FAIL_CATEGORIES}
    sorted_bf = sorted(taxonomy["build_fail_categories"].items(), key=lambda x: -x[1]["count"])
    for i, (cat_name, cat_data) in enumerate(sorted_bf, 1):
        count = cat_data["count"]
        pct = safe_percentage(count, total_bf)
        desc = cat_descs.get(cat_name, "")
        lines.append(f"| {i} | `{cat_name}` | {count} | {pct:.1f}% | {desc} |")
    lines.append(f"| | **Total** | **{total_bf}** | **100.0%** | |")
    lines.append("")

    # Table 3: RUN_FAIL categories
    lines.append("## Table 3: RUN_FAIL Root Cause Categories")
    lines.append("")
    total_rf = sum(c["count"] for c in taxonomy["run_fail_categories"].values())
    n_rf_cats = len(taxonomy['run_fail_categories'])
    lines.append(f"*{total_rf} total RUN_FAIL results classified into {n_rf_cats} {'category' if n_rf_cats == 1 else 'categories'}.*")
    lines.append("")
    lines.append("| # | Category | Count | % of RUN_FAIL | Description |")
    lines.append("|--:|----------|------:|-------------:|-------------|")
    cat_descs_rf = {name: desc for name, desc, _ in RUN_FAIL_CATEGORIES}
    sorted_rf = sorted(taxonomy["run_fail_categories"].items(), key=lambda x: -x[1]["count"])
    for i, (cat_name, cat_data) in enumerate(sorted_rf, 1):
        count = cat_data["count"]
        pct = safe_percentage(count, total_rf)
        desc = cat_descs_rf.get(cat_name, "")
        lines.append(f"| {i} | `{cat_name}` | {count} | {pct:.1f}% | {desc} |")
    lines.append(f"| | **Total** | **{total_rf}** | **100.0%** | |")
    lines.append("")

    # Table 4: EXTRACTION_FAIL categories
    lines.append("## Table 4: EXTRACTION_FAIL Root Cause Categories")
    lines.append("")
    total_ef = sum(c["count"] for c in taxonomy["extraction_fail_categories"].values())
    n_ef_cats = len(taxonomy['extraction_fail_categories'])
    lines.append(f"*{total_ef} total EXTRACTION_FAIL results classified into {n_ef_cats} {'category' if n_ef_cats == 1 else 'categories'}.*")
    lines.append("")
    lines.append("| # | Category | Count | % of EXTRACTION_FAIL | Description |")
    lines.append("|--:|----------|------:|--------------------:|-------------|")
    cat_descs_ef = {name: desc for name, desc, _ in EXTRACTION_FAIL_CATEGORIES}
    sorted_ef = sorted(taxonomy["extraction_fail_categories"].items(), key=lambda x: -x[1]["count"])
    for i, (cat_name, cat_data) in enumerate(sorted_ef, 1):
        count = cat_data["count"]
        pct = safe_percentage(count, total_ef)
        desc = cat_descs_ef.get(cat_name, "")
        lines.append(f"| {i} | `{cat_name}` | {count} | {pct:.1f}% | {desc} |")
    lines.append(f"| | **Total** | **{total_ef}** | **100.0%** | |")
    lines.append("")

    # Table 4b: VERIFY_FAIL categories
    lines.append("## Table 4b: VERIFY_FAIL Root Cause Categories")
    lines.append("")
    total_vf = sum(c["count"] for c in taxonomy.get("verify_fail_categories", {}).values())
    vf_cats_data = taxonomy.get("verify_fail_categories", {})
    n_vf_cats = len(vf_cats_data)
    lines.append(f"*{total_vf} total VERIFY_FAIL results classified into {n_vf_cats} {'category' if n_vf_cats == 1 else 'categories'}.*")
    lines.append("")
    if total_vf > 0:
        lines.append("| # | Category | Count | % of VERIFY_FAIL | Description |")
        lines.append("|--:|----------|------:|----------------:|-------------|")
        cat_descs_vf = {name: desc for name, desc in VERIFY_FAIL_CATEGORIES}
        sorted_vf = sorted(vf_cats_data.items(), key=lambda x: -x[1]["count"])
        for i, (cat_name, cat_data) in enumerate(sorted_vf, 1):
            count = cat_data["count"]
            pct = safe_percentage(count, total_vf)
            desc = cat_descs_vf.get(cat_name, "")
            lines.append(f"| {i} | `{cat_name}` | {count} | {pct:.1f}% | {desc} |")
        lines.append(f"| | **Total** | **{total_vf}** | **100.0%** | |")
    lines.append("")

    # Table 5: BUILD_FAIL by model
    lines.append("## Table 5: BUILD_FAIL Categories by Model")
    lines.append("")
    models = sorted(taxonomy["per_model"].keys())
    bf_cats = [cat for cat, _ in sorted_bf]
    header = "| Category | " + " | ".join(models) + " | Total |"
    sep = "|----------|" + "|".join(["------:" for _ in models]) + "|------:|"
    lines.append(header)
    lines.append(sep)
    for cat_name in bf_cats:
        cat_data = taxonomy["build_fail_categories"][cat_name]
        row = f"| `{cat_name}` |"
        for m in models:
            row += f" {cat_data['by_model'].get(m, 0)} |"
        row += f" {cat_data['count']} |"
        lines.append(row)
    # Total row
    row = "| **Total** |"
    for m in models:
        total_m = sum(taxonomy["per_model"][m]["build_fail"].values())
        row += f" **{total_m}** |"
    row += f" **{total_bf}** |"
    lines.append(row)
    lines.append("")

    # Table 6: BUILD_FAIL by direction
    lines.append("## Table 6: BUILD_FAIL Categories by Direction")
    lines.append("")
    directions = sorted(taxonomy["per_direction"].keys(),
                       key=lambda d: sum(taxonomy["per_direction"][d]["build_fail"].values()),
                       reverse=True)
    # Only include directions with BUILD_FAIL
    directions = [d for d in directions if sum(taxonomy["per_direction"][d]["build_fail"].values()) > 0]
    header = "| Category | " + " | ".join(d.replace("-to-", "→") for d in directions) + " | Total |"
    sep = "|----------|" + "|".join(["------:" for _ in directions]) + "|------:|"
    lines.append(header)
    lines.append(sep)
    for cat_name in bf_cats:
        cat_data = taxonomy["build_fail_categories"][cat_name]
        row = f"| `{cat_name}` |"
        for d in directions:
            row += f" {cat_data['by_direction'].get(d, 0)} |"
        row += f" {cat_data['count']} |"
        lines.append(row)
    # Total row
    row = "| **Total** |"
    for d in directions:
        total_d = sum(taxonomy["per_direction"][d]["build_fail"].values())
        row += f" **{total_d}** |"
    row += f" **{total_bf}** |"
    lines.append(row)
    lines.append("")

    # Table 7: RUN_FAIL by model
    lines.append("## Table 7: RUN_FAIL Categories by Model")
    lines.append("")
    rf_cats = [cat for cat, _ in sorted_rf]
    header = "| Category | " + " | ".join(models) + " | Total |"
    sep = "|----------|" + "|".join(["------:" for _ in models]) + "|------:|"
    lines.append(header)
    lines.append(sep)
    for cat_name in rf_cats:
        cat_data = taxonomy["run_fail_categories"][cat_name]
        row = f"| `{cat_name}` |"
        for m in models:
            row += f" {cat_data['by_model'].get(m, 0)} |"
        row += f" {cat_data['count']} |"
        lines.append(row)
    row = "| **Total** |"
    for m in models:
        total_m = sum(taxonomy["per_model"][m]["run_fail"].values())
        row += f" **{total_m}** |"
    row += f" **{total_rf}** |"
    lines.append(row)
    lines.append("")

    # Table 8: Per-model summary (all statuses)
    lines.append("## Table 8: Complete Status Distribution by Model")
    lines.append("")
    header = "| Model | Total | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL | ERROR | Pass Rate |"
    sep = "|-------|------:|-----:|----------:|--------:|----------:|--------------:|------:|----------:|"
    lines.append(header)
    lines.append(sep)
    for m in models:
        md = taxonomy["per_model"][m]
        total = md["total"]
        p = md["pass"]
        bf = sum(md["build_fail"].values())
        rf = sum(md["run_fail"].values())
        vf = sum(md["verify_fail"].values())
        ef = sum(md["extraction_fail"].values())
        er = md["error"]
        rate = safe_percentage(p, total)
        lines.append(f"| {m} | {total} | {p} | {bf} | {rf} | {vf} | {ef} | {er} | {rate:.1f}% |")
    lines.append("")

    # Table 9: Per-direction summary
    lines.append("## Table 9: Complete Status Distribution by Direction")
    lines.append("")
    all_dirs = sorted(taxonomy["per_direction"].keys())
    header = "| Direction | Total | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL | ERROR | Pass Rate |"
    sep = "|-----------|------:|-----:|----------:|--------:|----------:|--------------:|------:|----------:|"
    lines.append(header)
    lines.append(sep)
    for d in all_dirs:
        dd = taxonomy["per_direction"][d]
        total = dd["total"]
        p = dd["pass"]
        bf = sum(dd["build_fail"].values())
        rf = sum(dd["run_fail"].values())
        vf = sum(dd["verify_fail"].values())
        ef = sum(dd["extraction_fail"].values())
        er = dd["error"]
        rate = safe_percentage(p, total)
        lines.append(f"| {d.replace('-to-', '→')} | {total} | {p} | {bf} | {rf} | {vf} | {ef} | {er} | {rate:.1f}% |")
    lines.append("")

    # Table 10: Per-kernel summary (top-level)
    lines.append("## Table 10: Per-Kernel Pass Rate and Primary Failure Mode")
    lines.append("")
    header = "| Kernel | Total | PASS | Pass Rate | Primary Failure Mode | Primary Failure Count |"
    sep = "|--------|------:|-----:|----------:|---------------------|---------------------:|"
    lines.append(header)
    lines.append(sep)
    for k in sorted(taxonomy["per_kernel"].keys()):
        kd = taxonomy["per_kernel"][k]
        total = kd["total"]
        p = kd["pass"]
        rate = safe_percentage(p, total)
        # Find primary failure mode across all failure types
        all_fails = Counter()
        all_fails.update(kd["build_fail"])
        all_fails.update(kd["run_fail"])
        all_fails.update(kd["verify_fail"])
        all_fails.update(kd["extraction_fail"])
        if kd["error"] > 0:
            all_fails["api_error"] = kd["error"]
        if all_fails:
            top_fail, top_count = all_fails.most_common(1)[0]
        else:
            top_fail, top_count = "—", 0
        lines.append(f"| {k} | {total} | {p} | {rate:.1f}% | `{top_fail}` | {top_count} |")
    lines.append("")

    # Summary findings
    lines.append("## Key Findings")
    lines.append("")

    # Top 3 BUILD_FAIL
    lines.append("### Top BUILD_FAIL Root Causes")
    lines.append("")
    for i, (cat_name, cat_data) in enumerate(sorted_bf[:3], 1):
        pct = safe_percentage(cat_data["count"], total_bf)
        lines.append(f"{i}. **`{cat_name}`** — {cat_data['count']} ({pct:.1f}% of BUILD_FAIL)")
    lines.append("")

    # Top 3 RUN_FAIL
    lines.append("### Top RUN_FAIL Root Causes")
    lines.append("")
    for i, (cat_name, cat_data) in enumerate(sorted_rf[:3], 1):
        pct = safe_percentage(cat_data["count"], total_rf)
        lines.append(f"{i}. **`{cat_name}`** — {cat_data['count']} ({pct:.1f}% of RUN_FAIL)")
    lines.append("")

    # Top VERIFY_FAIL
    if total_vf > 0:
        lines.append("### Top VERIFY_FAIL Root Causes")
        lines.append("")
        for i, (cat_name, cat_data) in enumerate(sorted_vf[:3], 1):
            pct = safe_percentage(cat_data["count"], total_vf)
            lines.append(f"{i}. **`{cat_name}`** — {cat_data['count']} ({pct:.1f}% of VERIFY_FAIL)")
        lines.append("")

    # Direction asymmetry
    lines.append("### Direction Asymmetry")
    lines.append("")
    for d in all_dirs:
        dd = taxonomy["per_direction"][d]
        rate = safe_percentage(dd["pass"], dd["total"])
        lines.append(f"- {d.replace('-to-', '→')}: {dd['pass']}/{dd['total']} PASS ({rate:.1f}%)")
    lines.append("")

    # Compound failures (secondary categories)
    lines.append("### Compound Failures (Multiple Root Causes)")
    lines.append("")
    compound = [r for r in taxonomy["classified_results"] if r["secondary_categories"]]
    lines.append(f"**{len(compound)}** results ({safe_percentage(len(compound), taxonomy['total_failures']):.1f}% of failures) have secondary error categories.")
    lines.append("")
    if compound:
        # Count secondary category co-occurrences
        secondary_counts = Counter()
        for r in compound:
            for s in r["secondary_categories"]:
                secondary_counts[f"{r['primary_category']} + {s}"] += 1
        lines.append("| Primary + Secondary | Count |")
        lines.append("|---------------------|------:|")
        for combo, count in secondary_counts.most_common(10):
            lines.append(f"| {combo} | {count} |")
    lines.append("")

    # Canonical vs Augmentation split (if present)
    if "canonical" in taxonomy and "augmentation" in taxonomy:
        lines.append("---")
        lines.append("")
        lines.append("## Canonical Taxonomy (L0)")
        lines.append("")
        can = taxonomy["canonical"]
        lines.append(f"**Total:** {can['total_results']}  ")
        lines.append(f"**PASS:** {can['total_pass']} ({safe_percentage(can['total_pass'], can['total_results']):.1f}%)  ")
        lines.append(f"**Failures:** {can['total_failures']} ({safe_percentage(can['total_failures'], can['total_results']):.1f}%)")
        lines.append("")
        lines.append("| Status | Count | % |")
        lines.append("|--------|------:|--:|")
        for status in ["PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL", "ERROR"]:
            count = can["status_counts"].get(status, 0)
            if count > 0:
                lines.append(f"| {status} | {count} | {safe_percentage(count, can['total_results']):.1f}% |")
        lines.append("")

        lines.append("## Augmentation Taxonomy (L1-L4)")
        lines.append("")
        aug = taxonomy["augmentation"]
        lines.append(f"**Total:** {aug['total_results']}  ")
        lines.append(f"**PASS:** {aug['total_pass']} ({safe_percentage(aug['total_pass'], aug['total_results']):.1f}%)  ")
        lines.append(f"**Failures:** {aug['total_failures']} ({safe_percentage(aug['total_failures'], aug['total_results']):.1f}%)")
        lines.append("")

        if "augmentation_by_level" in taxonomy:
            lines.append("| Level | Total | PASS | Pass Rate |")
            lines.append("|-------|------:|-----:|----------:|")
            for lvl in ["L1", "L2", "L3", "L4"]:
                lvl_data = taxonomy["augmentation_by_level"].get(lvl, {})
                lt = lvl_data.get("total_results", 0)
                lp = lvl_data.get("total_pass", 0)
                lines.append(f"| {lvl} | {lt} | {lp} | {safe_percentage(lp, lt):.1f}% |")
            lines.append("")

        if "comparison" in taxonomy:
            comp = taxonomy["comparison"]
            lines.append("## Comparison: Canonical vs Augmentation")
            lines.append("")
            lines.append(f"- Canonical pass rate: {comp.get('canonical_pass_rate', 0)}%")
            lines.append(f"- Augmentation pass rate: {comp.get('augmentation_pass_rate', 0)}%")
            lines.append(f"- Delta: {comp.get('delta_pp', 0):+.1f} pp")
            lines.append(f"- *{comp.get('note', '')}*")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Build error taxonomy from eval results")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {project_root / 'results' / 'evaluation'}...")
    results = load_results(project_root)
    print(f"Loaded {len(results)} result JSONs")

    # Partition into canonical (L0) and augmentation (L1-L4)
    canonical_results = [r for r in results if r.get("augment_level", 0) == 0]
    augmentation_results = [r for r in results if r.get("augment_level", 0) > 0]
    print(f"  Canonical (L0): {len(canonical_results)}, Augmentation (L1-L4): {len(augmentation_results)}")

    print("Classifying canonical failures...")
    canonical_taxonomy = build_taxonomy(canonical_results)

    print("Classifying augmentation failures...")
    augmentation_taxonomy = build_taxonomy(augmentation_results)

    # Per-level augmentation breakdowns
    augmentation_by_level = {}
    for level in [1, 2, 3, 4]:
        level_results = [r for r in augmentation_results if r.get("augment_level") == level]
        if level_results:
            augmentation_by_level[f"L{level}"] = serialize_taxonomy(build_taxonomy(level_results))

    # Comparison
    can_pass_rate = safe_percentage(canonical_taxonomy["total_pass"], canonical_taxonomy["total_results"])
    aug_pass_rate = safe_percentage(augmentation_taxonomy["total_pass"], augmentation_taxonomy["total_results"])
    comparison = {
        "canonical_pass_rate": round(can_pass_rate, 1),
        "augmentation_pass_rate": round(aug_pass_rate, 1),
        "delta_pp": round(aug_pass_rate - can_pass_rate, 1),
        "note": "Augmentation runs only on tasks that passed canonical, so higher pass rate is expected",
    }

    # Also build combined taxonomy for backward compatibility
    print("Classifying all failures (combined)...")
    taxonomy = build_taxonomy(results)

    # Serialize and write JSON
    json_path = output_dir / "error_taxonomy.json"
    serialized = serialize_taxonomy(taxonomy)
    serialized["canonical"] = serialize_taxonomy(canonical_taxonomy)
    serialized["augmentation"] = serialize_taxonomy(augmentation_taxonomy)
    serialized["augmentation_by_level"] = augmentation_by_level
    serialized["comparison"] = comparison
    with open(json_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"Wrote {json_path}")

    # Generate markdown
    md_path = output_dir / "error_taxonomy.md"
    generate_markdown(serialized, md_path)
    print(f"Wrote {md_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total results:  {taxonomy['total_results']}")
    print(f"PASS:           {taxonomy['total_pass']} ({safe_percentage(taxonomy['total_pass'], taxonomy['total_results']):.1f}%)")
    print(f"Failures:       {taxonomy['total_failures']} ({safe_percentage(taxonomy['total_failures'], taxonomy['total_results']):.1f}%)")
    print()

    # BUILD_FAIL
    total_bf = sum(c["count"] for c in taxonomy["build_fail_categories"].values())
    print(f"BUILD_FAIL ({total_bf}):")
    sorted_bf = sorted(taxonomy["build_fail_categories"].items(), key=lambda x: -x[1]["count"])
    for cat_name, cat_data in sorted_bf:
        pct = safe_percentage(cat_data["count"], total_bf)
        print(f"  {cat_name:30s} {cat_data['count']:4d} ({pct:5.1f}%)")

    # RUN_FAIL
    total_rf = sum(c["count"] for c in taxonomy["run_fail_categories"].values())
    print(f"\nRUN_FAIL ({total_rf}):")
    sorted_rf = sorted(taxonomy["run_fail_categories"].items(), key=lambda x: -x[1]["count"])
    for cat_name, cat_data in sorted_rf:
        pct = safe_percentage(cat_data["count"], total_rf)
        print(f"  {cat_name:30s} {cat_data['count']:4d} ({pct:5.1f}%)")

    # EXTRACTION_FAIL
    total_ef = sum(c["count"] for c in taxonomy["extraction_fail_categories"].values())
    print(f"\nEXTRACTION_FAIL ({total_ef}):")
    for cat_name, cat_data in taxonomy["extraction_fail_categories"].items():
        pct = safe_percentage(cat_data["count"], total_ef)
        print(f"  {cat_name:30s} {cat_data['count']:4d} ({pct:5.1f}%)")

    # VERIFY_FAIL
    total_vf = sum(c["count"] for c in taxonomy["verify_fail_categories"].values())
    if total_vf > 0:
        print(f"\nVERIFY_FAIL ({total_vf}):")
        sorted_vf = sorted(taxonomy["verify_fail_categories"].items(), key=lambda x: -x[1]["count"])
        for cat_name, cat_data in sorted_vf:
            pct = safe_percentage(cat_data["count"], total_vf)
            print(f"  {cat_name:30s} {cat_data['count']:4d} ({pct:5.1f}%)")

    if taxonomy["error_count"] > 0:
        print(f"\nERROR: {taxonomy['error_count']} (API/infrastructure errors)")

    # Validation
    total_classified = sum(c["count"] for c in taxonomy["build_fail_categories"].values())
    total_classified += sum(c["count"] for c in taxonomy["run_fail_categories"].values())
    total_classified += sum(c["count"] for c in taxonomy["extraction_fail_categories"].values())
    total_classified += sum(c["count"] for c in taxonomy["verify_fail_categories"].values())
    total_classified += taxonomy["error_count"]

    print(f"\nVALIDATION: {total_classified} classified vs {taxonomy['total_failures']} failures", end="")
    if total_classified == taxonomy["total_failures"]:
        print(" ✓ MATCH")
    else:
        print(f" ✗ MISMATCH (delta={total_classified - taxonomy['total_failures']})")
        return 1

    # Check 'other' buckets
    other_bf = taxonomy["build_fail_categories"].get("other_build", {})
    other_bf_count = other_bf.get("count", 0) if isinstance(other_bf, dict) else other_bf["count"]
    other_bf_pct = safe_percentage(other_bf_count, total_bf)
    other_rf = taxonomy["run_fail_categories"].get("other_runtime", {})
    other_rf_count = other_rf.get("count", 0) if isinstance(other_rf, dict) else other_rf["count"]
    other_rf_pct = safe_percentage(other_rf_count, total_rf)

    print(f"\n'other_build' bucket: {other_bf_count}/{total_bf} ({other_bf_pct:.1f}%)", end="")
    if other_bf_pct > OTHER_BUCKET_REFINEMENT_THRESHOLD:
        print(" ⚠ NEEDS REFINEMENT (>20%)")
    else:
        print(" ✓ OK")

    print(f"'other_runtime' bucket: {other_rf_count}/{total_rf} ({other_rf_pct:.1f}%)", end="")
    if other_rf_pct > OTHER_BUCKET_REFINEMENT_THRESHOLD:
        print(" ⚠ NEEDS REFINEMENT (>20%)")
    else:
        print(" ✓ OK")

    other_vf = taxonomy["verify_fail_categories"].get("other_verify", {})
    other_vf_count = other_vf.get("count", 0) if isinstance(other_vf, dict) else other_vf["count"]
    other_vf_pct = safe_percentage(other_vf_count, total_vf)
    if total_vf > 0:
        print(f"'other_verify' bucket: {other_vf_count}/{total_vf} ({other_vf_pct:.1f}%)", end="")
        if other_vf_pct > OTHER_BUCKET_REFINEMENT_THRESHOLD:
            print(" ⚠ NEEDS REFINEMENT (>20%)")
        else:
            print(" ✓ OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
