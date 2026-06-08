#!/usr/bin/env python3
"""Analyze CUDA batch harness results from JSON logs.

Reads individual .json files from results/phase3/cuda_batch3_logs/
and produces a comprehensive analysis report.

Usage:
    cd .
    source env_parbench/bin/activate
    python scripts/analyze_cuda_batch.py
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(".")
LOG_DIR = PROJECT_ROOT / "results" / "phase3" / "cuda_batch3_logs"
REPORT_PATH = PROJECT_ROOT / "results" / "phase3" / "cuda_batch3_analysis.md"
REPORT_JSON = PROJECT_ROOT / "results" / "phase3" / "cuda_batch3_results.json"

# All 40 kernels we expect
BATCH2 = [
    "babelstream", "backprop", "ccsd-trpdrv", "deredundancy", "feynman-kac",
    "fpc", "ga", "keccaktreehash", "laplace3d", "lulesh",
    "maxpool3d", "md5hash", "pathfinder", "pso", "rmsnorm",
    "secp256k1", "softmax-online", "thomas", "tissue", "tsp",
]
BATCH3 = [
    "bezier-surface", "convolution3d", "crc64", "floydwarshall", "gaussian",
    "geglu", "heat2d", "iso2dfd", "jenkins-hash", "knn",
    "mandelbrot", "mis", "murmurhash3", "myocyte", "nw",
    "perplexity", "popcount", "sobol", "stencil1d", "triad",
]
ALL_KERNELS = BATCH2 + BATCH3


def load_json_result(kernel: str) -> dict | None:
    """Load the JSON result file for a kernel."""
    path = LOG_DIR / f"{kernel}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  WARNING: Failed to parse {path}: {e}", file=sys.stderr)
        return None


def load_log_result(kernel: str) -> str:
    """Load the full text log for a kernel."""
    path = LOG_DIR / f"{kernel}.log"
    if not path.exists():
        return ""
    return path.read_text()


def classify_result(data: dict | None, log_text: str) -> dict:
    """Classify a kernel result into structured fields."""
    result = {
        "kernel": "",
        "build_status": "MISSING",
        "build_time": 0.0,
        "build_stderr": "",
        "run_status": "MISSING",
        "run_time": 0.0,
        "run_exit_code": None,
        "run_stdout": "",
        "run_stderr": "",
        "verify_status": "MISSING",
        "verify_strategy": "",
        "verify_details": "",
        "metrics": [],
        "stdout_first5": "",
        "failure_category": None,
        "failure_details": None,
    }

    if data is None:
        result["failure_category"] = "NO_JSON_OUTPUT"
        result["failure_details"] = f"No JSON output found. Log excerpt: {log_text[:500]}"
        return result

    # Build
    build = data.get("build", {})
    result["build_status"] = build.get("status", "unknown").upper()
    result["build_time"] = build.get("duration_seconds", 0.0)
    result["build_stderr"] = build.get("stderr", "")

    # Run
    runs = data.get("runs", {})
    correctness = runs.get("correctness", {})
    result["run_status"] = correctness.get("status", "unknown").upper()
    result["run_time"] = correctness.get("duration_seconds", 0.0)
    result["run_exit_code"] = correctness.get("exit_code")
    result["run_stdout"] = correctness.get("stdout", "")
    result["run_stderr"] = correctness.get("stderr", "")

    # First 5 lines of stdout
    stdout_lines = result["run_stdout"].strip().split("\n")
    result["stdout_first5"] = "\n".join(stdout_lines[:5])

    # Verify
    ver = data.get("verification", {})
    result["verify_status"] = ver.get("status", "unknown").upper()
    result["verify_strategy"] = ver.get("strategy_used", "")
    result["verify_details"] = ver.get("details", "")

    # Metrics
    result["metrics"] = data.get("metrics", [])

    # Failure classification
    if result["build_status"] != "PASS":
        result["failure_category"] = "BUILD_FAIL"
        result["failure_details"] = _extract_build_error(result["build_stderr"])
    elif result["run_status"] == "TIMEOUT":
        result["failure_category"] = "RUN_TIMEOUT"
        result["failure_details"] = f"Timed out. stderr: {result['run_stderr'][:300]}"
    elif result["run_status"] != "PASS":
        result["failure_category"] = _classify_run_failure(result)
        result["failure_details"] = _extract_run_error(result)
    elif result["verify_status"] != "PASS":
        result["failure_category"] = "VERIFY_FAIL"
        result["failure_details"] = result["verify_details"]

    return result


def _extract_build_error(stderr: str) -> str:
    """Extract the most relevant build error lines."""
    lines = stderr.strip().split("\n")
    error_lines = [l for l in lines if "error" in l.lower() or "fatal" in l.lower()]
    if error_lines:
        return "\n".join(error_lines[:10])
    return "\n".join(lines[-10:]) if lines else "(empty stderr)"


def _classify_run_failure(result: dict) -> str:
    """Classify what kind of run failure occurred."""
    exit_code = result["run_exit_code"]
    stderr = result["run_stderr"]

    if exit_code == -11 or "segfault" in stderr.lower():
        return "RUN_SEGFAULT"
    elif exit_code == -6 or "abort" in stderr.lower():
        return "RUN_ABORT"
    elif "not found" in stderr.lower() or "No such file" in stderr.lower():
        return "RUN_FILE_NOT_FOUND"
    elif exit_code != 0:
        return f"RUN_FAIL(exit={exit_code})"
    return "RUN_FAIL"


def _extract_run_error(result: dict) -> str:
    """Extract relevant run error info."""
    parts = []
    if result["run_exit_code"] is not None:
        parts.append(f"exit_code={result['run_exit_code']}")
    if result["run_stderr"]:
        parts.append(f"stderr: {result['run_stderr'][:300]}")
    if result["run_stdout"]:
        parts.append(f"stdout(last 5 lines): {chr(10).join(result['run_stdout'].strip().split(chr(10))[-5:])}")
    return " | ".join(parts)


def generate_report(results: list[dict]) -> str:
    """Generate the full markdown analysis report."""
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Counts
    total = len(results)
    build_pass = sum(1 for r in results if r["build_status"] == "PASS")
    build_fail = sum(1 for r in results if r["build_status"] not in ("PASS", "MISSING"))
    run_pass = sum(1 for r in results if r["run_status"] == "PASS")
    run_fail = sum(1 for r in results if r["run_status"] == "FAIL")
    run_timeout = sum(1 for r in results if r["run_status"] == "TIMEOUT")
    run_error = sum(1 for r in results if r["run_status"] == "ERROR")
    verify_pass = sum(1 for r in results if r["verify_status"] == "PASS")
    verify_fail = sum(1 for r in results if r["verify_status"] not in ("PASS", "MISSING", "SKIP"))
    full_pass = sum(1 for r in results if r["build_status"] == "PASS" and r["run_status"] == "PASS" and r["verify_status"] == "PASS")
    missing = sum(1 for r in results if r["build_status"] == "MISSING")

    # Batch breakdown
    batch2_results = [r for r in results if r["kernel"] in BATCH2]
    batch3_results = [r for r in results if r["kernel"] in BATCH3]
    b2_pass = sum(1 for r in batch2_results if r["build_status"] == "PASS" and r["run_status"] == "PASS" and r["verify_status"] == "PASS")
    b3_pass = sum(1 for r in batch3_results if r["build_status"] == "PASS" and r["run_status"] == "PASS" and r["verify_status"] == "PASS")

    # Timing stats
    build_times = [r["build_time"] for r in results if r["build_status"] == "PASS"]
    run_times = [r["run_time"] for r in results if r["run_status"] == "PASS"]

    lines = []
    lines.append("# CUDA Batch 2+3 — Full Harness Analysis Report")
    lines.append("")
    lines.append(f"**Generated**: {now}")
    lines.append(f"**Platform**: Linux x86_64, NVIDIA GeForce RTX 4070 (sm_89, Ada Lovelace, 12 GB)")
    lines.append(f"**Compiler**: nvcc 12.3, gcc 12.4.0")
    lines.append(f"**Kernels tested**: {total} (Batch 2: {len(batch2_results)}, Batch 3: {len(batch3_results)})")
    lines.append("")

    # ── Summary table ──
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Batch 2 | Batch 3 | Total |")
    lines.append("|--------|---------|---------|-------|")

    def count_by_batch(results_list, pred):
        b2 = sum(1 for r in results_list if r["kernel"] in BATCH2 and pred(r))
        b3 = sum(1 for r in results_list if r["kernel"] in BATCH3 and pred(r))
        return b2, b3, b2 + b3

    for label, pred in [
        ("Kernels tested", lambda r: True),
        ("BUILD PASS", lambda r: r["build_status"] == "PASS"),
        ("BUILD FAIL", lambda r: r["build_status"] not in ("PASS", "MISSING")),
        ("RUN PASS", lambda r: r["run_status"] == "PASS"),
        ("RUN FAIL", lambda r: r["run_status"] == "FAIL"),
        ("RUN TIMEOUT", lambda r: r["run_status"] == "TIMEOUT"),
        ("RUN ERROR", lambda r: r["run_status"] == "ERROR"),
        ("VERIFY PASS", lambda r: r["verify_status"] == "PASS"),
        ("VERIFY FAIL", lambda r: r["verify_status"] not in ("PASS", "MISSING", "SKIP")),
        ("**Full PASS**", lambda r: r["build_status"] == "PASS" and r["run_status"] == "PASS" and r["verify_status"] == "PASS"),
    ]:
        b2, b3, tot = count_by_batch(results, pred)
        lines.append(f"| {label} | {b2} | {b3} | {tot} |")

    if missing:
        lines.append(f"| MISSING (no output) | — | — | {missing} |")
    lines.append("")

    # ── Timing statistics ──
    lines.append("## Timing Statistics")
    lines.append("")
    if build_times:
        lines.append(f"- **Build times**: min={min(build_times):.2f}s, max={max(build_times):.2f}s, avg={sum(build_times)/len(build_times):.2f}s, total={sum(build_times):.1f}s")
    if run_times:
        lines.append(f"- **Run times**: min={min(run_times):.3f}s, max={max(run_times):.3f}s, avg={sum(run_times)/len(run_times):.3f}s, total={sum(run_times):.1f}s")
    lines.append(f"- **Total wall time**: ~{sum(build_times) + sum(run_times):.0f}s (build+run, excludes clean)")
    lines.append("")

    # ── Detailed results table ──
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("Legend: ✅ PASS | ❌ FAIL | ⏱ TIMEOUT | 🚫 BUILD FAIL | ⚠ ERROR | ❓ MISSING")
    lines.append("")
    lines.append("| # | Kernel | Batch | Build | Build Time | Run | Run Time | Exit | Verify | Strategy | Stdout (first line) |")
    lines.append("|---|--------|-------|-------|------------|-----|----------|------|--------|----------|---------------------|")

    for i, r in enumerate(results, 1):
        batch = "B2" if r["kernel"] in BATCH2 else "B3"
        build_icon = {"PASS": "✅", "FAIL": "🚫", "TIMEOUT": "⏱", "ERROR": "⚠"}.get(r["build_status"], "❓")
        run_icon = {"PASS": "✅", "FAIL": "❌", "TIMEOUT": "⏱", "ERROR": "⚠"}.get(r["run_status"], "❓")
        verify_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⬜"}.get(r["verify_status"], "❓")

        first_line = r["stdout_first5"].split("\n")[0][:80] if r["stdout_first5"] else "(none)"
        first_line = first_line.replace("|", "\\|")

        lines.append(
            f"| {i} | {r['kernel']} | {batch} "
            f"| {build_icon} | {r['build_time']:.2f}s "
            f"| {run_icon} | {r['run_time']:.3f}s "
            f"| {r['run_exit_code'] if r['run_exit_code'] is not None else '—'} "
            f"| {verify_icon} | {r['verify_strategy']} "
            f"| {first_line} |"
        )

    lines.append("")

    # ── Passing kernels ──
    passing = [r for r in results if r["failure_category"] is None]
    if passing:
        lines.append(f"## Passing Kernels ({len(passing)}/{total})")
        lines.append("")
        for r in passing:
            metrics_str = ""
            if r["metrics"]:
                metrics_str = " — " + ", ".join(f"{m['name']}={m['value']}{m.get('unit','')}" for m in r["metrics"])
            lines.append(f"- **{r['kernel']}**: build={r['build_time']:.2f}s, run={r['run_time']:.3f}s, verify={r['verify_strategy']}{metrics_str}")
        lines.append("")

    # ── Failures breakdown ──
    failures = [r for r in results if r["failure_category"] is not None]
    if failures:
        lines.append(f"## Failures ({len(failures)}/{total})")
        lines.append("")

        # Group by failure category
        categories = {}
        for r in failures:
            cat = r["failure_category"]
            categories.setdefault(cat, []).append(r)

        for cat, items in sorted(categories.items()):
            lines.append(f"### {cat} ({len(items)} kernels)")
            lines.append("")
            lines.append("| Kernel | Batch | Details |")
            lines.append("|--------|-------|---------|")
            for r in items:
                batch = "B2" if r["kernel"] in BATCH2 else "B3"
                details = (r["failure_details"] or "").replace("\n", " ").replace("|", "\\|")[:200]
                lines.append(f"| {r['kernel']} | {batch} | {details} |")
            lines.append("")

        # Failure stdout/stderr details for debugging
        lines.append("### Detailed Failure Logs")
        lines.append("")
        for r in failures:
            lines.append(f"#### {r['kernel']} ({r['failure_category']})")
            lines.append("")
            lines.append(f"- **Build**: {r['build_status']} ({r['build_time']:.2f}s)")
            lines.append(f"- **Run**: {r['run_status']} ({r['run_time']:.3f}s, exit={r['run_exit_code']})")
            lines.append(f"- **Verify**: {r['verify_status']} ({r['verify_strategy']})")
            if r["run_stdout"]:
                lines.append(f"- **stdout** (last 10 lines):")
                lines.append("```")
                for line in r["run_stdout"].strip().split("\n")[-10:]:
                    lines.append(line)
                lines.append("```")
            if r["run_stderr"]:
                lines.append(f"- **stderr** (last 10 lines):")
                lines.append("```")
                for line in r["run_stderr"].strip().split("\n")[-10:]:
                    lines.append(line)
                lines.append("```")
            if r["build_stderr"] and r["build_status"] != "PASS":
                lines.append(f"- **build stderr** (last 10 lines):")
                lines.append("```")
                for line in r["build_stderr"].strip().split("\n")[-10:]:
                    lines.append(line)
                lines.append("```")
            lines.append("")

    # ── Verification strategy breakdown ──
    lines.append("## Verification Strategies Used")
    lines.append("")
    strat_counts = {}
    for r in results:
        s = r["verify_strategy"] or "(none)"
        strat_counts[s] = strat_counts.get(s, 0) + 1
    for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{s}**: {c} kernels")
    lines.append("")

    # ── Metrics summary ──
    lines.append("## Performance Metrics (from correctness runs)")
    lines.append("")
    lines.append("| Kernel | Metric | Value | Unit |")
    lines.append("|--------|--------|-------|------|")
    for r in results:
        for m in r["metrics"]:
            lines.append(f"| {r['kernel']} | {m['name']} | {m['value']} | {m.get('unit', '')} |")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"Analyzing results from: {LOG_DIR}")

    # Check if batch is done
    done_marker = LOG_DIR / "_done.marker"
    if done_marker.exists():
        print(f"  Batch completed: {done_marker.read_text().strip()}")
    else:
        print("  ⚠ Batch may still be running (_done.marker not found)")

    all_results = []
    for kernel in sorted(ALL_KERNELS):
        data = load_json_result(kernel)
        log_text = load_log_result(kernel)
        result = classify_result(data, log_text)
        result["kernel"] = kernel
        all_results.append(result)

    # Sort by batch then alphabetically
    all_results.sort(key=lambda r: (0 if r["kernel"] in BATCH2 else 1, r["kernel"]))

    # Generate markdown report
    report = generate_report(all_results)
    REPORT_PATH.write_text(report)
    print(f"  Markdown report: {REPORT_PATH}")

    # Save structured JSON for further processing
    json_data = {
        "generated": datetime.now().isoformat(),
        "total": len(all_results),
        "full_pass": sum(1 for r in all_results if r["failure_category"] is None),
        "results": all_results,
    }
    REPORT_JSON.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"  JSON data: {REPORT_JSON}")

    # Quick summary
    full_pass = sum(1 for r in all_results if r["failure_category"] is None)
    failures = [r for r in all_results if r["failure_category"] is not None]
    missing = [r for r in all_results if r["build_status"] == "MISSING"]
    print(f"\n  Results: {full_pass}/{len(all_results)} FULL PASS, {len(failures)} failures, {len(missing)} missing")

    if failures:
        print(f"\n  Failures:")
        for r in failures:
            print(f"    {r['kernel']}: {r['failure_category']} — {(r['failure_details'] or '')[:80]}")


if __name__ == "__main__":
    main()
