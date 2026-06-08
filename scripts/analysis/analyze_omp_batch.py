#!/usr/bin/env python3
"""Analyze OMP batch harness results from JSON logs."""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(".")
LOG_DIR = PROJECT_ROOT / "results" / "phase3" / "omp_batch_logs"
REPORT_PATH = PROJECT_ROOT / "results" / "phase3" / "omp_batch_analysis.md"
REPORT_JSON = PROJECT_ROOT / "results" / "phase3" / "omp_batch_results.json"

BATCH2 = [
    "babelstream", "backprop", "ccsd-trpdrv", "deredundancy", "feynman-kac",
    "fpc", "ga", "keccaktreehash", "laplace3d", "lulesh",
    "maxpool3d", "md5hash", "pathfinder", "pso", "rmsnorm",
    "secp256k1", "softmax-online", "thomas", "tissue", "tsp",
]
BATCH3_RUNNABLE = [
    "bezier-surface", "crc64", "floydwarshall", "gaussian",
    "geglu", "heat2d", "iso2dfd", "jenkins-hash", "knn",
    "mandelbrot", "mis", "murmurhash3", "myocyte", "nw",
    "perplexity", "popcount", "sobol", "stencil1d", "triad",
]
BATCH3_SKIP = ["convolution3d"]  # No OMP dir

ALL_KERNELS = BATCH2 + BATCH3_RUNNABLE


def load_json_result(kernel):
    path = LOG_DIR / f"{kernel}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_log_result(kernel):
    path = LOG_DIR / f"{kernel}.log"
    if not path.exists():
        return ""
    return path.read_text()


def classify_result(data, log_text):
    result = {
        "kernel": "",
        "build_status": "MISSING", "build_time": 0.0, "build_stderr": "",
        "run_status": "MISSING", "run_time": 0.0, "run_exit_code": None,
        "run_stdout": "", "run_stderr": "",
        "verify_status": "MISSING", "verify_strategy": "", "verify_details": "",
        "metrics": [], "stdout_first5": "",
        "failure_category": None, "failure_details": None,
    }

    if data is None:
        result["failure_category"] = "NO_JSON_OUTPUT"
        result["failure_details"] = f"No JSON output. Log: {log_text[:500]}"
        return result

    build = data.get("build", {})
    result["build_status"] = build.get("status", "unknown").upper()
    result["build_time"] = build.get("duration_seconds", 0.0)
    result["build_stderr"] = build.get("stderr", "")

    runs = data.get("runs", {})
    corr = runs.get("correctness", {})
    result["run_status"] = corr.get("status", "unknown").upper()
    result["run_time"] = corr.get("duration_seconds", 0.0)
    result["run_exit_code"] = corr.get("exit_code")
    result["run_stdout"] = corr.get("stdout", "")
    result["run_stderr"] = corr.get("stderr", "")

    stdout_lines = result["run_stdout"].strip().split("\n")
    result["stdout_first5"] = "\n".join(stdout_lines[:5])

    ver = data.get("verification", {})
    result["verify_status"] = ver.get("status", "unknown").upper()
    result["verify_strategy"] = ver.get("strategy_used", "")
    result["verify_details"] = ver.get("details", "")

    result["metrics"] = data.get("metrics", [])

    if result["build_status"] != "PASS":
        result["failure_category"] = "BUILD_FAIL"
        result["failure_details"] = result["build_stderr"][:300]
    elif result["run_status"] == "TIMEOUT":
        result["failure_category"] = "RUN_TIMEOUT"
        result["failure_details"] = f"Timed out. stderr: {result['run_stderr'][:200]}"
    elif result["run_status"] != "PASS":
        ec = result["run_exit_code"]
        if ec == -11:
            result["failure_category"] = "RUN_SEGFAULT"
        elif ec == -6:
            result["failure_category"] = "RUN_ABORT"
        else:
            result["failure_category"] = f"RUN_FAIL(exit={ec})"
        result["failure_details"] = f"exit={ec}, stderr: {result['run_stderr'][:200]}"
    elif result["verify_status"] != "PASS":
        result["failure_category"] = "VERIFY_FAIL"
        result["failure_details"] = result["verify_details"][:200]

    return result


def main():
    print(f"Analyzing OMP results from: {LOG_DIR}")

    done_marker = LOG_DIR / "_done.marker"
    if done_marker.exists():
        print(f"  Batch completed: {done_marker.read_text().strip()}")

    all_results = []
    for kernel in ALL_KERNELS:
        data = load_json_result(kernel)
        log_text = load_log_result(kernel)
        result = classify_result(data, log_text)
        result["kernel"] = kernel
        all_results.append(result)

    all_results.sort(key=lambda r: (0 if r["kernel"] in BATCH2 else 1, r["kernel"]))

    # Counts
    total = len(all_results)
    full_pass = sum(1 for r in all_results if r["failure_category"] is None)
    failures = [r for r in all_results if r["failure_category"] is not None]

    # Save JSON
    json_data = {
        "generated": datetime.now().isoformat(),
        "total": total,
        "skip": len(BATCH3_SKIP),
        "full_pass": full_pass,
        "results": all_results,
    }
    REPORT_JSON.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"  JSON data: {REPORT_JSON}")

    # Print summary
    print(f"\n  Results: {full_pass}/{total} FULL PASS (+{len(BATCH3_SKIP)} SKIP), {len(failures)} failures\n")

    if failures:
        print("  Failures:")
        for r in failures:
            print(f"    {r['kernel']}: {r['failure_category']}")
            # Show stdout for verify fails
            if "VERIFY" in str(r["failure_category"]):
                stdout_last = "\n".join(r["run_stdout"].strip().split("\n")[-5:])
                print(f"      details: {r['failure_details']}")
                print(f"      stdout(last 5): {stdout_last[:200]}")
            elif "RUN" in str(r["failure_category"]):
                print(f"      details: {r['failure_details']}")

    # Passing
    print(f"\n  Passing ({full_pass}):")
    for r in all_results:
        if r["failure_category"] is None:
            print(f"    {r['kernel']}: build={r['build_time']:.2f}s, run={r['run_time']:.3f}s, verify={r['verify_strategy']}")


if __name__ == "__main__":
    main()
