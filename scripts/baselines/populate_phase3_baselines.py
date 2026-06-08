#!/usr/bin/env python3
"""populate_phase3_baselines.py — Populate baseline_results for all passing specs
from Phase 3 harness results.

Reads JSON logs from:
  - results/phase3/cuda_batch3_logs/{kernel}.json  (Batch 2+3 CUDA)
  - results/phase3/omp_batch_logs/{kernel}.json    (Batch 2+3 OMP)

Only populates specs whose verification status == "pass".
Skips specs that already have baseline_results.
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path("/home/samyak/Desktop/parbench_sam")
SPECS_DIR = PROJECT_ROOT / "specs"
CUDA_LOGS = PROJECT_ROOT / "results" / "phase3" / "cuda_batch3_logs"
OMP_LOGS = PROJECT_ROOT / "results" / "phase3" / "omp_batch_logs"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
REF_PLATFORM = "rtx4070-r9-7900x"

def extract_kernel_name(spec_filename, api):
    """Extract kernel name from spec filename like hecbench-aes-cuda.json"""
    name = spec_filename.replace(".json", "").replace(f"hecbench-", "").replace(f"-{api}", "")
    return name

def populate_from_harness_json(spec_path, harness_json_path, api):
    """Read harness JSON output and populate baseline_results in the spec."""
    with open(harness_json_path) as f:
        result = json.load(f)

    # Check verification passed
    verification = result.get("verification", {})
    if verification.get("status") != "pass":
        return False, f"verification={verification.get('status')}"

    # Get the correctness run
    runs = result.get("runs", {})
    correctness = runs.get("correctness", {})
    if correctness.get("status") != "pass":
        return False, f"run={correctness.get('status')}"

    stdout = correctness.get("stdout", "")
    exit_code = correctness.get("exit_code", 0)
    wall_time = correctness.get("duration_seconds")
    stdout_snippet = stdout[:500] if stdout else ""

    # Extract performance metrics if available
    with open(spec_path) as f:
        spec = json.load(f)

    metrics = {}
    perf = spec.get("performance")
    if perf and perf.get("metrics"):
        for metric_def in perf["metrics"]:
            name = metric_def["name"]
            pattern = metric_def.get("extraction", {}).get("pattern", "")
            capture = metric_def.get("extraction", {}).get("capture_group", 1)
            unit = metric_def.get("unit", "")
            if pattern and stdout:
                m = re.search(pattern, stdout)
                if m:
                    try:
                        metrics[f"{name}_{unit}"] = float(m.group(capture))
                    except (ValueError, IndexError):
                        pass

    # Build baseline_results
    baseline = {
        "reference_platform": REF_PLATFORM,
        "timestamp": TIMESTAMP,
        "configurations": {
            "correctness": {
                "status": "pass",
                "exit_code": exit_code,
                "stdout_snippet": stdout_snippet,
                "wall_time_seconds": wall_time
            }
        }
    }

    if metrics:
        baseline["configurations"]["performance"] = {
            "status": "pass",
            "wall_time_seconds": wall_time,
            "metrics": metrics
        }

    spec["baseline_results"] = baseline

    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)
        f.write('\n')

    return True, f"{len(metrics)} metrics"


def main():
    cuda_ok = 0; cuda_skip = 0; cuda_fail = 0; cuda_nodata = 0
    omp_ok = 0; omp_skip = 0; omp_fail = 0; omp_nodata = 0

    # Process CUDA specs
    print("=== CUDA Specs ===")
    for spec_path in sorted(SPECS_DIR.glob("hecbench-*-cuda.json")):
        spec = json.load(open(spec_path))
        kernel = extract_kernel_name(spec_path.name, "cuda")

        if spec.get("baseline_results"):
            cuda_skip += 1
            continue

        log_path = CUDA_LOGS / f"{kernel}.json"
        if not log_path.exists():
            print(f"  NODATA {spec_path.name}: no harness JSON")
            cuda_nodata += 1
            continue

        ok, info = populate_from_harness_json(spec_path, log_path, "cuda")
        if ok:
            print(f"  OK {spec_path.name}: baseline populated ({info})")
            cuda_ok += 1
        else:
            print(f"  FAIL {spec_path.name}: {info}")
            cuda_fail += 1

    # Process OMP specs
    print("\n=== OMP Specs ===")
    for spec_path in sorted(SPECS_DIR.glob("hecbench-*-omp.json")):
        spec = json.load(open(spec_path))
        kernel = extract_kernel_name(spec_path.name, "omp")

        if spec.get("baseline_results"):
            omp_skip += 1
            continue

        log_path = OMP_LOGS / f"{kernel}.json"
        if not log_path.exists():
            print(f"  NODATA {spec_path.name}: no harness JSON")
            omp_nodata += 1
            continue

        ok, info = populate_from_harness_json(spec_path, log_path, "omp")
        if ok:
            print(f"  OK {spec_path.name}: baseline populated ({info})")
            omp_ok += 1
        else:
            print(f"  SKIP {spec_path.name}: {info} (expected for failing kernels)")
            omp_fail += 1

    print(f"\n=== Summary ===")
    print(f"CUDA: {cuda_ok} populated, {cuda_skip} already had baselines, {cuda_fail} failed verification, {cuda_nodata} no data")
    print(f"OMP:  {omp_ok} populated, {omp_skip} already had baselines, {omp_fail} failed/not-pass, {omp_nodata} no data")
    print(f"Total baselines now: CUDA {cuda_ok+cuda_skip}/60, OMP {omp_ok+omp_skip}/60")


if __name__ == "__main__":
    main()
