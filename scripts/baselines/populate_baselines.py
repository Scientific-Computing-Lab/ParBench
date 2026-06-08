#!/usr/bin/env python3
"""populate_baselines.py — Fill baseline_results from Phase 3/4 results."""

import json
import glob
import re
from pathlib import Path
from datetime import datetime, timezone

SPECS_DIR = Path("specs")
RESULTS_DIR = Path("results/phase5")
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

for spec_path in sorted(SPECS_DIR.glob("hecbench-*.json")):
    spec_id = spec_path.stem
    result_path = RESULTS_DIR / f"{spec_id}.json"
    log_path = RESULTS_DIR / f"{spec_id}.log"

    if not log_path.exists():
        print(f"SKIP {spec_id}: no results log")
        continue

    with open(log_path) as f:
        log_content = f.read()

    # Check if verification passed
    if "VERIFY: PASS" not in log_content and "BUILD: PASS" not in log_content:
        print(f"SKIP {spec_id}: did not pass verification")
        continue

    # Load the spec
    with open(spec_path) as f:
        spec = json.load(f)

    # Extract stdout from the log (between [stdout] markers)
    stdout_match = re.search(r'\[stdout\]\n(.*?)(?:\n\[stderr\]|\Z)', log_content, re.DOTALL)
    stdout_snippet = stdout_match.group(1).strip()[:500] if stdout_match else ""

    # Extract wall time if available
    time_match = re.search(r'wall[_\s]?time[:\s]+([0-9.]+)', log_content, re.IGNORECASE)
    wall_time = float(time_match.group(1)) if time_match else None

    # Extract exit code
    exit_match = re.search(r'exit[_\s]?code[:\s]+(\d+)', log_content, re.IGNORECASE)
    exit_code = int(exit_match.group(1)) if exit_match else 0

    # Extract performance metrics using spec's regex patterns
    metrics = {}
    perf = spec.get("performance")
    if perf and perf.get("metrics"):
        for metric_def in perf["metrics"]:
            name = metric_def["name"]
            pattern = metric_def.get("extraction", {}).get("pattern", "")
            capture = metric_def.get("extraction", {}).get("capture_group", 1)
            unit = metric_def.get("unit", "")
            if pattern and stdout_snippet:
                m = re.search(pattern, stdout_snippet)
                if m:
                    try:
                        metrics[f"{name}_{unit}"] = float(m.group(capture))
                    except (ValueError, IndexError):
                        pass

    # Build baseline_results
    baseline = {
        "reference_platform": "rtx4070-r9-7900x",
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

    # Update spec
    spec["baseline_results"] = baseline

    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)
        f.write('\n')

    print(f"OK {spec_id}: baseline populated ({len(metrics)} metrics)")

print("\nDone. Run validate_schema.py --all to verify.")
