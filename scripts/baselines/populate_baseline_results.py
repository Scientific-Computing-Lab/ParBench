#!/usr/bin/env python3
"""
populate_baseline_results.py

Reads per-kernel JSON logs from results/rodinia/logs/ and writes
baseline_results into each matching spec file under specs/.

Schema for baseline_results (per spec_schema.json):
  {
    "reference_platform": str,
    "timestamp": ISO-8601 str,
    "configurations": {
      "<config_name>": {
        "status": "pass"|"fail"|"error"|"timeout"|"skip",
        "exit_code": int|null,
        "stdout_snippet": str|null,   # first 1000 chars of stdout
        "wall_time_seconds": float|null,
        "metrics": {},
        "fail_reason": str,           # only when status != "pass"
        "fail_category": str          # only when status == "fail"
      }
    }
  }
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR  = REPO_ROOT / "results" / "rodinia" / "logs"
SPECS_DIR = REPO_ROOT / "specs"

REFERENCE_PLATFORM = "rtx4070-linux-x86_64"

STDOUT_SNIPPET_LIMIT = 1000  # chars


def load_log(log_path: Path) -> dict:
    with open(log_path) as f:
        return json.load(f)


def build_baseline(log: dict) -> dict:
    """Convert a harness JSON log into the baseline_results structure."""
    now   = datetime.now(timezone.utc).isoformat()
    build = log.get("build", {})
    runs  = log.get("runs", {})

    configurations = {}

    # Determine which configs were attempted
    config_names = list(runs.keys()) if runs else ["correctness"]

    for config in config_names:
        run_data = runs.get(config, {})

        if build.get("status") != "pass":
            stderr_snip = (build.get("stderr") or "")[:STDOUT_SNIPPET_LIMIT]
            configurations[config] = {
                "status":            "fail",
                "exit_code":         None,
                "stdout_snippet":    None,
                "wall_time_seconds": None,
                "metrics":           {},
                "fail_reason":       f"build failed: {stderr_snip}",
                "fail_category":     "build_failure",
            }
        elif not run_data:
            configurations[config] = {
                "status":      "error",
                "metrics":     {},
                "fail_reason": "no run data in log",
            }
        else:
            run_status = run_data.get("status", "error")
            exit_code  = run_data.get("exit_code")
            stdout_raw = run_data.get("stdout") or ""
            stderr_raw = run_data.get("stderr") or ""
            wall_time  = run_data.get("duration_seconds")

            if run_status == "pass":
                configurations[config] = {
                    "status":            "pass",
                    "exit_code":         exit_code,
                    "stdout_snippet":    stdout_raw[:STDOUT_SNIPPET_LIMIT],
                    "wall_time_seconds": wall_time,
                    "metrics":           _extract_metrics(log.get("metrics", [])),
                }
            else:
                reason = (stderr_raw or stdout_raw)[:STDOUT_SNIPPET_LIMIT]
                configurations[config] = {
                    "status":            "fail",
                    "exit_code":         exit_code,
                    "stdout_snippet":    stdout_raw[:STDOUT_SNIPPET_LIMIT],
                    "wall_time_seconds": wall_time,
                    "metrics":           {},
                    "fail_reason":       reason or f"exit_code={exit_code}",
                }

    return {
        "reference_platform": REFERENCE_PLATFORM,
        "timestamp":          now,
        "configurations":     configurations,
    }


def _extract_metrics(metrics_list: list) -> dict:
    out = {}
    for m in metrics_list:
        if isinstance(m, dict) and "name" in m and "value" in m:
            out[m["name"]] = m["value"]
    return out


def main():
    log_files = sorted(LOGS_DIR.glob("rodinia-*.json"))
    if not log_files:
        print(f"ERROR: no log files found in {LOGS_DIR}", file=sys.stderr)
        sys.exit(1)

    updated      = 0
    missing_spec = 0

    for log_path in log_files:
        spec_id   = log_path.stem          # e.g. rodinia-bfs-cuda
        spec_path = SPECS_DIR / f"{spec_id}.json"

        if not spec_path.exists():
            print(f"  WARN  {spec_id}: spec file not found, skipping")
            missing_spec += 1
            continue

        log = load_log(log_path)

        with open(spec_path) as f:
            spec = json.load(f)

        spec["baseline_results"] = build_baseline(log)

        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=4)
            f.write("\n")

        configs = spec["baseline_results"]["configurations"]
        overall = "PASS" if all(c.get("status") == "pass" for c in configs.values()) else "FAIL"
        print(f"  {overall:4s}  {spec_id}")
        updated += 1

    print(f"\nDone. {updated} specs updated, {missing_spec} spec files missing.")


if __name__ == "__main__":
    main()
