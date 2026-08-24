#!/usr/bin/env python3
"""Task 7 thread-provisioning sweep across offload baselines (E-05/E-12).

Re-runs the hecbench-lud-omp-class sweep on the source-complete host as a
freeze input: every curated, non-excluded omp / omp_target baseline whose
generated thread_provisioning check reported a finding is run pristine at a
provisioned OMP_NUM_THREADS (its largest declared requirement) and
deliberately under-provisioned at 1, judged by its own declared oracle both
times. The sweep logic is Task 4's (`build_source_baseline_dispositions`);
this wrapper only rebinds the output to the Task 7 artifact path so the
freeze manifest can validate and hash it independently.

A robust_at_both_levels outcome is a screen, not a prover - it shows the
declared oracle cannot distinguish the two provisioning levels on this host.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "thread-provisioning-v1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    root = args.project_root.resolve()

    import copy as _copy

    import build_source_baseline_dispositions as t4

    from harness.builder import build_spec
    from harness.models import Status

    t0 = time.time()
    targets = t4.sweep_target_specs(root)
    results: dict[str, dict] = {}
    for uid, meta in targets.items():
        spec = t4._load_spec(root, uid)
        pinned_env = (spec["run"].get("environment_variables") or {}).get(
            "OMP_NUM_THREADS")
        candidates = list(meta["literal_extents"])
        if pinned_env and str(pinned_env).isdigit():
            candidates.append(int(pinned_env))
        provisioned = max(candidates or [256])
        build = build_spec(spec, root)
        if build.status is not Status.PASS:
            results[uid] = {
                "outcome": "build_failed",
                "literal_extents": meta["literal_extents"],
                "details": build.stderr[-300:],
            }
            print(f"{uid}: BUILD_FAIL", flush=True)
            continue
        runs = {
            "provisioned": t4._run_with_omp_threads(
                _copy.deepcopy(spec), root, provisioned),
            "under_provisioned": t4._run_with_omp_threads(
                _copy.deepcopy(spec), root, 1),
        }
        prov_ok = runs["provisioned"]["oracle_status"] == "PASS"
        under_ok = runs["under_provisioned"]["oracle_status"] == "PASS"
        if prov_ok and under_ok:
            outcome = "robust_at_both_levels"
        elif prov_ok and not under_ok:
            outcome = "under_provisioning_detected_by_oracle"
        else:
            outcome = "provisioned_baseline_red"
        results[uid] = {
            "outcome": outcome,
            "literal_extents": meta["literal_extents"],
            "runs": runs,
        }
        print(f"{uid}: {outcome}", flush=True)

    payload = {
        "schema": SCHEMA,
        "generated_by": "scripts/spec_tools/run_thread_provisioning_sweep.py",
        "generated_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "description": (
            "Task 7 freeze-input sweep of the hecbench-lud-omp "
            "under-provisioning class; robust_at_both_levels is a screen, "
            "not a prover"
        ),
        "complete": True,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "specs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(results)} specs, "
          f"{payload['wall_clock_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
