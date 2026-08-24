#!/usr/bin/env python3
"""Baseline determinism repetitions for witnessed pairs (Task 7, E-12).

Every spec that participates in at least one witness-passing candidate pair
is built once and then run + verified ``--repetitions`` times with its
declared correctness input. A spec's baseline is ``stable`` when every
repetition's oracle verdict is PASS. Raw stdout hashes are recorded as data
(timing text makes byte-stability the wrong bar; the contract-level claim is
verdict stability under the declared oracle).

Resumable like the witness generator: the report rewrites after every spec
and ``--resume`` skips specs already measured under the same oracle hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_oracle_contracts import oracle_sha256  # noqa: E402

SCHEMA = "baseline-determinism-v1"


def measure_spec(
    spec: dict[str, Any], project_root: Path, *,
    repetitions: int, build_timeout: int, run_timeout_cap: int = 480,
) -> dict[str, Any]:
    import copy

    from harness.builder import build_spec
    from harness.models import Status
    from harness.runner import run_spec
    from harness.spec_loader import resolve_paths
    from harness.verifier import verify_run

    declared = spec.get("run", {}).get("timeout_seconds", 300)
    if declared > run_timeout_cap:
        spec = copy.deepcopy(spec)
        spec["run"]["timeout_seconds"] = run_timeout_cap

    t0 = time.time()
    entry: dict[str, Any] = {
        "oracle_sha256": oracle_sha256(spec),
        "repetitions": repetitions,
        "runs": [],
    }
    build = build_spec(spec, project_root, timeout=build_timeout)
    entry["build_status"] = build.status.name
    if build.status is not Status.PASS:
        entry["stable"] = False
        entry["details"] = (build.stderr or build.stdout or "")[-400:]
    else:
        resolved = resolve_paths(spec, project_root)["_resolved"]
        for _ in range(repetitions):
            run = run_spec(spec, project_root, "correctness")
            verdict = verify_run(
                spec, run, working_dir=resolved["working_dir"])
            entry["runs"].append({
                "status": verdict.status.name,
                "exit_code": run.exit_code,
                "strategy_used": verdict.strategy_used,
                "stdout_sha256": hashlib.sha256(
                    (run.stdout or "").encode("utf-8", "replace")
                ).hexdigest(),
            })
        entry["stable"] = bool(entry["runs"]) and all(
            r["status"] == "PASS" for r in entry["runs"])
    entry["duration_seconds"] = round(time.time() - t0, 2)
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--pair-witnesses", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=None,
                    help="Measure at most N un-cached specs, exit 3 if any "
                         "remain (bounded foreground chunk).")
    ap.add_argument("--timeout", type=int, default=480,
                    help="Build timeout per spec (seconds).")
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    witnesses = json.loads(args.pair_witnesses.read_text())
    eligible_specs = sorted({
        uid
        for row in witnesses["pairs"]
        if row["witness_status"] == "pass"
        for uid in (row["source_spec"], row["target_spec"])
    })
    spec_docs = {
        uid: json.loads((root / "specs" / f"{uid}.json").read_text())
        for uid in eligible_specs
    }

    report: dict[str, Any] = {}
    if args.output.exists():
        report = json.loads(args.output.read_text())
    entries: dict[str, Any] = dict(report.get("specs") or {})
    # Drop cached entries that no longer correspond to a witnessed spec or
    # whose oracle changed since measurement.
    entries = {
        u: e for u, e in entries.items()
        if u in spec_docs and e.get("oracle_sha256") == oracle_sha256(spec_docs[u])
    }

    def _write(complete: bool) -> None:
        stable = sum(1 for e in entries.values() if e.get("stable"))
        payload = {
            "schema": SCHEMA,
            "generated_by":
                "scripts/spec_tools/run_baseline_determinism.py",
            "host": platform.node(),
            "pair_witnesses": str(args.pair_witnesses),
            "repetitions": args.repetitions,
            "complete": complete,
            "summary": {
                "specs_total": len(eligible_specs),
                "specs_measured": len(entries),
                "specs_stable": stable,
            },
            "specs": {u: entries[u] for u in sorted(entries)},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    todo = [u for u in eligible_specs if u not in entries]
    budget = args.max_new if args.max_new is not None else len(todo)
    done_now = 0
    for uid in todo:
        if done_now >= budget:
            break
        entry = measure_spec(
            spec_docs[uid], root,
            repetitions=args.repetitions, build_timeout=args.timeout)
        entries[uid] = entry
        done_now += 1
        print(f"{uid:45s} stable={entry['stable']} "
              f"{entry['duration_seconds']:7.1f}s", flush=True)
        _write(complete=False)

    remaining = len(todo) - done_now
    complete = remaining == 0
    _write(complete=complete)
    stable = sum(1 for e in entries.values() if e.get("stable"))
    print(f"\nmeasured {len(entries)}/{len(eligible_specs)} "
          f"(+{done_now} this run, {remaining} remaining); "
          f"stable {stable} -> {args.output}")
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())
