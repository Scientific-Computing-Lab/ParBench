#!/usr/bin/env python3
"""Generate Linux shared-input baseline witnesses for every candidate pair
(Task 7, E-02/E-12).

For every spec involved in the populated candidate registry, build and run
the PRISTINE baseline with its declared correctness input and judge it with
its own declared oracle (the independent comparator is the declared strategy
set, evidenced by this report). A directed pair's witness passes only when
BOTH sides' pristine baselines pass. Exclusions (KNOWN_FAIL) and
performance-only specs are recorded as explicit ineligibility reasons, never
silently dropped.

The per-spec section is a resumable cache: the report is rewritten after
every spec, and ``--resume`` skips specs already recorded with the same
oracle hash, so the campaign can run in bounded foreground chunks
(``--max-new``).
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

from harness.constants import (  # noqa: E402
    EXCLUDED_SPECS,
    PERFORMANCE_ONLY_SPECS,
)

SCHEMA = "pair-witnesses-v1"


def _load_specs(spec_root: Path) -> dict[str, dict[str, Any]]:
    return {
        (s := json.loads(p.read_text()))["identity"]["unique_id"]: s
        for p in sorted(spec_root.glob("*.json"))
    }


def witness_spec(
    spec: dict[str, Any], project_root: Path, *, timeout: int,
    run_timeout_cap: int = 480,
) -> dict[str, Any]:
    """Build + run + verify one pristine baseline; never edits any tree.

    The run timeout is capped at ``run_timeout_cap`` seconds (recorded when
    the cap undercuts the spec's declared timeout) so the headless campaign
    stays bounded; a baseline that cannot finish under the cap is recorded
    as TIMEOUT and lands in the pair disposition table, never silently
    skipped.
    """
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
        "run_timeout_capped": declared > run_timeout_cap,
        "correctness_args": [
            str(a) for a in (
                (spec.get("run", {}).get("input_configurations") or {})
                .get("correctness") or {}
            ).get("arguments", [])
        ],
    }
    build = build_spec(spec, project_root, timeout=timeout)
    entry["build_status"] = build.status.name
    if build.status is not Status.PASS:
        entry.update(status="BUILD_FAIL",
                     details=(build.stderr or build.stdout or "")[-400:])
    else:
        run = run_spec(spec, project_root, "correctness")
        resolved = resolve_paths(spec, project_root)["_resolved"]
        verdict = verify_run(spec, run, working_dir=resolved["working_dir"])
        entry.update(
            status=verdict.status.name,
            run_exit_code=run.exit_code,
            strategy_used=verdict.strategy_used,
            details=(verdict.details or "")[:300],
            stdout_sha256=hashlib.sha256(
                (run.stdout or "").encode("utf-8", "replace")).hexdigest(),
        )
    entry["duration_seconds"] = round(time.time() - t0, 2)
    return entry


def pair_rows(
    pairs: list[dict[str, Any]], spec_entries: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        src, tgt = pair["source_spec"], pair["target_spec"]
        reasons: list[str] = []
        for side, uid in (("source", src), ("target", tgt)):
            if uid in EXCLUDED_SPECS:
                reasons.append(f"{side}_excluded_known_fail")
            if uid in PERFORMANCE_ONLY_SPECS:
                reasons.append(f"{side}_performance_only")
            entry = spec_entries.get(uid)
            if entry is None:
                reasons.append(f"{side}_witness_not_run")
            elif entry["status"] != "PASS":
                reasons.append(f"{side}_baseline_{entry['status'].lower()}")
        rows.append({
            "source_spec": src,
            "target_spec": tgt,
            "witness_status": "pass" if not reasons else "fail",
            "reasons": reasons,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--candidate-contracts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip specs already witnessed under the same oracle "
                         "hash (default: on).")
    ap.add_argument("--max-new", type=int, default=None,
                    help="Witness at most N un-cached specs, then stop with "
                         "exit 3 if specs remain (bounded foreground chunk).")
    ap.add_argument("--timeout", type=int, default=480,
                    help="Build timeout per spec (seconds).")
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    registry = json.loads(args.candidate_contracts.read_text())
    pairs = registry["pairs"]
    specs = _load_specs(root / "specs")

    involved = sorted({
        uid for p in pairs for uid in (p["source_spec"], p["target_spec"])
    })
    unknown = [u for u in involved if u not in specs]
    if unknown:
        print(f"error: registry references unknown specs: {unknown}",
              file=sys.stderr)
        return 2

    report: dict[str, Any] = {}
    if args.output.exists():
        report = json.loads(args.output.read_text())
    spec_entries: dict[str, Any] = dict(report.get("specs") or {})

    def _write(complete: bool) -> None:
        payload = {
            "schema": SCHEMA,
            "generated_by":
                "scripts/spec_tools/generate_pair_witnesses.py",
            "host": platform.node(),
            "comparator": {
                "type": "declared-strategy",
                "description": (
                    "each side's pristine baseline is judged by its own "
                    "declared verification strategies (conjunctive, "
                    "fail-closed harness verifier) on the source-complete "
                    "Linux host"
                ),
            },
            "complete": complete,
            "specs": {u: spec_entries[u] for u in sorted(spec_entries)},
            "pairs": pair_rows(pairs, spec_entries),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    todo: list[str] = []
    for uid in involved:
        cached = spec_entries.get(uid)
        if (args.resume and cached
                and cached.get("oracle_sha256") == oracle_sha256(specs[uid])):
            continue
        todo.append(uid)

    budget = args.max_new if args.max_new is not None else len(todo)
    done_now = 0
    for uid in todo:
        if done_now >= budget:
            break
        entry = witness_spec(specs[uid], root, timeout=args.timeout)
        spec_entries[uid] = entry
        done_now += 1
        print(f"{uid:45s} {entry['status']:>11} "
              f"{entry['duration_seconds']:7.1f}s", flush=True)
        _write(complete=False)

    remaining = len(todo) - done_now
    complete = remaining == 0
    _write(complete=complete)
    rows = pair_rows(pairs, spec_entries)
    npass = sum(1 for r in rows if r["witness_status"] == "pass")
    print(f"\nspecs witnessed {len(spec_entries)}/{len(involved)} "
          f"(+{done_now} this run, {remaining} remaining); "
          f"pairs pass {npass}/{len(rows)} -> {args.output}")
    if not complete:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
