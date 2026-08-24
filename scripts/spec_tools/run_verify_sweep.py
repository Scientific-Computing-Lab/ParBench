#!/usr/bin/env python3
"""Sweep `harness verify --config correctness` on all curated non-KNOWN_FAIL specs (T2.3).

Filter sources:
- CURATED set: config/sloc_corpus_kernels.json (the frozen 35-kernel registry;
  scripts/analysis/sloc_analysis.CORPUS_KERNELS loads it hash-verified).
- KNOWN_FAIL set: .claude/rules/known-issues.md (8 entries, updated 2026-04-09).

Exit 0 only when every non-KNOWN_FAIL curated spec PASSes. Emits FAIL details to stderr.
Run serially by default (concurrent nvcc can OOM).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# The 35-kernel curated corpus is LOADED from the hash-verified frozen registry
# (config/sloc_corpus_kernels.json), never re-copied as a literal here (gate22
# finding 3). sloc_analysis.load_frozen_corpus_kernels re-hashes the registry and
# fails closed on any drift, so this consumer shares that single source of truth;
# a stale literal copy could otherwise diverge silently.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "analysis"))
from sloc_analysis import CORPUS_KERNELS  # noqa: E402  (hash-verified at import)

CURATED: set[tuple[str, str]] = set(CORPUS_KERNELS)

# Canonical KNOWN_FAIL list, imported live so it can never drift. A stale 8-entry copy
# lived here until 2026-07-24: it predated fd554569, which added rodinia-backprop-opencl.
from harness.constants import EXCLUDED_SPECS as KNOWN_FAIL  # noqa: E402

EXPECTED_COUNT = 87


def select_specs(project_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for p in sorted((project_root / "specs").glob("*.json")):
        s = json.loads(p.read_text())
        ident = s["identity"]
        key = (ident["source_suite"], ident["kernel_name"])
        uid = ident["unique_id"]
        if key not in CURATED or uid in KNOWN_FAIL:
            continue
        out.append((uid, p))
    return out


def verify_one(project_root: Path, path: Path, timeout: int = 900):
    t0 = time.time()
    r = subprocess.run(
        ["python3", "-m", "harness", "verify", "--config", "correctness", str(path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument(
        "--exclude-known-fail",
        action="store_true",
        default=True,
        help="No-op flag (always True); kept for clarity.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallelism; default 1 to avoid concurrent nvcc OOM.",
    )
    args = ap.parse_args()

    specs = select_specs(args.project_root)
    if len(specs) != EXPECTED_COUNT:
        print(
            f"ERROR: expected {EXPECTED_COUNT} curated non-KNOWN_FAIL specs; got {len(specs)}",
            file=sys.stderr,
        )
        return 2

    print(f"{'spec':45s} {'rc':>3} {'time':>7}  verdict")
    print("-" * 90)
    fails: list[tuple[str, int, str, str]] = []
    for uid, path in specs:
        rc, out, err, dt = verify_one(args.project_root, path)
        last = (out.strip().splitlines() or [""])[-1]
        print(f"{uid:45s} {rc:>3} {dt:7.1f}  {last[:64]}", flush=True)
        if rc != 0:
            fails.append((uid, rc, out, err))

    ok = len(specs) - len(fails)
    print(f"\nPASS {ok}/{len(specs)}  FAIL {len(fails)}")
    if fails:
        print("\n=== FAIL DETAIL ===", file=sys.stderr)
        for uid, rc, out, err in fails:
            print(f"\n--- {uid} (rc={rc}) ---\n{(err or out)[-1500:]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
