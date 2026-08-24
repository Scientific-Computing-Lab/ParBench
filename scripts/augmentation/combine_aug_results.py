#!/usr/bin/env python3
"""scripts/combine_aug_results.py — Merge per-stream JSON results into aggregate reports.

After all 3 stream scripts (cuda/omp/opencl) complete, run this to produce:
  results/augmentation/phase3_smoke_results.md   — 3 specs × L1,L2,L4
  results/augmentation/rodinia_aug_results.md     — 51 Rodinia specs × L2
  results/augmentation/full_aug_results.md        — all specs × L1,L2,L4

Usage:
    python3 scripts/combine_aug_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
AUGDIR = PROJECT_ROOT / "results" / "augmentation"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "augmentation"))

from run_augment_batch import generate_markdown  # noqa: E402


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping", flush=True)
        return []
    return json.loads(path.read_text())


def main() -> int:
    print("=== Combining augmentation results ===", flush=True)

    # ---- Phase 3: smoke test (3 specs × L1,L2,L4) ----
    p3_results = (
        load_json(AUGDIR / "phase3_cuda.json")
        + load_json(AUGDIR / "phase3_omp.json")
        + load_json(AUGDIR / "phase3_opencl.json")
    )
    if p3_results:
        md = generate_markdown(p3_results, levels=[1, 2, 4],
                               title="Phase 3: Smoke Test (3 Specs × L1,L2,L4)")
        out = AUGDIR / "phase3_smoke_results.md"
        out.write_text(md)
        passes = sum(1 for r in p3_results if r["overall_status"] == "PASS")
        print(f"Phase 3: {passes}/{len(p3_results)} PASS  →  {out}", flush=True)
    else:
        print("Phase 3: no results found", flush=True)

    # ---- Phase 4: Rodinia batch (all Rodinia specs × L2) ----
    p4_results = (
        load_json(AUGDIR / "phase4_cuda.json")
        + load_json(AUGDIR / "phase4_omp.json")
        + load_json(AUGDIR / "phase4_opencl.json")
    )
    if p4_results:
        md = generate_markdown(p4_results, levels=[2],
                               title="Phase 4: Rodinia Batch — All Specs at L2")
        out = AUGDIR / "rodinia_aug_results.md"
        out.write_text(md)
        # Also write combined JSON
        (AUGDIR / "rodinia_aug_results.json").write_text(json.dumps(p4_results, indent=2))
        passes = sum(1 for r in p4_results if r["overall_status"] == "PASS")
        print(f"Phase 4: {passes}/{len(p4_results)} PASS  →  {out}", flush=True)
    else:
        print("Phase 4: no results found", flush=True)

    # ---- Phase 5: full batch (all specs × L1,L2,L4) ----
    p5_results = (
        load_json(AUGDIR / "phase5_cuda.json")
        + load_json(AUGDIR / "phase5_omp.json")
        + load_json(AUGDIR / "phase5_opencl.json")
    )
    if p5_results:
        md = generate_markdown(p5_results, levels=[1, 2, 4],
                               title="Phase 5: Full Batch — All Specs at L1,L2,L4")
        out = AUGDIR / "full_aug_results.md"
        out.write_text(md)
        (AUGDIR / "full_aug_results.json").write_text(json.dumps(p5_results, indent=2))
        passes = sum(1 for r in p5_results if r["overall_status"] == "PASS")
        print(f"Phase 5: {passes}/{len(p5_results)} PASS  →  {out}", flush=True)
    else:
        print("Phase 5: no results found", flush=True)

    print("=== Done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
