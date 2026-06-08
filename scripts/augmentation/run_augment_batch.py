#!/usr/bin/env python3
"""scripts/run_augment_batch.py — Batch augmentation verification runner.

Runs augment_verify for multiple specs × multiple levels, writes incremental
JSON results, and generates a Markdown report at the end.

Usage:
    python3 scripts/run_augment_batch.py specs/rodinia-*-cuda.json \
        --levels 1 2 4 --seed 42 \
        --out results/augmentation/phase4_cuda \
        --title "Phase 4: Rodinia CUDA"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "augmentation"))

from augment_verify import augment_verify  # noqa: E402

_SYM = {"PASS": "✓", "BUILD_FAIL": "✗", "FAIL": "✗", "ERROR": "!", "SKIP": "-"}


def run_batch(
    spec_paths: list[Path],
    levels: list[int],
    seed: int,
    config: str,
    out_prefix: str,
    verbose: bool = False,
) -> list[dict]:
    """Run (spec, level) combinations sequentially; write JSON incrementally."""
    results: list[dict] = []
    total = len(spec_paths) * len(levels)
    done = 0
    out_json = Path(out_prefix + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    for spec_path in sorted(spec_paths):
        for level in sorted(levels):
            done += 1
            spec_id = spec_path.stem
            print(f"[{done:4d}/{total}] {spec_id}  L{level}", flush=True)
            t0 = time.time()
            result = augment_verify(
                spec_path=spec_path,
                project_root=PROJECT_ROOT,
                augment_level=level,
                seed=seed,
                config=config,
                keep_temp=False,
                verbose=verbose,
            )
            elapsed = time.time() - t0
            overall = result.get("overall_status", "?")
            sym = _SYM.get(overall, "?")
            tfs = ", ".join(result.get("transforms_summary", [])) or "(none)"
            print(f"          {sym} {overall:<12}  {elapsed:.1f}s  [{tfs}]", flush=True)
            results.append(result)
            out_json.write_text(json.dumps(results, indent=2))  # incremental

    return results


def generate_markdown(results: list[dict], levels: list[int], title: str) -> str:
    by_spec: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in results:
        by_spec[r["spec_id"]][r["augment_level"]] = r

    now = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# {title}",
        f"**Date:** {now}  |  **Seed:** 42",
        "",
        "## Results Matrix",
        "",
    ]
    level_cols = " | ".join(f"L{l}" for l in sorted(levels))
    lines.append(f"| Spec | {level_cols} | Transforms Applied |")
    lines.append("|------|" + "|".join(["---"] * len(levels)) + "|---|")

    for spec_id in sorted(by_spec):
        row = by_spec[spec_id]
        cells = []
        for l in sorted(levels):
            if l in row:
                s = row[l]["overall_status"]
                sym = _SYM.get(s, "?")
                cells.append(f"{sym} {s}")
            else:
                cells.append("-")
        tfs: list[str] = []
        for l in sorted(levels, reverse=True):
            if l in row:
                tfs = row[l].get("transforms_summary", [])
                if tfs:
                    break
        tf_str = ", ".join(tfs) if tfs else "*(none)*"
        lines.append(f"| {spec_id} | {' | '.join(cells)} | {tf_str} |")

    lines += ["", "## Summary Statistics", ""]
    for l in sorted(levels):
        lr = [r for r in results if r["augment_level"] == l]
        if not lr:
            continue
        n = len(lr)
        passes = sum(1 for r in lr if r["overall_status"] == "PASS")
        bf = sum(1 for r in lr if r["overall_status"] == "BUILD_FAIL")
        fails = sum(1 for r in lr if r["overall_status"] == "FAIL")
        errors = sum(1 for r in lr if r["overall_status"] == "ERROR")
        pct = f"{100 * passes // n}%" if n else "0%"
        lines.append(
            f"**Level {l}:** {passes}/{n} PASS ({pct})"
            f" | BUILD_FAIL={bf} | FAIL={fails} | ERROR={errors}"
        )
    lines.append("")

    lines += ["## Transform Frequency", ""]
    tf_ctr: Counter = Counter()
    for r in results:
        for tf in r.get("transforms_summary", []):
            tf_ctr[tf] += 1
    if tf_ctr:
        lines.append("| Transform | Times Applied |")
        lines.append("|-----------|--------------|")
        for tf, count in tf_ctr.most_common():
            lines.append(f"| {tf} | {count} |")
    else:
        lines.append("*(no transforms applied)*")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Batch augmentation verifier")
    p.add_argument("specs", nargs="+", type=Path, help="Spec JSON files")
    p.add_argument("--levels", "-l", nargs="+", type=int, default=[2])
    p.add_argument("--seed", "-s", type=int, default=42)
    p.add_argument("--config", default="correctness")
    p.add_argument("--out", required=True, help="Output prefix (no extension)")
    p.add_argument("--title", default="Augmentation Batch Results")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    spec_paths = [s.resolve() for s in args.specs if s.is_file()]
    if not spec_paths:
        print("ERROR: No valid spec files found.", file=sys.stderr)
        return 1

    print(f"Specs={len(spec_paths)}  Levels={args.levels}  Seed={args.seed}", flush=True)
    print(f"Output prefix: {args.out}", flush=True)
    print("", flush=True)

    results = run_batch(spec_paths, args.levels, args.seed, args.config, args.out, args.verbose)

    md = generate_markdown(results, args.levels, args.title)
    out_md = Path(args.out + ".md")
    out_md.write_text(md)

    n = len(results)
    passes = sum(1 for r in results if r["overall_status"] == "PASS")
    print(f"\nDone: {passes}/{n} PASS", flush=True)
    print(f"  JSON → {args.out}.json", flush=True)
    print(f"  MD   → {args.out}.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
