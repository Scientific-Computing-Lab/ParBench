"""Derive L0 passer-set for a given model from a canonical eval directory.

Phase 2 / Plan 02-05. Implements D-16 through D-21.

Filter: pass@1-of-any — cell (source_spec, target_spec) is a passer iff
the set of samples for that cell at augment_level=0 contains at least one
result with overall_status == "PASS". Samples with missing / unparseable
JSONs are treated as non-passes (permissive, matches D-21 exactly).

Output schema (D-19): flat list of {"source_spec": str, "target_spec": str,
"augment_level": 0} dicts. `augment_level: 0` is a placeholder — the consuming
`run_eval_batch.py --task-list` invocation overrides via `--augment-levels 1 2 3 4`.

Usage:
    python3 -m scripts.evaluation.derive_l0_passers \\
        --canonical-dir results/evaluation/qwen_canonical \\
        --model together-qwen-3.5-397b-a17b \\
        --out .planning/eval-selections/l0_passers_qwen.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from harness.constants import EXCLUDED_SPECS

log = logging.getLogger(__name__)


def _load_result(path: Path) -> dict[str, Any] | None:
    """Load a result JSON, mirroring analyze_eval.py:_load_complexity_lookup's
    try/except-and-warn-continue idiom. Returns None on any parse error."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: failed to load {path}: {e}", file=sys.stderr)
        return None


def derive_passers(canonical_dir: Path, model: str,
                   direction: str | None = None) -> list[dict[str, Any]]:
    """Return sorted passer list. Pure function — no filesystem writes.

    Per D-18: cell is a passer iff >=1 of its (augment_level=0, model==model) samples
    has overall_status == "PASS".
    Per D-21: cells with <3 samples emit a stderr warning but are still included if
    they have >=1 PASS.

    S7c: Optional `direction` filter of the form "SRC-to-TGT" (e.g. "cuda-to-omp").
    When provided, only cells whose source_spec ends in -SRC and target_spec ends in
    -TGT are considered. API is the final hyphen-delimited segment of unique_id
    (safe: no parallel_api in manifest.jsonl contains a "-"). direction=None = legacy.
    """
    if not canonical_dir.is_dir():
        raise FileNotFoundError(f"canonical dir not found: {canonical_dir}")

    src_api: str | None = None
    tgt_api: str | None = None
    if direction is not None:
        if "-to-" not in direction:
            raise ValueError(f"--direction must match pattern SRC-to-TGT, got {direction!r}")
        src_api, tgt_api = direction.split("-to-", 1)

    # Group samples per cell.
    cell_samples: dict[tuple[str, str], list[str]] = defaultdict(list)

    for path in sorted(canonical_dir.rglob("*.json")):
        obj = _load_result(path)
        if obj is None:
            continue
        if obj.get("model") != model:
            continue
        if obj.get("augment_level") != 0:
            continue
        src = obj.get("source_spec")
        tgt = obj.get("target_spec")
        status = obj.get("overall_status")
        if not src or not tgt:
            print(f"warning: {path} missing source_spec/target_spec", file=sys.stderr)
            continue
        if src in EXCLUDED_SPECS or tgt in EXCLUDED_SPECS:
            continue
        if src_api is not None:
            if src.rsplit("-", 1)[-1] != src_api or tgt.rsplit("-", 1)[-1] != tgt_api:
                continue
        cell_samples[(src, tgt)].append(status)

    passers: list[dict[str, Any]] = []
    for cell in sorted(cell_samples):
        samples = cell_samples[cell]
        n = len(samples)
        if n < 3:
            print(f"warning: cell {cell} has {n}/3 samples (incomplete)", file=sys.stderr)
        if any(s == "PASS" for s in samples):
            passers.append({
                "source_spec": cell[0],
                "target_spec": cell[1],
                "augment_level": 0,
            })

    return passers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive L0-passer set (pass@1-of-any) for a model from a canonical eval directory.",
    )
    parser.add_argument("--canonical-dir", type=Path, required=True,
                        help="Directory of canonical (L0) result JSONs (recursively scanned).")
    parser.add_argument("--model", type=str, required=True,
                        help="Model id to filter on (e.g. together-qwen-3.5-397b-a17b).")
    parser.add_argument("--direction", type=str, default=None, metavar="SRC-to-TGT",
                        help="Optional. Filters passers to the given translation direction "
                             "(e.g. cuda-to-omp). API extracted from the last hyphen-delimited "
                             "segment of each unique_id. When omitted, both directions pass "
                             "(legacy behavior).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: .planning/eval-selections/l0_passers_{model}.json).")
    args = parser.parse_args(argv)

    try:
        passers = derive_passers(args.canonical_dir, args.model, direction=args.direction)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_path = args.out
    if out_path is None:
        # Slugify model for filename safety.
        slug = args.model.replace("/", "_")
        out_path = Path(".planning/eval-selections") / f"l0_passers_{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(passers, indent=2) + "\n")
    print(f"wrote {len(passers)} passers to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
