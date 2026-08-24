#!/usr/bin/env python3
"""Directly scan every spec's verification block for its oracle shape/strength.

This is the independent counterpart to the record-driven
``scripts/rebuttal/oracle_shape_census.py``. Where the census classifies the
oracle shapes that actually appear as translation *targets* in the evaluated
corpus, this scan walks the whole spec directory (206 specs) and classifies
each one directly from its ``verification`` block - it never looks at a result
record. ``scripts/analysis/compare_oracle_censuses.py`` then cross-checks the
two: every spec the census touched must carry the same shape here, and this
scan must cover the full expected spec count.

Two independent classifiers are recorded per spec so downstream comparison can
check both:
  * ``shape``  - the three-way partition (``numeric/hash`` | ``self-checking``
                 | ``banner-only``). Classified here by an INDEPENDENT
                 reimplementation (``_shape_from_verification``) that does not
                 import the census's ``oracle_shape``; the cross-check in
                 ``compare_oracle_censuses.py`` is only meaningful when the two
                 sides are separate code paths (gate1 finding 8, 2026-08-12).
  * ``generated_strength`` / ``audit_class`` - the audit classifier from
    ``scripts/spec_tools/audit_oracle_contracts.classify_oracle`` (this is the
    strength that the generated oracle-contract report records, recorded here for
    provenance; on its own it agrees with the report by construction).
  * ``independent_strength`` - an INDEPENDENT strength classifier
    (``_strength_from_verification``) that does NOT import the audit classifier
    (gate21 finding 5). ``compare_oracle_censuses.py`` checks the oracle-contract
    report against THIS strength, so the report cross-check is a genuine
    two-code-path comparison, not a tautology.

Usage:
    python3 scripts/spec_tools/direct_oracle_scan.py \\
        --spec-root specs \\
        --json-out results/analysis/final_direct_oracle_scan.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spec_tools.audit_oracle_contracts import (  # noqa: E402
    GENERATED_STRENGTH,
    classify_oracle,
)

# Independent self-check regex. Kept deliberately separate from the census's
# copy so the two classifiers are genuinely distinct code paths; a drift
# between them is exactly what compare_oracle_censuses.py exists to catch.
_SELF_CHECK = re.compile(r"(?i)\bpass\b|passed|verif|correct|match|\bok\b|success")


def _shape_from_verification(spec: dict) -> str:
    """Classify a spec's oracle shape directly from its verification block.

    Independent reimplementation of the census's DEFINED shape semantics - it
    shares no code with ``oracle_shape_census.oracle_shape`` (gate1 finding 8),
    so an edit to EITHER implementation makes ``compare_oracle_censuses`` fail.
    It reproduces the same definition (not a redefinition): 136 corpus specs and
    the committed rebuttal shape tables depend on that exact definition, so this
    cross-check verifies implementation fidelity, it does not reclassify.

    Definition (mirrored from the census):
      numeric/hash  - declares a file_hash or numeric_comparison strategy
      self-checking - the matched stdout pattern contains a self-verdict token
      banner-only   - otherwise
    The matched pattern is the top-level ``stdout_pattern`` if present, else the
    POSITIVE ``stdout_pattern`` strategy. A ``stdout_exclude_pattern`` strategy
    (Task 3 negative conditions) also carries a ``pattern`` field but is a
    negative guard, not a self-verdict, so it must never drive the shape
    (gate13 finding 1). The census applies the identical rule; this is an
    independent reimplementation of it, not a redefinition.
    """
    v = spec.get("verification") or {}
    strat = v.get("strategies")
    if strat is None:
        strat = v.get("strategy") or []
    if isinstance(strat, str):
        strat = [strat]

    declared_types = set()
    for entry in strat:
        if isinstance(entry, str):
            declared_types.add(entry)
        elif isinstance(entry, dict):
            declared_types.add(entry.get("type") or entry.get("name"))
    if declared_types & {"file_hash", "numeric_comparison"}:
        return "numeric/hash"

    pattern = v.get("stdout_pattern") or ""
    if not pattern:
        for entry in strat:
            if (isinstance(entry, dict)
                    and (entry.get("type") or entry.get("name")) == "stdout_pattern"
                    and entry.get("pattern")):
                pattern = entry["pattern"]
                break
    return "self-checking" if _SELF_CHECK.search(pattern) else "banner-only"


def _strength_from_verification(spec: dict) -> str:
    """Classify oracle STRENGTH directly from the verification block, INDEPENDENTLY
    of ``audit_oracle_contracts.classify_oracle`` (gate21 finding 5).

    The generated oracle-contract report records ``generated_oracle_strength`` via
    that audit classifier. Reproducing the strength by importing the SAME
    classifier made the report cross-check in ``compare_oracle_censuses.py``
    tautological (identical code on both sides). This is a genuinely separate code
    path that reproduces the same DEFINITION, so a drift between the two makes the
    cross-check fail - which is the point of the check.

    Definition (mirrors GENERATED_STRENGTH, computed here without sharing code):
      strong - an independent comparator strategy (file_hash / file_diff)
      medium - a program-reported numeric strategy (numeric_comparison)
      weak   - anything else (self-claimed banner / exit code / none)
    """
    v = spec.get("verification") or {}
    strat = v.get("strategies")
    if strat is None:
        strat = v.get("strategy") or []
    if isinstance(strat, str):
        strat = [strat]
    types = set()
    for entry in strat:
        if isinstance(entry, str):
            types.add(entry)
        elif isinstance(entry, dict):
            types.add(entry.get("type") or entry.get("name"))
    if types & {"file_hash", "file_diff"}:
        return "strong"
    if "numeric_comparison" in types:
        return "medium"
    return "weak"


def scan(spec_root: Path) -> dict:
    spec_paths = sorted(spec_root.glob("*.json"))
    shapes_by_spec: dict[str, str] = {}
    strength_by_spec: dict[str, dict] = {}
    shape_counts: Counter[str] = Counter()
    generated_strength_counts: Counter[str] = Counter()
    audit_class_counts: Counter[str] = Counter()

    for path in spec_paths:
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec_id = spec.get("unique_id", path.stem)
        shape = _shape_from_verification(spec)
        audit_class = classify_oracle(spec)
        generated_strength = GENERATED_STRENGTH[audit_class]
        independent_strength = _strength_from_verification(spec)
        tagged = (spec.get("verification") or {}).get("oracle_strength")

        shapes_by_spec[spec_id] = shape
        strength_by_spec[spec_id] = {
            "tagged": tagged,
            "audit_class": audit_class,
            "generated_strength": generated_strength,
            # gate21 finding 5: an INDEPENDENT strength, computed without the audit
            # classifier, so compare_oracle_censuses' report cross-check is real.
            "independent_strength": independent_strength,
        }
        shape_counts[shape] += 1
        generated_strength_counts[generated_strength] += 1
        audit_class_counts[audit_class] += 1

    return {
        "artifact": "direct_oracle_scan",
        "generated_by": "scripts/spec_tools/direct_oracle_scan.py",
        "spec_root": str(spec_root),
        "spec_count": len(spec_paths),
        "shape_counts": dict(sorted(shape_counts.items())),
        "generated_strength_counts": dict(sorted(generated_strength_counts.items())),
        "audit_class_counts": dict(sorted(audit_class_counts.items())),
        "shapes_by_spec": dict(sorted(shapes_by_spec.items())),
        "strength_by_spec": dict(sorted(strength_by_spec.items())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-root", type=Path, default=PROJECT_ROOT / "specs")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    spec_root = args.spec_root.resolve()
    if not spec_root.is_dir():
        print(f"ABORT: spec root {spec_root} is not a directory", file=sys.stderr)
        return 1

    doc = scan(spec_root)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")

    print(
        f"scanned {doc['spec_count']} specs; shapes {doc['shape_counts']}; "
        f"generated strengths {doc['generated_strength_counts']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
