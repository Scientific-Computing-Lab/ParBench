#!/usr/bin/env python3
"""Cross-check the record-driven oracle census against the direct spec scan.

Two independent producers classify each spec's oracle:
  * ``scripts/rebuttal/oracle_shape_census.py`` walks the evaluated result
    records and records the shape of each *target* spec it saw.
  * ``scripts/spec_tools/direct_oracle_scan.py`` walks every spec file and
    classifies it directly.

They must agree. This tool fails closed on:
  * any target spec in the census whose shape differs from the direct scan
    (or is absent from the direct scan);
  * a direct scan that does not cover ``--expected-specs`` specs;
  * (optional, gate21 finding 5) any spec in the generated oracle-contract report
    (``--oracle-report``) that is ABSENT from the direct scan, or whose report
    ``generated_oracle_strength`` disagrees with the scan's INDEPENDENT strength
    classifier (``independent_strength`` - a separate code path from the audit
    classifier the report used, so the check is not tautological).

Usage:
    python3 scripts/analysis/compare_oracle_censuses.py \\
        --census results/analysis/final_oracle_shape_census.json \\
        --direct-scan results/analysis/final_direct_oracle_scan.json \\
        --expected-specs 206
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def compare(
    census: dict,
    direct_scan: dict,
    expected_specs: int,
    oracle_report: dict | None = None,
) -> list[str]:
    """Return a list of disagreement strings; empty means the censuses agree."""
    failures: list[str] = []

    scan_count = direct_scan.get("spec_count")
    if scan_count != expected_specs:
        failures.append(
            f"direct scan covers {scan_count} specs, expected {expected_specs}"
        )

    census_shapes = census.get("shapes_by_spec") or {}
    scan_shapes = direct_scan.get("shapes_by_spec") or {}
    if not census_shapes:
        failures.append("census has no shapes_by_spec map")
    if not scan_shapes:
        failures.append("direct scan has no shapes_by_spec map")

    for spec_id, shape in sorted(census_shapes.items()):
        if spec_id not in scan_shapes:
            failures.append(f"{spec_id}: in census but absent from direct scan")
        elif scan_shapes[spec_id] != shape:
            failures.append(
                f"{spec_id}: census shape {shape!r} != direct scan "
                f"{scan_shapes[spec_id]!r}"
            )

    if oracle_report is not None:
        report_specs = oracle_report.get("specs") or {}
        strengths = direct_scan.get("strength_by_spec") or {}
        # gate21 finding 5: check the report against the scan's INDEPENDENT strength
        # classifier (not the one the report itself used - that would be
        # tautological), and fail closed on any report spec absent from the scan.
        # Iterating the REPORT (not the scan) is what closes the silent-skip hole:
        # a report spec the scan never covered used to pass unnoticed.
        for spec_id in sorted(report_specs):
            info = strengths.get(spec_id)
            if info is None:
                failures.append(
                    f"{spec_id}: in oracle report but absent from direct scan"
                )
                continue
            report_strength = (report_specs[spec_id] or {}).get("generated_oracle_strength")
            indep = info.get("independent_strength")
            if report_strength != indep:
                failures.append(
                    f"{spec_id}: independent scan strength {indep!r} "
                    f"!= oracle report {report_strength!r}"
                )

    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--direct-scan", type=Path, required=True)
    ap.add_argument("--expected-specs", type=int, required=True)
    ap.add_argument(
        "--oracle-report",
        type=Path,
        default=None,
        help="Optional generated oracle-contract report to check strengths against.",
    )
    args = ap.parse_args(argv)

    census = json.loads(args.census.read_text(encoding="utf-8"))
    direct_scan = json.loads(args.direct_scan.read_text(encoding="utf-8"))
    oracle_report = (
        json.loads(args.oracle_report.read_text(encoding="utf-8"))
        if args.oracle_report is not None
        else None
    )

    failures = compare(census, direct_scan, args.expected_specs, oracle_report)
    if failures:
        for f in failures:
            print(f"MISMATCH: {f}", file=sys.stderr)
        print(f"\n{len(failures)} disagreement(s) between censuses", file=sys.stderr)
        return 1

    print(
        f"OK: direct scan covers {direct_scan.get('spec_count')} specs; "
        f"all {len(census.get('shapes_by_spec') or {})} census target specs "
        "agree with the direct scan"
        + ("" if oracle_report is None else "; strengths agree with the oracle report")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
