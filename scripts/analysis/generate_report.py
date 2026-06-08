#!/usr/bin/env python3
"""
generate_report.py — ParBench pilot summary report generator.

Reads manifest.jsonl and all spec files, runs validation checks, and
produces a Markdown summary report.

Usage:
    python generate_report.py            # prints to stdout + saves to reports/pilot_report.md
    python generate_report.py --stdout   # prints to stdout only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifest.jsonl"
SPECS_DIR = PROJECT_ROOT / "specs"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load all entries from manifest.jsonl."""
    entries: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _translation_pairs(n: int) -> int:
    """Number of ordered translation pairs for n API variants."""
    return comb(n, 2) * 2  # A→B and B→A


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_data(
    manifest_path: Path,
) -> dict[str, Any]:
    """
    Gather all data needed for the report.

    Returns a dict with keys:
        manifest_entries, specs, kernel_apis, api_kernels,
        total_kernels, total_specs, total_pairs, per_kernel_pairs,
        all_apis, file_classifications
    """
    entries = _load_manifest(manifest_path)

    # (source_suite, kernel_name) → list[api]  — tuple key prevents cross-suite collisions
    kernel_apis: dict[tuple[str, str], list[str]] = defaultdict(list)
    # api → list[(source_suite, kernel_name)]
    api_kernels: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # (source_suite, kernel_name) → {api → spec_dict}
    specs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    # (source_suite, kernel_name) → {api → file_classifications}
    file_classifications: dict[tuple[str, str], dict[str, dict[str, list[str]]]] = defaultdict(dict)

    all_apis: set[str] = set()

    for entry in entries:
        suite = entry.get("source_suite", "unknown")
        kname = entry["kernel_name"]
        key = (suite, kname)
        api = entry["parallel_api"]
        kernel_apis[key].append(api)
        api_kernels[api].append(key)
        all_apis.add(api)

        spec_path = PROJECT_ROOT / entry["spec_file"]
        if spec_path.exists():
            spec = _load_json(spec_path)
            specs[key][api] = spec
            files_section = spec.get("files", {})
            file_classifications[key][api] = {
                "prompt_payload": files_section.get("prompt_payload", []),
                "support_files": files_section.get("support_files", []),
                "verification_only": files_section.get("verification_only", []),
            }

    total_kernels = len(kernel_apis)
    total_specs = len(entries)
    per_kernel_pairs: dict[tuple[str, str], int] = {}
    total_pairs = 0
    for key, apis in kernel_apis.items():
        p = _translation_pairs(len(apis))
        per_kernel_pairs[key] = p
        total_pairs += p

    return {
        "manifest_entries": entries,
        "specs": dict(specs),
        "kernel_apis": dict(kernel_apis),
        "api_kernels": dict(api_kernels),
        "total_kernels": total_kernels,
        "total_specs": total_specs,
        "total_pairs": total_pairs,
        "per_kernel_pairs": per_kernel_pairs,
        "all_apis": sorted(all_apis),
        "file_classifications": dict(file_classifications),
    }


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------


def run_validation() -> tuple[int, int, str]:
    """
    Run validate_schema.py --all and capture the result.

    Returns (pass_count, total_count, raw_output).
    """
    validate_script = PROJECT_ROOT / "scripts" / "validate_schema.py"
    result = subprocess.run(
        [sys.executable, str(validate_script), "--all"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    output = result.stdout + result.stderr

    # Count how many specs passed by looking for "✓ spec" lines
    pass_count = output.count("✓ spec")
    # Total is the number of spec files
    total_count = len(list(SPECS_DIR.glob("*.json"))) if SPECS_DIR.exists() else 0

    return pass_count, total_count, output


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(manifest_path: Path) -> str:
    """Generate the full Markdown report string."""
    data = collect_data(manifest_path)

    # Run validation
    val_pass, val_total, val_output = run_validation()

    lines: list[str] = []
    w = lines.append  # shorthand

    w("# ParBench Pilot Report")
    w("")
    w(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_")
    w("")

    # ---- Summary ----
    w("## Summary")
    w("")
    w(f"- **Total kernels:** {data['total_kernels']}")
    w(f"- **Total specs:** {data['total_specs']}")
    n_apis = len(data["all_apis"])
    pairs_per = comb(n_apis, 2) * 2
    w(
        f"- **Total translation pairs:** {data['total_pairs']} "
        f"({data['total_kernels']} kernels × {pairs_per} pairs each)"
    )
    w("")

    # ---- Kernel Coverage Matrix ----
    w("## Kernel Coverage")
    w("")
    api_order = sorted(data["all_apis"])

    # Table header
    header_cols = ["Kernel"] + api_order + ["APIs", "Pairs"]
    w("| " + " | ".join(header_cols) + " |")
    w("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for (suite, kname) in sorted(data["kernel_apis"].keys()):
        key = (suite, kname)
        apis_present = set(data["kernel_apis"][key])
        cells = [f"{suite}/{kname}"]
        for api in api_order:
            cells.append("✓" if api in apis_present else "—")
        cells.append(str(len(apis_present)))
        cells.append(str(data["per_kernel_pairs"][key]))
        w("| " + " | ".join(cells) + " |")

    w("")

    # ---- API Distribution ----
    w("## API Distribution")
    w("")
    for api in api_order:
        count = len(data["api_kernels"].get(api, []))
        w(f"- **{api}:** {count} kernel(s)")
    w("")

    # ---- File Classifications ----
    w("## File Classifications")
    w("")
    for (suite, kname) in sorted(data["file_classifications"].keys()):
        key = (suite, kname)
        w(f"### {suite}/{kname}")
        w("")
        for api in api_order:
            fc = data["file_classifications"].get(key, {}).get(api)
            if fc is None:
                continue
            w(f"**{api}:**")
            w(f"- prompt_payload: {', '.join(fc['prompt_payload']) or '(none)'}")
            w(f"- support_files: {', '.join(fc['support_files']) or '(none)'}")
            w(f"- verification_only: {', '.join(fc['verification_only']) or '(none)'}")
            w("")

    # ---- Validation Status ----
    w("## Validation Status")
    w("")
    if val_pass == val_total and val_total > 0:
        w(f"✓ **{val_pass}/{val_total}** specs pass all checks")
    else:
        w(f"**{val_pass}/{val_total}** specs pass all checks")
    w("")
    w("### Validation Output")
    w("")
    w("```")
    w(val_output.strip())
    w("```")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ParBench pilot summary report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to manifest.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout only (don't save to file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: reports/pilot_report.md)",
    )

    args = parser.parse_args()
    manifest_path = Path(args.manifest)

    report = generate_report(manifest_path)

    # Always print to stdout
    print(report)

    # Save to file unless --stdout-only
    if not args.stdout:
        output_path = (
            Path(args.output) if args.output else REPORTS_DIR / "pilot_report.md"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\n--- Report saved to {output_path} ---")


if __name__ == "__main__":
    main()
