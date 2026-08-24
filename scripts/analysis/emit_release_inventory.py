#!/usr/bin/env python3
"""Emit results/analysis/final/release_inventory.json - the canonical pointer
file recording the #9 release export inventory, so W-24's ledger entries can
cite it under the final/-only source allowlist.

The three record/file counts are RE-DERIVED here by running export_translations
against the immutable submitted corpus (results/evaluation), so any future
checker reproduces them deterministically over immutable inputs. export_
translations is allowlisted for that read; this generator only shells out to it
and never touches the sealed/immutable namespace directly.

The release zip's sha256 is a RECORDED provenance value: the scrubbed release
zip is an on-demand, uncommitted build artifact this generator cannot recompute
(it does not rebuild the zip). It enters via --zip-sha256 and is emitted as
`zip_sha256_recorded` alongside `source_commits` so no reader mistakes it for a
value the generator independently verified.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_SUMMARY_RE = re.compile(
    r"records_with_code=(\d+) extraction_fail_empty=(\d+) files=(\d+)")


def _export_counts(project_root: Path) -> tuple[int, int, int]:
    """Run export_translations over the immutable corpus into a throwaway dir
    and parse its summary line. Deterministic over immutable inputs."""
    script = project_root / "scripts" / "rebuttal" / "export_translations.py"
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, str(script),
             "--project-root", str(project_root),
             "--results-namespace", "results/evaluation",
             "--out-dir", tmp],
            capture_output=True, text=True, cwd=str(project_root),
        )
        if result.returncode != 0:
            raise SystemExit(
                f"ERROR: export_translations failed:\n{result.stderr}")
        match = _SUMMARY_RE.search(result.stdout)
        if not match:
            raise SystemExit(
                f"ERROR: could not parse export summary line:\n{result.stdout}")
        return int(match.group(1)), int(match.group(2)), int(match.group(3))


def build_inventory(project_root: Path, zip_sha256: str,
                    source_commits: list[str]) -> dict:
    with_source, without_source, generated_files = _export_counts(project_root)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root), stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_hash": git_hash,
        "source": {
            "translations_namespace": "results/evaluation",
            "release_result_namespace": "results/evaluation_final/parbench-final-v1",
            "counts_method": "re-derived by scripts/rebuttal/export_translations.py "
                             "over the immutable corpus",
        },
        "records_with_source": with_source,
        "records_without_source": without_source,
        "generated_files": generated_files,
        "zip_sha256_recorded": zip_sha256,
        "source_commits": list(source_commits),
        "note": "records_* and generated_files are re-derived from the immutable "
                "corpus by export_translations; zip_sha256_recorded is a recorded "
                "hash of the on-demand release zip build artifact, NOT recomputed "
                "here.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--output", default=None,
                    help="default: <project-root>/results/analysis/final/release_inventory.json")
    ap.add_argument("--zip-sha256", required=True,
                    help="recorded sha256 of the #9 release zip (build artifact, "
                         "not recomputed here)")
    ap.add_argument("--source-commits", nargs="+", required=True,
                    help="the #9 build commit ids that produced the release")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    inv = build_inventory(project_root, args.zip_sha256, args.source_commits)
    out = (Path(args.output) if args.output
           else project_root / "results" / "analysis" / "final" / "release_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, indent=2) + "\n")
    print(f"wrote {out}: records_with_source={inv['records_with_source']} "
          f"records_without_source={inv['records_without_source']} "
          f"generated_files={inv['generated_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
