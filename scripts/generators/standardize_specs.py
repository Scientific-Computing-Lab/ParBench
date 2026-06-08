#!/usr/bin/env python3
"""
scripts/generators/standardize_specs.py

Universal spec standardizer — sets files.translation_targets and
metadata.translation_complexity for ALL spec files across all benchmark suites.

Replaces populate_translation_targets.py (SESSION 1.5 Rodinia-only script).

Three API family rules:
  Family 1 (opencl): targets = only .cl files from prompt_payload. Always overwrite.
  Family 2 (omp, omp_target, openacc): preserve existing curated targets; else = prompt_payload.
  Family 3 (all others): targets = prompt_payload always.

Usage:
    python3 scripts/generators/standardize_specs.py \\
        --project-root /home/samyak/Desktop/parbench_sam

    # Preview changes without writing:
    python3 scripts/generators/standardize_specs.py \\
        --project-root /home/samyak/Desktop/parbench_sam --dry-run

    # Filter to a single suite:
    python3 scripts/generators/standardize_specs.py \\
        --project-root /home/samyak/Desktop/parbench_sam --suite rodinia
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API family classification
# ---------------------------------------------------------------------------

# Family 1: APIs where device code lives in separate files.
# key = api name, value = list of device file extensions.
SEPARATE_DEVICE_APIS: dict[str, list[str]] = {
    "opencl": [".cl"],  # OpenCL kernels in .cl files, host in .c/.cpp
}

# Family 2: APIs where parallel constructs are inline pragmas.
# Preserve existing source-verified curated targets if present; else use prompt_payload.
PRAGMA_APIS: set[str] = {"omp", "omp_target", "openacc"}

# Family 3 (implicit default): All other APIs — translation_targets = prompt_payload.
# cuda, hip, sycl, kokkos, raja, mpi, omp_mpi, tbb, stdpar, thrust, serial

# Expected APIs per suite (for missing-variant reporting only)
EXPECTED_APIS_BY_SUITE: dict[str, set[str]] = {
    "rodinia": {"cuda", "omp", "opencl"},
    "hecbench": {"cuda", "omp"},
    "xsbench": {"cuda", "omp", "opencl", "omp_target"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_indent(raw: str) -> int:
    """Detect JSON indentation level from raw file content."""
    for line in raw.splitlines()[1:]:  # skip opening brace
        stripped = line.lstrip()
        if stripped:
            return len(line) - len(stripped)
    return 2  # default


def _compute_targets(
    api: str,
    prompt_payload: list[str],
    existing_targets: list[str] | None,
) -> list[str]:
    """Compute translation_targets for a spec based on its API family."""
    if api in SEPARATE_DEVICE_APIS:
        # Family 1: only device file extensions — ALWAYS overwrite
        extensions = SEPARATE_DEVICE_APIS[api]
        targets = [f for f in prompt_payload if any(f.endswith(ext) for ext in extensions)]
        if not targets:
            log.warning("No %s files found in prompt_payload — using full payload", extensions)
            targets = list(prompt_payload)
        return targets

    if api in PRAGMA_APIS:
        # Family 2: preserve existing curated targets if present
        if existing_targets:
            return list(existing_targets)
        return list(prompt_payload)

    # Family 3: full payload (cuda, hip, sycl, kokkos, raja, mpi, etc.)
    return list(prompt_payload)


def _compute_complexity(
    spec_id: str,
    api: str,
    all_specs: dict[str, dict],
) -> str | None:
    """Compute translation_complexity for a spec as a translation TARGET.

    Family 3 (cuda, hip, etc.) → always None (source-side APIs).
    Family 1 and 2 → auto-computed from sibling spec file counts.

    Returns None if no sibling specs with translation_targets exist.
    """
    # Family 3: source-side APIs — complexity depends on which target is chosen
    if api not in SEPARATE_DEVICE_APIS and api not in PRAGMA_APIS:
        return None

    spec = all_specs[spec_id]
    ident = spec.get("identity") or {}
    kernel = ident.get("kernel_name", "?")
    suite = ident.get("source_suite", "?")

    tgt_targets = (spec.get("files") or {}).get("translation_targets") or []
    tgt_count = len(tgt_targets)

    # Find sibling specs (same suite + kernel, different API)
    src_count_candidates: list[int] = []
    for sid, other in all_specs.items():
        if sid == spec_id:
            continue
        oid = other.get("identity") or {}
        if (oid.get("kernel_name") == kernel
                and oid.get("source_suite") == suite
                and oid.get("parallel_api") != api):
            other_targets = (other.get("files") or {}).get("translation_targets") or []
            if other_targets:
                src_count_candidates.append(len(other_targets))

    if not src_count_candidates:
        return None

    src_count = min(src_count_candidates)  # most conservative
    if src_count == 1 and tgt_count == 1:
        return "single_file"
    elif src_count > 1 and tgt_count == 1:
        return "multi_to_single"
    elif src_count == 1 and tgt_count > 1:
        return "single_to_multi"
    else:
        return "multi_to_multi"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_all_specs(
    specs_dir: Path,
    suite_filter: str | None = None,
    dry_run: bool = False,
) -> int:
    """Load all specs, standardize fields, optionally write back.

    Returns the number of validation errors encountered.
    """
    # Load all specs into memory (needed for cross-spec complexity computation)
    all_specs: dict[str, dict] = {}
    spec_files: list[Path] = []
    spec_raw: dict[str, str] = {}  # raw JSON for indentation detection

    for fn in sorted(specs_dir.glob("*.json")):
        if suite_filter and not fn.name.startswith(f"{suite_filter}-"):
            continue
        raw = fn.read_text(encoding="utf-8")
        spec = json.loads(raw)
        spec_id = fn.stem
        all_specs[spec_id] = spec
        spec_files.append(fn)
        spec_raw[spec_id] = raw

    log.info("Loaded %d specs (filter: %s)", len(spec_files), suite_filter or "all")

    # --- Report missing API variants (informational only) ---
    kernel_apis: dict[tuple[str, str], set[str]] = defaultdict(set)
    for spec_id, spec in all_specs.items():
        ident = spec.get("identity") or {}
        suite = ident.get("source_suite", "?")
        kernel = ident.get("kernel_name", "?")
        api = ident.get("parallel_api", "?")
        kernel_apis[(suite, kernel)].add(api)

    print("\n=== Missing API Variants ===")
    any_missing = False
    for (suite, kernel), present_apis in sorted(kernel_apis.items()):
        expected = EXPECTED_APIS_BY_SUITE.get(suite, set())
        missing = expected - present_apis
        if missing:
            print(f"  {suite}/{kernel}: missing {sorted(missing)}")
            any_missing = True
    if not any_missing:
        print("  (none — all expected variants present)")
    print()

    # ── PASS 1: Compute and set translation_targets ──────────────────────
    errors = 0
    new_targets_map: dict[str, list[str]] = {}

    for fn in spec_files:
        spec_id = fn.stem
        spec = all_specs[spec_id]
        ident = spec.get("identity") or {}
        api = ident.get("parallel_api", "")
        prompt_payload: list[str] = (spec.get("files") or {}).get("prompt_payload") or []
        existing_targets: list[str] | None = (spec.get("files") or {}).get("translation_targets")

        new_targets = _compute_targets(api, prompt_payload, existing_targets)

        # Validate: all targets must be in prompt_payload
        payload_set = set(prompt_payload)
        invalid = [t for t in new_targets if t not in payload_set]
        if invalid:
            log.error("%s: translation_targets entries NOT in prompt_payload: %s",
                      spec_id, invalid)
            errors += 1
            new_targets = list(prompt_payload)  # safe fallback

        new_targets_map[spec_id] = new_targets

        # Update all_specs in memory so Pass 2 can see everyone's targets
        if spec.get("files") is None:
            spec["files"] = {}
        spec["files"]["translation_targets"] = new_targets

    # ── PASS 2: Compute translation_complexity ───────────────────────────
    new_complexity_map: dict[str, str | None] = {}

    for fn in spec_files:
        spec_id = fn.stem
        spec = all_specs[spec_id]
        ident = spec.get("identity") or {}
        api = ident.get("parallel_api", "")
        orig_metadata = json.loads(spec_raw[spec_id]).get("metadata") or {}
        has_complexity_field = "translation_complexity" in orig_metadata
        existing_complexity = orig_metadata.get("translation_complexity")

        if api in SEPARATE_DEVICE_APIS:
            # Family 1: always recompute (targets changed via .cl narrowing)
            new_complexity = _compute_complexity(spec_id, api, all_specs)
        elif api in PRAGMA_APIS:
            # Family 2: preserve existing if field is present; else compute
            if has_complexity_field:
                new_complexity = existing_complexity
            else:
                new_complexity = _compute_complexity(spec_id, api, all_specs)
        else:
            # Family 3: always null (source-side APIs)
            new_complexity = None

        new_complexity_map[spec_id] = new_complexity

    # ── Report and apply changes ─────────────────────────────────────────
    changes = 0
    for fn in spec_files:
        spec_id = fn.stem
        raw = spec_raw[spec_id]
        orig_spec = json.loads(raw)

        orig_targets = (orig_spec.get("files") or {}).get("translation_targets")
        orig_metadata = orig_spec.get("metadata") or {}
        orig_complexity = orig_metadata.get("translation_complexity")
        orig_has_complexity = "translation_complexity" in orig_metadata

        new_targets = new_targets_map[spec_id]
        new_complexity = new_complexity_map[spec_id]

        targets_changed = orig_targets != new_targets
        complexity_changed = orig_complexity != new_complexity or not orig_has_complexity

        if not targets_changed and not complexity_changed:
            continue

        changes += 1
        if targets_changed:
            old_n = len(orig_targets) if orig_targets else 0
            new_n = len(new_targets)
            print(f"  {spec_id}: targets {old_n} → {new_n} files")
            if orig_targets and new_n < old_n:
                removed = set(orig_targets) - set(new_targets)
                print(f"    removed: {sorted(removed)}")
            elif not orig_targets:
                print(f"    added: {new_targets}")
        if complexity_changed:
            print(f"  {spec_id}: complexity {orig_complexity!r} → {new_complexity!r}")

        if not dry_run:
            spec_to_write = json.loads(raw)
            if spec_to_write.get("files") is None:
                spec_to_write["files"] = {}
            spec_to_write["files"]["translation_targets"] = new_targets
            if spec_to_write.get("metadata") is None:
                spec_to_write["metadata"] = {}
            spec_to_write["metadata"]["translation_complexity"] = new_complexity

            indent = _detect_indent(raw)
            fn.write_text(json.dumps(spec_to_write, indent=indent) + "\n",
                          encoding="utf-8")

    action = "[DRY RUN] Would update" if dry_run else "Updated"
    print(f"\n{action} {changes}/{len(spec_files)} specs")
    if errors:
        print(f"ERRORS: {errors} specs had invalid translation_targets (check logs)")

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standardize translation_targets and translation_complexity "
                    "in all spec files."
    )
    parser.add_argument(
        "--project-root", required=True, type=Path,
        help="Root of the parbench project",
    )
    parser.add_argument(
        "--suite", default=None,
        help="Optional suite filter: 'rodinia', 'hecbench', etc.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    specs_dir = args.project_root / "specs"
    if not specs_dir.exists():
        log.error("specs/ directory not found at %s", specs_dir)
        return 1

    errors = process_all_specs(
        specs_dir=specs_dir,
        suite_filter=args.suite,
        dry_run=args.dry_run,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
