#!/usr/bin/env python3
"""
validate_schema.py — ParBench schema validator.

Validates manifest.jsonl entries and Level 2 spec JSON files against
their respective JSON Schemas, plus deep cross-cutting consistency checks.

Usage:
    python validate_schema.py --manifest manifest.jsonl
    python validate_schema.py --spec specs/rodinia-bfs-cuda.json
    python validate_schema.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    print("ERROR: jsonschema is required. Install with: pip install jsonschema")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths — everything is relative to this project root (parbench/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = PROJECT_ROOT / "schema"
MANIFEST_SCHEMA_PATH = SCHEMA_DIR / "manifest_schema.json"
SPEC_SCHEMA_PATH = SCHEMA_DIR / "spec_schema.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifest.jsonl"

# Load downloads_root from config/paths.json
_config_path = PROJECT_ROOT / "config" / "paths.json"
if _config_path.exists():
    with open(_config_path, "r", encoding="utf-8") as _f:
        _config = json.load(_f)
    DOWNLOADS_ROOT = Path(_config.get("downloads_root", str(PROJECT_ROOT))).resolve()
else:
    DOWNLOADS_ROOT = PROJECT_ROOT


# ---------------------------------------------------------------------------
# Schema loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_validator(schema_path: Path) -> Draft7Validator:
    """Build a Draft-07 validator from a schema file."""
    schema = _load_json(schema_path)
    Draft7Validator.check_schema(schema)  # fail fast if schema itself is bad
    return Draft7Validator(schema)


def _resolve_source_dir(repo_root_rel: str, source_path: str) -> Path:
    """Resolve the absolute path to a kernel's source directory."""
    return (DOWNLOADS_ROOT / repo_root_rel / source_path).resolve()


# ---------------------------------------------------------------------------
# Manifest-level validation
# ---------------------------------------------------------------------------


def validate_manifest_entry(entry: dict[str, Any]) -> list[str]:
    """
    Validate a single manifest.jsonl entry against the manifest schema.

    Returns a list of human-readable error strings (empty = valid).
    """
    errors: list[str] = []
    validator = _get_validator(MANIFEST_SCHEMA_PATH)

    for err in sorted(validator.iter_errors(entry), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.absolute_path) or "(root)"
        errors.append(f"[manifest] {path}: {err.message}")

    # Cross-check: spec_file exists on disk
    if "spec_file" in entry:
        spec_path = PROJECT_ROOT / entry["spec_file"]
        if not spec_path.exists():
            errors.append(
                f"[manifest] spec_file: '{entry['spec_file']}' does not exist on disk"
            )

    # Cross-check: source_dir exists on disk
    if "source_dir" in entry:
        source = (DOWNLOADS_ROOT / entry["source_dir"]).resolve()
        if not source.exists():
            errors.append(
                f"[manifest] source_dir: '{entry['source_dir']}' does not exist on disk"
            )

    return errors


# ---------------------------------------------------------------------------
# Spec-level validation
# ---------------------------------------------------------------------------


def validate_spec(
    spec: dict[str, Any], spec_file_path: Path | None = None
) -> list[str]:
    """
    Validate a Level 2 spec dict against the spec schema, plus cross-cutting
    consistency checks.

    Parameters
    ----------
    spec : dict
        Parsed spec JSON.
    spec_file_path : Path, optional
        Absolute path to the spec file (used for filename-vs-unique_id check).

    Returns a list of human-readable error strings (empty = valid).
    """
    errors: list[str] = []
    warnings: list[str] = []
    validator = _get_validator(SPEC_SCHEMA_PATH)

    # ---- JSON Schema validation ----
    for err in sorted(validator.iter_errors(spec), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.absolute_path) or "(root)"
        errors.append(f"[spec] {path}: {err.message}")

    # ---- Cross-cutting checks (only if basic structure is present) ----

    identity = spec.get("identity", {})
    unique_id = identity.get("unique_id", "")

    # 1. unique_id matches filename (minus .json)
    if spec_file_path and unique_id:
        expected_stem = spec_file_path.stem
        if unique_id != expected_stem:
            errors.append(
                f"[spec] identity.unique_id '{unique_id}' does not match "
                f"filename '{expected_stem}.json'"
            )

    # 2. unique_id format matches identity fields
    if unique_id and all(
        k in identity for k in ("source_suite", "kernel_name", "parallel_api")
    ):
        expected_id = (
            f"{identity['source_suite']}-{identity['kernel_name']}-"
            f"{identity['parallel_api']}"
        )
        if unique_id != expected_id:
            errors.append(
                f"[spec] identity.unique_id '{unique_id}' does not match expected "
                f"'{expected_id}' (from source_suite-kernel_name-parallel_api)"
            )

    # 3. implementation.api matches identity.parallel_api
    impl_api = spec.get("implementation", {}).get("api", "")
    id_api = identity.get("parallel_api", "")
    if impl_api and id_api and impl_api != id_api:
        errors.append(
            f"[spec] implementation.api '{impl_api}' != identity.parallel_api '{id_api}'"
        )

    # ---- (a) PATH RESOLUTION — check every classified file exists on disk ----

    provenance = spec.get("provenance", {})
    repo_root_rel = provenance.get("repo_root", "")
    source_path = provenance.get("source_path", "")

    if repo_root_rel:
        repo_root = (DOWNLOADS_ROOT / repo_root_rel).resolve()
    else:
        repo_root = None

    base_dir: Path | None = None
    if repo_root and repo_root.exists():
        base_dir = repo_root / source_path if source_path else repo_root

    files_section = spec.get("files", {})
    all_file_lists: dict[str, list[str]] = {
        "prompt_payload": files_section.get("prompt_payload", []),
        "support_files": files_section.get("support_files", []),
        "verification_only": files_section.get("verification_only", []),
    }

    if base_dir and base_dir.exists():
        for list_name, file_list in all_file_lists.items():
            for fname in file_list:
                full_path = base_dir / fname
                if not full_path.exists():
                    errors.append(
                        f"[spec] files.{list_name}: '{fname}' not found at "
                        f"'{full_path}'"
                    )

    # ---- (b) FILE CLASSIFICATION INTEGRITY ----

    prompt_set = set(all_file_lists["prompt_payload"])
    support_set = set(all_file_lists["support_files"])
    verif_set = set(all_file_lists["verification_only"])

    # No file in both prompt_payload and verification_only (ERROR)
    pv_overlap = prompt_set & verif_set
    if pv_overlap:
        errors.append(
            f"[spec] files: these files appear in BOTH prompt_payload and "
            f"verification_only (forbidden): {sorted(pv_overlap)}"
        )

    # No file in both prompt_payload and support_files (WARN)
    ps_overlap = prompt_set & support_set
    if ps_overlap:
        warnings.append(
            f"[spec] files: these files appear in BOTH prompt_payload and "
            f"support_files (may be intentional): {sorted(ps_overlap)}"
        )

    # Orphan-file check: every file in the source directory should be in at least one list
    all_classified = prompt_set | support_set | verif_set
    if base_dir and base_dir.exists():
        on_disk = {f.name for f in base_dir.iterdir() if f.is_file()}
        orphans = on_disk - all_classified
        if orphans:
            warnings.append(
                f"[spec] files: orphan files in source dir not in any list: "
                f"{sorted(orphans)}"
            )

    # ---- (d) TRANSLATION_TARGETS ⊂ PROMPT_PAYLOAD ----

    translation_targets = files_section.get("translation_targets", [])
    if translation_targets:
        tt_set = set(translation_targets)
        not_in_payload = tt_set - prompt_set
        if not_in_payload:
            errors.append(
                f"[spec] files: translation_targets entries not in prompt_payload: "
                f"{sorted(not_in_payload)}"
            )
        if len(translation_targets) != len(tt_set):
            warnings.append(
                "[spec] files: translation_targets contains duplicate entries"
            )

    # ---- (e) BUILD COMMAND SANITY ----

    build = spec.get("build", {})
    build_working_dir = build.get("working_directory", "")
    if build_working_dir and repo_root and repo_root.exists():
        working_abs = repo_root / build_working_dir
        if not working_abs.exists():
            errors.append(
                f"[spec] build.working_directory: '{build_working_dir}' "
                f"does not exist under repo_root"
            )

    outputs = build.get("outputs", {})
    executable = outputs.get("executable", "")
    if not executable:
        errors.append("[spec] build.outputs.executable: name is empty")

    commands = build.get("commands", {})
    build_cmd = commands.get("build", "")
    if build_cmd:
        # Find ${VAR} references in the build command
        var_refs = set(re.findall(r"\$\{(\w+)\}", build_cmd))
        if var_refs:
            variables = build.get("variables") or {}
            defined_vars = set(variables.keys())
            undefined = var_refs - defined_vars
            if undefined:
                warnings.append(
                    f"[spec] build.commands.build: undefined variable(s) "
                    f"referenced: {sorted(undefined)}"
                )

    # Prepend warnings with ⚠ marker so they're distinguishable
    return errors + [f"⚠ WARNING: {w}" for w in warnings]


# ---------------------------------------------------------------------------
# Manifest–Spec consistency checks (c)
# ---------------------------------------------------------------------------


def _load_manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    """Load all entries from manifest.jsonl."""
    entries: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return entries
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _check_manifest_spec_consistency(
    manifest_entries: list[dict[str, Any]],
) -> list[str]:
    """
    Cross-validate manifest entries against their linked spec files.

    Checks:
      - Every spec_file in manifest points to an existing file
      - kernel_name and parallel_api in manifest match those in spec
      - source_dir in manifest matches repo_root + source_path in spec
      - Every spec file in specs/ has a corresponding manifest entry
    """
    errors: list[str] = []
    manifest_spec_files: set[str] = set()

    for entry in manifest_entries:
        spec_rel = entry.get("spec_file", "")
        manifest_spec_files.add(spec_rel)
        spec_path = PROJECT_ROOT / spec_rel

        if not spec_path.exists():
            # Already reported by validate_manifest_entry; skip deeper checks
            continue

        try:
            spec = _load_json(spec_path)
        except json.JSONDecodeError:
            continue

        identity = spec.get("identity", {})

        # kernel_name must match
        m_kernel = entry.get("kernel_name", "")
        s_kernel = identity.get("kernel_name", "")
        if m_kernel and s_kernel and m_kernel != s_kernel:
            errors.append(
                f"[consistency] {spec_rel}: manifest kernel_name '{m_kernel}' "
                f"!= spec identity.kernel_name '{s_kernel}'"
            )

        # parallel_api must match
        m_api = entry.get("parallel_api", "")
        s_api = identity.get("parallel_api", "")
        if m_api and s_api and m_api != s_api:
            errors.append(
                f"[consistency] {spec_rel}: manifest parallel_api '{m_api}' "
                f"!= spec identity.parallel_api '{s_api}'"
            )

        # source_dir should correspond to repo_root / source_path
        m_source_dir = entry.get("source_dir", "")
        provenance = spec.get("provenance", {})
        s_repo_root = provenance.get("repo_root", "")
        s_source_path = provenance.get("source_path", "")
        if m_source_dir and s_repo_root:
            manifest_resolved = (PROJECT_ROOT / m_source_dir).resolve()
            spec_resolved = (PROJECT_ROOT / s_repo_root / s_source_path).resolve()
            if manifest_resolved != spec_resolved:
                errors.append(
                    f"[consistency] {spec_rel}: manifest source_dir resolves to "
                    f"'{manifest_resolved}' but spec repo_root+source_path "
                    f"resolves to '{spec_resolved}'"
                )

    # Every spec in specs/ should have a manifest entry
    specs_dir = PROJECT_ROOT / "specs"
    if specs_dir.exists():
        for json_file in sorted(specs_dir.glob("*.json")):
            rel = str(json_file.relative_to(PROJECT_ROOT))
            if rel not in manifest_spec_files:
                errors.append(
                    f"[consistency] {rel}: spec file has no corresponding "
                    f"manifest entry"
                )

    return errors


# ---------------------------------------------------------------------------
# Cross-kernel pairing check (d)
# ---------------------------------------------------------------------------


def _check_cross_kernel_pairing(
    manifest_entries: list[dict[str, Any]],
) -> tuple[list[str], dict[tuple[str, str], list[str]], int]:
    """
    For each unique (source_suite, kernel_name), collect all available APIs.
    Warn if a kernel has fewer than 2 APIs.
    Return the total number of valid translation pairs.

    Returns (warnings, kernel_api_map, total_pairs).
    Keys of kernel_api_map are (source_suite, kernel_name) tuples to prevent
    cross-suite collisions (e.g. rodinia/gaussian vs hecbench/gaussian).
    """
    warnings: list[str] = []
    kernel_apis: dict[tuple[str, str], list[str]] = defaultdict(list)

    for entry in manifest_entries:
        suite = entry.get("source_suite", "unknown")
        kname = entry.get("kernel_name", "")
        api = entry.get("parallel_api", "")
        if kname and api:
            kernel_apis[(suite, kname)].append(api)

    total_pairs = 0
    for (suite, kname), apis in sorted(kernel_apis.items()):
        n = len(apis)
        if n < 2:
            warnings.append(
                f"[pairing] {suite}/{kname} has only {n} API(s) "
                f"({apis}) — not useful for translation"
            )
        pairs = comb(n, 2) * 2  # ordered pairs (A→B and B→A)
        total_pairs += pairs

    return warnings, dict(kernel_apis), total_pairs


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _validate_manifest_file(manifest_path: Path) -> list[str]:
    """Validate every line in a manifest.jsonl file."""
    all_errors: list[str] = []
    if not manifest_path.exists():
        return [f"Manifest file not found: {manifest_path}"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {lineno}: invalid JSON — {e}")
                continue

            entry_errors = validate_manifest_entry(entry)
            for err in entry_errors:
                all_errors.append(f"Line {lineno}: {err}")

    if not all_errors:
        print(f"  ✓ manifest ({manifest_path.name}): all entries valid")
    return all_errors


def _validate_spec_file(spec_path: Path) -> list[str]:
    """Validate a single spec JSON file."""
    if not spec_path.exists():
        return [f"Spec file not found: {spec_path}"]

    try:
        spec = _load_json(spec_path)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {spec_path}: {e}"]

    errors = validate_spec(spec, spec_file_path=spec_path)
    non_warn = [e for e in errors if not e.startswith("⚠")]
    if not non_warn:
        print(f"  ✓ spec ({spec_path.name}): valid")
    return errors


def _validate_all() -> list[str]:
    """
    Validate the manifest and every spec file it references.
    Also validates any spec files in specs/ not yet in the manifest.
    Runs deep cross-cutting checks (consistency, pairing, build sanity).
    """
    all_errors: list[str] = []

    # 1. Validate manifest schema
    print("Validating manifest...")
    manifest_errors = _validate_manifest_file(DEFAULT_MANIFEST_PATH)
    all_errors.extend(manifest_errors)

    # 2. Collect spec files from manifest
    manifest_entries = _load_manifest_entries(DEFAULT_MANIFEST_PATH)
    referenced_specs: set[str] = {
        e["spec_file"] for e in manifest_entries if "spec_file" in e
    }

    # 3. Validate referenced specs
    print("\nValidating referenced specs...")
    for rel_path in sorted(referenced_specs):
        spec_path = PROJECT_ROOT / rel_path
        spec_errors = _validate_spec_file(spec_path)
        for err in spec_errors:
            all_errors.append(f"{rel_path}: {err}")

    # 4. Validate any spec files in specs/ not in manifest
    specs_dir = PROJECT_ROOT / "specs"
    if specs_dir.exists():
        print("\nChecking for unreferenced specs...")
        for json_file in sorted(specs_dir.glob("*.json")):
            rel = str(json_file.relative_to(PROJECT_ROOT))
            if rel not in referenced_specs:
                spec_errors = _validate_spec_file(json_file)
                for err in spec_errors:
                    all_errors.append(f"{rel} (unreferenced): {err}")

    # 5. (c) Manifest–Spec consistency
    print("\nRunning manifest–spec consistency checks...")
    consistency_errors = _check_manifest_spec_consistency(manifest_entries)
    if consistency_errors:
        all_errors.extend(consistency_errors)
    else:
        print("  ✓ manifest–spec consistency: all entries match")

    # 6. (d) Cross-kernel pairing check
    print("\nRunning cross-kernel pairing checks...")
    pairing_warnings, kernel_apis, total_pairs = _check_cross_kernel_pairing(
        manifest_entries
    )
    if pairing_warnings:
        for w in pairing_warnings:
            all_errors.append(f"⚠ WARNING: {w}")
    print(f"  Kernels: {len(kernel_apis)}")
    for (suite, kname), apis in sorted(kernel_apis.items()):
        print(f"    {suite}/{kname}: {', '.join(sorted(apis))}")
    print(f"  Total translation pairs: {total_pairs}")

    # 7. Oracle-strength summary + weak/unknown WARN (truncated to avoid noise)
    print("\nRunning oracle_strength audit...")
    counts = {"strong": 0, "medium": 0, "weak": 0, "unknown": 0, "missing": 0}
    flagged: list[str] = []  # weak / unknown / missing specs
    if specs_dir.exists():
        for json_file in sorted(specs_dir.glob("*.json")):
            try:
                spec = _load_json(json_file)
            except json.JSONDecodeError:
                continue
            strength = (spec.get("verification") or {}).get("oracle_strength")
            if strength in ("strong", "medium", "weak", "unknown"):
                counts[strength] += 1
                if strength in ("weak", "unknown"):
                    flagged.append(f"{json_file.name}: oracle_strength='{strength}'")
            else:
                counts["missing"] += 1
                flagged.append(f"{json_file.name}: oracle_strength not set")
    print(
        f"  oracle_strength: {counts['strong']} strong / {counts['medium']} medium / "
        f"{counts['weak']} weak / {counts['unknown']} unknown / {counts['missing']} missing"
    )
    if flagged:
        for entry in flagged[:5]:
            all_errors.append(f"⚠ WARNING: [oracle] {entry}")
        if len(flagged) > 5:
            all_errors.append(
                f"⚠ WARNING: [oracle] ... {len(flagged) - 5} more specs flagged "
                f"(total {len(flagged)}); S3 audit will populate oracle_strength"
            )

    return all_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParBench schema validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--manifest",
        type=str,
        metavar="PATH",
        help="Validate all entries in a manifest.jsonl file",
    )
    group.add_argument(
        "--spec",
        type=str,
        metavar="PATH",
        help="Validate a single Level 2 spec JSON file",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Validate manifest + all referenced and unreferenced specs",
    )

    args = parser.parse_args()

    errors: list[str] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        errors = _validate_manifest_file(manifest_path)

    elif args.spec:
        spec_path = Path(args.spec)
        if not spec_path.is_absolute():
            spec_path = PROJECT_ROOT / spec_path
        errors = _validate_spec_file(spec_path)

    elif args.all:
        errors = _validate_all()

    # Separate true errors from warnings
    true_errors = [e for e in errors if "⚠ WARNING:" not in e]
    warnings = [e for e in errors if "⚠ WARNING:" in e]

    # Report warnings
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):\n")
        for w in warnings:
            print(f"  {w}")

    # Report errors
    if true_errors:
        print(f"\n✗ {len(true_errors)} error(s) found:\n")
        for err in true_errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("\n✓ All validations passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
