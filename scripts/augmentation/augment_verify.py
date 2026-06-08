#!/usr/bin/env python3
"""
scripts/augment_verify.py — Augment → Build → Run → Verify pipeline.

Applies semantics-preserving AST transforms to a kernel's prompt_payload
files, builds the augmented code in a sibling temp directory, runs it, and
verifies the output matches the spec's verification strategy.

The temp directory is created as a sibling of the original working directory
so that relative paths to data files (e.g. ../../data/bfs/graph1MW_6.txt)
remain valid.

Usage:
    python3 scripts/augment_verify.py specs/rodinia-bfs-cuda.json
    python3 scripts/augment_verify.py specs/rodinia-bfs-cuda.json \\
        --augment_level 2 --seed 42 -v
    python3 scripts/augment_verify.py specs/rodinia-bfs-cuda.json \\
        --augment_level 4 --seed 42 --keep-temp --json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from harness.builder import build_spec
from harness.models import Status
from harness.runner import run_spec
from harness.spec_loader import load_spec, resolve_paths
from harness.verifier import extract_metrics, verify_run

# File extensions that the augmentation engine can process
AUGMENTABLE_SUFFIXES = {".c", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".cl", ".cc", ".dp.cpp"}


def _build_aug_config(augment_level: int):
    """Return (ci.Index, AugmentationConfig) for the given level, or (None, None)."""
    if augment_level <= 0:
        return None, None
    import clang.cindex as ci
    from c_augmentation.augment_dataset import (
        AugmentationConfig,
        ArithmeticTransform,
        ChangeFunctionNames,
        ChangeNames,
        PointerArithmeticToArrayIndex,
        SwapCondition,
        TypedefExpansion,
    )
    index = ci.Index.create()
    config = AugmentationConfig(
        level=augment_level,
        transforms=[
            ArithmeticTransform(level=augment_level),
            SwapCondition(level=augment_level),
            PointerArithmeticToArrayIndex(level=augment_level),
            TypedefExpansion(level=augment_level),
            ChangeNames(level=augment_level),
            ChangeFunctionNames(level=augment_level),
        ],
    )
    return index, config


def _augment_payload(
    spec: dict[str, Any],
    project_root: Path,
    augment_level: int,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Read prompt_payload files, apply augmentation, return contents + transforms.

    Returns
    -------
    contents : {filename: file_content_string}
    transforms : {filename: [transform_name, ...]}
    """
    resolved = resolve_paths(spec, project_root)["_resolved"]
    source_dir: Path = resolved["source_dir"]
    prompt_files: list[str] = spec.get("files", {}).get("prompt_payload", [])

    ci_index, aug_config = _build_aug_config(augment_level)

    contents: dict[str, str] = {}
    transforms: dict[str, list[str]] = {}

    for fname in prompt_files:
        fpath = source_dir / fname
        if not fpath.exists():
            contents[fname] = f"<FILE NOT FOUND: {fpath}>"
            transforms[fname] = []
            continue

        text = fpath.read_text(encoding="utf-8", errors="replace")

        if ci_index is not None and aug_config is not None and Path(fname).suffix in AUGMENTABLE_SUFFIXES:
            from c_augmentation.augment_dataset import augment_code
            augmented, applied = augment_code(text, aug_config, ci_index, filename=str(fpath))
            contents[fname] = augmented
            transforms[fname] = applied
        else:
            contents[fname] = text
            transforms[fname] = []

    return contents, transforms


def augment_verify(
    spec_path: Path,
    project_root: Path,
    augment_level: int = 1,
    seed: int | None = None,
    config: str = "correctness",
    keep_temp: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Full augment -> build -> run -> verify pipeline for one spec.

    Returns a result dict with keys:
        spec_id, augment_level, seed, config,
        transforms_applied, transforms_summary, files_changed,
        build_status, run_status, verify_status, overall_status,
        verify_details, wall_time_seconds, exit_code,
        tempdir (only if keep_temp), details
    """
    if seed is not None:
        random.seed(seed)

    spec = load_spec(spec_path)
    spec_id: str = spec.get("identity", {}).get("unique_id", spec_path.stem)

    # --- 1. Resolve original working directory ---
    original_resolved = resolve_paths(spec, project_root)["_resolved"]
    original_working_dir: Path = original_resolved["working_dir"]

    if not original_working_dir.exists():
        return {
            "spec_id": spec_id,
            "augment_level": augment_level,
            "seed": seed,
            "overall_status": "ERROR",
            "details": f"Working directory not found: {original_working_dir}",
        }

    # --- 2. Create sibling temp directory ---
    seed_tag = str(seed) if seed is not None else "noseed"
    tempdir = original_working_dir.parent / f"{original_working_dir.name}_aug_{seed_tag}"
    if tempdir.exists():
        shutil.rmtree(tempdir)
    shutil.copytree(str(original_working_dir), str(tempdir))

    try:
        # --- 3. Augment prompt_payload files ---
        augmented_contents, transforms_per_file = _augment_payload(
            spec, project_root, augment_level
        )

        # --- 4. Write augmented files into tempdir; track which files changed ---
        files_changed: list[str] = []
        for fname, content in augmented_contents.items():
            dest = tempdir / fname
            orig_path = original_working_dir / fname
            if orig_path.exists():
                original_text = orig_path.read_text(encoding="utf-8", errors="replace")
                if content != original_text:
                    files_changed.append(fname)
            dest.write_text(content, encoding="utf-8")

        # --- 5. Patch spec in memory so build/run use tempdir ---
        # Python's Path(repo_root) / Path("/absolute") == Path("/absolute"),
        # so setting working_directory to the absolute tempdir string makes
        # resolve_paths() return tempdir as working_dir for both build and run.
        patched_spec = copy.deepcopy(spec)
        patched_spec["build"]["working_directory"] = str(tempdir)

        # --- 6. Build ---
        if verbose:
            print(f"    Building in {tempdir} ...")
        build_result = build_spec(patched_spec, project_root, verbose=verbose)

        if build_result.status != Status.PASS:
            return {
                "spec_id": spec_id,
                "augment_level": augment_level,
                "seed": seed,
                "config": config,
                "transforms_applied": transforms_per_file,
                "transforms_summary": _flatten_transforms(transforms_per_file),
                "files_changed": files_changed,
                "build_status": build_result.status.value,
                "run_status": "skipped",
                "verify_status": "skipped",
                "overall_status": "BUILD_FAIL",
                "verify_details": "",
                "wall_time_seconds": None,
                "exit_code": None,
                "tempdir": str(tempdir) if keep_temp else None,
                "details": (build_result.stderr or "")[-600:],
            }

        # --- 7. Run ---
        if verbose:
            print(f"    Running [{config}] ...")
        run_result = run_spec(patched_spec, project_root, configuration=config, verbose=verbose)

        # --- 8. Verify + metrics ---
        ver_result = verify_run(patched_spec, run_result, working_dir=tempdir)
        _metrics = extract_metrics(patched_spec, run_result)

        overall = "PASS" if ver_result.status == Status.PASS else ver_result.status.value.upper()

        return {
            "spec_id": spec_id,
            "augment_level": augment_level,
            "seed": seed,
            "config": config,
            "transforms_applied": transforms_per_file,
            "transforms_summary": _flatten_transforms(transforms_per_file),
            "files_changed": files_changed,
            "build_status": build_result.status.value,
            "run_status": run_result.status.value,
            "verify_status": ver_result.status.value,
            "overall_status": overall,
            "verify_details": ver_result.details,
            "wall_time_seconds": run_result.duration_seconds,
            "exit_code": run_result.exit_code,
            "tempdir": str(tempdir) if keep_temp else None,
            "details": "",
        }

    finally:
        if not keep_temp and tempdir.exists():
            shutil.rmtree(tempdir)


def _flatten_transforms(transforms_per_file: dict[str, list[str]]) -> list[str]:
    return sorted({t for tlist in transforms_per_file.values() for t in tlist})


def _status_symbol(overall: str) -> str:
    return {"PASS": "✓", "BUILD_FAIL": "✗", "FAIL": "✗", "ERROR": "!", "SKIP": "-"}.get(overall, "?")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment a kernel's source, build it, and verify semantic equivalence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spec_file", type=Path, help="Path to spec JSON (absolute or relative to project root)")
    parser.add_argument(
        "--augment_level", "-l",
        type=int, choices=[0, 1, 2, 3, 4], default=1,
        help="Augmentation aggressiveness (0=none, 1=light, 4=aggressive). Default: 1",
    )
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--config", default="correctness", help="Run configuration name (default: correctness)")
    parser.add_argument("--project-root", default=None, help="Path to parbench/ root (default: auto-detected)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary build directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show subprocess stdout/stderr")
    parser.add_argument("--json", action="store_true", help="Also print machine-readable JSON result")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    spec_path = Path(args.spec_file)
    if not spec_path.is_absolute():
        spec_path = (project_root / spec_path).resolve()

    spec_id = spec_path.stem
    print(f"[{spec_id}]  level={args.augment_level}  seed={args.seed}  config={args.config}")

    result = augment_verify(
        spec_path=spec_path,
        project_root=project_root,
        augment_level=args.augment_level,
        seed=args.seed,
        config=args.config,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
    )

    overall = result.get("overall_status", "UNKNOWN")
    sym = _status_symbol(overall)
    build_s = result.get("build_status", "?").upper()
    run_s = result.get("run_status", "?").upper()
    ver_s = result.get("verify_status", "?").upper()
    changed = result.get("files_changed", [])
    transforms = result.get("transforms_summary", [])
    wall = result.get("wall_time_seconds")

    print(f"  {sym} BUILD:{build_s}  RUN:{run_s}  VERIFY:{ver_s}  [{overall}]")
    if changed:
        print(f"  Files changed : {', '.join(changed)}")
    if transforms:
        print(f"  Transforms    : {', '.join(transforms)}")
    if wall is not None:
        print(f"  Wall time     : {wall:.3f}s")
    if result.get("verify_details"):
        print(f"  Verify detail : {result['verify_details']}")
    if result.get("details"):
        print(f"  Error detail  : {result['details'][:300]}")
    if args.keep_temp and result.get("tempdir"):
        print(f"  Temp dir      : {result['tempdir']}")

    if args.json:
        print(json.dumps(result, indent=2))

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
