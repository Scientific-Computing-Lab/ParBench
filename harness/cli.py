"""harness.cli — Command-line entry point for the ParBench harness.

Usage examples::

    python -m harness build  specs/hecbench-nn-cuda.json
    python -m harness run    specs/hecbench-nn-cuda.json --config correctness
    python -m harness verify specs/hecbench-nn-cuda.json
    python -m harness prompt specs/hecbench-nn-cuda.json
    python -m harness info   specs/hecbench-nn-cuda.json
    python -m harness pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from harness.builder import build_spec
from harness.models import SpecResult, Status
from harness.reporter import (
    format_json,
    format_result,
    print_prompt_payload,
    print_spec_info,
    print_translation_pairs,
)
from harness.runner import run_all_configurations, run_spec
from harness.spec_loader import (
    find_translation_pairs,
    get_prompt_payload,
    load_manifest,
    load_spec,
    resolve_paths,
)
from harness.verifier import extract_metrics, verify_run

log = logging.getLogger("harness")


# ---------------------------------------------------------------------------
# Resolve common paths
# ---------------------------------------------------------------------------


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root).resolve()


def _manifest_path(args: argparse.Namespace) -> Path:
    return _project_root(args) / args.manifest


def _spec_path(args: argparse.Namespace) -> Path:
    p = Path(args.spec_file)
    if not p.is_absolute():
        p = _project_root(args) / p
    return p


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    """Build a single spec."""
    project_root = _project_root(args)
    spec = load_spec(_spec_path(args))
    spec_id = spec.get("identity", {}).get("unique_id", "?")

    print(f"Building {spec_id} ...")
    result = build_spec(spec, project_root, verbose=args.verbose)

    sr = SpecResult(spec_id=spec_id, build=result)
    print(format_result(sr))

    if args.verbose:
        if result.stdout:
            print(f"\n[stdout]\n{result.stdout}")
        if result.stderr:
            print(f"\n[stderr]\n{result.stderr}")

    return 0 if result.status == Status.PASS else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run (but do not build) a single spec with one configuration."""
    project_root = _project_root(args)
    spec = load_spec(_spec_path(args))
    spec_id = spec.get("identity", {}).get("unique_id", "?")

    print(f"Running {spec_id} [{args.config}] ...")
    result = run_spec(
        spec, project_root, configuration=args.config, verbose=args.verbose
    )

    sr = SpecResult(
        spec_id=spec_id,
        build=_dummy_build(),
        runs={args.config: result},
    )
    print(format_result(sr))

    if args.verbose:
        if result.stdout:
            print(f"\n[stdout]\n{result.stdout}")
        if result.stderr:
            print(f"\n[stderr]\n{result.stderr}")

    return 0 if result.status == Status.PASS else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Full pipeline: build → run → verify."""
    project_root = _project_root(args)
    spec = load_spec(_spec_path(args))
    spec_id = spec.get("identity", {}).get("unique_id", "?")

    # Build
    print(f"Building {spec_id} ...")
    build_result = build_spec(spec, project_root, verbose=args.verbose)
    if build_result.status != Status.PASS:
        sr = SpecResult(spec_id=spec_id, build=build_result)
        print(format_result(sr))
        return 1

    # Run
    config = args.config
    print(f"Running {spec_id} [{config}] ...")
    run_result = run_spec(
        spec, project_root, configuration=config, verbose=args.verbose
    )

    # Verify
    print(f"Verifying {spec_id} ...")
    resolved = resolve_paths(spec, project_root)["_resolved"]
    ver_result = verify_run(spec, run_result, working_dir=resolved["working_dir"])

    # Metrics
    metrics = extract_metrics(spec, run_result)

    sr = SpecResult(
        spec_id=spec_id,
        build=build_result,
        runs={config: run_result},
        verification=ver_result,
        metrics=metrics,
    )
    print(format_result(sr))

    if args.json:
        print(json.dumps(format_json(sr), indent=2))

    if args.verbose:
        if run_result.stdout:
            print(f"\n[stdout]\n{run_result.stdout}")
        if run_result.stderr:
            print(f"\n[stderr]\n{run_result.stderr}")

    return 0 if ver_result.status == Status.PASS else 1


def cmd_prompt(args: argparse.Namespace) -> int:
    """Print the prompt payload (what the LLM would see)."""
    project_root = _project_root(args)
    spec = load_spec(_spec_path(args))
    identity = spec.get("identity", {})

    payload = get_prompt_payload(spec, project_root, augment_level=getattr(args, 'augment_level', 0))
    print_prompt_payload(
        spec_id=identity.get("unique_id", "?"),
        api=identity.get("parallel_api", "?"),
        payload=payload,
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print spec info without building/running."""
    spec = load_spec(_spec_path(args))
    print_spec_info(spec)
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    """List all valid translation pairs from the manifest."""
    manifest = load_manifest(_manifest_path(args))
    pairs = find_translation_pairs(manifest)
    print_translation_pairs(pairs)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_build() -> Any:
    """Placeholder BuildResult when build is skipped (e.g. `run` command)."""
    from harness.models import BuildResult

    return BuildResult(
        status=Status.SKIP,
        duration_seconds=0.0,
        stdout="",
        stderr="(build skipped)",
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="harness",
        description="ParBench harness — build, run, and verify kernels from specs",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to parbench/ project root (default: cwd)",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.jsonl",
        help="Manifest filename relative to project root (default: manifest.jsonl)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (logs stdout/stderr of subprocesses)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print machine-readable JSON output",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- build --
    p_build = sub.add_parser("build", help="Build a kernel from its spec")
    p_build.add_argument("spec_file", help="Path to spec JSON file")
    p_build.set_defaults(func=cmd_build)

    # -- run --
    p_run = sub.add_parser("run", help="Run a compiled kernel")
    p_run.add_argument("spec_file", help="Path to spec JSON file")
    p_run.add_argument(
        "--config",
        default="correctness",
        help="Input configuration name (default: correctness)",
    )
    p_run.set_defaults(func=cmd_run)

    # -- verify --
    p_verify = sub.add_parser(
        "verify",
        help="Full pipeline: build → run → verify",
    )
    p_verify.add_argument("spec_file", help="Path to spec JSON file")
    p_verify.add_argument(
        "--config",
        default="correctness",
        help="Input configuration name (default: correctness)",
    )
    p_verify.set_defaults(func=cmd_verify)

    # -- prompt --
    p_prompt = sub.add_parser(
        "prompt",
        help="Print the prompt payload (what the LLM sees)",
    )
    p_prompt.add_argument("spec_file", help="Path to spec JSON file")
    p_prompt.add_argument(
        "--augment_level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help="Apply semantics-preserving C/C++ augmentation (0=none, 1=light, 4=aggressive)",
    )
    p_prompt.set_defaults(func=cmd_prompt)

    # -- info --
    p_info = sub.add_parser("info", help="Print spec summary (no build/run)")
    p_info.add_argument("spec_file", help="Path to spec JSON file")
    p_info.set_defaults(func=cmd_info)

    # -- pairs --
    p_pairs = sub.add_parser(
        "pairs",
        help="List all valid translation pairs from manifest",
    )
    p_pairs.set_defaults(func=cmd_pairs)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s: %(message)s",
    )

    try:
        rc = args.func(args)
    except Exception as exc:
        log.error("Unhandled error: %s", exc, exc_info=args.verbose)
        print(f"ERROR: {exc}", file=sys.stderr)
        rc = 2

    sys.exit(rc)


if __name__ == "__main__":
    main()
