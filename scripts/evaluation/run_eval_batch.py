#!/usr/bin/env python3
"""scripts/evaluation/run_eval_batch.py — Batch LLM evaluation runner.

Translates source kernels to target API using one or more LLMs, writes
incremental per-task JSON results, and generates a Markdown summary.

Usage:
    python3 scripts/evaluation/run_eval_batch.py \\
      --suite rodinia --direction cuda-to-omp \\
      --models claude-sonnet-4-20250514 azure-gpt-5.4 \\
      --project-root . \\
      --resume -v

    # Test resume behaviour (all 5 pilot results already exist):
    python3 scripts/evaluation/run_eval_batch.py \\
      --kernels bfs hotspot backprop nw srad \\
      --direction cuda-to-omp \\
      --models claude-sonnet-4-20250514 \\
      --project-root . -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.spec_loader import load_manifest  # noqa: E402
from scripts.evaluation.llm_evaluate import evaluate_translation  # noqa: E402


def _extract_kernel(r: dict) -> str:
    """Extract kernel name from result dict's source_spec field.

    source_spec has format "{suite}-{kernel}-{api}" (e.g. "rodinia-nw-cuda").
    Returns the middle portion (e.g. "nw").
    """
    spec_id = r.get("source_spec", "?")
    if spec_id == "?":
        return "?"
    parts = spec_id.split("-")
    if len(parts) < 3:
        return spec_id
    return "-".join(parts[1:-1])


# --------------------------------------------------------------------------- #
# Task building                                                                #
# --------------------------------------------------------------------------- #

def _build_tasks(
    manifest_path: Path,
    project_root: Path,
    direction: str,
    suite: str | None,
    kernels: list[str] | None,
    models: list[str],
    augment_levels: list[int] | None = None,
    num_samples: int = 1,
    excluded_specs: set[str] | None = None,
) -> list[dict]:
    """Return a list of task dicts: {src_spec, tgt_spec, kernel, model, augment_level, sample_id}."""
    src_api, tgt_api = direction.split("-to-")
    levels = augment_levels or [0]

    manifest = load_manifest(str(manifest_path))

    # Index by (suite, kernel, api) → spec_file
    index: dict[tuple[str, str, str], str] = {}
    for entry in manifest:
        key = (entry["source_suite"], entry["kernel_name"], entry["parallel_api"])
        index[key] = entry["spec_file"]

    tasks: list[dict] = []
    seen_kernels: set[str] = set()

    for entry in manifest:
        if suite and entry["source_suite"] != suite:
            continue
        if entry["parallel_api"] != src_api:
            continue
        kernel = entry["kernel_name"]
        if kernels and kernel not in kernels:
            continue
        tgt_suite = entry["source_suite"]
        tgt_key = (tgt_suite, kernel, tgt_api)
        if tgt_key not in index:
            continue  # no matching target spec

        src_spec_path = project_root / entry["spec_file"]
        tgt_spec_path = project_root / index[tgt_key]

        # Skip pairs where either spec file has been deleted (phantom specs)
        if not src_spec_path.exists():
            print(f"  ⚠ SKIP {kernel}: source spec not found: {src_spec_path}", flush=True)
            continue
        if not tgt_spec_path.exists():
            print(f"  ⚠ SKIP {kernel}: target spec not found: {tgt_spec_path}", flush=True)
            continue

        if excluded_specs:
            if src_spec_path.stem in excluded_specs or tgt_spec_path.stem in excluded_specs:
                continue

        for model in models:
            for level in levels:
                for sid in range(num_samples):
                    tasks.append({
                        "kernel": kernel,
                        "src_spec": src_spec_path,
                        "tgt_spec": tgt_spec_path,
                        "model": model,
                        "augment_level": level,
                        "sample_id": sid,
                        "num_samples": num_samples,
                        "src_id": src_spec_path.stem,
                        "tgt_id": tgt_spec_path.stem,
                    })

        seen_kernels.add(kernel)

    return tasks


def _build_tasks_from_task_list(
    task_list_path: Path,
    project_root: Path,
    direction: str,
    models: list[str],
    augment_levels: list[int],
    num_samples: int,
    manifest_path: Path,
    excluded_specs: set[str] | None = None,
) -> list[dict]:
    """Build task dict list from a passer-JSON file (D-25, plan 02-06).

    Each entry in the passer JSON looks like:
        {"source_spec": "rodinia-bfs-cuda", "target_spec": "rodinia-bfs-omp", "augment_level": 0}

    The `augment_level` field in the passer file is a placeholder (D-19) — we override
    via the supplied `augment_levels` arg. Cross-product with `models` × `augment_levels`
    × `num_samples`, mirroring the existing manifest-enumeration branch's task dict shape.

    Entries whose source_spec or target_spec are absent from the manifest emit a stderr
    warning and are skipped (D-26 case 2 — not crash).
    """
    src_api, tgt_api = direction.split("-to-")

    try:
        data = json.loads(task_list_path.read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"--task-list file not found: {task_list_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"--task-list file is not valid JSON: {task_list_path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(
            f"--task-list file must contain a JSON list of task dicts, got {type(data).__name__}: {task_list_path}"
        )

    manifest = load_manifest(str(manifest_path))
    # Index by unique_id (source_suite-kernel_name-parallel_api) → entry
    manifest_by_id: dict[str, dict] = {}
    for entry in manifest:
        uid = f"{entry['source_suite']}-{entry['kernel_name']}-{entry['parallel_api']}"
        manifest_by_id[uid] = entry

    tasks: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict) or "source_spec" not in entry or "target_spec" not in entry:
            print(
                f"warning: passer entry skipped — missing source_spec/target_spec: {entry!r}",
                file=sys.stderr,
            )
            continue
        src = entry["source_spec"]
        tgt = entry["target_spec"]
        if src not in manifest_by_id or tgt not in manifest_by_id:
            print(
                f"warning: passer entry skipped — not in manifest: "
                f"source_spec={src!r} target_spec={tgt!r}",
                file=sys.stderr,
            )
            continue

        src_entry = manifest_by_id[src]
        tgt_entry = manifest_by_id[tgt]
        # Sanity-check direction matches the passer entry.
        if src_entry["parallel_api"] != src_api or tgt_entry["parallel_api"] != tgt_api:
            print(
                f"warning: passer entry skipped — APIs do not match --direction {direction}: "
                f"source_spec={src!r}({src_entry['parallel_api']}) "
                f"target_spec={tgt!r}({tgt_entry['parallel_api']})",
                file=sys.stderr,
            )
            continue

        src_spec_path = project_root / src_entry["spec_file"]
        tgt_spec_path = project_root / tgt_entry["spec_file"]
        if not src_spec_path.exists() or not tgt_spec_path.exists():
            print(
                f"warning: passer entry skipped — spec file missing on disk: "
                f"{src_spec_path if not src_spec_path.exists() else tgt_spec_path}",
                file=sys.stderr,
            )
            continue

        if excluded_specs:
            if src_spec_path.stem in excluded_specs or tgt_spec_path.stem in excluded_specs:
                continue

        kernel = src_entry["kernel_name"]
        for model in models:
            for level in augment_levels:
                for sid in range(num_samples):
                    tasks.append({
                        "kernel": kernel,
                        "src_spec": src_spec_path,
                        "tgt_spec": tgt_spec_path,
                        "model": model,
                        "augment_level": level,
                        "sample_id": sid,
                        "num_samples": num_samples,
                        "src_id": src_spec_path.stem,
                        "tgt_id": tgt_spec_path.stem,
                    })

    return tasks


def _result_path(project_root: Path, model: str, src_id: str, tgt_id: str, augment_level: int = 0, sample_id: int = 0, num_samples: int = 1) -> Path:
    safe_model = model.replace("/", "_")
    level_tag = f"-L{augment_level}" if augment_level > 0 else ""
    sample_tag = f"-s{sample_id}" if num_samples > 1 else ""
    return project_root / "results" / "evaluation" / safe_model / f"{src_id}-to-{tgt_id}{level_tag}{sample_tag}.json"


# --------------------------------------------------------------------------- #
# Main batch loop                                                              #
# --------------------------------------------------------------------------- #

def run_batch(
    tasks: list[dict],
    project_root: Path,
    resume: bool,
    max_failures: int,
    use_cpu_timing: bool,
    verbose: bool,
    max_retries: int = 1,
    temperature: float = 0.0,
    thinking: str = "on",
) -> list[dict]:
    """Execute all tasks sequentially; return list of result dicts."""
    results: list[dict] = []
    total = len(tasks)
    consecutive_failures = 0

    for i, task in enumerate(tasks, 1):
        src_id = task["src_id"]
        tgt_id = task["tgt_id"]
        model = task["model"]
        augment_level = task.get("augment_level", 0)
        sample_id = task.get("sample_id", 0)
        num_samples = task.get("num_samples", 1)
        result_file = _result_path(project_root, model, src_id, tgt_id, augment_level, sample_id, num_samples)

        level_tag = f" L{augment_level}" if augment_level > 0 else ""
        sample_tag = f" s{sample_id}" if num_samples > 1 else ""
        prefix = f"[{i:3d}/{total}] {src_id} → {tgt_id} [{model}]{level_tag}{sample_tag}"

        if resume and result_file.exists():
            try:
                existing = json.loads(result_file.read_text())
            except (json.JSONDecodeError, ValueError):
                print(f"{prefix}  ↺ RETRY (corrupted file)", flush=True)
            else:
                existing_status = existing.get("overall_status", "")
                # Retry ERROR and EXTRACTION_FAIL results — these are transient
                # failures (API errors, partial code extraction) worth re-running.
                # All other statuses (PASS, BUILD_FAIL, RUN_FAIL, VERIFY_FAIL, SKIP)
                # are deterministic outcomes that won't change on retry.
                if existing_status not in ("ERROR", "EXTRACTION_FAIL"):
                    print(f"{prefix}  ⟳ SKIP ({existing_status})", flush=True)
                    results.append(existing)
                    consecutive_failures = 0
                    continue
                print(f"{prefix}  ↺ RETRY ({existing_status})", flush=True)

        print(f"{prefix}", flush=True)
        t0 = time.time()

        try:
            result = evaluate_translation(
                source_path=task["src_spec"],
                target_path=task["tgt_spec"],
                model=model,
                project_root=project_root,
                augment_level=augment_level,
                use_cpu_timing=use_cpu_timing,
                max_retries=max_retries,
                verbose=verbose,
                temperature=temperature,
                sample_id=sample_id,
                save_to_disk=False,  # batch runner owns file I/O (avoids dual-write)
                thinking=thinking,
                num_samples=num_samples,
            )
        except FileNotFoundError as exc:
            elapsed = time.time() - t0
            print(f"  ✗ SKIP (spec file missing: {exc})", flush=True)
            results.append({
                "overall_status": "SKIP",
                "source_spec": task["src_id"],
                "target_spec": task["tgt_id"],
                "model": model,
                "error_message": f"Spec file not found: {exc}",
            })
            continue

        elapsed = time.time() - t0
        status = result.get("overall_status", "?")
        sym = "✓" if status == "PASS" else ("!" if status == "ERROR" else "✗")
        print(f"  {sym} {status:<12}  {elapsed:.1f}s", flush=True)

        # Write result incrementally — warn if overwriting an existing PASS
        result_file.parent.mkdir(parents=True, exist_ok=True)
        if result_file.exists():
            existing_result = json.loads(result_file.read_text())
            if existing_result.get("overall_status") == "PASS":
                print(
                    f"  ⚠ WARNING: overwriting existing PASS result for "
                    f"{src_id} → {tgt_id} [{model}]. "
                    f"Remove --no-resume flag to skip completed tasks.",
                    flush=True,
                )
        result_file.write_text(json.dumps(result, indent=2))

        results.append(result)

        if status not in ("PASS", "SKIP"):
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        if max_failures > 0 and consecutive_failures >= max_failures:
            print(
                f"\nStopping after {consecutive_failures} consecutive failures "
                f"(--max-failures {max_failures}).",
                flush=True,
            )
            break

    return results


# --------------------------------------------------------------------------- #
# Report generation                                                            #
# --------------------------------------------------------------------------- #

def _generate_markdown(results: list[dict], models: list[str], title: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d")

    # Group by model → (kernel, level)
    by_model: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    all_levels: set[int] = set()
    for r in results:
        m = r.get("model", "?")
        k = r.get("kernel") or _extract_kernel(r)
        level = r.get("augment_level", 0)
        by_model[m][(k, level)] = r
        all_levels.add(level)
    multi_level = len(all_levels) > 1

    lines = [
        f"# {title}",
        f"**Date:** {now}  |  **Tasks:** {len(results)}",
        "",
    ]

    for model in models:
        model_results = list(by_model.get(model, {}).values())
        if not model_results:
            continue

        n = len(model_results)
        passes = sum(1 for r in model_results if r.get("overall_status") == "PASS")
        fails = sum(1 for r in model_results
                    if r.get("overall_status") not in {"PASS", "ERROR", "SKIP", None})
        errors = sum(1 for r in model_results if r.get("overall_status") == "ERROR")
        skips = sum(1 for r in model_results if r.get("overall_status") == "SKIP")
        rate = f"{100 * passes // n}%" if n else "0%"

        if multi_level:
            lines += [
                f"## {model}",
                f"**{passes}/{n} PASS ({rate})** | FAILURES={fails} | ERROR={errors} | SKIP={skips}",
                "",
                "| Kernel | Level | Status | Speedup | Timing Method | Tokens |",
                "|--------|-------|--------|---------|---------------|--------|",
            ]
        else:
            lines += [
                f"## {model}",
                f"**{passes}/{n} PASS ({rate})** | FAILURES={fails} | ERROR={errors} | SKIP={skips}",
                "",
                "| Kernel | Status | Speedup | Timing Method | Tokens |",
                "|--------|--------|---------|---------------|--------|",
            ]

        for (kernel, level), r in sorted(by_model.get(model, {}).items()):
            status = r.get("overall_status", "?")
            sym = "✓" if status == "PASS" else ("!" if status == "ERROR" else "✗")
            speedup = r.get("speedup_ratio")
            speedup_str = f"{speedup:.3f}×" if speedup else "—"
            timing = r.get("timing_method", "—")
            pt = r.get("prompt_tokens") or 0
            ct = r.get("completion_tokens") or 0
            tokens = f"{pt + ct}" if (pt or ct) else "—"
            if multi_level:
                lines.append(
                    f"| {kernel} | L{level} | {sym} {status} | {speedup_str} | {timing} | {tokens} |"
                )
            else:
                lines.append(
                    f"| {kernel} | {sym} {status} | {speedup_str} | {timing} | {tokens} |"
                )

        lines.append("")

    # Summary across models
    lines += ["## Cross-Model Summary", ""]
    all_keys: set[tuple[str, int]] = set()
    for r in results:
        k = r.get("kernel") or _extract_kernel(r)
        level = r.get("augment_level", 0)
        all_keys.add((k, level))

    col_header = " | ".join(models)
    if multi_level:
        lines.append(f"| Kernel | Level | {col_header} |")
        lines.append("|--------|-------|" + "|".join(["---"] * len(models)) + "|")
    else:
        lines.append(f"| Kernel | {col_header} |")
        lines.append("|--------|" + "|".join(["---"] * len(models)) + "|")

    for (kernel, level) in sorted(all_keys):
        cells = []
        for model in models:
            r = by_model.get(model, {}).get((kernel, level))
            if r is None:
                cells.append("—")
            else:
                s = r.get("overall_status", "?")
                sym = "✓" if s == "PASS" else ("!" if s == "ERROR" else "✗")
                cells.append(f"{sym} {s}")
        if multi_level:
            lines.append(f"| {kernel} | L{level} | {' | '.join(cells)} |")
        else:
            lines.append(f"| {kernel} | {' | '.join(cells)} |")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Extracted for testability (plan 02-06)."""
    parser = argparse.ArgumentParser(
        description="Batch LLM evaluation runner for ParBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # D-23 (lifted 2026-04-17, plan 02-10): --suite and --kernels COMPOSE
    # (filter intersection). Only --task-list is mutex with the other two;
    # enforced at runtime via parser.error() after parse_args(). The original
    # 02-06 D-23 LOCK was over-scoped — an argparse-native 3-way mutex blocked
    # the documented `--suite rodinia --kernels bfs` invocation and caused
    # 12/12 Qwen real-API tests to fail at argparse on 2026-04-17.
    parser.add_argument(
        "--suite",
        default=None,
        help="Filter to a single benchmark suite (e.g. 'rodinia'). Composes with --kernels; mutex with --task-list.",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=None,
        metavar="KERNEL",
        help="Restrict to specific kernel names (e.g. bfs hotspot). Composes with --suite; mutex with --task-list.",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=None,
        dest="task_list",
        help=(
            "Path to a passer-JSON list (e.g. from derive_l0_passers.py) — "
            "runs only the listed cells, bypassing manifest enumeration. "
            "Mutex with --suite and --kernels."
        ),
    )
    parser.add_argument(
        "--direction",
        required=True,
        metavar="SRC-to-TGT",
        help="Translation direction, e.g. 'cuda-to-omp'.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="Model IDs to evaluate (e.g. claude-sonnet-4-20250514 azure-gpt-5.4).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Absolute path to the parbench project root.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip tasks whose result file already exists (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-run all tasks even if result files exist.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N consecutive failures (0 = never stop).",
    )
    parser.add_argument(
        "--use-cpu-timing",
        action="store_true",
        default=False,
        help="Measure CPU time via /usr/bin/time -v (Linux/GNU time required).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PREFIX",
        help=(
            "Output prefix for the batch summary JSON and Markdown report. "
            "Defaults to results/evaluation/batch_{direction}_{timestamp}."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title for the Markdown report (auto-generated if omitted).",
    )
    parser.add_argument(
        "--augment-levels",
        nargs="+",
        type=int,
        default=[0],
        metavar="LEVEL",
        help=(
            "Augmentation levels to test (0-4). Each level generates separate tasks "
            "with result files tagged -L1, -L2, etc. (L0 has no tag). Default: 0."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        metavar="N",
        help="Max LLM attempts per task with error feedback (1=zero-shot, default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0=greedy, 0.5+ for pass@k multi-sampling). Default: 0.0",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent samples per task for pass@k (default: 1).",
    )
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="on",
        help=(
            "Enable LLM reasoning/thinking mode for capable models (default: on). "
            "Passed through to evaluate_translation(); no-op for models with "
            "supports_thinking=False in MODEL_REGISTRY."
        ),
    )
    parser.add_argument(
        "--excluded-specs",
        nargs="*",
        default=[],
        metavar="SPEC_ID",
        help="Spec IDs to exclude as source or target (e.g. KNOWN_FAIL specs).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # D-23 runtime mutex (plan 02-10, 2026-04-17): --task-list is mutex with
    # --suite and --kernels. parser.error() exits with code 2 — matches the
    # existing SystemExit(2) contract asserted by D-26 case 3 tests.
    if args.task_list and (args.suite or args.kernels):
        parser.error("--task-list is mutually exclusive with --suite and --kernels")

    project_root = args.project_root.resolve()
    excluded_specs = set(args.excluded_specs) if args.excluded_specs else set()
    manifest_path = project_root / "manifest.jsonl"

    # Validate direction format
    if "-to-" not in args.direction:
        parser.error("--direction must be in the form SRC-to-TGT (e.g. cuda-to-omp)")

    # Validate augment levels
    for lvl in args.augment_levels:
        if lvl not in range(5):
            parser.error(f"--augment-levels values must be 0-4, got {lvl}")

    # Build task list — branch on --task-list (D-25, plan 02-06).
    if args.task_list is not None:
        tasks = _build_tasks_from_task_list(
            task_list_path=args.task_list,
            project_root=project_root,
            direction=args.direction,
            models=args.models,
            augment_levels=args.augment_levels,
            num_samples=args.num_samples,
            manifest_path=manifest_path,
            excluded_specs=excluded_specs,
        )
    else:
        tasks = _build_tasks(
            manifest_path=manifest_path,
            project_root=project_root,
            direction=args.direction,
            suite=args.suite,
            kernels=args.kernels,
            models=args.models,
            augment_levels=args.augment_levels,
            num_samples=args.num_samples,
            excluded_specs=excluded_specs,
        )

    if not tasks:
        print("No tasks found. Check --suite/--kernels/--task-list/--direction arguments.", flush=True)
        sys.exit(1)

    print(
        f"Batch: {len(tasks)} task(s)  "
        f"direction={args.direction}  "
        f"models={args.models}  "
        f"augment_levels={args.augment_levels}  "
        f"temperature={args.temperature}  "
        f"num_samples={args.num_samples}  "
        f"thinking={args.thinking}  "
        f"resume={args.resume}",
        flush=True,
    )

    # Run
    t_start = time.time()
    results = run_batch(
        tasks=tasks,
        project_root=project_root,
        resume=args.resume,
        max_failures=args.max_failures,
        use_cpu_timing=args.use_cpu_timing,
        verbose=args.verbose,
        max_retries=args.max_retries,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    total_elapsed = time.time() - t_start

    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = args.out or (
        project_root / "results" / "evaluation"
        / f"batch_{args.direction}_{timestamp}"
    )
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Write batch summary JSON
    summary = {
        "direction": args.direction,
        "models": args.models,
        "suite": args.suite,
        "kernels": args.kernels,
        "task_list": str(args.task_list) if args.task_list is not None else None,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "thinking": args.thinking,
        "total_tasks": len(results),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": results,
    }
    summary_json = Path(str(out_prefix) + ".json")
    summary_json.write_text(json.dumps(summary, indent=2))

    # Write Markdown report
    title = args.title or f"Eval Batch: {args.direction} — {datetime.now().strftime('%Y-%m-%d')}"
    md = _generate_markdown(results, args.models, title)
    summary_md = Path(str(out_prefix) + ".md")
    summary_md.write_text(md)

    # Print final summary
    print(f"\nCompleted {len(results)}/{len(tasks)} tasks in {total_elapsed:.1f}s")
    passes = sum(1 for r in results if r.get("overall_status") == "PASS")
    print(f"PASS: {passes}/{len(results)}")
    print(f"Report: {summary_md}")


if __name__ == "__main__":
    main()
