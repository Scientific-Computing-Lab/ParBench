#!/usr/bin/env python3
"""
scripts/analysis/token_analysis.py

Token usage analysis across all 500 ParBench evaluation results.

Computes per-model, per-kernel, per-direction, and per-augmentation-level
token statistics including cost estimates, efficiency metrics, and
correlations between prompt size and translation success.

Pricing (per million tokens, as of 2026-04):
  Qwen 3.5 397B (Together): $0.60 input, $3.60 output
  Gemini 2.5 Flash:         $0.15 input, $0.60 output
  (Legacy models also defined in MODEL_PRICING for historical analysis)

Output: results/analysis/token_analysis.json + .md (5 tables)

Usage:
    python3 scripts/analysis/token_analysis.py \\
        --project-root .
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# The installed parbench package covers harness/ but not scripts/ (and on a
# host without the editable install, neither), so direct script invocation
# needs the repo root on sys.path before harness.* or scripts.* imports.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness.constants import CORRECTNESS_INELIGIBLE_SPECS  # noqa: E402
from scripts.evaluation.replay_records import promote_parent_metadata  # noqa: E402

# Per-million-token pricing
MODEL_PRICING = {
    "together-qwen-3.5-397b-a17b": {"input": 0.60, "output": 3.60, "display": "Qwen 3.5 397B (Together)"},
    "azure-gpt-5.4": {"input": 2.50, "output": 15.00, "display": "Azure GPT-5.4"},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "display": "Gemini 2.5 Flash"},
    # Legacy models (kept for historical result analysis)
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "display": "Claude Sonnet 4"},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30, "display": "Gemini 2.5 Flash-Lite"},
    "groq-llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79, "display": "Groq Llama 3.3 70B"},
}

# Field-name constants — single source of truth for JSON key names
FIELD_PROMPT_TOKENS = "prompt_tokens"
FIELD_COMPLETION_TOKENS = "completion_tokens"
FIELD_OVERALL_STATUS = "overall_status"
FIELD_SOURCE_SPEC = "source_spec"
FIELD_TARGET_SPEC = "target_spec"

# Immutable stored-translation properties. A Task 8 replay record carries the
# NEW verdict at top level but made no model call, so these fields live only on
# the referenced submitted record (input_record.path). They are recovered
# read-only so verdict-dependent token statistics use the replay verdict while
# the token counts stay the invariant originals (gate9 finding 1).
TOKEN_BACKFILL_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "llm_response_time_seconds",
    "kernel",
)

# Precision constants — controls rounding across the module
PRECISION_TOKENS = 1
PRECISION_RATE = 4
PRECISION_COST_DETAIL = 6
PRECISION_CORRELATION = 4


def extract_token_lists(results: list[dict]) -> tuple[list[int], list[int]]:
    """Extract parallel prompt and completion token lists from result dicts.

    Missing keys default to 0.  Returns (prompt_list, completion_list).
    """
    prompt_list = [r.get(FIELD_PROMPT_TOKENS, 0) for r in results]
    completion_list = [r.get(FIELD_COMPLETION_TOKENS, 0) for r in results]
    return prompt_list, completion_list


def _backfill_tokens_from_input(record: dict, project_root: Path) -> None:
    """Recover immutable token fields for a replay record from its submitted
    source (``input_record.path``), read-only, in place.

    Only replay records missing a field are touched, and only the immutable
    ``TOKEN_BACKFILL_FIELDS`` are copied - the record's own ``overall_status``
    (the replay verdict) is never overwritten (gate9 finding 1). A non-replay
    record, or one whose input record is unreadable, is left untouched.
    """
    if record.get("replay_kind") != "stored_translation_replay":
        return
    if all(record.get(k) is not None for k in TOKEN_BACKFILL_FIELDS):
        return
    input_record = record.get("input_record")
    if not isinstance(input_record, dict):
        return
    rel = input_record.get("path")
    if not rel:
        return
    for base in (project_root, Path.cwd()):
        candidate = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        try:
            src = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in TOKEN_BACKFILL_FIELDS:
            if record.get(key) is None and src.get(key) is not None:
                record[key] = src[key]
        return


def load_all_results(project_root: Path, results_root: Path | None = None) -> list[dict]:
    """Load all result JSONs from {results_root}/{model}/ directories.

    ``results_root`` defaults (historical, compatibility only — Task 5) to
    ``{project_root}/results/evaluation``.

    Results involving any spec in CORRECTNESS_INELIGIBLE_SPECS (KNOWN_FAIL or
    performance_only, as source or target) are filtered out to match the
    canonical denominator used by analyze_eval.py.
    """
    results = []
    eval_dir = results_root if results_root is not None else (
        project_root / "results" / "evaluation"
    )
    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for json_file in sorted(model_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                # Replay records carry identity under "parent" (Task 8 format)
                promote_parent_metadata(data)
                # ... and their immutable token fields on the submitted record
                # they reference; recover them without touching the verdict.
                _backfill_tokens_from_input(data, project_root)
                if FIELD_OVERALL_STATUS not in data:
                    continue
                # Filter out results involving excluded (KNOWN_FAIL) specs
                if data.get(FIELD_SOURCE_SPEC, "") in CORRECTNESS_INELIGIBLE_SPECS:
                    continue
                if data.get(FIELD_TARGET_SPEC, "") in CORRECTNESS_INELIGIBLE_SPECS:
                    continue
                data["_file"] = str(json_file.relative_to(project_root))
                results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def compute_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Compute cost in USD for a single result."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000


def evaluated_but_unpriced_models(results: list[dict]) -> list[str]:
    """Models present in ``results`` that have no MODEL_PRICING entry.

    gate8 finding 1: the per-model loop iterates MODEL_PRICING, so a model
    evaluated but absent from the pricing table (e.g. azure-gpt-5.3-codex) is
    silently dropped from the pooled grand total, understating a published
    'Total API cost'. Surface those models so the caller can fail closed rather
    than report a two-of-three-model figure as the pooled total.
    """
    evaluated = {r.get("model") for r in results if r.get("model")}
    return sorted(m for m in evaluated if m not in MODEL_PRICING)


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "total": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), PRECISION_TOKENS),
        "median": round(median(values), PRECISION_TOKENS),
        "std": round(std(values), PRECISION_TOKENS),
        "min": round(min(values), PRECISION_TOKENS),
        "max": round(max(values), PRECISION_TOKENS),
        "total": round(sum(values), PRECISION_TOKENS),
    }


def _average_ranks(vals: list[float]) -> list[float]:
    """Fractional (tie-aware) ranks: tied values share the mean of their ranks.

    Ranks are 1-based. This is what a validated Spearman implementation
    (scipy.stats.spearmanr) uses; the 1 - 6*Sum(d^2)/(n*(n^2-1)) shortcut is
    only exact when there are no ties, and the binary pass outcome is almost
    all ties (gate9 finding 3).
    """
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of the 1-based ranks i+1 .. j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute tie-aware Spearman rank correlation.

    Ranks both variables with average (fractional) ranks, then takes the
    Pearson correlation of the ranks - the tie-corrected definition that
    agrees with scipy.stats.spearmanr. Returns None when n < 3 or either
    ranked variable has zero variance (correlation undefined).
    """
    if len(xs) < 3:
        return None
    n = len(xs)
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return None
    return round(cov / math.sqrt(vx * vy), PRECISION_CORRELATION)


def get_direction(result: dict) -> str:
    """Extract translation direction from a result."""
    src_api = result.get(FIELD_SOURCE_SPEC, "").split("-")[-1]
    tgt_api = result.get(FIELD_TARGET_SPEC, "").split("-")[-1]
    return f"{src_api}-to-{tgt_api}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Explicit evaluation-results root to read (Task 5). The final "
             "analysis chain must always pass this; when omitted, the "
             "HISTORICAL default {project-root}/results/evaluation applies "
             "(compatibility only — never reached by the final-evidence "
             "orchestrator).",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    # Historical default retained for compatibility only (annotated above).
    results_root = (
        args.results_root.resolve() if args.results_root is not None else None
    )
    output_dir = args.output_dir or (project_root / "results" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(project_root, results_root)
    print(f"Loaded {len(results)} result files")
    if not results:
        print(
            f"ERROR: No result records found under "
            f"{results_root or project_root / 'results' / 'evaluation'}.",
            file=sys.stderr,
        )
        return 1

    # ── Per-model statistics ──────────────────────────────────────────
    # Iterate over every EVALUATED model, not the pricing table (gate9
    # finding 2): a model present in the corpus but absent from MODEL_PRICING
    # (e.g. azure-gpt-5.3-codex) must still receive its non-cost token and pass
    # statistics. Cost fields are null - not a wrong 0 - when the rate is
    # unavailable.
    by_model: dict[str, dict] = {}
    evaluated_models = sorted({r.get("model") for r in results if r.get("model")})
    for model_id in evaluated_models:
        model_results = [r for r in results if r.get("model") == model_id]
        if not model_results:
            continue

        pricing = MODEL_PRICING.get(model_id)
        priced = pricing is not None
        display_name = pricing["display"] if priced else model_id

        prompt_tokens, completion_tokens = extract_token_lists(model_results)
        response_times = [
            r.get("llm_response_time_seconds", 0)
            for r in model_results
            if r.get("llm_response_time_seconds")
        ]
        costs = [
            compute_cost(r.get(FIELD_PROMPT_TOKENS, 0), r.get(FIELD_COMPLETION_TOKENS, 0), model_id)
            for r in model_results
        ] if priced else []

        # Tokens per second (completion tokens / response time)
        tps = [
            r.get(FIELD_COMPLETION_TOKENS, 0) / r["llm_response_time_seconds"]
            for r in model_results
            if r.get("llm_response_time_seconds") and r["llm_response_time_seconds"] > 0
        ]

        pass_results = [r for r in model_results if r[FIELD_OVERALL_STATUS] == "PASS"]
        fail_results = [r for r in model_results if r[FIELD_OVERALL_STATUS] != "PASS"]

        pass_cost = sum(
            compute_cost(r.get(FIELD_PROMPT_TOKENS, 0), r.get(FIELD_COMPLETION_TOKENS, 0), model_id)
            for r in pass_results
        ) if priced else 0.0

        pass_prompt, pass_completion = extract_token_lists(pass_results)
        fail_prompt, fail_completion = extract_token_lists(fail_results)

        # Cost fields resolve to None for an unpriced model: a partial or zero
        # cost would misreport a two-of-three-model figure as complete.
        if priced:
            cost_usd = {
                "total": round(sum(costs), PRECISION_RATE),
                "mean_per_task": round(mean(costs), PRECISION_COST_DETAIL),
                "cost_per_pass": round(pass_cost / len(pass_results), PRECISION_COST_DETAIL) if pass_results else None,
                "total_pass_cost": round(pass_cost, PRECISION_RATE),
                "total_fail_cost": round(sum(costs) - pass_cost, PRECISION_RATE),
            }
        else:
            cost_usd = {
                "total": None,
                "mean_per_task": None,
                "cost_per_pass": None,
                "total_pass_cost": None,
                "total_fail_cost": None,
            }

        by_model[model_id] = {
            "display_name": display_name,
            "priced": priced,
            "total_results": len(model_results),
            "pass_count": len(pass_results),
            "pass_rate": round(len(pass_results) / len(model_results), PRECISION_RATE),
            "prompt_tokens": compute_stats(prompt_tokens),
            "completion_tokens": compute_stats(completion_tokens),
            "response_time_seconds": compute_stats(response_times),
            "tokens_per_second": compute_stats(tps),
            "cost_usd": cost_usd,
            # Pass vs fail token comparison
            "pass_prompt_tokens_mean": round(mean(pass_prompt), PRECISION_TOKENS) if pass_results else None,
            "fail_prompt_tokens_mean": round(mean(fail_prompt), PRECISION_TOKENS) if fail_results else None,
            "pass_completion_tokens_mean": round(mean(pass_completion), PRECISION_TOKENS) if pass_results else None,
            "fail_completion_tokens_mean": round(mean(fail_completion), PRECISION_TOKENS) if fail_results else None,
        }

    # ── Per-kernel statistics ─────────────────────────────────────────
    by_kernel: dict[str, dict] = {}
    kernel_groups = defaultdict(list)
    for r in results:
        kernel_groups[r.get("kernel", "unknown")].append(r)

    for kernel, kresults in sorted(kernel_groups.items()):
        prompt_tokens, completion_tokens = extract_token_lists(kresults)
        pass_count = sum(1 for r in kresults if r[FIELD_OVERALL_STATUS] == "PASS")
        by_kernel[kernel] = {
            "total_results": len(kresults),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / len(kresults), PRECISION_RATE),
            "prompt_tokens": compute_stats(prompt_tokens),
            "completion_tokens": compute_stats(completion_tokens),
        }

    # ── Per-direction statistics ──────────────────────────────────────
    by_direction: dict[str, dict] = {}
    direction_groups = defaultdict(list)
    for r in results:
        direction_groups[get_direction(r)].append(r)

    for direction, dresults in sorted(direction_groups.items()):
        prompt_tokens, completion_tokens = extract_token_lists(dresults)
        pass_count = sum(1 for r in dresults if r[FIELD_OVERALL_STATUS] == "PASS")
        by_direction[direction] = {
            "total_results": len(dresults),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / len(dresults), PRECISION_RATE),
            "prompt_tokens": compute_stats(prompt_tokens),
            "completion_tokens": compute_stats(completion_tokens),
        }

    # ── Per-augmentation-level statistics ─────────────────────────────
    by_level: dict[str, dict] = {}
    level_groups = defaultdict(list)
    for r in results:
        level = f"L{r.get('augment_level', 0)}"
        level_groups[level].append(r)

    for level, lresults in sorted(level_groups.items()):
        prompt_tokens, completion_tokens = extract_token_lists(lresults)
        pass_count = sum(1 for r in lresults if r[FIELD_OVERALL_STATUS] == "PASS")
        by_level[level] = {
            "total_results": len(lresults),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / len(lresults), PRECISION_RATE),
            "prompt_tokens": compute_stats(prompt_tokens),
            "completion_tokens": compute_stats(completion_tokens),
        }

    # ── Correlations ──────────────────────────────────────────────────
    # Per-kernel: mean prompt_tokens vs pass_rate
    kernel_prompt_means = [by_kernel[k]["prompt_tokens"]["mean"] for k in sorted(by_kernel)]
    kernel_pass_rates = [by_kernel[k]["pass_rate"] for k in sorted(by_kernel)]
    corr_prompt_vs_pass = spearman_correlation(kernel_prompt_means, kernel_pass_rates)

    # Individual result: prompt_tokens → pass (1) / fail (0) — point-biserial proxy
    all_prompt = [r.get(FIELD_PROMPT_TOKENS, 0) for r in results]
    all_pass = [1.0 if r[FIELD_OVERALL_STATUS] == "PASS" else 0.0 for r in results]
    pass_prompts = [p for p, s in zip(all_prompt, all_pass) if s == 1.0]
    fail_prompts = [p for p, s in zip(all_prompt, all_pass) if s == 0.0]

    # Completion tokens vs pass/fail correlation
    all_completion = [r.get(FIELD_COMPLETION_TOKENS, 0) for r in results]
    corr_completion_vs_pass = spearman_correlation(all_completion, all_pass)
    pass_completions = [c for c, s in zip(all_completion, all_pass) if s == 1.0]
    fail_completions = [c for c, s in zip(all_completion, all_pass) if s == 0.0]

    # ── Grand totals ──────────────────────────────────────────────────
    # gate8 finding 1: fail closed on any evaluated model without pricing. The
    # priced-only sum below is a partial estimate for the markdown; the pooled
    # cost field is None (not a wrong two-of-three-model number) when a model
    # lacks a verified rate.
    unpriced_models = evaluated_but_unpriced_models(results)
    # Sum only priced models; an unpriced model's total is None (gate9 finding 2).
    grand_total_cost = sum(
        m["cost_usd"]["total"] for m in by_model.values()
        if m["cost_usd"]["total"] is not None
    )
    grand_prompt_tokens = sum(r.get(FIELD_PROMPT_TOKENS, 0) for r in results)
    grand_completion_tokens = sum(r.get(FIELD_COMPLETION_TOKENS, 0) for r in results)
    grand_total_tokens = grand_prompt_tokens + grand_completion_tokens

    # gate9 finding 4: the hard-coded Together billing block (96.2M/24.3M
    # tokens, $145.37, "~46% coverage") was emitted as paper-reporting evidence
    # with no hash-bound source artifact in the repo, and the 46% figure did not
    # hold against the final-evidence token totals. Removed rather than
    # fabricate a source; provider-billing cost, if reported, must load a scoped
    # artifact whose hash is bound in the evidence manifest.

    output = {
        "analysis": "token_usage",
        "total_results": len(results),
        "grand_total_tokens": grand_total_tokens,
        "grand_prompt_tokens": grand_prompt_tokens,
        "grand_completion_tokens": grand_completion_tokens,
        # None when any evaluated model is unpriced (gate8 finding 1): a complete
        # pooled cost is not computable, so do not publish a partial one.
        "grand_total_cost_usd": (
            None if unpriced_models else round(grand_total_cost, PRECISION_RATE)
        ),
        "cost_priced_over_models": [
            mid for mid in by_model if mid in MODEL_PRICING
        ],
        "unpriced_evaluated_models": unpriced_models,
        "by_model": by_model,
        "by_kernel": by_kernel,
        "by_direction": by_direction,
        "by_augment_level": by_level,
        "correlations": {
            "kernel_mean_prompt_vs_pass_rate_spearman": corr_prompt_vs_pass,
            "pass_mean_prompt_tokens": round(mean(pass_prompts), PRECISION_TOKENS),
            "fail_mean_prompt_tokens": round(mean(fail_prompts), PRECISION_TOKENS),
            "completion_tokens_vs_pass_spearman": corr_completion_vs_pass,
            "pass_mean_completion_tokens": round(mean(pass_completions), PRECISION_TOKENS),
            "fail_mean_completion_tokens": round(mean(fail_completions), PRECISION_TOKENS),
            "note": (
                "Negative prompt correlation: larger kernels (more prompt tokens) "
                "are harder to translate. Completion correlation shows whether "
                "successful translations produce more or fewer output tokens."
            ),
        },
    }

    # Write JSON
    json_path = output_dir / "token_analysis.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    # ── Markdown report ───────────────────────────────────────────────
    md_lines = [
        "# Token Usage Analysis — ParBench Evaluation",
        "",
        f"**{len(results)} results** across {len(by_model)} models, "
        f"{len(by_kernel)} kernels, {len(by_direction)} directions.",
        f"Grand total (from result JSONs): **{grand_total_tokens:,} tokens** "
        f"({grand_prompt_tokens:,} input + {grand_completion_tokens:,} output), "
        + (
            f"estimated cost (priced models only, EXCLUDES unpriced "
            f"{', '.join(unpriced_models)}): **${grand_total_cost:.2f}**; a "
            f"complete pooled cost is not reported."
            if unpriced_models
            else f"estimated cost: **${grand_total_cost:.2f}**."
        ),
        "",
        "## Table 1: Per-Model Token Statistics",
        "",
        "| Model | N | Pass% | Prompt (mean) | Completion (mean) | tok/s (mean) | Total Cost | Cost/PASS |",
        "|-------|--:|------:|--------------:|------------------:|-------------:|-----------:|----------:|",
    ]

    for model_id in sorted(by_model):
        m = by_model[model_id]
        cpp = m["cost_usd"]["cost_per_pass"]
        cpp_str = f"${cpp:.4f}" if cpp is not None else "N/A"
        total = m["cost_usd"]["total"]
        total_str = f"${total:.2f}" if total is not None else "N/A"
        md_lines.append(
            f"| {m['display_name']} | {m['total_results']} | "
            f"{m['pass_rate']:.1%} | "
            f"{m['prompt_tokens']['mean']:,.0f} | "
            f"{m['completion_tokens']['mean']:,.0f} | "
            f"{m['tokens_per_second']['mean']:,.0f} | "
            f"{total_str} | {cpp_str} |"
        )

    md_lines.extend([
        "",
        "## Table 2: Per-Kernel Token Statistics (sorted by prompt size)",
        "",
        "| Kernel | N | Pass% | Prompt (mean) | Completion (mean) |",
        "|--------|--:|------:|--------------:|------------------:|",
    ])

    for kernel in sorted(by_kernel, key=lambda k: by_kernel[k]["prompt_tokens"]["mean"], reverse=True):
        k = by_kernel[kernel]
        md_lines.append(
            f"| {kernel} | {k['total_results']} | {k['pass_rate']:.1%} | "
            f"{k['prompt_tokens']['mean']:,.0f} | {k['completion_tokens']['mean']:,.0f} |"
        )

    md_lines.extend([
        "",
        "## Table 3: Per-Direction Token Statistics",
        "",
        "| Direction | N | Pass% | Prompt (mean) | Completion (mean) |",
        "|-----------|--:|------:|--------------:|------------------:|",
    ])

    for direction in sorted(by_direction, key=lambda d: by_direction[d]["total_results"], reverse=True):
        d = by_direction[direction]
        md_lines.append(
            f"| {direction} | {d['total_results']} | {d['pass_rate']:.1%} | "
            f"{d['prompt_tokens']['mean']:,.0f} | {d['completion_tokens']['mean']:,.0f} |"
        )

    md_lines.extend([
        "",
        "## Table 4: Per-Augmentation-Level Statistics",
        "",
        "| Level | N | Pass% | Prompt (mean) | Completion (mean) |",
        "|-------|--:|------:|--------------:|------------------:|",
    ])

    for level in sorted(by_level):
        lv = by_level[level]
        md_lines.append(
            f"| {level} | {lv['total_results']} | {lv['pass_rate']:.1%} | "
            f"{lv['prompt_tokens']['mean']:,.0f} | {lv['completion_tokens']['mean']:,.0f} |"
        )

    md_lines.extend([
        "",
        "## Table 5: Cost Analysis",
        "",
        "| Model | Total Cost | Cost on PASS | Cost on FAIL | Cost/PASS | Cost/Task |",
        "|-------|----------:|-----------:|-----------:|----------:|----------:|",
    ])

    for model_id in sorted(by_model):
        m = by_model[model_id]
        c = m["cost_usd"]
        cpp = f"${c['cost_per_pass']:.4f}" if c["cost_per_pass"] is not None else "N/A"
        total = f"${c['total']:.2f}" if c["total"] is not None else "N/A"
        pass_c = f"${c['total_pass_cost']:.2f}" if c["total_pass_cost"] is not None else "N/A"
        fail_c = f"${c['total_fail_cost']:.2f}" if c["total_fail_cost"] is not None else "N/A"
        mean_c = f"${c['mean_per_task']:.4f}" if c["mean_per_task"] is not None else "N/A"
        md_lines.append(
            f"| {m['display_name']} | {total} | "
            f"{pass_c} | {fail_c} | "
            f"{cpp} | {mean_c} |"
        )

    total_label = (
        f"**${grand_total_cost:.2f}** (priced models only; EXCLUDES "
        f"{', '.join(unpriced_models)})"
        if unpriced_models
        else f"**${grand_total_cost:.2f}**"
    )
    md_lines.extend([
        f"| **TOTAL** | {total_label} | | | | |",
        "",
        "## Correlations",
        "",
        f"- **Kernel-level (prompt)**: Spearman(mean prompt tokens, pass rate) = "
        f"**{corr_prompt_vs_pass}**",
        f"- **Result-level (prompt)**: Mean prompt tokens for PASS = "
        f"**{mean(pass_prompts):,.0f}**, for FAIL = **{mean(fail_prompts):,.0f}**",
        f"- **Result-level (completion)**: Spearman(completion tokens, pass) = "
        f"**{corr_completion_vs_pass}**; Mean completion for PASS = "
        f"**{mean(pass_completions):,.0f}**, for FAIL = **{mean(fail_completions):,.0f}**",
        "",
    ])

    md_path = output_dir / "token_analysis.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # Summary to stdout
    print(f"\n{'=' * 50}")
    print(f"Total tokens: {grand_total_tokens:,}")
    print(f"Total cost (priced models): ${grand_total_cost:.2f}")
    if unpriced_models:
        print(f"  (unpriced evaluated models EXCLUDED from cost: "
              f"{', '.join(unpriced_models)})")
    for model_id in sorted(by_model):
        m = by_model[model_id]
        total = m["cost_usd"]["total"]
        total_str = f"${total:.2f}" if total is not None else "cost N/A"
        print(f"  {m['display_name']}: {m['total_results']} results, {total_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
