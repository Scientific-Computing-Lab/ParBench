#!/usr/bin/env python3
"""
scripts/analysis/statistical_analysis.py

Statistical significance analysis for ParBench evaluation results.

Computes:
  - Wilson score 95% CI for every reported pass rate
  - Chi-squared / Fisher's exact tests for model comparison
  - Chi-squared test for augmentation level independence
  - Cochran-Armitage trend test for augmentation monotonic effect
  - Effect sizes: Cramer's V, odds ratios with Woolf CI, Cohen's h
  - Per-model augmentation curves with CIs
  - Direction asymmetry (McNemar's exact)
  - Sample size adequacy flags
  - Pass@k estimates (when multi-sample results exist)

Output:
  results/analysis/statistical_analysis.json
  results/analysis/statistical_analysis.md

Dependencies: scipy, numpy (install via pip in venv)

Usage:
    python3 scripts/analysis/statistical_analysis.py \\
        --project-root .
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Reuse result-loading infrastructure from analyze_eval.py to avoid duplication.
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.evaluation.analyze_eval import (
    _kernel_from_spec,
    load_results,
)


# --------------------------------------------------------------------------- #
# Predicates: deterministic vs stochastic records                              #
# --------------------------------------------------------------------------- #


def is_deterministic(record: dict) -> bool:
    """Return True if the record is from a deterministic run (temperature == 0)."""
    return (record.get("temperature") or 0) == 0.0


def is_stochastic(record: dict) -> bool:
    """Return True if the record is from a stochastic run (temperature > 0)."""
    return (record.get("temperature") or 0) > 0


# --------------------------------------------------------------------------- #
# Constants: magic-number replacements                                         #
# --------------------------------------------------------------------------- #

MIN_EXPECTED_CELL_COUNT: int = 5
"""Minimum expected cell count for chi-squared validity."""

MCNEMAR_EXACT_THRESHOLD: int = 25
"""Below this number of discordant pairs, use exact McNemar (binomial)."""

SAMPLE_SIZE_ADEQUACY_THRESHOLD: int = 10
"""Minimum n for a reliable CI; cells below this are flagged."""

# --------------------------------------------------------------------------- #
# Effect-size interpretation tables                                            #
# --------------------------------------------------------------------------- #

# Cohen's h: list of (threshold, label) tuples, checked in order.
# If |h| is below the threshold, that label is returned.
COHENS_H_INTERPRETATION: list[tuple[float, str]] = [
    (0.20, "small"),
    (0.80, "medium"),
]
"""Thresholds for Cohen's h.  Fall-through → 'large'."""

# Cramer's V: keyed by df* (min(rows, cols) - 1).
# Each value is a list of (threshold, label) tuples, checked in order.
CRAMERS_V_INTERPRETATIONS: dict[int, list[tuple[float, str]]] = {
    1: [(0.10, "negligible"), (0.30, "small"), (0.50, "medium")],
    2: [(0.07, "negligible"), (0.21, "small"), (0.35, "medium")],
}
"""Thresholds for Cramer's V by df*.  Fall-through → 'large'.
   df* not in the dict uses the default (df* >= 3) entry."""

_CRAMERS_V_DEFAULT: list[tuple[float, str]] = [
    (0.06, "negligible"),
    (0.17, "small"),
    (0.29, "medium"),
]


# --------------------------------------------------------------------------- #
# Wilson Score CI                                                              #
# --------------------------------------------------------------------------- #

def wilson_ci(passes: int, total: int, alpha: float = 0.05) -> dict:
    """Wilson score confidence interval for a binomial proportion.

    Wilson score is preferred over normal approximation because it:
    - Never produces intervals outside [0, 1]
    - Is accurate even for small n and extreme p
    - Is the default in modern clinical statistics

    Args:
        passes: Number of successes.
        total: Number of trials.
        alpha: Significance level (0.05 = 95% CI).

    Returns:
        {"rate": float, "ci_lower": float, "ci_upper": float, "n": int}
    """
    if total == 0:
        return {"rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = passes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom

    return {
        "rate": round(p_hat, 4),
        "ci_lower": round(max(0.0, center - spread), 4),
        "ci_upper": round(min(1.0, center + spread), 4),
        "n": total,
    }


# --------------------------------------------------------------------------- #
# Model comparison: omnibus chi-squared + pairwise Fisher's exact              #
# --------------------------------------------------------------------------- #

def test_model_comparison(
    records: list[dict], alpha: float = 0.05
) -> dict:
    """Test H0: all models have equal pass rates.

    Omnibus chi-squared on 2xK table (pass/fail x K models).
    Pairwise Fisher's exact with Bonferroni correction.

    Returns dict with omnibus test results and pairwise comparisons.
    """
    model_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        m = r.get("model", "unknown")
        model_counts[m]["total"] += 1
        if r.get("overall_status") == "PASS":
            model_counts[m]["pass"] += 1

    models = sorted(model_counts.keys())
    if len(models) < 2:
        return {"error": "Need at least 2 models for comparison"}

    # Omnibus chi-squared: rows = [PASS, FAIL], cols = models
    pass_row = [model_counts[m]["pass"] for m in models]
    fail_row = [model_counts[m]["total"] - model_counts[m]["pass"] for m in models]
    table = np.array([pass_row, fail_row])

    chi2, p_omni, dof, expected = sp_stats.chi2_contingency(table)
    n_total = int(table.sum())
    min_expected = float(expected.min())

    # Cramer's V
    df_star = min(table.shape) - 1
    cramers_v = math.sqrt(chi2 / (n_total * df_star)) if df_star > 0 and n_total > 0 else 0.0

    # Pairwise Fisher's exact with Bonferroni
    n_pairs = len(models) * (len(models) - 1) // 2
    alpha_pair = alpha / n_pairs if n_pairs > 0 else alpha
    pairwise = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            pair_table = np.array([
                [model_counts[m1]["pass"], model_counts[m2]["pass"]],
                [model_counts[m1]["total"] - model_counts[m1]["pass"],
                 model_counts[m2]["total"] - model_counts[m2]["pass"]],
            ])
            odds_ratio, p_fisher = sp_stats.fisher_exact(pair_table)
            # Odds ratio CI via Woolf's method
            or_ci = _odds_ratio_ci(pair_table, alpha=alpha)
            # Cohen's h
            p1 = model_counts[m1]["pass"] / max(model_counts[m1]["total"], 1)
            p2 = model_counts[m2]["pass"] / max(model_counts[m2]["total"], 1)
            h = cohens_h(p1, p2)
            pairwise.append({
                "pair": f"{m1} vs {m2}",
                "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else "inf",
                "or_ci_lower": or_ci["ci_lower"],
                "or_ci_upper": or_ci["ci_upper"],
                "p_value": round(p_fisher, 6),
                "p_corrected": round(min(p_fisher * n_pairs, 1.0), 6),
                "significant": p_fisher < alpha_pair,
                "cohens_h": round(h, 4),
                "effect_size": _interpret_h(h),
            })

    return {
        "omnibus_chi2": round(chi2, 4),
        "omnibus_p": round(p_omni, 6),
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
        "cramers_v_interpretation": _interpret_v(cramers_v, df_star),
        "min_expected_count": round(min_expected, 1),
        "low_expected_counts": min_expected < MIN_EXPECTED_CELL_COUNT,
        "n_total": n_total,
        "alpha_pairwise": round(alpha_pair, 6),
        "pairwise": pairwise,
    }


def test_model_comparison_task_level(
    records: list[dict], alpha: float = 0.05
) -> dict:
    """Task-level omnibus chi-squared + pairwise Fisher's exact.

    Within each task, 3 samples share the same kernel and direction, producing
    high intra-task correlation (79-89% deterministic outcomes).  Treating
    individual records as independent inflates the effective N.  This function
    classifies each L0 task as pass-any (>=1 sample passes) or all-fail, then
    runs the omnibus and pairwise tests on 142 tasks per model.
    """
    l0 = [r for r in records if r.get("augment_level") in (0, None)]
    if not l0:
        return {"error": "No L0 records found"}

    # Build per-(model, task_id) pass counts.  task_id = source_spec→target_spec.
    task_pass: dict[tuple[str, str], int] = defaultdict(int)
    task_total: dict[tuple[str, str], int] = defaultdict(int)
    for r in l0:
        m = r.get("model", "unknown")
        tid = f"{r.get('source_spec', '?')}->{r.get('target_spec', '?')}"
        task_total[(m, tid)] += 1
        if r.get("overall_status") == "PASS":
            task_pass[(m, tid)] += 1

    # Aggregate to per-model pass-any counts
    model_pass_any: dict[str, int] = defaultdict(int)
    model_tasks: dict[str, int] = defaultdict(int)
    for (m, tid), total in task_total.items():
        model_tasks[m] += 1
        if task_pass[(m, tid)] > 0:
            model_pass_any[m] += 1

    models = sorted(model_tasks.keys())
    if len(models) < 2:
        return {"error": "Need at least 2 models"}

    # Omnibus chi-squared: rows = [pass-any, all-fail], cols = models
    pass_row = [model_pass_any[m] for m in models]
    fail_row = [model_tasks[m] - model_pass_any[m] for m in models]
    table = np.array([pass_row, fail_row])

    chi2, p_omni, dof, expected = sp_stats.chi2_contingency(table)
    n_total = int(table.sum())
    min_expected = float(expected.min())

    df_star = min(table.shape) - 1
    cramers_v = math.sqrt(chi2 / (n_total * df_star)) if df_star > 0 and n_total > 0 else 0.0

    # Pairwise Fisher's exact with Bonferroni
    n_pairs = len(models) * (len(models) - 1) // 2
    alpha_pair = alpha / n_pairs if n_pairs > 0 else alpha
    pairwise = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            pair_table = np.array([
                [model_pass_any[m1], model_pass_any[m2]],
                [model_tasks[m1] - model_pass_any[m1],
                 model_tasks[m2] - model_pass_any[m2]],
            ])
            odds_ratio, p_fisher = sp_stats.fisher_exact(pair_table)
            or_ci = _odds_ratio_ci(pair_table, alpha=alpha)
            p1 = model_pass_any[m1] / max(model_tasks[m1], 1)
            p2 = model_pass_any[m2] / max(model_tasks[m2], 1)
            h = cohens_h(p1, p2)
            pairwise.append({
                "pair": f"{m1} vs {m2}",
                "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else "inf",
                "or_ci_lower": or_ci["ci_lower"],
                "or_ci_upper": or_ci["ci_upper"],
                "p_value": round(p_fisher, 6),
                "p_corrected": round(min(p_fisher * n_pairs, 1.0), 6),
                "significant": p_fisher < alpha_pair,
                "cohens_h": round(h, 4),
                "effect_size": _interpret_h(h),
            })

    return {
        "level": "task",
        "unit": "pass-any (>=1 of n samples passes) vs all-fail",
        "omnibus_chi2": round(chi2, 4),
        "omnibus_p": round(p_omni, 6),
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
        "cramers_v_interpretation": _interpret_v(cramers_v, df_star),
        "min_expected_count": round(min_expected, 1),
        "low_expected_counts": min_expected < MIN_EXPECTED_CELL_COUNT,
        "n_total": n_total,
        "per_model": {m: {"pass_any": model_pass_any[m], "tasks": model_tasks[m]}
                      for m in models},
        "alpha_pairwise": round(alpha_pair, 6),
        "pairwise": pairwise,
    }


# --------------------------------------------------------------------------- #
# Augmentation level independence: chi-squared per group                       #
# --------------------------------------------------------------------------- #

def test_augmentation_independence(
    records: list[dict],
    group_key: str = "model",
) -> list[dict]:
    """Test independence: augmentation level x pass/fail.

    For each group (model or direction), builds a contingency table:
      rows = augmentation levels (L0-L4)
      cols = [PASS, NOT_PASS]

    Uses chi-squared with Bonferroni correction.
    Also includes Fisher's exact for L0 vs aggregated L1-L4.
    """
    groups: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"pass": 0, "fail": 0})
    )
    for r in records:
        group = r.get(group_key, "unknown")
        level = r.get("augment_level", 0)
        if r.get("overall_status") == "PASS":
            groups[group][level]["pass"] += 1
        else:
            groups[group][level]["fail"] += 1

    results = []
    # Bonferroni denominator: count tests that will actually run (groups with >=2 levels)
    # +1 for the overall test if called from main() alongside per-group tests.
    # The caller can override via n_bonferroni parameter if needed.
    testable_groups = sum(1 for g in groups.values() if len(g) >= 2)
    n_tests = testable_groups + 1  # +1 for overall test (per methodology B1)

    for group, levels in sorted(groups.items()):
        if len(levels) < 2:
            continue

        level_keys = sorted(levels.keys())
        table = np.array(
            [[levels[lv]["pass"], levels[lv]["fail"]] for lv in level_keys]
        )

        # Check expected cell counts
        row_totals = table.sum(axis=1, keepdims=True)
        col_totals = table.sum(axis=0, keepdims=True)
        grand_total = table.sum()
        if grand_total == 0:
            continue
        expected = row_totals * col_totals / grand_total
        low_expected = bool((expected < MIN_EXPECTED_CELL_COUNT).any())

        # Skip if any row or column sums to zero (chi2_contingency would error)
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            continue

        chi2, p_value, dof, _ = sp_stats.chi2_contingency(table)
        p_corrected = min(p_value * n_tests, 1.0)

        min_dim = min(table.shape[0], table.shape[1]) - 1
        cramers_v = (
            math.sqrt(chi2 / (grand_total * min_dim))
            if min_dim > 0 and grand_total > 0
            else 0.0
        )

        # df* for Cramer's V interpretation is min(r,c)-1, NOT chi-squared dof
        # For a 5x2 table: df* = min(5,2)-1 = 1; chi2 dof = (5-1)*(2-1) = 4
        df_star_for_v = min_dim  # already min(table.shape[0], table.shape[1]) - 1

        result = {
            "group": group,
            "group_key": group_key,
            "levels_tested": level_keys,
            "contingency_table": table.tolist(),
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 6),
            "p_corrected_bonferroni": round(p_corrected, 6),
            "dof": dof,
            "cramers_v": round(cramers_v, 4),
            "cramers_v_interpretation": _interpret_v(cramers_v, df_star_for_v),
            "df_star": df_star_for_v,
            "low_expected_counts": low_expected,
            "significant_at_05": p_corrected < 0.05,
            "n_total": int(grand_total),
            "n_bonferroni_tests": n_tests,
        }

        # Fisher's exact for L0 vs aggregated L1-L4
        if len(level_keys) >= 2:
            l0_pass = levels.get(0, {"pass": 0})["pass"]
            l0_fail = levels.get(0, {"pass": 0, "fail": 0}).get("fail", 0)
            aug_pass = sum(levels[lv]["pass"] for lv in level_keys if lv > 0)
            aug_fail = sum(levels[lv]["fail"] for lv in level_keys if lv > 0)
            fisher_table = np.array([[l0_pass, l0_fail], [aug_pass, aug_fail]])
            if fisher_table.sum() > 0:
                odds_ratio, fisher_p = sp_stats.fisher_exact(fisher_table)
                result["fisher_exact_l0_vs_augmented"] = {
                    "table": fisher_table.tolist(),
                    "odds_ratio": (
                        round(odds_ratio, 4) if np.isfinite(odds_ratio) else "inf"
                    ),
                    "p_value": round(fisher_p, 6),
                    "p_corrected": round(min(fisher_p * n_tests, 1.0), 6),
                }

        results.append(result)

    return results


# --------------------------------------------------------------------------- #
# Cochran-Armitage trend test                                                  #
# --------------------------------------------------------------------------- #

def cochran_armitage_trend(
    pass_counts: list[int],
    total_counts: list[int],
    scores: list[int] | None = None,
) -> dict:
    """Cochran-Armitage test for trend in binomial proportions.

    More powerful than chi-squared for detecting ordered effects because
    it uses the ordinal structure of the levels.

    Args:
        pass_counts: Successes per group [L0, L1, ...].
        total_counts: Totals per group.
        scores: Ordinal scores (default: 0, 1, 2, ...).

    Returns:
        Dict with z-statistic, p-value, and trend direction.
    """
    k = len(pass_counts)
    if scores is None:
        scores = list(range(k))
    n = np.array(total_counts, dtype=float)
    r = np.array(pass_counts, dtype=float)
    s = np.array(scores, dtype=float)
    N = n.sum()
    R = r.sum()

    if N == 0:
        return {"z": 0.0, "p_value": 1.0, "trend_direction": "none"}

    p_bar = R / N
    t_bar = np.sum(n * s) / N

    numerator = np.sum(s * (r - n * p_bar))
    denominator_sq = p_bar * (1 - p_bar) * (np.sum(n * s**2) - N * t_bar**2)

    if denominator_sq <= 0:
        return {"z": 0.0, "p_value": 1.0, "trend_direction": "none"}

    denominator = math.sqrt(denominator_sq)
    z = float(numerator / denominator)
    p_value = float(2 * sp_stats.norm.sf(abs(z)))

    return {
        "z": round(z, 4),
        "p_value": round(p_value, 6),
        "trend_direction": "decreasing" if z < 0 else "increasing" if z > 0 else "none",
    }


def compute_augmentation_trends(records: list[dict], alpha: float = 0.05) -> dict:
    """Run Cochran-Armitage trend tests per model and overall (cuda-to-omp only).

    Restricts to cuda-to-omp, deterministic runs only (temp=0.0), and balanced
    subset (tasks present at ALL augmentation levels). This ensures the trend
    test compares equal-sized groups of identical tasks across levels.
    Bonferroni correction: N per-model + 1 overall tests.
    """
    # Filter to cuda-to-omp only
    filtered = [r for r in records if r.get("direction") == "cuda-to-omp"]
    if not filtered:
        return {"error": "No cuda-to-omp records found"}

    # Step 1: Filter to deterministic only (temp=0.0).
    # Stochastic samples (temp>0) inflate L0 counts relative to L1-L4,
    # creating an artificial imbalance that produces spurious trends.
    filtered = [r for r in filtered if is_deterministic(r)]
    if not filtered:
        return {"error": "No deterministic cuda-to-omp records found"}

    # Step 2: Find balanced subset — tasks present at ALL levels.
    # A task is identified by (kernel, model). Only tasks that appear at every
    # observed augmentation level are included, ensuring equal group sizes.
    task_levels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in filtered:
        k = _kernel_from_spec(r.get("source_spec", "?"))
        m = r.get("model", "unknown")
        lv = r.get("augment_level", 0)
        task_levels[(k, m)].add(lv)

    all_levels = set()
    for r in filtered:
        all_levels.add(r.get("augment_level", 0))

    balanced_tasks = {task for task, levels in task_levels.items()
                      if levels == all_levels}

    # Step 3: Restrict to balanced tasks only
    filtered = [
        r for r in filtered
        if (_kernel_from_spec(r.get("source_spec", "?")),
            r.get("model", "unknown")) in balanced_tasks
    ]
    if not filtered:
        return {"error": "No balanced cuda-to-omp tasks found across all levels"}

    # Group by (model, level)
    by_model_level: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"pass": 0, "total": 0})
    )
    overall_level: dict[int, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})

    for r in filtered:
        m = r.get("model", "unknown")
        lv = r.get("augment_level", 0)
        by_model_level[m][lv]["total"] += 1
        overall_level[lv]["total"] += 1
        if r.get("overall_status") == "PASS":
            by_model_level[m][lv]["pass"] += 1
            overall_level[lv]["pass"] += 1

    # Bonferroni: N per-model tests + 1 overall = N+1 tests
    n_tests = len(by_model_level) + 1
    alpha_corrected = alpha / n_tests

    results = {}

    # Overall trend
    levels_sorted = sorted(overall_level.keys())
    pass_counts = [overall_level[lv]["pass"] for lv in levels_sorted]
    total_counts = [overall_level[lv]["total"] for lv in levels_sorted]
    trend = cochran_armitage_trend(pass_counts, total_counts)
    trend["alpha_corrected"] = round(alpha_corrected, 6)
    trend["significant"] = trend["p_value"] < alpha_corrected
    trend["levels"] = levels_sorted
    trend["pass_counts"] = pass_counts
    trend["total_counts"] = total_counts
    results["overall"] = trend

    # Per-model trends
    for model in sorted(by_model_level.keys()):
        level_data = by_model_level[model]
        lvs = sorted(level_data.keys())
        pc = [level_data[lv]["pass"] for lv in lvs]
        tc = [level_data[lv]["total"] for lv in lvs]
        trend = cochran_armitage_trend(pc, tc)
        trend["alpha_corrected"] = round(alpha_corrected, 6)
        trend["significant"] = trend["p_value"] < alpha_corrected
        trend["levels"] = lvs
        trend["pass_counts"] = pc
        trend["total_counts"] = tc
        results[model] = trend

    return results


def compute_augmentation_trend_l1_l4(
    records: list[dict], alpha: float = 0.05
) -> dict:
    """Cochran-Armitage trend test on L1-L4 augmentation records only.

    The existing compute_augmentation_trends() requires deterministic (temp=0)
    records, which don't exist when all runs used temp>0.  L1-L4 augmentation
    records have n=1 per task per level, so there is no within-task correlation
    to inflate N.  This function also computes Fisher exact L1-vs-L4 per model.
    """
    aug = [r for r in records if r.get("augment_level", 0) >= 1]
    if not aug:
        return {"error": "No L1-L4 augmentation records found"}

    by_model_level: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"pass": 0, "total": 0})
    )
    for r in aug:
        m = r.get("model", "unknown")
        lv = r.get("augment_level", 0)
        by_model_level[m][lv]["total"] += 1
        if r.get("overall_status") == "PASS":
            by_model_level[m][lv]["pass"] += 1

    n_tests = len(by_model_level)
    alpha_corrected = alpha / max(n_tests, 1)
    results: dict[str, dict] = {}

    for model in sorted(by_model_level.keys()):
        level_data = by_model_level[model]
        lvs = sorted(level_data.keys())
        if len(lvs) < 2:
            results[model] = {"error": "Fewer than 2 augmentation levels"}
            continue

        pc = [level_data[lv]["pass"] for lv in lvs]
        tc = [level_data[lv]["total"] for lv in lvs]
        trend = cochran_armitage_trend(pc, tc)
        trend["alpha_corrected"] = round(alpha_corrected, 6)
        trend["significant"] = trend["p_value"] < alpha_corrected
        trend["levels"] = lvs
        trend["pass_counts"] = pc
        trend["total_counts"] = tc

        # Fisher exact: L1 vs L4 (first vs last level)
        l_first, l_last = lvs[0], lvs[-1]
        p1, n1 = level_data[l_first]["pass"], level_data[l_first]["total"]
        p4, n4 = level_data[l_last]["pass"], level_data[l_last]["total"]
        fisher_table = np.array([[p1, p4], [n1 - p1, n4 - p4]])
        fisher_or, fisher_p = sp_stats.fisher_exact(fisher_table)
        trend["fisher_l1_vs_l4"] = {
            "levels_compared": [l_first, l_last],
            "odds_ratio": round(fisher_or, 4) if np.isfinite(fisher_or) else "inf",
            "p_value": round(fisher_p, 6),
        }

        results[model] = trend

    return results


# --------------------------------------------------------------------------- #
# Direction asymmetry: McNemar's exact                                         #
# --------------------------------------------------------------------------- #

def _get_direction_pairs(
    d: str, directions: set[str]
) -> tuple[str, str] | None:
    """Return (forward, reverse) direction pair if the reverse exists in *directions*.

    Given direction ``'cuda-to-omp'`` and a set containing ``'omp-to-cuda'``,
    returns ``('cuda-to-omp', 'omp-to-cuda')``.  Returns ``None`` when the
    reverse direction is absent.
    """
    parts = d.split("-to-")
    if len(parts) != 2:
        return None
    reverse = f"{parts[1]}-to-{parts[0]}"
    if reverse not in directions:
        return None
    return (d, reverse)


def test_direction_asymmetry(records: list[dict], alpha: float = 0.05) -> list[dict]:
    """Test directional asymmetry using McNemar's exact test.

    For each direction pair (A->B vs B->A), builds a 2x2 table of
    concordant/discordant pairs at L0, then tests whether the asymmetry
    is significant.
    """
    # Only use L0 records to avoid augmentation confound.
    # Augmented levels (L1-L4) would create duplicate (kernel, model) pairs for
    # the same direction, inflating the sample size and violating McNemar's
    # assumption that each unit appears exactly once in each condition.
    l0_records = [r for r in records if r.get("augment_level") in (0, None)]

    # Build lookup: (kernel, model, direction) -> PASS/FAIL
    lookup: dict[tuple[str, str, str], bool] = {}
    for r in l0_records:
        kernel = _kernel_from_spec(r.get("source_spec", ""))
        model = r.get("model", "unknown")
        direction = r.get("direction", "unknown")
        passed = r.get("overall_status") == "PASS"
        lookup[(kernel, model, direction)] = passed

    # Identify direction pairs (A-to-B and B-to-A)
    directions = set()
    for r in l0_records:
        d = r.get("direction", "")
        if d:
            directions.add(d)

    pair_map: dict[tuple[str, str], tuple[str, str]] = {}
    for d in directions:
        pair = _get_direction_pairs(d, directions)
        if pair is not None:
            key = tuple(sorted(pair))
            if key not in pair_map:
                pair_map[key] = pair

    n_tests = len(pair_map) if pair_map else 1
    alpha_corrected = alpha / n_tests

    results = []
    for (d_fwd, d_rev) in pair_map.values():
        # Collect paired observations
        paired: list[tuple[bool, bool]] = []
        kernels_models = set()
        for (kernel, model, direction), passed in lookup.items():
            if direction == d_fwd:
                kernels_models.add((kernel, model))

        for (kernel, model) in kernels_models:
            fwd_pass = lookup.get((kernel, model, d_fwd))
            rev_pass = lookup.get((kernel, model, d_rev))
            if fwd_pass is not None and rev_pass is not None:
                paired.append((fwd_pass, rev_pass))

        if not paired:
            continue

        # Build McNemar's 2x2 table
        a = sum(1 for (f, r) in paired if f and r)       # both pass
        b = sum(1 for (f, r) in paired if not f and r)    # reverse only
        c = sum(1 for (f, r) in paired if f and not r)    # forward only
        d_val = sum(1 for (f, r) in paired if not f and not r)  # both fail
        discordant = b + c

        if discordant == 0:
            p_value = 1.0
            method = "exact"
        elif discordant < MCNEMAR_EXACT_THRESHOLD:
            # Exact McNemar's test (binomial)
            binom_result = sp_stats.binomtest(b, discordant, 0.5, alternative="two-sided")
            p_value = float(binom_result.pvalue)
            method = "exact"
        else:
            # Chi-squared McNemar's with continuity correction
            chi2_val = (abs(b - c) - 1) ** 2 / discordant if discordant > 0 else 0
            p_value = float(1 - sp_stats.chi2.cdf(chi2_val, df=1))
            method = "chi-squared"

        # Compute pass rates for each direction to report Cohen's h
        fwd_pass_count = a + c  # forward passed (both_pass + forward_only)
        rev_pass_count = a + b  # reverse passed (both_pass + reverse_only)
        n_p = len(paired)
        fwd_rate = fwd_pass_count / n_p if n_p > 0 else 0
        rev_rate = rev_pass_count / n_p if n_p > 0 else 0
        h = cohens_h(fwd_rate, rev_rate)

        results.append({
            "pair": f"{d_fwd} vs {d_rev}",
            "forward_direction": d_fwd,
            "reverse_direction": d_rev,
            "n_paired": n_p,
            "forward_pass_rate": round(fwd_rate, 4),
            "reverse_pass_rate": round(rev_rate, 4),
            "table": {"both_pass": a, "reverse_only": b, "forward_only": c, "both_fail": d_val},
            "discordant": discordant,
            "method": method,
            "p_value": round(p_value, 6),
            "alpha_corrected": round(alpha_corrected, 6),
            "significant": p_value < alpha_corrected,
            "cohens_h": round(h, 4),
            "effect_size": _interpret_h(h),
        })

    return results


# --------------------------------------------------------------------------- #
# Effect sizes                                                                  #
# --------------------------------------------------------------------------- #

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    |h| interpretation: <0.2 = small, 0.2-0.8 = medium, >=0.8 = large
    """
    return 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - 2 * math.asin(
        math.sqrt(max(0.0, min(1.0, p2)))
    )


def _interpret_h(h: float) -> str:
    """Interpret Cohen's h magnitude using COHENS_H_INTERPRETATION table."""
    abs_h = abs(h)
    for thresh, label in COHENS_H_INTERPRETATION:
        if abs_h < thresh:
            return label
    return "large"


def _interpret_v(v: float, df_star: int) -> str:
    """Interpret Cramer's V by df* using CRAMERS_V_INTERPRETATIONS table."""
    thresholds = CRAMERS_V_INTERPRETATIONS.get(df_star, _CRAMERS_V_DEFAULT)
    for thresh, label in thresholds:
        if v < thresh:
            return label
    return "large"


def _odds_ratio_ci(table_2x2: np.ndarray, alpha: float = 0.05) -> dict:
    """Odds ratio with Woolf's method CI (Haldane-corrected for zero cells)."""
    z = sp_stats.norm.ppf(1 - alpha / 2)
    a, b, c, d = table_2x2.flatten().astype(float)
    # Haldane correction if any cell is zero
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    OR = (a * d) / (b * c)
    log_OR = math.log(OR)
    SE = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lower = math.exp(log_OR - z * SE)
    ci_upper = math.exp(log_OR + z * SE)
    return {
        "or": round(OR, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


# --------------------------------------------------------------------------- #
# Augmentation curves with CIs                                                 #
# --------------------------------------------------------------------------- #

def augmentation_curves(records: list[dict], alpha: float = 0.05) -> dict:
    """Build per-model augmentation curves with Wilson CIs.

    Returns:
        {model: {level_str: {rate, ci_lower, ci_upper, n, pass, total}}}
    """
    groups: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"pass": 0, "total": 0}
    )
    for r in records:
        model = r.get("model", "unknown")
        level = r.get("augment_level", 0)
        groups[(model, level)]["total"] += 1
        if r.get("overall_status") == "PASS":
            groups[(model, level)]["pass"] += 1

    curves: dict[str, dict[str, dict]] = defaultdict(dict)
    for (model, level), counts in sorted(groups.items()):
        ci = wilson_ci(counts["pass"], counts["total"], alpha)
        ci["pass"] = counts["pass"]
        ci["total"] = counts["total"]
        curves[model][f"L{level}"] = ci

    return dict(curves)


# --------------------------------------------------------------------------- #
# Pass@k                                                                       #
# --------------------------------------------------------------------------- #

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).

    pass@k = 1 - C(n-c, k) / C(n, k)
    Computed in log space to avoid overflow.
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_ratio)


def compute_pass_at_k_table(
    records: list[dict], k_values: list[int] | None = None
) -> dict:
    """Compute pass@k for each (kernel, model, direction, level) group.

    Groups results by task identity (excluding sample_id). For each group,
    counts n (total samples) and c (PASS samples), then computes pass@k.

    Only stochastic samples (temperature > 0) are used. Deterministic runs
    (temperature == 0) violate the i.i.d. assumption underlying pass@k and
    are excluded.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    # Filter to stochastic samples only (temp > 0) — deterministic runs
    # violate the i.i.d. assumption required by the pass@k estimator.
    sample_records = [r for r in records if is_stochastic(r)]

    groups: dict[tuple, dict[str, int]] = defaultdict(lambda: {"n": 0, "c": 0})
    for r in sample_records:
        key = (
            _kernel_from_spec(r.get("source_spec", "?")),
            r.get("model", "?"),
            r.get("direction", "?"),
            r.get("augment_level", 0),
        )
        groups[key]["n"] += 1
        if r.get("overall_status") == "PASS":
            groups[key]["c"] += 1

    table = {}
    for key, counts in sorted(groups.items()):
        n, c = counts["n"], counts["c"]
        entry = {"n": n, "c": c}
        for k in k_values:
            if k <= n:
                entry[f"pass@{k}"] = round(pass_at_k(n, c, k), 4)
        table[str(key)] = entry

    return table


# --------------------------------------------------------------------------- #
# Sample size adequacy                                                         #
# --------------------------------------------------------------------------- #

def sample_size_adequacy(records: list[dict]) -> list[dict]:
    """Flag cells where n < SAMPLE_SIZE_ADEQUACY_THRESHOLD as insufficient for reliable CI.

    Checks per-model, per-direction, per-kernel, and per-(model x direction) cells.
    """
    flags = []

    def _check(label: str, n: int):
        if n < SAMPLE_SIZE_ADEQUACY_THRESHOLD:
            flags.append({"cell": label, "n": n, "adequate": False,
                          "note": "n < 10; CI width exceeds 30pp"})

    # Per-model
    model_n: dict[str, int] = defaultdict(int)
    dir_n: dict[str, int] = defaultdict(int)
    kernel_n: dict[str, int] = defaultdict(int)
    model_dir_n: dict[tuple[str, str], int] = defaultdict(int)

    for r in records:
        m = r.get("model", "unknown")
        d = r.get("direction", "unknown")
        k = _kernel_from_spec(r.get("source_spec", ""))
        model_n[m] += 1
        dir_n[d] += 1
        kernel_n[k] += 1
        model_dir_n[(m, d)] += 1

    for m, n in sorted(model_n.items()):
        _check(f"model:{m}", n)
    for d, n in sorted(dir_n.items()):
        _check(f"direction:{d}", n)
    for k, n in sorted(kernel_n.items()):
        _check(f"kernel:{k}", n)
    for (m, d), n in sorted(model_dir_n.items()):
        _check(f"model_x_dir:{m}/{d}", n)

    return flags


# --------------------------------------------------------------------------- #
# Compute all CIs for standard slices                                          #
# --------------------------------------------------------------------------- #

def compute_all_cis(records: list[dict], alpha: float = 0.05) -> dict:
    """Compute Wilson CIs for model, direction, kernel, and level slices."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    by_direction: dict[str, list[dict]] = defaultdict(list)
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    by_level: dict[int, list[dict]] = defaultdict(list)

    for r in records:
        by_model[r.get("model", "unknown")].append(r)
        by_direction[r.get("direction", "unknown")].append(r)
        by_kernel[_kernel_from_spec(r.get("source_spec", ""))].append(r)
        by_level[r.get("augment_level", 0)].append(r)

    def _ci_for_group(group_dict):
        result = {}
        for key, recs in sorted(group_dict.items()):
            passes = sum(1 for r in recs if r.get("overall_status") == "PASS")
            result[str(key) if not isinstance(key, str) else key] = wilson_ci(
                passes, len(recs), alpha
            )
        return result

    return {
        "by_model": _ci_for_group(by_model),
        "by_direction": _ci_for_group(by_direction),
        "by_kernel": _ci_for_group(by_kernel),
        "by_level": _ci_for_group({f"L{k}": v for k, v in by_level.items()}),
    }


# --------------------------------------------------------------------------- #
# Markdown report builder                                                      #
# --------------------------------------------------------------------------- #

def _build_markdown(data: dict) -> str:
    """Build publication-ready Markdown tables from statistical analysis."""
    lines = [
        "# ParBench Statistical Analysis",
        "",
        (
            f"**Generated:** {data['generated_at']}  |  "
            f"**Records:** {data['total_records']}  |  "
            f"**Alpha:** {data['alpha']}"
        ),
        "",
    ]

    # -- Wilson CIs --
    lines += [
        "## 1. Pass Rates with 95% Wilson Score CIs",
        "",
        "### By Model",
        "| Model | Rate | 95% CI | n |",
        "|-------|-----:|-------:|--:|",
    ]
    for model, ci in sorted(data["wilson_cis"]["by_model"].items()):
        lines.append(
            f"| {model} | {ci['rate']*100:.1f}% "
            f"| [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%] "
            f"| {ci['n']} |"
        )

    lines += [
        "",
        "### By Direction",
        "| Direction | Rate | 95% CI | n |",
        "|-----------|-----:|-------:|--:|",
    ]
    for direction, ci in sorted(data["wilson_cis"]["by_direction"].items()):
        lines.append(
            f"| {direction} | {ci['rate']*100:.1f}% "
            f"| [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%] "
            f"| {ci['n']} |"
        )

    lines += [
        "",
        "### By Kernel",
        "| Kernel | Rate | 95% CI | n |",
        "|--------|-----:|-------:|--:|",
    ]
    for kernel, ci in sorted(data["wilson_cis"]["by_kernel"].items()):
        flag = " *" if ci["n"] < SAMPLE_SIZE_ADEQUACY_THRESHOLD else ""
        lines.append(
            f"| {kernel} | {ci['rate']*100:.1f}% "
            f"| [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%] "
            f"| {ci['n']}{flag} |"
        )

    lines += [
        "",
        "### By Augmentation Level",
        "| Level | Rate | 95% CI | n |",
        "|-------|-----:|-------:|--:|",
    ]
    for level, ci in sorted(data["wilson_cis"]["by_level"].items()):
        lines.append(
            f"| {level} | {ci['rate']*100:.1f}% "
            f"| [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%] "
            f"| {ci['n']} |"
        )

    # -- Model comparison --
    mc = data.get("model_comparison", {})
    if "omnibus_chi2" in mc:
        lines += [
            "",
            "## 2. Model Comparison",
            "",
            (
                f"**Omnibus chi-squared:** chi2({mc['dof']}) = {mc['omnibus_chi2']:.2f}, "
                f"p = {mc['omnibus_p']:.2e}"
            ),
            f"**Cramer's V:** {mc['cramers_v']:.3f} ({mc['cramers_v_interpretation']})",
            "",
            "### Pairwise Comparisons (Fisher's exact, Bonferroni-corrected)",
            "| Pair | OR [95% CI] | p (corrected) | Cohen's h | Effect |",
            "|------|-------------|-------------:|----------:|:------:|",
        ]
        for pw in mc["pairwise"]:
            or_str = f"{pw['odds_ratio']}" if isinstance(pw["odds_ratio"], str) else f"{pw['odds_ratio']:.2f}"
            ci_str = f"[{pw['or_ci_lower']:.2f}, {pw['or_ci_upper']:.2f}]"
            sig = " **" if pw["significant"] else ""
            lines.append(
                f"| {pw['pair']} | {or_str} {ci_str} "
                f"| {pw['p_corrected']:.4f}{sig} "
                f"| {pw['cohens_h']:.3f} | {pw['effect_size']} |"
            )

    # -- Augmentation independence --
    lines += [
        "",
        "## 3. Augmentation Level Independence (Chi-Squared)",
        "",
        "**H0:** Pass rate is independent of augmentation level.",
        "**Correction:** Bonferroni for multiple comparisons.",
        "",
        "### By Model",
        "| Model | chi2 | p (corrected) | Cramer's V | Significant? | Low expected? |",
        "|-------|-----:|-------------:|----------:|:------------:|:-------------:|",
    ]
    for t in data.get("chi2_augmentation_by_model", []):
        sig = "Yes" if t["significant_at_05"] else "No"
        low = "Yes" if t["low_expected_counts"] else "No"
        v_interp = t.get("cramers_v_interpretation", "")
        lines.append(
            f"| {t['group']} | {t['chi2']:.2f} | {t['p_corrected_bonferroni']:.4f} "
            f"| {t['cramers_v']:.3f} ({v_interp}) | {sig} | {low} |"
        )

    # -- Cochran-Armitage trend --
    trends = data.get("augmentation_trends", {})
    if trends and "error" not in trends:
        lines += [
            "",
            "## 4. Augmentation Trend (Cochran-Armitage, cuda-to-omp)",
            "",
            "**H0:** No monotonic trend in pass rate across L0-L4.",
            "",
            "| Group | z | p-value | Trend | Significant? |",
            "|-------|--:|--------:|:------|:------------:|",
        ]
        for group, t in sorted(trends.items()):
            sig = "Yes" if t.get("significant", False) else "No"
            lines.append(
                f"| {group} | {t['z']:.3f} | {t['p_value']:.4f} "
                f"| {t['trend_direction']} | {sig} |"
            )

    # -- Direction asymmetry --
    asym = data.get("direction_asymmetry", [])
    if asym:
        lines += [
            "",
            "## 5. Direction Asymmetry (McNemar's Test, L0 only)",
            "",
            "| Direction Pair | n paired | Fwd Rate | Rev Rate | Cohen's h | p-value | Significant? |",
            "|----------------|--------:|--------:|--------:|----------:|--------:|:------------:|",
        ]
        for a in asym:
            sig = "Yes" if a["significant"] else "No"
            lines.append(
                f"| {a['pair']} | {a['n_paired']} "
                f"| {a.get('forward_pass_rate', 0)*100:.1f}% "
                f"| {a.get('reverse_pass_rate', 0)*100:.1f}% "
                f"| {a.get('cohens_h', 0):.3f} "
                f"| {a['p_value']:.4f} | {sig} |"
            )

    # -- Augmentation curves --
    lines += [
        "",
        "## 6. Augmentation Curves with CIs",
        "",
    ]
    for model, levels in sorted(data.get("augmentation_curves", {}).items()):
        lines.append(f"### {model}")
        lines.append("| Level | Rate | 95% CI | Pass/Total |")
        lines.append("|-------|-----:|-------:|----------:|")
        for level, ci in sorted(levels.items()):
            lines.append(
                f"| {level} | {ci['rate']*100:.1f}% "
                f"| [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%] "
                f"| {ci['pass']}/{ci['total']} |"
            )
        lines.append("")

    # -- Sample size adequacy --
    flags = data.get("sample_size_flags", [])
    if flags:
        lines += [
            "## 7. Sample Size Adequacy Flags",
            "",
            f"**{len(flags)} cells with n < 10** (insufficient for reliable CI):",
            "",
            "| Cell | n | Note |",
            "|------|--:|:-----|",
        ]
        for f in flags[:30]:  # Cap at 30 to keep report readable
            lines.append(f"| {f['cell']} | {f['n']} | {f['note']} |")
        if len(flags) > 30:
            lines.append(f"| ... | ... | ({len(flags) - 30} more) |")
        lines.append("")

    # -- Pass@k --
    pk = data.get("pass_at_k", {})
    if pk:
        lines += [
            "## 8. Pass@k Estimates",
            "",
            "| Task | n | c | pass@1 | pass@3 | pass@10 |",
            "|------|--:|--:|-------:|-------:|--------:|",
        ]
        for key, vals in sorted(pk.items()):
            p1 = vals.get("pass@1", "---")
            p3 = vals.get("pass@3", "---")
            p10 = vals.get("pass@10", "---")
            lines.append(
                f"| {key} | {vals['n']} | {vals['c']} | {p1} | {p3} | {p10} |"
            )
        lines.append("")

    # -- Methodology notes --
    lines += [
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Wilson score intervals** preferred over Wald (normal approximation) for "
        "binary outcomes: valid for small n and extreme proportions.",
        "- **Fisher's exact test** used for pairwise model comparisons (exact p-values, "
        "valid at any sample size).",
        "- **Bonferroni correction** applied to all families of tests to control FWER.",
        "- **Cochran-Armitage trend** uses ordinal structure of augmentation levels "
        "(more powerful than chi-squared for detecting monotonic trends).",
        "- **McNemar's exact** used for direction asymmetry (paired binary data, "
        "exact binomial when discordant pairs < 25).",
        "- **Cohen's h** thresholds: <0.2 small, 0.2-0.8 medium, >=0.8 large.",
        "- **Cramer's V** thresholds depend on df* (see Cohen 1988).",
        "- Cells with n < 10 flagged as insufficient for reliable inference.",
        "",
    ]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis for ParBench evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/analysis/statistical_analysis.py \\\n"
            "      --project-root .\n"
            "\n"
            "  python3 scripts/analysis/statistical_analysis.py --alpha 0.01\n"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory (default: auto-detected).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="k values for pass@k computation (default: 1 5 10).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_dir = project_root / "results" / "evaluation"
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    records = load_results(results_dir)
    if not records:
        print("ERROR: No result records found.", file=sys.stderr)
        sys.exit(1)
    if args.verbose:
        print(f"Loaded {len(records)} result records.")

    # 1. Wilson CIs for all slices
    if args.verbose:
        print("Computing Wilson CIs...")
    all_cis = compute_all_cis(records, args.alpha)

    # 2. Model comparison (omnibus + pairwise)
    if args.verbose:
        print("Running model comparison tests...")
    model_comp = test_model_comparison(records, args.alpha)

    # 3. Augmentation independence (chi-squared per model and direction)
    #    Only cuda-to-omp has L1-L4 augmented results; other directions are L0-only.
    if args.verbose:
        print("Testing augmentation level independence...")
    aug_records = [r for r in records if r.get("direction") == "cuda-to-omp"]
    chi2_by_model = test_augmentation_independence(aug_records, group_key="model")
    chi2_by_direction = test_augmentation_independence(aug_records, group_key="direction")

    # 4. Cochran-Armitage trend (cuda-to-omp, deterministic only — may be empty)
    if args.verbose:
        print("Running Cochran-Armitage trend tests...")
    aug_trends = compute_augmentation_trends(records, args.alpha)

    # 4b. Task-level model comparison (pass-any vs all-fail on L0 tasks)
    if args.verbose:
        print("Running task-level model comparison...")
    model_comp_task = test_model_comparison_task_level(records, args.alpha)

    # 4c. Cochran-Armitage on L1-L4 stochastic augmentation records
    if args.verbose:
        print("Running L1-L4 augmentation trend tests...")
    aug_trend_l1_l4 = compute_augmentation_trend_l1_l4(records, args.alpha)

    # 5. Direction asymmetry (McNemar's)
    if args.verbose:
        print("Testing direction asymmetry...")
    dir_asymmetry = test_direction_asymmetry(records, args.alpha)

    # 6. Augmentation curves
    if args.verbose:
        print("Building augmentation curves...")
    aug_curves = augmentation_curves(records, args.alpha)

    # 7. Sample size adequacy
    if args.verbose:
        print("Checking sample size adequacy...")
    size_flags = sample_size_adequacy(records)

    # 8. Pass@k (only if multi-sample data exists)
    has_samples = any(is_stochastic(r) for r in records)
    pass_k_table = {}
    if has_samples:
        if args.verbose:
            print("Computing pass@k estimates...")
        pass_k_table = compute_pass_at_k_table(records, args.k_values)
    elif args.verbose:
        print("No multi-sample data found; skipping pass@k.")

    # Assemble output
    output = {
        "generated_at": datetime.now().isoformat(),
        "alpha": args.alpha,
        "total_records": len(records),
        "wilson_cis": all_cis,
        "model_comparison": model_comp,
        "model_comparison_task_level": model_comp_task,
        "chi2_augmentation_by_model": chi2_by_model,
        "chi2_augmentation_by_direction": chi2_by_direction,
        "augmentation_trends": aug_trends,
        "augmentation_trend_l1_l4": aug_trend_l1_l4,
        "direction_asymmetry": dir_asymmetry,
        "augmentation_curves": aug_curves,
        "sample_size_flags": size_flags,
        "pass_at_k": pass_k_table,
    }

    # Write JSON
    json_path = output_dir / "statistical_analysis.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"JSON: {json_path}")

    # Write Markdown
    md = _build_markdown(output)
    md_path = output_dir / "statistical_analysis.md"
    md_path.write_text(md)
    print(f"Markdown: {md_path}")

    # Print summary to stdout
    print(f"\n=== Summary ({len(records)} records, alpha={args.alpha}) ===")
    for m, ci in sorted(all_cis["by_model"].items()):
        print(
            f"  {m}: {ci['rate']*100:.1f}% "
            f"[{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%] "
            f"(n={ci['n']})"
        )
    if "omnibus_p" in model_comp:
        print(
            f"  Model comparison: chi2={model_comp['omnibus_chi2']:.2f}, "
            f"p={model_comp['omnibus_p']:.2e}, "
            f"V={model_comp['cramers_v']:.3f} ({model_comp['cramers_v_interpretation']})"
        )
    if aug_trends and "overall" in aug_trends:
        t = aug_trends["overall"]
        print(
            f"  Augmentation trend (overall, cuda-to-omp): "
            f"z={t['z']:.3f}, p={t['p_value']:.4f}, {t['trend_direction']}"
        )
    if size_flags:
        print(f"  Sample size warnings: {len(size_flags)} cells with n < 10")


if __name__ == "__main__":
    main()
