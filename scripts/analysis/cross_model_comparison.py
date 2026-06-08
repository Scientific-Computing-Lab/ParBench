#!/usr/bin/env python3
"""Cross-model pairwise comparison for any two models.

Produces statistical comparison for NeurIPS 2026 paper using Phase 3 canonical+augmentation corpus.
Accepts any two paper_data JSON files via --model-a / --model-b.

Output: results/analysis/cross_model_comparison.json (default)

Statistical tests:
  - McNemar's test with Yates correction on paired task-level pass/fail (when passk_estimates available)
  - Chi-squared fallback on 2x2 contingency table (when passk_estimates absent)
  - Cohen's h effect sizes (overall + per-direction)
  - Per-direction paired comparison for common directions only
  - Per-kernel agreement matrix with four-way classification
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    Clamp inputs to [0, 1] to handle edge cases.
    Positive when p1 > p2.
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def classify_effect_size(h: float) -> str:
    """Classify |h| per Cohen's conventions.

    |h| < 0.2: negligible
    0.2 <= |h| < 0.5: small
    0.5 <= |h| < 0.8: medium
    |h| >= 0.8: large
    """
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


def compute_mcnemar(est_a: dict, est_b: dict) -> dict:
    common_tasks = sorted(set(est_a) & set(est_b))
    a_extra = len(set(est_a) - set(est_b))
    b_extra = len(set(est_b) - set(est_a))
    if a_extra or b_extra:
        print(f"WARNING: Task key mismatch: {a_extra} a-only, {b_extra} b-only. Using {len(common_tasks)} common tasks.", file=sys.stderr)

    both_pass = 0
    both_fail = 0
    a_only = 0
    b_only = 0

    for task in common_tasks:
        a_pass = est_a[task]["c"] >= 1
        b_pass = est_b[task]["c"] >= 1
        if a_pass and b_pass:
            both_pass += 1
        elif not a_pass and not b_pass:
            both_fail += 1
        elif a_pass:
            a_only += 1
        else:
            b_only += 1

    discordant = a_only + b_only
    if discordant > 0:
        # Yates continuity correction (matches statistical_analysis.py convention)
        mcnemar_chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
        p_value = float(1 - sp_stats.chi2.cdf(mcnemar_chi2, df=1))
    else:
        mcnemar_chi2 = 0.0
        p_value = 1.0

    return {
        "both_pass": both_pass,
        "both_fail": both_fail,
        "a_only": a_only,
        "b_only": b_only,
        "total": both_pass + both_fail + a_only + b_only,
        "mcnemar_chi2": round(mcnemar_chi2, 4),
        "p_value": round(p_value, 6),
    }


def build_comparison(data_a: dict, data_b: dict) -> dict:
    """Build the full cross-model comparison dict.

    Args:
        data_a: Parsed paper_data JSON for first model
        data_b: Parsed paper_data JSON for second model

    Returns:
        Dict with overall, per_direction, per_kernel_matrix sections.
    """
    name_a = data_a["model"]
    name_b = data_b["model"]

    pc_a = data_a.get("passk_campaign") or data_a["primary_campaign"]
    pc_b = data_b.get("passk_campaign") or data_b["primary_campaign"]

    # --- Determine common directions (intersection) ---
    dirs_a = set(pc_a["by_direction"].keys())
    dirs_b = set(pc_b["by_direction"].keys())
    common_dirs = sorted(dirs_a & dirs_b)

    missing = {}
    a_only_dirs = sorted(dirs_a - dirs_b)
    b_only_dirs = sorted(dirs_b - dirs_a)
    if a_only_dirs:
        missing[name_b] = a_only_dirs
    if b_only_dirs:
        missing[name_a] = b_only_dirs

    # --- Overall comparison ---
    overall_a = pc_a["overall"]
    overall_b = pc_b["overall"]

    pass_a = overall_a["pass"]
    total_a = overall_a["total"]
    fail_a = total_a - pass_a
    pass_b = overall_b["pass"]
    total_b = overall_b["total"]
    fail_b = total_b - pass_b

    rate_a = overall_a["pass_rate"]
    rate_b = overall_b["pass_rate"]
    h_overall = cohens_h(rate_a, rate_b)

    # McNemar on paired task-level data when passk_estimates available
    est_a = pc_a.get("passk_estimates")
    est_b = pc_b.get("passk_estimates")
    if est_a and est_b:
        mcnemar_result = compute_mcnemar(est_a, est_b)
        stat_key = "mcnemar"
        stat_value = mcnemar_result
    else:
        table = np.array([[pass_a, fail_a],
                          [pass_b, fail_b]])
        chi2, p_value_val, dof, _ = sp_stats.chi2_contingency(table)
        stat_key = "chi_squared"
        stat_value = {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value_val), 6),
            "dof": int(dof),
        }

    overall = {
        stat_key: stat_value,
        name_a: {
            "pass": int(pass_a),
            "total": int(total_a),
            "rate": round(float(rate_a), 4),
        },
        name_b: {
            "pass": int(pass_b),
            "total": int(total_b),
            "rate": round(float(rate_b), 4),
        },
        "cohens_h": round(float(h_overall), 4),
        "effect_size": classify_effect_size(h_overall),
    }

    # --- Per-direction comparison (common directions only) ---
    per_direction = {}
    for d in common_dirs:
        da = pc_a["by_direction"][d]
        db = pc_b["by_direction"][d]
        ra = da["pass_rate"]
        rb = db["pass_rate"]
        h = cohens_h(ra, rb)
        per_direction[d] = {
            name_a: {
                "pass": int(da["pass"]),
                "total": int(da["total"]),
                "rate": round(float(ra), 4),
            },
            name_b: {
                "pass": int(db["pass"]),
                "total": int(db["total"]),
                "rate": round(float(rb), 4),
            },
            "cohens_h": round(float(h), 4),
            "effect_size": classify_effect_size(h),
        }

    # --- Per-kernel agreement matrix (four-way per D-04) ---
    bk_a = pc_a.get("by_kernel", {})
    bk_b = pc_b.get("by_kernel", {})
    kernels_a = set(bk_a.keys())
    kernels_b = set(bk_b.keys())
    common_kernels = sorted(kernels_a & kernels_b)

    both_pass = []
    both_fail = []
    a_only_pass = []
    b_only_pass = []

    for kernel in common_kernels:
        ka = bk_a[kernel]
        kb = bk_b[kernel]
        a_passes = ka["pass"] > 0
        b_passes = kb["pass"] > 0
        if a_passes and b_passes:
            both_pass.append(kernel)
        elif not a_passes and not b_passes:
            both_fail.append(kernel)
        elif a_passes and not b_passes:
            a_only_pass.append(kernel)
        else:
            b_only_pass.append(kernel)

    per_kernel_matrix = {
        "total_common_kernels": len(common_kernels),
        "both_pass": sorted(both_pass),
        "both_fail": sorted(both_fail),
        f"{name_a}_only_pass": sorted(a_only_pass),
        f"{name_b}_only_pass": sorted(b_only_pass),
        "counts": {
            "both_pass": len(both_pass),
            "both_fail": len(both_fail),
            f"{name_a}_only_pass": len(a_only_pass),
            f"{name_b}_only_pass": len(b_only_pass),
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": [name_a, name_b],
        "common_directions": common_dirs,
        "missing_directions": missing,
        "overall": overall,
        "per_direction": per_direction,
        "per_kernel_matrix": per_kernel_matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model comparison for NeurIPS 2026 paper (Phase 3 canonical+augmentation corpus).",
    )
    parser.add_argument(
        "--model-a",
        type=Path,
        default=Path("results/analysis/paper_data_together-qwen-3.5-397b-a17b.json"),
        help="Path to first model's paper_data JSON",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=Path("results/analysis/paper_data_azure_gpt54.json"),
        help="Path to second model's paper_data JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/cross_model_comparison.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    # Resolve paths
    path_a = args.model_a if args.model_a.is_absolute() else Path.cwd() / args.model_a
    path_b = args.model_b if args.model_b.is_absolute() else Path.cwd() / args.model_b
    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output

    # Load data
    if not path_a.exists():
        print(f"ERROR: Model A data not found: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"ERROR: Model B data not found: {path_b}", file=sys.stderr)
        sys.exit(1)

    data_a = json.loads(path_a.read_text())
    data_b = json.loads(path_b.read_text())

    # Build comparison
    result = build_comparison(data_a, data_b)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output_path}")

    if args.verbose:
        o = result["overall"]
        name_a, name_b = result["models"]
        print(f"\n=== Cross-Model Comparison ===")
        print(f"{name_a}: {o[name_a]['pass']}/{o[name_a]['total']} ({o[name_a]['rate']:.1%})")
        print(f"{name_b}: {o[name_b]['pass']}/{o[name_b]['total']} ({o[name_b]['rate']:.1%})")
        if "mcnemar" in o:
            mcn = o["mcnemar"]
            print(f"McNemar: chi2={mcn['mcnemar_chi2']:.4f}, p={mcn['p_value']:.6f}")
            print(f"  Concordance: both_pass={mcn['both_pass']}, both_fail={mcn['both_fail']}, "
                  f"a_only={mcn['a_only']}, b_only={mcn['b_only']}")
        else:
            print(f"Chi-squared: chi2={o['chi_squared']['chi2']:.4f}, "
                  f"p={o['chi_squared']['p_value']:.6f}")
        print(f"Cohen's h: {o['cohens_h']:.4f} ({o['effect_size']})")
        print(f"\nCommon directions: {len(result['common_directions'])}")
        print(f"Missing: {result['missing_directions']}")
        km = result["per_kernel_matrix"]
        print(f"\nPer-kernel matrix ({km['total_common_kernels']} kernels):")
        print(f"  Both pass: {km['counts']['both_pass']}")
        print(f"  Both fail: {km['counts']['both_fail']}")
        print(f"  {name_a} only: {km['counts'][f'{name_a}_only_pass']}")
        print(f"  {name_b} only: {km['counts'][f'{name_b}_only_pass']}")
        print(f"\nPer-direction breakdown:")
        for d in result["common_directions"]:
            dd = result["per_direction"][d]
            print(f"  {d}: {name_a} {dd[name_a]['rate']:.1%} vs {name_b} {dd[name_b]['rate']:.1%} "
                  f"(h={dd['cohens_h']:.4f}, {dd['effect_size']})")


if __name__ == "__main__":
    main()
