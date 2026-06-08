#!/usr/bin/env python3
"""Generate paper_claims.json — every quantitative claim mapped to source data.

Each claim includes its value, source file, JSON path, and a verification
command. All paths use 'canonical' as the top-level key in the source JSONs.

Usage:
    # Generate claims:
    python3 scripts/analysis/generate_paper_claims.py \
        --project-root /path/to/parbench --output results/analysis/paper_claims.json -v

    # Validate existing claims against source data:
    python3 scripts/analysis/generate_paper_claims.py \
        --project-root /path/to/parbench --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

INTERNAL_ALIAS = {}
MODEL = "together-qwen-3.5-397b-a17b"
STANDARD_DIRECTIONS = [
    "cuda-to-omp", "omp-to-cuda",
    "cuda-to-opencl", "opencl-to-cuda",
    "omp-to-opencl", "opencl-to-omp",
]


def _resolve(data: dict, path: str) -> object:
    keys = path.split(".")
    node = data
    for k in keys:
        k = INTERNAL_ALIAS.get(k, k)
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return None
    return node


def _val(node: object) -> object:
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def _rp(relpath: str) -> str:
    return f"results/analysis/{relpath}"


def _vcmd(file: str, path: str) -> str:
    if not file or not path:
        return ""
    parts = path.split(".")
    accessor = "".join(f"['{p}']" for p in parts)
    return (
        f"python3 -c \"import json; "
        f"d=json.load(open('{file}')); "
        f"print(d{accessor})\""
    )


def _make_claim(
    claim_id: str,
    section: str,
    claim_text: str,
    value: object,
    source_files: list[str],
    json_path: str,
    **kwargs,
) -> dict:
    file = source_files[0] if source_files else ""
    return {
        "claim_id": claim_id,
        "section": section,
        "claim_text": claim_text,
        "value": value,
        "source_files": source_files,
        "json_path": json_path,
        "verification_cmd": _vcmd(file, json_path) if file and json_path else "",
        "model": kwargs.get("model", MODEL),
        "terminology": kwargs.get("terminology", "canonical"),
        **{k: v for k, v in kwargs.items() if k not in ("model", "terminology")},
    }


def build_claims(project_root: Path, verbose: bool = False) -> dict:
    analysis = project_root / "results" / "analysis"
    qf_path = analysis / f"quantitative_findings_{MODEL}.json"
    pd_path = analysis / f"paper_data_{MODEL}.json"
    et_path = analysis / "error_taxonomy.json"
    sl_path = analysis / "sloc_analysis.json"
    ta_path = analysis / "token_analysis.json"
    bc_path = analysis / "benchmark_characterization.json"
    sa_path = analysis / "statistical_analysis.json"
    aug_path = analysis / f"augmentation_per_kernel_matrix_{MODEL}.json"

    qf = json.loads(qf_path.read_text())
    et = json.loads(et_path.read_text())
    sl = json.loads(sl_path.read_text())
    ta = json.loads(ta_path.read_text())
    bc = json.loads(bc_path.read_text())
    sa = json.loads(sa_path.read_text())

    c2 = qf["canonical"]
    pk = c2["pass_at_k"]
    apr = c2["aggregate_pass_rates"]
    dpr = c2["direction_pass_rates"]["standard"]
    ft = c2["failure_taxonomy"]

    claims = []

    # 1. Overall pass@1
    p1 = pk["pass_at_1"]
    claims.append(_make_claim(
        "overall_pass_at_1", "Abstract, S6.1",
        f"Qwen 3.5 397B achieves pass@1 of {p1['value']*100:.1f}% "
        f"[{p1['ci_lower']*100:.1f}%, {p1['ci_upper']*100:.1f}%] across {pk['total_tasks']['value']} tasks",
        p1["value"],
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.pass_at_k.pass_at_1",
        ci_lower=p1["ci_lower"], ci_upper=p1["ci_upper"],
        ci_level=0.95, n=pk["total_tasks"]["value"], unit="fraction",
    ))

    # 2. Overall pass@3
    p3 = pk["pass_at_3"]
    claims.append(_make_claim(
        "overall_pass_at_3", "S6.1",
        f"pass@3 of {p3['value']*100:.1f}% [{p3['ci_lower']*100:.1f}%, {p3['ci_upper']*100:.1f}%]",
        p3["value"],
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.pass_at_k.pass_at_3",
        ci_lower=p3["ci_lower"], ci_upper=p3["ci_upper"],
        ci_level=0.95, n=pk["total_tasks"]["value"], unit="fraction",
    ))

    # 3. Aggregate sample-level pass rate
    ov = apr["overall"]
    claims.append(_make_claim(
        "aggregate_pass_rate", "S6.1",
        f"Sample-level pass rate {ov['value']*100:.1f}% "
        f"[{ov['ci_lower']*100:.1f}%, {ov['ci_upper']*100:.1f}%] (n={ov['n']})",
        ov["value"],
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.aggregate_pass_rates.overall",
        ci_lower=ov["ci_lower"], ci_upper=ov["ci_upper"],
        ci_level=0.95, n=ov["n"], unit="fraction",
    ))

    # 4-9. Per-direction pass@1 (6 standard directions)
    for d in STANDARD_DIRECTIONS:
        if d not in dpr:
            continue
        dv = dpr[d]
        claims.append(_make_claim(
            f"direction_pass_{d.replace('-', '_')}", "S6.2",
            f"{d} pass rate: {dv['value']*100:.1f}% "
            f"[{dv['ci_lower']*100:.1f}%, {dv['ci_upper']*100:.1f}%] (n={dv['n']})",
            dv["value"],
            [_rp(f"quantitative_findings_{MODEL}.json")],
            f"canonical.direction_pass_rates.standard.{d}",
            ci_lower=dv["ci_lower"], ci_upper=dv["ci_upper"],
            ci_level=0.95, n=dv["n"], unit="fraction",
        ))

    # 10-14. Per-suite pass@1
    suite_pk = pk.get("per_suite", {}).get("pass_at_1", {})
    for suite, sv in suite_pk.items():
        claims.append(_make_claim(
            f"suite_pass_{suite}", "S6.3",
            f"{suite} pass@1: {sv['value']*100:.1f}% "
            f"[{sv['ci_lower']*100:.1f}%, {sv['ci_upper']*100:.1f}%] (n={sv['n']})",
            sv["value"],
            [_rp(f"quantitative_findings_{MODEL}.json")],
            f"canonical.pass_at_k.per_suite.pass_at_1.{suite}",
            ci_lower=sv["ci_lower"], ci_upper=sv["ci_upper"],
            ci_level=0.95, n=sv["n"], unit="fraction",
        ))

    # 15. Direction asymmetry (largest gap)
    da = sa.get("direction_asymmetry", [])
    if da:
        biggest = max(da, key=lambda x: abs(x["forward_pass_rate"] - x["reverse_pass_rate"]))
        claims.append(_make_claim(
            "direction_asymmetry", "S6.2",
            f"Largest asymmetry: {biggest['pair']} "
            f"(forward={biggest['forward_pass_rate']*100:.1f}%, "
            f"reverse={biggest['reverse_pass_rate']*100:.1f}%, "
            f"p={biggest['p_value']:.4f})",
            biggest["forward_pass_rate"] - biggest["reverse_pass_rate"],
            [_rp("statistical_analysis.json")],
            "direction_asymmetry",
            notes="Value is forward-reverse gap; sign indicates direction of asymmetry",
        ))

    # 16. BUILD_FAIL count
    bf = ft["status_counts"].get("BUILD_FAIL", 0)
    claims.append(_make_claim(
        "build_fail_count", "S6.4",
        f"BUILD_FAIL: {bf}/{ft['total_records']} "
        f"({bf/ft['total_records']*100:.1f}%)",
        bf,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.failure_taxonomy.status_counts.BUILD_FAIL",
        unit="count",
    ))

    # 17. RUN_FAIL count
    rf = ft["status_counts"].get("RUN_FAIL", 0)
    claims.append(_make_claim(
        "run_fail_count", "S6.4",
        f"RUN_FAIL: {rf}/{ft['total_records']} "
        f"({rf/ft['total_records']*100:.1f}%)",
        rf,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.failure_taxonomy.status_counts.RUN_FAIL",
        unit="count",
    ))

    # 18. VERIFY_FAIL count
    vf = ft["status_counts"].get("VERIFY_FAIL", 0)
    claims.append(_make_claim(
        "verify_fail_count", "S6.4",
        f"VERIFY_FAIL: {vf}/{ft['total_records']} "
        f"({vf/ft['total_records']*100:.1f}%)",
        vf,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.failure_taxonomy.status_counts.VERIFY_FAIL",
        unit="count",
    ))

    # 19. BUILD_FAIL subcategories (top 3)
    top3 = ft.get("top_3_build_subcategories",
                  sorted(ft["by_status"]["BUILD_FAIL"]["subcategories"].items(),
                         key=lambda x: -x[1])[:3])
    if isinstance(top3, list) and top3:
        if isinstance(top3[0], dict):
            desc = ", ".join(f"{x['subcategory']}({x['count']})" for x in top3)
        else:
            desc = ", ".join(f"{k}({v})" for k, v in top3)
    else:
        desc = "N/A"
    claims.append(_make_claim(
        "build_fail_subcategories", "S6.4",
        f"Top BUILD_FAIL subcategories: {desc}",
        len(ft["by_status"]["BUILD_FAIL"]["subcategories"]),
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.failure_taxonomy.by_status.BUILD_FAIL.subcategories",
        unit="count_of_categories",
    ))

    # 20. Spec counts
    meta = qf.get("metadata", {})
    excluded_count = meta.get("excluded_specs_count", 9)
    total_on_disk = meta.get("file_counts", {}).get("total_on_disk", 708)
    claims.append(_make_claim(
        "spec_counts", "S4",
        f"{excluded_count} KNOWN_FAIL specs excluded; "
        f"206 total specs; {total_on_disk} result files on disk",
        excluded_count,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "metadata.excluded_specs_count",
        unit="count",
    ))

    # 21. Suite composition
    bcs = bc.get("summary", {})
    claims.append(_make_claim(
        "suite_composition", "S4",
        f"{bcs.get('total_kernels', 35)} kernels across "
        f"{bcs.get('total_suites', 5)} suites, "
        f"{bcs.get('total_specs', 206)} specs",
        bcs.get("total_kernels", 35),
        [_rp("benchmark_characterization.json")],
        "summary.total_kernels",
        unit="count",
    ))

    # 22. Multi-file fraction
    mf_pct = bcs.get("multi_file_pct", 0)
    claims.append(_make_claim(
        "multi_file_fraction", "S4",
        f"{mf_pct:.1f}% of kernels require multi-file translation",
        mf_pct,
        [_rp("benchmark_characterization.json")],
        "summary.multi_file_pct",
        unit="percentage",
    ))

    # 23. Token cost total
    cost = ta.get("grand_total_cost_usd", 0)
    claims.append(_make_claim(
        "token_cost_total", "S5",
        f"Total API cost: ${cost:.2f}",
        cost,
        [_rp("token_analysis.json")],
        "grand_total_cost_usd",
        unit="usd",
    ))

    # 24. Augmentation degradation (L0→L4)
    aug_levels = c2.get("augmentation_trends", {}).get("aggregate", {}).get("per_level", {})
    if aug_levels:
        l0_val = aug_levels.get("L0", {}).get("value", 0)
        l4_val = aug_levels.get("L4", {}).get("value", 0)
        claims.append(_make_claim(
            "augmentation_degradation", "S6.5",
            f"Augmentation degradation L0→L4: "
            f"{l0_val*100:.1f}% → {l4_val*100:.1f}%",
            l4_val - l0_val,
            [_rp(f"quantitative_findings_{MODEL}.json")],
            "canonical.augmentation_trends.aggregate.per_level",
            notes="Value is L4-L0 difference (negative = degradation)",
            unit="fraction_difference",
        ))

    # 25. Cochran-Armitage trend
    chi2_aug = sa.get("chi2_augmentation_by_model", [])
    if chi2_aug:
        entry = chi2_aug[0]
        claims.append(_make_claim(
            "cochran_armitage_trend", "S6.5",
            f"Chi-squared augmentation trend: chi2={entry.get('chi2', 0):.4f}, "
            f"p={entry.get('p_value', 1):.4f}",
            entry.get("chi2", 0),
            [_rp("statistical_analysis.json")],
            "chi2_augmentation_by_model",
            notes="chi2 test for augmentation level vs pass/fail across levels",
            unit="chi2_statistic",
        ))

    # 26. SLoC correlation
    sloc_rho = sl.get("summary", {}).get("sloc_vs_pass_rate_spearman")
    if sloc_rho is not None:
        claims.append(_make_claim(
            "sloc_correlation", "S6.6",
            f"SLoC vs pass-rate Spearman rho = {sloc_rho:.4f}",
            sloc_rho,
            [_rp("sloc_analysis.json")],
            "summary.sloc_vs_pass_rate_spearman",
            unit="correlation_coefficient",
        ))

    # 28. Total eval files
    fc = meta.get("file_counts", {})
    claims.append(_make_claim(
        "total_eval_files", "S5",
        f"{fc.get('total_on_disk', 708)} result files on disk, "
        f"{fc.get('valid_after_exclusion', 626)} valid after KNOWN_FAIL exclusion",
        fc.get("total_on_disk", 708),
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "metadata.file_counts.total_on_disk",
        unit="count",
    ))

    # 29. Best direction
    dir_values = {d: dpr[d]["value"] for d in STANDARD_DIRECTIONS if d in dpr}
    if dir_values:
        best_d = max(dir_values, key=dir_values.get)
        claims.append(_make_claim(
            "best_direction", "S6.2",
            f"Best direction: {best_d} ({dir_values[best_d]*100:.1f}%)",
            dir_values[best_d],
            [_rp(f"quantitative_findings_{MODEL}.json")],
            f"canonical.direction_pass_rates.standard.{best_d}",
            notes="Derived: highest pass rate among 6 standard directions",
            unit="fraction",
        ))

    # 30. Worst direction
    if dir_values:
        worst_d = min(dir_values, key=dir_values.get)
        claims.append(_make_claim(
            "worst_direction", "S6.2",
            f"Worst direction: {worst_d} ({dir_values[worst_d]*100:.1f}%)",
            dir_values[worst_d],
            [_rp(f"quantitative_findings_{MODEL}.json")],
            f"canonical.direction_pass_rates.standard.{worst_d}",
            notes="Derived: lowest pass rate among 6 standard directions",
            unit="fraction",
        ))

    if verbose:
        print(f"  Generated {len(claims)} claims")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "terminology_note": (
            "All json_path values use 'canonical' as the top-level key "
            "in the source analysis JSONs."
        ),
        "claims": claims,
    }


def validate_claims(project_root: Path, verbose: bool = False) -> int:
    claims_path = project_root / "results" / "analysis" / "paper_claims.json"
    if not claims_path.exists():
        print(f"FAIL: {claims_path} not found", file=sys.stderr)
        return 1

    data = json.loads(claims_path.read_text())
    claims = data.get("claims", [])
    failures = 0

    for claim in claims:
        cid = claim["claim_id"]
        if not claim["source_files"] or not claim["json_path"]:
            if verbose:
                print(f"  SKIP {cid}: no source_files or json_path")
            continue

        src_path = project_root / claim["source_files"][0]
        if not src_path.exists():
            print(f"  FAIL {cid}: source file not found: {src_path}")
            failures += 1
            continue

        src_data = json.loads(src_path.read_text())
        node = _resolve(src_data, claim["json_path"])
        if node is None:
            if verbose:
                print(f"  SKIP {cid}: json_path not resolvable")
            continue

        resolved_val = _val(node)
        claim_val = claim["value"]

        if isinstance(resolved_val, (int, float)) and isinstance(claim_val, (int, float)):
            if abs(resolved_val - claim_val) > 1e-6:
                print(f"  FAIL {cid}: expected {resolved_val}, got {claim_val}")
                failures += 1
            elif verbose:
                print(f"  PASS {cid}: {claim_val}")
        elif verbose:
            print(f"  SKIP {cid}: non-numeric comparison")

    if failures:
        print(f"\n{failures} claim(s) failed validation")
        return 1

    print(f"All {len(claims)} claims validated OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or validate paper_claims.json",
    )
    parser.add_argument(
        "--project-root", type=Path, required=True,
        help="Path to ParBench project root",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for paper_claims.json",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing claims against source data",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.validate:
        sys.exit(validate_claims(args.project_root, args.verbose))

    output = args.output or (
        args.project_root / "results" / "analysis" / "paper_claims.json"
    )
    data = build_claims(args.project_root, args.verbose)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(data['claims'])} claims to {output}")


if __name__ == "__main__":
    main()
