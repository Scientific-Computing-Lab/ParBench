#!/usr/bin/env python3
"""
scripts/analysis/validate_characterization.py

Independent cross-validation of benchmark_characterization.json against raw data sources.
Uses a completely different code path from benchmark_characterization.py to catch bugs.

Checks:
  1. Schema: top-level and nested key presence
  2. SLoC (CHAR-01): all 35 kernels present, SLoC values > 0, summary stats valid
  3. Categories (CHAR-02): 12 categories, kernel counts match manifest.jsonl
  4. API coverage (CHAR-03): 5x4 matrix matches spec file enumeration
  5. Multi-file (CHAR-04): counts match spec prompt_payload/translation_targets
  6. Language features (CHAR-05): valid tiers, coverage of corpus kernels
  7. Language standards (CHAR-06): distribution sums to 206, matches spec field scan
  8. Cross-metric: kernel counts consistent across sections

Usage:
    python3 scripts/analysis/validate_characterization.py \\
        --project-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Import CORPUS_KERNELS from sloc_analysis (same directory — ground truth)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sloc_analysis import CORPUS_KERNELS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_characterization(project_root: Path) -> dict:
    """Load the characterization JSON.  Exit if it does not exist."""
    path = project_root / "results" / "analysis" / "benchmark_characterization.json"
    if not path.exists():
        print(
            f"ERROR: {path} not found.  Run benchmark_characterization.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_valid_manifest(project_root: Path) -> list[dict]:
    """Parse manifest.jsonl, returning only entries whose spec file exists on disk."""
    manifest_path = project_root / "manifest.jsonl"
    entries: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            spec_file = entry.get("spec_file", "")
            if (project_root / spec_file).exists():
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Check 1: Schema validation
# ---------------------------------------------------------------------------
def check_schema(data: dict) -> list[str]:
    """Verify presence of all required top-level and nested keys."""
    errors: list[str] = []

    required_top = [
        "metadata", "sloc", "categories", "api_coverage",
        "multi_file", "language_features", "language_standards", "summary",
    ]
    for key in required_top:
        if key not in data:
            errors.append(f"Schema: missing top-level key '{key}'")

    # metadata sub-keys
    if "metadata" in data:
        for mkey in ["generated", "script"]:
            if mkey not in data["metadata"]:
                errors.append(f"Schema: metadata missing key '{mkey}'")

    # sloc sub-keys
    if "sloc" in data:
        for skey in ["per_kernel", "summary"]:
            if skey not in data["sloc"]:
                errors.append(f"Schema: sloc missing key '{skey}'")

    # multi_file sub-keys
    if "multi_file" in data:
        for mfkey in ["by_kernel", "aggregate"]:
            if mfkey not in data["multi_file"]:
                errors.append(f"Schema: multi_file missing key '{mfkey}'")

    # language_features sub-keys
    if "language_features" in data:
        if "per_kernel" not in data["language_features"]:
            errors.append("Schema: language_features missing key 'per_kernel'")

    # language_standards sub-keys
    if "language_standards" in data:
        for lskey in ["distribution", "total_specs"]:
            if lskey not in data["language_standards"]:
                errors.append(f"Schema: language_standards missing key '{lskey}'")

    # api_coverage sub-keys
    if "api_coverage" in data:
        for ackey in ["suites", "totals"]:
            if ackey not in data["api_coverage"]:
                errors.append(f"Schema: api_coverage missing key '{ackey}'")

    return errors


# ---------------------------------------------------------------------------
# Check 2: SLoC (CHAR-01)
# ---------------------------------------------------------------------------
def check_sloc(data: dict, project_root: Path) -> list[str]:
    """Validate SLoC section against CORPUS_KERNELS ground truth."""
    errors: list[str] = []
    sloc_section = data.get("sloc", {})
    per_kernel = sloc_section.get("per_kernel", {})

    # -- Verify kernel count
    expected_count = len(CORPUS_KERNELS)  # 35
    actual_count = len(per_kernel)
    if actual_count != expected_count:
        errors.append(
            f"SLoC (CHAR-01): expected {expected_count} kernels, got {actual_count}"
        )

    # -- Verify kernel names match ground truth
    expected_names = {k for _, k in CORPUS_KERNELS}
    actual_names = set(per_kernel.keys())
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing:
        errors.append(f"SLoC (CHAR-01): missing kernels: {sorted(missing)}")
    if extra:
        errors.append(f"SLoC (CHAR-01): unexpected kernels: {sorted(extra)}")

    # -- Verify each kernel has cuda_sloc > 0
    for kname, kdata in per_kernel.items():
        sloc_val = kdata.get("cuda_sloc", 0)
        if not sloc_val or sloc_val <= 0:
            errors.append(f"SLoC (CHAR-01): kernel '{kname}' has cuda_sloc={sloc_val}")

    # -- Verify summary stats
    summary = sloc_section.get("summary", {})
    for stat_key in ["min_sloc", "max_sloc", "mean_sloc", "median_sloc"]:
        if stat_key not in summary:
            errors.append(f"SLoC (CHAR-01): summary missing '{stat_key}'")

    if all(k in summary for k in ["min_sloc", "max_sloc", "median_sloc"]):
        if not (summary["min_sloc"] <= summary["median_sloc"] <= summary["max_sloc"]):
            errors.append(
                f"SLoC (CHAR-01): stats ordering violated: "
                f"min={summary['min_sloc']}, median={summary['median_sloc']}, "
                f"max={summary['max_sloc']}"
            )

    # -- Verify OMP/CUDA ratio present for at least some kernels (D-04)
    ratio_count = sum(
        1 for kdata in per_kernel.values()
        if kdata.get("omp_cuda_ratio") is not None
    )
    if ratio_count < 10:
        errors.append(
            f"SLoC (CHAR-01): only {ratio_count} kernels have omp_cuda_ratio "
            f"(expected >=10)"
        )

    return errors


# ---------------------------------------------------------------------------
# Check 3: Categories (CHAR-02)
# ---------------------------------------------------------------------------
def check_categories(data: dict, manifest_entries: list[dict]) -> list[str]:
    """Validate category distribution against independent manifest scan."""
    errors: list[str] = []

    # -- Independently compute categories from manifest
    cat_kernels: dict[str, set[str]] = defaultdict(set)
    cat_suite_kernels: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for entry in manifest_entries:
        category = entry.get("category", "other")
        kernel_name = entry.get("kernel_name", "")
        suite = entry.get("source_suite", "")
        if kernel_name:
            cat_kernels[category].add(kernel_name)
            cat_suite_kernels[category][suite].add(kernel_name)

    # Build expected structure
    expected: dict[str, dict] = {}
    for category in sorted(cat_kernels.keys()):
        suites: dict[str, int] = {}
        for suite_name in sorted(cat_suite_kernels[category].keys()):
            suites[suite_name] = len(cat_suite_kernels[category][suite_name])
        expected[category] = {
            "kernel_count": len(cat_kernels[category]),
            "suites": suites,
        }

    # -- Compare against characterization data
    actual_cats = data.get("categories", {})

    # Verify category count
    if len(actual_cats) != len(expected):
        errors.append(
            f"Categories (CHAR-02): expected {len(expected)} categories, "
            f"got {len(actual_cats)}"
        )

    # Verify each category
    for cat_name, exp_data in expected.items():
        if cat_name not in actual_cats:
            errors.append(f"Categories (CHAR-02): missing category '{cat_name}'")
            continue

        act_data = actual_cats[cat_name]

        # kernel_count
        exp_count = exp_data["kernel_count"]
        act_count = act_data.get("kernel_count", -1)
        if act_count != exp_count:
            errors.append(
                f"Categories (CHAR-02): '{cat_name}' kernel_count "
                f"expected {exp_count}, got {act_count}"
            )

        # suites sub-dict must exist (D-05)
        if "suites" not in act_data:
            errors.append(
                f"Categories (CHAR-02): '{cat_name}' missing 'suites' sub-dict"
            )
        else:
            # Compare suite breakdown
            for suite_name, exp_suite_count in exp_data["suites"].items():
                act_suite_count = act_data["suites"].get(suite_name, 0)
                if act_suite_count != exp_suite_count:
                    errors.append(
                        f"Categories (CHAR-02): '{cat_name}'.suites.{suite_name} "
                        f"expected {exp_suite_count}, got {act_suite_count}"
                    )

    # Check for unexpected categories in actual
    for cat_name in actual_cats:
        if cat_name not in expected:
            errors.append(
                f"Categories (CHAR-02): unexpected category '{cat_name}'"
            )

    return errors


# ---------------------------------------------------------------------------
# Check 4: API Coverage (CHAR-03)
# ---------------------------------------------------------------------------
def check_api_coverage(data: dict, manifest_entries: list[dict]) -> list[str]:
    """Validate API coverage matrix against independent manifest scan."""
    errors: list[str] = []

    apis = ["cuda", "omp", "opencl", "omp_target"]
    suite_order = ["rodinia", "hecbench", "xsbench", "rsbench", "mixbench"]

    # -- Independently compute: suite -> api -> set of kernel names
    matrix: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in manifest_entries:
        suite = entry.get("source_suite", "")
        api = entry.get("parallel_api", "")
        kernel = entry.get("kernel_name", "")
        if suite and api and kernel:
            matrix[suite][api].add(kernel)

    # -- Compare cell by cell
    actual_suites = data.get("api_coverage", {}).get("suites", {})
    actual_totals = data.get("api_coverage", {}).get("totals", {})

    for suite in suite_order:
        if suite not in actual_suites:
            errors.append(f"API coverage (CHAR-03): missing suite '{suite}'")
            continue

        for api in apis:
            expected_count = len(matrix[suite][api])
            actual_count = actual_suites[suite].get(api, -1)
            if actual_count != expected_count:
                errors.append(
                    f"API coverage (CHAR-03): [{suite}][{api}] "
                    f"expected {expected_count}, got {actual_count}"
                )

    # -- Verify totals are correct column sums
    for api in apis:
        expected_total = sum(len(matrix[s][api]) for s in suite_order)
        actual_total = actual_totals.get(api, -1)
        if actual_total != expected_total:
            errors.append(
                f"API coverage (CHAR-03): totals[{api}] "
                f"expected {expected_total}, got {actual_total}"
            )

    # -- Verify 4 API columns present (D-06)
    actual_api_list = data.get("api_coverage", {}).get("apis", [])
    if set(actual_api_list) != set(apis):
        errors.append(
            f"API coverage (CHAR-03): expected APIs {apis}, got {actual_api_list}"
        )

    # -- Verify grand total
    expected_grand_total = sum(
        len(matrix[s][a]) for s in suite_order for a in apis
    )
    actual_grand_total = actual_totals.get("total", -1)
    if actual_grand_total != expected_grand_total:
        errors.append(
            f"API coverage (CHAR-03): grand total "
            f"expected {expected_grand_total}, got {actual_grand_total}"
        )

    return errors


# ---------------------------------------------------------------------------
# Check 5: Multi-file (CHAR-04)
# ---------------------------------------------------------------------------
def check_multi_file(data: dict, project_root: Path) -> list[str]:
    """Validate multi-file classification against independent spec reads."""
    errors: list[str] = []
    specs_dir = project_root / "specs"
    mf_section = data.get("multi_file", {})
    by_kernel = mf_section.get("by_kernel", {})
    aggregate = mf_section.get("aggregate", {})

    # -- Independently check each corpus kernel's CUDA spec
    independent_multi = 0
    independent_single = 0

    for suite, kernel in CORPUS_KERNELS:
        # Find the CUDA spec (primary reference)
        cuda_spec_path = specs_dir / f"{suite}-{kernel}-cuda.json"
        if not cuda_spec_path.exists():
            # Skip kernels without a CUDA spec (should not happen for corpus)
            errors.append(
                f"Multi-file (CHAR-04): CUDA spec not found for {suite}-{kernel}"
            )
            continue

        with open(cuda_spec_path, encoding="utf-8") as f:
            spec = json.load(f)

        files_section = spec.get("files", {})
        payload = files_section.get("prompt_payload", [])
        targets = files_section.get("translation_targets", [])

        # Determine multi-file classification: based on translation_targets count
        is_multi = len(targets) > 1
        if is_multi:
            independent_multi += 1
        else:
            independent_single += 1

        # -- Compare against characterization data
        if kernel not in by_kernel:
            errors.append(
                f"Multi-file (CHAR-04): kernel '{kernel}' missing from by_kernel"
            )
            continue

        kdata = by_kernel[kernel]

        # Check headline values
        headline_multi = kdata.get("headline_multi_file")
        if headline_multi != is_multi:
            errors.append(
                f"Multi-file (CHAR-04): '{kernel}' headline_multi_file "
                f"expected {is_multi}, got {headline_multi}"
            )

        headline_payload = kdata.get("headline_payload_count")
        if headline_payload != len(payload):
            errors.append(
                f"Multi-file (CHAR-04): '{kernel}' headline_payload_count "
                f"expected {len(payload)}, got {headline_payload}"
            )

        headline_targets = kdata.get("headline_target_count")
        if headline_targets != len(targets):
            errors.append(
                f"Multi-file (CHAR-04): '{kernel}' headline_target_count "
                f"expected {len(targets)}, got {headline_targets}"
            )

        # Check CUDA-specific API data if present
        apis_data = kdata.get("apis", {})
        if "cuda" in apis_data:
            cuda_api = apis_data["cuda"]
            if cuda_api.get("prompt_payload_count") != len(payload):
                errors.append(
                    f"Multi-file (CHAR-04): '{kernel}' apis.cuda.prompt_payload_count "
                    f"expected {len(payload)}, got {cuda_api.get('prompt_payload_count')}"
                )
            if cuda_api.get("translation_targets_count") != len(targets):
                errors.append(
                    f"Multi-file (CHAR-04): '{kernel}' apis.cuda.translation_targets_count "
                    f"expected {len(targets)}, got {cuda_api.get('translation_targets_count')}"
                )

    # -- Verify aggregate totals
    exp_total = independent_multi + independent_single
    act_single = aggregate.get("single_file_count", -1)
    act_multi = aggregate.get("multi_file_count", -1)
    act_total = aggregate.get("total", -1)

    if act_single != independent_single:
        errors.append(
            f"Multi-file (CHAR-04): aggregate single_file_count "
            f"expected {independent_single}, got {act_single}"
        )
    if act_multi != independent_multi:
        errors.append(
            f"Multi-file (CHAR-04): aggregate multi_file_count "
            f"expected {independent_multi}, got {act_multi}"
        )
    if act_total != exp_total:
        errors.append(
            f"Multi-file (CHAR-04): aggregate total "
            f"expected {exp_total}, got {act_total}"
        )
    if act_single + act_multi != act_total:
        errors.append(
            f"Multi-file (CHAR-04): single+multi ({act_single}+{act_multi}) "
            f"!= total ({act_total})"
        )

    return errors


# ---------------------------------------------------------------------------
# Check 6: Language Features (CHAR-05)
# ---------------------------------------------------------------------------
VALID_TIERS = {
    "cuda_basic", "cuda_library", "cuda_9plus",
    "omp_basic", "omp_4.5", "omp_target",
    "opencl_1x", "opencl_2x",
    "undetected",
}


def check_language_features(data: dict) -> list[str]:
    """Validate language feature tiers."""
    errors: list[str] = []
    lf_section = data.get("language_features", {})
    per_kernel = lf_section.get("per_kernel", {})

    if not per_kernel:
        errors.append("Language features (CHAR-05): per_kernel is empty")
        return errors

    # -- Verify at least 25 kernels have entries
    if len(per_kernel) < 25:
        errors.append(
            f"Language features (CHAR-05): only {len(per_kernel)} kernels "
            f"have entries (expected >=25)"
        )

    # -- Verify all tier values are valid
    for kname, kdata in per_kernel.items():
        overall = kdata.get("overall_tier")
        if overall not in VALID_TIERS:
            errors.append(
                f"Language features (CHAR-05): kernel '{kname}' has invalid "
                f"overall_tier='{overall}' (valid: {sorted(VALID_TIERS)})"
            )

        # Check per-API tiers
        apis = kdata.get("apis", {})
        for api_name, api_data in apis.items():
            tier = api_data.get("tier")
            if tier not in VALID_TIERS:
                errors.append(
                    f"Language features (CHAR-05): '{kname}'.apis.{api_name} "
                    f"has invalid tier='{tier}'"
                )

    # -- Verify corpus kernel coverage
    corpus_names = {k for _, k in CORPUS_KERNELS}
    actual_names = set(per_kernel.keys())
    missing = corpus_names - actual_names
    if missing:
        errors.append(
            f"Language features (CHAR-05): corpus kernels missing: {sorted(missing)}"
        )

    return errors


# ---------------------------------------------------------------------------
# Check 7: Language Standards (CHAR-06)
# ---------------------------------------------------------------------------
def check_language_standards(data: dict, project_root: Path) -> list[str]:
    """Validate language standard distribution against independent spec scan."""
    errors: list[str] = []
    specs_dir = project_root / "specs"

    # -- Independently scan ALL spec files
    expected_dist: dict[str, int] = defaultdict(int)
    spec_count = 0

    for spec_path in sorted(specs_dir.glob("*.json")):
        with open(spec_path, encoding="utf-8") as f:
            spec = json.load(f)
        lang_std = (spec.get("implementation") or {}).get("language_standard")
        if lang_std is None:
            lang_std = "unspecified"
        expected_dist[lang_std] += 1
        spec_count += 1

    # -- Compare against characterization data
    ls_section = data.get("language_standards", {})
    actual_dist = ls_section.get("distribution", {})
    actual_total = ls_section.get("total_specs", -1)

    # Verify total sums to expected spec count
    actual_dist_sum = sum(actual_dist.values())
    if actual_dist_sum != spec_count:
        errors.append(
            f"Language standards (CHAR-06): distribution sums to "
            f"{actual_dist_sum}, expected {spec_count} (spec file count)"
        )

    if actual_total != spec_count:
        errors.append(
            f"Language standards (CHAR-06): total_specs={actual_total}, "
            f"expected {spec_count}"
        )

    # Compare per-standard counts
    all_standards = set(expected_dist.keys()) | set(actual_dist.keys())
    for std in sorted(all_standards):
        exp = expected_dist.get(std, 0)
        act = actual_dist.get(std, 0)
        if exp != act:
            errors.append(
                f"Language standards (CHAR-06): '{std}' "
                f"expected {exp}, got {act}"
            )

    return errors


# ---------------------------------------------------------------------------
# Check 8: Cross-metric consistency
# ---------------------------------------------------------------------------
def check_cross_metric(data: dict) -> list[str]:
    """Verify consistency between different sections of the characterization."""
    errors: list[str] = []

    sloc_count = len(data.get("sloc", {}).get("per_kernel", {}))
    mf_count = len(data.get("multi_file", {}).get("by_kernel", {}))
    lf_count = len(data.get("language_features", {}).get("per_kernel", {}))

    # -- SLoC kernel count == multi_file kernel count
    if sloc_count != mf_count:
        errors.append(
            f"Cross-metric: sloc kernel count ({sloc_count}) "
            f"!= multi_file kernel count ({mf_count})"
        )

    # -- SLoC kernel count == language_features kernel count
    if sloc_count != lf_count:
        errors.append(
            f"Cross-metric: sloc kernel count ({sloc_count}) "
            f"!= language_features kernel count ({lf_count})"
        )

    # -- Category kernel count total covers a reasonable range
    cats = data.get("categories", {})
    # Unique kernels across all categories (some appear in multiple)
    all_cat_kernels: set[str] = set()
    for cat_data in cats.values():
        all_cat_kernels.update(cat_data.get("kernels", []))
    # Must cover at least the 35 corpus kernels
    corpus_names = {k for _, k in CORPUS_KERNELS}
    missing_from_cats = corpus_names - all_cat_kernels
    if missing_from_cats:
        errors.append(
            f"Cross-metric: corpus kernels missing from categories: "
            f"{sorted(missing_from_cats)}"
        )

    # -- API coverage total specs should match language_standards total
    api_total = data.get("api_coverage", {}).get("totals", {}).get("total", -1)
    ls_total = data.get("language_standards", {}).get("total_specs", -1)
    if api_total != ls_total:
        errors.append(
            f"Cross-metric: api_coverage total ({api_total}) "
            f"!= language_standards total_specs ({ls_total})"
        )

    # -- Summary section consistency
    summary = data.get("summary", {})
    if summary.get("total_kernels") != sloc_count:
        errors.append(
            f"Cross-metric: summary.total_kernels ({summary.get('total_kernels')}) "
            f"!= sloc kernel count ({sloc_count})"
        )
    if summary.get("total_specs") != api_total:
        errors.append(
            f"Cross-metric: summary.total_specs ({summary.get('total_specs')}) "
            f"!= api_coverage total ({api_total})"
        )
    if summary.get("total_categories") != len(cats):
        errors.append(
            f"Cross-metric: summary.total_categories "
            f"({summary.get('total_categories')}) != category count ({len(cats)})"
        )

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent cross-validation of benchmark characterization data"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to the parbench project root",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    print("=" * 60)
    print("  Benchmark Characterization Validation")
    print("=" * 60)
    print()

    # Load data
    data = load_characterization(project_root)
    manifest_entries = load_valid_manifest(project_root)
    print(f"Loaded characterization JSON ({len(data)} top-level keys)")
    print(f"Loaded manifest ({len(manifest_entries)} valid entries)")
    print()

    # Run all checks
    checks: list[tuple[str, list[str]]] = [
        ("Schema check", check_schema(data)),
        ("SLoC (CHAR-01)", check_sloc(data, project_root)),
        ("Categories (CHAR-02)", check_categories(data, manifest_entries)),
        ("API coverage (CHAR-03)", check_api_coverage(data, manifest_entries)),
        ("Multi-file (CHAR-04)", check_multi_file(data, project_root)),
        ("Language features (CHAR-05)", check_language_features(data)),
        ("Language standards (CHAR-06)", check_language_standards(data, project_root)),
        ("Cross-metric consistency", check_cross_metric(data)),
    ]

    # Print results
    passed = 0
    failed = 0
    report_lines: list[str] = []
    report_lines.append("=" * 60)
    report_lines.append("  Benchmark Characterization Validation Report")
    report_lines.append(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append("=" * 60)
    report_lines.append("")

    for name, errs in checks:
        if not errs:
            status_line = f"[PASS] {name}"
            print(status_line)
            report_lines.append(status_line)
            passed += 1
        else:
            status_line = f"[FAIL] {name} ({len(errs)} error(s))"
            print(status_line)
            report_lines.append(status_line)
            for err in errs:
                detail = f"  - {err}"
                print(detail)
                report_lines.append(detail)
            failed += 1

    print()
    summary_line = f"RESULT: {passed}/{passed + failed} checks passed"
    if failed > 0:
        summary_line += f", {failed} FAILED"
    print(summary_line)
    report_lines.append("")
    report_lines.append(summary_line)

    # Write validation report
    report_path = project_root / "results" / "analysis" / "characterization_validation.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport written to: {report_path}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
