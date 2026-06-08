#!/usr/bin/env python3
"""
scripts/analysis/benchmark_characterization.py

Complete benchmark characterization for the SC26 ParBench paper.
Computes 6 metrics (CHAR-01 through CHAR-06) from on-disk spec files,
manifest.jsonl, and benchmark source directories.

Metrics:
  CHAR-01: SLoC analysis (per-kernel, aggregate stats, OMP/CUDA ratio)
  CHAR-02: Domain category distribution (12 categories, suite annotations)
  CHAR-03: API coverage cross-tab (5 suites x 4 APIs)
  CHAR-04: Multi-file translation breakdown (prompt_payload vs translation_targets)
  CHAR-05: Language feature tiers (CUDA/OMP/OpenCL feature grep)
  CHAR-06: Language standard distribution (from spec JSONs)

Output: results/analysis/benchmark_characterization.json + .md

Usage:
    python3 scripts/analysis/benchmark_characterization.py \\
        --project-root .
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Import from sloc_analysis (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sloc_analysis import CORPUS_KERNELS, count_physical_sloc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# CHAR-01: SLoC Analysis
# ---------------------------------------------------------------------------
def compute_sloc(project_root: Path) -> dict:
    """Compute SLoC for all 35 corpus kernels using CUDA spec source files.

    Also computes OMP SLoC and OMP/CUDA ratio where OMP spec exists.
    """
    specs_dir = project_root / "specs"
    per_kernel: dict[str, dict] = {}
    cuda_slocs: list[int] = []
    ratios: list[float] = []

    for suite, kernel in CORPUS_KERNELS:
        cuda_spec_path = specs_dir / f"{suite}-{kernel}-cuda.json"
        if not cuda_spec_path.exists():
            print(f"  WARNING: CUDA spec not found: {cuda_spec_path}", file=sys.stderr)
            continue

        cuda_spec = json.loads(cuda_spec_path.read_text(encoding="utf-8"))
        repo_root = cuda_spec["provenance"]["repo_root"]
        source_path = cuda_spec["provenance"]["source_path"]
        source_dir = project_root / repo_root / source_path
        payload_files = (cuda_spec.get("files") or {}).get("prompt_payload", [])
        target_files = (cuda_spec.get("files") or {}).get("translation_targets", [])

        # Count CUDA SLoC
        cuda_sloc = 0
        file_details = []
        if source_dir.exists():
            for fname in payload_files:
                fpath = source_dir / fname
                if fpath.exists():
                    code = fpath.read_text(encoding="utf-8", errors="replace")
                    sloc = count_physical_sloc(code)
                    file_details.append({"file": fname, "physical_sloc": sloc})
                    cuda_sloc += sloc
        else:
            print(f"  WARNING: source dir not found: {source_dir}", file=sys.stderr)

        cuda_slocs.append(cuda_sloc)

        # Compute OMP SLoC if OMP spec exists
        omp_spec_path = specs_dir / f"{suite}-{kernel}-omp.json"
        omp_sloc = None
        omp_cuda_ratio = None
        if omp_spec_path.exists():
            omp_spec = json.loads(omp_spec_path.read_text(encoding="utf-8"))
            omp_repo = omp_spec["provenance"]["repo_root"]
            omp_src_path = omp_spec["provenance"]["source_path"]
            omp_dir = project_root / omp_repo / omp_src_path
            omp_payload = (omp_spec.get("files") or {}).get("prompt_payload", [])
            if omp_dir.exists():
                omp_total = 0
                for fname in omp_payload:
                    fpath = omp_dir / fname
                    if fpath.exists():
                        code = fpath.read_text(encoding="utf-8", errors="replace")
                        omp_total += count_physical_sloc(code)
                omp_sloc = omp_total
                if cuda_sloc > 0:
                    omp_cuda_ratio = round(omp_total / cuda_sloc, 2)
                    ratios.append(omp_cuda_ratio)

        per_kernel[kernel] = {
            "kernel": kernel,
            "suite": suite,
            "cuda_sloc": cuda_sloc,
            "omp_sloc": omp_sloc,
            "omp_cuda_ratio": omp_cuda_ratio,
            "num_source_files": len(payload_files),
            "num_target_files": len(target_files),
            "file_details": file_details,
        }

    # Aggregate statistics
    n = len(cuda_slocs)
    sorted_slocs = sorted(cuda_slocs)
    mean_sloc = sum(cuda_slocs) / n if n > 0 else 0
    median_sloc = (
        sorted_slocs[n // 2]
        if n % 2 == 1
        else (sorted_slocs[n // 2 - 1] + sorted_slocs[n // 2]) / 2
    )
    std_sloc = math.sqrt(sum((s - mean_sloc) ** 2 for s in cuda_slocs) / n) if n > 0 else 0

    distribution = {
        "<100": sum(1 for s in cuda_slocs if s < 100),
        "100-500": sum(1 for s in cuda_slocs if 100 <= s < 500),
        "500-1000": sum(1 for s in cuda_slocs if 500 <= s < 1000),
        ">1000": sum(1 for s in cuda_slocs if s >= 1000),
    }

    # Ratio stats (only where both CUDA and OMP exist)
    ratio_summary = None
    if ratios:
        sorted_ratios = sorted(ratios)
        rn = len(sorted_ratios)
        ratio_median = (
            sorted_ratios[rn // 2]
            if rn % 2 == 1
            else round((sorted_ratios[rn // 2 - 1] + sorted_ratios[rn // 2]) / 2, 2)
        )
        ratio_summary = {
            "count": rn,
            "min": min(ratios),
            "max": max(ratios),
            "median": ratio_median,
            "mean": round(sum(ratios) / rn, 2),
        }

    summary = {
        "total_kernels": n,
        "min_sloc": min(cuda_slocs) if cuda_slocs else 0,
        "max_sloc": max(cuda_slocs) if cuda_slocs else 0,
        "mean_sloc": round(mean_sloc, 1),
        "median_sloc": median_sloc,
        "std_sloc": round(std_sloc, 1),
        "total_sloc": sum(cuda_slocs),
        "distribution": distribution,
        "omp_cuda_ratio": ratio_summary,
    }

    return {"per_kernel": per_kernel, "summary": summary}


# ---------------------------------------------------------------------------
# CHAR-02: Category Distribution
# ---------------------------------------------------------------------------
def compute_categories(project_root: Path) -> dict:
    """Compute domain category distribution from manifest.jsonl.

    Deduplicates kernel names within each category (a kernel appears in
    multiple API variants). Annotates with suite breakdown.
    """
    manifest_path = project_root / "manifest.jsonl"
    # category -> set of kernel names
    cat_kernels: dict[str, set[str]] = defaultdict(set)
    # category -> suite -> set of kernel names
    cat_suite_kernels: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Filter phantom entries
            spec_file = entry.get("spec_file", "")
            if not (project_root / spec_file).exists():
                continue
            category = entry.get("category", "other")
            kernel_name = entry.get("kernel_name", "")
            suite = entry.get("source_suite", "")
            if kernel_name:
                cat_kernels[category].add(kernel_name)
                cat_suite_kernels[category][suite].add(kernel_name)

    result = {}
    for category in sorted(cat_kernels.keys()):
        suites = {}
        for suite_name in sorted(cat_suite_kernels[category].keys()):
            suites[suite_name] = len(cat_suite_kernels[category][suite_name])
        result[category] = {
            "kernel_count": len(cat_kernels[category]),
            "kernels": sorted(cat_kernels[category]),
            "suites": suites,
        }

    return result


# ---------------------------------------------------------------------------
# CHAR-03: API Coverage Cross-Tab
# ---------------------------------------------------------------------------
def compute_api_coverage(project_root: Path) -> dict:
    """Build suite x API matrix of distinct kernel counts.

    Five suite rows: rodinia, hecbench, xsbench, rsbench, mixbench
    Four API columns: cuda, omp, opencl, omp_target
    """
    manifest_path = project_root / "manifest.jsonl"
    apis = ["cuda", "omp", "opencl", "omp_target"]
    suite_order = ["rodinia", "hecbench", "xsbench", "rsbench", "mixbench"]

    # suite -> api -> set of kernel names
    matrix: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            spec_file = entry.get("spec_file", "")
            if not (project_root / spec_file).exists():
                continue
            suite = entry.get("source_suite", "")
            api = entry.get("parallel_api", "")
            kernel = entry.get("kernel_name", "")
            if suite and api and kernel:
                matrix[suite][api].add(kernel)

    suites_result = {}
    for suite in suite_order:
        row = {}
        for api in apis:
            row[api] = len(matrix[suite][api])
        row["total"] = sum(row[api] for api in apis)
        suites_result[suite] = row

    totals = {}
    for api in apis:
        totals[api] = sum(len(matrix[suite][api]) for suite in suite_order)
    totals["total"] = sum(totals[api] for api in apis)

    return {
        "suites": suites_result,
        "totals": totals,
        "apis": apis,
        "suite_order": suite_order,
    }


# ---------------------------------------------------------------------------
# CHAR-04: Multi-File Translation Breakdown
# ---------------------------------------------------------------------------
def compute_multi_file(project_root: Path) -> dict:
    """Compute multi-file vs single-file breakdown for all 35 corpus kernels.

    Uses translation_targets count to classify: >1 = multi-file.
    Also computes per-API and per-suite aggregates.
    """
    specs_dir = project_root / "specs"
    per_kernel: dict[str, dict] = {}
    api_variants = ["cuda", "omp", "opencl", "omp_target"]

    for suite, kernel in CORPUS_KERNELS:
        kernel_data = {"kernel": kernel, "suite": suite, "apis": {}}

        for api in api_variants:
            spec_path = specs_dir / f"{suite}-{kernel}-{api}.json"
            if not spec_path.exists():
                continue
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            payload = (spec.get("files") or {}).get("prompt_payload", [])
            targets = (spec.get("files") or {}).get("translation_targets", [])
            kernel_data["apis"][api] = {
                "prompt_payload_count": len(payload),
                "translation_targets_count": len(targets),
                "is_multi_file": len(targets) > 1,
            }

        # Use CUDA spec for headline classification
        cuda_info = kernel_data["apis"].get("cuda", {})
        kernel_data["headline_multi_file"] = cuda_info.get("is_multi_file", False)
        kernel_data["headline_target_count"] = cuda_info.get("translation_targets_count", 0)
        kernel_data["headline_payload_count"] = cuda_info.get("prompt_payload_count", 0)

        per_kernel[kernel] = kernel_data

    # Aggregate stats from CUDA specs
    multi = sum(1 for k in per_kernel.values() if k["headline_multi_file"])
    single = sum(1 for k in per_kernel.values() if not k["headline_multi_file"])
    total = multi + single
    multi_pct = round(100 * multi / total, 1) if total > 0 else 0

    # Per-suite breakdown
    by_suite: dict[str, dict] = defaultdict(lambda: {"single": 0, "multi": 0})
    for k in per_kernel.values():
        s = k["suite"]
        if k["headline_multi_file"]:
            by_suite[s]["multi"] += 1
        else:
            by_suite[s]["single"] += 1

    # Per-API aggregate (across all specs, not just corpus kernels)
    # Count from all 206 specs
    api_aggregate: dict[str, dict] = defaultdict(lambda: {"single": 0, "multi": 0, "total": 0})
    for spec_path in sorted(specs_dir.glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        api = (spec.get("identity") or {}).get("parallel_api", "unknown")
        targets = (spec.get("files") or {}).get("translation_targets", [])
        is_multi = len(targets) > 1
        api_aggregate[api]["total"] += 1
        if is_multi:
            api_aggregate[api]["multi"] += 1
        else:
            api_aggregate[api]["single"] += 1

    return {
        "by_kernel": per_kernel,
        "aggregate": {
            "single_file_count": single,
            "multi_file_count": multi,
            "total": total,
            "multi_file_pct": multi_pct,
        },
        "by_suite": dict(by_suite),
        "by_api": dict(api_aggregate),
    }


# ---------------------------------------------------------------------------
# CHAR-05: Language Feature Tiers
# ---------------------------------------------------------------------------
def compute_language_features(project_root: Path) -> dict:
    """Grep source directories for language feature indicators.

    Assigns a tier to each kernel based on the highest-level feature found.
    """
    specs_dir = project_root / "specs"

    # Feature patterns by API
    cuda_patterns = {
        "cuda_basic": [
            r"__global__",
            r"__shared__",
            r"__device__",
        ],
        "cuda_library": [
            r"thrust::",
            r"cub::",
        ],
        "cuda_9plus": [
            r"cooperative_groups",
        ],
    }

    omp_patterns = {
        "omp_basic": [
            r"#pragma\s+omp\s+parallel\s+for",
            r"#pragma\s+omp\s+parallel",
        ],
        "omp_4.5": [
            r"#pragma\s+omp\s+simd",
            r"#pragma\s+omp\s+taskloop",
        ],
        "omp_target": [
            r"#pragma\s+omp\s+target",
        ],
    }

    opencl_patterns = {
        "opencl_1x": [
            r"clEnqueueNDRangeKernel",
            r"clCreateBuffer",
        ],
        "opencl_2x": [
            r"clSVMAlloc",
            r"pipe\s+",
        ],
    }

    # Tier ordering: higher index = more advanced
    cuda_tier_order = ["cuda_basic", "cuda_library", "cuda_9plus"]
    omp_tier_order = ["omp_basic", "omp_4.5", "omp_target"]
    opencl_tier_order = ["opencl_1x", "opencl_2x"]

    source_extensions = ("*.cu", "*.c", "*.cpp", "*.h", "*.hpp", "*.cl")

    def grep_dir(directory: Path, patterns: dict[str, list[str]]) -> dict[str, list[str]]:
        """Grep all source files in a directory for patterns."""
        found: dict[str, list[str]] = defaultdict(list)
        if not directory.exists():
            return dict(found)

        for ext in source_extensions:
            for fpath in directory.rglob(ext):
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for tier, pats in patterns.items():
                    for pat in pats:
                        if re.search(pat, content):
                            if pat not in found[tier]:
                                found[tier].append(pat)
        return dict(found)

    def highest_tier(found: dict[str, list[str]], tier_order: list[str]) -> str:
        """Return the highest tier that has matches, or 'undetected'."""
        best = None
        for tier in tier_order:
            if tier in found and found[tier]:
                best = tier
        return best if best is not None else "undetected"

    per_kernel: dict[str, dict] = {}

    for suite, kernel in CORPUS_KERNELS:
        entry: dict = {"kernel": kernel, "suite": suite, "apis": {}}
        source_dir_found = False

        # CUDA features
        cuda_spec_path = specs_dir / f"{suite}-{kernel}-cuda.json"
        if cuda_spec_path.exists():
            cuda_spec = json.loads(cuda_spec_path.read_text(encoding="utf-8"))
            repo_root = cuda_spec["provenance"]["repo_root"]
            src_path = cuda_spec["provenance"]["source_path"]
            cuda_dir = project_root / repo_root / src_path
            if cuda_dir.exists():
                source_dir_found = True
                cuda_found = grep_dir(cuda_dir, cuda_patterns)
                entry["apis"]["cuda"] = {
                    "features_found": cuda_found,
                    "tier": highest_tier(cuda_found, cuda_tier_order),
                }

        # OMP features (check both omp and omp_target specs)
        for omp_api in ["omp", "omp_target"]:
            omp_spec_path = specs_dir / f"{suite}-{kernel}-{omp_api}.json"
            if omp_spec_path.exists():
                omp_spec = json.loads(omp_spec_path.read_text(encoding="utf-8"))
                repo_root = omp_spec["provenance"]["repo_root"]
                src_path = omp_spec["provenance"]["source_path"]
                omp_dir = project_root / repo_root / src_path
                if omp_dir.exists():
                    source_dir_found = True
                    omp_found = grep_dir(omp_dir, omp_patterns)
                    entry["apis"][omp_api] = {
                        "features_found": omp_found,
                        "tier": highest_tier(omp_found, omp_tier_order),
                    }

        # OpenCL features
        opencl_spec_path = specs_dir / f"{suite}-{kernel}-opencl.json"
        if opencl_spec_path.exists():
            opencl_spec = json.loads(opencl_spec_path.read_text(encoding="utf-8"))
            repo_root = opencl_spec["provenance"]["repo_root"]
            src_path = opencl_spec["provenance"]["source_path"]
            opencl_dir = project_root / repo_root / src_path
            if opencl_dir.exists():
                source_dir_found = True
                opencl_found = grep_dir(opencl_dir, opencl_patterns)
                entry["apis"]["opencl"] = {
                    "features_found": opencl_found,
                    "tier": highest_tier(opencl_found, opencl_tier_order),
                }

        if not source_dir_found:
            print(
                f"  WARNING: no source directory found for {suite}-{kernel}",
                file=sys.stderr,
            )

        # Per-API max tiers (independent per API family — no cross-family comparison)
        per_api_max = {}
        for api_name, api_data in entry["apis"].items():
            t = api_data.get("tier")
            if t and t != "undetected":
                per_api_max[api_name] = t
        entry["per_api_max_tiers"] = per_api_max if per_api_max else {"note": "no features detected"}
        # Overall tier: highest within each API family, reported as the most complex single tier found
        # Uses per-family ordering to avoid arbitrary cross-family comparisons
        best_tier = "undetected"
        best_rank = -1
        for api_name, tier_val in per_api_max.items():
            if "cuda" in api_name:
                order = cuda_tier_order
            elif "omp" in api_name:
                order = omp_tier_order
            elif "opencl" in api_name:
                order = opencl_tier_order
            else:
                continue
            rank = order.index(tier_val) if tier_val in order else -1
            if rank > best_rank:
                best_rank = rank
                best_tier = tier_val
        entry["overall_tier"] = best_tier

        per_kernel[kernel] = entry

    # Tier distribution
    tier_dist: dict[str, int] = defaultdict(int)
    for kdata in per_kernel.values():
        for api_data in kdata["apis"].values():
            t = api_data.get("tier")
            if t:
                tier_dist[t] += 1

    return {
        "per_kernel": per_kernel,
        "tier_distribution": dict(tier_dist),
    }


# ---------------------------------------------------------------------------
# CHAR-06: Language Standard Distribution
# ---------------------------------------------------------------------------
def compute_language_standards(project_root: Path) -> dict:
    """Extract language_standard distribution from all 206 spec JSONs.

    Breaks down by API and by suite.
    """
    specs_dir = project_root / "specs"

    distribution: dict[str, int] = defaultdict(int)
    by_api: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_suite: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_specs = 0
    for spec_path in sorted(specs_dir.glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        lang_std = (spec.get("implementation") or {}).get("language_standard") or "unspecified"
        api = (spec.get("identity") or {}).get("parallel_api", "unknown")
        suite = (spec.get("identity") or {}).get("source_suite", "unknown")

        distribution[lang_std] += 1
        by_api[api][lang_std] += 1
        by_suite[suite][lang_std] += 1
        total_specs += 1

    return {
        "distribution": dict(distribution),
        "by_api": {k: dict(v) for k, v in by_api.items()},
        "by_suite": {k: dict(v) for k, v in by_suite.items()},
        "total_specs": total_specs,
    }


# ---------------------------------------------------------------------------
# Markdown Report Generation
# ---------------------------------------------------------------------------
def generate_markdown(data: dict) -> str:
    """Generate human-readable markdown tables from characterization data."""
    lines = [
        "# ParBench Benchmark Characterization",
        "",
        f"Generated: {data['metadata']['generated']}",
        "",
    ]

    # --- SLoC Section ---
    lines.extend([
        "## 1. Source Lines of Code (SLoC)",
        "",
        "Physical SLoC of CUDA source files (prompt_payload) per corpus kernel.",
        "",
        "| Kernel | Suite | CUDA SLoC | OMP SLoC | Ratio | Src Files | Tgt Files |",
        "|--------|-------|----------:|---------:|------:|----------:|----------:|",
    ])
    sloc = data["sloc"]
    for kernel in sorted(sloc["per_kernel"].values(), key=lambda x: x["cuda_sloc"], reverse=True):
        omp_str = str(kernel["omp_sloc"]) if kernel["omp_sloc"] is not None else "N/A"
        ratio_str = str(kernel["omp_cuda_ratio"]) if kernel["omp_cuda_ratio"] is not None else "N/A"
        lines.append(
            f"| {kernel['kernel']} | {kernel['suite']} | {kernel['cuda_sloc']:,} | "
            f"{omp_str} | {ratio_str} | {kernel['num_source_files']} | {kernel['num_target_files']} |"
        )

    s = sloc["summary"]
    lines.extend([
        "",
        "### SLoC Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total Kernels | {s['total_kernels']} |",
        f"| Min SLoC | {s['min_sloc']:,} |",
        f"| Max SLoC | {s['max_sloc']:,} |",
        f"| Mean SLoC | {s['mean_sloc']:,.1f} |",
        f"| Median SLoC | {s['median_sloc']:,.0f} |",
        f"| Std Dev | {s['std_sloc']:,.1f} |",
        f"| Total SLoC | {s['total_sloc']:,} |",
        "",
        "### SLoC Distribution",
        "",
        "| Range | Count |",
        "|-------|------:|",
    ])
    for bucket, count in s["distribution"].items():
        lines.append(f"| {bucket} | {count} |")

    if s["omp_cuda_ratio"]:
        r = s["omp_cuda_ratio"]
        lines.extend([
            "",
            "### OMP/CUDA SLoC Ratio",
            "",
            f"Computed over {r['count']} kernels with both CUDA and OMP specs.",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Min Ratio | {r['min']} |",
            f"| Max Ratio | {r['max']} |",
            f"| Median Ratio | {r['median']} |",
            f"| Mean Ratio | {r['mean']} |",
        ])

    # --- Categories Section ---
    lines.extend([
        "",
        "## 2. Domain Category Distribution",
        "",
        "| Category | Kernels | Suites |",
        "|----------|--------:|--------|",
    ])
    categories = data["categories"]
    for cat in sorted(categories.keys()):
        cdata = categories[cat]
        suite_str = ", ".join(f"{s}({n})" for s, n in sorted(cdata["suites"].items()))
        lines.append(f"| {cat} | {cdata['kernel_count']} | {suite_str} |")

    total_cat_kernels = sum(c["kernel_count"] for c in categories.values())
    lines.append(f"| **Total** | **{total_cat_kernels}** | |")

    # --- API Coverage Section ---
    lines.extend([
        "",
        "## 3. API Coverage Cross-Tab",
        "",
        "Distinct kernel count per (suite, API) cell.",
        "",
    ])
    api_cov = data["api_coverage"]
    apis = api_cov["apis"]
    header = "| Suite | " + " | ".join(apis) + " | Total |"
    sep = "|-------|" + "|".join(["------:" for _ in apis]) + "|------:|"
    lines.extend([header, sep])
    for suite in api_cov["suite_order"]:
        row = api_cov["suites"][suite]
        cells = " | ".join(str(row[a]) for a in apis)
        lines.append(f"| {suite} | {cells} | {row['total']} |")
    totals = api_cov["totals"]
    cells = " | ".join(str(totals[a]) for a in apis)
    lines.append(f"| **Total** | {cells} | {totals['total']} |")

    # --- Multi-File Section ---
    lines.extend([
        "",
        "## 4. Multi-File Translation Breakdown",
        "",
        "Classification based on `translation_targets` count (>1 = multi-file).",
        "",
    ])
    mf = data["multi_file"]
    agg = mf["aggregate"]
    lines.extend([
        f"- **Single-file:** {agg['single_file_count']} kernels",
        f"- **Multi-file:** {agg['multi_file_count']} kernels ({agg['multi_file_pct']}%)",
        "",
        "### Per-Kernel Detail",
        "",
        "| Kernel | Suite | Payload Files | Target Files | Multi-File |",
        "|--------|-------|-------------:|-------------:|:----------:|",
    ])
    for kernel in sorted(mf["by_kernel"].values(), key=lambda x: x["headline_target_count"], reverse=True):
        mf_str = "Yes" if kernel["headline_multi_file"] else "No"
        lines.append(
            f"| {kernel['kernel']} | {kernel['suite']} | "
            f"{kernel['headline_payload_count']} | {kernel['headline_target_count']} | {mf_str} |"
        )

    # By suite
    lines.extend([
        "",
        "### By Suite",
        "",
        "| Suite | Single | Multi |",
        "|-------|-------:|------:|",
    ])
    for suite_name in sorted(mf["by_suite"].keys()):
        sd = mf["by_suite"][suite_name]
        lines.append(f"| {suite_name} | {sd['single']} | {sd['multi']} |")

    # --- Language Features Section ---
    lines.extend([
        "",
        "## 5. Language Feature Tiers",
        "",
        "Features detected via regex grep of source directories.",
        "",
        "| Kernel | Suite | CUDA Tier | OMP Tier | OpenCL Tier |",
        "|--------|-------|-----------|----------|-------------|",
    ])
    lf = data["language_features"]
    for kernel in sorted(lf["per_kernel"].values(), key=lambda x: x["kernel"]):
        cuda_tier = (kernel["apis"].get("cuda") or {}).get("tier", "N/A") or "N/A"
        omp_tier = (kernel["apis"].get("omp") or kernel["apis"].get("omp_target") or {}).get("tier", "N/A") or "N/A"
        opencl_tier = (kernel["apis"].get("opencl") or {}).get("tier", "N/A") or "N/A"
        lines.append(
            f"| {kernel['kernel']} | {kernel['suite']} | {cuda_tier} | {omp_tier} | {opencl_tier} |"
        )

    lines.extend([
        "",
        "### Tier Distribution (across all API variants)",
        "",
        "| Tier | Count |",
        "|------|------:|",
    ])
    for tier, count in sorted(lf["tier_distribution"].items()):
        lines.append(f"| {tier} | {count} |")

    # --- Language Standards Section ---
    lines.extend([
        "",
        "## 6. Language Standard Distribution",
        "",
        "Extracted from `implementation.language_standard` across all spec files.",
        "",
        "| Standard | Count | Percentage |",
        "|----------|------:|-----------:|",
    ])
    ls = data["language_standards"]
    total_ls = ls["total_specs"]
    for std in sorted(ls["distribution"].keys()):
        count = ls["distribution"][std]
        pct = round(100 * count / total_ls, 1) if total_ls > 0 else 0
        lines.append(f"| {std} | {count} | {pct}% |")
    lines.append(f"| **Total** | **{total_ls}** | **100.0%** |")

    # By API
    lines.extend([
        "",
        "### By API",
        "",
    ])
    for api in sorted(ls["by_api"].keys()):
        lines.append(f"**{api}:**")
        for std, count in sorted(ls["by_api"][api].items()):
            lines.append(f"  - {std}: {count}")
        lines.append("")

    # --- Summary Section ---
    lines.extend([
        "## Summary",
        "",
    ])
    summary = data["summary"]
    for key, val in summary.items():
        lines.append(f"- **{key}:** {val}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="ParBench Benchmark Characterization (CHAR-01 through CHAR-06)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to parbench_sam root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/analysis/)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir or (project_root / "results" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Output dir: {output_dir}")
    print(f"Corpus: {len(CORPUS_KERNELS)} kernels\n")

    # Compute all 6 metrics
    print("Computing CHAR-01: SLoC analysis...")
    sloc = compute_sloc(project_root)
    print(f"  {sloc['summary']['total_kernels']} kernels analyzed\n")

    print("Computing CHAR-02: Category distribution...")
    categories = compute_categories(project_root)
    print(f"  {len(categories)} categories found\n")

    print("Computing CHAR-03: API coverage...")
    api_coverage = compute_api_coverage(project_root)
    print(f"  {len(api_coverage['suites'])} suites x {len(api_coverage['apis'])} APIs\n")

    print("Computing CHAR-04: Multi-file breakdown...")
    multi_file = compute_multi_file(project_root)
    agg = multi_file["aggregate"]
    print(f"  {agg['multi_file_count']}/{agg['total']} multi-file ({agg['multi_file_pct']}%)\n")

    print("Computing CHAR-05: Language features...")
    language_features = compute_language_features(project_root)
    print(f"  {len(language_features['per_kernel'])} kernels scanned\n")

    print("Computing CHAR-06: Language standards...")
    language_standards = compute_language_standards(project_root)
    print(f"  {language_standards['total_specs']} specs, "
          f"{len(language_standards['distribution'])} standards\n")

    # Assemble combined output
    data = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "project_root": str(project_root),
            "script": "benchmark_characterization.py",
            "corpus_size": len(CORPUS_KERNELS),
        },
        "sloc": sloc,
        "categories": categories,
        "api_coverage": api_coverage,
        "multi_file": multi_file,
        "language_features": language_features,
        "language_standards": language_standards,
        "summary": {
            "total_kernels": len(CORPUS_KERNELS),
            "total_specs": language_standards["total_specs"],
            "total_categories": len(categories),
            "total_suites": len(api_coverage["suites"]),
            "total_apis": len(api_coverage["apis"]),
            "sloc_range": f"{sloc['summary']['min_sloc']}-{sloc['summary']['max_sloc']}",
            "sloc_median": sloc["summary"]["median_sloc"],
            "multi_file_pct": multi_file["aggregate"]["multi_file_pct"],
        },
    }

    # Write JSON
    json_path = output_dir / "benchmark_characterization.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    # Write Markdown
    md_path = output_dir / "benchmark_characterization.md"
    md_content = generate_markdown(data)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Wrote {md_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK CHARACTERIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Kernels: {data['summary']['total_kernels']}")
    print(f"Specs: {data['summary']['total_specs']}")
    print(f"Categories: {data['summary']['total_categories']}")
    print(f"Suites: {data['summary']['total_suites']}")
    print(f"APIs: {data['summary']['total_apis']}")
    print(f"SLoC range: {data['summary']['sloc_range']}")
    print(f"SLoC median: {data['summary']['sloc_median']}")
    print(f"Multi-file: {data['summary']['multi_file_pct']}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
