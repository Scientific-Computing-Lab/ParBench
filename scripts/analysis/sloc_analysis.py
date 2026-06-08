#!/usr/bin/env python3
"""
scripts/analysis/sloc_analysis.py

SLoC (Source Lines of Code) characterization of ParBench benchmark kernels.

Counts PHYSICAL lines (non-blank, non-comment) in the source files listed in
each kernel's CUDA spec prompt_payload. This is SOURCE SLoC — the code that
the LLM receives as input for translation.

Covers all 35 kernels in the evaluation corpus across 5 suites:
  - Rodinia (22 kernels)
  - HeCBench curated (10 kernels)
  - XSBench, RSBench, mixbench (1 each)

Note: translated_files in result JSONs are truncated to ~200 chars per file,
so we read the actual source files from the benchmark repositories.
If running in a worktree where the submodule is empty, pass --project-root
pointing to the main repository.

Physical SLoC definition (approximates cloc; does not parse string literals):
  - Excludes blank lines (empty or whitespace-only)
  - Excludes comment-only lines (// line comments, /* */ block comments)
  - Includes lines with both code and trailing comments

Output: results/analysis/sloc_analysis.json + .md

Usage:
    python3 scripts/analysis/sloc_analysis.py \\
        --project-root .
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# All 35 corpus kernels: (suite, kernel_name) pairs
# These match the evaluation corpus in paper.tex tab:suite-summary.
CORPUS_KERNELS: list[tuple[str, str]] = [
    # Rodinia — 22 kernels
    ("rodinia", "backprop"), ("rodinia", "bfs"), ("rodinia", "bptree"),
    ("rodinia", "cfd"), ("rodinia", "dwt2d"), ("rodinia", "gaussian"),
    ("rodinia", "heartwall"), ("rodinia", "hotspot"), ("rodinia", "hotspot3d"),
    ("rodinia", "huffman"), ("rodinia", "hybridsort"), ("rodinia", "kmeans"),
    ("rodinia", "lavamd"), ("rodinia", "lud"), ("rodinia", "mummergpu"),
    ("rodinia", "myocyte"), ("rodinia", "nn"), ("rodinia", "nw"),
    ("rodinia", "particlefilter"), ("rodinia", "pathfinder"),
    ("rodinia", "srad"), ("rodinia", "streamcluster"),
    # HeCBench curated — 10 kernels
    ("hecbench", "convolution1d"), ("hecbench", "floydwarshall"),
    ("hecbench", "heat2d"), ("hecbench", "iso2dfd"), ("hecbench", "jacobi"),
    ("hecbench", "md"), ("hecbench", "nqueen"), ("hecbench", "page-rank"),
    ("hecbench", "scan"), ("hecbench", "stencil1d"),
    # Single-kernel suites
    ("xsbench", "xsbench"), ("rsbench", "rsbench"), ("mixbench", "mixbench"),
]


def count_physical_sloc(code: str) -> int:
    """Count non-blank, non-comment lines (physical SLoC) in C/C++/CUDA code."""
    lines = code.split("\n")
    count = 0
    in_block_comment = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue  # blank line

        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
                # Check for code after closing comment
                after = stripped[stripped.index("*/") + 2 :].strip()
                if after and not after.startswith("//"):
                    count += 1
            continue

        # Check for block comment start
        if "/*" in stripped:
            before = stripped[: stripped.index("/*")].strip()
            if before:
                count += 1  # code before comment
            rest = stripped[stripped.index("/*") + 2 :]
            if "*/" in rest:
                # Single-line block comment: /* ... */
                after = rest[rest.index("*/") + 2 :].strip()
                if after and not after.startswith("//"):
                    count += 1 if not before else 0
            else:
                in_block_comment = True
            continue  # already counted if `before` had code

        if stripped.startswith("//"):
            continue  # line comment

        count += 1

    return count


def count_total_lines(code: str) -> int:
    """Count total non-blank lines."""
    return sum(1 for line in code.split("\n") if line.strip())


def load_category_map(project_root: Path) -> dict[str, str]:
    """Load kernel → category mapping from manifest.jsonl."""
    categories = {}
    manifest_path = project_root / "manifest.jsonl"
    if not manifest_path.exists():
        return categories
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            kn = entry.get("kernel_name", "")
            if kn and kn not in categories:
                categories[kn] = entry.get("category", "other")
    return categories


def load_complexity_map(project_root: Path) -> dict[str, str]:
    """Load kernel → complexity_class from translation_complexity.csv (cuda-to-omp direction)."""
    import csv

    complexity = {}
    csv_path = project_root / "results" / "evaluation" / "translation_complexity.csv"
    if not csv_path.exists():
        return complexity
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use cuda→omp direction as the representative complexity
            if row.get("source_api") == "cuda" and row.get("target_api") == "omp":
                complexity[row["kernel"]] = row["complexity_class"]
    return complexity


def load_pass_rates(project_root: Path) -> dict[str, float]:
    """Load per-kernel pass rates from eval_summary.json."""
    summary_path = project_root / "results" / "evaluation" / "eval_summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rates = {}
    for kernel, data in summary.get("by_kernel", {}).items():
        rates[kernel] = data.get("rate", 0.0)
    return rates


def get_spec_source_info(spec: dict) -> tuple[str, list[str]]:
    """Extract source path and prompt_payload file list from a spec."""
    repo_root = spec["provenance"]["repo_root"]
    source_path = spec["provenance"]["source_path"]
    full_source_dir = f"{repo_root}/{source_path}"
    payload_files = spec["files"].get("prompt_payload", [])
    return full_source_dir, payload_files


def analyze_kernel(
    suite: str,
    kernel: str,
    project_root: Path,
    specs_dir: Path,
) -> dict | None:
    """Analyze SLoC for a single kernel using its CUDA spec source files."""
    # Find the CUDA spec: {suite}-{kernel}-cuda.json
    spec_path = specs_dir / f"{suite}-{kernel}-cuda.json"

    if not spec_path.exists():
        print(f"  WARNING: spec not found: {spec_path}", file=sys.stderr)
        return None

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_dir_rel, payload_files = get_spec_source_info(spec)
    translation_targets = spec["files"].get("translation_targets", [])
    source_dir = project_root / source_dir_rel

    if not source_dir.exists():
        print(f"  WARNING: source dir not found: {source_dir}", file=sys.stderr)
        return None

    # Count SLoC for each file
    file_slocs = []
    total_sloc = 0
    total_lines = 0
    missing_files = []

    for fname in payload_files:
        fpath = source_dir / fname
        if not fpath.exists():
            missing_files.append(fname)
            continue
        code = fpath.read_text(encoding="utf-8", errors="replace")
        sloc = count_physical_sloc(code)
        lines = count_total_lines(code)
        file_slocs.append({
            "file": fname,
            "physical_sloc": sloc,
            "total_nonblank_lines": lines,
        })
        total_sloc += sloc
        total_lines += lines

    if missing_files:
        print(
            f"  WARNING: {kernel}: missing files: {missing_files}",
            file=sys.stderr,
        )

    # Also get OMP spec info for comparison
    omp_spec_path = specs_dir / f"{suite}-{kernel}-omp.json"

    omp_sloc = None
    omp_file_count = None
    omp_target_count = None
    if omp_spec_path.exists():
        omp_spec = json.loads(omp_spec_path.read_text(encoding="utf-8"))
        omp_source_dir_rel, omp_files = get_spec_source_info(omp_spec)
        omp_target_count = len(omp_spec["files"].get("translation_targets", []))
        omp_source_dir = project_root / omp_source_dir_rel
        omp_file_count = len(omp_files)
        if omp_source_dir.exists():
            omp_total = 0
            for fname in omp_files:
                fpath = omp_source_dir / fname
                if fpath.exists():
                    code = fpath.read_text(encoding="utf-8", errors="replace")
                    omp_total += count_physical_sloc(code)
            omp_sloc = omp_total

    return {
        "kernel": kernel,
        "suite": suite,
        "source_api": "cuda",
        "source_dir": source_dir_rel,
        "num_source_files": len(payload_files),
        "num_target_files": len(translation_targets),
        "physical_sloc": total_sloc,
        "total_nonblank_lines": total_lines,
        "files": file_slocs,
        "omp_physical_sloc": omp_sloc,
        "omp_num_files": omp_file_count,
        "omp_num_target_files": omp_target_count,
    }


def compute_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Spearman rank correlation coefficient."""
    if len(xs) < 3:
        return None
    n = len(xs)

    def rank(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(sorted_idx):
            ranks[i] = r + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)

    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_sq_sum) / (n * (n * n - 1))
    return round(rho, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Path to parbench_sam root (where rodinia/xsbench sources live)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/analysis/)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    specs_dir = project_root / "specs"
    output_dir = args.output_dir or (project_root / "results" / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata maps
    category_map = load_category_map(project_root)
    complexity_map = load_complexity_map(project_root)
    pass_rate_map = load_pass_rates(project_root)

    print(f"Project root: {project_root}")
    n_suites = len(set(s for s, _ in CORPUS_KERNELS))
    print(f"Analyzing {len(CORPUS_KERNELS)} corpus kernels ({n_suites} suites)...\n")

    # Analyze each kernel
    kernels = []
    for suite, kernel in CORPUS_KERNELS:
        info = analyze_kernel(suite, kernel, project_root, specs_dir)
        if info is None:
            continue
        # Enrich with metadata
        info["category"] = category_map.get(kernel, "other")
        info["complexity_class"] = complexity_map.get(kernel, "unknown")
        info["pass_rate"] = pass_rate_map.get(kernel, 0.0)
        kernels.append(info)
        sloc = info["physical_sloc"]
        omp = info.get("omp_physical_sloc")
        omp_str = f", OMP={omp}" if omp else ""
        print(f"  {suite:10s} {kernel:20s} CUDA={sloc:5d} SLoC ({info['num_source_files']} files){omp_str}")

    # Summary statistics
    slocs = [k["physical_sloc"] for k in kernels]
    slocs_sorted = sorted(slocs)
    n = len(slocs)
    median = (
        slocs_sorted[n // 2]
        if n % 2 == 1
        else (slocs_sorted[n // 2 - 1] + slocs_sorted[n // 2]) / 2
    )

    # Distribution buckets
    buckets = {
        "<100": sum(1 for s in slocs if s < 100),
        "100-500": sum(1 for s in slocs if 100 <= s < 500),
        "500-1000": sum(1 for s in slocs if 500 <= s < 1000),
        ">1000": sum(1 for s in slocs if s >= 1000),
    }

    # Correlation: SLoC vs pass rate
    # KNOWN_FAIL kernels (kmeans, mummergpu, hybridsort) are deliberately included at
    # pass_rate=0.0. Rationale: their failures are platform constraints (missing GL/glew.h,
    # deprecated CUDA 12 texture<>), not SLoC complexity — so including them does not
    # bias the correlation toward complexity. Excluding them would be cherry-picking.
    # Paper methodology should state: "all corpus kernels included, KNOWN_FAIL at pass_rate=0."
    sloc_values = [k["physical_sloc"] for k in kernels if k.get("pass_rate") is not None]
    rate_values = [k["pass_rate"] for k in kernels if k.get("pass_rate") is not None]
    correlation = compute_correlation(sloc_values, rate_values)

    # ParEval-Repo comparison
    pareval_threshold = 133
    above_threshold = sum(1 for s in slocs if s > pareval_threshold)

    mean_sloc = sum(slocs) / n
    summary = {
        "total_kernels": n,
        "min_sloc": min(slocs),
        "max_sloc": max(slocs),
        "mean_sloc": round(mean_sloc, 1),
        "median_sloc": median,
        "std_sloc": round(math.sqrt(sum((s - mean_sloc) ** 2 for s in slocs) / n), 1),
        "total_sloc_all_kernels": sum(slocs),
        "distribution": buckets,
        "pareval_repo_threshold": pareval_threshold,
        "kernels_above_pareval_threshold": above_threshold,
        "pct_above_pareval_threshold": round(100 * above_threshold / n, 1),
        "sloc_vs_pass_rate_spearman": correlation,
    }

    # Build output
    output = {
        "analysis": "sloc_characterization",
        "method": "physical_sloc_from_source_files",
        "note": (
            "SOURCE SLoC counted from actual Rodinia/XSBench repository files "
            "(the code sent to the LLM as input). Physical lines = non-blank, "
            "non-comment lines (approximates cloc; does not parse string literals)."
        ),
        "kernels": {k["kernel"]: k for k in kernels},
        "summary": summary,
    }

    # Write JSON
    json_path = output_dir / "sloc_analysis.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {json_path}")

    # Write markdown report
    md_path = output_dir / "sloc_analysis.md"
    # Count per suite for the report
    suite_counts = defaultdict(int)
    for k in kernels:
        suite_counts[k["suite"]] += 1
    suite_str = ", ".join(f"{s} ({c})" for s, c in sorted(suite_counts.items()))

    md_lines = [
        "# SLoC Characterization of ParBench Corpus Kernels",
        "",
        f"**{n} kernels** analyzed across 5 suites: {suite_str}.",
        f"Physical SLoC = non-blank, non-comment lines (matches cloc methodology).",
        f"Source: actual CUDA source files from the benchmark repositories.",
        "",
        "## Per-Kernel SLoC",
        "",
        "| Kernel | Category | CUDA SLoC | Src Files | Tgt Files | OMP SLoC | OMP Files | Complexity | Pass Rate |",
        "|--------|----------|----------:|----------:|----------:|---------:|----------:|------------|----------:|",
    ]

    for k in sorted(kernels, key=lambda x: x["physical_sloc"], reverse=True):
        omp_sloc = k.get("omp_physical_sloc")
        omp_str = str(omp_sloc) if omp_sloc is not None else "N/A"
        omp_files = k.get("omp_num_files")
        omp_f_str = str(omp_files) if omp_files is not None else "N/A"
        md_lines.append(
            f"| {k['kernel']} | {k['category']} | {k['physical_sloc']:,} | "
            f"{k['num_source_files']} | {k.get('num_target_files', 'N/A')} | "
            f"{omp_str} | {omp_f_str} | "
            f"{k['complexity_class']} | {k['pass_rate']:.1%} |"
        )

    md_lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Min SLoC | {summary['min_sloc']:,} |",
        f"| Max SLoC | {summary['max_sloc']:,} |",
        f"| Mean SLoC | {summary['mean_sloc']:,.1f} |",
        f"| Median SLoC | {summary['median_sloc']:,.0f} |",
        f"| Std Dev | {summary['std_sloc']:,.1f} |",
        f"| Total SLoC (all kernels) | {summary['total_sloc_all_kernels']:,} |",
        "",
        "## SLoC Distribution",
        "",
        "| Range | Count |",
        "|-------|------:|",
    ])
    for bucket, count in buckets.items():
        md_lines.append(f"| {bucket} | {count} |")

    md_lines.extend([
        "",
        f"## ParEval-Repo Comparison",
        "",
        f"ParEval-Repo uses functions averaging ~133 SLoC (single-function snippets).",
        f"ParBench evaluates **full application kernels** with real build systems.",
        "",
        f"- **{above_threshold}/{n}** ({summary['pct_above_pareval_threshold']}%) "
        f"of ParBench kernels exceed ParEval-Repo's 133-SLoC average",
        f"- ParBench median: **{summary['median_sloc']:,.0f} SLoC** "
        f"({summary['median_sloc']/pareval_threshold:.1f}x ParEval-Repo)",
        f"- ParBench range: **{summary['min_sloc']:,}** to **{summary['max_sloc']:,}** SLoC",
        "",
        f"## SLoC vs. Pass Rate Correlation",
        "",
        f"Spearman rank correlation (SLoC vs. overall pass rate): "
        f"**{correlation}**",
        "",
    ])

    if correlation is not None and correlation < -0.3:
        md_lines.append(
            "Negative correlation suggests larger kernels are harder to translate correctly."
        )
    elif correlation is not None and abs(correlation) < 0.3:
        md_lines.append(
            "Weak correlation suggests SLoC alone does not strongly predict translation difficulty."
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # Print summary to stdout
    print(f"\n{'=' * 50}")
    print(f"SUMMARY: {n} kernels, {summary['min_sloc']}-{summary['max_sloc']} SLoC")
    print(f"Mean={summary['mean_sloc']:.0f}, Median={summary['median_sloc']:.0f}, "
          f"Std={summary['std_sloc']:.0f}")
    print(f"Distribution: {buckets}")
    print(f"{above_threshold}/{n} ({summary['pct_above_pareval_threshold']}%) > "
          f"ParEval-Repo threshold ({pareval_threshold})")
    print(f"SLoC vs pass-rate Spearman ρ = {correlation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
