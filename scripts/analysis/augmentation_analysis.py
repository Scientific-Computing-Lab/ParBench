#!/usr/bin/env python3
"""
scripts/analysis/augmentation_analysis.py

Complete augmentation analysis for the NeurIPS 2026 ParBench paper.
Builds per-kernel x per-level status matrices from raw eval result JSONs
(Phase 3 canonical+augmentation corpus), classifies kernel patterns, computes
aggregate statistics with Wilson CIs, and writes JSON + MD output artifacts.

Metrics:
  AUG-01: Per-kernel x per-level status matrix (cuda-to-omp, L0-L4)
  AUG-02: Pattern classification (stable_pass, stable_fail, degradation,
           improvement, other)

Output: results/analysis/augmentation_per_kernel_matrix.json + .md

Usage:
    python3 scripts/analysis/augmentation_analysis.py \\
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

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
except ImportError:
    pass  # graceful fallback — use default matplotlib style

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_FAIL_SOURCES = frozenset({
    "rodinia-kmeans-cuda",
    "rodinia-mummergpu-cuda",
})
"""Source specs whose CUDA source is KNOWN_FAIL (cannot build on this hardware)."""

EVAL_DIR_NAME = "together-qwen-3.5-397b-a17b"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# ---------------------------------------------------------------------------
# Okabe-Ito Colorblind-Safe Palette (Okabe & Ito, 2008)
# ---------------------------------------------------------------------------

STATUS_COLORS: dict[str, str] = {
    "PASS":            "#009E73",   # green
    "BUILD_FAIL":      "#D55E00",   # vermillion
    "RUN_FAIL":        "#E69F00",   # orange
    "VERIFY_FAIL":     "#0072B2",   # blue
    "EXTRACTION_FAIL": "#CC79A7",   # purple
}

STATUS_ORDER = ["PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL"]

STATUS_ABBREV = {
    "PASS": "P",
    "BUILD_FAIL": "BF",
    "RUN_FAIL": "RF",
    "VERIFY_FAIL": "VF",
    "EXTRACTION_FAIL": "EF",
}

# Pattern group ordering for heatmap rows
PATTERN_ORDER = ["stable_pass", "degradation", "improvement", "other", "stable_fail"]


# ---------------------------------------------------------------------------
# Wilson CI (re-implemented to avoid import complexity)
# ---------------------------------------------------------------------------

def wilson_ci(passes: int, total: int, alpha: float = 0.05) -> dict:
    """Wilson score confidence interval for a binomial proportion.

    Wilson score is preferred over normal approximation because it:
    - Never produces intervals outside [0, 1]
    - Is accurate even for small n and extreme p

    Returns:
        {"rate": float, "ci_lower": float, "ci_upper": float, "n": int}
    """
    if total == 0:
        return {"rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    # z-score for the given alpha (1.96 for 95% CI)
    # Use the inverse CDF of the standard normal distribution
    # For alpha=0.05: z = 1.959964 (approx 1.96)
    z = _norm_ppf(1 - alpha / 2)
    p_hat = passes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * total)) / total
    ) / denom

    return {
        "rate": round(p_hat, 4),
        "ci_lower": round(max(0.0, center - spread), 4),
        "ci_upper": round(min(1.0, center + spread), 4),
        "n": total,
    }


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (percent point function).

    Uses the rational approximation from Abramowitz and Stegun (26.2.23)
    for 0.5 < p < 1.0. Accurate to ~4.5e-4.
    """
    if p <= 0.5:
        return -_norm_ppf(1 - p)

    # Rational approximation constants
    t = math.sqrt(-2 * math.log(1 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


# ---------------------------------------------------------------------------
# Filename Parsing
# ---------------------------------------------------------------------------

def _augment_level_from_filename(stem: str) -> int:
    """Extract augmentation level from result file stem.

    Convention: {src_id}-to-{tgt_id}-L{N}.json -> N
                {src_id}-to-{tgt_id}.json      -> 0  (L0)
    """
    m = re.search(r"-L(\d+)(?:-s\d+)?$", stem)
    return int(m.group(1)) if m else 0


def _is_stochastic(stem: str) -> bool:
    """Return True if filename is a stochastic sample (temperature > 0)."""
    return bool(re.search(r"-s\d+$", stem))


def _consensus_status(statuses: list[str]) -> str:
    """Compute consensus status from multiple seed results.

    PASS if any seed passes (pass@1-of-any), matching L0-conditional filter.
    Otherwise, return the most common failure status.
    """
    if any(s == "PASS" for s in statuses):
        return "PASS"
    from collections import Counter
    counts = Counter(statuses)
    return counts.most_common(1)[0][0]


def _extract_api(spec_name: str) -> str:
    """Extract the API from a spec name like 'rodinia-backprop-cuda'.

    CRITICAL: Check for 'omp_target' FIRST since 'omp' is a substring.
    """
    if spec_name.endswith("-omp_target"):
        return "omp_target"
    if spec_name.endswith("-opencl"):
        return "opencl"
    if spec_name.endswith("-omp"):
        return "omp"
    if spec_name.endswith("-cuda"):
        return "cuda"
    # Fallback: last segment
    return spec_name.rsplit("-", 1)[-1] if spec_name else "unknown"


# ---------------------------------------------------------------------------
# Primary Matrix: cuda-to-omp
# ---------------------------------------------------------------------------

def build_primary_matrix(results_dir: Path, verbose: bool = False,
                         direction: str = "cuda-to-omp") -> dict:
    """Build per-kernel x per-level status matrix for a specific direction.

    For L0: collects all seed results and applies consensus (PASS if any passes).
    For L1-L4: uses single result per kernel (augmentation files are not seeded).

    Returns:
        dict with keys: direction, levels, kernel_count, per_kernel, aggregate,
        pattern_summary, exceptions
    """
    src_api_filter, tgt_api_filter = direction.split("-to-")

    per_kernel: dict[str, dict] = {}
    l0_seeds: dict[str, list[str]] = defaultdict(list)
    l0_meta: dict[str, dict] = {}

    for f in sorted(results_dir.glob("*.json")):
        stem = f.stem

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if verbose:
                print(f"  WARNING: Could not parse {f.name}", file=sys.stderr)
            continue

        source_spec = data.get("source_spec", "")
        target_spec = data.get("target_spec", "")

        src_api = _extract_api(source_spec)
        tgt_api = _extract_api(target_spec)

        if src_api != src_api_filter or tgt_api != tgt_api_filter:
            continue

        status = data.get("overall_status", "UNKNOWN")
        kernel = data.get("kernel", "unknown")
        level = data.get("augment_level")
        if level is None:
            level = _augment_level_from_filename(stem)

        if level < 0 or level > 4:
            continue

        suite = source_spec.split("-")[0] if source_spec else "unknown"

        if kernel not in per_kernel:
            per_kernel[kernel] = {
                "suite": suite,
                "source_spec": source_spec,
                "target_spec": target_spec,
                "known_fail": source_spec in KNOWN_FAIL_SOURCES,
            }

        if level == 0:
            l0_seeds[kernel].append(status)
            if kernel not in l0_meta:
                l0_meta[kernel] = {"suite": suite, "source_spec": source_spec,
                                   "target_spec": target_spec}
            continue

        if _is_stochastic(stem):
            continue

        per_kernel[kernel][f"L{level}"] = status

    # Apply consensus for L0 seeds
    for kernel, statuses in l0_seeds.items():
        if kernel not in per_kernel:
            meta = l0_meta.get(kernel, {})
            per_kernel[kernel] = {
                "suite": meta.get("suite", "unknown"),
                "source_spec": meta.get("source_spec", ""),
                "target_spec": meta.get("target_spec", ""),
                "known_fail": meta.get("source_spec", "") in KNOWN_FAIL_SOURCES,
            }
        per_kernel[kernel]["L0"] = _consensus_status(statuses)

    # Classify patterns
    pattern_summary = classify_patterns(per_kernel)

    for pattern_type, kernels in pattern_summary.items():
        for k in kernels:
            if k in per_kernel:
                per_kernel[k]["pattern"] = pattern_type

    aggregate = compute_aggregates(per_kernel)
    aggregate_excl_kf = compute_aggregates(per_kernel, exclude_known_fail=True)

    exceptions = identify_exceptions(per_kernel, pattern_summary)

    return {
        "direction": direction,
        "levels": LEVELS,
        "kernel_count": len(per_kernel),
        "per_kernel": per_kernel,
        "aggregate": aggregate,
        "aggregate_excluding_known_fail": aggregate_excl_kf,
        "pattern_summary": pattern_summary,
        "exceptions": exceptions,
    }


# ---------------------------------------------------------------------------
# Pattern Classification
# ---------------------------------------------------------------------------

def classify_patterns(per_kernel: dict) -> dict:
    """Classify each kernel into a pattern category based on L0-L4 statuses.

    Categories:
      - stable_pass: all L0-L4 are PASS
      - stable_fail: all L0-L4 are non-PASS AND all are the same status
      - degradation: L0 is PASS and any L1-L4 is non-PASS
      - improvement: L0 is non-PASS and any L1-L4 is PASS
      - other: everything else (e.g., mixed non-PASS statuses with no improvement)
    """
    result: dict[str, list[str]] = {
        "stable_pass": [],
        "stable_fail": [],
        "degradation": [],
        "improvement": [],
        "other": [],
    }

    for kernel, entry in per_kernel.items():
        statuses = [entry.get(f"L{i}", "UNKNOWN") for i in range(5)]
        l0 = statuses[0]
        l1_l4 = statuses[1:]

        if all(s == "PASS" for s in statuses):
            result["stable_pass"].append(kernel)
        elif l0 == "PASS" and any(s != "PASS" for s in l1_l4):
            result["degradation"].append(kernel)
        elif l0 != "PASS" and any(s == "PASS" for s in l1_l4):
            result["improvement"].append(kernel)
        elif all(s != "PASS" for s in statuses) and len(set(statuses)) == 1:
            # All same non-PASS status
            result["stable_fail"].append(kernel)
        else:
            # Mixed non-PASS statuses (all non-PASS, not all same)
            result["other"].append(kernel)

    # Sort each category for deterministic output
    for key in result:
        result[key].sort()

    return result


# ---------------------------------------------------------------------------
# Aggregate Statistics
# ---------------------------------------------------------------------------

def compute_aggregates(
    per_kernel: dict, exclude_known_fail: bool = False
) -> dict:
    """Compute aggregate pass rates with Wilson 95% CIs for each level.

    Args:
        per_kernel: The per-kernel matrix entries.
        exclude_known_fail: If True, exclude kernels with known_fail=True.

    Returns:
        Dict with L0-L4 keys, each containing rate, ci_lower, ci_upper, n.
    """
    agg: dict[str, dict] = {}

    for level in LEVELS:
        passes = 0
        total = 0
        for _kernel, entry in per_kernel.items():
            if exclude_known_fail and entry.get("known_fail"):
                continue
            status = entry.get(level, "UNKNOWN")
            if status != "UNKNOWN":
                total += 1
                if status == "PASS":
                    passes += 1

        agg[level] = wilson_ci(passes, total)

    return agg


# ---------------------------------------------------------------------------
# Exception Identification
# ---------------------------------------------------------------------------

def identify_exceptions(per_kernel: dict, pattern_summary: dict) -> list:
    """Identify and document exception kernels (degradation + improvement).

    For each non-stable kernel, build an exception entry with:
    - kernel name, suite, pattern type
    - L0 status and affected levels with statuses
    - Root cause note where available
    """
    exceptions: list[dict] = []

    # Root cause notes for known exceptions
    root_causes = {
        "backprop": (
            "Linker error: multiple definition of 'gettime' and 'main' -- "
            "LLM duplicated functions across files when given augmented source. "
            "Augmentation engine validated separately (backprop-cuda PASSES at "
            "all harness levels). This is genuine model brittleness, not a "
            "transform artifact."
        ),
    }

    exception_kernels = (
        pattern_summary.get("degradation", [])
        + pattern_summary.get("improvement", [])
    )

    for kernel in sorted(exception_kernels):
        entry = per_kernel.get(kernel, {})
        statuses = {f"L{i}": entry.get(f"L{i}", "UNKNOWN") for i in range(5)}

        # Determine pattern type
        pattern = entry.get("pattern", "unknown")

        # Find affected levels (non-PASS for degradation, PASS for improvement)
        if pattern == "degradation":
            affected = {
                k: v for k, v in statuses.items() if k != "L0" and v != "PASS"
            }
        else:  # improvement
            affected = {
                k: v for k, v in statuses.items() if k != "L0" and v == "PASS"
            }

        exceptions.append({
            "kernel": kernel,
            "suite": entry.get("suite", "unknown"),
            "pattern": pattern,
            "L0_status": statuses["L0"],
            "all_statuses": statuses,
            "affected_levels": affected,
            "root_cause": root_causes.get(kernel, "Not yet investigated"),
        })

    return exceptions


# ---------------------------------------------------------------------------
# Secondary Matrix: All Directions
# ---------------------------------------------------------------------------

def build_secondary_matrix(results_dir: Path, verbose: bool = False) -> dict:
    """Build aggregate pass rates for ALL directions (not just cuda-to-omp).

    For each direction found in the data, computes per-level aggregates
    with Wilson CIs.
    """
    # direction -> kernel -> level -> status
    dir_data: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for f in sorted(results_dir.glob("*.json")):
        stem = f.stem

        # Skip stochastic samples
        if _is_stochastic(stem):
            continue

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        source_spec = data.get("source_spec", "")
        target_spec = data.get("target_spec", "")

        src_api = _extract_api(source_spec)
        tgt_api = _extract_api(target_spec)
        direction = f"{src_api}-to-{tgt_api}"

        status = data.get("overall_status", "UNKNOWN")
        kernel = data.get("kernel", "unknown")
        level = data.get("augment_level")
        if level is None:
            level = _augment_level_from_filename(stem)

        if level < 0 or level > 4:
            continue

        dir_data[direction][kernel][f"L{level}"] = status

    # Compute per-direction aggregates
    per_direction_aggregate: dict[str, dict] = {}

    for direction in sorted(dir_data.keys()):
        kernels = dir_data[direction]
        agg: dict[str, dict] = {}

        for level in LEVELS:
            passes = 0
            total = 0
            for _kernel, levels_map in kernels.items():
                status = levels_map.get(level, "UNKNOWN")
                if status != "UNKNOWN":
                    total += 1
                    if status == "PASS":
                        passes += 1
            agg[level] = wilson_ci(passes, total)

        per_direction_aggregate[direction] = {
            "kernel_count": len(kernels),
            "levels": agg,
        }

    return {
        "direction_count": len(per_direction_aggregate),
        "per_direction_aggregate": per_direction_aggregate,
    }


# ---------------------------------------------------------------------------
# Figure Utilities
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save figure as both PDF and PNG (300 dpi for PNG)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def _text_color_for_bg(hex_color: str) -> str:
    """Return 'white' for dark backgrounds, 'black' for light ones.

    Uses ITU-R BT.601 luminance: 0.299*R + 0.587*G + 0.114*B.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 128 else "black"


# ---------------------------------------------------------------------------
# Figure: Per-Kernel x Per-Level Heatmap
# ---------------------------------------------------------------------------


def generate_heatmap(matrix_data: dict, output_dir: Path) -> None:
    """Generate per-kernel x per-level status heatmap (cuda-to-omp).

    Rows = kernels ordered by pattern group, columns = L0-L4,
    cells colored by status using Okabe-Ito palette.
    """
    pm = matrix_data["primary_matrix"]
    per_kernel = pm["per_kernel"]
    pattern_summary = pm["pattern_summary"]

    # Order kernels by pattern group then alphabetically within group
    ordered_kernels: list[str] = []
    group_boundaries: list[int] = []  # row index where each group starts
    group_labels: list[str] = []

    for pattern in PATTERN_ORDER:
        kernels_in_group = sorted(pattern_summary.get(pattern, []))
        if kernels_in_group:
            group_boundaries.append(len(ordered_kernels))
            group_labels.append(pattern)
            ordered_kernels.extend(kernels_in_group)

    n_kernels = len(ordered_kernels)
    n_levels = len(LEVELS)

    # Build 2D integer array: status -> index via STATUS_ORDER
    status_to_idx = {s: i for i, s in enumerate(STATUS_ORDER)}
    data_array = np.zeros((n_kernels, n_levels), dtype=int)

    for row_idx, kernel in enumerate(ordered_kernels):
        entry = per_kernel[kernel]
        for col_idx, level in enumerate(LEVELS):
            status = entry.get(level, "BUILD_FAIL")
            data_array[row_idx, col_idx] = status_to_idx.get(status, 1)

    # Build colormap
    colors = [STATUS_COLORS[s] for s in STATUS_ORDER]
    cmap = ListedColormap(colors)
    boundaries = [-0.5 + i for i in range(len(STATUS_ORDER) + 1)]
    norm = BoundaryNorm(boundaries, cmap.N)

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 6))
    ax.imshow(data_array, cmap=cmap, norm=norm, aspect="auto")

    # Axis labels
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(LEVELS)
    ax.set_yticks(range(n_kernels))
    ax.set_yticklabels(ordered_kernels, fontsize=7)

    # Cell annotations (abbreviated status with appropriate text color)
    for row_idx in range(n_kernels):
        for col_idx in range(n_levels):
            status_idx = data_array[row_idx, col_idx]
            status_name = STATUS_ORDER[status_idx]
            abbrev = STATUS_ABBREV[status_name]
            bg_color = colors[status_idx]
            txt_color = _text_color_for_bg(bg_color)
            ax.text(col_idx, row_idx, abbrev, ha="center", va="center",
                    fontsize=6, color=txt_color, fontweight="bold")

    # Pattern group separators (horizontal lines between groups)
    for boundary_idx in group_boundaries[1:]:
        ax.axhline(y=boundary_idx - 0.5, color="black", linewidth=0.8)

    # Legend
    legend_patches = [
        Patch(facecolor=STATUS_COLORS[s], label=s.replace("_", " "))
        for s in STATUS_ORDER
        if any(
            per_kernel[k].get(level, "") == s
            for k in ordered_kernels
            for level in LEVELS
        )
    ]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=6,
              frameon=True)

    ax.set_title("Per-Kernel Augmentation Status\n(CUDA-to-OMP)", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir, "aug_heatmap")


# ---------------------------------------------------------------------------
# Figure: Aggregate Trend Line with Wilson CIs
# ---------------------------------------------------------------------------


def generate_trend_line(matrix_data: dict, output_dir: Path) -> None:
    """Generate aggregate pass rate trend line (L0-L4) with Wilson 95% CI error bars."""
    agg = matrix_data["primary_matrix"]["aggregate"]

    levels_x = [0, 1, 2, 3, 4]
    labels = ["L0", "L1", "L2", "L3", "L4"]

    rates = []
    lower_errs = []
    upper_errs = []

    for level in LEVELS:
        a = agg[level]
        rate_pct = a["rate"] * 100
        rates.append(rate_pct)
        lower_errs.append((a["rate"] - a["ci_lower"]) * 100)
        upper_errs.append((a["ci_upper"] - a["rate"]) * 100)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.errorbar(
        levels_x, rates,
        yerr=[lower_errs, upper_errs],
        fmt="o-",
        color="#0072B2",
        capsize=4,
        linewidth=1.8,
        markersize=7,
        label="Pass Rate (Wilson 95% CI)",
    )

    ax.set_xlabel("Augmentation Level")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_xticks(levels_x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Aggregate Pass Rate by Augmentation Level", fontsize=9)

    fig.tight_layout()
    _save_figure(fig, output_dir, "aug_trend")


# ---------------------------------------------------------------------------
# Markdown Generation
# ---------------------------------------------------------------------------

def generate_markdown(data: dict) -> str:
    """Produce a markdown summary of the augmentation analysis."""
    lines: list[str] = []

    pm = data["primary_matrix"]
    sm = data["secondary_matrix"]

    # Header
    lines.append("# Per-Kernel Augmentation Matrix")
    lines.append("")
    lines.append(f"Generated: {data['generated_at']}")
    lines.append(f"Data source: `{data['data_source']}`")
    lines.append(f"Direction: {pm['direction']}")
    lines.append(f"Kernel count: {pm['kernel_count']}")
    lines.append("")

    # Per-kernel table
    lines.append("## Per-Kernel Status Table")
    lines.append("")
    lines.append(
        "| Kernel | Suite | L0 | L1 | L2 | L3 | L4 | Pattern | Known Fail |"
    )
    lines.append(
        "|--------|-------|----|----|----|----|----|---------|-----------:|"
    )

    for kernel in sorted(pm["per_kernel"].keys()):
        entry = pm["per_kernel"][kernel]
        row = [
            kernel,
            entry.get("suite", "?"),
            entry.get("L0", "?"),
            entry.get("L1", "?"),
            entry.get("L2", "?"),
            entry.get("L3", "?"),
            entry.get("L4", "?"),
            entry.get("pattern", "?"),
            "Yes" if entry.get("known_fail") else "",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Pattern Summary
    lines.append("## Pattern Summary")
    lines.append("")
    ps = pm["pattern_summary"]
    for pattern in ["stable_pass", "stable_fail", "degradation", "improvement", "other"]:
        kernels = ps.get(pattern, [])
        lines.append(f"- **{pattern}** ({len(kernels)}): {', '.join(kernels) if kernels else 'none'}")
    lines.append("")

    # Aggregate Pass Rates
    lines.append("## Aggregate Pass Rates")
    lines.append("")
    lines.append("### All Kernels")
    lines.append("")
    lines.append("| Level | Pass | Total | Rate | 95% CI |")
    lines.append("|-------|------|-------|------|--------|")
    agg = pm["aggregate"]
    for level in LEVELS:
        a = agg[level]
        passes = round(a["rate"] * a["n"])
        lines.append(
            f"| {level} | {passes} | {a['n']} | "
            f"{a['rate']:.1%} | [{a['ci_lower']:.1%}, {a['ci_upper']:.1%}] |"
        )
    lines.append("")

    lines.append("### Excluding KNOWN_FAIL")
    lines.append("")
    lines.append("| Level | Pass | Total | Rate | 95% CI |")
    lines.append("|-------|------|-------|------|--------|")
    agg_excl = pm["aggregate_excluding_known_fail"]
    for level in LEVELS:
        a = agg_excl[level]
        passes = round(a["rate"] * a["n"])
        lines.append(
            f"| {level} | {passes} | {a['n']} | "
            f"{a['rate']:.1%} | [{a['ci_lower']:.1%}, {a['ci_upper']:.1%}] |"
        )
    lines.append("")

    # Exceptions
    lines.append("## Exceptions")
    lines.append("")
    for exc in pm["exceptions"]:
        lines.append(f"### {exc['kernel']} ({exc['suite']}) -- {exc['pattern']}")
        lines.append(f"- L0: {exc['L0_status']}")
        lines.append(
            "- Affected: "
            + ", ".join(f"{k}={v}" for k, v in exc["affected_levels"].items())
        )
        lines.append(f"- Root cause: {exc['root_cause']}")
        lines.append("")

    # Secondary Directions
    lines.append("## Secondary Directions")
    lines.append("")
    lines.append(
        "| Direction | Kernels | L0 Rate | L1 Rate | L2 Rate | L3 Rate | L4 Rate |"
    )
    lines.append(
        "|-----------|---------|---------|---------|---------|---------|---------|"
    )
    for direction, ddata in sorted(sm["per_direction_aggregate"].items()):
        rates = []
        for level in LEVELS:
            r = ddata["levels"][level]["rate"]
            rates.append(f"{r:.1%}")
        lines.append(
            f"| {direction} | {ddata['kernel_count']} | "
            + " | ".join(rates) + " |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Orchestrate augmentation analysis: build matrices, classify, write output."""
    parser = argparse.ArgumentParser(
        description="Augmentation analysis for NeurIPS 2026 ParBench paper (Phase 3 canonical+augmentation corpus)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to project root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/analysis/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate publication-quality figures (aug_heatmap, aug_trend)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: docs/paper/figures/)",
    )
    parser.add_argument(
        "--model-dir",
        nargs="+",
        default=[EVAL_DIR_NAME, "azure-gpt-5.4"],
        help="Model results subdirectories under results/evaluation/ (space-separated; default: %(default)s)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / "results" / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_dir_name in args.model_dir:
        results_dir = project_root / "results" / "evaluation" / model_dir_name

        if not results_dir.exists():
            print(f"WARNING: Results directory not found, skipping: {results_dir}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"Project root: {project_root}")
            print(f"Results dir:  {results_dir}")
            print(f"Output dir:   {output_dir}")

        # Discover all directions from data
        all_directions: set[str] = set()
        for f in results_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            src_api = _extract_api(d.get("source_spec", ""))
            tgt_api = _extract_api(d.get("target_spec", ""))
            if src_api and tgt_api and src_api != "unknown" and tgt_api != "unknown":
                all_directions.add(f"{src_api}-to-{tgt_api}")

        # Build primary matrix (cuda-to-omp, backward compatible)
        print(f"[{model_dir_name}] Building primary matrix (cuda-to-omp)...")
        primary = build_primary_matrix(results_dir, verbose=args.verbose, direction="cuda-to-omp")
        print(f"  Found {primary['kernel_count']} kernels")
        print(f"  Patterns: {', '.join(f'{k}={len(v)}' for k, v in primary['pattern_summary'].items())}")

        # Build per-direction matrices for all augmentation directions
        per_direction_matrices = {}
        for direction in sorted(all_directions):
            print(f"[{model_dir_name}] Building matrix for {direction}...")
            mat = build_primary_matrix(results_dir, verbose=args.verbose, direction=direction)
            if mat["kernel_count"] > 0:
                per_direction_matrices[direction] = mat
                print(f"  Found {mat['kernel_count']} kernels")

        # Build secondary matrix (all directions)
        print(f"[{model_dir_name}] Building secondary matrix (all directions)...")
        secondary = build_secondary_matrix(results_dir, verbose=args.verbose)
        print(f"  Found {secondary['direction_count']} directions")

        # Assemble combined JSON
        combined = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": f"results/evaluation/{model_dir_name}",
            "primary_matrix": primary,
            "per_direction": per_direction_matrices,
            "secondary_matrix": secondary,
        }

        # Write JSON (per-model suffix when multiple models)
        suffix = f"_{model_dir_name}" if len(args.model_dir) > 1 else ""
        json_path = output_dir / f"augmentation_per_kernel_matrix{suffix}.json"
        json_path.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  Wrote: {json_path}")

        # Generate and write markdown
        md_content = generate_markdown(combined)
        md_path = output_dir / f"augmentation_per_kernel_matrix{suffix}.md"
        md_path.write_text(md_content, encoding="utf-8")
        print(f"  Wrote: {md_path}")

        # Generate figures if requested
        if args.figures:
            figures_dir = (
                args.figures_dir or project_root / "docs" / "paper" / "figures"
            ).resolve()
            figures_dir.mkdir(parents=True, exist_ok=True)

            print("Generating figures...")
            generate_heatmap(combined, figures_dir)
            print(f"  Wrote: {figures_dir}/aug_heatmap.{{pdf,png}}")

            generate_trend_line(combined, figures_dir)
            print(f"  Wrote: {figures_dir}/aug_trend.{{pdf,png}}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
