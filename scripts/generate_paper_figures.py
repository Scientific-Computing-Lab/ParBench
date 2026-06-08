#!/usr/bin/env python3
"""Generate all publication-quality figures for the NeurIPS 2026 paper.

Unified script combining main-body figures (F2--F7), appendix figures
(C.1--C.4), and the LaTeX model-comparison table (T2).

Usage:
    python3 scripts/generate_paper_figures.py \
      --project-root . \
      --figure all \
      --output-dir docs/paper/figures

    # Single figure:
    python3 scripts/generate_paper_figures.py \
      --project-root . \
      --figure F3 \
      --output-dir docs/paper/figures
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402, F401

plt.style.use(["science", "ieee", "no-latex"])

# ---------------------------------------------------------------------------
# Okabe-Ito Colorblind-Safe Palette (Okabe & Ito, 2008)
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "black":      "#000000",
}

STATUS_COLORS: dict[str, str] = {
    "PASS":            "#009E73",
    "BUILD_FAIL":      "#D55E00",
    "RUN_FAIL":        "#E69F00",
    "VERIFY_FAIL":     "#0072B2",
    "EXTRACTION_FAIL": "#CC79A7",
}

STATUS_HATCH: dict[str, str] = {
    "PASS":            "",
    "BUILD_FAIL":      "///",
    "RUN_FAIL":        "\\\\\\\\",
    "VERIFY_FAIL":     "xxx",
    "EXTRACTION_FAIL": "...",
}

STATUS_ABBREV: dict[str, str] = {
    "PASS": "P",
    "BUILD_FAIL": "BF",
    "RUN_FAIL": "RF",
    "VERIFY_FAIL": "VF",
    "EXTRACTION_FAIL": "EF",
}

STATUS_ORDER: list[str] = [
    "PASS", "BUILD_FAIL", "RUN_FAIL", "EXTRACTION_FAIL", "VERIFY_FAIL",
]

NA_COLOR = "#E0E0E0"

MODEL_COLORS: dict[str, str] = {
    "together-qwen-3.5-397b-a17b": OKABE_ITO["orange"],
    "azure-gpt-5.4":          OKABE_ITO["sky_blue"],
    "azure-gpt-5.3-codex":    OKABE_ITO["green"],
}

MODEL_DISPLAY: dict[str, str] = {
    "together-qwen-3.5-397b-a17b": "Qwen 3.5\n397B",
    "azure-gpt-5.4":          "GPT-5.4",
    "azure-gpt-5.3-codex":    "GPT-5.3\nCodex",
}

MODEL_DISPLAY_SHORT: dict[str, str] = {
    "together-qwen-3.5-397b-a17b": "Qwen 3.5 397B-A17B",
    "azure-gpt-5.4":          "Azure GPT-5.4",
    "azure-gpt-5.3-codex":    "Azure GPT-5.3-codex",
}

MODEL_LINESTYLE: dict[str, tuple[str, str]] = {
    "together-qwen-3.5-397b-a17b": ("D-.", "dashdot"),
    "azure-gpt-5.4":          ("v-", "solid"),
    "azure-gpt-5.3-codex":    ("s--", "dashed"),
}

MODEL_SLUG: dict[str, str] = {
    "together-qwen-3.5-397b-a17b": "qwen",
    "azure-gpt-5.4": "gpt",
    "azure-gpt-5.3-codex": "codex",
}

# Legacy palette references for survey figures (F2, C.4)
PALETTE = {
    "rose":         OKABE_ITO["vermillion"],
    "rose_tint":    "#F5D5C8",
    "rose_dark":    "#A34A00",
    "saffron":      OKABE_ITO["orange"],
    "saffron_tint": "#FBF0DD",
    "saffron_dark": "#B37D00",
    "teal":         OKABE_ITO["blue"],
    "teal_tint":    "#C8E0F0",
    "teal_dark":    "#004A75",
    "gold":         OKABE_ITO["yellow"],
    "gold_tint":    "#FDF5E3",
    "gold_dark":    "#B5A23B",
    "charcoal":     OKABE_ITO["black"],
    "slate":        "#636e72",
    "linen":        "#F5F0EB",
}

FIGURE_DPI: int = 600
FONT_SIZE_DEFAULT: int = 10
PDF_FONTTYPE: int = 42

# Canonical 6 directions for multi-direction figures
DIRECTIONS = [
    "cuda-to-omp", "omp-to-cuda",
    "cuda-to-opencl", "opencl-to-cuda",
    "omp-to-opencl", "opencl-to-omp",
]

ALL_DIRECTIONS = DIRECTIONS + ["cuda-to-omp_target", "omp_target-to-cuda"]

DIRECTION_LABELS = {
    "cuda-to-omp": "CUDA\u2192OMP",
    "omp-to-cuda": "OMP\u2192CUDA",
    "cuda-to-opencl": "CUDA\u2192OCL",
    "opencl-to-cuda": "OCL\u2192CUDA",
    "omp-to-opencl": "OMP\u2192OCL",
    "opencl-to-omp": "OCL\u2192OMP",
    "cuda-to-omp_target": "CUDA\u2192OMP-T",
    "omp_target-to-cuda": "OMP-T\u2192CUDA",
}

SUITE_ORDER = ["rodinia", "xsbench", "rsbench", "mixbench", "hecbench"]

SUITE_DISPLAY = {
    "rodinia": "Rodinia",
    "xsbench": "XSBench",
    "rsbench": "RSBench",
    "mixbench": "mixbench",
    "hecbench": "HeCBench",
}

# F2: Repository-level vs kernel-level API pair counts (survey data)
REPO_KERNEL_PAIRS: list[tuple[str, int, int]] = [
    ("CUDA\u2013OpenMP", 6, 472),
    ("CUDA\u2013HIP", 3, 633),
    ("CUDA\u2013SYCL", 2, 616),
]

# C.4: HeCBench kernel selection pipeline funnel stages (survey data)
# Stages 0-2: automated survey of HeCBench-master/src at commit 22785cdd
# Stages 3-4: manually curated (human judgment \u2014 self-checking strictness, domain diversity)
HECBENCH_FUNNEL_STAGES: list[tuple[str, int, str | None]] = [
    ("HeCBench kernels total", 522, None),
    ("All 4 API variants\n(CUDA, HIP, SYCL, OMP)", 329, "\u2212193: missing API variants"),
    ("With Makefiles", 327, "\u22122: no Makefile"),
    ("With self-checking\n(PASS/FAIL/verify patterns)", 242, "\u221285: no verification"),
    ("Final selected\n(complexity, deps, diversity)", 60, "\u2212182: complexity/deps/diversity"),
]



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def setup_rcparams() -> None:
    """Configure matplotlib for publication-quality output (NeurIPS 2026)."""
    plt.rcParams.update({
        "font.size": FONT_SIZE_DEFAULT,
        "axes.titlesize": 11,
        "axes.labelsize": FONT_SIZE_DEFAULT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "pdf.fonttype": PDF_FONTTYPE,
        "ps.fonttype": PDF_FONTTYPE,
    })


def _text_color_for_bg(hex_color: str) -> str:
    """Return 'white' or 'black' for readable text on the given background."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 145 else "black"


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save figure as both PNG (300 DPI) and vector PDF."""
    for fmt in ("png", "pdf"):
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300 if fmt == "png" else None)
        print(f"  Saved: {path}")


def aggregate_status_counts(
    records: list[dict], group_key: str,
) -> dict[str, dict[str, int]]:
    """Aggregate overall_status counts grouped by a record key."""
    result: dict[str, dict[str, int]] = {}
    for r in records:
        key = r.get(group_key, "UNKNOWN")
        if key not in result:
            result[key] = defaultdict(int)
        status = r.get("overall_status") or "UNKNOWN"
        result[key][status] += 1
    return result


def create_status_legend(
    statuses: list[str], *, include_hatch: bool = False,
) -> list[Patch]:
    """Create matplotlib Patch handles for a status legend."""
    patches = []
    for s in statuses:
        if s not in STATUS_COLORS:
            continue
        kwargs: dict = {
            "facecolor": STATUS_COLORS[s],
            "edgecolor": "black",
            "linewidth": 0.5,
            "label": s.replace("_", " "),
        }
        if include_hatch:
            kwargs["hatch"] = STATUS_HATCH[s]
        patches.append(Patch(**kwargs))
    return patches


def _extract_api(spec_name: str) -> str:
    """Extract API suffix from a spec name like 'rodinia-bfs-cuda' -> 'cuda'."""
    if spec_name.endswith("-omp_target"):
        return "omp_target"
    parts = spec_name.rsplit("-", 1)
    return parts[1] if len(parts) == 2 else "unknown"


def _extract_suite(spec_name: str) -> str:
    """Extract suite prefix from a spec name."""
    for prefix in ("rodinia", "xsbench", "hecbench", "rsbench", "mixbench"):
        if spec_name.startswith(prefix + "-"):
            return prefix
    return "other"



def _classify_initial_status(attempt: dict) -> str:
    """Classify the initial failure status from an attempt record."""
    if attempt.get("extraction_fail"):
        return "EXTRACTION_FAIL"
    bs = attempt.get("build_status", "")
    if bs and bs.lower() == "fail":
        return "BUILD_FAIL"
    rs = attempt.get("run_status")
    if rs is not None and str(rs).lower() not in ("pass", "none", ""):
        return "RUN_FAIL"
    vs = attempt.get("verify_status")
    if vs is not None and str(vs).lower() not in ("pass", "none", ""):
        return "VERIFY_FAIL"
    if bs and bs.lower() == "pass":
        return "PASS"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_results(
    project_root: Path, file_filter: str | None = None,
) -> list[dict]:
    """Load all per-task result JSONs from results/evaluation/{model}/*.json.

    Adds ``is_sample`` field to every record based on filename pattern
    (the ``-sN`` suffix), NOT the ``sample_id`` JSON field.

    Args:
        project_root: Project root path.
        file_filter: One of 'samples', 'augmented', 'base', or None (all).
    """
    results_dir = project_root / "results" / "evaluation"
    records: list[dict] = []
    if not results_dir.exists():
        return records

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for result_file in sorted(model_dir.glob("*.json")):
            stem = result_file.stem

            # Determine is_sample from filename pattern, NOT sample_id field
            is_sample = bool(re.search(r"-s\d+", stem))
            is_augmented = bool(re.search(r"-L\d+$", stem))
            is_base = not is_sample and not is_augmented

            if file_filter == "samples" and not is_sample:
                continue
            if file_filter == "augmented" and not is_augmented:
                continue
            if file_filter == "base" and not is_base:
                continue

            try:
                data = json.loads(result_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            source_spec = data.get("source_spec", "")
            target_spec = data.get("target_spec", "")
            if not source_spec or not target_spec:
                continue

            # Augment level from filename
            m_lvl = re.search(r"-L([1-4])$", stem)
            augment_level = int(m_lvl.group(1)) if m_lvl else 0

            src_api = _extract_api(source_spec)
            tgt_api = _extract_api(target_spec)

            kernel = data.get("kernel", "")
            if not kernel:
                parts = source_spec.split("-")
                kernel = "-".join(parts[1:-1]) if len(parts) >= 3 else source_spec

            records.append({
                "kernel": kernel,
                "suite": _extract_suite(source_spec),
                "direction": f"{src_api}-to-{tgt_api}",
                "model": data.get("model", model),
                "overall_status": data.get("overall_status", "UNKNOWN"),
                "sample_id": data.get("sample_id"),
                "augment_level": augment_level,
                "total_attempts": data.get("total_attempts", 1),
                "attempts": data.get("attempts", []),
                "source_spec": source_spec,
                "target_spec": target_spec,
                "is_sample": is_sample,
            })

    return records


def load_augmentation_eval_results(project_root: Path) -> list[dict]:
    """Load augmentation eval_*.json files (contain transform data)."""
    aug_dir = project_root / "results" / "augmentation"
    records = []
    for api in ("cuda", "omp", "opencl"):
        fpath = aug_dir / f"eval_{api}.json"
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text())
        for entry in data:
            entry["api"] = api
            records.append(entry)
    return records


def filter_records(
    records: list[dict],
    level: int = 0,
    suite: str | None = None,
    direction: str | None = None,
    *,
    samples_only: bool = False,
) -> list[dict]:
    """Filter result records by augmentation level, suite, direction, and sample flag."""
    out = [r for r in records if r.get("augment_level", 0) == level]
    if suite:
        out = [r for r in out if r.get("suite", "") == suite]
    if direction:
        out = [r for r in out if r.get("direction", "") == direction]
    if samples_only:
        out = [r for r in out if r.get("is_sample", False)]
    return out


def get_canonical_l0(records: list[dict]) -> list[dict]:
    """Return canonical L0 records: true base files, or sample_id=0 as fallback."""
    base = [r for r in records if not r.get("is_sample", False)]
    l0 = filter_records(base, level=0)
    if l0:
        return l0
    s0 = [
        r for r in records
        if r.get("is_sample", False)
        and r.get("augment_level", -1) == 0
        and r.get("sample_id") == 0
    ]
    return s0


def build_kernel_model_matrix(
    records: list[dict],
    level: int = 0,
    suite: str | None = None,
    direction: str | None = "cuda-to-omp",
) -> tuple[list[str], list[str], dict[tuple[str, str], str]]:
    """Build (kernel, model) -> overall_status lookup.

    Returns:
        kernels: sorted by total PASS count descending, then alphabetical
        models:  sorted by pass rate descending, then alphabetical
        lookup:  {(kernel, model_dir_name): overall_status}
    """
    filtered = filter_records(records, level=level, suite=suite, direction=direction)
    lookup: dict[tuple[str, str], str] = {}
    kernel_pass: dict[str, int] = defaultdict(int)
    model_pass: dict[str, int] = defaultdict(int)
    model_total: dict[str, int] = defaultdict(int)

    for r in filtered:
        k = r["kernel"]
        m = r["model"]
        status = r.get("overall_status", "UNKNOWN")
        lookup[(k, m)] = status
        if status == "PASS":
            kernel_pass[k] += 1
            model_pass[m] += 1
        model_total[m] += 1

    all_kernels = sorted({k for k, _ in lookup})
    kernels = sorted(all_kernels, key=lambda k: (-kernel_pass.get(k, 0), k))

    all_models = sorted({m for _, m in lookup})
    models = sorted(
        all_models,
        key=lambda m: (-model_pass.get(m, 0) / max(model_total.get(m, 1), 1), m),
    )

    return kernels, models, lookup


# ===================================================================
# F2: Repository-Level vs Kernel-Level Counts (survey data)
# ===================================================================


def generate_f2_repo_vs_kernel(
    output_dir: Path, verbose: bool,
) -> None:
    """F2: Grouped bar chart comparing repo-level vs kernel-level counts."""
    pairs = REPO_KERNEL_PAIRS
    labels = [p[0] for p in pairs]
    repo_counts = np.array([p[1] for p in pairs], dtype=float)
    kernel_counts = np.array([p[2] for p in pairs], dtype=float)
    multipliers = [f"{int(k / r)}x" for r, k in zip(repo_counts, kernel_counts)]

    if verbose:
        for lbl, r, k, m in zip(labels, repo_counts, kernel_counts, multipliers):
            print(f"  {lbl}: {int(r)} repos -> {int(k)} kernels ({m})")

    x = np.arange(len(labels))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    ax.bar(
        x - bar_width / 2, repo_counts, bar_width,
        color=PALETTE["teal_tint"], edgecolor=PALETTE["teal"],
        linewidth=1.2, hatch="///", label="Repository Count",
    )
    ax.bar(
        x + bar_width / 2, kernel_counts, bar_width,
        color=PALETTE["teal"], edgecolor=PALETTE["teal_dark"],
        linewidth=1.2, hatch="\\\\", label="Kernel Count",
    )

    ax.set_yscale("log")
    ax.set_ylim(3, 3000)
    # Clean integer-like y-axis ticks for log scale
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(["10", "100", "1,000"])
    ax.minorticks_off()

    # Value labels centered above each bar
    for i, (r, k) in enumerate(zip(repo_counts, kernel_counts)):
        ax.text(
            x[i] - bar_width / 2, r * 1.25, str(int(r)),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=PALETTE["teal_dark"],
        )
        ax.text(
            x[i] + bar_width / 2, k * 1.15, str(int(k)),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=PALETTE["charcoal"],
        )

    # Multiplier labels well above bars
    for i, mult in enumerate(multipliers):
        max_val = max(repo_counts[i], kernel_counts[i])
        ax.text(
            x[i], max_val * 2.5, mult,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=PALETTE["rose"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    # Add padding between x-axis labels and the bars
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("Count (log scale)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)
    ax.set_title(
        "Repo vs. Kernel Translation Pair Counts",
        fontsize=9, fontweight="bold", pad=8,
    )

    fig.tight_layout()
    _save_figure(fig, output_dir, "f2_repo_vs_kernel")
    plt.close(fig)


# ===================================================================
# F3: Per-Kernel Translation Outcomes Heatmap (all suites, 6 directions)
# ===================================================================


def generate_f3_kernel_heatmap(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
    project_root: Path | None = None,
    *,
    model_id: str,
) -> None:
    """F3: Single-panel heatmap of all benchmark kernels x directions, grouped by suite.

    Uses only non-sample, non-augmented (L0 base) records for the given model
    in all directions available for that model.  Kernels are grouped by suite
    (SUITE_ORDER) with horizontal divider lines and suite labels on the left margin.
    Includes ALL kernels from spec files, not just those with eval results.
    """
    slug = MODEL_SLUG[model_id]

    # ------------------------------------------------------------------
    # 1. Filter to base L0 records for this model's directions
    # ------------------------------------------------------------------
    l0_records = get_canonical_l0(records)
    directions = list(ALL_DIRECTIONS)  # always 8 directions for consistent layout
    std_records = [r for r in l0_records if r["direction"] in directions and r.get("model") == model_id]

    if not std_records:
        print("  WARNING: No L0 records for this model -- skipping F3.")
        return

    # ------------------------------------------------------------------
    # 2. Build (kernel, direction) -> status lookup and kernel->suite map
    # ------------------------------------------------------------------
    lookup: dict[tuple[str, str], str] = {}
    kernel_suite: dict[str, str] = {}
    kernel_pass_count: dict[str, int] = defaultdict(int)

    kernel_apis: dict[str, set[str]] = defaultdict(set)
    # Also scan ALL records (not just std) to detect omp_target-only kernels
    for r in [r for r in records if not r.get("is_sample", False) and r.get("model") == model_id]:
        rk = r.get("kernel", "")
        src_spec = r.get("source_spec", "")
        tgt_spec = r.get("target_spec", "")
        if src_spec:
            kernel_apis[rk].add(_extract_api(src_spec))
        if tgt_spec:
            kernel_apis[rk].add(_extract_api(tgt_spec))

    # Also scan spec files on disk to discover ALL kernels (even those without
    # eval results) and their available APIs
    if project_root:
        specs_dir = project_root / "specs"
        # Curated HeCBench kernels: those that have eval results
        curated_hec = set()
        for r in l0_records:
            if r.get("suite") == "hecbench":
                curated_hec.add(r.get("kernel", ""))
        for spec_file in specs_dir.glob("*.json"):
            parts = spec_file.stem.split("-")
            suite = parts[0]
            if suite not in SUITE_ORDER:
                continue
            api = parts[-1]
            kernel = "-".join(parts[1:-1])
            # For HeCBench, only include curated kernels (those with eval data)
            if suite == "hecbench" and kernel not in curated_hec:
                continue
            kernel_apis[kernel].add(api)
            if kernel not in kernel_suite:
                kernel_suite[kernel] = suite

    for r in std_records:
        k = r["kernel"]
        d = r["direction"]
        status = r.get("overall_status", "UNKNOWN")
        lookup[(k, d)] = status
        kernel_suite[k] = r.get("suite", "other")
        if status == "PASS":
            kernel_pass_count[k] += 1

    # ------------------------------------------------------------------
    # 3. Group kernels by suite, sort within group by PASS count desc
    # ------------------------------------------------------------------
    suite_groups: list[tuple[str, list[str]]] = []
    for suite in SUITE_ORDER:
        suite_kernels = sorted(
            [k for k, s in kernel_suite.items() if s == suite],
        )
        if suite_kernels:
            suite_groups.append((suite, suite_kernels))

    # Flatten to ordered kernel list and track suite boundaries
    kernels_ordered: list[str] = []
    suite_boundaries: list[int] = []  # row indices where dividers go
    suite_midpoints: list[tuple[str, float]] = []  # (suite_name, y_midpoint)
    for suite_name, suite_kernels in suite_groups:
        start = len(kernels_ordered)
        if kernels_ordered:
            suite_boundaries.append(start)
        kernels_ordered.extend(suite_kernels)
        mid = start + len(suite_kernels) / 2 - 0.5
        suite_midpoints.append((suite_name, mid))

    n_kernels = len(kernels_ordered)
    n_dirs = len(directions)

    if verbose:
        print(f"  Total kernels for {MODEL_DISPLAY_SHORT[model_id]}: {n_kernels}")
        for suite_name, suite_kernels in suite_groups:
            print(f"    {suite_name}: {len(suite_kernels)} kernels")

    # ------------------------------------------------------------------
    # 4. Build numeric matrix for colormap
    # ------------------------------------------------------------------
    status_to_idx = {s: i for i, s in enumerate(STATUS_ORDER)}
    na_idx = len(STATUS_ORDER)

    matrix = np.full((n_kernels, n_dirs), na_idx, dtype=int)
    present: set[str] = set()

    for i, kernel in enumerate(kernels_ordered):
        for j, direction in enumerate(directions):
            status = lookup.get((kernel, direction))
            if status and status in status_to_idx:
                matrix[i, j] = status_to_idx[status]
                present.add(status)

    # ------------------------------------------------------------------
    # 5. Render heatmap
    # ------------------------------------------------------------------
    fig_height = max(6, n_kernels * 0.28 + 2)
    fig, ax = plt.subplots(figsize=(7.16, fig_height))

    colors = [STATUS_COLORS[s] for s in STATUS_ORDER] + [NA_COLOR]
    cmap = mcolors.ListedColormap(colors)
    bounds = list(range(len(colors) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # API display names for missing-data labels
    _api_full = {"omp": "OpenMP", "cuda": "CUDA", "opencl": "OpenCL", "omp_target": "OMP-Target"}

    # Cell text: status abbreviation or specific missing-API reason
    for i, kernel in enumerate(kernels_ordered):
        for j, direction in enumerate(directions):
            status = lookup.get((kernel, direction))
            if status and status in STATUS_ABBREV:
                bg = STATUS_COLORS.get(status, "#FFFFFF")
                ax.text(
                    j, i, STATUS_ABBREV[status],
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color=_text_color_for_bg(bg),
                )
            else:
                # Determine which API is missing for this kernel+direction
                src_api, tgt_api = direction.split("-to-")
                apis = kernel_apis.get(kernel, set())
                src_missing = src_api not in apis
                tgt_missing = tgt_api not in apis
                if src_missing and tgt_missing:
                    reason = "No src\nor tgt\nimpl."
                elif src_missing:
                    api_name = _api_full.get(src_api, src_api)
                    reason = f"No {api_name}\nsrc impl."
                elif tgt_missing:
                    api_name = _api_full.get(tgt_api, tgt_api)
                    reason = f"No {api_name}\ntgt impl."
                else:
                    reason = "\u2014"
                ax.text(
                    j, i, reason,
                    ha="center", va="center",
                    fontsize=4.5, fontstyle="italic", color="#777777",
                    linespacing=0.85,
                )

    # ------------------------------------------------------------------
    # 6. Axis labels
    # ------------------------------------------------------------------
    ax.set_xticks(range(n_dirs))
    ax.set_xticklabels(
        [DIRECTION_LABELS[d] for d in directions],
        fontsize=8, rotation=0, ha="center",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_kernels))
    ax.set_yticklabels(kernels_ordered, fontsize=7)

    # ------------------------------------------------------------------
    # 7. Grid lines (white cell borders)
    # ------------------------------------------------------------------
    for i in range(n_kernels + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(n_dirs + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.5)

    # ------------------------------------------------------------------
    # 8. Suite grouping: thicker divider lines + suite labels
    # ------------------------------------------------------------------
    for boundary_y in suite_boundaries:
        ax.axhline(boundary_y - 0.5, color="black", linewidth=2.0, linestyle="-")

    for suite_name, y_mid in suite_midpoints:
        display = SUITE_DISPLAY.get(suite_name, suite_name)
        ax.text(
            -1.0, y_mid, display,
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            rotation=0,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                linewidth=0.5,
            ),
        )

    # ------------------------------------------------------------------
    # 9. Title
    # ------------------------------------------------------------------
    model_name = MODEL_DISPLAY_SHORT[model_id]
    ax.set_title(
        f"Per-Kernel Translation Outcomes (L0, All Suites, {model_name})\n"
        f"{n_kernels} kernels \u00d7 {n_dirs} directions",
        fontsize=10, fontweight="bold", pad=12,
    )

    # ------------------------------------------------------------------
    # 10. Legend
    # ------------------------------------------------------------------
    present_ordered = sorted(present, key=lambda s: STATUS_ORDER.index(s))
    legend_handles = create_status_legend(present_ordered)
    # Add N/A legend entry for missing data
    legend_handles.append(Patch(
        facecolor=NA_COLOR, edgecolor="black", linewidth=0.5,
        label="No data",
    ))
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_handles), frameon=False, fontsize=7,
    )

    # Explanatory annotations below legend (tighter spacing)
    fig.text(
        0.5, -0.01,
        '"No [API] src/tgt impl." = the benchmark suite has no implementation '
        "in that API for this kernel,\n"
        "so the translation task does not exist "
        "(e.g., dwt2d has no OpenMP code; gaussian has no OpenMP code).",
        ha="center", va="top", fontsize=6, fontstyle="italic", color="#555555",
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.28, bottom=0.06)
    _save_figure(fig, output_dir, f"f3_kernel_model_heatmap_{slug}")
    plt.close(fig)


# ===================================================================
# F4: Failure Taxonomy (status distribution by direction, all suites)
# ===================================================================


def generate_f4_failure_taxonomy(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
    *,
    model_id: str,
) -> None:
    """F4: Failure taxonomy -- status distribution by direction (per model)."""
    l0_records = get_canonical_l0(records)
    directions = list(ALL_DIRECTIONS)  # always 8 directions for consistent layout
    std_records = [r for r in l0_records if r["direction"] in directions and r.get("model") == model_id]

    if not std_records:
        print(f"  WARNING: No L0 records for {MODEL_DISPLAY_SHORT[model_id]} -- skipping F4.")
        return

    dir_status = aggregate_status_counts(std_records, "direction")

    if verbose:
        for d in directions:
            counts = dir_status.get(d, {})
            total = sum(counts.values())
            print(f"  {d}: {total} tasks")

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    active_dirs = list(ALL_DIRECTIONS)  # show all 8 directions, zero bars for missing
    x = np.arange(len(active_dirs))
    bar_width = 0.6

    bottoms = np.zeros(len(active_dirs))
    all_statuses: set[str] = set()
    for status in STATUS_ORDER:
        counts = np.array([
            dir_status.get(d, {}).get(status, 0) for d in active_dirs
        ], dtype=float)
        if counts.sum() > 0:
            all_statuses.add(status)
        ax.bar(
            x, counts, bar_width,
            bottom=bottoms,
            color=STATUS_COLORS[status],
            hatch=STATUS_HATCH[status],
            edgecolor="black", linewidth=0.5,
            label=status.replace("_", " "),
        )
        for i, (count, bottom) in enumerate(zip(counts, bottoms)):
            if count > 0:
                ax.text(
                    x[i], bottom + count / 2, str(int(count)),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=_text_color_for_bg(STATUS_COLORS[status]),
                )
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DIRECTION_LABELS.get(d, d) for d in active_dirs],
        fontsize=8, rotation=30, ha="right",
    )
    ax.tick_params(axis="x", length=0)  # remove x-axis tick marks
    ax.set_ylabel("Number of Tasks")
    ax.set_ylim(0, max(bottoms) * 1.08 if max(bottoms) > 0 else 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    model_name = MODEL_DISPLAY_SHORT[model_id]
    ax.set_title(
        f"Failure Taxonomy: Status Distribution by Direction\n(L0, All Suites, {model_name})",
        fontsize=9, fontweight="bold", pad=8,
    )

    present_ordered = [s for s in STATUS_ORDER if s in all_statuses]
    handles = create_status_legend(present_ordered, include_hatch=True)
    ax.legend(
        handles=handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=True, framealpha=0.9, fontsize=7,
        borderaxespad=0,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, output_dir, f"f4_failure_taxonomy_{MODEL_SLUG[model_id]}")
    plt.close(fig)


# ===================================================================
# F5: pass@k by Direction (BUG FIX: uses is_sample from filename)
# ===================================================================


def generate_f5_pass_at_k(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
    *,
    model_id: str,
) -> None:
    """F5: pass@k by translation direction (per model).

    CRITICAL: Filters to true sample files using the ``is_sample`` field
    (determined by filename ``-sN`` pattern), NOT the ``sample_id`` JSON field.
    """
    # Only true sample files for this model (all directions)
    sample_records = [
        r for r in records
        if r.get("is_sample", False) and r.get("model") == model_id
    ]

    if not sample_records:
        print(f"  WARNING: No sample records for {MODEL_DISPLAY_SHORT[model_id]} -- skipping F5.")
        return

    if verbose:
        print(f"  Sample records for pass@k ({MODEL_DISPLAY_SHORT[model_id]}): {len(sample_records)}")

    # Group by (kernel, direction), collect pass/fail per sample
    kernel_dir_samples: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in sample_records:
        key = (r["kernel"], r["direction"])
        kernel_dir_samples[key].append(1 if r["overall_status"] == "PASS" else 0)

    # Compute pass@k per direction
    dir_pass1: dict[str, list[float]] = defaultdict(list)
    dir_pass3: dict[str, list[float]] = defaultdict(list)

    for (kernel, direction), results in kernel_dir_samples.items():
        n = len(results)
        c = sum(results)

        # pass@1: unbiased estimator = c/n
        p1 = 1 - comb(n - c, 1) / comb(n, 1) if n >= 1 else 0

        # pass@k where k = min(n, 3)
        k = min(n, 3)
        if n >= k and comb(n, k) > 0:
            pk = 1 - comb(n - c, k) / comb(n, k) if comb(n - c, k) <= comb(n, k) else 1.0
        else:
            pk = 1.0 if c > 0 else 0.0

        dir_pass1[direction].append(p1)
        dir_pass3[direction].append(pk)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    sample_dirs = {r["direction"] for r in sample_records}
    x_dirs = list(ALL_DIRECTIONS)  # always 8 directions for consistent layout
    x_pos = np.arange(len(x_dirs))
    width = 0.35

    avg_pass1 = [np.mean(dir_pass1[d]) if dir_pass1[d] else 0.0 for d in x_dirs]
    avg_pass3 = [np.mean(dir_pass3[d]) if dir_pass3[d] else 0.0 for d in x_dirs]

    bars1 = ax.bar(x_pos - width / 2, avg_pass1, width,
                   label="pass@1", color=STATUS_COLORS["BUILD_FAIL"],
                   edgecolor="white")
    bars3 = ax.bar(x_pos + width / 2, avg_pass3, width,
                   label="pass@3", color=STATUS_COLORS["VERIFY_FAIL"],
                   edgecolor="white")

    for bars in [bars1, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=8, color=PALETTE["charcoal"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([DIRECTION_LABELS.get(d, d) for d in x_dirs],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Average pass@k")
    ax.set_ylim(0, 1.15)

    # Subtitle noting number of directions with sample data
    n_sample_dirs = len([d for d in x_dirs if dir_pass1[d]])
    n_total_dirs = len(ALL_DIRECTIONS)
    subtitle = f"({n_sample_dirs} of {n_total_dirs} directions with sample data)" if n_sample_dirs < n_total_dirs else ""

    model_name = MODEL_DISPLAY_SHORT[model_id]
    title = f"pass@k by Translation Direction\n({model_name})"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, output_dir, f"f5_pass_at_k_by_direction_{MODEL_SLUG[model_id]}")
    plt.close(fig)


# ===================================================================
# F6: Cross-Suite Pass Rate Comparison
# ===================================================================


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


# Suite bar colors (Okabe-Ito based)
SUITE_COLORS: dict[str, str] = {
    "rodinia":  OKABE_ITO["blue"],
    "xsbench":  OKABE_ITO["orange"],
    "rsbench":  OKABE_ITO["green"],
    "mixbench": OKABE_ITO["vermillion"],
    "hecbench": OKABE_ITO["purple"],
}


def generate_f6_cross_suite_comparison(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
    *,
    model_id: str,
) -> None:
    """F6: Cross-suite aggregate pass rate comparison bar chart (per model)."""
    l0_records = get_canonical_l0(records)
    directions = list(ALL_DIRECTIONS)  # always 8 directions for consistent layout
    std_records = [r for r in l0_records if r["direction"] in directions and r.get("model") == model_id]

    if not std_records:
        print(f"  WARNING: No L0 records for {MODEL_DISPLAY_SHORT[model_id]} -- skipping F6.")
        return

    # Compute per-suite stats
    suite_stats: list[dict] = []
    for suite in SUITE_ORDER:
        suite_recs = [r for r in std_records if r.get("suite") == suite]
        total = len(suite_recs)
        passed = sum(1 for r in suite_recs if r["overall_status"] == "PASS")
        rate = passed / total if total > 0 else 0
        ci_low, ci_high = _wilson_ci(passed, total)
        suite_stats.append({
            "suite": suite,
            "total": total,
            "pass": passed,
            "rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    if verbose:
        for s in suite_stats:
            print(
                f"  {s['suite']}: {s['pass']}/{s['total']} = {s['rate']:.1%} "
                f"[{s['ci_low']:.1%}, {s['ci_high']:.1%}]"
            )

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x_pos = np.arange(len(SUITE_ORDER))
    rates = [s["rate"] for s in suite_stats]
    bar_colors = [SUITE_COLORS.get(s["suite"], "#999999") for s in suite_stats]

    # Asymmetric error bars for Wilson CI
    yerr_low = [s["rate"] - s["ci_low"] for s in suite_stats]
    yerr_high = [s["ci_high"] - s["rate"] for s in suite_stats]

    bars = ax.bar(
        x_pos, rates, width=0.6,
        color=bar_colors, edgecolor="black", linewidth=0.5,
        yerr=[yerr_low, yerr_high],
        capsize=3, error_kw={"linewidth": 1.0, "color": "black"},
    )

    # Compute per-suite failure breakdowns for annotation
    suite_failures: dict[str, dict[str, int]] = {}
    for suite in SUITE_ORDER:
        suite_recs = [r for r in std_records if r.get("suite") == suite]
        fails: dict[str, int] = defaultdict(int)
        for r in suite_recs:
            st = r.get("overall_status", "UNKNOWN")
            if st != "PASS":
                fails[st] += 1
        suite_failures[suite] = dict(fails)

    # Annotate each bar with "pass/total" and failure breakdown for 0% suites
    for i, (bar, s) in enumerate(zip(bars, suite_stats)):
        h = bar.get_height()
        x_center = bar.get_x() + bar.get_width() / 2
        # pass/total label above bar
        ax.text(
            x_center, h + max(yerr_high[i], 0.02) + 0.02,
            f"{s['pass']}/{s['total']}", ha="center", va="bottom",
            fontsize=8, color=PALETTE["charcoal"],
        )
        # For 0% suites, show failure breakdown inside the bar area
        if s["pass"] == 0 and s["total"] > 0:
            fails = suite_failures.get(s["suite"], {})
            parts = []
            for ftype in ["BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL", "ERROR"]:
                cnt = fails.get(ftype, 0)
                if cnt > 0:
                    label = {"BUILD_FAIL": "build fail",
                             "RUN_FAIL": "run fail",
                             "VERIFY_FAIL": "verify fail",
                             "EXTRACTION_FAIL": "extract fail",
                             "ERROR": "API error"}[ftype]
                    parts.append(f"{cnt} {label}")
            if parts:
                ax.text(
                    x_center, 0.06,
                    "\n".join(parts),
                    ha="center", va="bottom",
                    fontsize=5.5, fontstyle="italic", color="#444444",
                    linespacing=1.2,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [SUITE_DISPLAY.get(s, s) for s in SUITE_ORDER],
        fontsize=8,
    )
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.15)
    model_name = MODEL_DISPLAY_SHORT[model_id]
    ax.set_title(
        f"L0 Pass Rate by Suite ({model_name})",
        fontsize=9, fontweight="bold", pad=8,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    # Conditional explanatory footnote: only show if any suite has 0% pass rate
    zero_suites = [s["suite"] for s in suite_stats if s["pass"] == 0 and s["total"] > 0]
    if zero_suites:
        zero_names = "/".join(SUITE_DISPLAY.get(s, s) for s in zero_suites)
        # Build footnote lines dynamically — avoid hardcoding suite-specific rates
        footnote_lines = (
            f"{zero_names} 0%: complex multi-file programs with inter-module\n"
            "dependencies \u2014 all translations failed (primarily build/linker errors)."
        )
        # Only mention HeCBench single-file advantage if HeCBench is NOT among the zero suites
        if "hecbench" not in zero_suites:
            hecbench_stat = next((s for s in suite_stats if s["suite"] == "hecbench"), None)
            if hecbench_stat and hecbench_stat["total"] > 0:
                footnote_lines += (
                    f"\nHeCBench {hecbench_stat['rate']:.0%}: single-file kernels."
                )
        fig.text(
            0.5, 0.02,
            footnote_lines,
            ha="center", va="top", fontsize=6.5, fontstyle="italic", color="#555555",
        )
        fig.subplots_adjust(bottom=0.18)
    _save_figure(fig, output_dir, f"f6_cross_suite_comparison_{MODEL_SLUG[model_id]}")
    plt.close(fig)


# ===================================================================
# F7: Augmentation Robustness Line Chart
# ===================================================================


def generate_f7_augmentation(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
) -> None:
    """F7: Augmentation robustness -- pass rate vs. augmentation level (dual-model)."""
    levels = [0, 1, 2, 3, 4]
    level_labels = ["L0\n(original)", "L1", "L2", "L3", "L4\n(max)"]

    # Canonical L0 + L1-L4 augmented, cuda-to-omp direction only
    l0_canonical = get_canonical_l0(records)
    augmented = [r for r in records if not r.get("is_sample", False)]
    combined = l0_canonical + augmented
    c2o_all = [r for r in combined
               if r.get("direction") == "cuda-to-omp"]

    if not c2o_all:
        print("  WARNING: No CUDA-to-OMP augmentation records -- skipping F7.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Annotation offsets: first model below line, second above
    annotation_configs = [(-5, "top"), (3, "bottom")]
    n_kernels = 34  # default; overwritten by first model's L0 count
    model_idx = 0

    for model_id in MODEL_COLORS:
        model_recs = [r for r in c2o_all if r["model"] == model_id]
        if not model_recs:
            continue

        # Aggregate pass counts and totals per level
        level_pass: dict[int, int] = defaultdict(int)
        level_total: dict[int, int] = defaultdict(int)
        for r in model_recs:
            lvl = r.get("augment_level", 0)
            level_total[lvl] += 1
            if r.get("overall_status") == "PASS":
                level_pass[lvl] += 1

        pass_counts = [level_pass.get(lvl, 0) for lvl in levels]
        totals_list = [level_total.get(lvl, 0) for lvl in levels]
        rates = [c / max(t, 1) * 100 for c, t in zip(pass_counts, totals_list)]

        # Use first model's L0 count for N in title
        if model_idx == 0:
            n_kernels = level_total.get(0, 34)

        # Wilson CIs for error bars
        ci_results = [_wilson_ci(level_pass.get(lvl, 0), level_total.get(lvl, 0))
                      for lvl in levels]
        ci_low_pct = [ci[0] * 100 for ci in ci_results]
        ci_high_pct = [ci[1] * 100 for ci in ci_results]
        yerr_low = [r - lo for r, lo in zip(rates, ci_low_pct)]
        yerr_high = [hi - r for r, hi in zip(rates, ci_high_pct)]

        if verbose:
            print(f"  {MODEL_DISPLAY_SHORT[model_id]}:")
            for lvl, pc, tot, rate in zip(levels, pass_counts, totals_list, rates):
                print(f"    L{lvl}: {pc}/{tot} = {rate:.1f}%")

        # Draw with errorbar
        fmt_str, _ = MODEL_LINESTYLE[model_id]
        ax.errorbar(
            levels, rates,
            yerr=[yerr_low, yerr_high],
            fmt=fmt_str,
            color=MODEL_COLORS[model_id],
            linewidth=1.8,
            markersize=7,
            label=MODEL_DISPLAY_SHORT[model_id],
            capsize=3,
            capthick=1.0,
        )

        # Per-point annotations
        y_offset, va = annotation_configs[min(model_idx, 1)]
        color = MODEL_COLORS[model_id]
        for lvl, rate in zip(levels, rates):
            ax.annotate(
                f"{rate:.1f}%",
                xy=(lvl, rate),
                xytext=(lvl, rate + y_offset),
                fontsize=7, ha="center", va=va, color=color,
                fontweight="bold",
            )

        model_idx += 1

    ax.set_xticks(levels)
    ax.set_xticklabels(level_labels)
    ax.set_xlabel("Augmentation Level")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(-2, 85)
    ax.set_xlim(-0.2, 4.7)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title(
        "Augmentation Robustness: Pass Rate across L0\u2013L4\n"
        f"(CUDA\u2192OpenMP, N={n_kernels} kernels/model, seed=42+L)",
        fontsize=10,
    )

    _save_figure(fig, output_dir, "f7_augmentation_robustness")
    plt.close(fig)


# ===================================================================
# C.1: Self-Repair Transition Matrix (was c-series C.2a)
# ===================================================================


def generate_c1_repair_transitions(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
) -> None:
    """C.1: Transition matrix -- initial failure -> final outcome."""
    multi = [r for r in records if r["total_attempts"] > 1 and r["attempts"]]

    if not multi:
        print("  WARNING: No multi-attempt results found -- skipping C.1.")
        return

    initial_statuses = ["BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL"]
    final_statuses = ["PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL"]

    transitions: Counter = Counter()
    for r in multi:
        initial = _classify_initial_status(r["attempts"][0])
        final = r["overall_status"]
        if initial in initial_statuses:
            transitions[(initial, final)] += 1

    if not transitions:
        print("  WARNING: No transitions found -- skipping C.1.")
        return

    active_initial = [s for s in initial_statuses
                      if any(transitions.get((s, f), 0) > 0 for f in final_statuses)]
    active_final = [s for s in final_statuses
                    if any(transitions.get((i, s), 0) > 0 for i in initial_statuses)]

    if not active_initial or not active_final:
        print("  WARNING: Empty transition matrix -- skipping C.1.")
        return

    matrix = np.zeros((len(active_initial), len(active_final)))
    for i, init in enumerate(active_initial):
        for j, fin in enumerate(active_final):
            matrix[i, j] = transitions.get((init, fin), 0)

    if verbose:
        print(f"  Transition matrix: {len(active_initial)} x {len(active_final)}")
        print(f"  Total multi-attempt records: {len(multi)}")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "yor", ["#FFFDE7", OKABE_ITO["orange"], OKABE_ITO["vermillion"]], N=256,
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(active_final)))
    ax.set_xticklabels([s.replace("_", "\n") for s in active_final], fontsize=8)
    ax.set_yticks(range(len(active_initial)))
    ax.set_yticklabels([s.replace("_", "\n") for s in active_initial], fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if matrix[i, j] > matrix.max() * 0.5 else PALETTE["charcoal"])

    ax.set_xlabel("Final Status", fontsize=9)
    ax.set_ylabel("Initial Status (Attempt 1)", fontsize=9)
    ax.set_title("Self-Repair Transition Matrix", fontsize=10, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)

    fig.tight_layout()
    _save_figure(fig, output_dir, "c1_repair_transition_matrix")
    plt.close(fig)


# ===================================================================
# C.2: Self-Repair Rate by Direction (was c-series C.2b)
# ===================================================================


def generate_c2_repair_rate(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
) -> None:
    """C.2: Self-repair rate by translation direction."""
    multi = [r for r in records if r["total_attempts"] > 1 and r["attempts"]]

    dir_stats: dict[str, dict] = defaultdict(
        lambda: {"initially_failing": 0, "repaired": 0, "attempts_to_success": []}
    )

    for r in multi:
        d = r["direction"]
        if d not in DIRECTIONS:
            continue
        initial = _classify_initial_status(r["attempts"][0])
        if initial != "PASS":
            dir_stats[d]["initially_failing"] += 1
            if r["overall_status"] == "PASS":
                dir_stats[d]["repaired"] += 1
                dir_stats[d]["attempts_to_success"].append(r["total_attempts"])

    if not dir_stats:
        print("  WARNING: No multi-attempt directional data found -- skipping C.2.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x_dirs = [d for d in DIRECTIONS if d in dir_stats]
    x_pos = np.arange(len(x_dirs))
    repair_rates = []
    avg_attempts = []

    for d in x_dirs:
        s = dir_stats[d]
        rate = s["repaired"] / s["initially_failing"] if s["initially_failing"] > 0 else 0
        repair_rates.append(rate)
        avg = np.mean(s["attempts_to_success"]) if s["attempts_to_success"] else 0
        avg_attempts.append(avg)

    if verbose:
        for d, rate in zip(x_dirs, repair_rates):
            s = dir_stats[d]
            print(f"  {d}: {s['repaired']}/{s['initially_failing']} = {rate:.1%}")

    bars = ax.bar(x_pos, repair_rates, color=STATUS_COLORS["PASS"],
                  edgecolor="white", linewidth=0.5, width=0.6)

    for i, (bar, avg, d) in enumerate(zip(bars, avg_attempts, x_dirs)):
        s = dir_stats[d]
        label = f"{s['repaired']}/{s['initially_failing']}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                label, ha="center", va="bottom", fontsize=7, color=PALETTE["charcoal"])
        if avg > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"avg {avg:.1f}\nattempts", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([DIRECTION_LABELS.get(d, d) for d in x_dirs],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Repair Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Self-Repair Rate by Translation Direction",
                 fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, output_dir, "c2_repair_rate_by_direction")
    plt.close(fig)


# ===================================================================
# C.3: Transform Frequency Heatmap (was c-series C.3)
# ===================================================================


def generate_c3_transform_frequency(
    project_root: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """C.3: Heatmap of transform frequency per kernel."""
    aug_records = load_augmentation_eval_results(project_root)

    kernel_transform_counts: dict[str, Counter] = defaultdict(Counter)

    for entry in aug_records:
        spec_id = entry.get("spec_id", "")
        parts = spec_id.split("-")
        suite = _extract_suite(spec_id)
        if suite in ("rodinia", "xsbench", "hecbench", "rsbench", "mixbench"):
            kernel = "-".join(parts[1:-1]) if len(parts) > 2 else parts[-1]
        else:
            kernel = spec_id

        transforms_applied = entry.get("transforms_applied", {})
        for filename, transform_list in transforms_applied.items():
            for transform_name in transform_list:
                kernel_transform_counts[kernel][transform_name] += 1

    if not kernel_transform_counts:
        print("  WARNING: No transform data found -- skipping C.3.")
        return

    all_transforms = sorted(set(
        t for counts in kernel_transform_counts.values() for t in counts
    ))
    if not all_transforms:
        print("  WARNING: No transforms applied in any entry -- skipping C.3.")
        return

    sorted_kernels = sorted(kernel_transform_counts.keys())

    matrix = np.zeros((len(sorted_kernels), len(all_transforms)))
    for i, kernel in enumerate(sorted_kernels):
        for j, transform in enumerate(all_transforms):
            matrix[i, j] = kernel_transform_counts[kernel].get(transform, 0)

    if verbose:
        print(f"  {len(sorted_kernels)} kernels x {len(all_transforms)} transforms")

    fig_height = max(4, len(sorted_kernels) * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(5, fig_height))

    saffron_cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_saffron", ["white", OKABE_ITO["orange"]], N=256,
    )

    im = ax.imshow(matrix, cmap=saffron_cmap, aspect="auto")

    # Abbreviate long transform names
    transform_abbrevs = {
        "ArithmeticTransform": "Arith\nTransform",
        "PointerArithmeticToArrayIndex": "PtrArith\n\u2192ArrIdx",
        "ChangeFunctionNames": "Change\nFuncNames",
        "ChangeNames": "Change\nNames",
        "SwapCondition": "Swap\nCondition",
        "TypedefExpansion": "Typedef\nExpansion",
    }
    x_labels = [transform_abbrevs.get(t, t) for t in all_transforms]

    ax.set_xticks(range(len(all_transforms)))
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(sorted_kernels)))
    ax.set_yticklabels(sorted_kernels, fontsize=7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center",
                        va="center", fontsize=6,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else PALETTE["charcoal"])

    ax.set_title("Transform Frequency Per Kernel",
                 fontsize=10, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax, label="Candidate Sites", shrink=0.8)

    fig.tight_layout()
    _save_figure(fig, output_dir, "c3_transform_frequency")
    plt.close(fig)


# ===================================================================
# C.4: HeCBench Kernel Selection Funnel (was f-series F4)
# ===================================================================


def generate_c4_selection_funnel(
    output_dir: Path, verbose: bool,
) -> None:
    """C.4: HeCBench kernel selection pipeline funnel diagram."""
    stages = HECBENCH_FUNNEL_STAGES
    values = [s[1] for s in stages]

    if verbose:
        for lbl, val, exc in stages:
            exc_str = f" ({exc})" if exc else ""
            print(f"  {lbl.replace(chr(10), ' ')}: {val}{exc_str}")

    n = len(stages)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    max_val = max(values)
    y_positions = list(range(n - 1, -1, -1))

    stage_colors = [
        PALETTE["teal_tint"],
        PALETTE["teal_tint"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["teal_dark"],
    ]

    for i, (label, value, exc) in enumerate(stages):
        y = y_positions[i]
        bar_width = value / max_val * 0.85
        ax.barh(
            y, bar_width, height=0.65,
            left=(1 - bar_width) / 2,
            color=stage_colors[i],
            edgecolor=PALETTE["charcoal"],
            linewidth=0.8,
        )
        text_color = _text_color_for_bg(stage_colors[i])
        ax.text(
            0.5, y, f"{value}",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color,
        )
        ax.text(
            -0.02, y, label,
            ha="right", va="center",
            fontsize=8, color=PALETTE["charcoal"],
        )
        if exc:
            ax.text(
                1.02, y, exc,
                ha="left", va="center",
                fontsize=8, color=PALETTE["rose_dark"], fontstyle="italic",
            )

    for i in range(n - 1):
        y_from = y_positions[i]
        y_to = y_positions[i + 1]
        ax.annotate(
            "", xy=(0.5, y_to + 0.35), xytext=(0.5, y_from - 0.35),
            arrowprops=dict(
                arrowstyle="-|>", color=PALETTE["slate"],
                lw=1.2, shrinkA=0, shrinkB=0,
            ),
        )

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.6, n - 0.4)
    ax.axis("off")
    ax.set_title(
        "HeCBench Kernel Selection Pipeline",
        fontsize=9, fontweight="bold", pad=8,
    )

    fig.tight_layout()
    _save_figure(fig, output_dir, "c4_selection_funnel")
    plt.close(fig)


# ===================================================================
# T2: Model Comparison LaTeX Table
# ===================================================================


def generate_t2_model_table(
    records: list[dict],
    output_dir: Path,
    verbose: bool,
) -> None:
    """T2: Model comparison LaTeX table -- per-model rows, all suites."""
    l0_records = get_canonical_l0(records)
    std_records = [r for r in l0_records if r["direction"] in DIRECTIONS]

    dir_headers = [
        DIRECTION_LABELS.get(d, d).replace("\u2192", r"$\to$")
        for d in DIRECTIONS
    ]

    lines = [
        r"\begin{tabular}{l" + "r" * (1 + len(DIRECTIONS)) + "}",
        r"\toprule",
        r"Model & Overall & " + " & ".join(dir_headers) + r" \\",
        r"\midrule",
    ]

    for model_id, display in MODEL_DISPLAY_SHORT.items():
        m_records = [r for r in std_records if r["model"] == model_id]
        m_total = len(m_records)
        if m_total == 0:
            continue
        m_pass = sum(1 for r in m_records if r["overall_status"] == "PASS")
        m_rate = m_pass / m_total * 100
        cells = [f"{m_pass}/{m_total} ({m_rate:.1f}\\%)"]
        for d in DIRECTIONS:
            d_recs = [r for r in m_records if r["direction"] == d]
            d_total = len(d_recs)
            d_pass = sum(1 for r in d_recs if r["overall_status"] == "PASS")
            rate = d_pass / d_total * 100 if d_total > 0 else 0
            cells.append(f"{d_pass}/{d_total} ({rate:.1f}\\%)")
        lines.append(f"{display} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    path = output_dir / "t2_model_comparison.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def _fmt_pval(p: float) -> str:
    """Format a p-value for LaTeX tables."""
    return f"{p:.3f}" if p >= 0.001 else "$<$0.001"


def _write_tex_table(
    path: Path, col_spec: str, header: str, body_lines: list[str],
) -> None:
    """Write a LaTeX tabular environment to a .tex file."""
    rows = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(rows) + "\n")
    print(f"  Saved: {path}")


# ===================================================================
# T1: Overall Pass Rates LaTeX Table
# ===================================================================


def generate_t1_overall_pass(
    project_root: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """T1: Overall pass rates with status breakdown and 95% Wilson CIs."""
    findings_dir = project_root / "results" / "analysis"
    statuses = ["PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL"]
    status_headers = ["PASS", "Build", "Run", "Verify", "Extract"]

    body = []
    for model_id, display in MODEL_DISPLAY_SHORT.items():
        path = findings_dir / f"quantitative_findings_{model_id}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        canon = data["canonical"]
        sc = canon["failure_taxonomy"]["status_counts"]
        total = canon["failure_taxonomy"]["total_records"]
        agg = canon["aggregate_pass_rates"]["overall"]
        rate = agg["value"] * 100
        ci_lo = agg["ci_lower"] * 100
        ci_hi = agg["ci_upper"] * 100

        cells = [str(sc.get(s, 0)) for s in statuses]
        cells.append(str(total))
        cells.append(f"{rate:.1f}\\% [{ci_lo:.1f}, {ci_hi:.1f}]")
        body.append(f"{display} & " + " & ".join(cells) + r" \\")

    _write_tex_table(
        output_dir / "t1_overall_pass.tex",
        "l" + "r" * len(statuses) + "rr",
        r"Model & " + " & ".join(status_headers) + r" & Total & Rate [95\% CI] \\",
        body,
    )


# Mapping from model_id to paper_data filename stem (inconsistent naming convention)
PAPER_DATA_STEMS: dict[str, str] = {
    "together-qwen-3.5-397b-a17b": "paper_data_together-qwen-3.5-397b-a17b",
    "azure-gpt-5.4": "paper_data_azure_gpt54",
    "azure-gpt-5.3-codex": "paper_data_azure-gpt-5.3-codex",
}


# ===================================================================
# T3: Pass@k Rates LaTeX Table
# ===================================================================


def generate_t3_passk(
    project_root: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """T3: Pass@k rates and task classification per model."""
    findings_dir = project_root / "results" / "analysis"

    body = []
    for model_id, display in MODEL_DISPLAY_SHORT.items():
        stem = PAPER_DATA_STEMS.get(model_id)
        if stem is None:
            continue
        pd_path = findings_dir / f"{stem}.json"
        qf_path = findings_dir / f"quantitative_findings_{model_id}.json"
        if not pd_path.exists() or not qf_path.exists():
            continue

        paper_data = json.loads(pd_path.read_text())
        qf = json.loads(qf_path.read_text())

        agg = paper_data["passk_campaign"]["aggregate_passk"]
        tc = qf["canonical"]["pass_at_k"]["task_classification"]

        cells = [
            str(agg["n_tasks"]),
            f"{agg['pass@1_macro_avg'] * 100:.1f}\\%",
            f"{agg['pass@3_macro_avg'] * 100:.1f}\\%",
            str(tc["always_pass"]),
            str(tc["noisy_fail"]),
            str(tc["hard_fail"]),
        ]
        body.append(f"{display} & " + " & ".join(cells) + r" \\")

    _write_tex_table(
        output_dir / "t3_passk.tex",
        "lrrrrrrr",
        r"Model & Tasks & pass@1 & pass@3 & Always-pass & Noisy & Hard-fail \\",
        body,
    )


# ===================================================================
# T4: Augmentation Rates LaTeX Table
# ===================================================================


def generate_t4_augmentation(
    project_root: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """T4: Augmentation level pass rates with chi-squared p-values."""
    findings_dir = project_root / "results" / "analysis"
    stats_path = findings_dir / "statistical_analysis.json"
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text())
    curves = stats["augmentation_curves"]
    chi2_list = stats["chi2_augmentation_by_model"]
    chi2_by_model = {entry["group"]: entry for entry in chi2_list}

    levels = ["L0", "L1", "L2", "L3", "L4"]

    body = []
    for model_id, display in MODEL_DISPLAY_SHORT.items():
        if model_id not in curves:
            continue
        mc = curves[model_id]
        chi2_entry = chi2_by_model.get(model_id, {})
        p_corr = chi2_entry.get("p_corrected_bonferroni", float("nan"))

        cells = []
        for lv in levels:
            if lv in mc:
                cells.append(f"{mc[lv]['rate'] * 100:.1f}\\%")
            else:
                cells.append("--")
        cells.append(_fmt_pval(p_corr))
        body.append(f"{display} & " + " & ".join(cells) + r" \\")

    _write_tex_table(
        output_dir / "t4_augmentation.tex",
        "l" + "r" * len(levels) + "r",
        r"Model & " + " & ".join(levels) + r" & $\chi^2$ $p$ (Bonf.) \\",
        body,
    )


# ===================================================================
# T5: Statistical Tests Summary LaTeX Table
# ===================================================================


def generate_t5_stats(
    project_root: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """T5: Pairwise McNemar tests with OR, CI, Cohen's h."""
    findings_dir = project_root / "results" / "analysis"
    stats_path = findings_dir / "statistical_analysis.json"
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text())
    mc = stats["model_comparison"]

    body = []
    for entry in mc["pairwise"]:
        pair = entry["pair"]
        or_val = entry["odds_ratio"]
        ci_lo = entry["or_ci_lower"]
        ci_hi = entry["or_ci_upper"]
        p_corr = entry["p_corrected"]
        h = entry["cohens_h"]
        effect = entry["effect_size"]

        cells = [
            f"{or_val:.3f}",
            f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            _fmt_pval(p_corr),
            f"{h:.3f}",
            effect,
        ]
        body.append(f"{pair} & " + " & ".join(cells) + r" \\")

    omnibus_chi2 = mc["omnibus_chi2"]
    omnibus_p = mc["omnibus_p"]
    cramers_v = mc["cramers_v"]
    interp = mc["cramers_v_interpretation"]
    body.append(r"\midrule")
    body.append(
        f"Omnibus ($\\chi^2$) & {omnibus_chi2:.2f} & -- & {_fmt_pval(omnibus_p)} & "
        f"$V$={cramers_v:.3f} & {interp}" + r" \\"
    )

    _write_tex_table(
        output_dir / "t5_stats.tex",
        "lrrrrl",
        r"Pair & OR & 95\% CI & $p$ (Bonf.) & Cohen's $h$ & Interpretation \\",
        body,
    )


# ===================================================================
# Figure registry and CLI
# ===================================================================

FIGURE_REGISTRY: dict[str, str] = {
    "F2":  "f2_repo_vs_kernel",
    "F3":  "f3_kernel_heatmap",
    "F4":  "f4_failure_taxonomy",
    "F5":  "f5_pass_at_k",
    "F6":  "f6_cross_suite_comparison",
    "F7":  "f7_augmentation",
    "C.1": "c1_repair_transitions",
    "C.2": "c2_repair_rate",
    "C.3": "c3_transform_frequency",
    "C.4": "c4_selection_funnel",
    "T1":  "t1_overall_pass",
    "T2":  "t2_model_table",
    "T3":  "t3_passk",
    "T4":  "t4_augmentation",
    "T5":  "t5_stats",
}

# Which figures need eval data loaded
EVAL_FIGURES = {"F3", "F4", "F5", "F6", "F7", "C.1", "C.2", "T2"}


def _normalize_figure_id(raw: str) -> str | None:
    """Normalize user input to a figure registry key, or None if unknown."""
    s = raw.strip().upper()
    if s in FIGURE_REGISTRY:
        return s
    # Try adding "C." prefix for appendix
    if s.startswith("C") and not s.startswith("C."):
        candidate = "C." + s[1:]
        if candidate in FIGURE_REGISTRY:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the NeurIPS 2026 paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Figure IDs: " + ", ".join(FIGURE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--project-root", type=Path, required=True,
        help="Absolute path to project root",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: {project-root}/docs/paper/figures/)",
    )
    parser.add_argument(
        "--figure", default="all",
        help="Figure ID to generate (F2, C.1, T2, etc.) or 'all' (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
    )
    args = parser.parse_args()

    setup_rcparams()

    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir or (project_root / "docs" / "paper" / "figures")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which figures to generate
    if args.figure.strip().lower() == "all":
        requested = set(FIGURE_REGISTRY.keys())
    else:
        fig_id = _normalize_figure_id(args.figure)
        if fig_id is None:
            print(
                f"ERROR: Unknown figure '{args.figure}'. "
                f"Valid: {', '.join(FIGURE_REGISTRY.keys())}, or 'all'",
                file=sys.stderr,
            )
            sys.exit(1)
        requested = {fig_id}

    # Load eval data if needed
    needs_eval = requested & EVAL_FIGURES
    records: list[dict] = []
    if needs_eval:
        print("Loading evaluation results...")
        records = load_eval_results(project_root)
        if not records:
            print(
                f"WARNING: No eval data found in {project_root / 'results' / 'evaluation'} -- "
                f"skipping figures {sorted(needs_eval)}",
                file=sys.stderr,
            )
            requested -= EVAL_FIGURES

    # Filter out sample records for figures that need only base records
    base_count = sum(1 for r in records if not r.get("is_sample", False))
    sample_count = sum(1 for r in records if r.get("is_sample", False))
    if records:
        print(f"  Total: {len(records)} records ({base_count} base, {sample_count} sample)")

    print(f"Output: {output_dir}")
    print(f"Generating: {sorted(requested)}")
    print()

    v = args.verbose

    # --- Main body figures ---

    if "F2" in requested:
        print("Generating F2: Repo vs Kernel-Level Counts...")
        generate_f2_repo_vs_kernel(output_dir, v)
        print()

    if "F3" in requested:
        for mid in MODEL_SLUG:
            print(f"Generating F3: Per-Kernel Translation Outcomes ({MODEL_DISPLAY_SHORT[mid]})...")
            generate_f3_kernel_heatmap(records, output_dir, v, project_root=project_root, model_id=mid)
            print()

    if "F4" in requested:
        for mid in MODEL_SLUG:
            print(f"Generating F4: Failure Taxonomy ({MODEL_DISPLAY_SHORT[mid]})...")
            generate_f4_failure_taxonomy(records, output_dir, v, model_id=mid)
            print()

    if "F5" in requested:
        for mid in MODEL_SLUG:
            print(f"Generating F5: pass@k by Direction ({MODEL_DISPLAY_SHORT[mid]})...")
            generate_f5_pass_at_k(records, output_dir, v, model_id=mid)
            print()

    if "F6" in requested:
        for mid in MODEL_SLUG:
            print(f"Generating F6: Cross-Suite Comparison ({MODEL_DISPLAY_SHORT[mid]})...")
            generate_f6_cross_suite_comparison(records, output_dir, v, model_id=mid)
            print()

    if "F7" in requested:
        print("Generating F7: Augmentation Robustness (L0-L4)...")
        generate_f7_augmentation(records, output_dir, v)
        print()

    # --- Appendix figures ---

    if "C.1" in requested:
        print("Generating C.1: Self-Repair Transition Matrix...")
        generate_c1_repair_transitions(records, output_dir, v)
        print()

    if "C.2" in requested:
        print("Generating C.2: Self-Repair Rate by Direction...")
        generate_c2_repair_rate(records, output_dir, v)
        print()

    if "C.3" in requested:
        print("Generating C.3: Transform Frequency Heatmap...")
        generate_c3_transform_frequency(project_root, output_dir, v)
        print()

    if "C.4" in requested:
        print("Generating C.4: HeCBench Selection Funnel...")
        generate_c4_selection_funnel(output_dir, v)
        print()

    # --- Tables ---

    if "T1" in requested:
        print("Generating T1: Overall Pass Rates LaTeX Table...")
        generate_t1_overall_pass(project_root, output_dir, v)
        print()

    if "T2" in requested:
        print("Generating T2: Model Comparison LaTeX Table...")
        generate_t2_model_table(records, output_dir, v)
        print()

    if "T3" in requested:
        print("Generating T3: Pass@k Rates LaTeX Table...")
        generate_t3_passk(project_root, output_dir, v)
        print()

    if "T4" in requested:
        print("Generating T4: Augmentation Rates LaTeX Table...")
        generate_t4_augmentation(project_root, output_dir, v)
        print()

    if "T5" in requested:
        print("Generating T5: Statistical Tests Summary LaTeX Table...")
        generate_t5_stats(project_root, output_dir, v)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
