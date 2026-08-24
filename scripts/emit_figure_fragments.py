#!/usr/bin/env python3
"""Emit corrected numbers into the manuscript's hand-tuned TikZ figure fragments (ruling D14).

This is the FORWARD direction of the figure chain: it rewrites ONLY the number-carrying
constructs inside the existing fragments under
``docs/paper/NeurIPS_ready_version/figures/figures_tek_version/``, leaving all styling, layout,
comments, and the commented-out prior revisions untouched. The LIVE line ranges and the
number-carrying construct per fragment are documented in that directory's
``FRAGMENT_EDIT_PATHS.md`` (ticket #27).

INDEPENDENCE CONTRACT (ruling D14): this emitter derives its values with its OWN record-walking
and status-derivation code. It deliberately does NOT import the figure code
(``generate_paper_figures``) or the comparator (``verify_figure_fragments``). The comparator is
the sole accepted proof of correctness: two independent codepaths agreeing over the same raw
records is the check. The only shared dependency is the ground-truth data contract
``harness.constants.CORRECTNESS_INELIGIBLE_SPECS`` (never hand-maintained).

Source of truth: the sealed replay ``results/evaluation_final/parbench-final-v1/`` (the corrected
canonical build), filtered to canonical L0 (s0-only), matching the analysis chain's scope.

Currently emits f3 (the kernel x model heatmap). Other fragment families are added one at a time,
each reaching zero comparator mismatches before the next (ruling D14, f3-first cadence).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# CORRECTNESS_INELIGIBLE_SPECS is the ground-truth data contract, never hand-maintained.
from harness.constants import CORRECTNESS_INELIGIBLE_SPECS  # noqa: E402

FIG_DIR = PROJECT_ROOT / "docs/paper/NeurIPS_ready_version/figures/figures_tek_version"
REPLAY_ROOT = PROJECT_ROOT / "results/evaluation_final/parbench-final-v1"

NA = "--"  # fragment code for "no such pair / not evaluated"

# overall_status -> fragment status code. Anything not listed emits NA (matches the analysis
# chain, which only abbreviates these five and treats every other verdict as no-data).
STATUS_CODE = {
    "PASS": "P",
    "BUILD_FAIL": "BF",
    "RUN_FAIL": "RF",
    "VERIFY_FAIL": "VF",
    "EXTRACTION_FAIL": "EF",
}

# --- f3 fragment layout (the fragment's OWN axis order; a structural fact, not a value) --------
F3_COL_DIRECTION = [
    "cuda-to-omp", "omp-to-cuda", "cuda-to-opencl", "opencl-to-cuda",
    "omp-to-opencl", "opencl-to-omp", "cuda-to-omp_target", "omp_target-to-cuda",
]
F3_ROW_KERNEL = [
    "backprop", "bfs", "bptree", "cfd", "dwt2d", "gaussian", "heartwall", "hotspot",
    "hotspot3d", "huffman", "hybridsort", "kmeans", "lavamd", "lud", "mummergpu", "myocyte",
    "nn", "nw", "particlefilter", "pathfinder", "srad", "streamcluster", "xsbench", "rsbench",
    "mixbench", "convolution1d", "floydwarshall", "heat2d", "iso2dfd", "jacobi", "md", "nqueen",
    "page-rank", "scan", "stencil1d",
]
F3_TITLE_MODEL = {
    "qwenshort": "together-qwen-3.5-397b-a17b",
    "gptnew": "azure-gpt-5.4",
    "codex": "azure-gpt-5.3-codex",
}

_HEATCELL = re.compile(r"(\\HeatCell\{(\d+)\}\{(\d+)\}\{)([^}]*)(\})")
_TITLE_MACRO = re.compile(r"title=\{[^}]*\\(qwenshort|gptnew|codex)\{\}")


# --- independent canonical-L0 loader ----------------------------------------------------------

_PARENT_IDENTITY = ("model", "source_spec", "target_spec", "augment_level", "sample_id")


def _promote_identity(rec: dict) -> None:
    """Lift replay identity fields from rec['parent'] when the top-level value is absent.

    Independent reimplementation of the identity-promotion half of the replay loader: a stored
    replay record keeps its corrected top-level overall_status but carries model/source/target/
    sample_id/augment_level only under 'parent'. An existing 0 is a real value and is kept.
    """
    parent = rec.get("parent")
    if isinstance(parent, dict):
        for key in _PARENT_IDENTITY:
            if rec.get(key) in (None, "") and parent.get(key) not in (None, ""):
                rec[key] = parent[key]


def _api_of(spec: str) -> str:
    if spec.endswith("-omp_target"):
        return "omp_target"
    parts = spec.rsplit("-", 1)
    return parts[1] if len(parts) == 2 else "unknown"


def _kernel_of(spec: str) -> str:
    parts = spec.split("-")
    return "-".join(parts[1:-1]) if len(parts) >= 3 else spec


def load_all_records(replay_root: Path = REPLAY_ROOT) -> list[dict]:
    """Return every replay record (base / sample / augmented) with independent code.

    Each record carries model / kernel / direction / overall_status / augment_level / sample_id /
    is_sample. Records whose source or target spec is correctness-ineligible are dropped on load.
    """
    out: list[dict] = []
    for model_dir in sorted(replay_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.glob("*.json")):
            stem = f.stem
            is_sample = bool(re.search(r"-s\d+", stem))
            aug = re.search(r"-L([1-4])$", stem)
            augment_level = int(aug.group(1)) if aug else 0
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            _promote_identity(data)
            src, tgt = data.get("source_spec", ""), data.get("target_spec", "")
            if not src or not tgt:
                continue
            if src in CORRECTNESS_INELIGIBLE_SPECS or tgt in CORRECTNESS_INELIGIBLE_SPECS:
                continue
            out.append({
                "model": data.get("model", model_dir.name),
                "kernel": data.get("kernel") or _kernel_of(src),
                "suite": src.split("-", 1)[0],
                "direction": f"{_api_of(src)}-to-{_api_of(tgt)}",
                "overall_status": data.get("overall_status", "UNKNOWN"),
                "augment_level": augment_level,
                "sample_id": data.get("sample_id"),
                "is_sample": is_sample,
            })
    return out


def load_l0_records(replay_root: Path = REPLAY_ROOT) -> list[dict]:
    """Canonical L0 (s0-only) records: base L0 files if any exist, else the s0 samples."""
    recs = load_all_records(replay_root)
    base = [r for r in recs if not r["is_sample"] and r["augment_level"] == 0]
    if base:
        return base
    return [r for r in recs if r["is_sample"] and r["augment_level"] == 0 and r["sample_id"] == 0]


def load_canonical_l0(replay_root: Path = REPLAY_ROOT) -> dict[tuple[str, str, str], str]:
    """Return {(model, kernel, direction): overall_status} from the canonical L0 record list."""
    grid: dict[tuple[str, str, str], str] = {}
    for r in load_l0_records(replay_root):
        grid[(r["model"], r["kernel"], r["direction"])] = r["overall_status"]
    return grid


def _status_code(overall_status: str | None) -> str:
    return STATUS_CODE.get(overall_status or "", NA)


# --- f3 emitter -------------------------------------------------------------------------------

def is_comment(line: str) -> bool:
    return line.lstrip().startswith("%")


def emit_f3(grid: dict[tuple[str, str, str], str], *, apply: bool) -> dict:
    """Rewrite the third arg of every LIVE \\HeatCell to the canonical status. Styling untouched."""
    path = FIG_DIR / "f3_kernel_model_heatmap_unified.tex"
    lines = path.read_text().splitlines(keepends=True)
    current_model: str | None = None
    changed = 0
    cells = 0

    for i, line in enumerate(lines):
        if is_comment(line):
            continue
        tm = _TITLE_MACRO.search(line)
        if tm:
            current_model = F3_TITLE_MODEL[tm.group(1)]
            continue
        if "\\HeatCell{" not in line:
            continue

        def repl(m: re.Match, current_model: str | None = current_model, line: str = line) -> str:
            nonlocal changed, cells
            cells += 1
            if current_model is None:
                raise ValueError(f"HeatCell before any panel title: {line!r}")
            col, row = int(m.group(2)), int(m.group(3))
            direction = F3_COL_DIRECTION[col]
            kernel = F3_ROW_KERNEL[row]
            want = _status_code(grid.get((current_model, kernel, direction)))
            if want != m.group(4):
                changed += 1
            return f"{m.group(1)}{want}{m.group(5)}"

        lines[i] = _HEATCELL.sub(repl, line)

    if apply:
        path.write_text("".join(lines))
    return {"fragment": "f3", "live_cells": cells, "changed": changed, "applied": apply}


# --- f4 emitter -------------------------------------------------------------------------------

# ffourpanel args 3-7 carry these five statuses in this order (per-direction (dir,count) lists).
F4_STATUS_ORDER = ["PASS", "BUILD_FAIL", "RUN_FAIL", "EXTRACTION_FAIL", "VERIFY_FAIL"]
F4_DIRECTIONS = F3_COL_DIRECTION[:6]  # f4 covers the 6 canonical directions only
F4_TITLE_MODEL = {
    "qwenshort": "together-qwen-3.5-397b-a17b",
    "gptnew": "azure-gpt-5.4",
    "codex": "azure-gpt-5.3-codex",
}


def _line_of(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start:end if end != -1 else len(text)]


def _brace_group(s: str, i: int) -> tuple[int, int]:
    """s[i] must be '{'. Return (inner_start, inner_end) of the balanced group."""
    assert s[i] == "{"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return i + 1, j
    raise ValueError("unbalanced braces")


def emit_f4(records: list[dict], *, apply: bool) -> dict:
    """Rewrite args 3-7 of each LIVE \\ffourpanel with canonical per-(model,direction) counts."""
    path = FIG_DIR / "f4_failure_taxonomy_unified.tex"
    text = path.read_text()

    # counts[(model, status, dir_idx)] over canonical L0
    counts: dict[tuple[str, str, int], int] = {}
    for model in F4_TITLE_MODEL.values():
        for di, direction in enumerate(F4_DIRECTIONS):
            recs = [r for r in records if r["model"] == model and r["direction"] == direction]
            for status in F4_STATUS_ORDER:
                counts[(model, status, di)] = sum(1 for r in recs if r["overall_status"] == status)

    edits: list[tuple[int, int, str]] = []  # (inner_start, inner_end, new_content)
    changed = 0
    for m in re.finditer(r"\\ffourpanel", text):
        if _line_of(text, m.start()).lstrip().startswith("%"):
            continue  # dead prior-revision call
        # Walk 7 brace groups, skipping " \t\n%" between them (line-continuation comments).
        j = m.end()
        groups: list[tuple[int, int]] = []
        while len(groups) < 7:
            while j < len(text) and text[j] in " \t\n%":
                j += 1
            if j >= len(text) or text[j] != "{":
                break
            gs, ge = _brace_group(text, j)
            groups.append((gs, ge))
            j = ge + 1
        if len(groups) < 7:
            continue
        title = text[groups[0][0]:groups[0][1]]
        model = next((mdl for macro, mdl in F4_TITLE_MODEL.items() if "\\" + macro in title), None)
        if model is None:
            raise ValueError(f"ffourpanel title without known model macro: {title!r}")
        # args 3-7 (groups index 2-6) are the five status coordinate lists.
        for status, (gs, ge) in zip(F4_STATUS_ORDER, groups[2:7], strict=True):
            new = " ".join(f"({di},{counts[(model, status, di)]})"
                           for di in range(len(F4_DIRECTIONS)))
            if text[gs:ge] != new:
                changed += 1
            edits.append((gs, ge, new))

    for gs, ge, new in sorted(edits, reverse=True):
        text = text[:gs] + new + text[ge:]

    if apply:
        path.write_text(text)
    return {"fragment": "f4", "live_cells": len(edits), "changed": changed, "applied": apply}


# --- f5 emitter -------------------------------------------------------------------------------

MODEL_FILE_SUFFIX = {
    "together-qwen-3.5-397b-a17b": "qwen",
    "azure-gpt-5.4": "gpt_5-4",
    "azure-gpt-5.3-codex": "gpt_5-3_codex",
}
DIRECTIONS_8 = F3_COL_DIRECTION


def _pass_at_k_by_direction(records: list[dict], model: str) -> tuple[dict, dict]:
    """(dir -> mean pass@1 %, dir -> mean pass@3 %) over this model's sample records.

    Independent reimplementation of the standard unbiased pass@k estimator, averaged per-kernel
    then over kernels within a direction.
    """
    kd: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        if r["is_sample"] and r["model"] == model:
            kd[(r["kernel"], r["direction"])].append(1 if r["overall_status"] == "PASS" else 0)
    d1: dict[str, list[float]] = defaultdict(list)
    d3: dict[str, list[float]] = defaultdict(list)
    for (_kernel, direction), res in kd.items():
        n, c = len(res), sum(res)
        p1 = c / n if n >= 1 else 0.0
        k = min(n, 3)
        if n >= k and comb(n, k) > 0:
            pk = 1 - comb(n - c, k) / comb(n, k)
        else:
            pk = 1.0 if c > 0 else 0.0
        d1[direction].append(p1)
        d3[direction].append(pk)
    exp1 = {d: sum(v) / len(v) * 100 for d, v in d1.items()}
    exp3 = {d: sum(v) / len(v) * 100 for d, v in d3.items()}
    return exp1, exp3


def _fmt_like(old: str, value: float) -> str:
    """Format value with the same number of decimals as the existing token."""
    decimals = len(old.split(".")[1]) if "." in old else 0
    return f"{value:.{decimals}f}"


def _rewrite_pgftable_rows(text: str, col_updates: dict[int, dict[int, float]]) -> tuple[str, int]:
    """Rewrite numeric columns of the single \\pgfplotstableread data rows, preserving layout.

    ``col_updates`` maps a data column index -> {row_dir_index: new_value}. The first column
    (index 0) is the row key (direction/level) and is never rewritten. Returns (new_text, changed).
    """
    m = re.search(r"(\\pgfplotstableread\{\s*\n)(.*?)(\n\s*\}\\\w+)", text, re.S)
    if not m:
        raise ValueError("no pgfplotstableread block")
    head, body, tail = m.group(1), m.group(2), m.group(3)
    lines = body.split("\n")
    header_seen = False
    changed = 0
    for li, line in enumerate(lines):
        if not line.strip():
            continue
        # split into (token, following-whitespace) pairs, preserving spacing exactly
        toks = re.findall(r"\S+|\s+", line)
        # token positions that are non-space
        idxs = [i for i, t in enumerate(toks) if not t.isspace()]
        if not header_seen:
            header_seen = True
            continue  # first non-blank line is the column header
        key = int(toks[idxs[0]])
        for col, updates in col_updates.items():
            if key in updates and col < len(idxs):
                old = toks[idxs[col]]
                new = _fmt_like(old, updates[key])
                if new != old:
                    changed += 1
                toks[idxs[col]] = new
        lines[li] = "".join(toks)
    return text[:m.start()] + head + "\n".join(lines) + tail + text[m.end():], changed


def emit_f5(records: list[dict], *, apply: bool) -> dict:
    """Rewrite p1/p3 columns of each per-model f5 table with canonical per-direction pass@k."""
    total_changed = 0
    for model, suffix in MODEL_FILE_SUFFIX.items():
        path = FIG_DIR / f"f5_pass_at_k_by_direction_{suffix}.tex"
        text = path.read_text()
        exp1, exp3 = _pass_at_k_by_direction(records, model)
        # header order is `dir p1 p3` -> columns 1 and 2
        p1_by_dir = {di: exp1.get(DIRECTIONS_8[di], 0.0) for di in range(len(DIRECTIONS_8))}
        p3_by_dir = {di: exp3.get(DIRECTIONS_8[di], 0.0) for di in range(len(DIRECTIONS_8))}
        new_text, changed = _rewrite_pgftable_rows(text, {1: p1_by_dir, 2: p3_by_dir})
        total_changed += changed
        if apply:
            path.write_text(new_text)
    return {"fragment": "f5", "live_cells": len(MODEL_FILE_SUFFIX) * 2, "changed": total_changed,
            "applied": apply}


# --- f6 emitter -------------------------------------------------------------------------------

SUITE_ORDER = ["rodinia", "xsbench", "rsbench", "mixbench", "hecbench"]


def _wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI (independent reimplementation of the analysis-chain definition)."""
    if total == 0:
        return (0.0, 0.0)
    ph = passed / total
    den = 1 + z * z / total
    center = (ph + z * z / (2 * total)) / den
    spread = z * ((ph * (1 - ph) / total + z * z / (4 * total * total)) ** 0.5) / den
    return (max(0.0, center - spread), min(1.0, center + spread))


def emit_f6(l0_records: list[dict], *, apply: bool) -> dict:
    """Rewrite each per-model f6 table (y/errm/errp) and its {passed/total} node labels."""
    total_changed = 0
    for model, suffix in MODEL_FILE_SUFFIX.items():
        path = FIG_DIR / f"f6_cross_suite_comparison_{suffix}.tex"
        text = path.read_text()
        std = [r for r in l0_records if r["model"] == model and r["direction"] in DIRECTIONS_8]
        y_by: dict[int, float] = {}
        em_by: dict[int, float] = {}
        ep_by: dict[int, float] = {}
        pt_by: dict[int, tuple[int, int]] = {}
        for xi, suite in enumerate(SUITE_ORDER):
            recs = [r for r in std if r["suite"] == suite]
            total = len(recs)
            passed = sum(1 for r in recs if r["overall_status"] == "PASS")
            rate = passed / total if total else 0.0
            lo, hi = _wilson(passed, total)
            y_by[xi], em_by[xi], ep_by[xi] = rate, rate - lo, hi - rate
            pt_by[xi] = (passed, total)
        # header order is `x y errm errp` -> columns 1,2,3
        text, changed = _rewrite_pgftable_rows(text, {1: y_by, 2: em_by, 3: ep_by})
        total_changed += changed

        # {passed/total} node labels: rewrite only the count, keep the label position (layout).
        def repl_node(m: re.Match, pt_by: dict = pt_by) -> str:
            nonlocal total_changed
            xi = int(m.group(2))
            if xi not in pt_by:
                return m.group(0)
            passed, total = pt_by[xi]
            new = f"{passed}/{total}"
            if new != m.group(4):
                total_changed += 1
            return f"{m.group(1)}{m.group(2)}{m.group(3)}{new}{m.group(5)}"

        text = re.sub(r"(at \(axis cs:)(\d+)(,[^)]*\)\s*\{)(\d+/\d+)(\})", repl_node, text)
        if apply:
            path.write_text(text)
    return {"fragment": "f6", "live_cells": len(MODEL_FILE_SUFFIX) * 5 * 4,
            "changed": total_changed, "applied": apply}


# --- f7 emitter -------------------------------------------------------------------------------

# f7 table column base per model: qw at 1, g54 at 4, cdx at 7 (each: rate, lo_halfwidth, hi_halfwidth).
F7_MODEL_BASECOL = {"together-qwen-3.5-397b-a17b": 1, "azure-gpt-5.4": 4, "azure-gpt-5.3-codex": 7}
F7_MIRROR_LABEL = {"Qwen": "together-qwen-3.5-397b-a17b",
                   "GPT-5.4": "azure-gpt-5.4", "GPT-5.3-codex": "azure-gpt-5.3-codex"}


def _f7_expected(all_records: list[dict], l0_records: list[dict]) -> dict:
    """(model, level) -> (rate%, ci_low%, ci_high%) for cuda-to-omp, L0 from s0 + L1-4 augmented."""
    augmented = [r for r in all_records if not r["is_sample"]]
    c2o = [r for r in (l0_records + augmented) if r["direction"] == "cuda-to-omp"]
    out: dict[tuple[str, int], tuple[float, float, float]] = {}
    for model in F7_MODEL_BASECOL:
        recs = [r for r in c2o if r["model"] == model]
        for level in range(5):
            lv = [r for r in recs if r["augment_level"] == level]
            total = len(lv)
            passed = sum(1 for r in lv if r["overall_status"] == "PASS")
            rate = passed / total * 100 if total else 0.0
            lo, hi = _wilson(passed, total)
            out[(model, level)] = (rate, lo * 100, hi * 100)
    return out


def emit_f7(all_records: list[dict], l0_records: list[dict], *, apply: bool) -> dict:
    """Rewrite the f7 \\fviidata table (rate + Wilson half-widths) and its comment mirror."""
    path = FIG_DIR / "f7_augmentation_robustness.tex"
    text = path.read_text()
    exp = _f7_expected(all_records, l0_records)

    col_updates: dict[int, dict[int, float]] = {}
    for model, base in F7_MODEL_BASECOL.items():
        rate_by = {lv: exp[(model, lv)][0] for lv in range(5)}
        lo_hw = {lv: exp[(model, lv)][0] - exp[(model, lv)][1] for lv in range(5)}
        hi_hw = {lv: exp[(model, lv)][2] - exp[(model, lv)][0] for lv in range(5)}
        col_updates[base] = rate_by
        col_updates[base + 1] = lo_hw
        col_updates[base + 2] = hi_hw
    text, changed = _rewrite_pgftable_rows(text, col_updates)

    # Comment mirror: "Ln=rate [lo,hi]" per model, model set by a leading label, held across lines.
    out_lines = []
    current: str | None = None
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("%"):
            for label, model in F7_MIRROR_LABEL.items():
                if f"{label}:" in line:
                    current = model
                    break
            if current is not None and re.search(r"L\d=[\d.]+\s*\[[\d.]+,[\d.]+\]", line):
                model = current

                def repl(m: re.Match, model: str = model) -> str:
                    nonlocal changed
                    lv = int(m.group(1))
                    rate, cilo, cihi = exp[(model, lv)]
                    nr = _fmt_like(m.group(2), rate)
                    nlo = _fmt_like(m.group(3), cilo)
                    nhi = _fmt_like(m.group(4), cihi)
                    if (nr, nlo, nhi) != (m.group(2), m.group(3), m.group(4)):
                        changed += 1
                    return f"L{m.group(1)}={nr} [{nlo},{nhi}]"

                line = re.sub(r"L(\d)=([\d.]+)\s*\[([\d.]+),([\d.]+)\]", repl, line)
        out_lines.append(line)
    text = "".join(out_lines)

    if apply:
        path.write_text(text)
    return {"fragment": "f7", "live_cells": len(F7_MODEL_BASECOL) * 5 * 3, "changed": changed,
            "applied": apply}


# --- aug_heatmap emitter ----------------------------------------------------------------------

# 12 augmented kernels, alphabetical (D14-EXT); row 0 top .. row 11 bottom. Must match the
# comparator's AUG_ROW_KERNEL byte-for-byte.
AUG_ROW_KERNEL = [
    "bfs", "cfd", "floydwarshall", "heat2d", "hotspot3d", "iso2dfd",
    "lud", "nw", "particlefilter", "pathfinder", "srad", "stencil1d",
]
# status -> (fill color, node letter, node text color). Cosmetic text color is not checked by the
# comparator; it asserts on the color name and the letter.
AUG_STATUS_STYLE = {
    "PASS": ("augPass", "P", "white"),
    "BUILD_FAIL": ("augBF", "BF", "white"),
    "RUN_FAIL": ("augRF", "RF", "black"),
    "VERIFY_FAIL": ("augVF", "VF", "white"),
    "EXTRACTION_FAIL": ("augEF", "EF", "black"),
}
AUG_FILLS_BEGIN = "% >>> AUTOGEN aug cell fills (emit_figure_fragments.py) >>>"
AUG_FILLS_END = "% <<< AUTOGEN aug cell fills <<<"
AUG_LABELS_BEGIN = "% >>> AUTOGEN aug cell labels (emit_figure_fragments.py) >>>"
AUG_LABELS_END = "% <<< AUTOGEN aug cell labels <<<"


def _region_body(text: str, begin: str, end: str) -> str:
    b = text.index(begin) + len(begin)
    e = text.index(end)
    return text[b:e].strip("\n")


def _replace_region(text: str, begin: str, end: str, body: str) -> str:
    b = text.index(begin) + len(begin)
    e = text.index(end)
    return text[:b] + "\n" + body + "\n" + text[e:]


def emit_aug_heatmap(all_records: list[dict], l0_records: list[dict], *, apply: bool) -> dict:
    """Regenerate the aug_heatmap per-cell fills and labels (both carriers) from the records.

    The static scaffold (colors, axis, legend, grid) is hand-maintained; this owns only the two
    AUTOGEN cell regions. cuda-to-omp Qwen: L0 from the canonical s0 record, L1-L4 from augmented.
    """
    path = FIG_DIR / "aug_heatmap.tex"
    text = path.read_text()

    model = "together-qwen-3.5-397b-a17b"
    augmented = [r for r in all_records if not r["is_sample"]]
    c2o = [r for r in (l0_records + augmented)
           if r["direction"] == "cuda-to-omp" and r["model"] == model]
    status_by = {(r["kernel"], r["augment_level"]): r["overall_status"] for r in c2o}

    fills: list[str] = []
    labels: list[str] = []
    for row, kernel in enumerate(AUG_ROW_KERNEL):
        for col in range(5):  # L0-L4
            status = status_by.get((kernel, col))
            if status not in AUG_STATUS_STYLE:
                raise ValueError(f"aug_heatmap: {kernel} L{col} has status {status!r} "
                                 "with no palette entry (record missing or unmapped verdict)")
            color, letter, text_color = AUG_STATUS_STYLE[status]
            fills.append(
                f"\\fill[{color}] (axis cs:{col - 0.5},{row - 0.5}) "
                f"rectangle (axis cs:{col + 0.5},{row + 0.5});")
            labels.append(
                f"\\node[font=\\tiny\\bfseries, text={text_color}] "
                f"at (axis cs:{col},{row}) {{{letter}}};")

    # changed = number of cell lines (fill or label) that differ from what is already in the file
    old_fills = _region_body(text, AUG_FILLS_BEGIN, AUG_FILLS_END).splitlines()
    old_labels = _region_body(text, AUG_LABELS_BEGIN, AUG_LABELS_END).splitlines()
    changed = sum(1 for a, b in zip(old_fills, fills, strict=False) if a != b)
    changed += abs(len(old_fills) - len(fills))
    changed += sum(1 for a, b in zip(old_labels, labels, strict=False) if a != b)
    changed += abs(len(old_labels) - len(labels))

    text = _replace_region(text, AUG_FILLS_BEGIN, AUG_FILLS_END, "\n".join(fills))
    text = _replace_region(text, AUG_LABELS_BEGIN, AUG_LABELS_END, "\n".join(labels))
    if apply:
        path.write_text(text)
    return {"fragment": "aug_heatmap", "live_cells": len(fills), "changed": changed,
            "applied": apply}


EMITTERS = {"f3": emit_f3, "f4": emit_f4, "f5": emit_f5, "f6": emit_f6, "f7": emit_f7,
            "aug_heatmap": emit_aug_heatmap}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    ap.add_argument("--fragment", default="all",
                    choices=["all", "f3", "f4", "f5", "f6", "f7", "aug_heatmap"],
                    help="which fragment to emit")
    args = ap.parse_args()

    apply = not args.dry_run
    verb = "would change" if args.dry_run else "changed"
    all_records = load_all_records()
    l0_records = load_l0_records()
    grid = {(r["model"], r["kernel"], r["direction"]): r["overall_status"] for r in l0_records}
    all_frags = ["f3", "f4", "f5", "f6", "f7", "aug_heatmap"]
    targets = all_frags if args.fragment == "all" else [args.fragment]
    for frag in targets:
        if frag == "f3":
            r = emit_f3(grid, apply=apply)
        elif frag == "f4":
            r = emit_f4(l0_records, apply=apply)
        elif frag == "f5":
            r = emit_f5(all_records, apply=apply)
        elif frag == "f6":
            r = emit_f6(l0_records, apply=apply)
        elif frag == "f7":
            r = emit_f7(all_records, l0_records, apply=apply)
        else:
            r = emit_aug_heatmap(all_records, l0_records, apply=apply)
        print(f"{r['fragment']}: {r['live_cells']} live constructs, {verb} {r['changed']}")


if __name__ == "__main__":
    main()
