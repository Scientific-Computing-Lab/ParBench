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

D12 claim_id convention (ruling 2026-08-17): every MODEL-SCOPED claim_id carries
a `_<model>` suffix - the primary model included - so no bare id silently defaults
to one model. Corpus-level claims stay bare. The 23 primary-model ids renamed at
this ruling (old -> new; the new id is always old + '_together-qwen-3.5-397b-a17b'):

    overall_pass_at_1                -> overall_pass_at_1_together-qwen-3.5-397b-a17b
    overall_pass_at_3                -> overall_pass_at_3_together-qwen-3.5-397b-a17b
    aggregate_pass_rate              -> aggregate_pass_rate_together-qwen-3.5-397b-a17b
    direction_pass_cuda_to_omp       -> direction_pass_cuda_to_omp_together-qwen-3.5-397b-a17b
    direction_pass_omp_to_cuda       -> direction_pass_omp_to_cuda_together-qwen-3.5-397b-a17b
    direction_pass_cuda_to_opencl    -> direction_pass_cuda_to_opencl_together-qwen-3.5-397b-a17b
    direction_pass_opencl_to_cuda    -> direction_pass_opencl_to_cuda_together-qwen-3.5-397b-a17b
    direction_pass_omp_to_opencl     -> direction_pass_omp_to_opencl_together-qwen-3.5-397b-a17b
    direction_pass_opencl_to_omp     -> direction_pass_opencl_to_omp_together-qwen-3.5-397b-a17b
    suite_pass_hecbench              -> suite_pass_hecbench_together-qwen-3.5-397b-a17b
    suite_pass_rodinia               -> suite_pass_rodinia_together-qwen-3.5-397b-a17b
    suite_pass_rsbench               -> suite_pass_rsbench_together-qwen-3.5-397b-a17b
    suite_pass_xsbench               -> suite_pass_xsbench_together-qwen-3.5-397b-a17b
    build_fail_count                 -> build_fail_count_together-qwen-3.5-397b-a17b
    run_fail_count                   -> run_fail_count_together-qwen-3.5-397b-a17b
    verify_fail_count                -> verify_fail_count_together-qwen-3.5-397b-a17b
    build_fail_subcategories         -> build_fail_subcategories_together-qwen-3.5-397b-a17b
    augmentation_degradation         -> augmentation_degradation_together-qwen-3.5-397b-a17b
    chi2_augmentation_trend          -> chi2_augmentation_trend_together-qwen-3.5-397b-a17b
    total_eval_files                 -> total_eval_files_together-qwen-3.5-397b-a17b
    valid_eval_files_count           -> valid_eval_files_count_together-qwen-3.5-397b-a17b
    best_direction                   -> best_direction_together-qwen-3.5-397b-a17b
    worst_direction                  -> worst_direction_together-qwen-3.5-397b-a17b

The 8 corpus-level ids that STAY bare: direction_asymmetry, spec_counts,
total_specs_count, suite_composition, total_suites_count, multi_file_fraction,
token_cost_total, sloc_correlation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

INTERNAL_ALIAS = {}
MODEL = "together-qwen-3.5-397b-a17b"


def _default_analysis_root(project_root: Path) -> Path:
    # Default analysis-input root when no explicit --analysis-root is given: the
    # canonical three-model final build (results/analysis/final), NOT the
    # top-level results/analysis (superseded single-model files carrying
    # pre-exclusion denominators and the wrong GPT-5.4 id). batch-review
    # 2026-08-17 HIGH finding: the old top-level default let the documented
    # no-flag invocation emit camera-ready claims from stale data.
    return project_root / "results" / "analysis" / "final"
# Count of evaluated models the camera-ready reports (qwen + azure-gpt-5.4 +
# azure-gpt-5.3-codex). The model IDS are discovered from the paper_data_*.json
# file set on disk, never hard-coded (D12); this arity guards that glob against
# silent SHRINKAGE - a missing or renamed per-model file would otherwise emit
# claims for a subset and still pass. Adding an evaluated model is a deliberate
# change that must bump this number.
EXPECTED_EVALUATED_MODELS = 3
STANDARD_DIRECTIONS = [
    "cuda-to-omp", "omp-to-cuda",
    "cuda-to-opencl", "opencl-to-cuda",
    "omp-to-opencl", "opencl-to-omp",
]


def _mid(base: str, model: str) -> str:
    """Model-scoped claim_id: base + model suffix (D12 uniform convention,
    ruling 2026-08-17). EVERY claim whose value depends on the model carries the
    suffix - including the primary MODEL - so no bare id silently defaults to one
    model. Corpus-level claims (spec/suite counts, pooled cross-model
    statistics) keep a bare id; suffixing those would be false precision."""
    return f"{base}_{model}"


def _resolve(data: dict, path: str) -> Any:
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


def _require(container: object, key: str, source: str) -> object:
    """Read ``container[key]`` or fail. The plan (Task 13) forbids any output
    depending on a historical hand-written count, so a missing key must abort
    rather than silently fall back to a baked-in literal (gate2 finding 5)."""
    if not isinstance(container, dict) or key not in container:
        raise SystemExit(
            f"ERROR: required key '{key}' absent from {source}; paper claims "
            f"must not fall back to a hand-written constant (plan Task 13: no "
            f"output depends on a historical count)."
        )
    return container[key]


def _eval_derivation(doc: dict, der: dict) -> object:
    """Evaluate a derived claim's recipe against its source document.

    Derived claims (a count, a difference, a selected list element) have no
    single stored scalar, so a container json_path cannot resolve them. The
    recipe is explicit and deterministic so validation recomputes it instead of
    silently skipping a non-scalar resolution (gate2 finding 4)."""
    op = der["op"]
    if op == "len":
        node = _resolve(doc, der["path"])
        return len(node) if node is not None else None
    if op == "diff":
        a = _val(_resolve(doc, der["minuend"]))
        b = _val(_resolve(doc, der["subtrahend"]))
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        return a - b
    if op == "max_abs_gap":
        node = _resolve(doc, der["path"])
        if not isinstance(node, list) or not node:
            return None
        fk, rk = der["forward"], der["reverse"]
        best = max(node, key=lambda x: abs(x[fk] - x[rk]))
        # gate21 finding 3: with a ``field`` the recipe selects ONE component of
        # the max-gap entry (forward_pass_rate / reverse_pass_rate / p_value), so
        # every number reported for that entry is resolvable, not just the gap.
        if der.get("field"):
            return best.get(der["field"])
        return best[fk] - best[rk]
    if op == "index_field":
        node = _resolve(doc, der["path"])
        if not isinstance(node, list) or len(node) <= der["index"]:
            return None
        return node[der["index"]].get(der["field"])
    if op == "top_n_items":
        node = _resolve(doc, der["path"])
        return _top_n_items(node, der["n"])
    raise SystemExit(f"unknown derivation op: {op!r}")


def _top_n_items(mapping: object, n: int) -> Any:
    """The n highest-count (key, count) pairs of a ``{name: count}`` mapping,
    as an ordered list of ``[name, count]``.

    Ties break by name so the result is deterministic regardless of dict
    insertion order (gate3 finding 2): a claim that states the top-three
    BUILD_FAIL subcategories must compare the ordered VALUES, not just how many
    categories exist - otherwise a change from 63/55/54 to 61/56/55 is invisible
    to the old-vs-new comparison because the category count is unchanged."""
    if not isinstance(mapping, dict) or not mapping:
        return None
    ordered = sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[k, v] for k, v in ordered[:n]]


def resolve_claim_value(doc: dict, claim: dict) -> object:
    """Resolve a claim's scalar value from its source document.

    A plain claim reads ``json_path`` (a scalar or a ``{value, ...}`` metric); a
    derived claim carries a ``derivation`` recipe. Shared by validation here and
    by the old-vs-new comparison in generate_final_evidence so both resolve
    every claim - including derived ones - to a scalar (gate2 findings 3/4)."""
    der = claim.get("derivation")
    if der:
        return _eval_derivation(doc, der)
    return _val(_resolve(doc, claim.get("json_path") or ""))


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


def _model_entry_index(entries: list, model: str) -> int | None:
    """Index of the per-model statistics entry for `model`, matched by key.

    gate7 finding 3: statistical_analysis emits per-model entries sorted by model
    id under a ``group`` (or ``model``) field. Selecting entry 0 attributed a
    different model's statistic to the paper model. Returns None when no entry
    names this model, so the caller omits the claim rather than mislabel it.
    """
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        if e.get("group") == model or e.get("model") == model:
            return i
    return None


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
    # gate21 finding 3: every NUMERIC component the claim reports (not just the
    # scalar `value`) is represented as a resolvable component so the old-vs-new
    # comparison and the independent verifier can compare/recompute each one.
    # A pass-rate claim's Wilson interval bounds and denominator resolve from the
    # SAME metric dict its json_path names (dict keys ci_lower/ci_upper/n), so they
    # get auto-generated components here; claims carrying additional reported
    # numbers (p-values, endpoints) pass an explicit `components` list.
    components = list(kwargs.get("components") or [])
    if json_path and kwargs.get("ci_lower") is not None:
        for field in ("ci_lower", "ci_upper", "n"):
            if kwargs.get(field) is not None:
                components.append({
                    "name": field, "value": kwargs[field],
                    "json_path": f"{json_path}.{field}",
                })
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
        "components": components,
        **{k: v for k, v in kwargs.items()
           if k not in ("model", "terminology", "components")},
    }


def _discover_additional_models(analysis: Path) -> list[str]:
    """Per-model paper_data files present in the analysis dir, minus the primary
    MODEL (D12, ruling 2026-08-17).

    The primary model (MODEL) has a rich quantitative_findings_{MODEL}.json and
    keeps its own claim builders; the OTHER evaluated models expose only a
    paper_data_{model}.json (a different schema whose real data lives under
    passk_campaign). The model list is discovered from the file set on disk, so
    adding an evaluated model needs no code change and no hand-maintained roster
    (ticket #34: model ids from ground truth, never a new hard-coded list)."""
    prefix, suffix = "paper_data_", ".json"
    found = [p.name[len(prefix):-len(suffix)]
             for p in sorted(analysis.glob(f"{prefix}*{suffix}"))]
    # Arity guard: in a real analysis namespace the per-model set must be exactly
    # the evaluated models, so a shrunk/renamed glob fails loudly instead of
    # emitting a silent subset. Minimal unit fixtures carry no paper_data files
    # (found == []); those legitimately produce no additional-model claims and
    # skip the guard.
    if found and len(found) != EXPECTED_EVALUATED_MODELS:
        raise SystemExit(
            f"ERROR: expected {EXPECTED_EVALUATED_MODELS} paper_data_*.json "
            f"files under {analysis}, found {len(found)}: {sorted(found)}. "
            "The per-model claim set is derived from this glob; a missing or "
            "renamed file must not silently drop a model."
        )
    if found and MODEL not in found:
        raise SystemExit(
            f"ERROR: primary model {MODEL!r} has no paper_data file under "
            f"{analysis}; found {sorted(found)}."
        )
    return [m for m in found if m != MODEL]


def build_paper_data_claims(
    model: str, analysis: Path, rp,
) -> list[dict]:
    """Additive per-model claims for a model whose only per-model source is
    paper_data_{model}.json (D12).

    Sourced from its passk_campaign block, which is metric-consistent with the
    primary model's quantitative_findings claims: verified 2026-08-16 that
    aggregate_passk.pass@1_macro_avg / pass@3_macro_avg equal the primary
    model's canonical pass_at_1 / pass_at_3 values. Every claim carries a
    model-suffixed claim_id so the primary model's 31 bare ids stay stable.
    Confidence intervals come only from nodes that actually store them
    (passk_campaign.overall and each by_direction entry); the macro-averaged
    pass@k figures carry no stored CI, so those claims report the point estimate
    without one rather than borrow a different denominator's interval."""
    pd = json.loads((analysis / f"paper_data_{model}.json").read_text())
    src = [rp(f"paper_data_{model}.json")]
    pk = pd["passk_campaign"]
    ov = pk["overall"]
    agg = pk["aggregate_passk"]
    bd = pk["by_direction"]
    claims: list[dict] = []

    # pass@1 / pass@3 (macro-averaged over tasks; no stored CI in paper_data).
    claims.append(_make_claim(
        _mid("overall_pass_at_1", model), "Abstract, S6.1",
        f"{model} achieves pass@1 of {agg['pass@1_macro_avg']*100:.1f}% "
        f"across {agg['n_tasks']} tasks",
        agg["pass@1_macro_avg"], src,
        "passk_campaign.aggregate_passk.pass@1_macro_avg",
        model=model, n=agg["n_tasks"], unit="fraction",
        notes="Macro-averaged pass@1 over tasks; paper_data stores no CI for it",
    ))
    claims.append(_make_claim(
        _mid("overall_pass_at_3", model), "S6.1",
        f"{model} pass@3 of {agg['pass@3_macro_avg']*100:.1f}%",
        agg["pass@3_macro_avg"], src,
        "passk_campaign.aggregate_passk.pass@3_macro_avg",
        model=model, n=agg["n_tasks"], unit="fraction",
        notes="Macro-averaged pass@3 over tasks; paper_data stores no CI for it",
    ))

    # Aggregate sample-level pass rate (carries a Wilson CI in the source).
    claims.append(_make_claim(
        _mid("aggregate_pass_rate", model), "S6.1",
        f"{model} sample-level pass rate {ov['pass_rate']*100:.1f}% "
        f"[{ov['ci_lower']*100:.1f}%, {ov['ci_upper']*100:.1f}%] (n={ov['total']})",
        ov["pass_rate"], src,
        "passk_campaign.overall.pass_rate",
        model=model, ci_level=0.95, unit="fraction",
        components=[
            {"name": "ci_lower", "value": ov["ci_lower"],
             "json_path": "passk_campaign.overall.ci_lower"},
            {"name": "ci_upper", "value": ov["ci_upper"],
             "json_path": "passk_campaign.overall.ci_upper"},
            {"name": "n", "value": ov["total"],
             "json_path": "passk_campaign.overall.total"},
        ],
    ))

    # Per-direction pass@1 (6 standard directions; each entry stores its CI).
    for d in STANDARD_DIRECTIONS:
        if d not in bd:
            continue
        dv = bd[d]
        claims.append(_make_claim(
            _mid(f"direction_pass_{d.replace('-', '_')}", model), "S6.2",
            f"{model} {d} pass rate: {dv['pass_rate']*100:.1f}% "
            f"[{dv['ci_lower']*100:.1f}%, {dv['ci_upper']*100:.1f}%] (n={dv['total']})",
            dv["pass_rate"], src,
            f"passk_campaign.by_direction.{d}.pass_rate",
            model=model, ci_level=0.95, unit="fraction",
            components=[
                {"name": "ci_lower", "value": dv["ci_lower"],
                 "json_path": f"passk_campaign.by_direction.{d}.ci_lower"},
                {"name": "ci_upper", "value": dv["ci_upper"],
                 "json_path": f"passk_campaign.by_direction.{d}.ci_upper"},
                {"name": "n", "value": dv["total"],
                 "json_path": f"passk_campaign.by_direction.{d}.total"},
            ],
        ))

    # Failure status counts (from the sample-level status histogram).
    status = ov.get("by_status", {})
    total = ov.get("total", 0)
    for st, cid in (("BUILD_FAIL", "build_fail_count"),
                    ("RUN_FAIL", "run_fail_count"),
                    ("VERIFY_FAIL", "verify_fail_count")):
        cnt = status.get(st, 0)
        pct = f" ({cnt/total*100:.1f}%)" if total else ""
        claims.append(_make_claim(
            _mid(cid, model), "S6.4",
            f"{model} {st}: {cnt}/{total}{pct}",
            cnt, src,
            f"passk_campaign.overall.by_status.{st}",
            model=model, unit="count",
        ))

    # Best / worst standard direction (derived selections).
    dir_values = {d: bd[d]["pass_rate"] for d in STANDARD_DIRECTIONS if d in bd}
    if dir_values:
        best_d = max(dir_values, key=lambda d: dir_values[d])
        worst_d = min(dir_values, key=lambda d: dir_values[d])
        claims.append(_make_claim(
            _mid("best_direction", model), "S6.2",
            f"{model} best direction: {best_d} ({dir_values[best_d]*100:.1f}%)",
            dir_values[best_d], src,
            f"passk_campaign.by_direction.{best_d}.pass_rate",
            model=model, unit="fraction",
            notes="Derived: highest pass rate among 6 standard directions",
        ))
        claims.append(_make_claim(
            _mid("worst_direction", model), "S6.2",
            f"{model} worst direction: {worst_d} ({dir_values[worst_d]*100:.1f}%)",
            dir_values[worst_d], src,
            f"passk_campaign.by_direction.{worst_d}.pass_rate",
            model=model, unit="fraction",
            notes="Derived: lowest pass rate among 6 standard directions",
        ))

    return claims


def build_file_count_claims(model, analysis, rp):
    """Per-model eval-file counts (total on disk / valid after exclusion /
    excluded), each read from that model's quantitative_findings_{model}.json
    metadata.file_counts. Emitted for ALL evaluated models (D12 uniform
    model-suffixed ids). The three per-model totals sum to 2,344, the valid
    counts to 2,160, and the excluded counts to 184 (the canonical build)."""
    qf = json.loads((analysis / f"quantitative_findings_{model}.json").read_text())
    fc = _require(
        _require(qf, "metadata", f"quantitative_findings_{model}.json"),
        "file_counts", f"quantitative_findings_{model}.json metadata")
    total_on_disk = _require(fc, "total_on_disk",
                             f"quantitative_findings_{model}.json metadata.file_counts")
    valid_after = _require(fc, "valid_after_exclusion",
                           f"quantitative_findings_{model}.json metadata.file_counts")
    excluded_files = _require(fc, "excluded_known_fail",
                              f"quantitative_findings_{model}.json metadata.file_counts")
    src = [rp(f"quantitative_findings_{model}.json")]
    return [
        _make_claim(
            _mid("total_eval_files", model), "S5",
            f"{total_on_disk} result files on disk, "
            f"{valid_after} valid after KNOWN_FAIL and performance-only exclusion",
            total_on_disk, src,
            "metadata.file_counts.total_on_disk",
            unit="count", model=model,
        ),
        _make_claim(
            _mid("valid_eval_files_count", model), "S5",
            f"{valid_after} result files valid after KNOWN_FAIL and "
            f"performance-only exclusion",
            valid_after, src,
            "metadata.file_counts.valid_after_exclusion",
            unit="count", model=model,
        ),
        _make_claim(
            _mid("excluded_eval_files_count", model), "S5",
            f"{excluded_files} result files excluded "
            f"(KNOWN_FAIL and performance-only)",
            excluded_files, src,
            "metadata.file_counts.excluded_known_fail",
            unit="count", model=model,
        ),
    ]


def build_claims(
    project_root: Path,
    verbose: bool = False,
    analysis_root: Path | None = None,
) -> dict:
    # Default is the canonical final build (see _default_analysis_root); the
    # final analysis chain still passes an explicit --analysis-root.
    analysis = analysis_root or _default_analysis_root(project_root)

    # Every claim's source_files must point at the analysis dir actually
    # consumed here, so the declared provenance and the emitted value come from
    # the SAME file (gate1 finding 2). The module-level _rp hard-codes
    # results/analysis/, which is correct only for the submitted-era default;
    # when a nested --analysis-root (e.g. results/analysis/final) is used the
    # claims must declare that path. This local _rp shadows the module one for
    # every claim built below.
    try:
        _rel = analysis.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        _rel = analysis.as_posix()

    def _rp(relpath: str) -> str:
        return f"{_rel}/{relpath}"

    qf_path = analysis / f"quantitative_findings_{MODEL}.json"
    et_path = analysis / "error_taxonomy.json"
    sl_path = analysis / "sloc_analysis.json"
    ta_path = analysis / "token_analysis.json"
    bc_path = analysis / "benchmark_characterization.json"
    sa_path = analysis / "statistical_analysis.json"

    for required in (qf_path, et_path, sl_path, ta_path, bc_path, sa_path):
        if not required.is_file():
            raise SystemExit(
                f"ERROR: required analysis input not found under {analysis}: "
                f"{required.name}"
            )

    qf = json.loads(qf_path.read_text())
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
        _mid("overall_pass_at_1", MODEL), "Abstract, S6.1",
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
        _mid("overall_pass_at_3", MODEL), "S6.1",
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
        _mid("aggregate_pass_rate", MODEL), "S6.1",
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
            _mid(f"direction_pass_{d.replace('-', '_')}", MODEL), "S6.2",
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
            _mid(f"suite_pass_{suite}", MODEL), "S6.3",
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
            derivation={"op": "max_abs_gap", "path": "direction_asymmetry",
                        "forward": "forward_pass_rate", "reverse": "reverse_pass_rate"},
            # gate21 finding 3: the forward/reverse rates and the p-value the
            # claim_text reports must each be comparable and independently
            # verifiable, not just the gap.
            components=[
                {"name": "forward_pass_rate", "value": biggest["forward_pass_rate"],
                 "derivation": {"op": "max_abs_gap", "path": "direction_asymmetry",
                                "forward": "forward_pass_rate", "reverse": "reverse_pass_rate",
                                "field": "forward_pass_rate"}},
                {"name": "reverse_pass_rate", "value": biggest["reverse_pass_rate"],
                 "derivation": {"op": "max_abs_gap", "path": "direction_asymmetry",
                                "forward": "forward_pass_rate", "reverse": "reverse_pass_rate",
                                "field": "reverse_pass_rate"}},
                {"name": "p_value", "value": round(biggest["p_value"], 6),
                 "derivation": {"op": "max_abs_gap", "path": "direction_asymmetry",
                                "forward": "forward_pass_rate", "reverse": "reverse_pass_rate",
                                "field": "p_value"}},
            ],
            notes="Value is forward-reverse gap; sign indicates direction of asymmetry",
            # The paired cells pool all evaluated models (codex, gpt-5.4, qwen),
            # so this is a cross-model statistic, not a Qwen-only one (gate16
            # finding 3). Labelling it 'pooled' prevents misattribution.
            model="pooled",
        ))

    # 16. BUILD_FAIL count
    bf = ft["status_counts"].get("BUILD_FAIL", 0)
    claims.append(_make_claim(
        _mid("build_fail_count", MODEL), "S6.4",
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
        _mid("run_fail_count", MODEL), "S6.4",
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
        _mid("verify_fail_count", MODEL), "S6.4",
        f"VERIFY_FAIL: {vf}/{ft['total_records']} "
        f"({vf/ft['total_records']*100:.1f}%)",
        vf,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "canonical.failure_taxonomy.status_counts.VERIFY_FAIL",
        unit="count",
    ))

    # 19. BUILD_FAIL subcategories (top 3). The claim VALUE is the ordered
    # top-three (subcategory, count) pairs, not the count of categories
    # (gate3 finding 2): the reported numbers - not merely how many buckets
    # exist - are what the old-vs-new comparison and the verifier must check, so
    # a 63/55/54 -> 61/56/55 shift is caught. Ties break by name (see
    # _top_n_items) so the value is deterministic across dict orderings.
    _subcats_path = "canonical.failure_taxonomy.by_status.BUILD_FAIL.subcategories"
    subcats = ft["by_status"]["BUILD_FAIL"]["subcategories"]
    top3 = _top_n_items(subcats, 3)
    desc = ", ".join(f"{k}({v})" for k, v in top3) if top3 else "N/A"
    claims.append(_make_claim(
        _mid("build_fail_subcategories", MODEL), "S6.4",
        f"Top BUILD_FAIL subcategories: {desc}",
        top3,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        _subcats_path,
        derivation={"op": "top_n_items", "path": _subcats_path, "n": 3},
        unit="ordered_top_subcategories",
    ))

    # 20. Spec counts. Every number below is read from a generated artifact and
    # fails if absent - no hand-written corpus constant (gate2 finding 5).
    meta = _require(qf, "metadata", f"quantitative_findings_{MODEL}.json")
    excluded_count = _require(meta, "excluded_specs_count",
                              f"quantitative_findings_{MODEL}.json metadata")
    _fc = _require(meta, "file_counts", f"quantitative_findings_{MODEL}.json metadata")
    total_on_disk = _require(_fc, "total_on_disk",
                             f"quantitative_findings_{MODEL}.json metadata.file_counts")
    total_specs = _require(_require(bc, "summary", "benchmark_characterization.json"),
                           "total_specs", "benchmark_characterization.json summary")
    claims.append(_make_claim(
        "spec_counts", "S4",
        f"{excluded_count} KNOWN_FAIL specs excluded; "
        f"{total_specs} total specs; {total_on_disk} result files on disk",
        excluded_count,
        [_rp(f"quantitative_findings_{MODEL}.json")],
        "metadata.excluded_specs_count",
        unit="count",
        # Corpus-level: the KNOWN_FAIL exclusion set is EXCLUDED_SPECS (global),
        # identical across models, so the id stays bare and the label is pooled
        # even though the count is read from the primary model's metadata file
        # (D12 ruling: suffixing a corpus-level claim would be false precision).
        model="pooled",
    ))

    # 20a. Total spec count as its own atomic claim (gate14 finding 3). The
    # compound spec_counts/suite_composition texts both NAME the total-spec count
    # but neither resolves it (spec_counts resolves the excluded count,
    # suite_composition resolves the kernel count), so 206 was validated by no
    # claim and never compared old-vs-new. Emit it standalone so every reported
    # number is independently traceable to a generated artifact.
    claims.append(_make_claim(
        "total_specs_count", "S4",
        f"{total_specs} total specs",
        total_specs,
        [_rp("benchmark_characterization.json")],
        "summary.total_specs",
        unit="count",
        model="pooled",  # spec-derived, model-independent (gate7 finding 3)
    ))

    # 21. Suite composition
    bcs = _require(bc, "summary", "benchmark_characterization.json")
    _tk = _require(bcs, "total_kernels", "benchmark_characterization.json summary")
    _ts = _require(bcs, "total_suites", "benchmark_characterization.json summary")
    claims.append(_make_claim(
        "suite_composition", "S4",
        f"{_tk} kernels across {_ts} suites, {total_specs} specs",
        _tk,
        [_rp("benchmark_characterization.json")],
        "summary.total_kernels",
        unit="count",
        model="pooled",  # spec-derived, model-independent (gate7 finding 3)
    ))

    # 21a. Total suite count as its own atomic claim (gate14 finding 3): the
    # suite_composition text names the suite count but resolves only the kernel
    # count.
    claims.append(_make_claim(
        "total_suites_count", "S4",
        f"{_ts} suites",
        _ts,
        [_rp("benchmark_characterization.json")],
        "summary.total_suites",
        unit="count",
        model="pooled",  # spec-derived, model-independent (gate7 finding 3)
    ))

    # 22. Multi-file fraction
    mf_pct = _require(bcs, "multi_file_pct", "benchmark_characterization.json summary")
    claims.append(_make_claim(
        "multi_file_fraction", "S4",
        f"{mf_pct:.1f}% of kernels require multi-file translation",
        mf_pct,
        [_rp("benchmark_characterization.json")],
        "summary.multi_file_pct",
        unit="percentage",
        model="pooled",  # spec-derived, model-independent (gate7 finding 3)
    ))

    # 23. Token cost total
    # gate8 finding 1: token_analysis reports grand_total_cost_usd=None when an
    # evaluated model lacks verified pricing, because a complete pooled cost is
    # not computable. Emit a removed-class claim (no value/source) rather than a
    # wrong two-of-three-model total labelled 'Total API cost'.
    cost = ta.get("grand_total_cost_usd")
    if cost is None:
        unpriced = ta.get("unpriced_evaluated_models") or []
        unpriced_str = ", ".join(unpriced) if unpriced else "one or more models"
        claims.append(_make_claim(
            "token_cost_total", "S5",
            "Total API cost: not reported (a complete pooled cost across all "
            f"evaluated models is unavailable because {unpriced_str} lack "
            "verified pricing).",
            None,
            [],
            "",
            unit="usd",
            model="pooled",
        ))
    else:
        claims.append(_make_claim(
            "token_cost_total", "S5",
            f"Total API cost: ${cost:.2f}",
            cost,
            [_rp("token_analysis.json")],
            "grand_total_cost_usd",
            unit="usd",
            model="pooled",  # grand total across all evaluated models (gate7 finding 3)
        ))

    # 24. Augmentation degradation (L0→L4)
    aug_levels = c2.get("augmentation_trends", {}).get("aggregate", {}).get("per_level", {})
    if aug_levels:
        l0_val = aug_levels.get("L0", {}).get("value", 0)
        l4_val = aug_levels.get("L4", {}).get("value", 0)
        claims.append(_make_claim(
            _mid("augmentation_degradation", MODEL), "S6.5",
            f"Augmentation degradation L0→L4: "
            f"{l0_val*100:.1f}% → {l4_val*100:.1f}%",
            l4_val - l0_val,
            [_rp(f"quantitative_findings_{MODEL}.json")],
            "canonical.augmentation_trends.aggregate.per_level",
            derivation={"op": "diff",
                        "minuend": "canonical.augmentation_trends.aggregate.per_level.L4.value",
                        "subtrahend": "canonical.augmentation_trends.aggregate.per_level.L0.value"},
            # gate21 finding 3: the L0 and L4 endpoints the claim_text reports are
            # each represented so both the comparison and the verifier check them,
            # not only their difference.
            components=[
                {"name": "l0_rate", "value": l0_val,
                 "json_path": "canonical.augmentation_trends.aggregate.per_level.L0.value"},
                {"name": "l4_rate", "value": l4_val,
                 "json_path": "canonical.augmentation_trends.aggregate.per_level.L4.value"},
            ],
            notes="Value is L4-L0 difference (negative = degradation)",
            unit="fraction_difference",
        ))

    # 25. Chi-squared augmentation trend
    # gate7 finding 3: select the entry for the PAPER model by its group key, not
    # list position. The list is sorted by model id, so entry 0 belonged to a
    # different model and its chi2 was mislabeled as the paper model's.
    # gate12 finding 1: this claim carries a chi-squared independence statistic,
    # NOT the Cochran-Armitage trend z-statistic. quantitative_findings.py emits
    # `cochran_armitage_trend` for the genuine z-statistic, so reusing that id
    # here made one key denote two different statistics. Emit under the distinct
    # id `chi2_augmentation_trend`.
    chi2_aug = sa.get("chi2_augmentation_by_model", [])
    aug_idx = _model_entry_index(chi2_aug, MODEL)
    if chi2_aug and aug_idx is not None:
        entry = chi2_aug[aug_idx]
        claims.append(_make_claim(
            _mid("chi2_augmentation_trend", MODEL), "S6.5",
            f"Chi-squared augmentation trend: chi2={entry.get('chi2', 0):.4f}, "
            f"p={entry.get('p_value', 1):.4f}",
            entry.get("chi2", 0),
            [_rp("statistical_analysis.json")],
            "chi2_augmentation_by_model",
            derivation={"op": "index_field", "path": "chi2_augmentation_by_model",
                        "index": aug_idx, "field": "chi2",
                        "selected_by": f"group == {MODEL}"},
            # gate21 finding 3: the p-value the claim_text reports is represented so
            # the comparison and the verifier check it alongside the chi2 statistic.
            components=[
                {"name": "p_value", "value": round(entry.get("p_value", 1), 6),
                 "derivation": {"op": "index_field", "path": "chi2_augmentation_by_model",
                                "index": aug_idx, "field": "p_value",
                                "selected_by": f"group == {MODEL}"}},
            ],
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
            model="pooled",  # per-kernel pass rates pooled over all models (gate7 finding 3)
        ))

    # 28/28a/28b. Per-model eval-file counts (total on disk / valid after
    # exclusion / excluded), emitted for the primary model here and for every
    # additional model in the loop below, via the shared build_file_count_claims
    # (D12 uniform model-suffixed ids). The three per-model totals sum to 2,344,
    # valid to 2,160, excluded to 184 (tested).
    claims.extend(build_file_count_claims(MODEL, analysis, _rp))

    # 29. Best direction
    dir_values = {d: dpr[d]["value"] for d in STANDARD_DIRECTIONS if d in dpr}
    if dir_values:
        best_d = max(dir_values, key=lambda d: dir_values[d])
        claims.append(_make_claim(
            _mid("best_direction", MODEL), "S6.2",
            f"Best direction: {best_d} ({dir_values[best_d]*100:.1f}%)",
            dir_values[best_d],
            [_rp(f"quantitative_findings_{MODEL}.json")],
            f"canonical.direction_pass_rates.standard.{best_d}",
            notes="Derived: highest pass rate among 6 standard directions",
            unit="fraction",
        ))

    # 30. Worst direction
    if dir_values:
        worst_d = min(dir_values, key=lambda d: dir_values[d])
        claims.append(_make_claim(
            _mid("worst_direction", MODEL), "S6.2",
            f"Worst direction: {worst_d} ({dir_values[worst_d]*100:.1f}%)",
            dir_values[worst_d],
            [_rp(f"quantitative_findings_{MODEL}.json")],
            f"canonical.direction_pass_rates.standard.{worst_d}",
            notes="Derived: lowest pass rate among 6 standard directions",
            unit="fraction",
        ))

    # Additive per-model claims (D12): every evaluated model that exposes only a
    # paper_data_{model}.json (no rich quantitative_findings file) gets a
    # model-suffixed claim set, sourced from its passk_campaign block. The
    # primary MODEL's 31 claims above keep their bare, stable claim_ids.
    additional_models = _discover_additional_models(analysis)
    for m in additional_models:
        claims.extend(build_paper_data_claims(m, analysis, _rp))
        # Per-model eval-file counts now that every model has a
        # quantitative_findings_{model}.json (D10/D12 per-model appendix).
        claims.extend(build_file_count_claims(m, analysis, _rp))

    if verbose:
        print(f"  Generated {len(claims)} claims "
              f"({len(additional_models)} additional model(s): "
              f"{', '.join(additional_models) or 'none'})")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "models": [MODEL, *additional_models],
        "terminology_note": (
            "All json_path values use 'canonical' as the top-level key "
            "in the source analysis JSONs."
        ),
        "claims": claims,
    }


def validate_claims(
    project_root: Path,
    verbose: bool = False,
    analysis_root: Path | None = None,
) -> int:
    # Default is the canonical final build (see _default_analysis_root).
    analysis = analysis_root or _default_analysis_root(project_root)
    claims_path = analysis / "paper_claims.json"
    if not claims_path.exists():
        print(f"FAIL: {claims_path} not found", file=sys.stderr)
        return 1

    data = json.loads(claims_path.read_text())
    claims = data.get("claims", [])
    failures = 0

    for claim in claims:
        cid = claim["claim_id"]
        # A sourceless claim is 'removed'-class (deferred namespace); only valid
        # when it carries no value to resolve.
        if not claim["source_files"] or not claim["json_path"]:
            if claim.get("value") is not None:
                print(f"  FAIL {cid}: value present but no source_files/json_path")
                failures += 1
            elif verbose:
                print(f"  SKIP {cid}: no source (removed-class)")
            continue

        src_path = project_root / claim["source_files"][0]
        if not src_path.exists():
            print(f"  FAIL {cid}: source file not found: {src_path}")
            failures += 1
            continue

        src_data = json.loads(src_path.read_text())
        # Resolve via the shared resolver so a derived claim recomputes its
        # recipe and a container json_path can never silently pass as a SKIP
        # (gate2 finding 4).
        resolved_val = resolve_claim_value(src_data, claim)
        claim_val = claim["value"]

        if isinstance(claim_val, (int, float)):
            if not isinstance(resolved_val, (int, float)):
                print(f"  FAIL {cid}: numeric claim did not resolve to a scalar "
                      f"(got {type(resolved_val).__name__}) from "
                      f"{claim['json_path']}")
                failures += 1
            elif abs(resolved_val - claim_val) > 1e-6:
                print(f"  FAIL {cid}: expected {resolved_val}, got {claim_val}")
                failures += 1
            elif verbose:
                print(f"  PASS {cid}: {claim_val}")
        else:
            if resolved_val != claim_val:
                print(f"  FAIL {cid}: non-numeric mismatch: "
                      f"{resolved_val!r} != {claim_val!r}")
                failures += 1
            elif verbose:
                print(f"  PASS {cid}: {claim_val!r}")

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
    parser.add_argument(
        "--analysis-root", type=Path, default=None,
        help="Explicit analysis-input root to consume (Task 5). When omitted, "
             "the default is the canonical three-model final build "
             "{project-root}/results/analysis/final; the top-level "
             "results/analysis (superseded per-model files) is never used.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve() if args.analysis_root else None

    if args.validate:
        sys.exit(
            validate_claims(args.project_root, args.verbose, analysis_root)
        )

    output = args.output or (
        (analysis_root or _default_analysis_root(args.project_root))
        / "paper_claims.json"
    )
    data = build_claims(args.project_root, args.verbose, analysis_root)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(data['claims'])} claims to {output}")


if __name__ == "__main__":
    main()
