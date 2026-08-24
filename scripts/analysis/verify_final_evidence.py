#!/usr/bin/env python3
"""Independently verify the ParBench final evidence manifest (plan Task 13).

This is a deliberately SEPARATE recomputation. It imports none of the analysis
modules it checks (statistical_analysis, quantitative_findings, build_error_
taxonomy, generate_paper_data, generate_final_evidence, ...). It recomputes the
record-level headline counts, rates, transition counts, and the PASS->RUN_FAIL
mechanism split directly from the record JSONs named by the manifest, then fails
on any disagreement with the manifest. It also verifies that every recorded
output hash still matches the file on disk and that every claim key resolves to
a generated claim.

Only ``harness.constants`` (the canonical exclusion set - not an analysis
module) is imported; parent-identity promotion is reimplemented inline so this
verifier shares no code path with the generator.

Usage:
    python3 scripts/analysis/verify_final_evidence.py \\
        --manifest results/analysis/final_evidence_manifest.json \\
        --expected-direction-pairs 5 \\
        --require-output results/analysis/sloc_analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math  # noqa: E402

from harness.constants import CORRECTNESS_INELIGIBLE_SPECS  # noqa: E402

# oracle_shape is the semantic classifier, not an analysis module the verifier
# checks; importing it lets the verifier independently recompute the semantic
# classification's denominators and estimates (gate2 finding 6). It is NOT on
# the forbidden-import list guarded by test_verify_is_independent_of_analysis_modules.
from scripts.rebuttal.oracle_shape_census import oracle_shape  # noqa: E402

# Wilson 95% score interval, reimplemented inline (no scipy, no analysis-module
# import) so the semantic-classification interval is independently recomputed.
# z = norm.ppf(0.975); rounding matches quantitative_findings.wilson_ci.
_WILSON_Z = 1.959963984540054


def _wilson(passes: int, total: int) -> dict:
    if total == 0:
        return {"value": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    z = _WILSON_Z
    p = passes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return {
        "value": round(p, 4),
        "ci_lower": round(max(0.0, center - spread), 4),
        "ci_upper": round(min(1.0, center + spread), 4),
    }


# --------------------------------------------------------------------------- #
# Inline claim resolver (independent of generate_paper_claims)                 #
# --------------------------------------------------------------------------- #

def _resolve(doc, dotted):
    cur = doc
    for part in (dotted or "").split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _scalar(node):
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def _eval_derivation(doc, der):
    op = der["op"]
    if op == "len":
        node = _resolve(doc, der["path"])
        return len(node) if node is not None else None
    if op == "diff":
        a = _scalar(_resolve(doc, der["minuend"]))
        b = _scalar(_resolve(doc, der["subtrahend"]))
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        return a - b
    if op == "max_abs_gap":
        node = _resolve(doc, der["path"])
        if not isinstance(node, list) or not node:
            return None
        fk, rk = der["forward"], der["reverse"]
        best = max(node, key=lambda x: abs(x[fk] - x[rk]))
        # gate21 finding 3: independent copy of the generator's ``field`` selector.
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
        # Independent copy of the generator's ordered-top-n rule (gate3 finding
        # 2): highest count first, ties broken by name for determinism.
        if not isinstance(node, dict) or not node:
            return None
        ordered = sorted(node.items(), key=lambda kv: (-kv[1], kv[0]))
        return [[k, v] for k, v in ordered[:der["n"]]]
    return None


def _resolve_claim_value(doc, claim):
    der = claim.get("derivation")
    if der:
        return _eval_derivation(doc, der)
    return _scalar(_resolve(doc, claim.get("json_path")))

STATUSES = (
    "PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL",
    "ERROR", "EXTRACTION_FAIL", "NOT_REPLAYABLE", "SKIP",
)

# Generators whose code MUST be hash-bound in the manifest (gate6 finding 3,
# 2026-08-13). Declared independently here (the verifier imports no analysis
# module it checks); kept in sync with generate_final_evidence.GENERATOR_SCRIPTS
# by test_required_generators_match_orchestrator.
REQUIRED_GENERATORS = (
    "scripts/analysis/generate_final_evidence.py",
    "scripts/analysis/verify_final_evidence.py",
    "scripts/analysis/generate_paper_claims.py",
    "scripts/analysis/sloc_analysis.py",
    "scripts/rebuttal/oracle_shape_census.py",
    "scripts/spec_tools/direct_oracle_scan.py",
    "scripts/analysis/compare_oracle_censuses.py",
    "scripts/provenance/inventory_immutable_inputs.py",
    # gate7 finding 5: the executed producers that were previously unbound.
    "scripts/analysis/statistical_analysis.py",
    "scripts/analysis/quantitative_findings.py",
    "scripts/analysis/build_error_taxonomy.py",
    "scripts/analysis/benchmark_characterization.py",
    "scripts/analysis/token_analysis.py",
    "scripts/analysis/generate_paper_data.py",
    "scripts/analysis/cross_model_comparison.py",
    "scripts/analysis/verify_direction_pairs.py",
    "scripts/evaluation/analyze_eval.py",
    "scripts/rebuttal/suite_composition.py",
    "scripts/rebuttal/generate_comparator_labels.py",
    # gate8 finding 5: imported (non-argv) producers that shape promoted
    # metadata, direction aggregation, and oracle strengths.
    "scripts/evaluation/replay_records.py",
    "scripts/analysis/paired_tasks.py",
    "scripts/spec_tools/audit_oracle_contracts.py",
)


# Inputs whose bytes MUST be hash-bound in the manifest when they exist under the
# project root (gate12 finding 4, 2026-08-13): a live source-of-truth the chain
# reads but the manifest previously omitted, so a post-generation edit was
# undetectable. Kept in sync with the generate_final_evidence inputs list by
# test_required_inputs_bound_by_orchestrator.
REQUIRED_INPUTS = (
    "config/final_pair_contracts.json",
    # gate22 finding 1: the frozen SLoC corpus registry defines the population the
    # SLoC and suite-composition evidence is computed over (loaded by
    # sloc_analysis.CORPUS_KERNELS, reused by benchmark_characterization). Bind and
    # re-hash it so a post-generation registry edit is caught.
    "config/sloc_corpus_kernels.json",
)


# --------------------------------------------------------------------------- #
# Independent record loading (no analysis-module imports)                      #
# --------------------------------------------------------------------------- #

def _load_records(root: Path) -> list[dict]:
    records = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.json")):
            if path.name.startswith("batch_") or path.name == "eval_summary.json":
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            # Inline parent-identity promotion (do not import replay_records).
            parent = rec.get("parent")
            if isinstance(parent, dict):
                for k in ("model", "source_spec", "target_spec", "augment_level", "sample_id"):
                    if rec.get(k) in (None, "") and parent.get(k) not in (None, ""):
                        rec[k] = parent[k]
            rec["_model"] = model_dir.name
            # gate20 finding 5: the run_fail records_by_bucket key is
            # "<model>/<file-stem>"; the generator's load_records carries the
            # stem, so the verifier must too to reproduce the exact record keys.
            rec["_stem"] = path.stem
            records.append(rec)
    return records


# The paper's headline model (declared independently; the verifier imports no
# analysis module). Kept equal to generate_final_evidence.CLAIMS_MODEL.
CLAIMS_MODEL = "together-qwen-3.5-397b-a17b"

# Together AI pricing (USD per 1M tokens); mirrors quantitative_findings.
_TOGETHER_INPUT_PRICE_PER_M = 0.60
_TOGETHER_OUTPUT_PRICE_PER_M = 3.60


def independent_token_cost(records: list[dict], project_root: Path) -> dict:
    """Recompute token totals/cost from records WITHOUT importing the analysis.

    gate7 finding 4: the verifier used to only resolve the shipped value against
    the shipped JSON, so a $0.00 token cost (finding 2) passed. This reproduces the
    recovery independently: replay records carry no token fields, so read them from
    the submitted record named by input_record.path.
    """
    ti = to = 0
    for r in records:
        pt = r.get("prompt_tokens") or 0
        ct = r.get("completion_tokens") or 0
        if not (pt or ct):
            ref = (r.get("input_record") or {}).get("path")
            if ref:
                rp = Path(ref)
                if not rp.is_absolute():
                    rp = project_root / ref
                try:
                    src = json.loads(rp.read_text(encoding="utf-8"))
                    pt = src.get("prompt_tokens") or 0
                    ct = src.get("completion_tokens") or 0
                except (OSError, json.JSONDecodeError):
                    pt = ct = 0
        ti += pt
        to += ct
    cost = ti * _TOGETHER_INPUT_PRICE_PER_M / 1_000_000 + to * _TOGETHER_OUTPUT_PRICE_PER_M / 1_000_000
    return {"total_input_tokens": ti, "total_output_tokens": to, "total_cost": round(cost, 4)}


def independent_augmentation(records: list[dict]) -> dict:
    """Recompute the MATCHED augmentation per-level counts (gate7 findings 1/4).

    Independent of quantitative_findings: restrict to tasks carrying an augmented
    (L>=1) record and collapse each (task, level) to one any-sample-success cell,
    so every level spans the same task set. Returns levels/pass_counts/total_counts
    the verifier compares to the shipped cochran_armitage block.
    """
    by_cell: dict[tuple, list[str]] = {}
    for r in records:
        key = (r.get("source_spec"), r.get("target_spec"), r.get("augment_level", 0) or 0)
        by_cell.setdefault(key, []).append(r.get("overall_status", ""))
    matched = {(s, t) for (s, t, lv) in by_cell if lv >= 1}
    levels = sorted({lv for (s, t, lv) in by_cell if (s, t) in matched})
    pass_counts: list[int] = []
    total_counts: list[int] = []
    for lv in levels:
        p = n = 0
        for (s, t, l2), statuses in by_cell.items():
            if l2 != lv or (s, t) not in matched:
                continue
            n += 1
            if any(x == "PASS" for x in statuses):
                p += 1
        pass_counts.append(p)
        total_counts.append(n)
    return {"levels": levels, "pass_counts": pass_counts, "total_counts": total_counts}


def _contract_eligibility_failures(records: list[dict], contract_path: Path) -> list[str]:
    """Bind correctness eligibility to the FROZEN contract, not just a code
    constant (gate6 finding 4, 2026-08-13).

    Two independent checks, reimplemented here (no analysis-module import):
      1. The contract's ``exclusions.correctness_ineligible_specs`` must equal the
         canonical ``CORRECTNESS_INELIGIBLE_SPECS`` the analysis uses. Otherwise
         both paths can agree on a stale exclusion set.
      2. Every correctness-eligible record's (source_spec, target_spec) pair must
         appear in the contract's frozen ``eligible_task_list``. A spec-eligible
         pair absent from the frozen registry must be rejected, never counted.
    """
    failures: list[str] = []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"contract manifest unreadable for eligibility check: {e}"]

    excl = set((contract.get("exclusions") or {}).get("correctness_ineligible_specs") or [])
    if excl != set(CORRECTNESS_INELIGIBLE_SPECS):
        failures.append(
            "contract correctness_ineligible_specs != canonical "
            f"CORRECTNESS_INELIGIBLE_SPECS (contract has {len(excl)}, "
            f"canonical {len(CORRECTNESS_INELIGIBLE_SPECS)})"
        )

    eligible_pairs = {
        (t.get("source_spec"), t.get("target_spec"))
        for t in (contract.get("eligible_task_list") or {}).get("tasks", [])
    }
    if not eligible_pairs:
        failures.append("contract eligible_task_list is empty or missing")
        return failures

    offenders = set()
    for rec in records:
        s, t = rec.get("source_spec"), rec.get("target_spec")
        if s in CORRECTNESS_INELIGIBLE_SPECS or t in CORRECTNESS_INELIGIBLE_SPECS:
            continue
        if (s, t) not in eligible_pairs:
            offenders.add((s, t))
    for s, t in sorted(offenders):
        failures.append(f"correctness-eligible pair absent from frozen registry: {s} -> {t}")
    return failures


def _submitted_aggregate(root: Path) -> dict:
    """Independent aggregate hash over every record JSON under ``root``.

    Reimplemented here (shares no code with the generator) so the check is a
    genuine cross-verification (gate1 finding 5). Must match the generator's
    _aggregate_sha256 byte-for-byte: sorted "relpath\\0filehash" joined by \\n.
    """
    lines = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.json")):
            if path.name.startswith("batch_") or path.name == "eval_summary.json":
                continue
            rel = path.relative_to(root).as_posix()
            lines.append(f"{rel}\0{hashlib.sha256(path.read_bytes()).hexdigest()}")
    agg = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {"aggregate_sha256": agg, "file_count": len(lines)}


def _stats(records: list[dict]) -> dict:
    status_counts: dict[str, int] = {}
    per_model: dict[str, dict[str, int]] = {}
    elig_pass = elig_total = 0
    for rec in records:
        status = rec.get("overall_status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
        model = rec.get("model") or rec.get("_model") or "unknown"
        per_model.setdefault(model, {})[status] = per_model.setdefault(model, {}).get(status, 0) + 1
        # Correctness eligibility excludes KNOWN_FAIL AND performance_only specs
        # (gate3 finding 1), matching the canonical analysis denominator.
        if (rec.get("source_spec") not in CORRECTNESS_INELIGIBLE_SPECS
                and rec.get("target_spec") not in CORRECTNESS_INELIGIBLE_SPECS):
            elig_total += 1
            if status == "PASS":
                elig_pass += 1
    total = len(records)
    passes = status_counts.get("PASS", 0)
    return {
        "total_records": total,
        "status_counts": {k: v for k, v in status_counts.items() if v},
        "per_model_status_counts": {m: per_model[m] for m in sorted(per_model)},
        "raw_pass_count": passes,
        "raw_pass_rate": round(passes / total, 6) if total else 0.0,
        "eligible_records": elig_total,
        "eligible_pass_count": elig_pass,
        "eligible_pass_rate": round(elig_pass / elig_total, 6) if elig_total else 0.0,
    }


def _canonical_overall(records: list[dict]) -> dict:
    """Reproduce canonical.aggregate_pass_rates.overall from scratch.

    Canonical set: every record whose source/target spec is NOT correctness-
    ineligible (KNOWN_FAIL or performance-only). The corpus carries no temp=0
    legacy records, so the temp>0 canonical filter is a no-op here. Value is
    round(PASS / total, 4) - the same rounding quantitative_findings.wilson_ci
    applies. This shares no code with the analysis package.
    """
    total = passes = 0
    for rec in records:
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        total += 1
        if rec.get("overall_status") == "PASS":
            passes += 1
    return {"value": round(passes / total, 4) if total else 0.0, "n": total}


def _transitions(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for rec in records:
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        counts[f"{old}->{new}"] = counts.get(f"{old}->{new}", 0) + 1
    return dict(sorted(counts.items()))


def _strata_transitions(records: list[dict]) -> dict:
    """Independent per-strata old->new transition counts (gate13 finding 2).

    Reimplements the generator's populations (per-model/suite over all levels,
    per-direction L0 only, all correctness-eligible) but shares no code with it,
    so a divergence in either fails the comparison recompute."""
    by_model: dict[str, dict[str, int]] = {}
    by_suite: dict[str, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}

    def _suite(spec):
        return spec.split("-", 1)[0] if spec else "unknown"

    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    def _bump(agg: dict, key: str, transition: str) -> None:
        cell = agg.setdefault(key, {})
        cell[transition] = cell.get(transition, 0) + 1

    for rec in records:
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        transition = f"{old}->{new}"
        _bump(by_model, rec.get("model") or rec.get("_model") or "unknown", transition)
        _bump(by_suite, _suite(rec.get("source_spec")), transition)
        if rec.get("augment_level") == 0:
            _bump(by_direction,
                  f"{_api(rec.get('source_spec'))}-to-{_api(rec.get('target_spec'))}",
                  transition)

    def _sort(agg: dict) -> dict:
        return {k: dict(sorted(agg[k].items())) for k in sorted(agg)}

    return {"by_model": _sort(by_model), "by_suite": _sort(by_suite),
            "by_direction": _sort(by_direction)}


def _semantic_transitions(records: list[dict], specs_dir: Path) -> dict:
    """Independent per-oracle-shape old->new transition counts (gate13 finding 2).
    Same population as _semantic_classification; shares no code with the
    generator."""
    agg: dict[str, dict[str, int]] = {}
    for rec in records:
        if rec.get("augment_level") != 0:
            continue
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        shape = oracle_shape(rec.get("target_spec"), specs_dir)
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        cell = agg.setdefault(shape, {})
        cell[f"{old}->{new}"] = cell.get(f"{old}->{new}", 0) + 1
    return {s: dict(sorted(agg[s].items())) for s in sorted(agg)}


def _headline_cell_transitions(records: list[dict]) -> dict:
    """Independent per-cell old->new transition matrices behind every changing
    headline-table cell (gate17 finding 2). Reimplements
    generate_final_evidence.headline_cell_transitions: L0 correctness-eligible
    records; direction = source_api-to-target_api; suite = source-spec prefix;
    kernel = middle of the source spec; task cells collapsed by any-sample
    success on new and parent (old) status. Shares no code with the generator."""
    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    def _suite(spec):
        return spec.split("-", 1)[0] if spec else "unknown"

    def _kernel(spec):
        if not spec:
            return "unknown"
        parts = spec.split("-")
        return "-".join(parts[1:-1]) if len(parts) >= 3 else spec

    cm_rec: dict[str, dict[str, dict[str, int]]] = {}
    suite_rec: dict[str, dict[str, int]] = {}
    rod_rec: dict[str, dict[str, int]] = {"rodinia": {}, "others": {}}
    new_cell: dict[tuple, list[str]] = {}
    old_cell: dict[tuple, list[str]] = {}
    for rec in records:
        if rec.get("augment_level") != 0:
            continue
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        model = rec.get("model") or rec.get("_model") or "unknown"
        suite = _suite(rec.get("source_spec"))
        direction = f"{_api(rec.get('source_spec'))}-to-{_api(rec.get('target_spec'))}"
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        tr = f"{old}->{new}"
        cm_rec.setdefault(model, {}).setdefault(direction, {})
        cm_rec[model][direction][tr] = cm_rec[model][direction].get(tr, 0) + 1
        cell = suite_rec.setdefault(suite, {})
        cell[tr] = cell.get(tr, 0) + 1
        grp = "rodinia" if suite == "rodinia" else "others"
        rod_rec[grp][tr] = rod_rec[grp].get(tr, 0) + 1
        key = (model, suite, _kernel(rec.get("source_spec")), direction)
        new_cell.setdefault(key, []).append(new)
        old_cell.setdefault(key, []).append(old)

    cm_task: dict[str, dict[str, dict[str, int]]] = {}
    rod_task: dict[str, dict[str, int]] = {"rodinia": {}, "others": {}}
    for key, new_st in new_cell.items():
        model, suite, _k, direction = key
        new = "PASS" if any(x == "PASS" for x in new_st) else "FAIL"
        old = "PASS" if any(x == "PASS" for x in old_cell[key]) else "FAIL"
        tr = f"{old}->{new}"
        cm_task.setdefault(model, {}).setdefault(direction, {})
        cm_task[model][direction][tr] = cm_task[model][direction].get(tr, 0) + 1
        grp = "rodinia" if suite == "rodinia" else "others"
        rod_task[grp][tr] = rod_task[grp].get(tr, 0) + 1

    def _s2(agg):
        return {m: {d: dict(sorted(agg[m][d].items())) for d in sorted(agg[m])}
                for m in sorted(agg)}

    def _s1(agg):
        return {k: dict(sorted(agg[k].items())) for k in sorted(agg)}

    return {
        "cross_model": {
            "record_unit_by_model_direction": _s2(cm_rec),
            "task_unit_by_model_direction": _s2(cm_task),
        },
        "suite_composition": {
            "per_suite_record_unit": _s1(suite_rec),
            "record_unit_rodinia_vs_others": _s1(rod_rec),
            "task_unit_rodinia_vs_others": _s1(rod_task),
        },
    }


# gate18 finding 3: independent copies of the collapse helpers. The verifier
# shares no code with the generator; these must produce byte-identical structures.
def _v_api(spec):
    return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"


def _v_suite(spec):
    return spec.split("-", 1)[0] if spec else "unknown"


def _v_kernel(spec):
    if not spec:
        return "unknown"
    parts = spec.split("-")
    return "-".join(parts[1:-1]) if len(parts) >= 3 else spec


def _v_record_key(rec: dict) -> str:
    return "|".join([
        str(rec.get("model") or rec.get("_model") or "unknown"),
        str(rec.get("source_spec") or ""),
        str(rec.get("target_spec") or ""),
        f"L{rec.get('augment_level', 0) or 0}",
        f"s{rec.get('sample_id', 0)}",
    ])


def _v_collapse(statuses: list[str]) -> str:
    return "PASS" if any(s == "PASS" for s in statuses) else "FAIL"


def _v_matrix(rows: list[dict]) -> dict:
    m: dict[str, int] = {}
    for row in rows:
        tr = f"{row['old']}->{row['new']}"
        m[tr] = m.get(tr, 0) + 1
    return dict(sorted(m.items()))


def _v_task_cell_rows(records: list[dict]) -> list[dict]:
    new_by: dict[tuple, list[str]] = {}
    old_by: dict[tuple, list[str]] = {}
    keys_by: dict[tuple, list[str]] = {}
    for r in records:
        k = (r.get("source_spec"), r.get("target_spec"))
        new_by.setdefault(k, []).append(r.get("overall_status", ""))
        old_by.setdefault(k, []).append((r.get("parent") or {}).get("overall_status", ""))
        keys_by.setdefault(k, []).append(_v_record_key(r))
    rows = []
    for (s, t) in sorted(new_by):
        rows.append({
            "source_spec": s, "target_spec": t,
            "direction": f"{_v_api(s)}-to-{_v_api(t)}",
            "suite": _v_suite(s),
            "old": _v_collapse(old_by[(s, t)]),
            "new": _v_collapse(new_by[(s, t)]),
            "records": sorted(keys_by[(s, t)]),
        })
    return rows


def _v_raw_record_rows(records: list[dict]) -> list[dict]:
    """Independent one-row-per-RAW-record table (gate19 finding 2). old/new are
    the record's verbatim parent/replay overall_status; sorted by record key."""
    rows = []
    for r in records:
        rows.append({
            "source_spec": r.get("source_spec"),
            "target_spec": r.get("target_spec"),
            "direction": f"{_v_api(r.get('source_spec'))}-to-"
                         f"{_v_api(r.get('target_spec'))}",
            "suite": _v_suite(r.get("source_spec")),
            "old": (r.get("parent") or {}).get("overall_status", ""),
            "new": r.get("overall_status", ""),
            "record": _v_record_key(r),
        })
    return sorted(rows, key=lambda x: x["record"])


def _v_raw_record_table(records: list[dict]) -> dict:
    rows = _v_raw_record_rows(records)
    return {"unit": "raw_record", "rows": rows, "matrix": _v_matrix(rows)}


def _v_task_cn_rows(records: list[dict]) -> list[dict]:
    """Independent per-task c/n table (gate19 finding 2): pass count / sample count
    per task on both sides, the unit pass@1 and per-suite pass@1 macro-average."""
    old_pass: dict[tuple, int] = {}
    new_pass: dict[tuple, int] = {}
    n: dict[tuple, int] = {}
    keys_by: dict[tuple, list[str]] = {}
    for r in records:
        k = (r.get("source_spec"), r.get("target_spec"))
        n[k] = n.get(k, 0) + 1
        if r.get("overall_status") == "PASS":
            new_pass[k] = new_pass.get(k, 0) + 1
        if (r.get("parent") or {}).get("overall_status") == "PASS":
            old_pass[k] = old_pass.get(k, 0) + 1
        keys_by.setdefault(k, []).append(_v_record_key(r))
    rows = []
    for (s, t) in sorted(n):
        rows.append({
            "source_spec": s, "target_spec": t,
            "direction": f"{_v_api(s)}-to-{_v_api(t)}",
            "suite": _v_suite(s),
            "old_pass": old_pass.get((s, t), 0), "old_n": n[(s, t)],
            "new_pass": new_pass.get((s, t), 0), "new_n": n[(s, t)],
            "records": sorted(keys_by[(s, t)]),
        })
    return rows


def _v_task_cn_table(records: list[dict]) -> dict:
    rows = _v_task_cn_rows(records)
    matrix = {
        "old_pass": sum(r["old_pass"] for r in rows),
        "old_n": sum(r["old_n"] for r in rows),
        "new_pass": sum(r["new_pass"] for r in rows),
        "new_n": sum(r["new_n"] for r in rows),
        # gate20 finding 6: independent per-RAW-record old->new transition counts,
        # mirroring generate_final_evidence._task_cn_table. The comparison's
        # claim_scoped_transitions byte-compare enforces this against the file, so
        # a task-cn change with a fabricated/absent transition matrix is rejected.
        "transitions": _v_matrix(_v_raw_record_rows(records)),
    }
    return {"unit": "task_cn", "rows": rows, "matrix": matrix}


def _claim_scoped_transitions(records: list[dict], claims_model: str) -> dict:
    """Independent per-claim-population transition tables for the headline model,
    with ROWS = the claim's statistical UNIT (gate18 finding 3; gate19 finding 2:
    pass@1 is per-task c/n, pass@3 is any-sample task cells, the per-direction
    rate is raw records, per-suite pass@1 is per-task c/n). Mirrors
    generate_final_evidence.claim_scoped_transitions exactly but shares no code."""
    hl = [r for r in records
          if (r.get("model") or r.get("_model")) == claims_model
          and r.get("augment_level") == 0
          and r.get("source_spec") not in CORRECTNESS_INELIGIBLE_SPECS
          and r.get("target_spec") not in CORRECTNESS_INELIGIBLE_SPECS]

    def _partition(field, table_fn):
        keys = sorted({
            (f"{_v_api(r.get('source_spec'))}-to-{_v_api(r.get('target_spec'))}"
             if field == "direction" else _v_suite(r.get("source_spec")))
            for r in hl})
        out = {}
        for key in keys:
            if field == "direction":
                sub = [r for r in hl
                       if f"{_v_api(r.get('source_spec'))}-to-"
                          f"{_v_api(r.get('target_spec'))}" == key]
            else:
                sub = [r for r in hl if _v_suite(r.get("source_spec")) == key]
            out[key] = table_fn(sub)
        return out

    return {
        "model": claims_model,
        "overall_L0": {"unit": "task_cell", "rows": _v_task_cell_rows(hl),
                       "matrix": _v_matrix(_v_task_cell_rows(hl))},
        "overall_L0_pass_at_1": _v_task_cn_table(hl),
        "by_direction_L0": _partition("direction", _v_raw_record_table),
        "by_suite_L0": _partition("suite", _v_task_cn_table),
        "claims_model_all_levels": _claims_model_all_levels_tr(records, claims_model),
        "matched_augmentation_by_level": _matched_augmentation_tr(records, claims_model),
        "pooled_by_direction_L0": _pooled_paired_cells(records),
        "pooled_overall_L0": _kernel_group_table(records),
    }


def _claims_model_all_levels_tr(records: list[dict], claims_model: str) -> dict:
    """Independent overall old->new matrix over the canonical (all-augment-level,
    correctness-eligible) headline-model population (gate17 finding 3)."""
    overall: dict[str, int] = {}
    for rec in records:
        if (rec.get("model") or rec.get("_model")) != claims_model:
            continue
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        overall[f"{old}->{new}"] = overall.get(f"{old}->{new}", 0) + 1
    return dict(sorted(overall.items()))


def _matched_augmentation_tr(records: list[dict], claims_model: str) -> dict:
    """Independent matched-augmentation cells for the headline model in the
    CUDA->OMP population (gate18 finding 3). ROWS are matched TASK CELLS per level,
    each carrying its record keys. Mirrors the generator exactly."""
    new_by: dict[tuple, list[str]] = {}
    old_by: dict[tuple, list[str]] = {}
    keys_by: dict[tuple, list[str]] = {}
    for rec in records:
        if (rec.get("model") or rec.get("_model")) != claims_model:
            continue
        s, t = rec.get("source_spec"), rec.get("target_spec")
        if f"{_v_api(s)}-to-{_v_api(t)}" != "cuda-to-omp":
            continue
        lv = rec.get("augment_level", 0) or 0
        new_by.setdefault((s, t, lv), []).append(rec.get("overall_status", ""))
        old_by.setdefault((s, t, lv), []).append(
            (rec.get("parent") or {}).get("overall_status", ""))
        keys_by.setdefault((s, t, lv), []).append(_v_record_key(rec))
    matched = {(s, t) for (s, t, lv) in new_by if lv >= 1}
    by_level: dict[int, list[dict]] = {}
    for (s, t, lv) in sorted(new_by):
        if (s, t) not in matched:
            continue
        by_level.setdefault(lv, []).append({
            "source_spec": s, "target_spec": t,
            "old": _v_collapse(old_by[(s, t, lv)]),
            "new": _v_collapse(new_by[(s, t, lv)]),
            "records": sorted(keys_by[(s, t, lv)]),
        })
    return {
        "unit": "matched_task_cell",
        "direction": "cuda-to-omp",
        "by_level": {
            f"L{lv}": {"unit": "matched_task_cell", "rows": by_level[lv],
                       "matrix": _v_matrix(by_level[lv])}
            for lv in sorted(by_level)
        },
    }


def _chi2_from_table(table: list[list[int]]) -> float:
    """Pearson chi-square of a contingency table, no Yates correction (matches
    scipy.chi2_contingency for tables larger than 2x2 - the augmentation table is
    5x2). Reimplemented inline so the verifier needs neither scipy nor an analysis
    module (gate18 finding 1)."""
    rows = len(table)
    cols = len(table[0]) if rows else 0
    grand = sum(sum(r) for r in table)
    if grand == 0:
        return 0.0
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    chi = 0.0
    for i in range(rows):
        for j in range(cols):
            exp = row_tot[i] * col_tot[j] / grand
            if exp > 0:
                chi += (table[i][j] - exp) ** 2 / exp
    return chi


def _matched_augmentation_chi2(records: list[dict],
                               claims_model: str,
                               direction: str = "cuda-to-omp") -> float | None:
    """Independent MATCHED-collapse augmentation chi-square for the headline model
    (gate18 finding 1). Mirrors statistical_analysis.test_augmentation_independence
    AFTER the finding-1 fix: restrict to the model's records in the augmented
    direction, collapse each (task, level) to one cell by any-sample success, keep
    only tasks carrying an augmented (L>=1) record, then chi-square the
    level x [pass, fail] table. Returns None when fewer than two levels survive or
    a row/column sums to zero (the generator skips the test in those cases)."""
    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    by_cell: dict[tuple, list[str]] = {}
    for rec in records:
        if (rec.get("model") or rec.get("_model")) != claims_model:
            continue
        s, t = rec.get("source_spec"), rec.get("target_spec")
        if f"{_api(s)}-to-{_api(t)}" != direction:
            continue
        lv = rec.get("augment_level", 0) or 0
        by_cell.setdefault((s, t, lv), []).append(rec.get("overall_status", ""))
    matched = {(s, t) for (s, t, lv) in by_cell if lv >= 1}
    levels: dict[int, dict[str, int]] = {}
    for (s, t, lv), statuses in by_cell.items():
        if (s, t) not in matched:
            continue
        passed = any(x == "PASS" for x in statuses)
        cell = levels.setdefault(lv, {"pass": 0, "fail": 0})
        cell["pass" if passed else "fail"] += 1
    if len(levels) < 2:
        return None
    table = [[levels[lv]["pass"], levels[lv]["fail"]] for lv in sorted(levels)]
    if any(sum(r) == 0 for r in table):
        return None
    col_tot = [sum(table[i][j] for i in range(len(table))) for j in range(2)]
    if any(c == 0 for c in col_tot):
        return None
    return _chi2_from_table(table)


def _pooled_paired_cells(records: list[dict]) -> dict:
    """Independent direction-paired cells over the ALL-model, L0 population
    (gate18 finding 3). ROWS are the PAIRED CELLS the McNemar asymmetry test uses,
    each carrying both directions' record keys. Mirrors the generator exactly."""
    collapsed: dict[tuple, dict] = {}
    for rec in records:
        if rec.get("augment_level") != 0:
            continue
        # gate19 finding 3: correctness eligibility (drop KNOWN_FAIL + mixbench).
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        s = rec.get("source_spec")
        key = (rec.get("model") or rec.get("_model") or "unknown",
               _v_suite(s), _v_kernel(s),
               f"{_v_api(s)}-to-{_v_api(rec.get('target_spec'))}")
        cell = collapsed.setdefault(
            key, {"pass": False, "old_pass": False, "records": []})
        cell["pass"] = cell["pass"] or (rec.get("overall_status") == "PASS")
        cell["old_pass"] = cell["old_pass"] or (
            (rec.get("parent") or {}).get("overall_status") == "PASS")
        cell["records"].append(_v_record_key(rec))
    directions = {k[3] for k in collapsed}
    pairs: dict[str, dict] = {}
    for d in sorted(directions):
        parts = d.split("-to-")
        if len(parts) != 2:
            continue
        rev = f"{parts[1]}-to-{parts[0]}"
        if rev not in directions:
            continue
        fwd, rvs = tuple(sorted([d, rev]))
        label = f"{fwd} vs {rvs}"
        if label in pairs:
            continue
        rows = []
        for (m, su, ke, direction) in sorted(collapsed):
            if direction != fwd:
                continue
            rev_cell = collapsed.get((m, su, ke, rvs))
            if rev_cell is None:
                continue
            fwd_cell = collapsed[(m, su, ke, fwd)]
            rows.append({
                "model": m, "suite": su, "kernel": ke,
                "forward": fwd, "reverse": rvs,
                "forward_pass": fwd_cell["pass"],
                "reverse_pass": rev_cell["pass"],
                "forward_old_pass": fwd_cell["old_pass"],
                "reverse_old_pass": rev_cell["old_pass"],
                "records": sorted(fwd_cell["records"] + rev_cell["records"]),
            })
        pairs[label] = {
            "unit": "direction_pair_cell", "rows": rows,
            "matrix": _v_pair_matrix(rows, "forward_pass", "reverse_pass"),
            "old_matrix": _v_pair_matrix(rows, "forward_old_pass", "reverse_old_pass"),
            "transition": _v_pair_transition(rows),
        }
    return pairs


def _v_pair_state(fwd: bool, rev: bool) -> str:
    return f"{'PASS' if fwd else 'FAIL'}/{'PASS' if rev else 'FAIL'}"


def _v_pair_matrix(rows: list[dict], fwd_key: str, rev_key: str) -> dict:
    m: dict[str, int] = {}
    for r in rows:
        k = _v_pair_state(r[fwd_key], r[rev_key])
        m[k] = m.get(k, 0) + 1
    return dict(sorted(m.items()))


def _v_pair_transition(rows: list[dict]) -> dict:
    m: dict[str, int] = {}
    for r in rows:
        old = _v_pair_state(r["forward_old_pass"], r["reverse_old_pass"])
        new = _v_pair_state(r["forward_pass"], r["reverse_pass"])
        k = f"{old}->{new}"
        m[k] = m.get(k, 0) + 1
    return dict(sorted(m.items()))


def _kernel_group_table(records: list[dict]) -> dict:
    """Independent kernel groups over the correctness-eligible, all-level, all-model
    population (gate18 finding 3) - the SLoC-vs-pass correlation unit. ROWS are the
    kernel groups, each carrying its record keys. Mirrors the generator exactly."""
    groups: dict[str, dict] = {}
    transition: dict[str, int] = {}
    for rec in records:
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        k = _v_kernel(rec.get("source_spec"))
        g = groups.setdefault(
            k, {"records": [], "pass": 0, "fail": 0, "old_pass": 0, "old_fail": 0})
        g["records"].append(_v_record_key(rec))
        new = rec.get("overall_status")
        old = (rec.get("parent") or {}).get("overall_status")
        if new == "PASS":
            g["pass"] += 1
        else:
            g["fail"] += 1
        if old == "PASS":
            g["old_pass"] += 1
        else:
            g["old_fail"] += 1
        tk = f"{'PASS' if old == 'PASS' else 'FAIL'}->{'PASS' if new == 'PASS' else 'FAIL'}"
        transition[tk] = transition.get(tk, 0) + 1
    rows = []
    total_pass = total_fail = total_old_pass = total_old_fail = 0
    for k in sorted(groups):
        g = groups[k]
        total_pass += g["pass"]
        total_fail += g["fail"]
        total_old_pass += g["old_pass"]
        total_old_fail += g["old_fail"]
        rows.append({
            "kernel": k, "n_records": len(g["records"]),
            "pass": g["pass"], "fail": g["fail"],
            "old_pass": g["old_pass"], "old_fail": g["old_fail"],
            "records": sorted(g["records"]),
        })
    return {"unit": "kernel_group", "rows": rows,
            "matrix": {"PASS": total_pass, "FAIL": total_fail},
            "old_matrix": {"PASS": total_old_pass, "FAIL": total_old_fail},
            "transition": dict(sorted(transition.items()))}


# Independent copy of the record-derived claim-source set (gate13 finding 2);
# kept equal to generate_final_evidence.RECORD_DERIVED_CLAIM_SOURCES by a source
# test. The verifier shares no code with the generator's classifier.
_RECORD_DERIVED_CLAIM_SOURCES = frozenset({
    "statistical_analysis.json",
    "cross_model_comparison.json",
    "suite_composition.json",
    "error_taxonomy.json",
})


# gate17 finding 3: the record-derived claim ids. Independent copy of
# generate_final_evidence.RECORD_DERIVED_CLAIM_IDS, kept equal by a source test.
_RECORD_DERIVED_CLAIM_IDS = frozenset({
    "overall_pass_at_1", "overall_pass_at_3",
    "aggregate_pass_rate",
    "best_direction", "worst_direction",
    "build_fail_count", "run_fail_count", "verify_fail_count",
    "build_fail_subcategories",
    "augmentation_degradation", "chi2_augmentation_trend",
    "direction_asymmetry", "sloc_correlation",
})

# gate17 finding 3: json_path prefixes / (source, json_path) pairs whose value is
# a function of result-record STATUS. This is an INDEPENDENT signal from the
# claim id set above: the verifier asserts the two agree, so a claim that is
# record-derived by its source/json_path but mislabeled not_applicable is caught.
_RECORD_DERIVED_JSON_PREFIXES = (
    "canonical.aggregate_pass_rates",
    "canonical.pass_at_k",
    "canonical.direction_pass_rates",
    "canonical.failure_taxonomy.status_counts",
    "canonical.failure_taxonomy.by_status",
    "canonical.augmentation_trends",
)
_RECORD_DERIVED_SOURCE_PATHS = frozenset({
    "statistical_analysis.json:direction_asymmetry",
    "statistical_analysis.json:chi2_augmentation_by_model",
    "sloc_analysis.json:summary.sloc_vs_pass_rate_spearman",
})


def _claim_is_record_derived(claim: dict) -> bool:
    """Independent classification (gate17 finding 3): True when the claim value
    is a function of result-record status, so not_applicable is FORBIDDEN.
    Classifies from the claim's source + json_path (NOT from the id list or the
    scope), so a mislabeled scope on a record-derived claim is caught."""
    src = (claim.get("source_files") or [None])[0]
    jp = claim.get("json_path") or ""
    if src is None or not jp:
        return False                       # removed-class: no value to move
    base = Path(src).name
    if base.startswith("quantitative_findings"):
        return any(jp.startswith(p) for p in _RECORD_DERIVED_JSON_PREFIXES)
    return f"{base}:{jp}" in _RECORD_DERIVED_SOURCE_PATHS


def _claim_transition_scope(claim: dict) -> str:
    # gate17 finding 3: every record-derived claim points at its OWN
    # exact-population cell; only a non-record claim is 'not_applicable', never
    # 'corpus'. Independent copy of generate_final_evidence.claim_transition_scope
    # (kept in sync by a source test).
    cid = claim.get("claim_id") or ""
    if cid.startswith("direction_pass_"):
        return ("claim_scoped:by_direction_L0:"
                + cid[len("direction_pass_"):].replace("_", "-"))
    if cid.startswith("suite_pass_"):
        return "claim_scoped:by_suite_L0:" + cid[len("suite_pass_"):]
    if cid in ("best_direction", "worst_direction"):
        jp = claim.get("json_path") or ""
        if "direction_pass_rates.standard." in jp:
            return "claim_scoped:by_direction_L0:" + jp.rsplit(".", 1)[-1]
        return "claim_scoped:overall_L0"
    # gate19 finding 2: pass@1 (per-task c/n) and pass@3 (any-sample task cell)
    # point at different-unit tables.
    if cid == "overall_pass_at_1":
        return "claim_scoped:overall_L0_pass_at_1"
    if cid == "overall_pass_at_3":
        return "claim_scoped:overall_L0"
    if cid in ("aggregate_pass_rate", "build_fail_count", "run_fail_count",
               "verify_fail_count", "build_fail_subcategories"):
        return "claim_scoped:claims_model_all_levels"
    if cid in ("augmentation_degradation", "chi2_augmentation_trend"):
        return "claim_scoped:matched_augmentation_by_level"
    if cid == "direction_asymmetry":
        return "claim_scoped:pooled_by_direction_L0"
    if cid == "sloc_correlation":
        return "claim_scoped:pooled_overall_L0"
    return "not_applicable"


def _claim_transition_reason(claim: dict) -> str:
    # Independent copy of generate_final_evidence.claim_transition_reason
    # (gate17 finding 3): only non-record claims reach here.
    src = (claim.get("source_files") or [None])[0]
    if not src:
        return "no source artifact; not a record-status claim"
    return ("value is a spec-, file-count-, or token-cost quantity, not a "
            "function of result-record status; it carries no old->new status "
            "transition")


# gate21 finding 4: neither claim is source-bound-by-hashing any more (hashing a
# producer output proves consistency, not correctness):
#   - build_fail_subcategories is RECORD-derived - recomputed below by an
#     INDEPENDENT copy of the build-fail classifier (_v_classify_build_fail), so
#     it leaves this set and takes the generic record-recompute path.
#   - sloc_correlation is NOT purely record-derived (per-kernel SLoC is a
#     benchmark-source property that no record carries), so it cannot use the
#     record-only path; it stays here but is INDEPENDENTLY recomputed by a
#     dedicated Spearman over its verified per-kernel pairs (see the comparison
#     loop), not merely re-resolved from its source JSON.
_CLAIM_VALUE_SOURCE_ONLY = frozenset({"sloc_correlation"})


# gate21 finding 4: an INDEPENDENT copy of quantitative_findings._BUILD_FAIL_
# PATTERNS + _classify_build_fail, so build_fail_subcategories is recomputed from
# raw records instead of re-resolved from the analysis output. A source-sync test
# (test_final_evidence) asserts this copy stays byte-identical to the generator's
# ordered ruleset, so the two classifiers cannot drift. The verifier imports no
# analysis module; the ruleset is copied, not imported.
_V_BUILD_FAIL_PATTERNS = [
    ("retained_cuda_api", re.compile(
        r"cudaMalloc|cudaFree|cudaMemcpy|cudaMemset|cudaDeviceSynchronize"
        r"|__global__|__device__|__shared__"
        r"|<<<|>>>"
        r"|cudaGetLastError|cudaSetDevice|cudaGetDevice"
        r"|cuda_runtime\.h|cudaError_t|cudaSuccess"
        r"|blockIdx|threadIdx|blockDim|gridDim"
        r"|__syncthreads|atomicAdd\b|atomicSub\b|atomicExch\b"
        r"|atomicMin\b|atomicMax\b|atomicCAS\b"
        r"|calling a __device__ function.*from a __host__",
        re.IGNORECASE), lambda d: not d.endswith("-to-cuda")),
    ("retained_cuda_types", re.compile(
        r"\bfloat3\b|\bfloat4\b|\bint3\b|\bint4\b|\bdouble3\b|\bdouble4\b"
        r"|\bdim3\b|cudaStream_t|cudaEvent_t"
        r"|texture<|surf<|__constant__",
        re.IGNORECASE), lambda d: not d.endswith("-to-cuda")),
    ("retained_opencl_api", re.compile(
        r"clCreateBuffer|clEnqueueWriteBuffer|clEnqueueReadBuffer"
        r"|clBuildProgram|clCreateKernel|clSetKernelArg"
        r"|clEnqueueNDRangeKernel|clReleaseMemObject"
        r"|get_global_id|get_local_id|__kernel\b|cl_mem\b",
        re.IGNORECASE), lambda d: not d.endswith("-to-opencl")),
    ("missing_header", re.compile(
        r"No such file or directory"
        r"|fatal error:.*not found"
        r"|cannot find.*include",
        re.IGNORECASE), None),
    ("linker_error", re.compile(
        r"undefined reference to"
        r"|multiple definition of"
        r"|collect2:.*error"
        r"|ld returned \d+ exit",
        re.IGNORECASE), None),
    ("undeclared_identifier", re.compile(
        r"\bundeclared\b"
        r"|was not declared in this scope"
        r"|identifier.*is undefined"
        r"|does not name a type"
        r"|unknown type name"
        r"|has no member named",
        re.IGNORECASE), None),
    ("type_mismatch", re.compile(
        r"cannot convert"
        r"|incompatible type"
        r"|no matching function"
        r"|conflicting types"
        r"|array subscript is not an integer"
        r"|no viable conversion"
        r"|no instance of overloaded"
        r"|invalid.*operands? to binary"
        r"|cannot be used to initialize",
        re.IGNORECASE), None),
    ("syntax_error", re.compile(
        r"expected.*before"
        r"|expected.*token"
        r"|expected a [\"'(]|expected a declaration"
        r"|stray.*in program"
        r"|parse error"
        r"|unterminated"
        r"|invalid.*form.*pragma"
        r"|is not valid for.*#pragma"
        r"|must be closely nested"
        r"|too few arguments to function"
        r"|too many arguments (?:to function|in function call)"
        r"|no storage class or type specifier"
        r"|missing closing quote",
        re.IGNORECASE), None),
    ("implicit_declaration", re.compile(
        r"implicit declaration of function", re.IGNORECASE), None),
    ("redefinition", re.compile(
        r"redefinition of"
        r"|redeclaration of"
        r"|duplicate member"
        r"|previously declared here",
        re.IGNORECASE), None),
]


def _v_classify_build_fail(rec: dict) -> str:
    """Independent copy of quantitative_findings._classify_build_fail (gate21
    finding 4): primary BUILD_FAIL subcategory from the final attempt's build-error
    snippet, ordered ruleset, ``other_build`` fallback."""
    snippet = rec.get("build_error_snippet") or ""
    attempts = rec.get("attempts") or []
    if attempts:
        last = attempts[-1].get("build_error_snippet")
        if last:
            snippet = last
    src, tgt = rec.get("source_spec", ""), rec.get("target_spec", "")
    direction = rec.get("direction") or (
        f"{src.rsplit('-', 1)[-1] if src else 'unknown'}-to-"
        f"{tgt.rsplit('-', 1)[-1] if tgt else 'unknown'}")
    for name, pattern, dfilter in _V_BUILD_FAIL_PATTERNS:
        if dfilter is not None and not dfilter(direction):
            continue
        if pattern.search(snippet):
            return name
    return "other_build"


def _spearman_rho(xs, ys):
    """Spearman rank correlation, hand-implemented (the verifier imports no scipy
    or analysis module). Average-rank ties, Pearson on the ranks. Returns None for
    fewer than two points or a degenerate (zero-variance) ranking - matching the
    analysis, which reports no correlation in those cases."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None

    def _ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0            # average rank (1-based)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n))
    dy = sum((ry[i] - my) ** 2 for i in range(n))
    if dx == 0 or dy == 0:
        return None
    return num / math.sqrt(dx * dy)


def _independent_record_claim_value(claim: dict, records: list[dict]):
    """Recompute a record-derived claim's VALUE directly from raw records using
    its declared population and statistical unit (gate20 finding 3). Returns the
    recomputed value (rounded to the source's 4 decimals for rates), or None when
    the claim is not one this function covers. Shares no code with the analysis
    modules; the population filters mirror _claim_scoped_transitions. Closes the
    hole where a self-consistent forged source JSON (e.g. a fabricated Qwen
    cuda-to-omp rate of 9.0) passed because the value was only re-resolved from
    that same forged file."""
    cid = claim.get("claim_id") or ""
    jp = claim.get("json_path") or ""

    def _elig(r):
        return (r.get("source_spec") not in CORRECTNESS_INELIGIBLE_SPECS
                and r.get("target_spec") not in CORRECTNESS_INELIGIBLE_SPECS)

    def _is_model(r):
        return (r.get("model") or r.get("_model")) == CLAIMS_MODEL

    hl = [r for r in records
          if _is_model(r) and r.get("augment_level") == 0 and _elig(r)]
    all_lev = [r for r in records if _is_model(r) and _elig(r)]

    def _frac_pass(rows):
        return round(sum(1 for r in rows if r.get("overall_status") == "PASS")
                     / len(rows), 4) if rows else None

    # Direction pass rate (raw records, Qwen L0 eligible) - direction from json_path.
    if "direction_pass_rates.standard." in jp:
        d = jp.rsplit(".", 1)[-1]
        return _frac_pass([r for r in hl
                           if f"{_v_api(r.get('source_spec'))}-to-"
                              f"{_v_api(r.get('target_spec'))}" == d])
    # Per-suite pass@1 (raw records, Qwen L0 eligible) - suite from json_path.
    if "pass_at_k.per_suite.pass_at_1." in jp:
        s = jp.rsplit(".", 1)[-1]
        return _frac_pass([r for r in hl if _v_suite(r.get("source_spec")) == s])
    # Overall pass@1 (micro, Qwen L0 eligible).
    if jp == "canonical.pass_at_k.pass_at_1":
        return _frac_pass(hl)
    # Overall pass@3 (macro: fraction of tasks with any-sample PASS, Qwen L0).
    if jp == "canonical.pass_at_k.pass_at_3":
        tasks: dict[tuple, list[str]] = {}
        for r in hl:
            tasks.setdefault((r.get("source_spec"), r.get("target_spec")), []).append(
                r.get("overall_status"))
        return round(sum(1 for k in tasks if any(x == "PASS" for x in tasks[k]))
                     / len(tasks), 4) if tasks else None
    # Aggregate pass rate (PASS fraction, Qwen ALL levels eligible).
    if jp == "canonical.aggregate_pass_rates.overall":
        return _frac_pass(all_lev)
    # Failure-taxonomy status counts (Qwen ALL levels eligible).
    if jp.startswith("canonical.failure_taxonomy.status_counts."):
        st = jp.rsplit(".", 1)[-1]
        return sum(1 for r in all_lev if r.get("overall_status") == st)
    # Aggregate augmentation degradation = L4.value - L0.value over the matched
    # (any-direction) per-level population, each rate rounded to 4 first.
    if cid == "augmentation_degradation":
        new_by: dict[tuple, list[str]] = {}
        for r in records:
            if not _is_model(r) or not _elig(r):
                continue
            k = (r.get("source_spec"), r.get("target_spec"),
                 r.get("augment_level", 0) or 0)
            new_by.setdefault(k, []).append(r.get("overall_status"))
        matched = {(s, t) for (s, t, lv) in new_by if lv >= 1}
        per_level: dict[int, list[int]] = {}
        for (s, t, lv), sts in new_by.items():
            if (s, t) not in matched:
                continue
            cell = per_level.setdefault(lv, [0, 0])
            cell[1] += 1
            if any(x == "PASS" for x in sts):
                cell[0] += 1
        if 0 not in per_level or 4 not in per_level:
            return None
        r0 = round(per_level[0][0] / per_level[0][1], 4)
        r4 = round(per_level[4][0] / per_level[4][1], 4)
        return round(r4 - r0, 4)
    # Direction asymmetry = max over paired directions of |fwd_rate - rev_rate|,
    # each rate rounded to 4 before the gap (mirrors the source's stored rates).
    if cid == "direction_asymmetry":
        best = 0.0
        for tab in _pooled_paired_cells(records).values():
            rows = tab.get("rows") or []
            if not rows:
                continue
            fr = round(sum(1 for r in rows if r["forward_pass"]) / len(rows), 4)
            rr = round(sum(1 for r in rows if r["reverse_pass"]) / len(rows), 4)
            best = max(best, round(abs(fr - rr), 4))
        return round(best, 4)
    # Matched-collapse augmentation chi-square (reuses the independent helper).
    if cid == "chi2_augmentation_trend":
        v = _matched_augmentation_chi2(records, CLAIMS_MODEL)
        return round(v, 4) if v is not None else None
    # gate21 finding 4: top-3 BUILD_FAIL subcategories, recomputed from records by
    # the INDEPENDENT classifier copy - no longer re-resolved from the analysis
    # output. Population is the model's all-levels eligible BUILD_FAIL records (the
    # denominator build_fail_count recomputes); order matches _top_n_items (count
    # desc, name asc), and the value is the [[name, count], ...] top-3 list.
    if cid == "build_fail_subcategories":
        counts: dict[str, int] = {}
        for r in all_lev:
            if r.get("overall_status") == "BUILD_FAIL":
                k = _v_classify_build_fail(r)
                counts[k] = counts.get(k, 0) + 1
        if not counts:
            return None
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [[k, v] for k, v in ordered[:3]]
    return None


def _independent_record_claim_components(claim: dict, records: list[dict]) -> dict:
    """Recompute the record-derivable NUMERIC COMPONENTS of a claim directly from
    raw records (gate21 finding 3), so the Wilson interval bounds, denominators,
    and augmentation endpoints are not merely re-resolved from a hash-pinned source
    JSON (a self-consistent forgery would pass that). Returns ``{component_name:
    recomputed_value}`` for every component this function covers; a component absent
    from the returned dict is verified by source-resolution only (the p-values and
    the forward/reverse orientation need the statistical test / PUBLISHED_FORWARD
    orientation - their record-derived INPUTS, the chi2 statistic and the asymmetry
    gap, ARE independently recomputed as the claim VALUE). Shares the population
    filters with _independent_record_claim_value."""
    cid = claim.get("claim_id") or ""
    jp = claim.get("json_path") or ""

    def _elig(r):
        return (r.get("source_spec") not in CORRECTNESS_INELIGIBLE_SPECS
                and r.get("target_spec") not in CORRECTNESS_INELIGIBLE_SPECS)

    def _is_model(r):
        return (r.get("model") or r.get("_model")) == CLAIMS_MODEL

    hl = [r for r in records
          if _is_model(r) and r.get("augment_level") == 0 and _elig(r)]
    all_lev = [r for r in records if _is_model(r) and _elig(r)]

    def _wilson_of(rows):
        passes = sum(1 for r in rows if r.get("overall_status") == "PASS")
        w = _wilson(passes, len(rows))
        return {"ci_lower": w["ci_lower"], "ci_upper": w["ci_upper"], "n": len(rows)}

    # Wilson-over-records interval claims (the analysis uses wilson_ci(passes,
    # total) for these two): recompute the SAME micro Wilson from records. The
    # pass@1 / pass@3 / per-suite intervals instead use a macro normal-approx over
    # per-task pass@k floats, so recomputing them as Wilson would spuriously
    # disagree; their bounds stay source-resolve-verified (their point estimates
    # ARE independently recomputed as the claim VALUE), and they are still compared
    # old-vs-new. Recompute here only what the analysis computes as Wilson-over-
    # records, so a match is meaningful rather than an artefact of a wrong method.
    rows = None
    if "direction_pass_rates.standard." in jp:
        d = jp.rsplit(".", 1)[-1]
        rows = [r for r in hl
                if f"{_v_api(r.get('source_spec'))}-to-"
                   f"{_v_api(r.get('target_spec'))}" == d]
    elif jp == "canonical.aggregate_pass_rates.overall":
        rows = all_lev
    if rows is not None:
        return _wilson_of(rows)

    # Augmentation endpoints: the L0 and L4 matched-per-level pass rates.
    if cid == "augmentation_degradation":
        by: dict[tuple, list[str]] = {}
        for r in records:
            if not _is_model(r) or not _elig(r):
                continue
            k = (r.get("source_spec"), r.get("target_spec"),
                 r.get("augment_level", 0) or 0)
            by.setdefault(k, []).append(r.get("overall_status"))
        matched = {(s, t) for (s, t, lv) in by if lv >= 1}
        per_level: dict[int, list[int]] = {}
        for (s, t, lv), sts in by.items():
            if (s, t) not in matched:
                continue
            cell = per_level.setdefault(lv, [0, 0])
            cell[1] += 1
            if any(x == "PASS" for x in sts):
                cell[0] += 1
        out: dict = {}
        if 0 in per_level:
            out["l0_rate"] = round(per_level[0][0] / per_level[0][1], 4)
        if 4 in per_level:
            out["l4_rate"] = round(per_level[4][0] / per_level[4][1], 4)
        return out

    return {}


def _claim_transitions(claim: dict) -> dict:
    # Independent copy of generate_final_evidence.claim_transitions
    # (gate16 finding 2).
    scope = _claim_transition_scope(claim)
    if scope.startswith("claim_scoped:"):
        return {"scope": scope}
    return {"not_applicable": True, "reason": _claim_transition_reason(claim)}


def _semantic_classification(records: list[dict], specs_dir: Path) -> dict:
    """Independent per-oracle-shape denominator/pass rate/Wilson interval over
    the correctness-eligible L0 records (gate2 finding 6; KNOWN_FAIL and
    performance_only both excluded - gate3 finding 1). Mirrors the generator's
    population and estimator but shares no code with it."""
    agg: dict[str, dict[str, int]] = {}
    for rec in records:
        if rec.get("augment_level") != 0:
            continue
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        shape = oracle_shape(rec.get("target_spec"), specs_dir)
        a = agg.setdefault(shape, {"n": 0, "passes": 0})
        a["n"] += 1
        if rec.get("overall_status") == "PASS":
            a["passes"] += 1
    out = {}
    for shape in sorted(agg):
        n, passes = agg[shape]["n"], agg[shape]["passes"]
        ci = _wilson(passes, n)
        out[shape] = {
            "n": n, "passes": passes, "pass_rate": ci["value"],
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
        }
    return out


def _strata(records: list[dict]) -> dict:
    """Independent per-model/suite/direction denominator/pass rate/Wilson interval
    over the correctness-eligible records (gate3 finding 3). Reimplements the
    generator's population and estimator; shares no code with it. Per-model and
    per-suite span all levels; per-direction is L0 only."""
    by_model: dict[str, dict[str, int]] = {}
    by_suite: dict[str, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}

    def _acc(agg: dict, key: str, is_pass: bool) -> None:
        cell = agg.setdefault(key, {"n": 0, "passes": 0})
        cell["n"] += 1
        cell["passes"] += 1 if is_pass else 0

    def _suite(spec):
        return spec.split("-", 1)[0] if spec else "unknown"

    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    def _finalize(agg: dict) -> dict:
        out = {}
        for name in sorted(agg):
            n, passes = agg[name]["n"], agg[name]["passes"]
            ci = _wilson(passes, n)
            out[name] = {
                "n": n, "passes": passes, "pass_rate": ci["value"],
                "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            }
        return out

    for rec in records:
        if (rec.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or rec.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        is_pass = rec.get("overall_status") == "PASS"
        _acc(by_model, rec.get("model") or rec.get("_model") or "unknown", is_pass)
        _acc(by_suite, _suite(rec.get("source_spec")), is_pass)
        if rec.get("augment_level") == 0:
            _acc(by_direction, f"{_api(rec.get('source_spec'))}-to-"
                 f"{_api(rec.get('target_spec'))}", is_pass)
    return {
        "by_model": _finalize(by_model),
        "by_suite": _finalize(by_suite),
        "by_direction": _finalize(by_direction),
    }


def _cross_model_headline(records: list[dict], models: list[str],
                          common_dirs: list[str]) -> tuple[dict, dict]:
    """Independent per-model overall + per-direction pass/total/rate for the
    cross-model headline table (gate12 finding 3). Population mirrors the
    generator's pass@k campaign: L0, correctness-eligible records. `overall`
    spans all directions for the model (matching pass@k `overall`); the
    per-direction map is restricted to `common_dirs`. Shares no code with
    cross_model_comparison; rate is round(pass/total, 4) as the generator does."""
    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    overall = {m: [0, 0] for m in models}                      # [passes, total]
    per_dir = {d: {m: [0, 0] for m in models} for d in common_dirs}
    for r in records:
        if r.get("augment_level") != 0:
            continue
        m = r.get("model") or r.get("_model")
        if m not in overall:
            continue
        if (r.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or r.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        is_pass = r.get("overall_status") == "PASS"
        overall[m][1] += 1
        overall[m][0] += 1 if is_pass else 0
        d = f"{_api(r.get('source_spec'))}-to-{_api(r.get('target_spec'))}"
        if d in per_dir:
            per_dir[d][m][1] += 1
            per_dir[d][m][0] += 1 if is_pass else 0

    def _cell(passes, total):
        return {"pass": passes, "total": total,
                "rate": round(passes / total, 4) if total else 0.0}

    return (
        {m: _cell(*overall[m]) for m in models},
        {d: {m: _cell(*per_dir[d][m]) for m in models} for d in common_dirs},
    )


def _parent_binding_failures(new_records: list[dict], submitted_root: Path,
                             project_root: Path) -> list[str]:
    """Bind every replay record's embedded parent to the referenced submitted
    record before its status is used for a transition (gate14 finding 4).

    The old->new transition counts read ``parent.overall_status`` as the OLD
    verdict. Nothing previously proved that ``parent`` describes the submitted
    record named by ``input_record.path``: a seal built with stale or mispaired
    parent metadata would fabricate transitions and still pass. For each replay
    record this requires:
      * an ``input_record.path`` that resolves to a real submitted JSON,
      * the file's SHA-256 equals the recorded ``input_record.sha256``,
      * a one-to-one path (no submitted record referenced twice), and
      * every ``parent`` identity field AND ``overall_status`` equals the
        submitted record's own value.
    Shares no code with the sealing/replay tooling."""
    failures: list[str] = []
    seen: dict[str, str] = {}
    id_fields = ("model", "source_spec", "target_spec", "augment_level",
                 "sample_id", "overall_status")
    for rec in new_records:
        stem = rec.get("_stem", "?")
        ir = rec.get("input_record") or {}
        ref = ir.get("path")
        if not ref:
            failures.append(
                f"replay record {stem}: no input_record.path to bind parent to")
            continue
        if ref in seen:
            failures.append(
                f"input_record.path {ref} referenced by multiple replay records "
                f"({seen[ref]}, {stem}); binding must be one-to-one")
        seen[ref] = stem
        rp = Path(ref)
        if not rp.is_absolute():
            rp = project_root / ref
        try:
            raw = rp.read_bytes()
        except OSError:
            failures.append(
                f"replay record {stem}: submitted record not found at {ref}")
            continue
        want_sha = ir.get("sha256")
        got_sha = hashlib.sha256(raw).hexdigest()
        if want_sha and got_sha != want_sha:
            failures.append(
                f"replay record {stem}: input_record.sha256 mismatch for {ref} "
                f"(recorded {want_sha}, file {got_sha})")
            continue
        try:
            submitted = json.loads(raw)
        except json.JSONDecodeError:
            failures.append(
                f"replay record {stem}: submitted record {ref} is not valid JSON")
            continue
        parent = rec.get("parent") or {}
        for field in id_fields:
            if parent.get(field) != submitted.get(field):
                failures.append(
                    f"replay record {stem}: parent.{field}="
                    f"{parent.get(field)!r} != submitted[{field}]="
                    f"{submitted.get(field)!r} at {ref}")
    return failures


def _model_task_pass_counts(records: list[dict], model: str) -> dict[str, int]:
    """{kernel:direction -> passing-sample count} over L0 correctness-eligible
    records for one model (gate17 finding 2). Mirrors generate_paper_data's
    passk_estimates task keying so McNemar concordance recomputes from records."""
    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    def _kernel(spec):
        parts = (spec or "").split("-")
        return "-".join(parts[1:-1]) if len(parts) >= 3 else (spec or "unknown")

    c: dict[str, int] = {}
    for r in records:
        if r.get("augment_level") != 0:
            continue
        if (r.get("model") or r.get("_model")) != model:
            continue
        if (r.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or r.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        key = (f"{_kernel(r.get('source_spec'))}:"
               f"{_api(r.get('source_spec'))}-to-{_api(r.get('target_spec'))}")
        c[key] = c.get(key, 0) + (1 if r.get("overall_status") == "PASS" else 0)
    return c


def _model_kernel_pass(records: list[dict], model: str) -> dict[str, int]:
    """{kernel -> passing-sample count} over L0 correctness-eligible records for
    one model (gate17 finding 2). A kernel 'passes' for the per-kernel matrix
    when its count is > 0 (any sample passed)."""
    def _kernel(spec):
        parts = (spec or "").split("-")
        return "-".join(parts[1:-1]) if len(parts) >= 3 else (spec or "unknown")

    p: dict[str, int] = {}
    for r in records:
        if r.get("augment_level") != 0:
            continue
        if (r.get("model") or r.get("_model")) != model:
            continue
        if (r.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or r.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        k = _kernel(r.get("source_spec"))
        p[k] = p.get(k, 0) + (1 if r.get("overall_status") == "PASS" else 0)
    return p


def _cross_model_mcnemar(records: list[dict], model_a: str, model_b: str) -> dict:
    """Independent McNemar concordance counts (gate17 finding 2). Mirrors
    cross_model_comparison.compute_mcnemar over the paired task set: a task
    'passes' for a model when its passing-sample count >= 1."""
    est_a = _model_task_pass_counts(records, model_a)
    est_b = _model_task_pass_counts(records, model_b)
    both_pass = both_fail = a_only = b_only = 0
    for task in sorted(set(est_a) & set(est_b)):
        ap, bp = est_a[task] >= 1, est_b[task] >= 1
        if ap and bp:
            both_pass += 1
        elif not ap and not bp:
            both_fail += 1
        elif ap:
            a_only += 1
        else:
            b_only += 1
    return {"both_pass": both_pass, "both_fail": both_fail,
            "a_only": a_only, "b_only": b_only,
            "total": both_pass + both_fail + a_only + b_only}


def _cross_model_per_kernel_counts(records: list[dict], model_a: str,
                                   model_b: str) -> dict:
    """Independent per-kernel four-way agreement counts (gate17 finding 2).
    Mirrors cross_model_comparison's per_kernel_matrix over common kernels; a
    model 'passes' a kernel when any sample passed."""
    pa = _model_kernel_pass(records, model_a)
    pb = _model_kernel_pass(records, model_b)
    both_pass = both_fail = a_only = b_only = 0
    common = sorted(set(pa) & set(pb))
    for k in common:
        ap, bp = pa[k] > 0, pb[k] > 0
        if ap and bp:
            both_pass += 1
        elif not ap and not bp:
            both_fail += 1
        elif ap:
            a_only += 1
        else:
            b_only += 1
    return {"total_common_kernels": len(common),
            "both_pass": both_pass, "both_fail": both_fail,
            f"{model_a}_only_pass": a_only, f"{model_b}_only_pass": b_only}


def _cross_model_headline_failures(doc: dict, records: list[dict],
                                   side: str) -> list[str]:
    """Independently recompute a cross-model comparison document from records.

    gate14 finding 2: the verifier previously recomputed ONLY the top-level
    Codex-vs-GPT-5.4 table and ignored every `pairwise` entry. For a >2-model
    panel the headline model (Qwen) appears ONLY in pairwise, so its cross-model
    counts, denominators, and rates went unchecked. This recomputes the
    top-level table AND every pairwise entry, each against ``records`` with the
    entry's own ``models`` and ``common_directions``. Shares no code with
    cross_model_comparison."""
    failures: list[str] = []

    def _check(sub: dict, label: str) -> None:
        models = sub.get("models") or []
        common = sub.get("common_directions") or []
        r_overall, r_dir = _cross_model_headline(records, models, common)
        for m in models:
            have = (sub.get("overall") or {}).get(m, {})
            got = r_overall.get(m, {})
            for k in ("pass", "total", "rate"):
                if not _num_eq(have.get(k), got.get(k)):
                    failures.append(
                        f"cross_model_comparison {side}{label} overall"
                        f"[{m}].{k}: file={have.get(k)!r} "
                        f"recomputed={got.get(k)!r}")
        for d in common:
            for m in models:
                have = ((sub.get("per_direction") or {}).get(d) or {}).get(m, {})
                got = r_dir.get(d, {}).get(m, {})
                for k in ("pass", "total", "rate"):
                    if not _num_eq(have.get(k), got.get(k)):
                        failures.append(
                            f"cross_model_comparison {side}{label} "
                            f"per_direction[{d}][{m}].{k}: "
                            f"file={have.get(k)!r} recomputed={got.get(k)!r}")
        # gate17 finding 2: recompute the McNemar concordance totals and the
        # per-kernel four-way counts from records - byte-checking the overall
        # pass/total/rate left these numeric headline fields unverified.
        if len(models) == 2:
            mc = (sub.get("overall") or {}).get("mcnemar")
            if isinstance(mc, dict):
                got_mc = _cross_model_mcnemar(records, models[0], models[1])
                for k in ("both_pass", "both_fail", "a_only", "b_only", "total"):
                    if not _num_eq(mc.get(k), got_mc.get(k)):
                        failures.append(
                            f"cross_model_comparison {side}{label} "
                            f"mcnemar.{k}: file={mc.get(k)!r} "
                            f"recomputed={got_mc.get(k)!r}")
            pk = sub.get("per_kernel_matrix")
            if isinstance(pk, dict):
                got_pk = _cross_model_per_kernel_counts(
                    records, models[0], models[1])
                counts = pk.get("counts") or {}
                if not _num_eq(pk.get("total_common_kernels"),
                               got_pk["total_common_kernels"]):
                    failures.append(
                        f"cross_model_comparison {side}{label} "
                        f"per_kernel_matrix.total_common_kernels: "
                        f"file={pk.get('total_common_kernels')!r} "
                        f"recomputed={got_pk['total_common_kernels']!r}")
                for k in ("both_pass", "both_fail",
                          f"{models[0]}_only_pass", f"{models[1]}_only_pass"):
                    if not _num_eq(counts.get(k), got_pk.get(k)):
                        failures.append(
                            f"cross_model_comparison {side}{label} "
                            f"per_kernel_matrix.counts[{k}]: "
                            f"file={counts.get(k)!r} recomputed={got_pk.get(k)!r}")

    _check(doc, "")
    for key in sorted(doc.get("pairwise") or {}):
        _check(doc["pairwise"][key], f" pairwise[{key}]")
    return failures


def _suite_record_unit(records: list[dict]) -> dict:
    """Independent per-source-suite record-unit n/passes/rate for the
    suite-composition headline table (gate12 finding 3). Population mirrors
    suite_composition.load_l0: L0, correctness-eligible records grouped by the
    source spec's suite. rate is round(passes/n, 4)."""
    def _suite(spec):
        return spec.split("-", 1)[0] if spec else "unknown"

    agg: dict[str, list[int]] = {}
    for r in records:
        if r.get("augment_level") != 0:
            continue
        if (r.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or r.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        cell = agg.setdefault(_suite(r.get("source_spec")), [0, 0])
        cell[1] += 1
        cell[0] += 1 if r.get("overall_status") == "PASS" else 0
    return {s: {"n": t, "passes": p, "rate": round(p / t, 4) if t else 0.0}
            for s, (p, t) in agg.items()}


def _suite_composition_units(records: list[dict]) -> dict:
    """Independent suite-composition scope + record/task-unit rodinia-vs-others
    counts (gate17 finding 2). Mirrors suite_composition.py: L0 correctness-
    eligible records; scope.records = record count, scope.tasks = number of
    (model, suite, kernel, direction) task cells; record_unit counts every
    sample, task_unit collapses each cell by any-sample success. Only the
    record-derived counts are recomputed (n, passes); CIs/rates are checked
    against those counts elsewhere."""
    def _suite(spec):
        return spec.split("-", 1)[0] if spec else "unknown"

    def _api(spec):
        return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"

    def _kernel(spec):
        parts = (spec or "").split("-")
        return "-".join(parts[1:-1]) if len(parts) >= 3 else (spec or "unknown")

    rec_group = {"rodinia": [0, 0], "others": [0, 0]}     # [passes, n]
    cells_new: dict[tuple, bool] = {}
    for r in records:
        if r.get("augment_level") != 0:
            continue
        if (r.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                or r.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
            continue
        suite = _suite(r.get("source_spec"))
        grp = "rodinia" if suite == "rodinia" else "others"
        is_pass = r.get("overall_status") == "PASS"
        rec_group[grp][1] += 1
        rec_group[grp][0] += 1 if is_pass else 0
        key = (r.get("model") or r.get("_model") or "unknown", suite,
               _kernel(r.get("source_spec")),
               f"{_api(r.get('source_spec'))}-to-{_api(r.get('target_spec'))}")
        cells_new[key] = cells_new.get(key, False) or is_pass

    task_group = {"rodinia": [0, 0], "others": [0, 0]}
    for (model, suite, _k, _d), ok in cells_new.items():
        grp = "rodinia" if suite == "rodinia" else "others"
        task_group[grp][1] += 1
        task_group[grp][0] += 1 if ok else 0

    records_n = rec_group["rodinia"][1] + rec_group["others"][1]
    return {
        "scope": {"records": records_n, "tasks": len(cells_new)},
        "record_unit": {g: {"n": rec_group[g][1], "passes": rec_group[g][0]}
                        for g in ("rodinia", "others")},
        "task_unit": {g: {"n": task_group[g][1], "passes": task_group[g][0]}
                      for g in ("rodinia", "others")},
    }


def _namespace_file_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under ``root`` except the seal marker.

    Mirrors scripts/provenance/seal_result_namespace._hash_namespace byte-for-
    byte so the sealed replay namespace can be recomputed here (gate4 finding 1).
    """
    files: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == ".parbench-seal.json":
            continue
        files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return files


def _seal_aggregate(files: dict[str, str]) -> str:
    """Aggregate hash matching seal_result_namespace._aggregate: sorted
    ``"{rel}\\t{filehash}\\n"`` (gate4 finding 1)."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\t{files[rel]}\n".encode())
    return h.hexdigest()


def _num_eq(a, b) -> bool:
    """Numeric-tolerant equality; mirrors the generator's supported/qualified
    rule so claim old/new values re-resolve identically (gate4 finding 2)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return a == b


def _claim_status(old_v, new_v) -> str:
    """Re-derive a claim's supported/qualified disposition from its old and new
    value, matching generate_final_evidence.build_claim_status (gate4 finding 2)."""
    if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
        return "supported" if abs(float(new_v) - float(old_v)) <= 1e-9 else "qualified"
    return "supported" if old_v == new_v else "qualified"


def _run_fail_split(records: list[dict]) -> dict:
    buckets = {"run_timeout": 0, "nonzero_exit_or_crash": 0, "other": 0}
    for rec in records:
        if (rec.get("parent") or {}).get("overall_status") != "PASS":
            continue
        if rec.get("overall_status") != "RUN_FAIL":
            continue
        rs = rec.get("run_status")
        if rs == "timeout":
            buckets["run_timeout"] += 1
        elif rs == "fail":
            buckets["nonzero_exit_or_crash"] += 1
        else:
            buckets["other"] += 1
    return buckets


def _run_fail_split_full(records: list[dict]) -> dict:
    """gate20 finding 5: independently recompute the COMPLETE run_fail mechanism
    object the comparison carries - per-model counts, exact record keys, and each
    record's exit-code/runtime - not just the three aggregate bucket counts the
    manifest headline holds. Mirrors generate_final_evidence.run_fail_mechanism_
    split's record-derived fields exactly (the static ``note`` prose is not a
    record quantity, so it is not recomputed here)."""
    buckets: dict[str, list] = {"run_timeout": [], "nonzero_exit_or_crash": [], "other": []}
    by_model: dict[str, dict[str, int]] = {}
    for rec in records:
        old = (rec.get("parent") or {}).get("overall_status")
        new = rec.get("overall_status")
        if old != "PASS" or new != "RUN_FAIL":
            continue
        rs = rec.get("run_status")
        bucket = ("run_timeout" if rs == "timeout"
                  else "nonzero_exit_or_crash" if rs == "fail" else "other")
        model = rec.get("model") or rec.get("_model") or "unknown"
        buckets[bucket].append({
            "record": f"{model}/{rec.get('_stem')}",
            "model": model,
            "run_status": rs,
            "run_exit_code": rec.get("run_exit_code"),
            "run_time_seconds": rec.get("run_time_seconds"),
        })
        bm = by_model.setdefault(
            model, {"run_timeout": 0, "nonzero_exit_or_crash": 0, "other": 0})
        bm[bucket] += 1
    return {
        "total_pass_to_run_fail": sum(len(v) for v in buckets.values()),
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "by_model": {m: by_model[m] for m in sorted(by_model)},
        "records_by_bucket": {
            k: sorted(v, key=lambda r: r["record"]) for k, v in buckets.items()
        },
    }


# --------------------------------------------------------------------------- #
# Verification                                                                 #
# --------------------------------------------------------------------------- #

def verify(manifest: dict, project_root: Path,
           expected_direction_pairs: int | None,
           require_outputs: list[Path],
           immutable_baseline: Path | None) -> list[str]:
    failures: list[str] = []

    replay_root = Path(manifest["replay_root"]["path"])
    submitted_root = Path(manifest["submitted_root"]["path"])

    new_records = _load_records(replay_root)
    old_records = _load_records(submitted_root)

    # 0a. Bind every replay record's embedded parent to the submitted record it
    # references (gate14 finding 4). The transition counts trust
    # parent.overall_status as the OLD verdict, so this must run BEFORE any
    # transition is believed: a stale/mispaired parent that fabricates a
    # transition fails here.
    failures.extend(_parent_binding_failures(
        new_records, submitted_root, project_root))

    # 0. Correctness eligibility is bound to the FROZEN contract, not merely the
    # code constant (gate6 finding 4). Enforced whenever the manifest records a
    # resolvable contract path; the real manifest always does. Its recorded sha
    # is re-checked in the input-hash pass below.
    cm = manifest.get("contract_manifest") or {}
    cm_path = cm.get("path")
    if cm_path:
        cm_resolved = Path(cm_path) if Path(cm_path).is_absolute() else project_root / cm_path
        failures.extend(_contract_eligibility_failures(new_records, cm_resolved))

    hl = manifest.get("headline") or {}

    # 1. Record-level counts/rates: recompute and compare.
    for label, recorded, computed in [
        ("new", hl.get("new"), _stats(new_records)),
        ("old", hl.get("old"), _stats(old_records)),
    ]:
        if recorded is None:
            failures.append(f"manifest headline missing '{label}' block")
            continue
        for key in ("total_records", "status_counts", "per_model_status_counts",
                    "raw_pass_count", "raw_pass_rate", "eligible_records",
                    "eligible_pass_count", "eligible_pass_rate"):
            if recorded.get(key) != computed.get(key):
                failures.append(
                    f"{label}.{key}: manifest={recorded.get(key)!r} "
                    f"recomputed={computed.get(key)!r}"
                )

    # 1b. Canonical overall pass rate + denominator: recompute independently.
    rec_canon = hl.get("canonical_overall") or {}
    for label, records in (("new", new_records), ("old", old_records)):
        recorded = rec_canon.get(label)
        if recorded is None:
            failures.append(f"manifest headline missing canonical_overall '{label}'")
            continue
        computed = _canonical_overall(records)
        for key in ("value", "n"):
            if recorded.get(key) != computed.get(key):
                failures.append(
                    f"canonical_overall.{label}.{key}: manifest={recorded.get(key)!r} "
                    f"recomputed={computed.get(key)!r}"
                )

    # 1c. Per-model/suite/direction strata: independently recompute every
    # denominator, rate, and interval and compare (gate3 finding 3). The headline
    # previously carried only the overall canonical rate.
    rec_strata = hl.get("strata") or {}
    for label, records in (("new", new_records), ("old", old_records)):
        recorded = rec_strata.get(label)
        if recorded is None:
            failures.append(f"manifest headline missing strata '{label}'")
            continue
        computed = _strata(records)
        for group in ("by_model", "by_suite", "by_direction"):
            rg, cg = recorded.get(group) or {}, computed.get(group)
            if set(rg) != set(cg):
                failures.append(
                    f"strata.{label}.{group} keys: manifest={sorted(rg)} "
                    f"recomputed={sorted(cg)}"
                )
            for name in sorted(set(rg) & set(cg)):
                for key in ("n", "passes", "pass_rate", "ci_lower", "ci_upper"):
                    if rg[name].get(key) != cg[name].get(key):
                        failures.append(
                            f"strata.{label}.{group}[{name}].{key}: "
                            f"manifest={rg[name].get(key)!r} "
                            f"recomputed={cg[name].get(key)!r}"
                        )

    # 2. Transition counts.
    rec_tr = hl.get("transition_counts")
    got_tr = _transitions(new_records)
    if rec_tr != got_tr:
        failures.append(f"transition_counts: manifest={rec_tr} recomputed={got_tr}")

    # 3. PASS->RUN_FAIL mechanism split.
    rec_split = hl.get("run_fail_mechanism_split")
    got_split = _run_fail_split(new_records)
    if rec_split != got_split:
        failures.append(f"run_fail_mechanism_split: manifest={rec_split} recomputed={got_split}")

    # 4. Every recorded output hash still matches the file on disk.
    for out in manifest.get("outputs", []):
        p = project_root / out["path"] if not Path(out["path"]).is_absolute() else Path(out["path"])
        if not p.exists():
            failures.append(f"output missing: {out['path']}")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != out.get("sha256"):
            failures.append(f"output hash mismatch: {out['path']}")

    # 4b. Every recorded INPUT hash still matches the file on disk, and every
    # generator's recorded hash matches its script (gate1 finding 5): provenance
    # is bound to the generator bytes, not to a commit that may post-date them.
    def _resolve(rel: str) -> Path:
        return project_root / rel if not Path(rel).is_absolute() else Path(rel)

    for label, meta in (manifest.get("inputs") or {}).items():
        sha = meta.get("sha256")
        if sha is None:
            continue
        p = _resolve(meta["path"])
        if not p.exists():
            failures.append(f"input missing: {label} ({meta['path']})")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            failures.append(f"input hash mismatch: {label} ({meta['path']})")

    recorded_generators = manifest.get("generators") or {}
    for rel, sha in recorded_generators.items():
        p = _resolve(rel)
        if not p.exists():
            failures.append(f"generator missing: {rel}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            failures.append(f"generator hash mismatch: {rel}")

    # Every mandatory generator that exists under the project root MUST be
    # hash-bound in the manifest (gate6 finding 3, 2026-08-13). The inventory
    # helper produces the gate-5 input closure yet was previously unbound, so the
    # manifest's commit/generator hashes did not cover the code that built the
    # evidence. The exists() guard keeps hermetic tests (a tmp project root with
    # no real scripts) vacuous while enforcing in the real run.
    for rel in REQUIRED_GENERATORS:
        if _resolve(rel).exists() and rel not in recorded_generators:
            failures.append(f"required generator not bound in manifest: {rel}")

    # Every mandatory INPUT that exists under the project root MUST be hash-bound
    # in the manifest (gate12 finding 4, 2026-08-13). generate_comparator_labels
    # reads config/final_pair_contracts.json to label every comparator pair, so
    # the registry feeds the generated evidence; without binding its bytes a
    # post-generation registry edit is invisible to this verifier (E-16 live
    # source-of-truth synchronization). The exists() guard keeps hermetic tests
    # (a tmp root with no registry) vacuous while enforcing in the real run.
    bound_input_paths = {
        _resolve(meta.get("path", "")).resolve()
        for meta in (manifest.get("inputs") or {}).values()
        if meta.get("path")
    }
    for rel in REQUIRED_INPUTS:
        p = _resolve(rel)
        if p.exists() and p.resolve() not in bound_input_paths:
            failures.append(f"required input not bound in manifest: {rel}")

    # 4c. The immutable submitted corpus aggregate hash must still match.
    sub = manifest.get("submitted_root") or {}
    recorded_agg = sub.get("aggregate_sha256")
    if recorded_agg is not None:
        got = _submitted_aggregate(submitted_root)
        if got["aggregate_sha256"] != recorded_agg:
            failures.append(
                "submitted_root aggregate hash mismatch "
                f"(manifest file_count={sub.get('file_count')}, "
                f"recomputed={got['file_count']})"
            )

    # 4d. Required frozen deliverables must be present and hash-recorded as
    # outputs (gate1 finding 6): the oracle-contract report, the pair-
    # comparability disposition table, and the source-baseline dispositions.
    fd = manifest.get("frozen_deliverables") or {}
    out_paths = {o["path"] for o in manifest.get("outputs", [])}
    for cat in ("oracle_contract_report", "pair_comparability_dispositions",
                "source_baseline_dispositions"):
        listed = fd.get(cat) or []
        if not listed:
            failures.append(f"frozen deliverable category missing/empty: {cat}")
            continue
        for rel in listed:
            if rel not in out_paths:
                failures.append(
                    f"frozen deliverable not recorded as a hashed output: {rel}"
                )

    # 5. Every claim key resolves to a generated claim, AND every claim's value
    # independently resolves from its hashed source to the stated value (gate2
    # findings 3/4): presence alone let a container-valued json_path or a wrong
    # value pass. A derived claim is recomputed from its recipe here.
    claim_keys = manifest.get("claim_keys") or []
    if not claim_keys:
        failures.append("manifest records no claim keys")
    pc_rel = manifest.get("paper_claims_path", "results/analysis/paper_claims.json")
    pc_path = project_root / pc_rel if not Path(pc_rel).is_absolute() else Path(pc_rel)
    if pc_path.exists():
        pc = json.loads(pc_path.read_text(encoding="utf-8"))
        by_id = {c.get("claim_id"): c for c in pc.get("claims", [])}
        for key in claim_keys:
            if key not in by_id:
                failures.append(f"claim key not found in paper_claims.json: {key}")
        for claim in pc.get("claims", []):
            cid = claim.get("claim_id")
            srcs = claim.get("source_files") or []
            jp = claim.get("json_path")
            val = claim.get("value")
            if not srcs or not jp:
                if val is not None:
                    failures.append(f"claim {cid}: value present but no source/json_path")
                continue
            src_path = project_root / srcs[0] if not Path(srcs[0]).is_absolute() else Path(srcs[0])
            if not src_path.exists():
                failures.append(f"claim {cid}: source missing {srcs[0]}")
                continue
            doc = json.loads(src_path.read_text(encoding="utf-8"))
            resolved = _resolve_claim_value(doc, claim)
            if isinstance(val, (int, float)):
                if not isinstance(resolved, (int, float)):
                    failures.append(
                        f"claim {cid}: numeric value did not resolve to a scalar "
                        f"(got {type(resolved).__name__}) from {jp}"
                    )
                elif abs(resolved - val) > 1e-6:
                    failures.append(
                        f"claim {cid}: source resolves {resolved}, claim says {val}"
                    )
            elif resolved != val:
                failures.append(
                    f"claim {cid}: non-numeric mismatch {resolved!r} != {val!r}"
                )
            # gate21 finding 3: every NUMERIC component the claim reports (Wilson
            # interval bounds, denominators, and the extra reported numbers -
            # asymmetry p-values, augmentation endpoints, chi-square p-values) must
            # ALSO resolve from the source and match; the scalar `value` alone left
            # them uncertified. Each component resolves through the same recipe
            # resolver as the value.
            for comp in claim.get("components") or []:
                c_val = comp.get("value")
                c_res = _resolve_claim_value(doc, comp)
                if isinstance(c_val, (int, float)):
                    if not isinstance(c_res, (int, float)):
                        failures.append(
                            f"claim {cid} component {comp.get('name')}: did not "
                            f"resolve to a scalar (got {type(c_res).__name__})"
                        )
                    elif abs(c_res - c_val) > 1e-6:
                        failures.append(
                            f"claim {cid} component {comp.get('name')}: source "
                            f"resolves {c_res}, claim says {c_val}"
                        )
                elif c_res != c_val:
                    failures.append(
                        f"claim {cid} component {comp.get('name')}: "
                        f"{c_res!r} != {c_val!r}"
                    )
    else:
        failures.append("results/analysis/paper_claims.json is missing")

    # 5a. Independently recompute the augmentation, token, and chi2-selection
    # values from the records/statistics (gate7 finding 4). Resolving a shipped
    # value against its own shipped JSON cannot catch a value that was WRONG in
    # both (finding 1 unmatched augmentation, finding 2 $0 token cost, finding 3
    # chi2 attributed to the wrong model). These recomputations do not import the
    # analysis modules and fail closed on mismatch.
    if pc_path.exists():
        analysis_dir = pc_path.parent
        # Match the analysis population: the paper model's records after the SAME
        # correctness-ineligible exclusion quantitative_findings.exclude_known_fail
        # applies (KNOWN_FAIL + performance-only specs). Recomputing over the raw
        # corpus would disagree by construction, not by defect.
        _ineligible = set(CORRECTNESS_INELIGIBLE_SPECS)
        model_recs = [
            r for r in new_records
            if r.get("_model") == CLAIMS_MODEL
            and r.get("source_spec") not in _ineligible
            and r.get("target_spec") not in _ineligible
        ]

        qf_path = analysis_dir / f"quantitative_findings_{CLAIMS_MODEL}.json"
        if qf_path.exists() and model_recs:
            qf_doc = json.loads(qf_path.read_text(encoding="utf-8"))
            canon = qf_doc.get("canonical") or {}

            # Augmentation: matched per-level counts must reproduce.
            ca = ((canon.get("augmentation_trends") or {}).get("aggregate") or {}).get(
                "cochran_armitage") or {}
            got_aug = independent_augmentation(model_recs)
            for field in ("pass_counts", "total_counts"):
                if ca.get(field) != got_aug[field]:
                    failures.append(
                        f"augmentation {field}: shipped={ca.get(field)} "
                        f"recomputed={got_aug[field]}"
                    )
            # The matched population makes every level denominator equal; a level
            # whose total differs from L0 is an unmatched-population regression.
            if got_aug["total_counts"] and len(set(got_aug["total_counts"])) != 1:
                failures.append(
                    f"augmentation total_counts not matched across levels: "
                    f"{got_aug['total_counts']}"
                )

            # Token cost: recovered total must be > 0 and match the shipped value.
            tc = (canon.get("token_cost") or {}).get("total_cost") or {}
            shipped_cost = tc.get("value") if isinstance(tc, dict) else None
            got_cost = independent_token_cost(model_recs, project_root)
            if got_cost["total_cost"] <= 0.0:
                failures.append("token cost recomputed to <= 0 (finding 2 regression)")
            if isinstance(shipped_cost, (int, float)) and \
                    abs(shipped_cost - got_cost["total_cost"]) > 1e-4:
                failures.append(
                    f"token total_cost: shipped={shipped_cost} "
                    f"recomputed={got_cost['total_cost']}"
                )

        # chi2 augmentation claim must name the paper model's own statistic.
        sa_path = analysis_dir / "statistical_analysis.json"
        if sa_path.exists():
            sa_doc = json.loads(sa_path.read_text(encoding="utf-8"))
            entries = sa_doc.get("chi2_augmentation_by_model") or []
            model_entry = next(
                (e for e in entries
                 if e.get("group") == CLAIMS_MODEL or e.get("model") == CLAIMS_MODEL),
                None,
            )
            # gate12 finding 1: this claim carries the chi-squared statistic and
            # is now keyed `chi2_augmentation_trend` (the id `cochran_armitage_trend`
            # is reserved for the genuine Cochran-Armitage z-statistic).
            claim = by_id.get("chi2_augmentation_trend")
            if model_entry is not None and claim is not None:
                want = model_entry.get("chi2")
                if isinstance(want, (int, float)) and \
                        abs((claim.get("value") or 0) - want) > 1e-4:
                    failures.append(
                        f"chi2_augmentation_trend chi2: claim={claim.get('value')} "
                        f"but {CLAIMS_MODEL} entry={want} (wrong-model selection)"
                    )
                # gate18 finding 1: the shipped statistic must be the MATCHED
                # per-task, any-sample collapse, not the unmatched raw-record
                # comparison. Recompute it independently from the replay records
                # and reject a stale unmatched value.
                indep = _matched_augmentation_chi2(new_records, CLAIMS_MODEL)
                if indep is not None and isinstance(want, (int, float)) and \
                        abs(want - indep) > 1e-3:
                    failures.append(
                        f"chi2_augmentation_trend chi2: shipped={want} but the "
                        f"independent matched-collapse recompute={indep:.4f} "
                        "(unmatched raw-record statistic must not survive)"
                    )

    # 5b. Semantic-classification deliverable: independently recompute the
    # denominator and pass rate (and Wilson interval) per target-oracle shape
    # from the replay records and compare (gate2 finding 6).
    sem_rel = manifest.get("semantic_classification_path")
    if sem_rel:
        sem_path = project_root / sem_rel if not Path(sem_rel).is_absolute() else Path(sem_rel)
        if not sem_path.exists():
            failures.append(f"semantic_classification deliverable missing: {sem_rel}")
        else:
            recorded = json.loads(sem_path.read_text(encoding="utf-8")).get("by_shape", {})
            computed = _semantic_classification(new_records, project_root / "specs")
            if set(recorded) != set(computed):
                failures.append(
                    f"semantic_classification shapes: manifest={sorted(recorded)} "
                    f"recomputed={sorted(computed)}"
                )
            for shape in sorted(set(recorded) & set(computed)):
                for key in ("n", "passes", "pass_rate", "ci_lower", "ci_upper"):
                    if recorded[shape].get(key) != computed[shape].get(key):
                        failures.append(
                            f"semantic_classification[{shape}].{key}: "
                            f"file={recorded[shape].get(key)!r} "
                            f"recomputed={computed[shape].get(key)!r}"
                        )
    else:
        failures.append("manifest records no semantic_classification_path")

    # 6. Direction-pairs expectation.
    if expected_direction_pairs is not None:
        recorded_pairs = manifest.get("expected_direction_pairs")
        if recorded_pairs != expected_direction_pairs:
            failures.append(
                f"expected_direction_pairs: manifest={recorded_pairs} "
                f"arg={expected_direction_pairs}"
            )

    # 7. Required outputs must be manifest-recorded (hashed) outputs, not just
    # any file that happens to exist. A stale top-level file that is NOT a
    # manifest output must fail, so a required deliverable cannot be satisfied
    # by a leftover from a prior run (gate1 finding 3).
    manifest_out_paths = set()
    for out in manifest.get("outputs", []):
        op = out["path"]
        manifest_out_paths.add(op)
        rp = project_root / op if not Path(op).is_absolute() else Path(op)
        manifest_out_paths.add(str(rp.resolve()))
    for req in require_outputs:
        p = req if req.is_absolute() else (project_root / req)
        if not p.exists():
            failures.append(f"--require-output missing: {req}")
            continue
        if (str(req) not in manifest_out_paths
                and str(p.resolve()) not in manifest_out_paths):
            failures.append(
                f"--require-output not a manifest output (unhashed/stale): {req}"
            )

    # 8. Optional immutable-baseline: submitted records unchanged.
    # gate8 finding 3: compare the COMPLETE expected and actual file sets, not
    # only the hashes of files that still exist. The old check skipped a baseline
    # entry whose file was gone (so a deletion passed) and never looked for files
    # on disk that the baseline does not name (so an addition passed). A synthetic
    # baseline naming a nonexistent record still returned VERIFY OK. Now a
    # deletion, an addition, and a hash change each fail.
    if immutable_baseline is not None:
        base = json.loads(immutable_baseline.read_text(encoding="utf-8"))
        # inventory stores a map under evaluation_results / files; tolerate both
        inv = base.get("evaluation_results") or base.get("files") or {}
        file_hashes = inv.get("files") if isinstance(inv, dict) and "files" in inv else inv
        if isinstance(file_hashes, dict) and file_hashes:
            def _resolve_rel(rel: str) -> Path:
                if Path(rel).is_absolute():
                    return Path(rel)
                cand = project_root / rel
                return cand if cand.exists() else (submitted_root.parent.parent / rel)

            # Expected: every baseline entry must exist and match its hash.
            for rel, sha in file_hashes.items():
                target = _resolve_rel(rel)
                if not target.exists():
                    failures.append(
                        f"immutable baseline: expected submitted file missing "
                        f"(deleted): {rel}"
                    )
                    continue
                if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                    failures.append(
                        f"immutable baseline: submitted file changed: {rel}"
                    )

            # Actual: every file under the submitted namespace must be named by
            # the baseline (else it was added after the baseline was taken).
            expected_rel = set(file_hashes)
            if submitted_root.exists():
                for p in sorted(submitted_root.rglob("*")):
                    if not p.is_file():
                        continue
                    try:
                        rel_key = p.relative_to(project_root).as_posix()
                    except ValueError:
                        rel_key = str(p)
                    if rel_key not in expected_rel and str(p) not in expected_rel:
                        failures.append(
                            f"immutable baseline: unexpected submitted file added: "
                            f"{rel_key}"
                        )

    # 9. Sealed replay namespace: recompute the namespace inventory against
    # .parbench-seal.json and the manifest (gate4 finding 1). The submitted
    # corpus gets an aggregate check (4c); the sealed replay root previously got
    # none, so a post-seal record mutation was accepted with zero failures. The
    # check fires whenever a seal exists on disk OR the manifest records a seal
    # aggregate, so it cannot be bypassed by nulling the manifest field.
    seal_file = replay_root / ".parbench-seal.json"
    recorded_seal_agg = (manifest.get("replay_root") or {}).get("seal_aggregate_sha256")
    if seal_file.exists() or recorded_seal_agg:
        if not seal_file.exists():
            failures.append(
                "replay seal missing (.parbench-seal.json) though the manifest "
                "records a seal aggregate"
            )
        else:
            seal = json.loads(seal_file.read_text(encoding="utf-8"))
            computed = _namespace_file_hashes(replay_root)
            computed_agg = _seal_aggregate(computed)
            if computed_agg != seal.get("aggregate_sha256"):
                failures.append(
                    "replay seal aggregate mismatch: the sealed namespace was "
                    "mutated after sealing"
                )
            if recorded_seal_agg is not None and recorded_seal_agg != seal.get("aggregate_sha256"):
                failures.append(
                    "manifest seal_aggregate_sha256 does not match .parbench-seal.json"
                )
            sealed_files = seal.get("files") or {}
            for rel, h in sorted(sealed_files.items()):
                if computed.get(rel) != h:
                    failures.append(f"replay record changed since sealing: {rel}")
            for rel in sorted(set(computed) - set(sealed_files)):
                failures.append(f"replay file added since sealing (not in seal): {rel}")

    # 10. Input inventories: the manifest must bind (and the verifier re-verify)
    # the OTHER inputs the analysis chain reads - manifest.jsonl (in inputs,
    # re-hashed by 4b) plus the spec JSONs and benchmark-source bytes that
    # benchmark_characterization.py and sloc_analysis.py read (gate4 finding 3).
    inv = manifest.get("input_inventories")
    if inv is None:
        failures.append(
            "manifest records no input_inventories (spec + benchmark-source bytes)"
        )
    else:
        from scripts.provenance.inventory_immutable_inputs import (  # noqa: E402
            aggregate_sha256,
            inventory_characterization_source_files,
            inventory_spec_referenced_files,
            sha256_file,
        )
        rec_specs = inv.get("specs") or {}
        spec_hashes = {p.name: sha256_file(p)
                       for p in sorted((project_root / "specs").glob("*.json"))}
        if rec_specs.get("aggregate_sha256") != aggregate_sha256(spec_hashes):
            failures.append("input_inventories.specs aggregate mismatch")
        if rec_specs.get("file_count") != len(spec_hashes):
            failures.append("input_inventories.specs file_count mismatch")
        rec_src = inv.get("benchmark_sources") or {}
        got_src = inventory_spec_referenced_files(project_root)
        if rec_src.get("aggregate_sha256") != got_src["aggregate_sha256"]:
            failures.append("input_inventories.benchmark_sources aggregate mismatch")
        if rec_src.get("file_count") != got_src["file_count"]:
            failures.append("input_inventories.benchmark_sources file_count mismatch")
        # gate5 finding 3: the manifest must bind the EXACT recursive file set
        # benchmark_characterization reads (build-tree files included), not just
        # the spec-referenced subset. Recompute and require the block.
        rec_char = inv.get("characterization_sources")
        if rec_char is None:
            failures.append(
                "input_inventories records no characterization_sources "
                "(exact recursive benchmark_characterization read set)"
            )
        else:
            got_char = inventory_characterization_source_files(project_root)
            if rec_char.get("aggregate_sha256") != got_char["aggregate_sha256"]:
                failures.append("input_inventories.characterization_sources aggregate mismatch")
            if rec_char.get("file_count") != got_char["file_count"]:
                failures.append("input_inventories.characterization_sources file_count mismatch")

    # 11. Old-vs-new comparison contents (gate4 finding 2). Section 4 only hashed
    # this file, so a corrupted old claim value + disposition passed once the
    # output hash was updated. Recompute every record-derived section, and
    # re-resolve each claim's new value (from its hashed source), old value (from
    # hashed old evidence), and disposition.
    comp_rel = manifest.get("comparison")
    if not comp_rel:
        failures.append("manifest records no comparison document (old_vs_new)")
    else:
        comp_path = _resolve(comp_rel)
        # gate5 finding 1: the comparison must itself be a hashed output (section
        # 4 re-hashes every `outputs` entry). Without this, an attacker who drops
        # the comparison's output-hash entry AND empties its sections passes
        # every guard below silently. Require the exact path in `outputs`.
        if comp_rel not in {o.get("path") for o in manifest.get("outputs", [])}:
            failures.append(
                f"comparison document {comp_rel} is not a hashed manifest output"
            )
        if not comp_path.exists():
            failures.append(f"comparison document missing: {comp_rel}")
        else:
            comp = json.loads(comp_path.read_text(encoding="utf-8"))
            # gate5 finding 1: every record-derived section is REQUIRED, not
            # optional. The prior `if "X" in comp` guards let a section be
            # removed with zero failures. Absence is now a failure, then the
            # contents are validated.
            # gate12 finding 3: semantic_classification is REQUIRED and its old
            # and new tables are independently recomputed from records - the
            # comparison presents them old-beside-new but the verifier previously
            # neither required nor recomputed them.
            for sect in ("record_level", "canonical_overall", "strata",
                         "semantic_classification", "transition_counts",
                         "strata_transitions", "semantic_transitions",
                         "claim_scoped_transitions", "headline_cell_transitions",
                         "claim_status",
                         # gate20 findings 4 & 5: the human-facing headline table,
                         # the record status movement, and the full RUN_FAIL
                         # mechanism object were neither required nor recomputed.
                         "headline", "status_movement", "run_fail_mechanism_split"):
                if sect not in comp:
                    failures.append(f"comparison missing required section: {sect}")
            # gate20 finding 4: independently recompute the headline table (the
            # canonical overall rate, its denominator, and its Wilson interval,
            # old and new) from records and compare - a forged headline value,
            # denominator, or interval must not pass just because it is present.
            if "headline" in comp:
                _hl_recompute = {}
                for _lbl, _recs in (("new", new_records), ("old", old_records)):
                    _tot = _pass = 0
                    for _r in _recs:
                        if (_r.get("source_spec") in CORRECTNESS_INELIGIBLE_SPECS
                                or _r.get("target_spec") in CORRECTNESS_INELIGIBLE_SPECS):
                            continue
                        _tot += 1
                        if _r.get("overall_status") == "PASS":
                            _pass += 1
                    _w = _wilson(_pass, _tot)
                    _hl_recompute[_lbl] = {
                        "canonical_overall_pass_rate": _w["value"],
                        "canonical_overall_denominator": _tot,
                        "canonical_overall_ci_lower": _w["ci_lower"],
                        "canonical_overall_ci_upper": _w["ci_upper"],
                    }
                _seen_hl = set()
                for _e in comp["headline"]:
                    _k = _e.get("key")
                    _seen_hl.add(_k)
                    for _side, _val in (("new", _e.get("new_value")),
                                        ("old", _e.get("old_value"))):
                        _exp = _hl_recompute[_side].get(_k)
                        if _exp is None:
                            failures.append(
                                f"comparison.headline unknown key {_k!r}")
                        elif not _num_eq(_val, _exp):
                            failures.append(
                                f"comparison.headline[{_k}].{_side}_value "
                                f"{_val!r} != recomputed {_exp!r}")
                    _od = _e.get("delta")
                    if (isinstance(_e.get("new_value"), (int, float))
                            and isinstance(_e.get("old_value"), (int, float))):
                        _exp_d = round(_e["new_value"] - _e["old_value"], 6)
                        if not _num_eq(_od, _exp_d):
                            failures.append(
                                f"comparison.headline[{_e.get('key')}].delta "
                                f"{_od!r} != recomputed {_exp_d!r}")
                _need_hl = set(_hl_recompute["new"])
                if _seen_hl != _need_hl:
                    failures.append(
                        f"comparison.headline keys {sorted(_seen_hl)} != "
                        f"required {sorted(_need_hl)}")
            # gate20 finding 4: independently recompute the record status movement
            # (per-status old/new counts and delta) from records.
            if "status_movement" in comp:
                _sc_new = _stats(new_records)["status_counts"]
                _sc_old = _stats(old_records)["status_counts"]
                _seen_sm = set()
                for _e in comp["status_movement"]:
                    _st = _e.get("status")
                    _seen_sm.add(_st)
                    _en, _eo = _sc_new.get(_st, 0), _sc_old.get(_st, 0)
                    if _e.get("new_count") != _en or _e.get("old_count") != _eo:
                        failures.append(
                            f"comparison.status_movement[{_st}] counts "
                            f"old={_e.get('old_count')!r}/new={_e.get('new_count')!r} "
                            f"!= recomputed old={_eo}/new={_en}")
                    if _e.get("delta") != _en - _eo:
                        failures.append(
                            f"comparison.status_movement[{_st}].delta "
                            f"{_e.get('delta')!r} != recomputed {_en - _eo}")
                _need_sm = {s for s in set(_sc_new) | set(_sc_old)
                            if _sc_new.get(s, 0) or _sc_old.get(s, 0)}
                if _seen_sm != _need_sm:
                    failures.append(
                        f"comparison.status_movement statuses {sorted(_seen_sm)} "
                        f"!= required {sorted(_need_sm)}")
            # gate20 finding 5: independently recompute the COMPLETE RUN_FAIL
            # mechanism object (per-model counts, exact record keys, exit codes,
            # runtimes) - not only the three aggregate bucket counts the manifest
            # headline holds.
            if "run_fail_mechanism_split" in comp:
                _rf_have = comp["run_fail_mechanism_split"]
                _rf_got = _run_fail_split_full(new_records)
                for _fld in ("total_pass_to_run_fail", "bucket_counts",
                             "by_model", "records_by_bucket"):
                    if _rf_have.get(_fld) != _rf_got[_fld]:
                        failures.append(
                            f"comparison.run_fail_mechanism_split.{_fld} "
                            f"mismatch (recomputed from sealed records)")
            if "record_level" in comp:
                for lbl, recs in (("new", new_records), ("old", old_records)):
                    if comp["record_level"].get(lbl) != _stats(recs):
                        failures.append(f"comparison.record_level.{lbl} mismatch")
            if "canonical_overall" in comp:
                for lbl, recs in (("new", new_records), ("old", old_records)):
                    if comp["canonical_overall"].get(lbl) != _canonical_overall(recs):
                        failures.append(f"comparison.canonical_overall.{lbl} mismatch")
            if "strata" in comp:
                for lbl, recs in (("new", new_records), ("old", old_records)):
                    if comp["strata"].get(lbl) != _strata(recs):
                        failures.append(f"comparison.strata.{lbl} mismatch")
            if "semantic_classification" in comp:
                # The generator writes ci_level per shape; the independent
                # recompute omits it, so compare only the recomputed fields.
                _sem_keys = ("n", "passes", "pass_rate", "ci_lower", "ci_upper")
                for lbl, recs in (("new", new_records), ("old", old_records)):
                    recorded = comp["semantic_classification"].get(lbl) or {}
                    computed = _semantic_classification(recs, project_root / "specs")
                    if set(recorded) != set(computed):
                        failures.append(
                            f"comparison.semantic_classification.{lbl} shapes: "
                            f"file={sorted(recorded)} recomputed={sorted(computed)}")
                    for shape in sorted(set(recorded) & set(computed)):
                        for key in _sem_keys:
                            if not _num_eq(recorded[shape].get(key),
                                           computed[shape].get(key)):
                                failures.append(
                                    f"comparison.semantic_classification.{lbl}"
                                    f"[{shape}].{key}: file="
                                    f"{recorded[shape].get(key)!r} recomputed="
                                    f"{computed[shape].get(key)!r}")
            if "transition_counts" in comp and comp["transition_counts"] != _transitions(new_records):
                failures.append("comparison.transition_counts mismatch")
            # gate13 finding 2: scoped per-record transitions must reproduce.
            if "strata_transitions" in comp:
                if comp["strata_transitions"] != _strata_transitions(new_records):
                    failures.append("comparison.strata_transitions mismatch")
            if "semantic_transitions" in comp:
                if comp["semantic_transitions"] != _semantic_transitions(
                        new_records, project_root / "specs"):
                    failures.append("comparison.semantic_transitions mismatch")
            # gate17 finding 2: per-cell headline transition matrices must
            # reproduce - the per-(model, direction) cross-model cells and the
            # suite-composition record/task-unit cells the pooled matrices miss.
            if "headline_cell_transitions" in comp:
                if comp["headline_cell_transitions"] != _headline_cell_transitions(
                        new_records):
                    failures.append("comparison.headline_cell_transitions mismatch")
            # gate14 finding 1: claim-scoped transitions (headline model, per
            # claim population) must reproduce, and every claim whose
            # transition_scope points at a claim-scoped cell must resolve to a
            # cell that actually exists in the table.
            cst = comp.get("claim_scoped_transitions")
            if cst is not None:
                recomputed_cst = _claim_scoped_transitions(new_records, CLAIMS_MODEL)
                if cst != recomputed_cst:
                    failures.append("comparison.claim_scoped_transitions mismatch")
                for entry in comp.get("claim_status", []):
                    scope = entry.get("transition_scope") or ""
                    if not scope.startswith("claim_scoped:"):
                        continue
                    _, _, rest = scope.partition("claim_scoped:")
                    parts = rest.split(":", 1)
                    table = parts[0]
                    # gate17 finding 3: a scope names either a whole matrix/table
                    # (flat overall matrix, or a keyed table the claim aggregates
                    # in full) or a single cell in a keyed table (by_direction_L0:
                    # <dir>, by_suite_L0:<suite>). Every record-derived claim's
                    # scope must resolve to a non-empty matrix/cell.
                    WHOLE_TABLES = ("overall_L0", "overall_L0_pass_at_1",
                                    "claims_model_all_levels",
                                    "pooled_overall_L0",
                                    "matched_augmentation_by_level",
                                    "pooled_by_direction_L0")
                    CELL_TABLES = ("by_direction_L0", "by_suite_L0")
                    if len(parts) > 1:
                        if table not in CELL_TABLES:
                            failures.append(
                                f"claim {entry.get('claim_id')} unknown "
                                f"claim-scoped transition table: {scope}")
                        elif not (cst.get(table) or {}).get(parts[1]):
                            failures.append(
                                f"claim {entry.get('claim_id')} scope {scope} "
                                f"resolves to no cell in {table}")
                    elif table in WHOLE_TABLES:
                        if not cst.get(table):
                            failures.append(
                                f"claim {entry.get('claim_id')} scope {scope} "
                                f"resolves to no records")
                    else:
                        failures.append(
                            f"claim {entry.get('claim_id')} unknown claim-scoped "
                            f"transition table: {scope}")

            oce = manifest.get("old_claim_evidence") or {}
            pc_by_id = {}
            if pc_path.exists():
                pc_by_id = {c.get("claim_id"): c
                            for c in json.loads(pc_path.read_text(encoding="utf-8")).get("claims", [])}
            # gate5 finding 1: the set of claims presented in the comparison must
            # be EXACTLY the manifest's claim-key set - a dropped claim_status
            # entry must not pass because the loop simply iterates fewer items.
            comp_claim_ids = {e.get("claim_id") for e in comp.get("claim_status", [])}
            manifest_claim_keys = set(manifest.get("claim_keys") or [])
            if comp_claim_ids != manifest_claim_keys:
                failures.append(
                    "comparison.claim_status id set != manifest claim_keys "
                    f"(missing {sorted(manifest_claim_keys - comp_claim_ids)}, "
                    f"extra {sorted(comp_claim_ids - manifest_claim_keys)})"
                )
            for entry in comp.get("claim_status", []):
                cid = entry.get("claim_id")
                claim = pc_by_id.get(cid)
                if claim is None:
                    failures.append(f"comparison claim {cid}: not in paper_claims.json")
                    continue
                # gate13 finding 2: each claim must carry a transition_scope, and
                # it must match the independent record-derived classification.
                # gate16 finding 2: it must never be 'corpus', and the per-claim
                # transitions object must reproduce independently - so a corpus
                # matrix smuggled onto a subset claim is rejected here.
                exp_scope = _claim_transition_scope(claim)
                if entry.get("transition_scope") != exp_scope:
                    failures.append(
                        f"comparison claim {cid}: transition_scope "
                        f"{entry.get('transition_scope')!r} != derived {exp_scope!r}")
                if entry.get("transition_scope") == "corpus":
                    failures.append(
                        f"comparison claim {cid}: transition_scope 'corpus' is "
                        f"forbidden (gate16 finding 2)")
                # gate17 finding 3: classify record-derived from an INDEPENDENT
                # signal (source + json_path, not the id list or the scope). The
                # emitted flag must match, and a record-derived claim must never
                # be not_applicable - the not_applicable escape is legal ONLY for
                # a claim whose value is not computed from record status.
                exp_rd = _claim_is_record_derived(claim)
                if entry.get("record_derived") != exp_rd:
                    failures.append(
                        f"comparison claim {cid}: record_derived "
                        f"{entry.get('record_derived')!r} != derived {exp_rd!r}")
                if exp_rd and (entry.get("transition_scope") == "not_applicable"
                               or (entry.get("transitions") or {}).get(
                                   "not_applicable")):
                    failures.append(
                        f"comparison claim {cid}: record-derived claim must carry "
                        f"an exact-population transition, not not_applicable "
                        f"(gate17 finding 3)")
                exp_transitions = _claim_transitions(claim)
                if entry.get("transitions") != exp_transitions:
                    failures.append(
                        f"comparison claim {cid}: transitions "
                        f"{entry.get('transitions')!r} != derived {exp_transitions!r}")
                old_v, new_v, status = (entry.get("old_value"),
                                        entry.get("new_value"), entry.get("status"))
                # New value: re-resolve from the claim's hashed source.
                nsrc = (claim.get("source_files") or [None])[0]
                njp = claim.get("json_path")
                if nsrc and njp:
                    nspath = _resolve(nsrc)
                    if nspath.exists():
                        rnew = _resolve_claim_value(
                            json.loads(nspath.read_text(encoding="utf-8")), claim)
                        if not _num_eq(rnew, new_v):
                            failures.append(
                                f"comparison claim {cid}: new_value {new_v!r} "
                                f"!= resolved {rnew!r}")
                # gate20 finding 3: re-resolving from the (hash-pinned) source JSON
                # proves only self-consistency - a self-consistent forgery of that
                # file passes. Recompute every record-derived claim's value DIRECTLY
                # from the raw records and require it to match. Two claims whose
                # value needs an analysis module the verifier may not import stay
                # source-bound (_CLAIM_VALUE_SOURCE_ONLY); every other must recompute.
                if exp_rd and cid not in _CLAIM_VALUE_SOURCE_ONLY:
                    indep = _independent_record_claim_value(claim, new_records)
                    if indep is None:
                        failures.append(
                            f"comparison claim {cid}: record-derived value has no "
                            f"independent record recomputation (gate20 finding 3)")
                    elif not (abs(float(indep) - float(new_v)) <= 1e-4
                              if isinstance(new_v, (int, float))
                              and isinstance(indep, (int, float))
                              else indep == new_v):
                        failures.append(
                            f"comparison claim {cid}: new_value {new_v!r} != "
                            f"independently recomputed {indep!r} from records "
                            f"(gate20 finding 3)")
                    # gate21 finding 3: recompute the record-derivable COMPONENTS
                    # (Wilson interval bounds, denominators, augmentation endpoints)
                    # directly from records and require each to match the claim's
                    # component - re-resolving them from the hash-pinned source JSON
                    # alone would pass a self-consistent forgery, exactly as for the
                    # scalar value above.
                    # Verify each DECLARED component the claim reports; a claim that
                    # reports no interval (e.g. best_direction, a selector) simply
                    # has none to recompute - the verifier checks what is claimed,
                    # it does not manufacture components a claim never stated.
                    rc = _independent_record_claim_components(claim, new_records)
                    for ccomp in claim.get("components") or []:
                        name = ccomp.get("name")
                        if name not in rc:
                            continue
                        claimed, recomputed = ccomp.get("value"), rc[name]
                        if not (abs(float(recomputed) - float(claimed)) <= 1e-4
                                if isinstance(claimed, (int, float))
                                and isinstance(recomputed, (int, float))
                                else recomputed == claimed):
                            failures.append(
                                f"comparison claim {cid} component {name}: claim "
                                f"{claimed!r} != independently recomputed "
                                f"{recomputed!r} from records (gate21 finding 3)")
                # gate21 finding 4: sloc_correlation is not purely record-derived
                # (per-kernel SLoC is a benchmark-source property no record carries),
                # so it cannot take the record-only path above. Recompute the
                # Spearman rho INDEPENDENTLY from the per-kernel (SLoC, pass_rate)
                # pairs in the (hash-bound) sloc_analysis.json - a dedicated
                # rank-correlation, NOT a re-resolve of the shipped scalar, so a
                # forged rho with intact pairs is caught.
                if cid == "sloc_correlation" and nsrc and _resolve(nsrc).exists():
                    sdoc = json.loads(_resolve(nsrc).read_text(encoding="utf-8"))
                    kmap = sdoc.get("kernels") or {}
                    kvals = kmap.values() if isinstance(kmap, dict) else kmap
                    pairs = [(k.get("physical_sloc"), k.get("pass_rate"))
                             for k in kvals
                             if isinstance(k.get("physical_sloc"), (int, float))
                             and isinstance(k.get("pass_rate"), (int, float))]
                    rho = _spearman_rho([p[0] for p in pairs], [p[1] for p in pairs])
                    if rho is None:
                        failures.append(
                            "comparison claim sloc_correlation: could not "
                            "independently recompute Spearman rho from the "
                            "per-kernel pairs (gate21 finding 4)")
                    elif not (isinstance(new_v, (int, float))
                              and abs(round(rho, 4) - float(new_v)) <= 1e-4):
                        failures.append(
                            f"comparison claim sloc_correlation: new_value {new_v!r} "
                            f"!= independently recomputed rho {round(rho, 4)!r} "
                            f"(gate21 finding 4)")
                # Old value + disposition: removed-class has no old counterpart.
                src = entry.get("source_file")
                if src is None or njp is None or claim.get("value") is None:
                    if status != "removed":
                        failures.append(
                            f"comparison claim {cid}: no old evidence so status "
                            f"must be 'removed', got {status!r}")
                else:
                    ev = oce.get(Path(src).name)
                    if not ev:
                        failures.append(
                            f"comparison claim {cid}: no old evidence recorded for "
                            f"{Path(src).name}")
                    else:
                        evp = _resolve(ev["path"])
                        if not evp.exists():
                            failures.append(
                                f"comparison claim {cid}: old evidence missing "
                                f"{ev['path']}")
                        elif hashlib.sha256(evp.read_bytes()).hexdigest() != ev.get("sha256"):
                            failures.append(
                                f"comparison claim {cid}: old evidence hash mismatch "
                                f"{Path(src).name}")
                        else:
                            rold = _resolve_claim_value(
                                json.loads(evp.read_text(encoding="utf-8")), claim)
                            if not _num_eq(rold, old_v):
                                failures.append(
                                    f"comparison claim {cid}: old_value {old_v!r} "
                                    f"!= resolved {rold!r}")
                            exp = _claim_status(rold, new_v)
                            if status != exp:
                                failures.append(
                                    f"comparison claim {cid}: status {status!r} "
                                    f"!= derived {exp!r}")

            # 12. Headline tables placed old-beside-new (gate5 finding 2). Each
            # root-dependent headline deliverable (cross-model comparison, suite
            # composition) is embedded old+new in the comparison and persisted as
            # hashed outputs. Require the section, require the manifest mapping,
            # require both backing files are hashed outputs, and confirm the
            # embedded documents equal the persisted hashed bytes.
            _HEADLINE_TABLE_FILES = ("cross_model_comparison.json",
                                     "suite_composition.json")
            htab = comp.get("headline_tables")
            hmap = manifest.get("headline_tables") or {}
            out_paths_all = {o.get("path") for o in manifest.get("outputs", [])}
            if htab is None:
                failures.append("comparison missing required section: headline_tables")
            else:
                # persisted[name][side] = the hashed backing document, captured so
                # the record-level recompute below reads the tamper-evident bytes,
                # not the (separately byte-checked) embedded copy.
                persisted: dict[str, dict[str, dict]] = {}
                for name in _HEADLINE_TABLE_FILES:
                    embedded = htab.get(name)
                    if not embedded or "old" not in embedded or "new" not in embedded:
                        failures.append(
                            f"comparison.headline_tables missing old/new for {name}")
                        continue
                    entry = hmap.get(name)
                    if not entry:
                        failures.append(
                            f"manifest.headline_tables has no mapping for {name}")
                        continue
                    for side, key in (("new", "new_output"), ("old", "old_evidence")):
                        rel = entry.get(key)
                        if rel not in out_paths_all:
                            failures.append(
                                f"headline_tables {name}.{key} not a hashed output: {rel}")
                            continue
                        fp = _resolve(rel)
                        if not fp.exists():
                            failures.append(f"headline_tables {name}.{key} missing: {rel}")
                            continue
                        persisted_doc = json.loads(fp.read_text(encoding="utf-8"))
                        persisted.setdefault(name, {})[side] = persisted_doc
                        if persisted_doc != embedded[side]:
                            failures.append(
                                f"headline_tables {name}.{side} embedded document "
                                f"!= persisted {rel}")

                # gate12 finding 3: byte equality between the embedded and the
                # persisted headline table proves nothing about record fidelity -
                # a wrong-but-consistent value passes. Independently recompute the
                # load-bearing headline denominators, counts, and rates from the
                # replay/submitted records and compare to the hash-bound file. The
                # structural gate keys on the persisted NEW document (a hashed
                # output, so its shape cannot be stripped undetected), which keeps
                # non-headline fixtures vacuous while enforcing on the real files.
                _recs_for = {"new": new_records, "old": old_records}
                cmc = persisted.get("cross_model_comparison.json", {})
                if "models" in (cmc.get("new") or {}):
                    for side in ("new", "old"):
                        doc = cmc.get(side) or {}
                        # Recompute the top-level table AND every pairwise entry
                        # (gate14 finding 2) from the records - no pairwise cell
                        # is left unverified.
                        failures.extend(_cross_model_headline_failures(
                            doc, _recs_for[side], side))
                sc = persisted.get("suite_composition.json", {})
                if "per_suite_record_unit" in (sc.get("new") or {}):
                    for side in ("new", "old"):
                        doc = sc.get(side) or {}
                        have = doc.get("per_suite_record_unit") or {}
                        got = _suite_record_unit(_recs_for[side])
                        if set(have) != set(got):
                            failures.append(
                                f"suite_composition {side} per_suite_record_unit "
                                f"suites: file={sorted(have)} "
                                f"recomputed={sorted(got)}")
                        for s in sorted(set(have) & set(got)):
                            for k in ("n", "passes", "rate"):
                                if not _num_eq(have[s].get(k), got[s].get(k)):
                                    failures.append(
                                        f"suite_composition {side} "
                                        f"per_suite_record_unit[{s}].{k}: "
                                        f"file={have[s].get(k)!r} "
                                        f"recomputed={got[s].get(k)!r}")
                # gate17 finding 2: recompute the suite-composition scope totals
                # and the record/task-unit rodinia-vs-others counts from records -
                # byte-checking per_suite_record_unit left scope.records,
                # scope.tasks, and the record_unit / task_unit headline cells
                # (the load-bearing rodinia-vs-others gap numbers) unverified.
                if "record_unit" in (sc.get("new") or {}):
                    for side in ("new", "old"):
                        doc = sc.get(side) or {}
                        got = _suite_composition_units(_recs_for[side])
                        for fld in ("records", "tasks"):
                            if not _num_eq((doc.get("scope") or {}).get(fld),
                                           got["scope"][fld]):
                                failures.append(
                                    f"suite_composition {side} scope.{fld}: "
                                    f"file={(doc.get('scope') or {}).get(fld)!r} "
                                    f"recomputed={got['scope'][fld]!r}")
                        for unit in ("record_unit", "task_unit"):
                            have = doc.get(unit) or {}
                            for grp in ("rodinia", "others"):
                                for k in ("n", "passes"):
                                    hv = (have.get(grp) or {}).get(k)
                                    if not _num_eq(hv, got[unit][grp][k]):
                                        failures.append(
                                            f"suite_composition {side} "
                                            f"{unit}[{grp}].{k}: file={hv!r} "
                                            f"recomputed={got[unit][grp][k]!r}")
                                # gate20 finding 4: the point estimate and the
                                # Wilson interval are deterministic functions of
                                # the (verified) n/passes; recompute and compare
                                # them so a forged rate or interval is rejected.
                                _gn = got[unit][grp]["n"]
                                _gp = got[unit][grp]["passes"]
                                _w = _wilson(_gp, _gn)
                                for _k, _exp in (("value", _w["value"]),
                                                 ("ci_lower", _w["ci_lower"]),
                                                 ("ci_upper", _w["ci_upper"])):
                                    _hv = (have.get(grp) or {}).get(_k)
                                    if not _num_eq(_hv, _exp):
                                        failures.append(
                                            f"suite_composition {side} "
                                            f"{unit}[{grp}].{_k}: file={_hv!r} "
                                            f"recomputed={_exp!r}")

    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--expected-direction-pairs", type=int, default=None)
    ap.add_argument("--require-output", type=Path, action="append", default=[])
    ap.add_argument("--immutable-baseline", type=Path, default=None)
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    project_root = args.project_root.resolve()

    failures = verify(
        manifest, project_root,
        args.expected_direction_pairs,
        args.require_output,
        args.immutable_baseline,
    )

    if failures:
        for f in failures:
            print(f"VERIFY FAIL: {f}", file=sys.stderr)
        print(f"\n{len(failures)} independent-verification failure(s)", file=sys.stderr)
        return 1

    hl = manifest.get("headline", {})
    new = hl.get("new", {})
    print(
        "VERIFY OK: independent recomputation agrees with the manifest "
        f"({new.get('total_records')} replay records, raw pass "
        f"{new.get('raw_pass_count')}, {len(manifest.get('outputs', []))} output "
        f"hashes, {len(manifest.get('claim_keys', []))} claim keys)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
