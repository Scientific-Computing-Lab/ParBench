#!/usr/bin/env python3
"""Regenerate the complete ParBench final evidence package (plan Task 13).

This orchestrator runs the existing analysis chain against the sealed replay
namespace (the "new" engineering outcome) and, for the old-to-new comparison,
against the immutable submitted namespace (the "old" outcome). It never makes a
model or provider call, never writes into a sealed namespace, and never edits a
submitted record. Both columns of the comparison come from the SAME analysis
code; only the input root differs, and favorable and unfavorable movements are
presented identically.

It produces, all under ``results/analysis/``:
  * statistical_analysis.{json,md}, quantitative_findings.{json,md} (+ per-model),
    error_taxonomy.{json,md}, token_analysis.{json,md},
    benchmark_characterization.{json,md}, sloc_analysis.{json,md},
    paper_data_<model>.json, cross_model_comparison.json,
    suite_composition.json, suite_composition_claim_status.json,
    comparator_labels.json, paper_claims.json,
    final_oracle_shape_census.json, final_direct_oracle_scan.json;
  * ``old_vs_new_comparison.json`` - the claims-checkpoint document; and
  * ``final_evidence_manifest.json`` - input/output hashes, commands, commit,
    host, contract hash, and generated claim keys.

Token COUNTS are a property of the stored translation (no model call happens in
replay, and the replay records do not carry token fields); they are recovered
read-only from each replay record's referenced submitted record. But the token
analysis also reports verdict-dependent quantities (pass counts, pass rates, and
token-vs-pass correlations), so the NEW token_analysis is computed over the
sealed replay root to use the replay verdicts (gate9 finding 1), with the token
counts backfilled from the submitted originals. Every verdict-derived quantity is
computed over the sealed replay root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness.constants import CORRECTNESS_INELIGIBLE_SPECS  # noqa: E402

# The orchestrator (unlike the independent verifier) may reuse the generators:
# resolve_claim_value keeps the old-value resolution in lock-step with the claim
# emitter, oracle_shape is the semantic classifier, wilson_ci the interval.
from scripts.analysis.generate_paper_claims import resolve_claim_value  # noqa: E402
from scripts.analysis.quantitative_findings import wilson_ci  # noqa: E402
from scripts.evaluation.replay_records import promote_parent_metadata  # noqa: E402

# Provenance primitives (not analysis modules): bind the OTHER inputs the
# analysis chain reads - manifest.jsonl, spec JSONs, benchmark-source bytes
# (gate4 finding 3).
from scripts.provenance.inventory_immutable_inputs import (  # noqa: E402
    aggregate_sha256,
    inventory_characterization_source_files,
    inventory_spec_referenced_files,
    sha256_file,
)
from scripts.rebuttal.oracle_shape_census import oracle_shape  # noqa: E402

MODELS = (
    "azure-gpt-5.3-codex",
    "azure-gpt-5.4",
    "together-qwen-3.5-397b-a17b",
)
CLAIMS_MODEL = "together-qwen-3.5-397b-a17b"
STATUSES = (
    "PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL",
    "ERROR", "EXTRACTION_FAIL", "NOT_REPLAYABLE", "SKIP",
)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Generators whose bytes define this evidence package. Their hashes are
# recorded in the manifest so provenance does not depend on a commit that may
# post-date generation (gate1 finding 5): at generation time HEAD is still the
# parent commit, which does not yet contain these scripts.
GENERATOR_SCRIPTS = (
    "scripts/analysis/generate_final_evidence.py",
    "scripts/analysis/verify_final_evidence.py",
    "scripts/analysis/generate_paper_claims.py",
    "scripts/analysis/sloc_analysis.py",
    "scripts/rebuttal/oracle_shape_census.py",
    "scripts/spec_tools/direct_oracle_scan.py",
    "scripts/analysis/compare_oracle_censuses.py",
    # The characterization/spec-source inventory helper supplies the gate-5
    # input closure at build_final_evidence; its code must be hash-bound too, or
    # the manifest does not bind the code that produced the evidence (gate6
    # finding 3, 2026-08-13).
    "scripts/provenance/inventory_immutable_inputs.py",
    # gate7 finding 5: every producer the orchestrator EXECUTES must be hash-bound,
    # not just the census/claims chain. quantitative_findings.py in particular
    # changed after the commit the manifest records, so binding only its commit
    # left the code that built quantitative_findings.json unrecorded. These run
    # via Chain.run in main(); executed_producer_scripts() also binds them
    # dynamically, but listing them here keeps REQUIRED_GENERATORS enforcing them.
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
    # gate8 finding 5: producers IMPORTED (not invoked as argv) that materially
    # shape the evidence - promoted-parent metadata, direction aggregation, and
    # oracle strengths. executed_producer_scripts() cannot see them because they
    # never appear as a python entry-point token, so bind them explicitly or the
    # manifest omits the code that produced the promoted records.
    "scripts/evaluation/replay_records.py",
    "scripts/analysis/paired_tasks.py",
    "scripts/spec_tools/audit_oracle_contracts.py",
)


def executed_producer_scripts(log: list[dict]) -> set[str]:
    """Repo-relative producer scripts actually executed in a Chain command log.

    gate7 finding 5: the manifest must bind the code it RAN. Each Chain entry
    records the full argv; a producer is the first ``scripts/...py`` token of a
    python invocation. Used to bind the executed set in the manifest even if a new
    producer is added to main() without updating GENERATOR_SCRIPTS.
    """
    scripts: set[str] = set()
    for entry in log:
        argv = entry.get("command") or []
        if len(argv) < 2:
            continue
        if Path(argv[0]).name not in ("python3", "python"):
            continue
        for tok in argv[1:]:
            if tok.startswith("scripts/") and tok.endswith(".py"):
                scripts.add(tok)
                break
    return scripts


def _aggregate_sha256(root: Path) -> dict:
    """Aggregate hash over every record JSON under ``root`` (order-independent).

    Hashes ``sorted("relpath\\0filehash")`` so a change to any submitted record
    (or a missing/extra one) moves the aggregate. Used to bind the immutable
    submitted corpus into the manifest (gate1 finding 5).
    """
    lines = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.json")):
            if path.name.startswith("batch_") or path.name == "eval_summary.json":
                continue
            rel = path.relative_to(root).as_posix()
            lines.append(f"{rel}\0{_sha256(path)}")
    agg = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {"aggregate_sha256": agg, "file_count": len(lines)}


def _git_commit(project_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _host() -> str:
    import socket
    return socket.gethostname()


def _within(child: Path, ancestor: Path) -> bool:
    """True when ``child`` is ``ancestor`` or lives beneath it (both resolved)."""
    try:
        child.relative_to(ancestor)
        return True
    except ValueError:
        return False


def _sealed_ancestor(path: Path) -> Path | None:
    """The nearest ancestor of ``path`` (or ``path`` itself) that is a sealed
    namespace - a directory holding ``.parbench-seal.json`` - or None. Walks up so
    a sealed root catches an output nested any depth below it (gate19 finding 1)."""
    for cand in [path, *path.parents]:
        if (cand / ".parbench-seal.json").exists():
            return cand
    return None


# gate21 finding 1: an output must also stay out of the append-only manifest,
# the frozen benchmark source trees, and the spec/config surfaces. These carry no
# .parbench-seal.json, so the sealed/corpus machinery above never engages for
# them; a direct `--output manifest.jsonl` or `--comparison-out rodinia/.../bfs.cu`
# would overwrite an append-only or benchmark input. Names are project-root
# relative (CLAUDE.md: manifest.jsonl append-only; rodinia/ HeCBench-master/ ...
# frozen benchmark sources; specs/ + config/ hold the frozen spec + contract).
_PROTECTED_OUTPUT_SUBTREES = (
    "rodinia", "HeCBench-master", "hecbench", "xsbench", "xsbench-src",
    "rsbench", "rsbench-src", "mixbench", "mixbench-src", "specs", "config",
)
_PROTECTED_OUTPUT_FILES = ("manifest.jsonl",)


def _unsafe_output_reasons(labeled_paths, submitted_root: Path,
                           replay_root: Path,
                           corpus_root: Path | None = None,
                           project_root: Path | None = None,
                           input_paths=None) -> list[str]:
    """Reject any output path that resolves inside the immutable submitted corpus,
    inside the sealed replay root, or inside any sealed namespace (gate19 finding
    1). The Bash hook does not protect a direct CLI run, and this generator
    rmtree's <final-dir>/old_evidence, so an output dir inside an immutable/sealed
    tree must be refused BEFORE any directory is created.

    gate20 finding 2: ``submitted_root`` is caller-supplied, so a decoy
    ``--submitted-root`` under /tmp used to leave the REAL corpus unprotected.
    ``corpus_root`` (the canonical <project-root>/results/evaluation) is ALWAYS
    protected, independent of ``--submitted-root``.

    gate21 finding 1: when ``project_root`` is supplied, also reject an output
    inside the append-only manifest, the frozen benchmark source trees, or the
    spec/config surfaces (``_PROTECTED_OUTPUT_SUBTREES`` / ``_PROTECTED_OUTPUT_
    FILES``); and reject any output that aliases (equals or is nested under) a
    declared input in ``input_paths`` - output/input aliasing would clobber an
    input the manifest simultaneously hashes."""
    reasons: list[str] = []
    immutable_roots = [submitted_root]
    if corpus_root is not None and not any(_within(corpus_root, r) or _within(r, corpus_root)
                                           for r in immutable_roots):
        immutable_roots.append(corpus_root)
    protected_subtrees = []
    protected_files = []
    if project_root is not None:
        protected_subtrees = [(project_root / s).resolve()
                              for s in _PROTECTED_OUTPUT_SUBTREES]
        protected_files = [(project_root / f).resolve()
                           for f in _PROTECTED_OUTPUT_FILES]
    inputs = [Path(p).resolve() for p in (input_paths or [])]
    for label, p in labeled_paths:
        p = Path(p).resolve()
        hit = next((r for r in immutable_roots if _within(p, r)), None)
        if hit is not None:
            reasons.append(
                f"{label}={p} is inside the immutable submitted corpus {hit}")
        elif _within(p, replay_root):
            reasons.append(
                f"{label}={p} is inside the sealed replay root {replay_root}")
        elif any(p == f for f in protected_files):
            reasons.append(
                f"{label}={p} is an append-only input (never an output target)")
        elif next((s for s in protected_subtrees if _within(p, s)), None) is not None:
            sub = next(s for s in protected_subtrees if _within(p, s))
            reasons.append(
                f"{label}={p} is inside a frozen benchmark/spec/config surface ({sub})")
        elif next((i for i in inputs if _within(p, i)), None) is not None:
            alias = next(i for i in inputs if _within(p, i))
            reasons.append(
                f"{label}={p} aliases a declared input ({alias}); "
                f"an output must not overwrite an input")
        else:
            sealed = _sealed_ancestor(p)
            if sealed is not None:
                reasons.append(
                    f"{label}={p} is inside a sealed namespace ({sealed})")
    return reasons


def load_records(root: Path) -> list[dict]:
    """Load every result record under ``root`` with parent identity promoted.

    Skips the replay companions (replay_manifest.json, replay_transitions.json)
    and the seal marker, which live at the namespace root, and any
    eval_summary/batch bookkeeping files inside model dirs.
    """
    records: list[dict] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.json")):
            if path.name.startswith("batch_") or path.name == "eval_summary.json":
                continue
            data = promote_parent_metadata(json.loads(path.read_text(encoding="utf-8")))
            data["_stem"] = path.stem
            data["_model"] = model_dir.name
            records.append(data)
    return records


def contract_eligibility_failures(records: list[dict], contract_path: Path) -> list[str]:
    """Bind correctness eligibility to the FROZEN contract (gate6 finding 4).

    (1) the contract's exclusion set must equal the canonical constant, and
    (2) every correctness-eligible record pair must appear in the frozen
    eligible_task_list. Mirrors verify_final_evidence._contract_eligibility_
    failures (the two paths are deliberately independent); both must reject a
    spec-eligible pair that is absent from the frozen registry.
    """
    failures: list[str] = []
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    excl = set((contract.get("exclusions") or {}).get("correctness_ineligible_specs") or [])
    if excl != set(CORRECTNESS_INELIGIBLE_SPECS):
        failures.append(
            "contract correctness_ineligible_specs != canonical "
            f"CORRECTNESS_INELIGIBLE_SPECS (contract {len(excl)}, "
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


def _eligible(rec: dict) -> bool:
    # Correctness eligibility, not merely KNOWN_FAIL exclusion (gate3 finding 1):
    # performance_only specs (mixbench) carry no correctness oracle, so they can
    # never enter a pass-rate denominator. CORRECTNESS_INELIGIBLE_SPECS is the
    # union the canonical analysis (quantitative_findings) already uses.
    return (
        rec.get("source_spec") not in CORRECTNESS_INELIGIBLE_SPECS
        and rec.get("target_spec") not in CORRECTNESS_INELIGIBLE_SPECS
    )


def record_stats(records: list[dict]) -> dict:
    """Unambiguous, trivially reproducible record-level headline counts/rates.

    These are what verify_final_evidence recomputes independently. Raw = over
    every record; the eligibility-adjusted, L0-collapsed rates live in the
    analysis outputs consumed by the old-vs-new comparison.
    """
    status_counts: dict[str, int] = {s: 0 for s in STATUSES}
    per_model: dict[str, dict[str, int]] = {}
    eligible_pass = 0
    eligible_total = 0
    for rec in records:
        status = rec.get("overall_status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
        model = rec.get("model") or rec.get("_model") or "unknown"
        pm = per_model.setdefault(model, {})
        pm[status] = pm.get(status, 0) + 1
        if _eligible(rec):
            eligible_total += 1
            if status == "PASS":
                eligible_pass += 1
    total = len(records)
    passes = status_counts.get("PASS", 0)
    return {
        "total_records": total,
        "status_counts": {k: v for k, v in status_counts.items() if v},
        "per_model_status_counts": {
            m: {k: v for k, v in sc.items()} for m, sc in sorted(per_model.items())
        },
        "raw_pass_count": passes,
        "raw_pass_rate": round(passes / total, 6) if total else 0.0,
        "eligible_records": eligible_total,
        "eligible_pass_count": eligible_pass,
        "eligible_pass_rate": round(eligible_pass / eligible_total, 6)
        if eligible_total else 0.0,
    }


def transition_counts(replay_records: list[dict]) -> dict:
    """Old->new status transition counts recomputed from the replay records."""
    counts: dict[str, int] = {}
    for rec in replay_records:
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        counts[f"{old}->{new}"] = counts.get(f"{old}->{new}", 0) + 1
    return dict(sorted(counts.items()))


def strata_transitions(replay_records: list[dict]) -> dict:
    """Per-record old->new transition counts scoped to each strata cell.

    The plan requires the old-vs-new comparison to carry the per-record
    transition counts BEHIND EACH CHANGE, not one corpus-wide matrix (gate13
    finding 2). This attaches a scoped transition matrix to each record-derived
    headline cell, using the SAME record sets as ``strata_pass_rates``:
    per-model and per-suite span every augmentation level; per-direction is L0
    only; all three are restricted to correctness-eligible records. The verifier
    recomputes these independently.
    """
    by_model: dict[str, dict[str, int]] = {}
    by_suite: dict[str, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}

    def _bump(agg: dict, key: str, transition: str) -> None:
        cell = agg.setdefault(key, {})
        cell[transition] = cell.get(transition, 0) + 1

    for rec in replay_records:
        if not _eligible(rec):
            continue
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        transition = f"{old}->{new}"
        _bump(by_model, rec.get("model") or rec.get("_model") or "unknown", transition)
        _bump(by_suite, _spec_suite(rec.get("source_spec")), transition)
        if rec.get("augment_level") == 0:
            direction = (f"{_spec_api(rec.get('source_spec'))}-to-"
                         f"{_spec_api(rec.get('target_spec'))}")
            _bump(by_direction, direction, transition)

    def _sort(agg: dict) -> dict:
        return {k: dict(sorted(agg[k].items())) for k in sorted(agg)}

    return {
        "by_model": _sort(by_model),
        "by_suite": _sort(by_suite),
        "by_direction": _sort(by_direction),
    }


def semantic_transitions(replay_records: list[dict], specs_dir: Path) -> dict:
    """Per-record old->new transition counts scoped to each target-oracle shape.

    Same record set as ``semantic_classification`` (correctness-eligible L0
    records grouped by the target spec's oracle shape); the per-record moves
    behind each semantic-classification cell's change (gate13 finding 2)."""
    agg: dict[str, dict[str, int]] = {}
    for rec in replay_records:
        if rec.get("augment_level") != 0:
            continue
        if not _eligible(rec):
            continue
        shape = oracle_shape(rec.get("target_spec"), specs_dir)
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        cell = agg.setdefault(shape, {})
        cell[f"{old}->{new}"] = cell.get(f"{old}->{new}", 0) + 1
    return {s: dict(sorted(agg[s].items())) for s in sorted(agg)}


# Analysis-output basenames whose claim values are a function of record STATUS
# (pass rates, status distributions, chi-squared over statuses). A claim
# resolving from one of these is record-derived, so its change is explained by
# the per-record transition counts; every other claim source (token counts,
# SLoC, spec-derived characterization) is transition-irrelevant (gate13 finding
# 2). Kept in sync with the verifier's independent copy by a source test.
RECORD_DERIVED_CLAIM_SOURCES = frozenset({
    "statistical_analysis.json",
    "cross_model_comparison.json",
    "suite_composition.json",
    "error_taxonomy.json",
})

# gate17 finding 3: the claim ids whose VALUE is a function of result-record
# status. Each MUST carry an exact-population transition matrix (never
# not_applicable). Enumerated explicitly so the classification is auditable and
# the verifier can mirror it from an independent signal. The per-direction /
# per-suite pass-rate claims match by prefix, handled in
# ``claim_is_record_derived``; every other record-derived claim is listed here.
# Claims deliberately ABSENT (value not a function of record status, so
# not_applicable is legal): spec_counts, total_specs_count, suite_composition,
# total_suites_count, multi_file_fraction (spec-derived); token_cost_total
# (token-cost); total_eval_files, valid_eval_files_count (file/spec counts).
RECORD_DERIVED_CLAIM_IDS = frozenset({
    "overall_pass_at_1", "overall_pass_at_3",
    "aggregate_pass_rate",
    "best_direction", "worst_direction",
    "build_fail_count", "run_fail_count", "verify_fail_count",
    "build_fail_subcategories",
    "augmentation_degradation", "chi2_augmentation_trend",
    "direction_asymmetry", "sloc_correlation",
})


# --------------------------------------------------------------------------- #
# gate18 finding 3: claim-scoped transition tables whose ROWS are the claim's   #
# statistical UNITS (task cells, matched cells, paired cells, kernel groups),    #
# never raw records, each carrying the collapse mapping unit -> record keys so   #
# an external reader can rebuild it.                                             #
# --------------------------------------------------------------------------- #

def _record_key(rec: dict) -> str:
    """Stable per-record identity for a collapse mapping (gate18 finding 3):
    (model, source, target, level, sample). Unique across the corpus."""
    return "|".join([
        str(rec.get("model") or rec.get("_model") or "unknown"),
        str(rec.get("source_spec") or ""),
        str(rec.get("target_spec") or ""),
        f"L{rec.get('augment_level', 0) or 0}",
        f"s{rec.get('sample_id', 0)}",
    ])


def _collapse_status(statuses: list[str]) -> str:
    return "PASS" if any(s == "PASS" for s in statuses) else "FAIL"


def _transition_matrix(rows: list[dict]) -> dict:
    m: dict[str, int] = {}
    for row in rows:
        tr = f"{row['old']}->{row['new']}"
        m[tr] = m.get(tr, 0) + 1
    return dict(sorted(m.items()))


def _unit_table(unit: str, rows: list[dict], matrix: dict) -> dict:
    return {"unit": unit, "rows": rows, "matrix": matrix}


def _task_cell_rows(records: list[dict]) -> list[dict]:
    """One row per (source, target) task cell - the any-sample collapse pass@k and
    the per-direction/per-suite pass rates use - with old=parent, new=replay, and
    the record keys the cell collapses (gate18 finding 3)."""
    new_by: dict[tuple, list[str]] = {}
    old_by: dict[tuple, list[str]] = {}
    keys_by: dict[tuple, list[str]] = {}
    for r in records:
        k = (r.get("source_spec"), r.get("target_spec"))
        new_by.setdefault(k, []).append(r.get("overall_status", ""))
        old_by.setdefault(k, []).append((r.get("parent") or {}).get("overall_status", ""))
        keys_by.setdefault(k, []).append(_record_key(r))
    rows = []
    for (s, t) in sorted(new_by):
        rows.append({
            "source_spec": s, "target_spec": t,
            "direction": f"{_spec_api(s)}-to-{_spec_api(t)}",
            "suite": _spec_suite(s),
            "old": _collapse_status(old_by[(s, t)]),
            "new": _collapse_status(new_by[(s, t)]),
            "records": sorted(keys_by[(s, t)]),
        })
    return rows


def _task_cell_table(records: list[dict]) -> dict:
    rows = _task_cell_rows(records)
    return _unit_table("task_cell", rows, _transition_matrix(rows))


def _raw_record_rows(records: list[dict]) -> list[dict]:
    """One row per RAW record (gate19 finding 2) - the unit
    compute_direction_pass_rates uses (wilson_ci over PASS count / total L0
    records). old/new are the record's VERBATIM parent/replay overall_status
    (not the any-sample collapse), each row carrying its own record key. Sorted
    by record key so the table is deterministic."""
    rows = []
    for r in records:
        rows.append({
            "source_spec": r.get("source_spec"),
            "target_spec": r.get("target_spec"),
            "direction": f"{_spec_api(r.get('source_spec'))}-to-"
                         f"{_spec_api(r.get('target_spec'))}",
            "suite": _spec_suite(r.get("source_spec")),
            "old": (r.get("parent") or {}).get("overall_status", ""),
            "new": r.get("overall_status", ""),
            "record": _record_key(r),
        })
    return sorted(rows, key=lambda x: x["record"])


def _raw_record_table(records: list[dict]) -> dict:
    rows = _raw_record_rows(records)
    return _unit_table("raw_record", rows, _transition_matrix(rows))


def _task_cn_rows(records: list[dict]) -> list[dict]:
    """One row per (source, target) task cell carrying the pass-count/sample-count
    on BOTH sides (gate19 finding 2) - the unit pass@1 and per-suite pass@1 use
    (macro-average of c/n per task). Distinct from the any-sample collapse: a task
    with new samples PASS,FAIL,FAIL is c/n = 1/3 here but PASS (any-pass) in the
    task-cell table. Each row carries the record keys it aggregates."""
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
        keys_by.setdefault(k, []).append(_record_key(r))
    rows = []
    for (s, t) in sorted(n):
        rows.append({
            "source_spec": s, "target_spec": t,
            "direction": f"{_spec_api(s)}-to-{_spec_api(t)}",
            "suite": _spec_suite(s),
            "old_pass": old_pass.get((s, t), 0), "old_n": n[(s, t)],
            "new_pass": new_pass.get((s, t), 0), "new_n": n[(s, t)],
            "records": sorted(keys_by[(s, t)]),
        })
    return rows


def _task_cn_table(records: list[dict]) -> dict:
    rows = _task_cn_rows(records)
    matrix = {
        "old_pass": sum(r["old_pass"] for r in rows),
        "old_n": sum(r["old_n"] for r in rows),
        "new_pass": sum(r["new_pass"] for r in rows),
        "new_n": sum(r["new_n"] for r in rows),
        # gate20 finding 6: the marginal old/new pass totals hide the per-record
        # old->new movement behind a pass@1 / per-suite pass@1 change. Emit the
        # per-RAW-record old->new transition counts so the movement is auditable
        # and the verifier can recompute it (it reproduces this table exactly).
        "transitions": _transition_matrix(_raw_record_rows(records)),
    }
    return _unit_table("task_cn", rows, matrix)


def claim_scoped_transitions(replay_records: list[dict], claims_model: str) -> dict:
    """Old->new transitions over EXACTLY each paper claim's record set,
    restricted to the headline (claims) model (gate14 finding 1).

    strata_transitions.by_direction pools every model (L0, all models) and
    by_model pools every direction/level, so neither matches a per-model-
    per-direction claim's records (e.g. Qwen cuda-to-omp L0). This emits the
    transitions behind the claims-model record sets the manuscript reports:
      overall_L0       - claims_model, L0 (records of overall_pass_at_1)
      by_direction_L0  - claims_model, L0, grouped by direction (records of
                         direction_pass_rates.standard.<d>)
      by_suite_L0      - claims_model, L0, grouped by source suite (records of
                         the per-suite pass@1 claims; pass@1 macro-averages c/n
                         per task over exactly these records)
    Mirrors compute_direction_pass_rates / compute_pass_at_k over the claim
    population (L0, with KNOWN_FAIL AND performance-only specs excluded via the
    same correctness-eligibility filter the claims apply upstream at
    quantitative_findings.exclude_known_fail - drop any record whose source or
    target spec is in CORRECTNESS_INELIGIBLE_SPECS; gate15 finding 2). Direction
    = source_api-to-target_api and suite = source-spec prefix (the replay records
    carry no top-level ``direction`` field, so quantitative_findings derives it
    the same way). Without the eligibility filter the scope would report
    ineligible records that are NOT in the claim's denominator (e.g. Qwen L0
    overall 504 records vs the claim's 408). The verifier recomputes this
    independently."""
    # gate19 finding 2: pass@1, pass@3, the per-direction rates, and per-suite
    # pass@1 are DIFFERENT statistical units - one any-sample collapse cannot back
    # all four. pass@3 = fraction of tasks with any pass (any-sample TASK CELL);
    # pass@1 and per-suite pass@1 = macro-average of c/n per task (per-task c/n);
    # the per-direction rate = wilson over RAW L0 records (raw record). Build one
    # eligible headline-L0 record set, then emit each claim family in its own unit.
    hl = [r for r in replay_records
          if (r.get("model") or r.get("_model")) == claims_model
          and r.get("augment_level") == 0
          and r.get("source_spec") not in CORRECTNESS_INELIGIBLE_SPECS
          and r.get("target_spec") not in CORRECTNESS_INELIGIBLE_SPECS]

    def _partition_records(field: str, table_fn) -> dict:
        keys = sorted({
            (f"{_spec_api(r.get('source_spec'))}-to-"
             f"{_spec_api(r.get('target_spec'))}" if field == "direction"
             else _spec_suite(r.get("source_spec")))
            for r in hl})
        out = {}
        for key in keys:
            if field == "direction":
                sub = [r for r in hl
                       if f"{_spec_api(r.get('source_spec'))}-to-"
                          f"{_spec_api(r.get('target_spec'))}" == key]
            else:
                sub = [r for r in hl if _spec_suite(r.get("source_spec")) == key]
            out[key] = table_fn(sub)
        return out

    return {
        "model": claims_model,
        # pass@3 - any-sample task cells
        "overall_L0": _task_cell_table(hl),
        # pass@1 - per-task c/n (gate19 finding 2)
        "overall_L0_pass_at_1": _task_cn_table(hl),
        # per-direction pass rate - raw records (gate19 finding 2)
        "by_direction_L0": _partition_records("direction", _raw_record_table),
        # per-suite pass@1 - per-task c/n (gate19 finding 2)
        "by_suite_L0": _partition_records("suite", _task_cn_table),
        # gate17 finding 3: the record-derived claims whose population is NOT the
        # headline-model L0 task set still need their OWN exact-population
        # transition matrix - not_applicable is reserved for claims not computed
        # from records at all. Each table below is the exact denominator behind a
        # family of record-derived claims, recomputed independently by the
        # verifier.
        #  * claims_model_all_levels - the canonical (all-augment-level,
        #    correctness-eligible) headline-model population behind
        #    aggregate_pass_rate and the failure-status counts (n=604 in the
        #    final corpus). Overall old->new matrix.
        #  * matched_augmentation_by_level (gate18 finding 3) - rows are MATCHED
        #    TASK CELLS in the CUDA->OMP population (the augmentation sweep's own
        #    population, not all directions), grouped by augment level, each cell
        #    carrying its record keys. Behind chi2_augmentation_trend / degradation.
        #  * pooled_by_direction_L0 (gate18 finding 3) - rows are PAIRED CELLS
        #    ((model, suite, kernel) present in both directions of a pair), the unit
        #    the McNemar direction-asymmetry test pairs, each carrying both
        #    directions' record keys.
        #  * pooled_overall_L0 (gate18 finding 3) - rows are KERNEL GROUPS (the
        #    correctness-eligible, all-level, all-model records grouped by kernel -
        #    30 groups / 2160 records), the unit the SLoC-vs-pass correlation uses,
        #    each carrying its record keys.
        "claims_model_all_levels": _claims_model_all_levels_transitions(
            replay_records, claims_model),
        "matched_augmentation_by_level": _matched_augmentation_transitions(
            replay_records, claims_model),
        "pooled_by_direction_L0": _pooled_paired_cells(replay_records),
        "pooled_overall_L0": _kernel_group_table(replay_records),
    }


def _claims_model_all_levels_transitions(replay_records: list[dict],
                                         claims_model: str) -> dict:
    """Overall old->new matrix over the canonical headline-model population:
    correctness-eligible records for ``claims_model`` at EVERY augment level
    (the aggregate_pass_rates / failure_taxonomy denominator). Independent of
    the L0-only overall_L0 table above (gate17 finding 3)."""
    overall: dict[str, int] = {}
    for rec in replay_records:
        if (rec.get("model") or rec.get("_model")) != claims_model:
            continue
        if not _eligible(rec):
            continue
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        overall[f"{old}->{new}"] = overall.get(f"{old}->{new}", 0) + 1
    return dict(sorted(overall.items()))


def _matched_augmentation_transitions(replay_records: list[dict],
                                      claims_model: str) -> dict:
    """Matched augmentation cells for ``claims_model`` in the CUDA->OMP population
    (gate18 finding 3). Restrict to cuda-to-omp (the augmentation sweep's own
    direction), keep tasks carrying at least one augmented (L>=1) record, collapse
    each (task, level) to one cell by any-sample success on BOTH replay (new) and
    parent (old) status. ROWS are the matched TASK CELLS, grouped by level, each
    carrying its record keys - the exact population behind augmentation_degradation
    and chi2_augmentation_trend."""
    new_by: dict[tuple, list[str]] = {}
    old_by: dict[tuple, list[str]] = {}
    keys_by: dict[tuple, list[str]] = {}
    for rec in replay_records:
        if (rec.get("model") or rec.get("_model")) != claims_model:
            continue
        s, t = rec.get("source_spec"), rec.get("target_spec")
        if f"{_spec_api(s)}-to-{_spec_api(t)}" != "cuda-to-omp":
            continue
        lv = rec.get("augment_level", 0) or 0
        new_by.setdefault((s, t, lv), []).append(rec.get("overall_status", ""))
        old_by.setdefault((s, t, lv), []).append(
            (rec.get("parent") or {}).get("overall_status", ""))
        keys_by.setdefault((s, t, lv), []).append(_record_key(rec))
    matched = {(s, t) for (s, t, lv) in new_by if lv >= 1}
    by_level: dict[int, list[dict]] = {}
    for (s, t, lv) in sorted(new_by):
        if (s, t) not in matched:
            continue
        by_level.setdefault(lv, []).append({
            "source_spec": s, "target_spec": t,
            "old": _collapse_status(old_by[(s, t, lv)]),
            "new": _collapse_status(new_by[(s, t, lv)]),
            "records": sorted(keys_by[(s, t, lv)]),
        })
    return {
        "unit": "matched_task_cell",
        "direction": "cuda-to-omp",
        "by_level": {
            f"L{lv}": _unit_table("matched_task_cell", by_level[lv],
                                  _transition_matrix(by_level[lv]))
            for lv in sorted(by_level)
        },
    }


def _pooled_paired_cells(replay_records: list[dict]) -> dict:
    """Direction-paired cells over the ALL-model, L0 population (gate18 finding 3).
    Collapse each (model, suite, kernel, direction) cell by any-sample success,
    then pair the two directions of a pair within the same (model, suite, kernel).
    ROWS are the PAIRED CELLS the McNemar asymmetry test uses, keyed by the
    canonical 'fwd vs rev' pair, each carrying both directions' record keys.

    gate19 finding 3: apply correctness eligibility (drop KNOWN_FAIL and the
    performance-only mixbench specs the McNemar denominator excludes), and carry
    each cell's OLD (submitted/parent) outcome beside its NEW (replay) outcome so
    the pair table emits an old->new transition, not just a replay snapshot."""
    collapsed: dict[tuple, dict] = {}
    for rec in replay_records:
        if rec.get("augment_level") != 0:
            continue
        if not _eligible(rec):
            continue
        s = rec.get("source_spec")
        key = (rec.get("model") or rec.get("_model") or "unknown",
               _spec_suite(s), _spec_kernel(s),
               f"{_spec_api(s)}-to-{_spec_api(rec.get('target_spec'))}")
        cell = collapsed.setdefault(
            key, {"pass": False, "old_pass": False, "records": []})
        cell["pass"] = cell["pass"] or (rec.get("overall_status") == "PASS")
        cell["old_pass"] = cell["old_pass"] or (
            (rec.get("parent") or {}).get("overall_status") == "PASS")
        cell["records"].append(_record_key(rec))
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
            "unit": "direction_pair_cell",
            "rows": rows,
            "matrix": _pair_matrix(rows, "forward_pass", "reverse_pass"),
            "old_matrix": _pair_matrix(rows, "forward_old_pass", "reverse_old_pass"),
            "transition": _pair_transition(rows),
        }
    return pairs


def _pair_state(fwd: bool, rev: bool) -> str:
    return f"{'PASS' if fwd else 'FAIL'}/{'PASS' if rev else 'FAIL'}"


def _pair_matrix(rows: list[dict], fwd_key: str, rev_key: str) -> dict:
    m: dict[str, int] = {}
    for r in rows:
        k = _pair_state(r[fwd_key], r[rev_key])
        m[k] = m.get(k, 0) + 1
    return dict(sorted(m.items()))


def _pair_transition(rows: list[dict]) -> dict:
    """Old pair state -> new pair state counts (gate19 finding 3)."""
    m: dict[str, int] = {}
    for r in rows:
        old = _pair_state(r["forward_old_pass"], r["reverse_old_pass"])
        new = _pair_state(r["forward_pass"], r["reverse_pass"])
        k = f"{old}->{new}"
        m[k] = m.get(k, 0) + 1
    return dict(sorted(m.items()))


def _kernel_group_table(replay_records: list[dict]) -> dict:
    """Kernel groups over the correctness-eligible, all-level, all-model population
    (gate18 finding 3) - the unit the SLoC-vs-pass correlation uses (30 groups /
    2160 records in the final corpus). ROWS are the kernel groups, each carrying
    its record keys and pass/fail counts."""
    groups: dict[str, dict] = {}
    transition: dict[str, int] = {}
    for rec in replay_records:
        if not _eligible(rec):
            continue
        k = _spec_kernel(rec.get("source_spec"))
        g = groups.setdefault(
            k, {"records": [], "pass": 0, "fail": 0, "old_pass": 0, "old_fail": 0})
        g["records"].append(_record_key(rec))
        new = rec.get("overall_status")
        old = (rec.get("parent") or {}).get("overall_status")
        if new == "PASS":
            g["pass"] += 1
        else:
            g["fail"] += 1
        # gate19 finding 3: carry the OLD (submitted) outcome and an old->new
        # per-record transition, not only the replay pass/fail count.
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


def claim_transition_scope(claim: dict) -> str:
    """The exact record population behind a claim's status change, or
    'not_applicable'. The corpus-wide fallback is REMOVED (gate16 finding 2).

    gate17 finding 3: not_applicable is legal ONLY for a claim whose value is
    NOT computed from result-record status (a spec count, a file count, a token
    cost). EVERY record-derived claim carries an exact-population transition
    matrix - never not_applicable, never the full-corpus matrix. Each claim
    points at its OWN cell in ``claim_scoped_transitions``, recomputed
    independently by the verifier:
      * overall pass@1/@3 and best/worst direction -> the headline-model L0 task
        population (overall_L0 / the named by_direction_L0 cell);
      * per-direction / per-suite pass rates -> their by_direction_L0 /
        by_suite_L0 cell;
      * aggregate_pass_rate and the failure-status counts -> the canonical
        all-level headline-model population (claims_model_all_levels);
      * augmentation effects -> the matched-augmentation-by-level population;
      * pooled cross-model statistics (direction asymmetry, SLoC correlation) ->
        the ALL-model L0 pooled population."""
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
    # are different units - each points at its own table.
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


def claim_is_record_derived(claim: dict) -> bool:
    """True when the claim's value is a function of result-record status, so a
    not_applicable transition is FORBIDDEN (gate17 finding 3). Derived from the
    claim's own id/source/json_path - the same signal the verifier classifies on
    independently - not from the scope, so a mislabeled scope is caught."""
    cid = claim.get("claim_id") or ""
    if cid in RECORD_DERIVED_CLAIM_IDS:
        return True
    if cid.startswith("direction_pass_") or cid.startswith("suite_pass_"):
        return True
    return False


def claim_transition_reason(claim: dict) -> str:
    """One-line reason a 'not_applicable' claim carries no exact-population
    transition matrix (gate17 finding 3). Only non-record claims reach here:
    spec counts, file counts, and token cost carry no old->new record status."""
    src = (claim.get("source_files") or [None])[0]
    if not src:
        return "no source artifact; not a record-status claim"
    return ("value is a spec-, file-count-, or token-cost quantity, not a "
            "function of result-record status; it carries no old->new status "
            "transition")


def claim_transitions(claim: dict) -> dict:
    """The per-claim transition evidence object (gate17 finding 3). A
    record-derived claim points at its exact-population cell in
    ``claim_scoped_transitions`` (the verifier recomputes that cell from
    records); a non-record claim is explicitly
    ``{"not_applicable": true, "reason": ...}`` - never the corpus matrix."""
    scope = claim_transition_scope(claim)
    if scope.startswith("claim_scoped:"):
        return {"scope": scope}
    return {"not_applicable": True, "reason": claim_transition_reason(claim)}


def semantic_classification(records: list[dict], specs_dir: Path) -> dict:
    """Per-semantic-classification (target oracle shape) denominator, point
    estimate, and descriptive interval (plan Task 13; gate2 finding 6).

    The plan requires these for every model, suite, direction, AND semantic
    classification; the analysis chain already emits the first three, and this
    closes the semantic-classification gap. Population: correctness-eligible L0
    replay records (KNOWN_FAIL and performance_only both excluded - gate3
    finding 1), grouped by the target spec's oracle shape. Denominator = record
    count, estimate = pass rate, interval = the Wilson 95% CI - the same
    estimator the rest of the analysis uses."""
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
    by_shape = {}
    for shape in sorted(agg):
        n, passes = agg[shape]["n"], agg[shape]["passes"]
        ci = wilson_ci(passes, n)
        by_shape[shape] = {
            "n": n, "passes": passes,
            "pass_rate": ci["value"],
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "ci_level": ci["ci_level"],
        }
    return by_shape


def _spec_kernel(spec: str | None) -> str:
    """'rodinia-bezier-surface-cuda' -> 'bezier-surface' (mirrors
    generate_paper_data._kernel_from_spec)."""
    if not spec:
        return "unknown"
    parts = spec.split("-")
    return "-".join(parts[1:-1]) if len(parts) >= 3 else spec


def headline_cell_transitions(replay_records: list[dict]) -> dict:
    """Per-cell old->new transition matrices behind EVERY changing headline-table
    cell (gate17 finding 2 / directive 2).

    The corpus, strata, and claim-scoped matrices elsewhere do not isolate a
    single headline cell: a cross-model per-(model, direction) cell (e.g. Codex
    cuda-to-omp, 69 records) is covered neither by strata_transitions.by_direction
    (all models pooled) nor by claim_scoped_transitions.by_direction_L0 (headline
    model only). This emits, for each headline table:
      * cross_model: record-unit AND task-unit per-(model, direction) matrices
        over the L0 correctness-eligible records (the cross_model_comparison
        overall / per_direction cells and their McNemar/per-kernel task cells);
      * suite_composition: the L0 per-suite record-unit matrices, and the
        rodinia-vs-others record-unit AND task-unit matrices (the record_unit /
        task_unit headline cells).
    Task-unit collapses (model, suite, kernel, direction) cells by any-sample
    success on both the new and parent (old) status. The verifier recomputes all
    of these independently."""
    def _dir(rec):
        return f"{_spec_api(rec.get('source_spec'))}-to-{_spec_api(rec.get('target_spec'))}"

    # --- record-unit, cross-model per (model, direction) and per suite ---
    cm_rec: dict[str, dict[str, dict[str, int]]] = {}
    suite_rec: dict[str, dict[str, int]] = {}
    rod_rec: dict[str, dict[str, int]] = {"rodinia": {}, "others": {}}
    # --- task cells: (model, suite, kernel, direction) -> collapsed status ---
    new_cell: dict[tuple, list[str]] = {}
    old_cell: dict[tuple, list[str]] = {}
    for rec in replay_records:
        if rec.get("augment_level") != 0 or not _eligible(rec):
            continue
        model = rec.get("model") or rec.get("_model") or "unknown"
        suite = _spec_suite(rec.get("source_spec"))
        direction = _dir(rec)
        old = (rec.get("parent") or {}).get("overall_status", "UNKNOWN")
        new = rec.get("overall_status", "UNKNOWN")
        tr = f"{old}->{new}"
        cm_rec.setdefault(model, {}).setdefault(direction, {})
        cm_rec[model][direction][tr] = cm_rec[model][direction].get(tr, 0) + 1
        cell = suite_rec.setdefault(suite, {})
        cell[tr] = cell.get(tr, 0) + 1
        grp = "rodinia" if suite == "rodinia" else "others"
        rod_rec[grp][tr] = rod_rec[grp].get(tr, 0) + 1
        key = (model, suite, _spec_kernel(rec.get("source_spec")), direction)
        new_cell.setdefault(key, []).append(new)
        old_cell.setdefault(key, []).append(old)

    # --- task-unit collapse (any-sample success on new and old) ---
    cm_task: dict[str, dict[str, dict[str, int]]] = {}
    rod_task: dict[str, dict[str, int]] = {"rodinia": {}, "others": {}}
    for key, new_st in new_cell.items():
        model, suite, _kernel, direction = key
        new = "PASS" if any(x == "PASS" for x in new_st) else "FAIL"
        old = "PASS" if any(x == "PASS" for x in old_cell[key]) else "FAIL"
        tr = f"{old}->{new}"
        cm_task.setdefault(model, {}).setdefault(direction, {})
        cm_task[model][direction][tr] = cm_task[model][direction].get(tr, 0) + 1
        grp = "rodinia" if suite == "rodinia" else "others"
        rod_task[grp][tr] = rod_task[grp].get(tr, 0) + 1

    def _s2(agg):  # sort a {model|group: {dir: matrix}}
        return {m: {d: dict(sorted(agg[m][d].items())) for d in sorted(agg[m])}
                for m in sorted(agg)}

    def _s1(agg):  # sort a {key: matrix}
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


def _spec_suite(spec: str | None) -> str:
    return spec.split("-", 1)[0] if spec else "unknown"


def _spec_api(spec: str | None) -> str:
    return spec.rsplit("-", 1)[1] if spec and "-" in spec else "unknown"


def _strata_cell(agg: dict[str, dict[str, int]]) -> dict:
    out = {}
    for name in sorted(agg):
        n, passes = agg[name]["n"], agg[name]["passes"]
        ci = wilson_ci(passes, n)
        out[name] = {
            "n": n, "passes": passes, "pass_rate": ci["value"],
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
        }
    return out


def strata_pass_rates(records: list[dict]) -> dict:
    """Per-model, per-suite, and per-direction denominator, point estimate, and
    Wilson interval over the correctness-eligible records (plan Task 13; gate3
    finding 3 - the old-vs-new headline previously carried only the overall
    rate).

    Population: records whose source AND target spec are correctness-eligible
    (KNOWN_FAIL and performance_only both excluded, via _eligible). Per-model and
    per-suite span every augmentation level (the canonical corpus); per-direction
    is L0 only, matching the analysis chain's direction_pass_rates. Suite = the
    source spec's suite prefix; direction = source_api-to-target_api. Estimate =
    Wilson pass rate, interval = Wilson 95% CI - the estimator used elsewhere."""
    by_model: dict[str, dict[str, int]] = {}
    by_suite: dict[str, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}

    def _acc(agg: dict, key: str, is_pass: bool) -> None:
        cell = agg.setdefault(key, {"n": 0, "passes": 0})
        cell["n"] += 1
        cell["passes"] += 1 if is_pass else 0

    for rec in records:
        if not _eligible(rec):
            continue
        is_pass = rec.get("overall_status") == "PASS"
        _acc(by_model, rec.get("model") or rec.get("_model") or "unknown", is_pass)
        _acc(by_suite, _spec_suite(rec.get("source_spec")), is_pass)
        if rec.get("augment_level") == 0:
            direction = (f"{_spec_api(rec.get('source_spec'))}-to-"
                         f"{_spec_api(rec.get('target_spec'))}")
            _acc(by_direction, direction, is_pass)
    return {
        "by_model": _strata_cell(by_model),
        "by_suite": _strata_cell(by_suite),
        "by_direction": _strata_cell(by_direction),
    }


# --------------------------------------------------------------------------- #
# Command runner                                                               #
# --------------------------------------------------------------------------- #

class Chain:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.log: list[dict] = []

    def run(self, argv: list[str], label: str) -> None:
        print(f"\n>>> [{label}] {' '.join(argv)}", flush=True)
        proc = subprocess.run(argv, cwd=str(self.project_root))
        self.log.append({"label": label, "command": argv, "exit_code": proc.returncode})
        if proc.returncode != 0:
            raise SystemExit(
                f"ABORT: step '{label}' exited {proc.returncode}: {' '.join(argv)}"
            )


def _py(script: str, *args: str) -> list[str]:
    return ["python3", script, *args]


# --------------------------------------------------------------------------- #
# Old-vs-new comparison                                                        #
# --------------------------------------------------------------------------- #

def _resolve_json_path(doc: dict, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def build_claim_status(new_claims: list[dict], analysis_dir: Path, old_dir: Path) -> list[dict]:
    """Mark each generated claim supported / qualified / removed, mechanically.

    Rule (deterministic, no interpretation):
      * removed  - the claim's new value cannot be regenerated (its evidence
                   file/path is missing), e.g. a deferred-namespace claim.
      * supported - the value is byte-stable old->new (|new-old| within 1e-9,
                    or equal for non-numerics): replay did not move it.
      * qualified - the value moved: the manuscript must use the new number;
                    the owner decides at the checkpoint whether the framing holds.
    """
    out = []
    for claim in new_claims:
        cid = claim.get("claim_id")
        new_val = claim.get("value")
        src = (claim.get("source_files") or [None])[0]
        jpath = claim.get("json_path")
        old_val = None
        old_doc = None
        if src is None or jpath is None or new_val is None:
            # A claim with no regenerable new value or no source is removed-class
            # (e.g. a deferred namespace). It carries no old counterpart.
            status = "removed"
        else:
            # Every remaining claim MUST have an old counterpart, resolved through
            # the SAME resolver the emitter uses - so a derived claim (a count, a
            # difference, a selected element) resolves to a scalar instead of a
            # container, and a missing/unresolvable old value FAILS rather than
            # being silently labeled 'qualified' (gate2 findings 2 and 4).
            old_file = old_dir / Path(src).name
            if not old_file.exists():
                raise SystemExit(
                    f"ABORT: claim '{cid}' has no old evidence ({old_file} "
                    f"missing). Every claim needs an old counterpart for the "
                    f"old-vs-new comparison (gate2 finding 2)."
                )
            old_doc = json.loads(old_file.read_text(encoding="utf-8"))
            old_val = resolve_claim_value(old_doc, claim)
            if old_val is None:
                raise SystemExit(
                    f"ABORT: claim '{cid}' old value did not resolve from "
                    f"{old_file} via {claim.get('derivation') or jpath} "
                    f"(gate2 finding 2/4)."
                )
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                status = "supported" if abs(float(new_val) - float(old_val)) <= 1e-9 else "qualified"
            else:
                status = "supported" if old_val == new_val else "qualified"
        # gate21 finding 3: present every NUMERIC component (Wilson interval bounds,
        # denominators, and the extra reported numbers - asymmetry p-values,
        # augmentation endpoints, chi-square p-values) old-beside-new, not just the
        # scalar value. Each component resolves through the SAME resolver as the
        # value, so a moved interval or endpoint is visible at the checkpoint.
        component_status = []
        for comp in claim.get("components") or []:
            c_new = comp.get("value")
            c_old = resolve_claim_value(old_doc, comp) if old_doc is not None else None
            if isinstance(c_old, (int, float)) and isinstance(c_new, (int, float)):
                c_stat = "supported" if abs(float(c_new) - float(c_old)) <= 1e-9 else "qualified"
            else:
                c_stat = "supported" if c_old == c_new else "qualified"
            component_status.append({
                "name": comp.get("name"), "old_value": c_old,
                "new_value": c_new, "status": c_stat,
            })
        out.append({
            "claim_id": cid,
            "section": claim.get("section"),
            "claim_text": claim.get("claim_text"),
            "old_value": old_val,
            "new_value": new_val,
            "unit": claim.get("unit"),
            "status": status,
            "component_status": component_status,
            "json_path": jpath,
            "source_file": src,
            # Per-record transition context (gate13 finding 2; gate16 finding 2;
            # gate17 finding 3): every RECORD-DERIVED claim is claim-scoped to its
            # exact record population (the verifier recomputes that cell); only a
            # claim whose value is not computed from record status is
            # not_applicable. record_derived is emitted so the verifier can assert
            # not_applicable is never used on a record-derived claim.
            "record_derived": claim_is_record_derived(claim),
            "transition_scope": claim_transition_scope(claim),
            "transitions": claim_transitions(claim),
        })
    # gate17 finding 3: a record-derived claim must never be not_applicable. This
    # is a generator-side invariant (the verifier re-checks it independently); if
    # the classification and the scope ever disagree, fail loudly rather than
    # ship a not_applicable escape on a record-derived claim.
    for entry in out:
        if entry["record_derived"] and entry["transition_scope"] == "not_applicable":
            raise SystemExit(
                f"ABORT: claim '{entry['claim_id']}' is record-derived but its "
                f"transition scope is not_applicable (gate17 finding 3)."
            )
    return out


def run_fail_mechanism_split(replay_root: Path, replay_records: list[dict]) -> dict:
    """Split the PASS->RUN_FAIL transitions by run mechanism (the owner's question).

    Buckets, from each sealed replay record's ``run_status``:
      (a) run_timeout        - run_status == 'timeout' (hit the run-time cap)
      (b) nonzero_exit_or_crash - run_status == 'fail' (nonzero/signal exit)
      (c) other              - anything else
    No interpretation - counts per model plus the exact record keys per bucket.
    """
    buckets = {"run_timeout": [], "nonzero_exit_or_crash": [], "other": []}
    by_model: dict[str, dict[str, int]] = {}
    for rec in replay_records:
        old = (rec.get("parent") or {}).get("overall_status")
        new = rec.get("overall_status")
        if old != "PASS" or new != "RUN_FAIL":
            continue
        run_status = rec.get("run_status")
        if run_status == "timeout":
            bucket = "run_timeout"
        elif run_status == "fail":
            bucket = "nonzero_exit_or_crash"
        else:
            bucket = "other"
        model = rec.get("model") or rec.get("_model") or "unknown"
        key = f"{model}/{rec.get('_stem')}"
        buckets[bucket].append({
            "record": key,
            "model": model,
            "run_status": run_status,
            "run_exit_code": rec.get("run_exit_code"),
            "run_time_seconds": rec.get("run_time_seconds"),
        })
        bm = by_model.setdefault(model, {"run_timeout": 0, "nonzero_exit_or_crash": 0, "other": 0})
        bm[bucket] += 1
    return {
        "note": (
            "PASS (submitted) -> RUN_FAIL (replay) transitions split by run "
            "mechanism from the sealed records. No interpretation; the owner "
            "rules on the split at the claims checkpoint. run_timeout = hit "
            "the 300s run-time cap (slow but possibly correct); "
            "nonzero_exit_or_crash = nonzero/signal exit; other = neither."
        ),
        "total_pass_to_run_fail": sum(len(v) for v in buckets.values()),
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "by_model": {m: by_model[m] for m in sorted(by_model)},
        "records_by_bucket": {
            k: sorted(v, key=lambda r: r["record"]) for k, v in buckets.items()
        },
    }


# Root-dependent headline deliverables placed old-beside-new in the comparison
# (gate5 finding 2). Each is generated over BOTH roots by the same code; the
# comparison embeds both documents and the verifier re-reads the hashed
# persisted copies to confirm the embedded old/new match.
HEADLINE_TABLE_FILES = ("cross_model_comparison.json", "suite_composition.json")


def build_headline_tables(new_dir: Path, old_dir: Path) -> dict:
    """Embed each root-dependent headline table old-beside-new.

    Both columns are the full document produced by the same generator over each
    root (new = replay, old = submitted). The verifier independently re-reads the
    hashed persisted copies and confirms they equal these embedded documents, so
    a corrupted headline table cannot slip through the comparison bytes alone.
    """
    tables: dict[str, dict] = {}
    for name in HEADLINE_TABLE_FILES:
        new_p, old_p = new_dir / name, old_dir / name
        if not (new_p.exists() and old_p.exists()):
            raise SystemExit(
                f"ABORT: headline table '{name}' missing an old/new counterpart "
                f"({new_p} / {old_p}); the old chain must generate it "
                f"(gate5 finding 2)."
            )
        tables[name] = {
            "new": json.loads(new_p.read_text(encoding="utf-8")),
            "old": json.loads(old_p.read_text(encoding="utf-8")),
        }
    return tables


def build_comparison(
    project_root: Path,
    analysis_dir: Path,
    old_dir: Path,
    replay_root: Path,
    submitted_root: Path,
    replay_records: list[dict],
    submitted_records: list[dict],
    contract_sha: str,
    seal_sha: str,
    commit: str,
) -> dict:
    new_stats = record_stats(replay_records)
    old_stats = record_stats(submitted_records)

    def _qf(root_dir: Path) -> dict:
        p = root_dir / "quantitative_findings.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    old_qf = _qf(old_dir)
    new_qf = _qf(analysis_dir)

    # Headline table: the same analysis-code aggregate over each root.
    headline = []
    for key, dotted, unit in [
        ("canonical_overall_pass_rate", "canonical.aggregate_pass_rates.overall.value", "rate"),
        ("canonical_overall_denominator", "canonical.aggregate_pass_rates.overall.n", "count"),
        ("canonical_overall_ci_lower", "canonical.aggregate_pass_rates.overall.ci_lower", "rate"),
        ("canonical_overall_ci_upper", "canonical.aggregate_pass_rates.overall.ci_upper", "rate"),
    ]:
        ov = _resolve_json_path(old_qf, dotted)
        nv = _resolve_json_path(new_qf, dotted)
        delta = None
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            delta = round(nv - ov, 6)
        headline.append({
            "key": key, "unit": unit, "old_value": ov,
            "new_value": nv, "delta": delta,
            "source": "quantitative_findings.json (same code, old=submitted / new=replay)",
        })

    # Canonical overall pass rate, taken from the analysis output. The
    # independent verifier recomputes this from scratch (PASS / eligible over
    # CORRECTNESS_INELIGIBLE_SPECS) and must agree - so a from-scratch recount
    # confirms the analysis code's headline (gate1 finding 3).
    def _canon(qf: dict) -> dict:
        ov = _resolve_json_path(qf, "canonical.aggregate_pass_rates.overall") or {}
        return {"value": ov.get("value"), "n": ov.get("n")}

    canonical_overall = {"new": _canon(new_qf), "old": _canon(old_qf)}

    # Record-level status movement, presented identically for both directions.
    status_movement = []
    for status in STATUSES:
        ov = old_stats["status_counts"].get(status, 0)
        nv = new_stats["status_counts"].get(status, 0)
        if ov or nv:
            status_movement.append({
                "status": status, "old_count": ov, "new_count": nv, "delta": nv - ov,
            })

    # Per-model / per-suite / per-direction and per-semantic-classification
    # tables, old next to new, each with denominator, point estimate, and Wilson
    # interval (gate3 finding 3 - the headline previously carried only the
    # overall rate). Both columns are recomputed from the record JSONs by the
    # SAME code over each root; the independent verifier reproduces the strata.
    specs_dir = project_root / "specs"
    strata = {
        "new": strata_pass_rates(replay_records),
        "old": strata_pass_rates(submitted_records),
    }
    semantic = {
        "new": semantic_classification(replay_records, specs_dir),
        "old": semantic_classification(submitted_records, specs_dir),
    }

    # Claim status.
    new_claims_path = analysis_dir / "paper_claims.json"
    new_claims = json.loads(new_claims_path.read_text(encoding="utf-8"))["claims"] \
        if new_claims_path.exists() else []
    claim_status = build_claim_status(new_claims, analysis_dir, old_dir)

    return {
        "artifact": "old_vs_new_comparison",
        "generated_by": "scripts/analysis/generate_final_evidence.py",
        "generated_at": _now(),
        "commit": commit,
        "host": _host(),
        "note": (
            "OLD = submitted namespace verdicts (results/evaluation). NEW = "
            "sealed replay verdicts under the frozen contract "
            "(results/evaluation_final/parbench-final-v1). Both columns come "
            "from the same analysis code; only the input root differs. "
            "Favorable and unfavorable movements are presented identically. "
            "This document goes to the owner's claims checkpoint BEFORE any "
            "manuscript regeneration."
        ),
        "roots": {
            "submitted": str(submitted_root),
            "replay": str(replay_root),
        },
        "provenance": {
            "contract_manifest_sha256": contract_sha,
            "seal_aggregate_sha256": seal_sha,
        },
        "record_level": {"old": old_stats, "new": new_stats},
        "canonical_overall": canonical_overall,
        "strata": strata,
        "semantic_classification": semantic,
        "headline_tables": build_headline_tables(analysis_dir, old_dir),
        "transition_counts": transition_counts(replay_records),
        # Scoped per-record transitions behind each record-derived change
        # (gate13 finding 2): the corpus matrix above alone does not say which
        # records drove a given strata/semantic cell's delta.
        "strata_transitions": strata_transitions(replay_records),
        "semantic_transitions": semantic_transitions(replay_records, specs_dir),
        # Transitions over the headline model's exact claim populations
        # (gate14 finding 1): a per-model-per-direction claim's records, which
        # the coarse strata tables above do not isolate.
        "claim_scoped_transitions": claim_scoped_transitions(
            replay_records, CLAIMS_MODEL),
        # Per-cell transition matrix behind every changing headline-table cell
        # (gate17 finding 2): the per-(model, direction) cross-model cells and the
        # suite-composition record/task-unit cells, which the pooled strata and
        # headline-model claim-scoped matrices above do not isolate.
        "headline_cell_transitions": headline_cell_transitions(replay_records),
        "status_movement": status_movement,
        "headline": headline,
        "claim_status": claim_status,
        "run_fail_mechanism_split": run_fail_mechanism_split(replay_root, replay_records),
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--contract-manifest", type=Path,
                    default=Path("config/final_contract_manifest.json"))
    ap.add_argument("--replay-root", type=Path, required=True)
    ap.add_argument("--submitted-root", type=Path, default=Path("results/evaluation"))
    ap.add_argument("--comparison-out", type=Path,
                    default=Path("results/analysis/old_vs_new_comparison.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/analysis/final_evidence_manifest.json"))
    ap.add_argument("--analysis-dir", type=Path,
                    default=Path("results/analysis"),
                    help="Top-level analysis dir; holds the plan-named Task 13 "
                    "deliverables (old_vs_new_comparison, final_evidence_manifest, "
                    "final_oracle_*). The submitted-era canonical analysis JSONs "
                    "here are NOT clobbered.")
    ap.add_argument("--final-dir", type=Path,
                    default=Path("results/analysis/final"),
                    help="Where the replay-based final analysis chain is written, "
                    "kept separate from the committed submitted-era analysis so "
                    "the new numbers are not adopted before the owner's checkpoint.")
    args = ap.parse_args(argv)

    project_root = args.project_root.resolve()
    replay_root = (project_root / args.replay_root).resolve() \
        if not args.replay_root.is_absolute() else args.replay_root.resolve()
    submitted_root = (project_root / args.submitted_root).resolve() \
        if not args.submitted_root.is_absolute() else args.submitted_root.resolve()
    top_dir = (project_root / args.analysis_dir).resolve() \
        if not args.analysis_dir.is_absolute() else args.analysis_dir.resolve()
    final_dir = (project_root / args.final_dir).resolve() \
        if not args.final_dir.is_absolute() else args.final_dir.resolve()
    contract_manifest = (project_root / args.contract_manifest).resolve() \
        if not args.contract_manifest.is_absolute() else args.contract_manifest.resolve()
    comparison_out = (project_root / args.comparison_out).resolve() \
        if not args.comparison_out.is_absolute() else args.comparison_out.resolve()
    output_path = (project_root / args.output).resolve() \
        if not args.output.is_absolute() else args.output.resolve()

    # gate19 finding 1: refuse to write outputs inside the immutable submitted
    # corpus or a sealed namespace BEFORE creating any directory or deleting
    # <final-dir>/old_evidence. A direct CLI run is not covered by the Bash hook.
    unsafe = _unsafe_output_reasons(
        [("--analysis-dir", top_dir), ("--final-dir", final_dir),
         ("--comparison-out", comparison_out), ("--output", output_path)],
        submitted_root, replay_root,
        corpus_root=(project_root / "results" / "evaluation").resolve(),
        project_root=project_root,
        input_paths=[contract_manifest, replay_root, submitted_root,
                     (project_root / "manifest.jsonl").resolve()])
    if unsafe:
        print("REFUSING to write: " + "; ".join(unsafe), file=sys.stderr)
        return 2

    top_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit(project_root)
    contract_sha = _sha256(contract_manifest)
    seal_path = replay_root / ".parbench-seal.json"
    seal_sha = json.loads(seal_path.read_text())["aggregate_sha256"] \
        if seal_path.exists() else None

    import tempfile
    stage = Path(tempfile.mkdtemp(prefix="parbench_final_"))
    old_dir = stage / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    sloc_stage = stage / "sloc_new"
    sloc_stage.mkdir(parents=True, exist_ok=True)

    ch = Chain(project_root)
    RR, SR, AD = str(replay_root), str(submitted_root), str(final_dir)

    # --- NEW (replay-root) canonical outputs -------------------------------- #
    ch.run(_py("scripts/analysis/statistical_analysis.py",
               "--results-root", RR, "--output-dir", AD), "statistical(new)")

    # SLoC runs BEFORE quantitative_findings (gate7 finding 5): quantitative_findings
    # reads sloc_analysis.json for its per-suite SLoC characteristics and SLoC
    # correlation. It used to hard-read the submitted-era results/analysis copy - a
    # hidden, unbound input inconsistent with the replay-derived sloc claim. Generate
    # the final replay-derived SLoC first and pass it explicitly, so the value is
    # consistent and its file is a bound manifest input.
    # eval_summary over the replay root (needed for sloc pass-rates), written
    # OUTSIDE the seal into the staging dir. This is external, read-only,
    # derived metadata - not a mirror of the sealed records.
    eval_summary_new = sloc_stage / "eval_summary.json"
    ch.run(_py("scripts/evaluation/analyze_eval.py",
               "--results-dir", RR,
               "--out-json", str(eval_summary_new),
               "--out-md", str(sloc_stage / "eval_summary.md")),
           "eval_summary(new)")
    # translation_complexity.csv is spec-pair derived (model-independent); the
    # authoritative submitted copy is passed read-only. SLoC runs against the
    # SEALED replay root directly, reading both metadata files from outside it
    # via explicit flags - no staging mirror of records (gate1 finding 7).
    tc_src = submitted_root / "translation_complexity.csv"
    sloc_new = final_dir / "sloc_analysis.json"
    ch.run(_py("scripts/analysis/sloc_analysis.py",
               "--project-root", ".", "--results-root", RR,
               "--eval-summary", str(eval_summary_new),
               "--translation-complexity", str(tc_src),
               "--output", str(sloc_new)), "sloc(new)")

    # --cross-check-root AD (gate8 finding 4): cross-check against the
    # replay-derived final outputs in this run, never submitted-era
    # results/analysis. statistical(new) already wrote statistical_analysis.json
    # into AD above, so the cross-check has replay-consistent data to compare.
    ch.run(_py("scripts/analysis/quantitative_findings.py",
               "--results-root", RR, "--output-dir", AD,
               "--cross-check-root", AD,
               "--sloc-analysis", str(sloc_new)), "quantitative_pooled(new)")
    ch.run(_py("scripts/analysis/quantitative_findings.py",
               "--results-root", RR, "--output-dir", AD,
               "--cross-check-root", AD,
               "--sloc-analysis", str(sloc_new),
               "--model-dir", CLAIMS_MODEL), "quantitative_model(new)")
    ch.run(_py("scripts/analysis/build_error_taxonomy.py",
               "--project-root", ".", "--results-root", RR,
               "--output-dir", AD), "error_taxonomy(new)")
    ch.run(_py("scripts/analysis/benchmark_characterization.py",
               "--project-root", ".", "--output-dir", AD), "benchmark_char")
    # gate9 finding 1: the NEW token analysis is verdict-dependent (pass counts,
    # pass rates, and prompt/completion-vs-pass correlations), so it must run
    # over the REPLAY root to use the replay verdicts. The token counts
    # themselves are stored-translation properties recovered read-only from each
    # replay record's referenced submitted record (input_record.path), so the
    # token values stay the invariant originals. The OLD chain below still runs
    # token_analysis over the submitted root for the comparison's old column.
    ch.run(_py("scripts/analysis/token_analysis.py",
               "--results-root", RR, "--output-dir", AD), "token_analysis(new)")

    # Per-model paper data + cross-model comparison (over the replay root).
    model_data_args: list[str] = []
    for model in MODELS:
        out = final_dir / f"paper_data_{model}.json"
        ch.run(_py("scripts/analysis/generate_paper_data.py",
                   "--results-dir", str(replay_root / model),
                   "--output", str(out)), f"paper_data:{model}")
        model_data_args += ["--model-data", str(out)]
    ch.run(_py("scripts/analysis/cross_model_comparison.py",
               *model_data_args,
               "--output", str(final_dir / "cross_model_comparison.json")),
           "cross_model")

    # Suite composition (over the replay root).
    ch.run(_py("scripts/rebuttal/suite_composition.py",
               "--results-root", RR,
               "--output", str(final_dir / "suite_composition.json"),
               "--claim-status", str(final_dir / "suite_composition_claim_status.json")),
           "suite_composition(new)")

    # Comparator labels (passthrough of the oracle report; root-independent).
    ch.run(_py("scripts/rebuttal/generate_comparator_labels.py",
               "--project-root", ".",
               "--output", str(final_dir / "comparator_labels.json")),
           "comparator_labels")

    # Oracle census trio -> the plan-named Task 13 deliverables at the top-level
    # analysis dir (these are new files, not submitted-era canonical outputs).
    ch.run(_py("scripts/rebuttal/oracle_shape_census.py",
               "--project-root", ".", "--results-root", RR,
               "--json-out", str(top_dir / "final_oracle_shape_census.json")),
           "oracle_census(new)")
    ch.run(_py("scripts/spec_tools/direct_oracle_scan.py",
               "--spec-root", "specs",
               "--json-out", str(top_dir / "final_direct_oracle_scan.json")),
           "direct_oracle_scan")
    ch.run(_py("scripts/analysis/compare_oracle_censuses.py",
               "--census", str(top_dir / "final_oracle_shape_census.json"),
               "--direct-scan", str(top_dir / "final_direct_oracle_scan.json"),
               "--expected-specs", "206",
               "--oracle-report", str(top_dir / "oracle_contract_report.json")),
           "compare_oracle_censuses")

    # Paper claims (consumes the new analysis outputs in the final dir).
    ch.run(_py("scripts/analysis/generate_paper_claims.py",
               "--project-root", ".", "--analysis-root", AD,
               "--output", str(final_dir / "paper_claims.json")), "paper_claims(new)")

    # E-01 direction-pair cross-check on the new outputs.
    ch.run(_py("scripts/analysis/verify_direction_pairs.py",
               "--statistical", str(final_dir / "statistical_analysis.json"),
               "--quantitative", str(final_dir / "quantitative_findings.json"),
               "--expected-pairs", "5"), "verify_direction_pairs(new)")

    # --- OLD (submitted-root) values for the comparison --------------------- #
    # gate11 finding 1: the OLD chain MIRRORS the new chain - same code, only the
    # root differs. So statistical(old) runs FIRST (it is the cross-check input),
    # then quantitative(old) passes --cross-check-root old_dir. Without the flag,
    # quantitative_findings.cross_check fell back to the submitted-era
    # results/analysis JSONs - a hidden input outside the evidence manifest - so
    # the old column was not a same-code/only-root-diff mirror of the new column.
    ch.run(_py("scripts/analysis/statistical_analysis.py",
               "--results-root", SR, "--output-dir", str(old_dir)),
           "statistical(old)")
    # SLoC(old) then quantitative(old) with an explicit --sloc-analysis, for the
    # same reason as the new chain (gate7 finding 5): quantitative_findings must
    # read the submitted-root SLoC file it is paired with, not a hidden global read.
    eval_summary_old = stage / "sloc_old" / "eval_summary.json"
    eval_summary_old.parent.mkdir(parents=True, exist_ok=True)
    ch.run(_py("scripts/evaluation/analyze_eval.py",
               "--results-dir", SR,
               "--out-json", str(eval_summary_old),
               "--out-md", str(eval_summary_old.parent / "eval_summary.md")),
           "eval_summary(old)")
    sloc_old = old_dir / "sloc_analysis.json"
    ch.run(_py("scripts/analysis/sloc_analysis.py",
               "--project-root", ".", "--results-root", SR,
               "--eval-summary", str(eval_summary_old),
               "--translation-complexity", str(tc_src),
               "--output", str(sloc_old)), "sloc(old)")
    ch.run(_py("scripts/analysis/quantitative_findings.py",
               "--results-root", SR, "--output-dir", str(old_dir),
               "--cross-check-root", str(old_dir),
               "--sloc-analysis", str(sloc_old)),
           "quantitative_pooled(old)")
    ch.run(_py("scripts/analysis/quantitative_findings.py",
               "--results-root", SR, "--output-dir", str(old_dir),
               "--cross-check-root", str(old_dir),
               "--sloc-analysis", str(sloc_old),
               "--model-dir", CLAIMS_MODEL), "quantitative_model(old)")
    ch.run(_py("scripts/analysis/build_error_taxonomy.py",
               "--project-root", ".", "--results-root", SR,
               "--output-dir", str(old_dir)), "error_taxonomy(old)")
    # Complete the old chain so EVERY claim source has an old counterpart
    # (gate2 finding 2): token (submitted root, invariant under replay) and
    # benchmark characterization (spec-derived, root-independent). Without these,
    # the token_cost / suite_composition / multi_file claims had old_value null
    # yet were labeled 'qualified' (a measured change that was never measured).
    # Ordered error_taxonomy -> benchmark_char -> token to mirror the new chain.
    ch.run(_py("scripts/analysis/benchmark_characterization.py",
               "--project-root", ".", "--output-dir", str(old_dir)),
           "benchmark_char(old)")
    ch.run(_py("scripts/analysis/token_analysis.py",
               "--results-root", SR, "--output-dir", str(old_dir)),
           "token_analysis(old)")

    # Old-side headline tables (gate5 finding 2): the per-model paper data,
    # cross-model comparison, and suite composition are root-dependent headline
    # deliverables generated for the replay root above but were never generated
    # for the submitted root, so the comparison could not place these headline
    # tables old-beside-new. Run the SAME generators over the submitted root.
    old_model_data_args: list[str] = []
    for model in MODELS:
        out = old_dir / f"paper_data_{model}.json"
        ch.run(_py("scripts/analysis/generate_paper_data.py",
                   "--results-dir", str(submitted_root / model),
                   "--output", str(out)), f"paper_data:{model}(old)")
        old_model_data_args += ["--model-data", str(out)]
    ch.run(_py("scripts/analysis/cross_model_comparison.py",
               *old_model_data_args,
               "--output", str(old_dir / "cross_model_comparison.json")),
           "cross_model(old)")
    ch.run(_py("scripts/rebuttal/suite_composition.py",
               "--results-root", SR,
               "--output", str(old_dir / "suite_composition.json"),
               "--claim-status", str(old_dir / "suite_composition_claim_status.json")),
           "suite_composition(old)")

    # --- Build the comparison + manifest ------------------------------------ #
    replay_records = load_records(replay_root)
    submitted_records = load_records(submitted_root)

    # Eligibility is bound to the FROZEN contract, not merely the code constant
    # (gate6 finding 4): validate the contract's exclusion set against the
    # canonical constant and reject any correctness-eligible replay pair absent
    # from the frozen eligible_task_list before any count is emitted.
    elig_failures = contract_eligibility_failures(replay_records, contract_manifest)
    if elig_failures:
        print("CONTRACT ELIGIBILITY FAILURES:", file=sys.stderr)
        for f in elig_failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    comparison = build_comparison(
        project_root, final_dir, old_dir, replay_root, submitted_root,
        replay_records, submitted_records, contract_sha, seal_sha or "", commit,
    )
    comparison_out = (project_root / args.comparison_out).resolve() \
        if not args.comparison_out.is_absolute() else args.comparison_out.resolve()
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_out.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {comparison_out}")

    # Semantic-classification deliverable (gate2 finding 6): denominator, point
    # estimate, and descriptive interval per target-oracle shape over the sealed
    # replay root. The oracle-contract report carries the shape COUNTS but no
    # pass-rate estimate/interval; this closes that gap. Written under the final
    # dir and recorded as a hashed output; the verifier recomputes it.
    sem_doc = {
        "artifact": "oracle_semantic_classification",
        "generated_by": "scripts/analysis/generate_final_evidence.py",
        "generated_at": _now(),
        "results_root": str(replay_root),
        "population": (
            "correctness-eligible L0 replay records (KNOWN_FAIL and "
            "performance-only specs both excluded), grouped by the target "
            "spec's oracle shape"
        ),
        "by_shape": semantic_classification(replay_records, project_root / "specs"),
    }
    sem_path = final_dir / "oracle_semantic_classification.json"
    sem_path.write_text(json.dumps(sem_doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {sem_path}")

    # Replay-based final analysis chain, written under the final dir (kept
    # separate from the submitted-era committed canonical outputs).
    final_chain = [
        "oracle_semantic_classification.json",
        "statistical_analysis.json", "statistical_analysis.md",
        "quantitative_findings.json", "quantitative_findings.md",
        f"quantitative_findings_{CLAIMS_MODEL}.json",
        f"quantitative_findings_{CLAIMS_MODEL}.md",
        "error_taxonomy.json", "error_taxonomy.md",
        "benchmark_characterization.json", "benchmark_characterization.md",
        "token_analysis.json", "token_analysis.md",
        "sloc_analysis.json", "sloc_analysis.md",
        "cross_model_comparison.json",
        "suite_composition.json", "suite_composition_claim_status.json",
        "comparator_labels.json",
        "paper_claims.json",
    ] + [f"paper_data_{m}.json" for m in MODELS]
    # Plan-named Task 13 deliverables at the top-level analysis dir.
    top_deliverables = [
        "final_oracle_shape_census.json", "final_direct_oracle_scan.json",
    ]

    paper_claims_path = final_dir / "paper_claims.json"
    outputs = []
    for name in final_chain:
        p = final_dir / name
        if p.exists():
            outputs.append({"path": str(p.relative_to(project_root)),
                            "sha256": _sha256(p)})
    for name in top_deliverables:
        p = top_dir / name
        if p.exists():
            outputs.append({"path": str(p.relative_to(project_root)),
                            "sha256": _sha256(p)})
    outputs.append({"path": str(comparison_out.relative_to(project_root)),
                    "sha256": _sha256(comparison_out)})

    # Persist the OLD-side evidence each claim resolves from, so the verifier can
    # independently re-resolve every old claim value and re-derive its
    # disposition instead of trusting the comparison file's bytes (gate4 finding
    # 2). Without this the old claim value + disposition were only hash-bound and
    # a matching corruption passed. Only the distinct source files the claims
    # reference are persisted (from the old-root analysis chain in old_dir).
    import shutil
    old_evidence_dir = final_dir / "old_evidence"
    # Reconcile the directory to EXACTLY this run's output set (gate10 finding 2).
    # This dir is rewritten every run from the claim/headline evidence below; a
    # file left by a prior run (e.g. a token_analysis.json that no claim resolves
    # from anymore) would otherwise stay tracked, unbound by any manifest output
    # hash, yet still pass verification. Clearing first makes the on-disk set and
    # the manifest output set identical by construction.
    if old_evidence_dir.exists():
        shutil.rmtree(old_evidence_dir)
    old_evidence_dir.mkdir(parents=True, exist_ok=True)
    old_claim_evidence: dict[str, dict] = {}
    for entry in comparison["claim_status"]:
        src = entry.get("source_file")
        if not src:
            continue
        base = Path(src).name
        if base in old_claim_evidence:
            continue
        old_src = old_dir / base
        if old_src.exists():
            dest = old_evidence_dir / base
            shutil.copyfile(old_src, dest)
            rel = str(dest.relative_to(project_root))
            old_claim_evidence[base] = {"path": rel, "sha256": _sha256(dest)}
            outputs.append({"path": rel, "sha256": _sha256(dest)})

    # Persist the OLD headline tables (gate5 finding 2) as hashed outputs so the
    # verifier can re-read them and confirm the comparison's embedded old-beside-
    # new documents are exactly the persisted bytes (the NEW copies are already
    # hashed outputs in the final chain). Record a name->{new_output, old_evidence}
    # map so the verifier knows which hashed files back each headline table.
    headline_evidence_dir = final_dir / "old_evidence"
    headline_evidence_dir.mkdir(parents=True, exist_ok=True)
    headline_table_map: dict[str, dict] = {}
    for name in HEADLINE_TABLE_FILES:
        new_rel = str((final_dir / name).relative_to(project_root))
        old_src = old_dir / name
        dest = headline_evidence_dir / f"old_{name}"
        shutil.copyfile(old_src, dest)
        old_rel = str(dest.relative_to(project_root))
        outputs.append({"path": old_rel, "sha256": _sha256(dest)})
        headline_table_map[name] = {"new_output": new_rel, "old_evidence": old_rel}

    # Required frozen deliverables (plan Task 13 "Generated outputs"): the
    # oracle-contract report (which also carries the semantic-audit strength
    # estimates), the 86-row pair-comparability disposition table, and the ten
    # source-baseline disposition artifacts. These are produced by the upstream
    # Task 3/4 generators over the FROZEN contract (not regenerable in a
    # replay-only run), so they are recorded and hashed here so the evidence
    # package and the verifier cover the full plan-required output set
    # (gate1 finding 6, 2026-08-12). Categorized so the verifier can require
    # each category to be present.
    frozen_deliverables: dict[str, list[str]] = {
        "oracle_contract_report": [],
        "pair_comparability_dispositions": [],
        "source_baseline_dispositions": [],
    }
    for cat, paths in [
        ("oracle_contract_report",
         [top_dir / "oracle_contract_report.json",
          top_dir / "oracle_contract_report.md"]),
        ("pair_comparability_dispositions",
         [top_dir / "pair_contract_dispositions.json"]),
        ("source_baseline_dispositions",
         sorted((top_dir / "source_baseline_dispositions").glob("*.json"))),
    ]:
        for p in paths:
            if p.exists():
                rel = str(p.relative_to(project_root))
                frozen_deliverables[cat].append(rel)
                outputs.append({"path": rel, "sha256": _sha256(p)})

    claim_keys = [c.get("claim_id") for c in comparison["claim_status"]]

    inputs = {}
    for label, p in [
        ("contract_manifest", contract_manifest),
        ("seal", seal_path),
        ("replay_transitions", replay_root / "replay_transitions.json"),
        ("oracle_contract_report", top_dir / "oracle_contract_report.json"),
        ("translation_complexity_csv", submitted_root / "translation_complexity.csv"),
        # manifest.jsonl is a real input of benchmark_characterization.py and
        # sloc_analysis.py (kernel->category map); bind its bytes (gate4 finding 3).
        ("manifest_jsonl", project_root / "manifest.jsonl"),
        # config/final_pair_contracts.json is read by generate_comparator_labels.py
        # to label every comparator pair; bind its bytes so a post-generation
        # registry edit is caught by verify_final_evidence (gate12 finding 4,
        # E-16 live source-of-truth synchronization).
        ("pair_registry", project_root / "config" / "final_pair_contracts.json"),
        # config/sloc_corpus_kernels.json is the hash-verified frozen corpus
        # population loaded by sloc_analysis.CORPUS_KERNELS and reused by
        # benchmark_characterization; it defines the SLoC and suite-composition
        # population, so bind its bytes too (gate22 finding 1, E-16).
        ("sloc_corpus_registry",
         project_root / "config" / "sloc_corpus_kernels.json"),
    ]:
        if p.exists():
            inputs[label] = {"path": str(p), "sha256": _sha256(p)}

    # Spec JSONs and the benchmark-source bytes are also read by
    # benchmark_characterization.py and sloc_analysis.py. Bind them as
    # aggregates the verifier independently recomputes (gate4 finding 3): a
    # single-file hash per spec-referenced source, aggregated the same way
    # scripts/provenance/inventory_immutable_inputs does.
    _spec_hashes = {p.name: sha256_file(p)
                    for p in sorted((project_root / "specs").glob("*.json"))}
    _bench = inventory_spec_referenced_files(project_root)
    # benchmark_characterization.grep_dir rglobs the WHOLE source directory per
    # corpus kernel, a superset of the spec-referenced files (build-tree CMake /
    # CUDA-fe artifacts included). Bind that EXACT recursive read set too so the
    # input closure is complete and tamper-evident (gate5 finding 3).
    _char = inventory_characterization_source_files(project_root)
    input_inventories = {
        "note": (
            "specs + benchmark-source bytes read by benchmark_characterization "
            "and sloc_analysis; aggregates recomputed by verify_final_evidence. "
            "characterization_sources is the EXACT recursive file set "
            "benchmark_characterization reads (superset of spec-referenced)."
        ),
        "specs": {"aggregate_sha256": aggregate_sha256(_spec_hashes),
                  "file_count": len(_spec_hashes)},
        "benchmark_sources": {"aggregate_sha256": _bench["aggregate_sha256"],
                              "file_count": _bench["file_count"],
                              "missing_count": _bench["missing_count"]},
        "characterization_sources": {"aggregate_sha256": _char["aggregate_sha256"],
                                     "file_count": _char["file_count"],
                                     "dir_count": _char["dir_count"]},
    }

    manifest = {
        "artifact": "final_evidence_manifest",
        "generated_by": "scripts/analysis/generate_final_evidence.py",
        "generated_at": _now(),
        "commit": commit,
        "commit_note": (
            "HEAD at generation time; it may PRE-date the commit that lands "
            "these generators (they are staged/committed after this file is "
            "written). Provenance is bound by the 'generators' hashes below, "
            "not by this commit (gate1 finding 5)."
        ),
        "host": _host(),
        "python_version": sys.version.split()[0],
        "contract_manifest": {"path": str(contract_manifest), "sha256": contract_sha},
        "replay_root": {"path": str(replay_root), "seal_aggregate_sha256": seal_sha},
        "submitted_root": {
            "path": str(submitted_root),
            **_aggregate_sha256(submitted_root),
        },
        "generators": {
            rel: _sha256(project_root / rel)
            for rel in sorted(set(GENERATOR_SCRIPTS) | executed_producer_scripts(ch.log))
            if (project_root / rel).exists()
        },
        "expected_direction_pairs": 5,
        "inputs": inputs,
        "input_inventories": input_inventories,
        "old_claim_evidence": old_claim_evidence,
        "outputs": outputs,
        "frozen_deliverables": frozen_deliverables,
        "headline_tables": headline_table_map,
        "commands": ch.log,
        "headline": {
            "note": (
                "Record-level counts recomputed directly from record JSONs; "
                "verify_final_evidence.py reproduces these independently."
            ),
            "new": comparison["record_level"]["new"],
            "old": comparison["record_level"]["old"],
            "canonical_overall": comparison["canonical_overall"],
            "strata": comparison["strata"],
            "transition_counts": comparison["transition_counts"],
            "run_fail_mechanism_split": comparison["run_fail_mechanism_split"]["bucket_counts"],
        },
        "claim_keys": claim_keys,
        "paper_claims_path": str(paper_claims_path.relative_to(project_root)),
        "semantic_classification_path": str(sem_path.relative_to(project_root)),
        "comparison": str(comparison_out.relative_to(project_root)),
        "note": (
            "The replay-based final analysis chain is written under "
            f"{final_dir.relative_to(project_root)}/ and is NOT adopted into the "
            "submitted-era canonical results/analysis/*.json; adoption is decided "
            "at the owner's claims checkpoint from old_vs_new_comparison.json."
        ),
    }
    out = (project_root / args.output).resolve() \
        if not args.output.is_absolute() else args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"\nDONE: {len(outputs)} outputs, {len(claim_keys)} claim keys, "
          f"{manifest['headline']['run_fail_mechanism_split']} run_fail split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
