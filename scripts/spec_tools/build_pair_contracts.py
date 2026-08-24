#!/usr/bin/env python3
"""Enumerate and build the pair-contract registry (Task 2, gate E-02).

This script is the ONLY writer of the pair-contract registry files under
``config/`` (never under ``specs/``, where validators and analyses treat
every JSON as a kernel spec). The CURRENT registry is
``config/pair_contracts_v2.json`` (kernel-only pairs carry TARGET args,
owner OD-3 follow-up ruling 2026-08-21). The frozen
``config/final_pair_contracts.json`` backs the sealed NeurIPS evidence
package (bytes hash-bound by ``final_evidence_manifest.json``) and is never
regenerated.

Task 2 implements ``--enumerate-candidates``: deterministically enumerate
every candidate directed cross-API pair (same suite, same kernel, different
parallel API) from the spec corpus, with every entry explicitly unresolved.
Enumeration does NOT certify production eligibility: Task 4 populates
spec-side data (``--populate-from-specs``) and Task 7 finalizes with Linux
witness evidence (``--finalize``); both extend this script when they land.

Usage:
    python3 scripts/spec_tools/build_pair_contracts.py \\
      --enumerate-candidates \\
      --output config/pair_contract_candidates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.llm_evaluate import (  # noqa: E402
    _is_kernel_only_translation,
)
from scripts.evaluation.pair_contracts import (  # noqa: E402
    CONTRACT_VERSION_CURRENT,
    load_pair_contracts,
)


def enumerate_candidate_pairs(spec_root: Path) -> list[dict]:
    """Return unresolved candidate entries for every directed cross-API pair.

    Pairs are scoped to one (source_suite, kernel_name) group; each group
    with *n* distinct APIs yields ``n*(n-1)`` ordered pairs. Output order is
    deterministic: sorted by (source_spec, target_spec).
    """
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for spec_path in sorted(spec_root.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        identity = spec.get("identity") or {}
        suite = identity.get("source_suite")
        kernel = identity.get("kernel_name")
        api = identity.get("parallel_api")
        unique_id = identity.get("unique_id")
        if not (suite and kernel and api and unique_id):
            raise ValueError(f"spec missing identity fields: {spec_path}")
        groups.setdefault((suite, kernel), []).append((api, unique_id))

    entries: list[dict] = []
    for (_suite, _kernel), members in groups.items():
        for src_api, src_id in members:
            for tgt_api, tgt_id in members:
                if src_api == tgt_api:
                    continue
                entries.append({
                    "source_spec": src_id,
                    "target_spec": tgt_id,
                    "baseline_witness": {"status": "unresolved"},
                    "comparability": "unresolved",
                })
    entries.sort(key=lambda e: (e["source_spec"], e["target_spec"]))
    return entries


_PAIR_STRATEGY_FIELDS = {
    # per-type field allowlists mirroring schema/pair_contract_schema.json's
    # oracle_strategy definition (additionalProperties: false); spec-only
    # fields like tolerance_type must be stripped, never copied through.
    "exit_code": ("expected", "description"),
    "stdout_pattern": ("pattern", "description"),
    "numeric_comparison": ("extract_regex", "expected", "tolerance",
                           "description"),
    "file_hash": ("path", "expected_sha256", "description"),
    "file_diff": ("path", "reference_file", "description"),
}


def _spec_index(spec_root: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for spec_path in sorted(spec_root.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        index[spec["identity"]["unique_id"]] = spec
    return index


def _correctness_args(spec: dict) -> list[str]:
    cfg = (spec.get("run", {}).get("input_configurations") or {}).get(
        "correctness") or {}
    return [str(a) for a in (cfg.get("arguments") or [])]


def _pair_oracle_strategies(spec: dict) -> tuple[list[dict], list[dict]]:
    """Split the target spec's declared strategies into (positive oracle,
    negative checks) in pair-schema shape."""
    positive: list[dict] = []
    negative: list[dict] = []
    for strat in spec.get("verification", {}).get("strategies", []):
        stype = strat.get("type")
        if stype == "stdout_exclude_pattern":
            entry = {"type": stype, "pattern": strat["pattern"]}
            if strat.get("description"):
                entry["description"] = strat["description"]
            negative.append(entry)
            continue
        allowed = _PAIR_STRATEGY_FIELDS.get(stype)
        if allowed is None:
            raise ValueError(
                f"{spec['identity']['unique_id']}: strategy type {stype!r} "
                f"has no pair-contract mapping"
            )
        entry = {"type": stype}
        for field in allowed:
            if field in strat and strat[field] is not None:
                entry[field] = strat[field]
        positive.append(entry)
    return positive, negative


def populate_from_specs(candidates: dict, spec_root: Path) -> dict:
    """Task 4: fill spec-side data into every enumerated candidate pair.

    Populates shared_input (each side's correctness arguments as declared on
    disk), translated_run (see below), target_oracle (the target spec's
    positive strategies), and negative_checks (the target spec's declared
    exclude patterns).

    translated_run.args carries the arguments of the program that actually
    runs (OD-3 follow-up, owner ruling 2026-08-21): for a KERNEL-ONLY pair
    (all translation_targets end in .cl) the target-native host binary runs,
    so the TARGET spec's correctness args apply; for a full-program pair the
    model keeps source argument parsing, so the SOURCE spec's args apply.
    Before this fix every pair got source args, which broke kernel-only
    targets whose host demands its own argument shape (streamcluster: host
    checks argc<11, source declares 9 args - every future run would have
    been a spurious RUN_FAIL).

    The Linux witness is deliberately NOT populated: baseline_witness stays
    {"status": "unresolved"} and comparability stays "unresolved", so
    pair_contracts.require_resolved() keeps rejecting every pair until
    Task 7 attaches real witness evidence from the source-complete host.
    """
    index = _spec_index(spec_root)
    populated_pairs: list[dict] = []
    for pair in candidates["pairs"]:
        src = index.get(pair["source_spec"])
        tgt = index.get(pair["target_spec"])
        if src is None or tgt is None:
            missing = pair["source_spec"] if src is None else pair["target_spec"]
            raise ValueError(f"candidate pair references unknown spec: {missing}")
        positive, negative = _pair_oracle_strategies(tgt)
        entry = {
            "source_spec": pair["source_spec"],
            "target_spec": pair["target_spec"],
            "shared_input": {
                "source_args": _correctness_args(src),
                "target_args": _correctness_args(tgt),
            },
            "translated_run": {
                "args": _correctness_args(
                    tgt if _is_kernel_only_translation(tgt) else src
                )
            },
            "target_oracle": {"strategies": positive},
            "baseline_witness": {"status": "unresolved"},
            "comparability": "unresolved",
        }
        if negative:
            entry["negative_checks"] = negative
        populated_pairs.append(entry)
    populated_pairs.sort(key=lambda e: (e["source_spec"], e["target_spec"]))
    return {
        # Every regeneration stamps the CURRENT version, never the frozen
        # v1 string, so a new registry is always distinguishable from the
        # sealed one at run time.
        "contract_version": CONTRACT_VERSION_CURRENT,
        "pairs": populated_pairs,
    }


WITNESS_EVIDENCE = "results/analysis/pair_witnesses.json"


def finalize_pairs(
    candidates: dict,
    witnesses: dict,
    determinism: dict,
) -> tuple[list[dict], list[dict]]:
    """Task 7: split the populated candidates into (final resolved pairs,
    disposition rows) from the generated Linux evidence.

    A pair becomes eligible only when its witness row passed (both pristine
    baselines PASS their own declared oracles, neither side excluded or
    performance-only) AND both sides' baseline-determinism repetitions are
    stable. Every other pair lands in the disposition table with its
    generated reasons - failing pairs are moved, never forced eligible.
    """
    witness_rows = {
        (r["source_spec"], r["target_spec"]): r for r in witnesses["pairs"]
    }
    det_specs = determinism.get("specs") or {}
    final: list[dict] = []
    dispositions: list[dict] = []
    for pair in candidates["pairs"]:
        key = (pair["source_spec"], pair["target_spec"])
        row = witness_rows.get(key)
        reasons: list[str] = []
        if row is None:
            reasons.append("no_witness_row")
        elif row["witness_status"] != "pass":
            reasons.extend(row["reasons"])
        else:
            for side, uid in (("source", key[0]), ("target", key[1])):
                det = det_specs.get(uid)
                if det is None:
                    reasons.append(f"{side}_determinism_not_measured")
                elif not det.get("stable"):
                    reasons.append(f"{side}_baseline_nondeterministic")
        if reasons:
            dispositions.append({
                "source_spec": key[0],
                "target_spec": key[1],
                "disposition": "ineligible",
                "reasons": sorted(set(reasons)),
            })
            continue
        entry = dict(pair)
        entry["baseline_witness"] = {
            "status": "pass",
            "evidence": WITNESS_EVIDENCE,
        }
        entry["comparator"] = {
            "type": "declared-strategy",
            "evidence": WITNESS_EVIDENCE,
        }
        entry["comparability"] = "eligible"
        final.append(entry)
    final.sort(key=lambda e: (e["source_spec"], e["target_spec"]))
    dispositions.sort(key=lambda e: (e["source_spec"], e["target_spec"]))
    return final, dispositions


def _frozen_artifact_paths() -> set[Path]:
    """Resolved paths hash-bound by the Task 7 contract freeze.

    config/final_contract_manifest.json records the sha256 of every artifact
    of the frozen v1 generation; those files back the sealed NeurIPS
    evidence package and must never be rewritten by a regeneration.
    """
    manifest_path = PROJECT_ROOT / "config" / "final_contract_manifest.json"
    if not manifest_path.is_file():
        return set()
    manifest = json.loads(manifest_path.read_text())
    frozen: set[Path] = set()

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            if "sha256" in node and isinstance(node.get("path"), str):
                frozen.add((PROJECT_ROOT / node["path"]).resolve())
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for value in node:
                _collect(value)

    _collect(manifest)
    return frozen


def write_registry(entries: list[dict], output: Path) -> None:
    if output.resolve() in _frozen_artifact_paths():
        raise SystemExit(
            f"refusing to write {output}: its bytes are hash-bound by "
            f"config/final_contract_manifest.json (frozen v1 generation "
            f"backing the sealed evidence package). Write a new versioned "
            f"file instead."
        )
    payload = {"contract_version": CONTRACT_VERSION_CURRENT, "pairs": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    # Self-check: the file we just wrote must load under the fail-closed
    # loader (schema validation + duplicate-key detection).
    load_pair_contracts(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the pair-contract registry "
                    "(Task 2: candidates; Task 4: spec-side population).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--enumerate-candidates", action="store_true",
                      help="Enumerate candidate directed pairs from the spec corpus.")
    mode.add_argument("--populate-from-specs", action="store_true",
                      help="Task 4: populate spec-side data into an enumerated "
                           "candidate registry; Linux witness stays unresolved "
                           "so the runner rejects every pair until Task 7.")
    mode.add_argument("--finalize", action="store_true",
                      help="Task 7: write the production registry from the "
                           "generated Linux witness + determinism evidence; "
                           "ineligible pairs move to the disposition table.")
    parser.add_argument("--witnesses", type=Path,
                        default=PROJECT_ROOT / "results" / "analysis"
                        / "pair_witnesses.json",
                        help="Generated pair-witness report (--finalize).")
    parser.add_argument("--determinism", type=Path,
                        default=PROJECT_ROOT / "results" / "analysis"
                        / "baseline_determinism.json",
                        help="Generated determinism report (--finalize).")
    parser.add_argument("--dispositions", type=Path,
                        default=PROJECT_ROOT / "results" / "analysis"
                        / "pair_contract_dispositions_v2.json",
                        help="Disposition table output (--finalize). The v1 "
                             "path is frozen (hash-bound) and refused.")
    parser.add_argument("--candidate-registry", type=Path,
                        default=PROJECT_ROOT / "config" / "pair_contract_candidates.json",
                        help="Enumerated candidate registry to populate "
                             "(--populate-from-specs only).")
    parser.add_argument("--spec-root", type=Path, default=PROJECT_ROOT / "specs",
                        help="Directory holding the kernel spec JSONs (default: specs/).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Registry file to write (e.g. config/pair_contract_candidates.json).")
    args = parser.parse_args(argv)

    if not args.spec_root.is_dir():
        print(f"error: spec root not found: {args.spec_root}", file=sys.stderr)
        return 2

    if args.enumerate_candidates:
        entries = enumerate_candidate_pairs(args.spec_root)
        write_registry(entries, args.output)
        print(f"Enumerated {len(entries)} candidate directed pair(s) from "
              f"{args.spec_root} -> {args.output} (all unresolved; eligibility "
              f"is certified in Task 7, never here)")
        return 0

    if not args.candidate_registry.is_file():
        print(f"error: candidate registry not found: {args.candidate_registry}",
              file=sys.stderr)
        return 2
    candidates = json.loads(args.candidate_registry.read_text())

    if args.finalize:
        for label, path in (("witnesses", args.witnesses),
                            ("determinism", args.determinism)):
            if not path.is_file():
                print(f"error: {label} report not found: {path}",
                      file=sys.stderr)
                return 2
        witnesses = json.loads(args.witnesses.read_text())
        determinism = json.loads(args.determinism.read_text())
        for label, doc, path in (("witness", witnesses, args.witnesses),
                                 ("determinism", determinism,
                                  args.determinism)):
            if not doc.get("complete"):
                print(f"error: {label} report {path} is not complete; "
                      f"finish its campaign before finalizing",
                      file=sys.stderr)
                return 2
        final, dispositions = finalize_pairs(candidates, witnesses,
                                             determinism)
        if not final:
            print("error: no pair is eligible; refusing to write an empty "
                  "production registry", file=sys.stderr)
            return 2
        write_registry(final, args.output)
        registry = load_pair_contracts(args.output)
        for contract in registry.contracts():
            contract.require_resolved()  # every production entry resolved
        if args.dispositions.resolve() in _frozen_artifact_paths():
            raise SystemExit(
                f"refusing to write {args.dispositions}: its bytes are "
                f"hash-bound by config/final_contract_manifest.json (frozen "
                f"v1 generation). Use a new versioned file."
            )
        args.dispositions.parent.mkdir(parents=True, exist_ok=True)
        args.dispositions.write_text(json.dumps({
            "schema": "pair-contract-dispositions-v1",
            "generated_by": "scripts/spec_tools/build_pair_contracts.py "
                            "--finalize",
            "candidate_pairs": len(candidates["pairs"]),
            "eligible_pairs": len(final),
            "ineligible_pairs": len(dispositions),
            "dispositions": dispositions,
        }, indent=2) + "\n")
        print(f"Finalized {len(final)} eligible pair(s) -> {args.output}; "
              f"{len(dispositions)} ineligible pair(s) with reasons -> "
              f"{args.dispositions}")
        return 0

    populated = populate_from_specs(candidates, args.spec_root)
    write_registry(populated["pairs"], args.output)
    print(f"Populated {len(populated['pairs'])} pair(s) from {args.spec_root} "
          f"-> {args.output} (spec-side data only; every baseline_witness "
          f"remains unresolved until Task 7's Linux witnesses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
