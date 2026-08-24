#!/usr/bin/env python3
"""Freeze the final ParBench contract (Task 7, E-02/E-03/E-12).

Sole writer of ``config/final_contract_manifest.json``. The freeze validates
the schema, hashes, and completion status of every input artifact and
REJECTS an unresolved pair or a missing artifact - it never merely hashes
what it was given:

- source-complete host report (``complete: true`` required),
- pair-witness report (complete, and covering every final pair),
- baseline-determinism report (complete; every final-pair spec stable),
- thread-provisioning report (no red outcome on a final-pair spec),
- blind-control preregistration + blinded labels + unblinding record +
  result summary (one sealed pool hash across all four; every preregistered
  control has a recorded outcome - misses stay recorded, never repaired),
- oracle contract report (every spec dispositioned),
- the canonical exclusion set (imported live from ``harness.constants``),
- the production pair registry (loads fail-closed; every pair resolved;
  every candidate pair accounted for in the registry or the generated
  disposition table).

The manifest records SHA-256 hashes of every input artifact, every spec,
the harness code, the toolchain versions, and the final eligible task list.
Its content is deterministic (no timestamp), so a rerun over unchanged
inputs is byte-identical and the manifest's own SHA-256 is stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.constants import (  # noqa: E402
    CORRECTNESS_INELIGIBLE_SPECS,
    EXCLUDED_SPECS,
    PERFORMANCE_ONLY_SPECS,
)
from scripts.evaluation.pair_contracts import (  # noqa: E402
    PairContractError,
    load_pair_contracts,
)

MANIFEST_VERSION = "parbench-final-contract-v1"

ARTIFACTS = {
    "host_report": "results/analysis/source_complete_host.json",
    "pair_witnesses": "results/analysis/pair_witnesses.json",
    "baseline_determinism": "results/analysis/baseline_determinism.json",
    "thread_provisioning": "results/analysis/thread_provisioning.json",
    "blind_control_preregistration":
        "results/provenance/blind_control_prereg_sealed.json",
    "blind_control_labels": "results/provenance/blind_control_labels.json",
    "blind_control_unblinding":
        "results/provenance/blind_control_unblinding.json",
    "blind_control_result": "results/analysis/blind_control_result.json",
    "oracle_report": "results/analysis/oracle_contract_report.json",
    "pair_dispositions": "results/analysis/pair_contract_dispositions.json",
    "candidate_registry": "config/pair_contract_candidates_populated.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, problems: list[str], label: str) -> dict | None:
    if not path.is_file():
        problems.append(f"missing artifact: {label} ({path})")
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"unreadable artifact: {label} ({path}): {exc}")
        return None


def _require(cond: bool, problems: list[str], message: str) -> None:
    if not cond:
        problems.append(message)


def validate_inputs(
    root: Path, registry_path: Path
) -> tuple[list[str], dict[str, Any]]:
    """Validate every freeze input; return (problems, manifest fields)."""
    problems: list[str] = []
    docs: dict[str, dict] = {}
    for label, rel in ARTIFACTS.items():
        doc = _load(root / rel, problems, label)
        if doc is not None:
            docs[label] = doc

    fields: dict[str, Any] = {}
    if problems:
        return problems, fields

    host = docs["host_report"]
    _require(host.get("schema") == "source-complete-host-v1", problems,
             "host report: unexpected schema")
    _require(host.get("complete") is True, problems,
             "host report: host is not source-complete")

    witnesses = docs["pair_witnesses"]
    _require(witnesses.get("schema") == "pair-witnesses-v1", problems,
             "pair witnesses: unexpected schema")
    _require(witnesses.get("complete") is True, problems,
             "pair witnesses: campaign incomplete")
    witness_rows = {
        (r["source_spec"], r["target_spec"]): r
        for r in witnesses.get("pairs", [])
    }

    determinism = docs["baseline_determinism"]
    _require(determinism.get("schema") == "baseline-determinism-v1", problems,
             "determinism report: unexpected schema")
    _require(determinism.get("complete") is True, problems,
             "determinism report: campaign incomplete")
    det_specs = determinism.get("specs") or {}

    provisioning = docs["thread_provisioning"]
    _require(provisioning.get("schema") == "thread-provisioning-v1", problems,
             "thread-provisioning report: unexpected schema")
    _require(provisioning.get("complete") is True, problems,
             "thread-provisioning report: incomplete")

    prereg = docs["blind_control_preregistration"]
    sealed_sha = prereg.get("sealed_sha256")
    count = prereg.get("count")
    _require(bool(sealed_sha) and isinstance(count, int) and count > 0,
             problems, "blind-control preregistration: malformed")
    result = docs["blind_control_result"]
    labels = docs["blind_control_labels"]
    unblind = docs["blind_control_unblinding"]
    for label, doc in (("result", result), ("labels", labels),
                       ("unblinding", unblind)):
        _require(doc.get("sealed_sha256") == sealed_sha, problems,
                 f"blind-control {label}: sealed hash does not match the "
                 f"preregistration")
    outcomes = result.get("controls", [])
    _require(len(outcomes) == count, problems,
             f"blind-control result: {len(outcomes)} outcomes for "
             f"{count} preregistered controls")
    _require(len(labels.get("labels", [])) == count, problems,
             "blind-control labels: incomplete")
    _require(len(unblind.get("controls", [])) == count, problems,
             "blind-control unblinding: incomplete")
    _require(all(o.get("outcome") for o in outcomes), problems,
             "blind-control result: an outcome is missing")
    # Control IDs must be unique and consistent across the three artifacts
    # (T7 gate finding, 2026-08-12: the freeze previously trusted counts
    # alone, so duplicated or mismatched control IDs passed silently).
    for label, doc, key in (("result", result, "controls"),
                            ("labels", labels, "labels"),
                            ("unblinding", unblind, "controls")):
        ids = [c.get("control_id") or c.get("id") for c in doc.get(key, [])]
        _require(all(ids) and len(set(ids)) == len(ids), problems,
                 f"blind-control {label}: control IDs missing or duplicated")
    id_sets = [
        {c.get("control_id") or c.get("id") for c in result.get("controls", [])},
        {c.get("control_id") or c.get("id") for c in labels.get("labels", [])},
        {c.get("control_id") or c.get("id") for c in unblind.get("controls", [])},
    ]
    _require(id_sets[0] == id_sets[1] == id_sets[2], problems,
             "blind-control artifacts: control ID sets differ across "
             "result/labels/unblinding")

    oracle = docs["oracle_report"]
    o_summary = oracle.get("summary") or {}
    _require(
        o_summary.get("eligible_specs") == o_summary.get("total_specs"),
        problems, "oracle report: not every spec is eligible/dispositioned")

    # The production registry: fail-closed load, every pair resolved.
    final_pairs: list[tuple[str, str]] = []
    try:
        registry = load_pair_contracts(registry_path)
        if len(registry) == 0:
            problems.append("production registry declares no pairs")
        for contract in registry.contracts():
            contract.require_resolved()
            key = (contract.source_spec, contract.target_spec)
            final_pairs.append(key)
            for uid in key:
                if uid in CORRECTNESS_INELIGIBLE_SPECS:
                    problems.append(
                        f"final pair includes correctness-ineligible spec "
                        f"{uid}")
            row = witness_rows.get(key)
            if row is None or row["witness_status"] != "pass":
                problems.append(
                    f"final pair {key[0]} -> {key[1]} has no passing "
                    f"witness row")
            for uid in key:
                det = det_specs.get(uid)
                if det is None or not det.get("stable"):
                    problems.append(
                        f"final pair spec {uid} has no stable "
                        f"determinism result")
    except PairContractError as exc:
        problems.append(f"production registry rejected: {exc}")

    # Completeness: every candidate pair is finalized or dispositioned.
    cand_keys = {
        (p["source_spec"], p["target_spec"])
        for p in docs["candidate_registry"].get("pairs", [])
    }
    disp = docs["pair_dispositions"]
    disp_keys = {
        (p["source_spec"], p["target_spec"])
        for p in disp.get("dispositions", [])
    }
    _require(all(p.get("reasons") for p in disp.get("dispositions", [])),
             problems, "disposition table: an entry has no reasons")
    final_set = set(final_pairs)
    for key in sorted(cand_keys - final_set - disp_keys):
        problems.append(f"candidate pair {key[0]} -> {key[1]} is neither "
                        f"finalized nor dispositioned")
    for key in sorted(final_set & disp_keys):
        problems.append(f"pair {key[0]} -> {key[1]} is both finalized and "
                        f"dispositioned")
    # No red thread-provisioning outcome may involve a final-pair spec.
    final_specs = {uid for key in final_pairs for uid in key}
    for uid, entry in (provisioning.get("specs") or {}).items():
        if (entry.get("outcome") in ("provisioned_baseline_red",
                                     "build_failed")
                and uid in final_specs):
            problems.append(
                f"thread-provisioning red outcome on final-pair spec {uid}")

    fields["blind_controls"] = {
        "sealed_sha256": sealed_sha,
        "count": count,
        "summary": result.get("summary"),
    }
    fields["final_pairs"] = sorted(final_pairs)
    return problems, fields


def toolchain_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for tool in ("nvcc", "nvc++", "gcc"):
        path = shutil.which(tool)
        if path is None:
            out[tool] = None
            continue
        try:
            r = subprocess.run([tool, "--version"], capture_output=True,
                               text=True, timeout=30)
            out[tool] = (r.stdout or r.stderr).strip().splitlines()[0]
        except (OSError, subprocess.TimeoutExpired):
            out[tool] = None
    return out


def aggregate_hash(files: dict[str, str]) -> str:
    canon = json.dumps(sorted(files.items()), separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--pair-contracts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    problems, fields = validate_inputs(root, args.pair_contracts)
    toolchains = toolchain_versions()
    for tool, version in toolchains.items():
        if version is None:
            problems.append(
                f"toolchain {tool} is not invocable on this host (is the "
                f"HPC SDK compilers directory on PATH?)")
    if problems:
        for p in problems:
            print(f"FREEZE REJECTED: {p}", file=sys.stderr)
        print(f"NOT FROZEN: {len(problems)} problem(s); no manifest written",
              file=sys.stderr)
        return 1

    spec_hashes = {
        p.name: _sha256(p) for p in sorted((root / "specs").glob("*.json"))
    }
    harness_hashes = {
        p.name: _sha256(p) for p in sorted((root / "harness").glob("*.py"))
    }
    artifact_hashes = {
        label: {"path": rel, "sha256": _sha256(root / rel)}
        for label, rel in sorted(ARTIFACTS.items())
    }
    registry_rel = str(args.pair_contracts.resolve().relative_to(root))
    task_list = [
        {"source_spec": s, "target_spec": t} for s, t in fields["final_pairs"]
    ]
    task_list_sha256 = hashlib.sha256(
        json.dumps(task_list, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_by": "scripts/spec_tools/freeze_final_contract.py",
        "project_root_note": "content-deterministic: no timestamp; rerun "
                             "over unchanged inputs is byte-identical",
        "exclusions": {
            "excluded_specs": sorted(EXCLUDED_SPECS),
            "performance_only_specs": sorted(PERFORMANCE_ONLY_SPECS),
            "correctness_ineligible_specs":
                sorted(CORRECTNESS_INELIGIBLE_SPECS),
            # Aggregate over the three sorted lists so a one-entry edit is a
            # visible hash change (T7 gate finding, 2026-08-12). Absent from
            # the immutable 2026-08-11 manifest; future freezes carry it.
            "aggregate_sha256": hashlib.sha256(json.dumps(
                {
                    "excluded_specs": sorted(EXCLUDED_SPECS),
                    "performance_only_specs": sorted(PERFORMANCE_ONLY_SPECS),
                    "correctness_ineligible_specs":
                        sorted(CORRECTNESS_INELIGIBLE_SPECS),
                },
                separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        },
        "input_artifacts": artifact_hashes,
        "pair_registry": {
            "path": registry_rel,
            "sha256": _sha256(args.pair_contracts),
            "pairs": len(fields["final_pairs"]),
        },
        "blind_controls": fields["blind_controls"],
        "specs": {
            "count": len(spec_hashes),
            "aggregate_sha256": aggregate_hash(spec_hashes),
            "files": spec_hashes,
        },
        "harness": {
            "count": len(harness_hashes),
            "aggregate_sha256": aggregate_hash(harness_hashes),
            "files": harness_hashes,
        },
        "toolchains": toolchains,
        "eligible_task_list": {
            "count": len(task_list),
            "sha256": task_list_sha256,
            "tasks": task_list,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FROZEN: {len(task_list)} eligible task(s), task-list SHA-256 "
          f"{task_list_sha256}")
    print(f"manifest -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
