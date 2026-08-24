#!/usr/bin/env python3
"""Check run arguments against the source's literal argc check.

Rules 13+14d merged package (ruled by the owner 2026-08-15). Doc-based "fixes" to
run args silently broke rodinia-nw-omp (needle.cpp:249, argc==4) and
rodinia-hotspot-omp (hotspot_openmp.cpp:282, argc!=8) for weeks; the pair-contract
gate checks contract-vs-spec agreement only, never args-vs-source. This script is
the args-vs-source check, in two passes over the same argc-parsing machinery.

Per-spec pass (--spec / --all): read the sources named in files.prompt_payload,
find literal `argc <op> N` checks, and compare against
len(run.input_configurations.correctness.arguments) + 1.

Pair-contract pass (--contracts, OD-3 ruled 2026-08-20): for each resolved pair
contract, validate translated_run.args against the argc checks of the tree that
actually runs. For full-program pairs the model keeps source argument parsing,
so the SOURCE tree's parser governs; for kernel-only pairs (all
translation_targets end in .cl; predicate _is_kernel_only_translation in
scripts/evaluation/llm_evaluate.py) the target-native host binary runs, so the
TARGET tree's parser governs. Since 2026-08-21 the generator emits args per
that same rule, and the audit default is the CURRENT registry
config/pair_contracts_v2.json (the frozen final_pair_contracts.json keeps the
old all-source-args shape by design; it backs the sealed evidence package).
Same argc machinery for both passes, governing spec chosen per pair.

Outcomes (both passes):

  OK                    every literal argc check is satisfied
  MISMATCH              a literal argc check fails (exit 1)
  SKIP (flag-parsed)    an argument starts with '-'; arg-count reasoning does
                        not apply to flag-parsed programs
  SKIP (no-argc)        no literal argc comparison in any payload source
  SKIP (no-args)        no arguments are declared to check
  SKIP (source-missing) a payload source is absent on this machine
  SPEC-MISSING          (contracts pass) a pair references an absent spec -
                        an integrity error, fails the run (exit 1)

Every SKIP list prints in full at the end: it is a manual-review surface.
Comparison semantics assume the idiomatic error guard: == and != demand an exact
count; `argc < N` demands at least N; `argc > N` demands at most N.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from scripts.evaluation.pair_contracts import PairContract

_REPO_ROOT = Path(__file__).resolve().parents[2]

ARGC_RE = re.compile(r"\bargc\s*(==|!=|<=|>=|<|>)\s*(\d+)")

# Real mismatches found by this checker and awaiting an owner ruling. Each is
# printed loudly but does not fail the run - the same explained-errors pattern
# health_check's schema baseline uses. Identity is pinned per entry (spec id
# AND the exact detail string), so a DIFFERENT future mismatch on the same
# spec is a real failure, not the known one (2026-08-16 review finding).
# Under --all, an entry that never fires is stale (the spec was fixed or
# changed) and fails the run until the entry is removed.
KNOWN_MISMATCHES: dict[str, dict[str, str]] = {
    # Empty since 2026-08-16: the one entry (hecbench-myocyte-omp, found
    # 2026-08-15) was resolved by the owner's ruling - the spec now passes the
    # source's required '-time <n>' flag form. History: commit 80f74384.
}

# Contracts-pass allowlist, same explained-errors pattern, keyed by the
# directed pair id "source -> target" and pinned to the exact detail string
# (2026-08-21 Codex architecture review, item d): a clean corpus PASSes, an
# allowlisted finding WARNs in health_check, and a NEW finding or a stale
# entry fails the run.
# Known limit (shared with KNOWN_MISMATCHES): classification reports the
# FIRST failing argc check per pair, so a second, different defect behind a
# known one surfaces only after the first is fixed - at which point it fires
# as a new mismatch, never absorbed.
KNOWN_CONTRACT_ARGC_MISMATCHES: dict[str, dict[str, str]] = {
    # Empty since 2026-08-21: the two streamcluster -> opencl entries (found
    # the same day by this gate) were resolved by the owner's ruling - the
    # generator now emits TARGET args for kernel-only pairs and the current
    # registry is config/pair_contracts_v2.json. The frozen
    # config/final_pair_contracts.json keeps the old args BY DESIGN: its
    # bytes are hash-bound into the sealed evidence package and it backs the
    # NeurIPS artifact, so it is never regenerated.
}

# Test seam: the mechanism tests shell out to this script, so they inject a
# synthetic entry here instead of depending on a real corpus defect existing.
_override = os.environ.get("CHECK_SPEC_ARGC_KNOWN_JSON")
if _override:
    KNOWN_MISMATCHES = json.loads(_override)
_contract_override = os.environ.get("CHECK_CONTRACT_ARGC_KNOWN_JSON")
if _contract_override:
    KNOWN_CONTRACT_ARGC_MISMATCHES = json.loads(_contract_override)


def check_argc(op: str, n: int, argc: int) -> bool:
    """True when `argc` is consistent with the literal check `argc <op> N`.

    == and != are direction-free: whichever branch is the error, the program
    expects exactly N. < and <= are the universal minimum-arg error guard
    (`if (argc < N) usage()`). > and >= are NOT enforced: in this corpus they
    guard optional arguments (`if (argc > 1) { read optional }`), so treating
    them as error guards false-flags 12 healthy specs. Asymmetric tolerance:
    a false negative is fine, a false positive is not.
    """
    if op in ("==", "!="):
        return argc == n
    if op == "<":
        return argc >= n
    if op == "<=":
        return argc > n
    return True  # > and >=: optional-argument idiom, non-definitive


def classify_args_against_source(
    arguments: list, spec: dict, project_root: Path, *, subject: str = "spec provides"
) -> tuple[str, str]:
    """Classify a run-argument list against a spec's source argc checks.

    Shared by the per-spec pass and the pair-contract pass: it owns the
    flag-parse skip, the source read (binary-safe for non-ISO myocyte
    sources), the literal `argc <op> N` scan, and the comparison. ``subject``
    only shapes the MISMATCH wording ("spec provides" vs "contract declares");
    the per-spec wording is preserved verbatim because the KNOWN_MISMATCHES
    allowlist pins mismatch identity on the exact detail string.
    """
    if any(str(a).startswith("-") for a in arguments):
        return "SKIP (flag-parsed)", f"args {arguments}"

    prov = spec.get("provenance", {})
    src_dir = project_root / prov.get("repo_root", "") / prov.get("source_path", "")
    payload = spec.get("files", {}).get("prompt_payload", [])
    argc = len(arguments) + 1

    checks: list[tuple[str, str, int]] = []
    for name in payload:
        src = src_dir / name
        if not src.is_file():
            return "SKIP (source-missing)", str(src)
        # Non-ISO extended-ASCII sources (myocyte) raise on a strict UTF-8 read.
        text = src.read_bytes().decode("utf-8", errors="replace")
        checks.extend((name, m.group(1), int(m.group(2))) for m in ARGC_RE.finditer(text))

    if not checks:
        return "SKIP (no-argc)", "no literal argc comparison"
    for name, op, n in checks:
        if not check_argc(op, n, argc):
            return "MISMATCH", (
                f"{name} checks `argc {op} {n}` but {subject} "
                f"{len(arguments)} arguments (argc={argc})"
            )
    return "OK", f"argc={argc} satisfies {len(checks)} check(s)"


def check_spec(spec_path: Path, project_root: Path) -> tuple[str, str]:
    """Return (outcome, detail). Outcome is OK, MISMATCH, or a SKIP reason."""
    spec = json.loads(spec_path.read_text())
    run = spec.get("run", {})
    correctness = run.get("input_configurations", {}).get("correctness")
    if not correctness or "arguments" not in correctness:
        return "SKIP (no-args)", "no correctness arguments"
    return classify_args_against_source(correctness["arguments"], spec, project_root)


def _load_spec_index(spec_root: Path) -> tuple[dict[str, dict], list[str]]:
    """Map unique_id -> spec dict for every JSON in ``spec_root``.

    Returns (index, problems). A malformed body, a missing
    ``identity.unique_id``, or a duplicate id is a problem line, never a
    silent drop (2026-08-21 Codex finding: silent drops turned integrity
    errors into successful skips).
    """
    index: dict[str, dict] = {}
    problems: list[str] = []
    for spec_path in sorted(spec_root.glob("*.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"malformed spec JSON {spec_path.name}: {exc}")
            continue
        uid = (spec.get("identity") or {}).get("unique_id")
        if not uid:
            problems.append(f"spec {spec_path.name} has no identity.unique_id")
        elif uid in index:
            problems.append(f"duplicate unique_id {uid} ({spec_path.name})")
        else:
            index[uid] = spec
    return index, problems


def check_contract(
    contract: PairContract,
    spec_index: dict[str, dict],
    is_kernel_only: Callable[[dict], bool],
    project_root: Path,
) -> tuple[str, str]:
    """Validate one pair contract's translated_run.args against the parser of
    the tree that actually runs.

    Governing tree (OD-3 semantics): kernel-only pairs run the target-native
    host binary, so the TARGET spec's source governs; full-program pairs keep
    source argument parsing, so the SOURCE spec's source governs. ``contract``
    is a scripts.evaluation.pair_contracts.PairContract; ``is_kernel_only`` is
    scripts.evaluation.llm_evaluate._is_kernel_only_translation (the single
    source of truth for the predicate).
    """
    src = spec_index.get(contract.source_spec)
    tgt = spec_index.get(contract.target_spec)
    if src is None or tgt is None:
        missing = contract.source_spec if src is None else contract.target_spec
        return "SPEC-MISSING", f"spec {missing} not in spec root"
    translated_run = contract.translated_run
    if "args" not in translated_run:
        return "SKIP (no-args)", "contract declares no translated_run.args"

    kernel_only = is_kernel_only(tgt)
    if kernel_only:
        governing, side, gov_id = tgt, "target", contract.target_spec
    else:
        governing, side, gov_id = src, "source", contract.source_spec
    outcome, detail = classify_args_against_source(
        translated_run["args"], governing, project_root, subject="contract declares"
    )
    kind = "kernel_only" if kernel_only else "full_program"
    return outcome, f"governed by {side} {gov_id} ({kind}): {detail}"


def run_contracts_pass(
    registry_path: Path, spec_root: Path, project_root: Path
) -> int:
    """Check every pair contract's translated_run.args against its governing
    tree's parser.

    Exit codes: 0 clean or only allowlisted known mismatches; 1 on a NEW
    mismatch, a stale allowlist entry, a contract-referenced missing spec, or
    a spec-index integrity problem; 2 when the checker itself cannot run
    (unloadable registry, absent spec root). Report-only toward the corpus:
    real mismatches are findings for the owner, never a signal to edit a spec
    or contract. Imports are lazy so the per-spec pass stays free of the eval
    stack.
    """
    # Import the eval stack from THIS repo, not from --project-root (which may
    # point at a synthetic tree in a test); project_root only resolves the
    # benchmark source dirs the argc scan reads.
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.evaluation.llm_evaluate import (  # noqa: E402
        _is_kernel_only_translation,
    )
    from scripts.evaluation.pair_contracts import (  # noqa: E402
        PairContractError,
        load_pair_contracts,
    )

    try:
        registry = load_pair_contracts(registry_path)
    except PairContractError as exc:
        print(f"FAIL: cannot load pair-contract registry {registry_path}: {exc}")
        return 2
    if not spec_root.is_dir():
        print(f"FAIL: spec root {spec_root} is not a directory")
        return 2

    spec_index, index_problems = _load_spec_index(spec_root)
    for problem in index_problems:
        print(f"SPEC-INDEX {problem}")

    mismatches = 0
    known = 0
    missing = 0
    fired: set[str] = set()
    skips: dict[str, list[str]] = {}
    for contract in registry.contracts():
        outcome, detail = check_contract(
            contract, spec_index, _is_kernel_only_translation, project_root
        )
        pair_id = f"{contract.source_spec} -> {contract.target_spec}"
        entry = KNOWN_CONTRACT_ARGC_MISMATCHES.get(pair_id)
        if outcome == "MISMATCH" and entry is not None:
            if detail == entry["detail"]:
                known += 1
                fired.add(pair_id)
                print(f"MISMATCH (known) {pair_id}: {detail}")
                print(f"  ruling pending: {entry['note']}")
                continue
            # Same pair, DIFFERENT defect: the allowlist must not absorb it.
            print(f"MISMATCH {pair_id}: {detail}")
            print(f"  note: differs from the recorded known mismatch "
                  f"({entry['detail']}) - treating as a new defect")
            mismatches += 1
            continue
        print(f"{outcome} {pair_id}: {detail}")
        if outcome == "MISMATCH":
            mismatches += 1
        elif outcome == "SPEC-MISSING":
            missing += 1
        elif outcome.startswith("SKIP"):
            skips.setdefault(outcome, []).append(pair_id)

    stale = sorted(set(KNOWN_CONTRACT_ARGC_MISMATCHES) - fired)
    for pair_id in stale:
        print(f"STALE known-mismatch entry: {pair_id} no longer mismatches - "
              f"remove its KNOWN_CONTRACT_ARGC_MISMATCHES entry")

    for reason in sorted(skips):
        ids = skips[reason]
        print(f"\n{reason}: {len(ids)} pair(s)")
        for pair_id in ids:
            print(f"  {pair_id}")
    skipped = sum(len(v) for v in skips.values())
    checked = len(registry) - skipped - missing
    print(f"\ncontracts_checked={checked} mismatches={mismatches} "
          f"known_mismatches={known} stale_known={len(stale)} "
          f"spec_missing={missing} index_problems={len(index_problems)} "
          f"skipped={skipped}")
    return 1 if (mismatches or stale or missing or index_problems) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check run arguments against source argc checks "
                    "(per-spec and pair-contract passes)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="check every spec in specs/")
    group.add_argument("--spec", type=Path, help="check a single spec JSON")
    group.add_argument(
        "--contracts", action="store_true",
        help="check every pair contract's translated_run.args against the "
             "governing tree's parser (OD-3)",
    )
    parser.add_argument(
        "--registry", type=Path, default=None,
        help="pair-contract registry for --contracts "
             "(default: <project-root>/config/pair_contracts_v2.json, the "
             "CURRENT registry; the frozen final_pair_contracts.json backs "
             "the sealed artifact and is not the audit target)",
    )
    parser.add_argument(
        "--spec-root", type=Path, default=None,
        help="specs directory for --contracts (default: <project-root>/specs)",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=_REPO_ROOT,
        help="root against which repo_root paths resolve (default: repo root)",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()

    if args.contracts:
        registry_path = args.registry or (root / "config" / "pair_contracts_v2.json")
        spec_root = args.spec_root or (root / "specs")
        return run_contracts_pass(registry_path, spec_root, root)

    if args.all:
        spec_paths = sorted((root / "specs").glob("*.json"))
    else:
        spec_paths = [args.spec]

    mismatches = 0
    known = 0
    fired: set[str] = set()
    skips: dict[str, list[str]] = {}
    for spec_path in spec_paths:
        outcome, detail = check_spec(spec_path, root)
        spec_id = spec_path.stem
        entry = KNOWN_MISMATCHES.get(spec_id)
        if outcome == "MISMATCH" and entry is not None:
            if detail == entry["detail"]:
                known += 1
                fired.add(spec_id)
                print(f"MISMATCH (known) {spec_id}: {detail}")
                print(f"  ruling pending: {entry['note']}")
                continue
            # Same spec, DIFFERENT defect: the allowlist must not absorb it.
            print(f"MISMATCH {spec_id}: {detail}")
            print(f"  note: differs from the recorded known mismatch "
                  f"({entry['detail']}) - treating as a new defect")
            mismatches += 1
            continue
        print(f"{outcome} {spec_id}: {detail}")
        if outcome == "MISMATCH":
            mismatches += 1
        elif outcome.startswith("SKIP"):
            skips.setdefault(outcome, []).append(spec_id)

    stale = sorted(set(KNOWN_MISMATCHES) - fired) if args.all else []
    for spec_id in stale:
        print(f"STALE known-mismatch entry: {spec_id} no longer mismatches - "
              f"remove its KNOWN_MISMATCHES entry")

    # The SKIP lists are a visible manual-review surface; print them in full.
    for reason in sorted(skips):
        ids = skips[reason]
        print(f"\n{reason}: {len(ids)} spec(s)")
        for spec_id in ids:
            print(f"  {spec_id}")
    checked = len(spec_paths) - sum(len(v) for v in skips.values())
    print(f"\nchecked={checked} mismatches={mismatches} known_mismatches={known} "
          f"stale_known={len(stale)} skipped={sum(len(v) for v in skips.values())}")
    return 1 if (mismatches or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
