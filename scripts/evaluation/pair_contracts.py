#!/usr/bin/env python3
"""Pair-contract registry: loader and fail-closed validation (Task 2, gate E-02).

A pair contract declares, for one directed cross-API translation pair, the
shared input under which the pair is comparable, the run configuration for
the translated binary, the target-side oracle and negative checks, the
independent comparator, and the baseline witness evidence. The registry file
validates against ``schema/pair_contract_schema.json``.

Everything here fails closed: a missing file, malformed JSON, schema
violation, duplicate pair key, missing entry, or unresolved entry raises
:class:`PairContractError`. Consumers (``llm_evaluate.py``,
``run_eval_batch.py``) must reject a task before any model invocation when
its pair cannot be resolved.

CLI:
    python3 -m scripts.evaluation.pair_contracts \\
      --registry config/final_pair_contracts.json --check-completeness \\
      [--tasks tasks.json]

Without ``--tasks``, ``--check-completeness`` requires every entry in the
registry to be fully resolved; with ``--tasks`` it additionally requires the
registry to cover every cross-API pair in the task list.
"""
from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = _REPO_ROOT / "schema" / "pair_contract_schema.json"

CONTRACT_VERSION = "parbench-final-v1"
# Current generation (2026-08-21, owner OD-3 follow-up ruling): kernel-only
# pairs carry TARGET args in translated_run. New registries stamp this
# version; the frozen parbench-final-v1 registry backs the sealed NeurIPS
# evidence and is never regenerated. The schema accepts both strings, and
# every result JSON records which registry (path, version, sha256) it ran
# under, so the two generations can never silently mix in analysis.
CONTRACT_VERSION_CURRENT = "parbench-contracts-v2"


class PairContractError(ValueError):
    """Fail-closed error for a missing, malformed, or unresolved pair contract."""


@dataclass(frozen=True)
class PairContract:
    """One directed pair entry. ``data`` is the raw registry entry."""

    source_spec: str
    target_spec: str
    data: dict[str, Any] = field(compare=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_spec, self.target_spec)

    @property
    def shared_input(self) -> dict[str, Any]:
        return self.data.get("shared_input") or {}

    @property
    def translated_run(self) -> dict[str, Any]:
        return self.data.get("translated_run") or {}

    @property
    def target_oracle(self) -> dict[str, Any]:
        return self.data.get("target_oracle") or {}

    @property
    def negative_checks(self) -> list[dict[str, Any]]:
        return self.data.get("negative_checks") or []

    @property
    def comparator(self) -> dict[str, Any]:
        return self.data.get("comparator") or {}

    @property
    def baseline_witness(self) -> dict[str, Any]:
        return self.data.get("baseline_witness") or {}

    @property
    def comparability(self) -> str:
        return self.data.get("comparability", "unresolved")

    def require_resolved(self) -> None:
        """Raise :class:`PairContractError` unless this entry is fully resolved.

        Resolved means: comparability is ``eligible``, the baseline witness
        passed with recorded evidence, a comparator with evidence is declared,
        the shared input is declared for both sides, the translated run has
        arguments, and the target oracle declares at least one strategy.
        """
        problems: list[str] = []
        if self.comparability != "eligible":
            problems.append(f"comparability is {self.comparability!r}, not 'eligible'")
        witness = self.baseline_witness
        if witness.get("status") != "pass":
            problems.append(
                f"baseline witness status is {witness.get('status')!r}, not 'pass'"
            )
        if not witness.get("evidence"):
            problems.append("baseline witness has no evidence path")
        else:
            _check_evidence_path("baseline witness", witness["evidence"], problems)
        comparator = self.comparator
        if not comparator.get("type") or not comparator.get("evidence"):
            problems.append("comparator with type and evidence is not declared")
        elif comparator.get("evidence"):
            _check_evidence_path("comparator", comparator["evidence"], problems)
        shared = self.shared_input
        if "source_args" not in shared or "target_args" not in shared:
            problems.append("shared_input does not declare source_args and target_args")
        if "args" not in self.translated_run:
            problems.append("translated_run does not declare args")
        if not self.target_oracle.get("strategies"):
            problems.append("target_oracle declares no strategies")
        if problems:
            raise PairContractError(
                f"pair contract {self.source_spec} -> {self.target_spec} "
                f"is not resolved: " + "; ".join(problems)
            )


def _check_evidence_path(label: str, value: str, problems: list[str]) -> None:
    """Evidence must be a repo-relative, readable regular file inside the repo.

    Fail closed on an absolute path, a path escaping the repository root, a
    path that does not exist, a directory, or an unreadable file —
    self-reported evidence a reviewer cannot open must not certify
    eligibility.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        problems.append(f"{label} evidence path is absolute, not repo-relative: {value}")
        return
    resolved = (_REPO_ROOT / candidate).resolve()
    if not str(resolved).startswith(str(_REPO_ROOT) + os.sep):
        problems.append(f"{label} evidence path escapes the repository: {value}")
        return
    if not resolved.is_file():
        problems.append(f"{label} evidence path is not an existing regular file: {value}")
        return
    if not os.access(resolved, os.R_OK):
        problems.append(f"{label} evidence file is not readable: {value}")


def _spec_correctness_args(spec: dict) -> list[str]:
    """The spec's declared correctness arguments (empty list if none)."""
    return list(
        ((spec.get("run") or {}).get("input_configurations", {}).get("correctness") or {})
        .get("arguments", [])
    )


def validate_contract_against_specs(
    contract: PairContract, source_spec: dict, target_spec: dict
) -> None:
    """Full pre-model validation of one contract against its two spec dicts.

    Raises :class:`PairContractError` when the contract is unresolved, names
    a different pair than the supplied specs, or declares shared inputs that
    do not match the specs' actual correctness arguments (mismatched inputs).
    """
    source_id = source_spec.get("identity", {}).get("unique_id", "?")
    target_id = target_spec.get("identity", {}).get("unique_id", "?")
    if (contract.source_spec, contract.target_spec) != (source_id, target_id):
        raise PairContractError(
            f"pair contract is for {contract.source_spec} -> "
            f"{contract.target_spec}, not {source_id} -> {target_id}"
        )
    contract.require_resolved()
    declared_source = contract.shared_input.get("source_args")
    actual_source = _spec_correctness_args(source_spec)
    if declared_source != actual_source:
        raise PairContractError(
            f"pair contract {source_id} -> {target_id}: declared shared_input "
            f"source_args {declared_source!r} do not match the source spec's "
            f"correctness arguments {actual_source!r}"
        )
    declared_target = contract.shared_input.get("target_args")
    actual_target = _spec_correctness_args(target_spec)
    if declared_target != actual_target:
        raise PairContractError(
            f"pair contract {source_id} -> {target_id}: declared shared_input "
            f"target_args {declared_target!r} do not match the target spec's "
            f"correctness arguments {actual_target!r}"
        )


class PairContractRegistry:
    """All pair contracts from one registry file, keyed by (source, target)."""

    def __init__(
        self,
        contract_version: str,
        pairs: dict[tuple[str, str], PairContract],
        path: Path | None = None,
    ) -> None:
        self.contract_version = contract_version
        self._pairs = pairs
        self.path = path
        # File-bytes hash, set by load_pair_contracts; None for registries
        # built in memory (tests).
        self.sha256: str | None = None

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._pairs

    def keys(self) -> set[tuple[str, str]]:
        return set(self._pairs)

    def contracts(self) -> list[PairContract]:
        return [self._pairs[k] for k in sorted(self._pairs)]

    def get(self, source_spec: str, target_spec: str) -> PairContract:
        """Return the contract for a directed pair; raise if none is declared."""
        try:
            return self._pairs[(source_spec, target_spec)]
        except KeyError:
            raise PairContractError(
                f"no pair contract declared for {source_spec} -> {target_spec}"
                + (f" in {self.path}" if self.path else "")
            ) from None

    def resolve(self, source_spec: str, target_spec: str) -> PairContract:
        """Return the contract for a directed pair, requiring it resolved."""
        contract = self.get(source_spec, target_spec)
        contract.require_resolved()
        return contract


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def _reject_nonfinite(token: str) -> Any:
    """Refuse Python's non-standard JSON tokens (Codex gate pass 3).

    ``json.loads`` accepts ``Infinity``, ``-Infinity``, and ``NaN`` by
    default, which would slip a non-finite tolerance past the schema's
    numeric bound and let any value verify as PASS.
    """
    raise ValueError(f"non-finite JSON token {token!r} is not allowed")


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse JSON objects with duplicate member names (Codex gate pass 6).

    ``json.loads`` silently keeps the LAST duplicate, so
    ``"tolerance": 0, "tolerance": 1e308`` would validate as 0 in spirit and
    verify as 1e308 in practice.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r} is not allowed")
        result[key] = value
    return result


def _parse_finite_float(token: str) -> float:
    """Refuse floats that binary64 cannot faithfully represent (gate pass 4;
    raw-token range/underflow validation added at contract finalization,
    Task 7, per the ratified Task 2 residual ruling).

    ``parse_constant`` never sees exponent overflow — ``1e999`` parses as a
    standard-looking number and becomes ``float('inf')`` — so every float is
    checked for finiteness here. Two further representability escapes are
    caught on the RAW token via ``decimal.Decimal`` before float rounding
    can hide them: silent underflow (``1e-999`` -> ``0.0``) and values just
    beyond DBL_MAX that ROUND DOWN to it (``1.7976931348623158e308``), which
    the post-parse schema bound cannot see.
    """
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {token!r} is not allowed")
    try:
        exact = decimal.Decimal(token)
    except decimal.InvalidOperation as exc:
        raise ValueError(f"unparseable JSON number {token!r}") from exc
    if exact != 0 and value == 0.0:
        raise ValueError(
            f"JSON number {token!r} silently underflows binary64 to 0.0")
    if abs(exact) > decimal.Decimal("1.7976931348623157e308"):
        raise ValueError(
            f"JSON number {token!r} exceeds the finite binary64 range")
    return value


def load_pair_contracts(path: Path) -> PairContractRegistry:
    """Load and validate a pair-contract registry file. Fails closed."""
    path = Path(path)
    if not path.is_file():
        raise PairContractError(f"pair-contract registry not found: {path}")
    # Read the bytes ONCE: the same buffer is parsed and hashed, so the
    # recorded sha256 always identifies exactly the bytes that were parsed
    # (a second read could race a concurrent file replacement).
    raw = path.read_bytes()
    try:
        data = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_members,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise PairContractError(
            f"pair-contract registry is not valid JSON: {path}: {exc}"
        ) from exc

    try:
        jsonschema.validate(data, _load_schema())
    except jsonschema.ValidationError as exc:
        raise PairContractError(
            f"pair-contract registry violates schema: {path}: {exc.message} "
            f"(at {'/'.join(str(p) for p in exc.absolute_path) or '<root>'})"
        ) from exc

    pairs: dict[tuple[str, str], PairContract] = {}
    for entry in data["pairs"]:
        key = (entry["source_spec"], entry["target_spec"])
        if key in pairs:
            raise PairContractError(
                f"duplicate pair key {key[0]} -> {key[1]} in {path}"
            )
        pairs[key] = PairContract(key[0], key[1], entry)

    registry = PairContractRegistry(data["contract_version"], pairs, path)
    # Provenance stamp: result JSONs record which registry bytes a run used
    # (2026-08-21 Codex finding: v1/v2 were indistinguishable at run time).
    registry.sha256 = hashlib.sha256(raw).hexdigest()
    return registry


def required_pair_keys(tasks: list[dict]) -> set[tuple[str, str]]:
    """Directed pair keys a task list requires contracts for.

    Accepts both the batch runner's task dicts (``src_id``/``tgt_id``) and
    passer-style dicts (``source_spec``/``target_spec``). Same-spec tasks
    (same-API, augmentation-only) need no pair contract.
    """
    keys: set[tuple[str, str]] = set()
    for task in tasks:
        src = task.get("source_spec") or task.get("src_id")
        tgt = task.get("target_spec") or task.get("tgt_id")
        if not src or not tgt:
            raise PairContractError(
                f"task has no identifiable source/target spec ids: {task!r}"
            )
        if src != tgt:
            keys.add((src, tgt))
    return keys


def validate_registry_completeness(
    registry: PairContractRegistry, tasks: list[dict]
) -> None:
    """Require a resolved contract for every cross-API pair in ``tasks``.

    Raises :class:`PairContractError` on the first gap: a missing entry, or
    any required entry that does not resolve. Returns None when complete.
    """
    required = required_pair_keys(tasks)
    missing = sorted(required - registry.keys())
    if missing:
        listing = ", ".join(f"{s} -> {t}" for s, t in missing)
        raise PairContractError(
            f"pair-contract registry is missing {len(missing)} required "
            f"pair(s): {listing}"
        )
    for source_spec, target_spec in sorted(required):
        registry.resolve(source_spec, target_spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a pair-contract registry (fail closed).",
    )
    parser.add_argument("--registry", type=Path, required=True,
                        help="Path to the registry JSON.")
    parser.add_argument("--check-completeness", action="store_true",
                        help="Require every entry (or every --tasks pair) resolved.")
    parser.add_argument("--tasks", type=Path, default=None,
                        help="Optional JSON task list to check coverage against.")
    args = parser.parse_args(argv)

    try:
        registry = load_pair_contracts(args.registry)
    except PairContractError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"Loaded {len(registry)} pair contract(s) "
          f"(version {registry.contract_version}) from {args.registry}")

    if not args.check_completeness:
        return 0

    try:
        if args.tasks is not None:
            tasks = json.loads(args.tasks.read_text())
            validate_registry_completeness(registry, tasks)
            print(f"OK: registry covers all {len(required_pair_keys(tasks))} "
                  f"required pair(s) with resolved contracts")
        else:
            for contract in registry.contracts():
                contract.require_resolved()
            print(f"OK: all {len(registry)} pair contract(s) are resolved")
    except PairContractError as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
