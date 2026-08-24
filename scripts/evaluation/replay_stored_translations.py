#!/usr/bin/env python3
"""Replay stored translations under the final pair contracts (Task 5).

Rebuilds and re-verifies the ``translated_files`` already stored in immutable
``results/evaluation`` records. It NEVER makes a model or provider request:
on startup it replaces the provider entry point (``llm_evaluate.call_llm``)
with a function that raises :class:`ProviderCallForbidden` immediately.

Safety contract (plan Task 5):
  - An output root at or under ``results/evaluation`` aborts BEFORE any file
    is opened (exit 2).
  - An existing output record whose recorded provenance (input SHA-256 +
    contract registry SHA-256) does not match the current inputs aborts
    rather than overwrites (exit 2). With ``--resume``, a matching existing
    output is skipped; without it, any existing output refuses to overwrite.
  - Old-to-new status transitions are written as a generated companion
    artifact (``replay_transitions.json``) in the new namespace, alongside a
    hash manifest (``replay_manifest.json``) consumed by
    ``scripts/provenance/seal_result_namespace.py``.

Verification ERROR attribution (Task 2 note): a harness-side verification
ERROR yields ``overall_status: "ERROR"``, never ``VERIFY_FAIL``.

Command contract:
    python3 scripts/evaluation/replay_stored_translations.py \\
      --input-root results/evaluation \\
      --output-root results/evaluation_final/parbench-final-v1 \\
      --pair-contracts config/final_pair_contracts.json \\
      --resume
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.builder import build_spec  # noqa: E402
from harness.constants import (  # noqa: E402
    CORRECTNESS_INELIGIBLE_SPECS,
    EXCLUDED_SPECS,
    PERFORMANCE_ONLY_SPECS,
)
from harness.models import Status  # noqa: E402
from harness.runner import run_spec  # noqa: E402
from harness.spec_loader import load_spec, resolve_paths  # noqa: E402
from harness.verifier import verify_run  # noqa: E402
from scripts.evaluation import llm_evaluate  # noqa: E402
from scripts.evaluation.llm_evaluate import (  # noqa: E402
    _build_cross_api_run_spec,
    _build_cross_api_verify_spec,
    _check_stdout_error_indicators,
    _head_tail,
    _stage_support_headers,
    _unstage_support_headers,
    backup_files,
    restore_files,
)
from scripts.evaluation.pair_contracts import (  # noqa: E402
    PairContract,
    PairContractError,
    PairContractRegistry,
    load_pair_contracts,
    validate_contract_against_specs,
)

logger = logging.getLogger("replay_stored_translations")

REPLAY_SCHEMA_VERSION = 1
TRANSITIONS_NAME = "replay_transitions.json"
MANIFEST_NAME = "replay_manifest.json"
COMPANION_FILES = {TRANSITIONS_NAME, MANIFEST_NAME, ".parbench-seal.json"}


class ProviderCallForbidden(RuntimeError):
    """Raised when any provider entry point is invoked during replay."""


def _forbidden_provider_call(*args: Any, **kwargs: Any) -> Any:
    raise ProviderCallForbidden(
        "replay runs in no-provider mode: llm_evaluate.call_llm must never "
        "be invoked while replaying stored translations"
    )


def install_no_provider_mode() -> None:
    """Make any attempted provider call fail immediately (design contract)."""
    llm_evaluate.call_llm = _forbidden_provider_call


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_under_results_evaluation(path: Path, project_root: Path) -> bool:
    """True when *path* is at or under any ``results/evaluation`` directory.

    Checked on BOTH the lexically-resolved absolute path (covers paths that
    do not exist yet) and the fully symlink-resolved path (T5 gate finding
    2026-08-12: a symlink alias resolving inside results/evaluation must not
    pass the gate).
    """
    forbidden = (project_root / "results" / "evaluation").resolve()
    for candidate in (Path(os.path.abspath(path)), Path(path).resolve()):
        parts = candidate.parts
        for i in range(len(parts) - 1):
            if parts[i] == "results" and parts[i + 1] == "evaluation":
                return True
        try:
            candidate.relative_to(forbidden)
            return True
        except ValueError:
            pass
    return False


def _tool_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else None


def collect_provenance() -> tuple[dict[str, Any], dict[str, Any]]:
    """Toolchain and platform provenance recorded into every output record."""
    toolchain = {
        "cc": _tool_version(["cc", "--version"]),
        "gcc": _tool_version(["gcc", "--version"]),
        "nvcc": _tool_version(["nvcc", "--version"]),
        "nvc++": _tool_version(["nvc++", "--version"]),
    }
    plat = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    return toolchain, plat


def _spec_hash_entry(spec_path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": os.path.relpath(spec_path, project_root),
        "sha256": _sha256_bytes(spec_path.read_bytes()),
    }


def replay_one(
    record: dict[str, Any],
    project_root: Path,
    registry: PairContractRegistry,
    verbose: bool = False,
) -> dict[str, Any]:
    """Rebuild, run, and verify one stored translation. No provider call.

    Returns the pipeline-outcome portion of the output record:
    build/run/verify statuses plus the new ``overall_status``. Statuses
    mirror the live pipeline, with the Task-2 attribution rule: a
    verification-stage ERROR is ``ERROR``, never ``VERIFY_FAIL``.
    """
    outcome: dict[str, Any] = {
        "build_status": None,
        "build_time_seconds": None,
        "build_error_snippet": None,
        "run_status": None,
        "run_exit_code": None,
        "run_time_seconds": None,
        "run_stdout_snippet": None,
        "run_stderr_snippet": None,
        "verify_status": None,
        "verify_strategy": None,
        "verify_details": None,
        "run_arguments_used": None,
        "overall_status": "ERROR",
        "error_message": None,
    }

    translated_files: dict[str, str] = record.get("translated_files") or {}
    if not translated_files:
        outcome["overall_status"] = "NOT_REPLAYABLE"
        outcome["error_message"] = (
            "record stores no translated_files; nothing to rebuild"
        )
        return outcome

    source_id = record["source_spec"]
    target_id = record["target_spec"]
    source_spec_path = project_root / "specs" / f"{source_id}.json"
    target_spec_path = project_root / "specs" / f"{target_id}.json"
    for p in (source_spec_path, target_spec_path):
        if not p.is_file():
            outcome["error_message"] = f"spec file not found: {p}"
            return outcome

    source_spec = load_spec(source_spec_path)
    target_spec = load_spec(target_spec_path)
    target_spec_resolved = resolve_paths(target_spec, project_root)
    source_api = source_spec.get("identity", {}).get("parallel_api", "")
    target_api = target_spec.get("identity", {}).get("parallel_api", "")

    pair_contract: PairContract | None = None
    if source_api != target_api:
        try:
            pair_contract = registry.resolve(source_id, target_id)
            validate_contract_against_specs(pair_contract, source_spec, target_spec)
        except PairContractError as exc:
            outcome["error_message"] = f"pair contract unresolved: {exc}"
            return outcome

    resolved: dict[str, Any] = target_spec_resolved.get("_resolved", {})
    source_dir: Path = resolved.get("source_dir", project_root)
    target_filenames: list[str] = target_spec["files"]["translation_targets"]
    target_file_paths = [source_dir / fname for fname in target_filenames]

    # The stored file set must equal the spec's translation_targets (Task A
    # finding 7, 2026-08-12): a partial set (e.g. a partial-extraction parent)
    # would rebuild against untouched reference files and record a misleading
    # BUILD_FAIL for a contaminated build.
    stored_names = set(translated_files)
    if stored_names != set(target_filenames):
        missing = sorted(set(target_filenames) - stored_names)
        extra = sorted(stored_names - set(target_filenames))
        outcome["overall_status"] = "NOT_REPLAYABLE"
        outcome["error_message"] = (
            "stored translated_files do not match translation_targets "
            f"(missing: {missing}, extraneous: {extra}); a partial file set "
            "would rebuild against reference code"
        )
        return outcome

    backup_info = backup_files(target_file_paths)
    staged_headers = _stage_support_headers(
        source_spec, target_spec_resolved, project_root
    )
    try:
        # Write the STORED translated files over the reference tree.
        for fname, code in translated_files.items():
            fp = source_dir / fname
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(code, encoding="utf-8")

        build_result = build_spec(target_spec_resolved, project_root, verbose=verbose)
        outcome["build_status"] = build_result.status.value
        outcome["build_time_seconds"] = round(build_result.duration_seconds, 3)
        if build_result.status != Status.PASS:
            outcome["build_error_snippet"] = _head_tail(build_result.stderr or "")
            outcome["overall_status"] = "BUILD_FAIL"
            outcome["error_message"] = (
                f"Build failed: {(build_result.stderr or '')[-200:].strip()}"
            )
            return outcome

        if source_api != target_api:
            run_spec_dict = _build_cross_api_run_spec(
                target_spec_resolved, source_spec, pair_contract
            )
        else:
            run_spec_dict = target_spec_resolved
        run_configs = (run_spec_dict.get("run") or {}).get("input_configurations", {})
        outcome["run_arguments_used"] = list(
            (run_configs.get("correctness") or {}).get("arguments", [])
        )

        run_result = run_spec(run_spec_dict, project_root, verbose=verbose)
        outcome["run_status"] = run_result.status.value
        outcome["run_exit_code"] = run_result.exit_code
        outcome["run_time_seconds"] = round(run_result.duration_seconds, 3)
        outcome["run_stdout_snippet"] = (run_result.stdout or "")[-500:]
        outcome["run_stderr_snippet"] = (run_result.stderr or "")[-500:]
        if run_result.status != Status.PASS:
            outcome["overall_status"] = "RUN_FAIL"
            outcome["error_message"] = f"Run failed (exit code {run_result.exit_code})"
            return outcome

        if source_api != target_api:
            verify_spec = _build_cross_api_verify_spec(
                target_spec, source_spec, pair_contract
            )
        else:
            verify_spec = target_spec
        verify_result = verify_run(
            verify_spec,
            run_result,
            working_dir=target_spec_resolved["_resolved"]["working_dir"],
        )
        outcome["verify_status"] = verify_result.status.value
        outcome["verify_strategy"] = verify_result.strategy_used
        outcome["verify_details"] = verify_result.details

        if verify_result.status == Status.PASS:
            reject_reason = _check_stdout_error_indicators(run_result.stdout or "")
            if reject_reason:
                outcome["overall_status"] = "VERIFY_FAIL"
                outcome["error_message"] = f"False positive rejected: {reject_reason}"
            else:
                outcome["overall_status"] = "PASS"
        elif verify_result.status == Status.ERROR:
            # Task 2 attribution rule: a harness-side verification ERROR is
            # an infrastructure error, distinct from VERIFY_FAIL.
            outcome["overall_status"] = "ERROR"
            outcome["error_message"] = f"Verification error: {verify_result.details}"
        else:
            outcome["overall_status"] = "VERIFY_FAIL"
            outcome["error_message"] = f"Verify failed: {verify_result.details}"
        return outcome
    finally:
        restore_files(backup_info)
        _unstage_support_headers(staged_headers)


def _spec_api(spec_id: str) -> str:
    """Trailing API segment of a spec id (``rodinia-bfs-cuda`` -> ``cuda``).

    ``omp_target`` uses an underscore, so splitting on the final dash is
    exact for every spec id in the corpus.
    """
    return spec_id.rsplit("-", 1)[-1] if spec_id else "unknown"


def _write_companions(output_root: Path) -> None:
    """Regenerate the transitions and hash-manifest companion artifacts."""
    records: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    for path in sorted(output_root.rglob("*.json")):
        rel = path.relative_to(output_root).as_posix()
        if path.name in COMPANION_FILES:
            continue
        files[rel] = _sha256_bytes(path.read_bytes())
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "output_record": rel,
                "input_record": data.get("input_record", {}).get("path"),
                "model": data.get("parent", {}).get("model"),
                "source_spec": data.get("parent", {}).get("source_spec"),
                "target_spec": data.get("parent", {}).get("target_spec"),
                "augment_level": data.get("parent", {}).get("augment_level"),
                "old_status": data.get("parent", {}).get("overall_status"),
                "new_status": data.get("overall_status"),
            }
        )

    counts: dict[str, int] = {}
    by_model: dict[str, dict[str, int]] = {}
    by_suite: dict[str, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}
    by_augment_level: dict[str, dict[str, int]] = {}
    by_failure_stage: dict[str, dict[str, int]] = {}
    for entry in records:
        key = f"{entry['old_status']}->{entry['new_status']}"
        counts[key] = counts.get(key, 0) + 1
        src = entry["source_spec"] or ""
        tgt = entry["target_spec"] or ""
        level = entry["augment_level"]
        dims = {
            "model": entry["model"] or "unknown",
            "suite": src.split("-", 1)[0] if src else "unknown",
            "direction": f"{_spec_api(src)}-to-{_spec_api(tgt)}",
            "level": f"L{level}" if isinstance(level, int) else "unknown",
        }
        for table, dim_key in (
            (by_model, dims["model"]),
            (by_suite, dims["suite"]),
            (by_direction, dims["direction"]),
            (by_augment_level, dims["level"]),
        ):
            cell = table.setdefault(dim_key, {})
            cell[key] = cell.get(key, 0) + 1
        stage = by_failure_stage.setdefault(str(entry["old_status"]), {})
        new = str(entry["new_status"])
        stage[new] = stage.get(new, 0) + 1

    def _sorted_nested(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}

    transitions = {
        "generated_by": "scripts/evaluation/replay_stored_translations.py",
        "generated_at": _utc_now(),
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "by_model": _sorted_nested(by_model),
        "by_suite": _sorted_nested(by_suite),
        "by_direction": _sorted_nested(by_direction),
        "by_augment_level": _sorted_nested(by_augment_level),
        "by_failure_stage": _sorted_nested(by_failure_stage),
        "records": records,
    }
    (output_root / TRANSITIONS_NAME).write_text(
        json.dumps(transitions, indent=1) + "\n", encoding="utf-8"
    )

    manifest = {
        "generated_by": "scripts/evaluation/replay_stored_translations.py",
        "generated_at": _utc_now(),
        "file_count": len(files),
        "files": files,
    }
    (output_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_spec_sha(spec_id: str, project_root: Path,
                      cache: dict[str, str | None]) -> str | None:
    """SHA-256 of the current spec file for *spec_id* (None if absent)."""
    if spec_id not in cache:
        p = project_root / "specs" / f"{spec_id}.json"
        cache[spec_id] = _sha256_bytes(p.read_bytes()) if p.is_file() else None
    return cache[spec_id]


def _stale_spec_side(out: dict[str, Any], project_root: Path,
                     cache: dict[str, str | None]) -> str | None:
    """Name the first side whose recorded spec hash is missing or differs
    from the current spec file (Task A finding 4: a changed spec must not
    leave replay outputs silently accepted as current). Fail closed: a record
    without a recorded hash or parent spec id is never "current" (every
    sealed parbench-final-v1 record carries both, verified 2026-08-12).
    Returns None only when both sides match."""
    for side, key in (("source", "source_spec"), ("target", "target_spec")):
        recorded = ((out.get("spec_hashes") or {}).get(side) or {}).get("sha256")
        spec_id = (out.get("parent") or {}).get(key)
        if recorded is None or not spec_id:
            return side
        if recorded != _current_spec_sha(spec_id, project_root, cache):
            return side
    return None


def _contract_manifest_problems(
    manifest: dict[str, Any],
    manifest_path: Path,
    registry_path: Path,
    registry_sha256: str,
    project_root: Path,
    *,
    require_harness: bool,
) -> list[str]:
    """Verify the frozen contract manifest against the current checkout.

    Fail closed on: registry mismatch (manifest vs current registry file, and
    vs the --pair-contracts actually in use), spec drift, and exclusion-list
    drift. Harness drift fails only when *require_harness* (replay/resume:
    new records must not claim the frozen contract under changed harness
    code); for read-only completeness checks it is reported as a WARNING,
    because post-seal harness fixes do not alter already-sealed bytes.
    """
    problems: list[str] = []
    reg = manifest.get("pair_registry") or {}
    if reg.get("sha256") != registry_sha256:
        problems.append(
            f"--pair-contracts {registry_path} does not match the frozen "
            f"manifest's pair_registry sha256 ({manifest_path})"
        )
    # Absent declarations are a fail, not a skip (fixcheck2, 2026-08-12): a
    # manifest with an empty spec map or no exclusion lists verifies nothing.
    if not ((manifest.get("specs") or {}).get("files") or {}):
        problems.append("frozen manifest declares no spec hashes")
    for key in ("excluded_specs", "performance_only_specs",
                "correctness_ineligible_specs"):
        if (manifest.get("exclusions") or {}).get(key) is None:
            problems.append(f"frozen manifest declares no {key} list")
    for section, subdir, required in (
        ("specs", "specs", True),
        ("harness", "harness", require_harness),
    ):
        files = (manifest.get(section) or {}).get("files") or {}
        drifted = []
        for name, sha in files.items():
            p = project_root / subdir / name
            if not p.is_file() or _sha256_bytes(p.read_bytes()) != sha:
                drifted.append(name)
        if drifted:
            msg = (f"{section} drift vs frozen manifest: {len(drifted)} "
                   f"file(s), e.g. {drifted[:3]}")
            if required:
                problems.append(msg)
            else:
                print(f"WARNING: {msg} (post-seal drift; sealed bytes are "
                      "unaffected)", file=sys.stderr)
    exclusions = manifest.get("exclusions") or {}
    current = {
        "excluded_specs": EXCLUDED_SPECS,
        "performance_only_specs": PERFORMANCE_ONLY_SPECS,
        "correctness_ineligible_specs": CORRECTNESS_INELIGIBLE_SPECS,
    }
    for key, cur in current.items():
        rec = exclusions.get(key)
        if rec is not None and sorted(rec) != sorted(cur):
            problems.append(f"exclusion list {key} differs from the frozen manifest")
    return problems


def check_complete(
    input_root: Path,
    output_root: Path,
    registry_sha256: str,
    *,
    project_root: Path | None = None,
    contract_manifest_sha: str | None = None,
) -> int:
    """Read-only input-vs-output key-set comparison (Task 8 Checks block).

    Complete means: every translation record under *input_root* has an
    output record at the same relative path whose recorded provenance
    (input-record SHA-256 + contract-registry SHA-256) matches the current
    inputs, and no output record exists without an input record. Input JSONs
    that are not translation records (no ``source_spec``/``target_spec``)
    are reported as an explicit non-record disposition, never silently
    dropped. Writes nothing; exit 1 on any omission, orphan, or mismatch.
    """
    problems: list[str] = []
    non_records: list[str] = []
    input_shas: dict[str, str] = {}
    for path in sorted(p for p in input_root.glob("*/*.json") if p.is_file()):
        rel = path.relative_to(input_root).as_posix()
        try:
            data = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            problems.append(f"input record is not valid JSON: {rel}")
            continue
        if "source_spec" not in data or "target_spec" not in data:
            non_records.append(rel)
            continue
        input_shas[rel] = _sha256_bytes(path.read_bytes())

    outputs: dict[str, dict[str, Any]] = {}
    if output_root.is_dir():
        for path in sorted(output_root.rglob("*.json")):
            if path.name in COMPANION_FILES:
                continue
            rel = path.relative_to(output_root).as_posix()
            try:
                outputs[rel] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                problems.append(f"output record is not valid JSON: {rel}")
                outputs[rel] = {}

    matched = 0
    status_counts: dict[str, int] = {}
    dispositions: dict[str, int] = {}
    spec_sha_cache: dict[str, str | None] = {}
    for rel, sha in input_shas.items():
        out = outputs.get(rel)
        if out is None:
            problems.append(f"OMISSION: no replay output for input record {rel}")
            continue
        if out.get("input_record", {}).get("sha256") != sha:
            problems.append(f"provenance mismatch (input sha256): {rel}")
            continue
        if out.get("contract", {}).get("registry_sha256") != registry_sha256:
            problems.append(f"provenance mismatch (contract registry sha256): {rel}")
            continue
        if contract_manifest_sha is not None and (
            (out.get("contract_manifest") or {}).get("sha256")
            != contract_manifest_sha
        ):
            problems.append(
                f"provenance mismatch (contract manifest sha256): {rel}"
            )
            continue
        if project_root is not None:
            stale = _stale_spec_side(out, project_root, spec_sha_cache)
            if stale:
                problems.append(
                    f"provenance mismatch ({stale} spec changed since replay): {rel}"
                )
                continue
        matched += 1
        status = str(out.get("overall_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        # Explicit dispositions for records that could not be re-executed.
        message = out.get("error_message") or ""
        if status == "NOT_REPLAYABLE":
            key = (
                "not_replayable_partial_file_set"
                if "do not match translation_targets" in message
                else "not_replayable_no_stored_files"
            )
            dispositions[key] = dispositions.get(key, 0) + 1
        elif status == "ERROR" and message.startswith("pair contract unresolved"):
            dispositions["contract_comparability_disposition"] = (
                dispositions.get("contract_comparability_disposition", 0) + 1
            )
    for rel in sorted(set(outputs) - set(input_shas)):
        problems.append(f"ORPHAN: output record has no input record: {rel}")

    transitions_path = output_root / TRANSITIONS_NAME
    if transitions_path.is_file():
        transitions = json.loads(transitions_path.read_text(encoding="utf-8"))
        if transitions.get("record_count") != len(outputs):
            problems.append(
                f"transitions record_count {transitions.get('record_count')} "
                f"!= output record count {len(outputs)}"
            )
    else:
        problems.append(f"missing companion artifact: {TRANSITIONS_NAME}")

    print(
        f"input records: {len(input_shas)} (+{len(non_records)} non-record "
        f"JSON(s): {non_records or 'none'})"
    )
    print(f"output records matched with current provenance: {matched}")
    print(f"new-status breakdown: {dict(sorted(status_counts.items()))}")
    print(f"explicit dispositions: {dict(sorted(dispositions.items())) or 'none'}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        print(f"CHECK-COMPLETE FAILED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(
        "CHECK-COMPLETE OK: every input translation record has a replay "
        "output with matching provenance; no orphans."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Immutable input corpus (e.g. results/evaluation)")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New replay namespace (must NOT be under results/evaluation)")
    parser.add_argument("--pair-contracts", type=Path, required=True,
                        help="Pair-contract registry JSON")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="ParBench project root (default: cwd)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip records whose output already exists with matching provenance")
    parser.add_argument("--limit", type=int, default=None,
                        help="Replay at most N records (smoke runs)")
    parser.add_argument("--check-complete", action="store_true",
                        help="Read-only completeness check: compare input and "
                             "output key sets and provenance; replay nothing")
    parser.add_argument("--contract-manifest", type=Path, default=None,
                        help="Frozen contract manifest (e.g. config/"
                             "final_contract_manifest.json); its SHA-256 is "
                             "bound into every new record and validated on "
                             "resume and --check-complete")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    project_root = args.project_root.resolve()

    # SAFETY GATE — runs before ANY file is opened, created, or read.
    if _is_under_results_evaluation(args.output_root, project_root):
        print(
            f"ABORT: output root {args.output_root} is at or under "
            "results/evaluation — the immutable corpus is never a replay "
            "target. No file was opened.",
            file=sys.stderr,
        )
        return 2

    # No-provider mode: any attempted provider call now raises immediately.
    install_no_provider_mode()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        print(f"ABORT: input root not found: {input_root}", file=sys.stderr)
        return 2

    try:
        registry = load_pair_contracts(args.pair_contracts)
    except PairContractError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    registry_sha256 = _sha256_bytes(Path(args.pair_contracts).read_bytes())

    contract_manifest_sha: str | None = None
    if args.contract_manifest is not None:
        if not args.contract_manifest.is_file():
            print(f"ABORT: contract manifest not found: {args.contract_manifest}",
                  file=sys.stderr)
            return 2
        manifest_bytes = args.contract_manifest.read_bytes()
        contract_manifest_sha = _sha256_bytes(manifest_bytes)
        try:
            manifest_doc = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            print(f"ABORT: contract manifest is not valid JSON: {exc}",
                  file=sys.stderr)
            return 2
        manifest_problems = _contract_manifest_problems(
            manifest_doc, args.contract_manifest, Path(args.pair_contracts),
            registry_sha256, project_root,
            require_harness=not args.check_complete,
        )
        if manifest_problems:
            for p in manifest_problems:
                print(f"ABORT: {p}", file=sys.stderr)
            return 2

    if args.check_complete:
        return check_complete(
            input_root, args.output_root.resolve(), registry_sha256,
            project_root=project_root,
            contract_manifest_sha=contract_manifest_sha,
        )

    record_paths = sorted(
        p for p in input_root.glob("*/*.json") if p.is_file()
    )
    if args.limit is not None:
        record_paths = record_paths[: args.limit]
    if not record_paths:
        print(f"ABORT: no input records under {input_root}", file=sys.stderr)
        return 2

    toolchain, plat = collect_provenance()
    output_root = args.output_root.resolve()

    # A sealed namespace (its own or an ancestor's .parbench-seal.json) is
    # complete and immutable - never a replay target (T5 gate, 2026-08-12).
    probe = output_root
    while True:
        if (probe / ".parbench-seal.json").is_file():
            print(
                f"ABORT: output root {output_root} is inside the sealed "
                f"namespace {probe}; sealed namespaces are immutable.",
                file=sys.stderr,
            )
            return 2
        if probe.parent == probe:
            break
        probe = probe.parent

    replayed = skipped = 0
    for rec_path in record_paths:
        rel = rec_path.relative_to(input_root)
        rec_bytes = rec_path.read_bytes()
        rec_sha = _sha256_bytes(rec_bytes)
        try:
            record = json.loads(rec_bytes)
        except json.JSONDecodeError as exc:
            print(f"ABORT: input record is not valid JSON: {rec_path}: {exc}",
                  file=sys.stderr)
            return 2
        if "source_spec" not in record or "target_spec" not in record:
            logger.info("skipping non-record JSON: %s", rec_path)
            continue

        out_path = output_root / rel
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            same_provenance = (
                existing.get("input_record", {}).get("sha256") == rec_sha
                and existing.get("contract", {}).get("registry_sha256")
                == registry_sha256
            )
            if not same_provenance:
                print(
                    f"ABORT: existing output {out_path} has mismatched "
                    "provenance (input record or contract registry changed); "
                    "refusing to overwrite.",
                    file=sys.stderr,
                )
                return 2
            if args.resume:
                # A resumed record is only "current" if the specs it was
                # replayed under are unchanged and (when a frozen contract
                # manifest is given) it is bound to that manifest.
                spec_sha_cache: dict[str, str | None] = {}
                stale = _stale_spec_side(existing, project_root, spec_sha_cache)
                if stale:
                    print(
                        f"ABORT: existing output {out_path} was replayed "
                        f"under a different {stale} spec; the replay is not "
                        "current. Refusing to resume over it.",
                        file=sys.stderr,
                    )
                    return 2
                if contract_manifest_sha is not None and (
                    (existing.get("contract_manifest") or {}).get("sha256")
                    != contract_manifest_sha
                ):
                    print(
                        f"ABORT: existing output {out_path} is not bound to "
                        "the given contract manifest; refusing to resume "
                        "over it.",
                        file=sys.stderr,
                    )
                    return 2
                skipped += 1
                continue
            print(
                f"ABORT: output already exists: {out_path} "
                "(use --resume to skip existing records; overwrite is never "
                "performed)",
                file=sys.stderr,
            )
            return 2

        try:
            input_rel_path = os.path.relpath(rec_path, project_root)
        except ValueError:  # different drive (never on Linux)
            input_rel_path = str(rec_path)

        outcome = replay_one(record, project_root, registry, verbose=args.verbose)

        source_spec_path = project_root / "specs" / f"{record['source_spec']}.json"
        target_spec_path = project_root / "specs" / f"{record['target_spec']}.json"
        spec_hashes = {
            side: (_spec_hash_entry(p, project_root) if p.is_file() else None)
            for side, p in (
                ("source", source_spec_path),
                ("target", target_spec_path),
            )
        }

        out_record: dict[str, Any] = {
            "replay_schema_version": REPLAY_SCHEMA_VERSION,
            "replay_kind": "stored_translation_replay",
            "timestamp": _utc_now(),
            "input_record": {"path": input_rel_path, "sha256": rec_sha},
            "parent": {
                "model": record.get("model"),
                "source_spec": record.get("source_spec"),
                "target_spec": record.get("target_spec"),
                "augment_level": record.get("augment_level"),
                "sample_id": record.get("sample_id"),
                "overall_status": record.get("overall_status"),
            },
            "contract": {
                "version": registry.contract_version,
                "registry_path": str(args.pair_contracts),
                "registry_sha256": registry_sha256,
            },
            "spec_hashes": spec_hashes,
            "toolchain": toolchain,
            "platform": plat,
            **outcome,
        }
        if contract_manifest_sha is not None:
            out_record["contract_manifest"] = {
                "path": str(args.contract_manifest),
                "sha256": contract_manifest_sha,
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_record, indent=2) + "\n", encoding="utf-8")
        replayed += 1
        if args.verbose:
            logger.info(
                "%s: %s -> %s", rel, record.get("overall_status"),
                out_record["overall_status"],
            )

    _write_companions(output_root)
    print(
        f"Replayed {replayed} record(s), skipped {skipped} existing; "
        f"companions written to {output_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
