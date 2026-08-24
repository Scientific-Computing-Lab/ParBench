#!/usr/bin/env python3
"""Seal a replay result namespace (Task 5).

Verifies the namespace is complete against its replay hash manifest
(``replay_manifest.json``, written by
``scripts/evaluation/replay_stored_translations.py``), hashes every file, and
writes ``.parbench-seal.json``. Once the marker exists, the seal-aware hooks
(``.claude/hooks/result-immutability.sh`` and
``.claude/hooks/protect-eval-results.sh``) block edit, overwrite, redirect,
and deletion anywhere inside the namespace.

Refusals (non-zero exit, no seal written or modified):
  - the namespace is at or under ``results/evaluation`` (never sealed here —
    that corpus has its own immutability hooks and predates sealing),
  - the namespace is empty,
  - a manifest entry is missing on disk or hashes differently,
  - a record file exists that the manifest does not list,
  - no manifest exists and ``--allow-no-manifest`` was not given,
  - a seal already exists and the current bytes differ from it (resealing
    different bytes is forbidden; identical bytes exit 0 without a rewrite).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

SEAL_NAME = ".parbench-seal.json"
MANIFEST_NAME = "replay_manifest.json"
SEAL_VERSION = 1

# Companion files excluded from the manifest completeness check (they are
# generated after the records and are hashed into the seal itself).
NON_RECORD_FILES = {SEAL_NAME, MANIFEST_NAME, "replay_transitions.json"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _aggregate(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\t{files[rel]}\n".encode())
    return h.hexdigest()


def _is_under_results_evaluation(path: Path) -> bool:
    parts = Path(os.path.abspath(path)).parts
    return any(
        parts[i] == "results" and parts[i + 1] == "evaluation"
        for i in range(len(parts) - 1)
    )


def _hash_namespace(namespace: Path) -> dict[str, str]:
    """SHA-256 of every regular file in the namespace, except the seal."""
    files: dict[str, str] = {}
    for p in sorted(namespace.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(namespace).as_posix()
        if p.name == SEAL_NAME:
            continue
        files[rel] = _sha256_file(p)
    return files


def check_completeness(namespace: Path, files: dict[str, str]) -> list[str]:
    """Verify the record set against the replay hash manifest.

    Returns a list of problems (empty = complete). Requires the manifest to
    exist; callers gate the no-manifest case behind ``--allow-no-manifest``.
    """
    problems: list[str] = []
    manifest_path = namespace / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed: dict[str, str] = manifest.get("files") or {}

    for rel, expected in sorted(listed.items()):
        actual = files.get(rel)
        if actual is None:
            problems.append(f"manifest entry missing on disk: {rel}")
        elif actual != expected:
            problems.append(
                f"hash mismatch vs manifest: {rel} "
                f"(manifest {expected[:12]}, disk {actual[:12]})"
            )

    record_files = {
        rel for rel in files
        if Path(rel).name not in NON_RECORD_FILES
    }
    for rel in sorted(record_files - set(listed)):
        problems.append(f"record not listed in manifest: {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", type=Path, required=True,
                        help="Result namespace directory to seal")
    parser.add_argument("--allow-no-manifest", action="store_true",
                        help="Seal on file hashes alone when no replay "
                             "manifest exists (non-replay namespaces)")
    parser.add_argument("--contract-manifest", type=Path, default=None,
                        help="Frozen contract manifest whose SHA-256 is bound "
                             "into the seal (and verified by --check)")
    parser.add_argument("--check", action="store_true",
                        help="Read-only: verify the existing seal matches the "
                             "current bytes (and the contract manifest, if "
                             "given); write nothing")
    parser.add_argument("--expect-records", type=int, default=None,
                        help="Externally-derived record count (e.g. the input "
                             "corpus size); sealing refuses if the namespace's "
                             "non-companion record count differs. Closes the "
                             "self-referential completeness gap: the replay "
                             "manifest lists only what exists, so a --limit "
                             "run would otherwise seal as 'complete'.")
    args = parser.parse_args(argv)

    namespace = args.namespace.resolve()
    if _is_under_results_evaluation(namespace):
        print(
            f"REFUSE: {namespace} is at or under results/evaluation; the "
            "immutable submitted corpus is never sealed by this tool.",
            file=sys.stderr,
        )
        return 2
    if not namespace.is_dir():
        print(f"REFUSE: namespace not found: {namespace}", file=sys.stderr)
        return 2

    files = _hash_namespace(namespace)
    if not files:
        print(f"REFUSE: namespace is empty: {namespace}", file=sys.stderr)
        return 2

    manifest_sha = None
    if args.contract_manifest is not None:
        if not args.contract_manifest.is_file():
            print(f"REFUSE: contract manifest not found: "
                  f"{args.contract_manifest}", file=sys.stderr)
            return 2
        manifest_sha = _sha256_file(args.contract_manifest)

    seal_path = namespace / SEAL_NAME

    if args.check:
        if not seal_path.is_file():
            print(f"CHECK FAIL: no seal at {seal_path}", file=sys.stderr)
            return 1
        try:
            existing = json.loads(seal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"CHECK FAIL: seal is unreadable: {seal_path}",
                  file=sys.stderr)
            return 1
        problems = []
        if existing.get("files") != files:
            problems.append("file inventory differs from the sealed inventory")
        if existing.get("aggregate_sha256") != _aggregate(files):
            problems.append("aggregate hash differs from the seal")
        if manifest_sha is not None:
            bound = (existing.get("contract_manifest") or {}).get("sha256")
            if bound is None:
                problems.append(
                    "seal predates contract-manifest binding (no "
                    "contract_manifest field); cannot verify the binding"
                )
            elif bound != manifest_sha:
                problems.append("seal is bound to a DIFFERENT contract manifest")
        if problems:
            for p in problems:
                print(f"CHECK FAIL: {p}", file=sys.stderr)
            return 1
        print(f"CHECK OK: {namespace} matches its seal "
              f"({len(files)} file(s))")
        return 0
    if seal_path.is_file():
        try:
            existing = json.loads(seal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(
                f"REFUSE: existing seal is unreadable: {seal_path}; "
                "refusing to replace it.",
                file=sys.stderr,
            )
            return 2
        if existing.get("files") == files:
            print(f"OK: {namespace} is already sealed with identical bytes.")
            return 0
        print(
            f"REFUSE: {namespace} is already sealed and the current bytes "
            "differ from the sealed inventory; resealing different bytes is "
            "forbidden.",
            file=sys.stderr,
        )
        return 2

    if args.expect_records is not None:
        record_count = sum(
            1 for rel in files if Path(rel).name not in NON_RECORD_FILES
        )
        if record_count != args.expect_records:
            print(
                f"REFUSE: namespace holds {record_count} record file(s) but "
                f"--expect-records demands {args.expect_records}; a partial "
                "(e.g. --limit) replay must not be sealed as complete.",
                file=sys.stderr,
            )
            return 1

    if (namespace / MANIFEST_NAME).is_file():
        problems = check_completeness(namespace, files)
        if problems:
            for p in problems:
                print(f"INCOMPLETE: {p}", file=sys.stderr)
            print(
                f"REFUSE: namespace fails completeness ({len(problems)} "
                "problem(s)); no seal written.",
                file=sys.stderr,
            )
            return 1
    elif not args.allow_no_manifest:
        print(
            f"REFUSE: no {MANIFEST_NAME} in {namespace}; pass "
            "--allow-no-manifest to seal on file hashes alone.",
            file=sys.stderr,
        )
        return 1

    seal = {
        "seal_version": SEAL_VERSION,
        "sealed_at": datetime.datetime.now(datetime.UTC)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sealed_by": "scripts/provenance/seal_result_namespace.py",
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": _aggregate(files),
    }
    if manifest_sha is not None:
        seal["contract_manifest"] = {
            "path": str(args.contract_manifest),
            "sha256": manifest_sha,
        }
    seal_path.write_text(json.dumps(seal, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(
        f"SEALED: {namespace} ({len(files)} file(s), "
        f"aggregate {seal['aggregate_sha256'][:16]})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
