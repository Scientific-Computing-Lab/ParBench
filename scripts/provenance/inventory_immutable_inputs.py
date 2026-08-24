#!/usr/bin/env python3
"""Inventory the immutable inputs of the ParBench engineering campaign.

Produces (or verifies against) a manifest recording:
  - every file under ``results/evaluation`` with its SHA-256,
  - every benchmark source file referenced by a spec (resolved through the
    spec's ``provenance.repo_root``/``source_path``), with its SHA-256,
  - nested benchmark-tree HEADs and dirty status where a tree carries history.

This manifest, not parent-repository ``git diff``, is the before-state for
every later immutability check. Manifests are host-specific: a benchmark tree
absent on one host records its files as MISSING there.

The script only ever reads ``results/evaluation``; it never writes into it.
The ``archives`` section of an existing output manifest is preserved on
regeneration: it is owned by ``make_private_archive.py``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import socket
import subprocess
import sys

BENCHMARK_TREES = (
    "rodinia",
    "HeCBench-master",
    "mixbench/mixbench-src",
    "xsbench/xsbench-src",
    "rsbench/rsbench-src",
)

SPEC_FILE_LISTS = ("prompt_payload", "support_files", "verification_only")

MISSING = "MISSING"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate_sha256(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\t{files[rel]}\n".encode())
    return h.hexdigest()


def _git(project_root: pathlib.Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def inventory_evaluation_results(project_root: pathlib.Path) -> dict:
    root = project_root / "results" / "evaluation"
    files: dict[str, str] = {}
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files[p.relative_to(project_root).as_posix()] = sha256_file(p)
    return {
        "root": "results/evaluation",
        "root_exists": root.is_dir(),
        "file_count": len(files),
        "aggregate_sha256": aggregate_sha256(files),
        "files": files,
    }


def inventory_benchmark_trees(project_root: pathlib.Path) -> dict:
    trees: dict[str, dict] = {}
    for name in BENCHMARK_TREES:
        tree = project_root / name
        # `-e`, not `-d`: a submodule's .git is a file. Never use
        # `git rev-parse --git-dir` here -- it walks UP into the parent repo.
        has_git = (tree / ".git").exists()
        entry: dict = {"exists": tree.is_dir(), "has_git": has_git,
                       "head": None, "dirty_files": []}
        if has_git:
            entry["head"] = _git(tree, "rev-parse", "HEAD")
            porcelain = _git(tree, "status", "--porcelain")
            if porcelain:
                entry["dirty_files"] = sorted(porcelain.splitlines())
        trees[name] = entry
    return trees


# The exact source-file extensions benchmark_characterization.grep_dir globs,
# and the API specs it consults per corpus kernel. Kept here (a provenance
# module, not the analysis computation) so both the generator and the
# independent verifier bind the EXACT recursive read set without importing the
# analysis grep logic (gate5 finding 3).
CHARACTERIZATION_SOURCE_EXTENSIONS = ("*.cu", "*.c", "*.cpp", "*.h", "*.hpp", "*.cl")
CHARACTERIZATION_APIS = ("cuda", "omp", "omp_target", "opencl")

# Independent copy of benchmark_characterization.GENERATED_BUILD_DIR_MARKERS
# (this provenance module never imports analysis code). Files inside a generated
# CMake/out-of-source build tree are mutable build OUTPUTS, not benchmark
# source; benchmark_characterization.grep_dir now skips them, so this inventory -
# which must bind EXACTLY what characterization reads - excludes them too
# (gate15 finding 1). A source test keeps the two copies equal.
GENERATED_BUILD_DIR_MARKERS = frozenset({"build", "CMakeFiles"})


def is_generated_build_path(relpath) -> bool:
    """True if ``relpath`` (relative to the scanned source directory) sits inside
    a generated build tree. Mirrors benchmark_characterization.is_generated_
    build_path exactly (gate15 finding 1)."""
    return any(part in GENERATED_BUILD_DIR_MARKERS
               for part in pathlib.Path(relpath).parts)


def inventory_characterization_source_files(project_root: pathlib.Path) -> dict:
    """Inventory the EXACT recursive file set benchmark_characterization reads.

    ``benchmark_characterization.grep_dir`` does not read the spec-declared file
    list; it ``rglob``s every ``CHARACTERIZATION_SOURCE_EXTENSIONS`` file under
    each corpus kernel's ``provenance.repo_root``/``source_path`` directory,
    EXCLUDING generated build trees (``is_generated_build_path``). That set - a
    superset of the spec-referenced files, but with mutable build outputs
    removed - is what actually feeds the tier detection, so it is what the
    evidence package must bind (gate5 finding 3; build-tree exclusion added in
    gate15 finding 1). This reproduces that enumeration independently of the
    analysis grep code so the verifier can recompute and detect any
    added/removed/mutated input file.
    """
    # CORPUS_KERNELS is a static corpus definition (a data constant), not the
    # analysis computation; import it lazily. Resolve it from THIS module's own
    # repo (parents[2]/scripts/analysis), not project_root, so the corpus list is
    # deterministic regardless of the project_root passed in.
    analysis_dir = str(pathlib.Path(__file__).resolve().parents[2]
                       / "scripts" / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from sloc_analysis import CORPUS_KERNELS  # noqa: E402

    spec_root = project_root / "specs"
    files: dict[str, str] = {}
    dirs_seen: set[str] = set()
    for suite, kernel in CORPUS_KERNELS:
        for api in CHARACTERIZATION_APIS:
            spec_path = spec_root / f"{suite}-{kernel}-{api}.json"
            if not spec_path.is_file():
                continue
            try:
                spec = json.loads(spec_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            prov = spec.get("provenance") or {}
            repo_root = prov.get("repo_root")
            source_path = prov.get("source_path")
            if not repo_root or source_path is None:
                continue
            src_dir = (project_root / repo_root / source_path)
            if not src_dir.is_dir():
                continue
            dirs_seen.add(src_dir.resolve().as_posix())
            for ext in CHARACTERIZATION_SOURCE_EXTENSIONS:
                for fpath in src_dir.rglob(ext):
                    if not fpath.is_file():
                        continue
                    # Skip generated build-tree artifacts; grep_dir skips them
                    # too, so the inventory keeps binding exactly what
                    # characterization reads (gate15 finding 1).
                    if is_generated_build_path(fpath.relative_to(src_dir)):
                        continue
                    key = fpath.resolve().relative_to(project_root.resolve()).as_posix()
                    files.setdefault(key, sha256_file(fpath))
    return {
        "file_count": len(files),
        "dir_count": len(dirs_seen),
        "aggregate_sha256": aggregate_sha256(files),
        "files": files,
    }


def inventory_spec_referenced_files(project_root: pathlib.Path) -> dict:
    spec_root = project_root / "specs"
    files: dict[str, str] = {}
    spec_errors: list[str] = []
    for spec_path in sorted(spec_root.glob("*.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            spec_errors.append(f"{spec_path.name}: {exc}")
            continue
        prov = spec.get("provenance") or {}
        repo_root = prov.get("repo_root")
        source_path = prov.get("source_path")
        if not repo_root or source_path is None:
            continue
        file_lists = spec.get("files") or {}
        for list_name in SPEC_FILE_LISTS:
            for fname in file_lists.get(list_name) or []:
                rel = pathlib.PurePosixPath(repo_root) / source_path / fname
                key = rel.as_posix()
                if key in files:
                    continue
                target = project_root / rel
                if target.is_file():
                    files[key] = sha256_file(target)
                elif target.is_dir():
                    # a spec may reference a whole support/data directory;
                    # inventory every file inside it
                    for sub in sorted(target.rglob("*")):
                        if sub.is_file():
                            sub_key = (rel / sub.relative_to(target)).as_posix()
                            files.setdefault(sub_key, sha256_file(sub))
                else:
                    files[key] = MISSING
    hashed = {k: v for k, v in files.items() if v != MISSING}
    return {
        "file_count": len(files),
        "hashed_count": len(hashed),
        "missing_count": len(files) - len(hashed),
        "aggregate_sha256": aggregate_sha256(hashed),
        "files": files,
        "spec_errors": spec_errors,
    }


def build_inventory(project_root: pathlib.Path) -> dict:
    return {
        "inventory_version": 1,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "git": {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "status_short": sorted(
                (_git(project_root, "status", "--short") or "").splitlines()
            ),
            "submodules": (_git(project_root, "submodule", "status") or "").splitlines(),
        },
        "evaluation_results": inventory_evaluation_results(project_root),
        "benchmark_sources": {
            "trees": inventory_benchmark_trees(project_root),
            "spec_referenced": inventory_spec_referenced_files(project_root),
        },
        "archives": {},
    }


def _diff_file_maps(recorded: dict[str, str], current: dict[str, str],
                    tree_roots_absent: set[str]) -> tuple[list[str], list[str]]:
    """Return (failures, warnings) comparing recorded vs current file hashes.

    A file whose entire benchmark tree root is absent on this host is a
    host-property warning, not a mutation; a file missing while its tree
    exists is a failure.
    """
    failures: list[str] = []
    warnings: list[str] = []
    for rel, rec_hash in sorted(recorded.items()):
        cur_hash = current.get(rel)
        if cur_hash == rec_hash:
            continue
        tree_absent = any(rel.startswith(root + "/") for root in tree_roots_absent)
        if cur_hash in (None, MISSING) and tree_absent:
            warnings.append(f"absent-tree on this host: {rel}")
        elif cur_hash is None:
            failures.append(f"removed: {rel}")
        elif cur_hash == MISSING and rec_hash != MISSING:
            failures.append(f"missing: {rel} (recorded {rec_hash[:12]})")
        elif rec_hash == MISSING:
            warnings.append(f"newly present (was MISSING when recorded): {rel}")
        else:
            failures.append(f"changed: {rel} {rec_hash[:12]} -> {cur_hash[:12]}")
    for rel in sorted(set(current) - set(recorded)):
        failures.append(f"added: {rel}")
    return failures, warnings


def verify(project_root: pathlib.Path, manifest_path: pathlib.Path, scope: str) -> int:
    recorded = json.loads(manifest_path.read_text())
    failures: list[str] = []
    warnings: list[str] = []

    if scope in ("all", "evaluation-results"):
        rec = recorded["evaluation_results"]
        cur = inventory_evaluation_results(project_root)
        f, w = _diff_file_maps(rec["files"], cur["files"], set())
        failures += [f"evaluation-results: {m}" for m in f]
        warnings += [f"evaluation-results: {m}" for m in w]

    if scope in ("all", "benchmark-sources"):
        rec_src = recorded["benchmark_sources"]
        cur_trees = inventory_benchmark_trees(project_root)
        cur_src = inventory_spec_referenced_files(project_root)
        absent_roots = {name for name, t in cur_trees.items() if not t["exists"]}
        f, w = _diff_file_maps(
            rec_src["spec_referenced"]["files"], cur_src["files"], absent_roots
        )
        failures += [f"benchmark-sources: {m}" for m in f]
        warnings += [f"benchmark-sources: {m}" for m in w]
        for name, rec_tree in rec_src["trees"].items():
            cur_tree = cur_trees.get(name, {})
            if not cur_tree.get("exists"):
                if rec_tree.get("exists"):
                    warnings.append(f"tree absent on this host: {name}")
                continue
            if rec_tree.get("head") and cur_tree.get("head") != rec_tree["head"]:
                failures.append(
                    f"tree HEAD changed: {name} {rec_tree['head']} -> {cur_tree.get('head')}"
                )

    for msg in warnings:
        print(f"WARN  {msg}")
    for msg in failures:
        print(f"FAIL  {msg}")
    if failures:
        print(f"VERIFY FAILED: {len(failures)} mismatch(es), {len(warnings)} warning(s)")
        return 1
    print(f"VERIFY OK: scope={scope}, {len(warnings)} warning(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True, type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path,
                    help="write the inventory manifest to this path")
    ap.add_argument("--verify-against", type=pathlib.Path,
                    help="verify current state against an existing manifest")
    ap.add_argument("--scope", choices=("all", "evaluation-results", "benchmark-sources"),
                    default="all")
    args = ap.parse_args(argv)

    project_root = args.project_root.resolve()
    if args.verify_against:
        return verify(project_root, args.verify_against, args.scope)
    if not args.output:
        ap.error("one of --output or --verify-against is required")

    # Never write into a sealed namespace (fixcheck2 finding, 2026-08-12).
    probe = args.output.resolve().parent
    while True:
        if (probe / ".parbench-seal.json").is_file():
            print(f"REFUSE: output {args.output} is inside the sealed "
                  f"namespace {probe}.", file=sys.stderr)
            return 2
        if probe.parent == probe:
            break
        probe = probe.parent

    inventory = build_inventory(project_root)
    if args.output.is_file():
        try:
            previous = json.loads(args.output.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("archives"):
            inventory["archives"] = previous["archives"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=1, sort_keys=True) + "\n")
    ev = inventory["evaluation_results"]
    src = inventory["benchmark_sources"]["spec_referenced"]
    print(f"Wrote {args.output}")
    print(f"  evaluation-results: {ev['file_count']} files, "
          f"aggregate {ev['aggregate_sha256'][:16]}")
    print(f"  spec-referenced sources: {src['hashed_count']} hashed, "
          f"{src['missing_count']} missing, aggregate {src['aggregate_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
