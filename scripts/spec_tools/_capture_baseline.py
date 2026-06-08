#!/usr/bin/env python3
"""scripts.spec_tools._capture_baseline — private one-shot baseline helper.

Single-shot, side-effecting helper. NOT a contributor CLI. Run ONCE per spec
during the Tier 2 oracle-upgrade campaign; paste the suggested JSON snippet
into the spec by hand; commit the reference file + spec edit together.

USAGE
-----
    python3 scripts/spec_tools/_capture_baseline.py specs/<spec>.json \\
        --project-root .

CRITICAL — same-args invariant
-----------------------------
The helper MUST use exactly ``run.input_configurations.correctness.arguments``
from the spec. That is the SAME argument set Phase 3 eval passes to the
binary at verify-time. Capturing with any different argument set produces a
SHA-256 that will NOT match at eval time, causing every spec upgraded to
``file_hash`` to false-FAIL. There is NO ``--strategy`` flag. There is NO
``--arguments`` flag. There is no override surface. Misuse breaks Rule 12
(file_hash determinism) silently and campaign-wide.

Campaign Rule 12 (HANDOFF §3 — inviolable)
------------------------------------------
The helper runs the binary TWICE with identical arguments and compares the
SHA-256 of every candidate output file across both runs. On mismatch it
emits a LOUD warning: the spec is non-deterministic and MUST NOT be
upgraded to ``file_hash``. Recommend ``oracle_strength: "weak"`` +
Threats-to-Validity entry instead.

Output
------
stdout only. The helper does not mutate ``specs/``, ``manifest.jsonl``,
``results/``, or benchmark source code. Samyak inspects the suggested JSON
snippet and pastes it into the spec manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))

from harness.builder import build_spec  # noqa: E402
from harness.models import Status  # noqa: E402
from harness.runner import run_spec  # noqa: E402
from harness.spec_loader import load_spec, resolve_paths  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _snapshot_dir(working_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {relative_posix_path: (mtime_ns, size_bytes)} for every regular
    file under *working_dir* (recursive). Symlinks are skipped so snapshot
    comparison is not fooled by retargeted links."""
    snap: dict[str, tuple[int, int]] = {}
    if not working_dir.is_dir():
        return snap
    for f in working_dir.rglob("*"):
        if not f.is_file() or f.is_symlink():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        rel = f.relative_to(working_dir).as_posix()
        snap[rel] = (st.st_mtime_ns, st.st_size)
    return snap


def _diff_snapshots(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
    exclude_basenames: set[str],
) -> list[str]:
    """Return sorted relative paths that are new-in-*after* or whose mtime/size
    changed. Drops any file whose basename is in *exclude_basenames* (the
    spec's prompt_payload — inputs, never outputs)."""
    out: list[str] = []
    for rel, stats in after.items():
        if Path(rel).name in exclude_basenames:
            continue
        if rel not in before or before[rel] != stats:
            out.append(rel)
    return sorted(out)


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of *path* read in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _head_tail(text: str, lines: int = 20) -> str:
    """First + last *lines* of text joined by an elision marker."""
    rows = text.splitlines()
    if len(rows) <= lines * 2:
        return text
    elided = len(rows) - lines * 2
    return "\n".join(rows[:lines] + [f"... [{elided} lines elided] ..."] + rows[-lines:])


# Exit codes:
#   0 = OK (single helper run completed; user inspects the printed snippet)
#   2 = invalid input (spec file missing, correctness config missing)
#   3 = build failed (spec cannot be baselined until build is fixed)
EXIT_OK = 0
EXIT_INPUT = 2
EXIT_BUILD = 3


def capture_baseline(spec_path: Path, project_root: Path) -> int:
    """Run a single baseline capture. See module docstring for invariants.

    Prints human-readable report to stdout (+ warnings to stderr). Does NOT
    mutate ``specs/``, ``manifest.jsonl``, ``results/``, or benchmark source.
    """
    if not spec_path.is_file():
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        return EXIT_INPUT

    spec = load_spec(spec_path)
    resolved = resolve_paths(spec, project_root)["_resolved"]
    working_dir: Path = resolved["working_dir"]

    correctness = (
        spec.get("run", {}).get("input_configurations", {}).get("correctness")
    )
    if not correctness or "arguments" not in correctness:
        print(
            "ERROR: spec has no run.input_configurations.correctness.arguments "
            "(same-args invariant cannot be satisfied)",
            file=sys.stderr,
        )
        return EXIT_INPUT
    correctness_args = correctness["arguments"]
    unique_id = spec.get("identity", {}).get("unique_id", spec_path.stem)

    print(f"=== _capture_baseline: {unique_id} ===")
    print(f"spec:            {spec_path}")
    print(f"working_dir:     {working_dir}")
    print(f"correctness args: {correctness_args}   (same-args invariant — DO NOT override)")
    print()

    print("--- build ---")
    br = build_spec(spec, project_root)
    if br.status != Status.PASS:
        print(
            f"BUILD_FAIL: cannot capture baseline ({br.status.value}). "
            "Fix the build first, then re-run this helper.",
            file=sys.stderr,
        )
        stderr_tail = (br.stderr or "").splitlines()[-20:]
        if stderr_tail:
            print("\n".join(stderr_tail), file=sys.stderr)
        return EXIT_BUILD
    print(f"build: PASS ({br.duration_seconds:.2f}s)")
    print()

    # Exclude prompt_payload basenames from candidate-output detection —
    # those files are LLM inputs, never kernel outputs.
    prompt_payload = (spec.get("files") or {}).get("prompt_payload") or []
    exclude = {Path(p).name for p in prompt_payload}

    # Snapshot ONCE before run 1, then diff both runs against the same baseline.
    # This ensures run 2's candidate set matches run 1's even when the binary
    # re-writes identical bytes (mtime may be unchanged, and diff(before_2,
    # after_2) would otherwise be empty). Same-baseline diff correctly surfaces
    # every output file produced by EITHER run.
    # NOTE: scenario (d) — binary writes on run 1 only — will appear deterministic here (leftover bytes re-hashed). Acceptable limitation.
    baseline_snapshot = _snapshot_dir(working_dir)

    run_hashes: list[dict[str, str]] = []
    for idx in (1, 2):
        print(f"--- run {idx} ---")
        rr = run_spec(spec, project_root, configuration="correctness")
        after = _snapshot_dir(working_dir)
        cands = _diff_snapshots(baseline_snapshot, after, exclude)

        hashes: dict[str, str] = {}
        for c in cands:
            fp = working_dir / c
            try:
                hashes[c] = _sha256_file(fp)
            except OSError as exc:
                hashes[c] = f"<ERROR: {exc}>"

        if rr.status != Status.PASS:
            print(
                f"run {idx} NON-PASS: {rr.status.value} (exit={rr.exit_code}) — "
                "capture still recorded but inspect before using for file_hash",
                file=sys.stderr,
            )
        print(
            f"run {idx} exit={rr.exit_code} duration={rr.duration_seconds:.3f}s"
        )
        print(f"run {idx} stdout (head+tail):")
        print(_head_tail(rr.stdout or ""))
        print()
        run_hashes.append(hashes)

    hashes1, hashes2 = run_hashes[0], run_hashes[1]

    all_cands = sorted(set(hashes1) | set(hashes2))
    print("--- candidate outputs (diff vs prompt_payload exclusion) ---")
    if not all_cands:
        print("(no new / modified files detected — spec may rely on stdout)")

    deterministic: list[tuple[str, str]] = []
    nondeterministic: list[str] = []
    for c in all_cands:
        h1 = hashes1.get(c, "<not produced in run1>")
        h2 = hashes2.get(c, "<not produced in run2>")
        try:
            size = (working_dir / c).stat().st_size
        except OSError:
            size = -1
        print(f"  {c}   size={size}B")
        print(f"    run1 sha256: {h1}")
        print(f"    run2 sha256: {h2}")
        if h1 == h2 and not h1.startswith("<"):
            deterministic.append((c, h1))
        else:
            nondeterministic.append(c)
    print()

    if nondeterministic:
        banner = "*" * 72
        print(banner, file=sys.stderr)
        print(
            "!! NON-DETERMINISTIC OUTPUT — Rule 12 violation if file_hash used !!",
            file=sys.stderr,
        )
        for c in nondeterministic:
            print(f"   - {c}", file=sys.stderr)
        print(
            "RECOMMENDATION: set oracle_strength=\"weak\" and add a Threats-to-"
            "Validity note. DO NOT upgrade this spec to file_hash.",
            file=sys.stderr,
        )
        print(banner, file=sys.stderr)
        print()

    print("--- suggested JSON snippets (paste into spec manually) ---")
    if deterministic and not nondeterministic:
        # Slug = kernel_name from identity for the reference_files path.
        slug = spec.get("identity", {}).get("kernel_name", unique_id)
        strategies = [
            {"type": "file_hash", "path": c, "expected_sha256": h}
            for c, h in deterministic
        ]
        reference_files = [
            {
                "path": f"specs/references/{slug}/{Path(c).name}",
                "sha256": h,
                "description": f"Reference {Path(c).name} from _capture_baseline",
            }
            for c, h in deterministic
        ]
        print("verification.strategies (file_hash):")
        print(json.dumps(strategies, indent=2))
        print()
        print("verification.reference_files:")
        print(json.dumps(reference_files, indent=2))
    elif nondeterministic and deterministic:
        print(
            "MIXED DETERMINISM: at least one candidate output is non-deterministic "
            "— this spec must be marked oracle_strength=\"weak\" + Threats note. "
            "file_hash snippet WITHHELD to prevent paste-misuse."
        )
    else:
        print(
            "(no deterministic candidate outputs — do not use file_hash; "
            "either add a numeric_comparison on a stdout metric, or mark "
            "oracle_strength=\"weak\" with a Threats note)"
        )

    print()
    print("=== end baseline capture ===")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI entry (intentionally minimal — no --strategy, no --arguments overrides)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Private one-shot baseline capture helper. Runs the spec's binary "
            "twice with run.input_configurations.correctness.arguments and "
            "prints a paste-ready verification snippet. See module docstring."
        ),
        epilog=(
            "There is intentionally NO --strategy / --arguments override "
            "flag: the same-args invariant (module docstring) must be "
            "preserved or file_hash-upgraded specs will false-FAIL at eval."
        ),
    )
    parser.add_argument("spec", type=Path, help="Path to the spec JSON file")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help="Project root (default: two levels up from this script)",
    )
    args = parser.parse_args(argv)
    return capture_baseline(args.spec.resolve(), args.project_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
