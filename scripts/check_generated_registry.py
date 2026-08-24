#!/usr/bin/env python3
"""Validate .claude/generated-outputs.tsv, the registry read by generated-file-guard.sh.

The guard fails open on infrastructure problems, so a registry that has quietly rotted
would disable protection with no signal. This script is that signal. It is a mandatory
universal check in /validate.

Checks, per row:
  1. exactly 3 tab-separated fields
  2. mode is literally `block` or `warn` (no CR, no padding, no case variance) - the
     guard blocks a malformed row rather than degrading it to `warn`, so a bad mode
     turns into a surprise block at edit time
  3. every generator path named in the row exists on disk
  4. the glob matches at least one git-tracked file (a row matching nothing is either
     a typo or a stale entry, and both read as protection that is not there)

And file-level:
  5. no CR anywhere (a CRLF editor on the macOS half of this project)
  6. the file ends with a newline (`while read` drops an unterminated final line, which
     is exactly the line an "add a generator" commit appends)

Exit 0 = clean, 1 = problems found (printed to stdout, one per line).

`--uncovered` additionally lists git-tracked files under the directories the registry
already reaches into that NO row covers. That is advisory, not a failure: it is the
worklist for closing coverage gaps, not a claim that every file there must be registered.

`--fail-uncovered` (Task 3, gate E-03) is the enforcing sibling: an uncovered file
under `--dirs` is a FAILURE (exit 1). It scans tracked AND untracked files; with
`--changed-only` the scan narrows to files changed by the current work (git status:
staged, modified, or untracked). A supplied directory that is missing or empty is
tolerated — e.g. results/augmentation_final, whose producing task is deferred.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALID_MODES = {"block", "warn"}


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.splitlines()


def untracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.splitlines()


def changed_files(root: Path) -> list[str]:
    """Files changed by the current work: staged, modified, or untracked."""
    out = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain",
         "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    )
    paths: list[str] = []
    for line in out.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        # Renames are shown as "old -> new"; the new path is the changed one.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def in_dirs(path: str, dirs: list[str]) -> bool:
    return any(path.startswith(d.rstrip("/") + "/") for d in dirs)


def parse_rows(raw: str) -> list[tuple[int, str]]:
    """Return (1-based line number, line) for non-comment, non-blank lines."""
    rows = []
    for i, line in enumerate(raw.split("\n"), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append((i, line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncovered", action="store_true",
                    help="also list tracked files under --dirs that no row covers")
    ap.add_argument("--fail-uncovered", action="store_true",
                    help="enforcing mode: an uncovered tracked-or-untracked file "
                         "under --dirs is a failure (exit 1); missing or empty "
                         "supplied directories are tolerated")
    ap.add_argument("--changed-only", action="store_true",
                    help="with --fail-uncovered: scan only files changed by the "
                         "current work (staged, modified, or untracked)")
    ap.add_argument("--dirs", nargs="+",
                    default=["results/analysis", "results/augmentation"],
                    help="directories the --uncovered advisory scans "
                         "(default: the two named in CLAUDE.md invariant 2)")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repository root (default: this checkout; used by tests)")
    args = ap.parse_args()

    root = args.root.resolve()
    registry = root / ".claude" / "generated-outputs.tsv"

    if not registry.exists():
        print(f"FAIL: registry missing: {registry}")
        return 1

    # read_bytes, not read_text: text mode applies universal-newline translation, which
    # silently rewrites \r\n to \n and makes the CR check below unable to ever fire.
    raw = registry.read_bytes().decode("utf-8")
    problems: list[str] = []

    if "\r" in raw:
        problems.append("FAIL: registry contains CR characters; the guard would see a "
                        "mode of 'block\\r' and treat the row as malformed")
    if not raw.endswith("\n"):
        problems.append("FAIL: registry does not end with a newline; the guard's read "
                        "loop would drop the final row")

    files = tracked_files(root)
    globs: list[str] = []

    for lineno, line in parse_rows(raw):
        fields = line.split("\t")
        if len(fields) != 3:
            problems.append(f"FAIL: line {lineno}: expected 3 tab-separated fields, got {len(fields)}")
            continue

        glob, generator, mode = fields
        globs.append(glob)

        if mode not in VALID_MODES:
            problems.append(f"FAIL: line {lineno}: mode {mode!r} is not 'block' or 'warn'")

        # A row may name more than one generator, joined with '+'.
        for gen in (g.strip() for g in generator.split("+")):
            if not gen:
                problems.append(f"FAIL: line {lineno}: empty generator name")
            elif not (root / gen).exists():
                problems.append(f"FAIL: line {lineno}: generator does not exist: {gen}")

        if not any(fnmatch.fnmatch(f, glob) for f in files):
            problems.append(f"FAIL: line {lineno}: glob matches no tracked file: {glob}")

    for p in problems:
        print(p)

    if args.uncovered:
        uncovered = [
            f for f in files
            if in_dirs(f, args.dirs)
            and not any(fnmatch.fnmatch(f, g) for g in globs)
        ]
        if uncovered:
            print(f"\nADVISORY: {len(uncovered)} tracked files under registered directories "
                  f"are covered by no row:")
            for f in uncovered:
                print(f"  {f}")

    if args.fail_uncovered:
        # Tracked plus untracked; --changed-only narrows to current work.
        if args.changed_only:
            candidates = changed_files(root)
        else:
            candidates = files + untracked_files(root)
        # A missing or empty supplied directory is fine (a deferred producing
        # task, e.g. results/augmentation_final); only files that exist can be
        # uncovered.
        enforce_uncovered = sorted({
            f for f in candidates
            if in_dirs(f, args.dirs)
            and not any(fnmatch.fnmatch(f, g) for g in globs)
        })
        if enforce_uncovered:
            print(f"\nFAIL: {len(enforce_uncovered)} file(s) under enforced "
                  f"directories are covered by no registry row:")
            for f in enforce_uncovered:
                print(f"  {f}")
            problems.extend(f"uncovered: {f}" for f in enforce_uncovered)

    if problems:
        print(f"\n{len(problems)} problem(s) found.")
        return 1

    print(f"Registry OK: {len(globs)} rows, all globs match tracked files, "
          f"all generators exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
