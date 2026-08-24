#!/usr/bin/env python3
"""Deterministic project-health checks for ParBench.

Absorbs the fixed shell checks previously spread across four archived
review agents (Phase 1B, 2026-08-05): build-validator (lint, test
collection, import smoke), regression-checker (spec/test/manifest
baselines, key infra files), test-synthesizer (changed-file import /
spec / shell-syntax / frontmatter checks), and spec-auditor (the
deterministic parts: slug regex, category enum, manifest cross-check).

`.claude/agents/verify-app.md` runs this script and interprets FAIL/WARN
output; the judgment calls each archived agent used to make in prose
(e.g. "is this schema error count material?") stay agent-side.

DEFERRED BY DESIGN (Codex review, 2026-08-05): check_import_smoke() covers only
the two package roots (scripts.evaluation, scripts.spec_tools), not every
changed module — actually importing each changed file would execute arbitrary
top-level code from a git diff. py_compile (syntax-only, argv-only invocation)
plus the two package-root imports is the deliberate scope; do not widen it to
"import every changed .py file" without addressing that execution risk.

Usage:
    python3 scripts/health_check.py            # run all checks
    python3 scripts/health_check.py --self-test # positive/negative controls

Exit 0 if every check is PASS or WARN; exit 1 if any check is FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
)

# --- Baselines (see .claude/rules/known-issues.md) -------------------------

RODINIA_SPEC_BASELINE = 60
XSBENCH_SPEC_MIN = 4
UNIT_TEST_MIN = 15
KEY_INFRA_FILES = [
    "harness/__main__.py",
    "harness/cli.py",
    "scripts/validate_schema.py",
    "c_augmentation/test_transforms.py",
]
# The 5 deleted phantom Rodinia specs (manifest.jsonl is append-only, so
# their manifest entries remain). Each contributes 1 "spec_file not found"
# + 1 "source_dir not found" manifest error = 2 identities per spec.
PHANTOM_SPECS = {
    "gaussian-omp",
    "huffman-omp",
    "huffman-opencl",
    "hybridsort-omp",
    "mummergpu-opencl",
}
# Direct tree-presence probes, one per suite whose "-src" tree can legitimately be
# absent (gitignored working copy, or an uninitialized submodule in a worktree —
# see known-issues.md "Git Worktrees and Submodules"). Classification is by CAUSE
# (is the tree actually there?), never by host path prefix — a host guess cannot
# know which trees a given checkout happens to have populated.
SUITE_TREE_PRESENT = {
    "rodinia/": lambda root: (root / "rodinia" / "rodinia-src" / ".git").exists(),
    "HeCBench-master/": lambda root: (root / "HeCBench-master" / "src").is_dir(),
    "xsbench/": lambda root: (root / "xsbench" / "xsbench-src").is_dir(),
    "rsbench/": lambda root: (root / "rsbench" / "rsbench-src").is_dir(),
    "mixbench/": lambda root: (root / "mixbench" / "mixbench-src").is_dir(),
}
VALID_CATEGORIES = {
    "ml",
    "graph",
    "physics",
    "linear_algebra",
    "stencil",
    "reduction",
    "sort",
    "molecular_dynamics",
    "image",
    "crypto",
    "financial",
    "other",
}
UNIQUE_ID_RE = re.compile(r"^[a-z0-9_]+-[a-z0-9_][a-z0-9_-]*-[a-z0-9_]+$")


class ChangedFilesUnavailable(RuntimeError):
    """Raised when `git diff --name-status HEAD` fails — callers must treat this
    as FAIL, never as an empty (= "nothing changed") changed-file list."""


class Result:
    def __init__(self, name: str, status: str, detail: str = ""):
        self.name = name
        self.status = status  # PASS / WARN / FAIL
        self.detail = detail

    def line(self) -> str:
        s = f"[{self.status}] {self.name}"
        if self.detail:
            s += f" — {self.detail}"
        return s


def _run(cmd: list[str], cwd: Path = PROJECT_ROOT, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


def _venv_python() -> list[str]:
    """Prefer the project venv's python3 if present; else the ambient one."""
    venv_py = PROJECT_ROOT / "env_parbench" / "bin" / "python3"
    return [str(venv_py)] if venv_py.exists() else [sys.executable]


def _parse_changed_files(name_status_output: str) -> list[str]:
    """Parse `git diff --name-status` output, dropping deleted paths (a per-file
    validator dispatched at a deleted path would misread or false-fail) and
    resolving renames/copies (status R/C carries old\\tnew — keep the new path)."""
    files = []
    for line in name_status_output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("D"):
            continue
        files.append(parts[-1])
    return files


def _decide_changed_files(returncode: int, stdout: str, stderr: str) -> list[str]:
    """A nonzero exit means git failed to compute the diff — that must never be
    conflated with an empty (= "nothing changed") diff. Raise instead."""
    if returncode != 0:
        raise ChangedFilesUnavailable(
            f"git diff --name-status HEAD exited {returncode}: {(stderr or stdout).strip()[:200]}"
        )
    return _parse_changed_files(stdout)


def _changed_files() -> list[str]:
    r = _run(["git", "diff", "--name-status", "HEAD"])
    return _decide_changed_files(r.returncode, r.stdout, r.stderr)


# --- Checks ------------------------------------------------------------


def _classify_lint(violation_lines: list[str], changed_py: set[str]) -> Result:
    """FAIL only on a violation in a file this diff touches; pre-existing debt
    elsewhere in the repo (e.g. c_augmentation/augment_dataset.py:1945, untouched
    since March) is reported as WARN with the file list, not a hard block."""
    in_diff = [line for line in violation_lines if any(line.startswith(f + ":") for f in changed_py)]
    if in_diff:
        return Result("lint (ruff)", "FAIL", f"{len(in_diff)} issue(s) in changed files, e.g. {in_diff[0][:180]}")
    if violation_lines:
        return Result(
            "lint (ruff)",
            "WARN",
            f"{len(violation_lines)} pre-existing issue(s) outside this diff (e.g. {violation_lines[0][:150]})",
        )
    return Result("lint (ruff)", "PASS")


def _judge_lint(returncode: int, stdout: str, stderr: str, changed_py: set[str]) -> Result:
    if returncode == 0:
        return Result("lint (ruff)", "PASS")
    violation_lines = [line for line in stdout.splitlines() if line.strip()]
    if not violation_lines:
        # ruff crashed (config error, etc.) and put its diagnostics on stderr —
        # zero parseable violation lines must not read as a clean PASS.
        tail = (stdout + stderr).strip().splitlines()
        return Result(
            "lint (ruff)",
            "FAIL",
            f"ruff exited {returncode} with 0 parseable violations — likely crashed: {tail[-1][:200] if tail else '(no output)'}",
        )
    return _classify_lint(violation_lines, changed_py)


def _ruff() -> list[str]:
    """Prefer the project venv's pinned ruff; a PATH ruff of another version
    applies different default rules and produces phantom failures."""
    venv_ruff = PROJECT_ROOT / "env_parbench" / "bin" / "ruff"
    return [str(venv_ruff)] if venv_ruff.exists() else ["ruff"]


def check_lint() -> Result:
    try:
        changed_py = {f for f in _changed_files() if f.endswith(".py")}
    except ChangedFilesUnavailable as e:
        return Result("lint (ruff)", "FAIL", f"cannot scope lint - {e}")
    r = _run(_ruff() + ["check", ".", "--output-format", "concise"])
    return _judge_lint(r.returncode, r.stdout, r.stderr, changed_py)


def check_test_collection() -> Result:
    """Repo-wide (~737 tests): --collect-only ONLY, by design — actually running
    the full tree is too slow for a quick health check. This means a repo-wide
    test FAILURE (as opposed to a collection error) is NOT caught here; the one
    suite fast and deterministic enough to run for real is check_unit_tests()
    below, which restores the parity the archived verify-app had."""
    r = _run(_venv_python() + ["-m", "pytest", "--collect-only", "-q"], timeout=120)
    if r.returncode == 0:
        return Result("test collection", "PASS")
    tail = (r.stdout + r.stderr).strip().splitlines()
    return Result("test collection", "FAIL", tail[-1] if tail else "collection error")


def check_import_smoke() -> Result:
    """ALL listed packages must import cleanly — OR-semantics would let a broken
    second module hide behind a working first one."""
    modules = ("scripts.evaluation", "scripts.spec_tools")
    failures = []
    for pkg in modules:
        r = _run(_venv_python() + ["-c", f"import {pkg}"])
        if r.returncode != 0:
            failures.append(pkg)
    if failures:
        return Result("import smoke", "FAIL", f"failed to import: {', '.join(failures)}")
    return Result("import smoke", "PASS", ", ".join(modules))


def check_spec_counts() -> list[Result]:
    results = []
    rodinia_n = len(list((PROJECT_ROOT / "specs").glob("rodinia-*.json")))
    if rodinia_n < RODINIA_SPEC_BASELINE:
        results.append(Result("rodinia spec count", "FAIL", f"{rodinia_n} < baseline {RODINIA_SPEC_BASELINE}"))
    elif rodinia_n > RODINIA_SPEC_BASELINE:
        results.append(Result("rodinia spec count", "WARN", f"{rodinia_n} > baseline {RODINIA_SPEC_BASELINE}"))
    else:
        results.append(Result("rodinia spec count", "PASS", str(rodinia_n)))

    xsbench_n = len(list((PROJECT_ROOT / "specs").glob("xsbench-*.json")))
    if xsbench_n < XSBENCH_SPEC_MIN:
        results.append(Result("xsbench spec count", "FAIL", f"{xsbench_n} < min {XSBENCH_SPEC_MIN}"))
    else:
        results.append(Result("xsbench spec count", "PASS", str(xsbench_n)))
    return results


def check_unit_tests(target: str = "c_augmentation/test_transforms.py") -> Result:
    """ACTUALLY RUNS the suite (15 tests, ~0.5s for the default target) rather
    than only collecting it — --collect-only would let a real failing test
    inside this fast suite through as PASS. Restores parity with the archived
    verify-app, which ran `pytest c_augmentation/test_transforms.py -v`.
    `target` is overridable so self-test can point this at a fixture without
    touching the real suite."""
    r = _run(_venv_python() + ["-m", "pytest", target, "-q"], timeout=60)
    tail = (r.stdout + r.stderr).strip().splitlines()
    m = re.search(r"(\d+) passed", r.stdout)
    count = int(m.group(1)) if m else 0
    label = f"unit tests ({target})"
    if r.returncode != 0:
        return Result(label, "FAIL", tail[-1] if tail else "test run failed")
    if count < UNIT_TEST_MIN:
        return Result(label, "FAIL", f"{count} passed < min {UNIT_TEST_MIN}")
    return Result(label, "PASS", f"{count} passed")


def check_key_infra_files() -> Result:
    missing = [f for f in KEY_INFRA_FILES if not (PROJECT_ROOT / f).is_file()]
    if missing:
        return Result("key infra files", "FAIL", f"missing: {', '.join(missing)}")
    return Result("key infra files", "PASS", f"{len(KEY_INFRA_FILES)}/{len(KEY_INFRA_FILES)} present")


def check_manifest() -> list[Result]:
    """manifest.jsonl is append-only (CLAUDE.md invariant 1). The baseline is
    HEAD's own line count, read via git — never a literal constant, which goes
    stale on the next legitimate append (exactly the class check_stale_counts.py
    exists to catch)."""
    results = []
    manifest = PROJECT_ROOT / "manifest.jsonl"
    if not manifest.is_file():
        return [Result("manifest present", "FAIL", "manifest.jsonl missing")]
    working_count = sum(1 for _ in manifest.open())

    head = _run(["git", "show", "HEAD:manifest.jsonl"])
    head_count = len(head.stdout.splitlines()) if head.returncode == 0 else 0
    if working_count < head_count:
        results.append(
            Result("manifest line count", "FAIL", f"{working_count} < HEAD's {head_count} — lines disappeared")
        )
    else:
        results.append(Result("manifest line count", "PASS", f"{working_count} lines (HEAD: {head_count})"))

    r = _run(["git", "diff", "HEAD", "--", "manifest.jsonl"])
    deleted = [line for line in r.stdout.splitlines() if line.startswith("-") and not line.startswith("---")]
    if deleted:
        results.append(Result("manifest append-only", "FAIL", f"{len(deleted)} line(s) deleted from manifest.jsonl"))
    else:
        results.append(Result("manifest append-only", "PASS"))

    results.append(_judge_manifest_categories(manifest.read_text()))
    return results


def _judge_manifest_categories(manifest_text: str) -> Result:
    """The category enum (spec-conventions.md rule 33) lives on MANIFEST entries,
    not on spec JSONs - no spec carries a category field (ruling 34 fix 4,
    2026-08-15: the old spec-side check validated a field that never exists)."""
    bad = []
    for i, line in enumerate(manifest_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            category = json.loads(line).get("category", "")
        except json.JSONDecodeError:
            bad.append(f"line {i}: unparseable")
            continue
        if category not in VALID_CATEGORIES:
            bad.append(f"line {i}: '{category}'")
    if bad:
        return Result("manifest categories", "FAIL", f"{len(bad)} outside the enum, e.g. {bad[0]}")
    return Result("manifest categories", "PASS", "all entries in the 12-value enum")


def check_python_import(path: Path) -> Result:
    """Verify a changed .py file is syntactically valid. This is py_compile —
    syntax-only; it does NOT execute the module, so it cannot catch a broken
    top-level import or other runtime error. Deeper checking for changed files
    is check_import_smoke() (module-level) and pytest collection (test files).

    Invoked as `python3 -m py_compile -- <path>` with path as its own argv
    element — never interpolated into a `-c` source string, where a
    git-controlled filename containing a quote could inject code."""
    r = _run(_venv_python() + ["-m", "py_compile", "--", str(path)])
    if r.returncode == 0:
        return Result(f"python compile: {path}", "PASS")
    return Result(f"python compile: {path}", "FAIL", (r.stderr or r.stdout).strip().splitlines()[-1][:200])


def check_spec_json(path: Path) -> list[Result]:
    """Deterministic parts of spec-auditor: JSON validity, slug, category, manifest xref, schema."""
    results = []
    full = PROJECT_ROOT / path
    try:
        data = json.loads(full.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [Result(f"spec JSON valid: {path}", "FAIL", str(e))]
    results.append(Result(f"spec JSON valid: {path}", "PASS"))

    unique_id = data.get("identity", {}).get("unique_id", "")
    if not UNIQUE_ID_RE.match(unique_id):
        results.append(Result(f"spec slug: {path}", "FAIL", f"unique_id '{unique_id}' fails slug regex"))
    else:
        results.append(Result(f"spec slug: {path}", "PASS"))

    # No spec JSON carries a category field (checked 2026-08-15: 0 of 206 at
    # either metadata.category or top level) - the enum is validated on
    # manifest entries by _judge_manifest_categories() instead.

    manifest = PROJECT_ROOT / "manifest.jsonl"
    if manifest.is_file() and unique_id:
        found = any(unique_id in line for line in manifest.open())
        results.append(
            Result(f"spec manifest xref: {path}", "PASS" if found else "FAIL", "" if found else f"'{unique_id}' not in manifest.jsonl")
        )

    r = _run(
        _venv_python() + ["scripts/validate_schema.py", "--spec", str(path)],
        timeout=60,
    )
    results.append(Result(f"spec schema: {path}", "PASS" if r.returncode == 0 else "FAIL", "" if r.returncode == 0 else (r.stdout + r.stderr).strip()[-200:]))
    return results


def check_shell_syntax(path: Path) -> Result:
    r = _run(["bash", "-n", str(PROJECT_ROOT / path)])
    if r.returncode == 0:
        return Result(f"shell syntax: {path}", "PASS")
    return Result(f"shell syntax: {path}", "FAIL", r.stderr.strip()[:200])


def check_agent_frontmatter(path: Path) -> Result:
    full = PROJECT_ROOT / path
    content = full.read_text()
    m = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return Result(f"agent frontmatter: {path}", "FAIL", "no frontmatter block")
    fm: dict[str, str] = {}
    for line in m.group(1).strip().split("\n"):
        kv = line.split(":", 1)
        if len(kv) == 2:
            fm[kv[0].strip()] = kv[1].strip().strip("\"'")
    required = ["name", "description", "tools", "model"]
    missing = [k for k in required if k not in fm]
    if missing:
        return Result(f"agent frontmatter: {path}", "FAIL", f"missing fields: {missing}")
    model = fm.get("model", "")
    if model == "opus":
        return Result(
            f"agent frontmatter: {path}",
            "FAIL",
            "bare 'opus' alias is banned project-wide — pin an explicit claude-opus-4-8[1m]-style id",
        )
    if model not in ("sonnet", "haiku") and not model.startswith("claude-"):
        return Result(f"agent frontmatter: {path}", "FAIL", f"invalid model: {model}")
    return Result(f"agent frontmatter: {path}", "PASS")


def check_changed_files() -> list[Result]:
    try:
        changed = _changed_files()
    except ChangedFilesUnavailable as e:
        return [Result("changed files", "FAIL", f"cannot enumerate changed files - {e}")]
    results: list[Result] = []
    for f in changed:
        p = Path(f)
        if f.endswith(".py") and not p.name.startswith("test_"):
            results.append(check_python_import(p))
        elif f.startswith("specs/") and f.endswith(".json"):
            results.extend(check_spec_json(p))
        elif f.endswith(".sh"):
            results.append(check_shell_syntax(p))
        elif f.startswith(".claude/agents/") and f.endswith(".md"):
            results.append(check_agent_frontmatter(p))
    return results


_API_DIR_NAME = {"omp": "openmp", "opencl": "opencl", "cuda": "cuda"}


def _phantom_spec_patterns() -> list[str]:
    """Every substring a phantom spec's error line can contain: the deleted
    spec_file name, and its source_dir path under rodinia/rodinia-src/<api>/<kernel>."""
    patterns = []
    for slug in PHANTOM_SPECS:
        kernel, api = slug.rsplit("-", 1)
        patterns.append(f"rodinia-{kernel}-{api}.json")
        dir_name = _API_DIR_NAME.get(api, api)
        patterns.append(f"/{dir_name}/{kernel}'")
    return patterns


def classify_schema_errors(
    error_lines: list[str], *, project_root: Path | None = None
) -> tuple[list[str], list[str]]:
    """Split validate_schema.py --all error lines into (expected, unexplained).

    Expected = traces to one of the 5 deleted phantom Rodinia specs (always —
    regardless of tree state, since those specs are gone for good), OR a
    source_dir miss under a suite whose tree is CONFIRMED ABSENT by a direct
    probe (SUITE_TREE_PRESENT), never by guessing from a host path prefix.
    A source_dir miss under a suite whose tree IS present is unexplained — that
    is a real regression, not a platform artifact. Everything else is also
    unexplained and should FAIL.
    """
    root = project_root or PROJECT_ROOT
    expected, unexplained = [], []
    phantom_patterns = _phantom_spec_patterns()
    absent_suite_dirs = [prefix for prefix, present in SUITE_TREE_PRESENT.items() if not present(root)]
    for line in error_lines:
        if any(p in line for p in phantom_patterns) or (
            "does not exist on disk" in line and any(d in line for d in absent_suite_dirs)
        ):
            expected.append(line)
        else:
            unexplained.append(line)
    return expected, unexplained


def _judge_schema_baseline(
    returncode: int, stdout: str, stderr: str, *, project_root: Path | None = None
) -> Result:
    root = project_root or PROJECT_ROOT
    error_lines = [line for line in stdout.splitlines() if line.strip().startswith("•")]
    n = len(error_lines)
    if returncode != 0 and n == 0:
        # The validator crashed (traceback, missing dep, etc.) rather than running
        # and finding zero errors — a bare `n == 0` here would misreport as PASS.
        tail = (stdout + stderr).strip().splitlines()
        return Result(
            "schema baseline",
            "FAIL",
            f"validate_schema.py exited {returncode} with 0 parseable errors — likely crashed: "
            f"{tail[-1][:200] if tail else '(no output)'}",
        )
    _, unexplained = classify_schema_errors(error_lines, project_root=root)
    absent_trees = [prefix.rstrip("/") for prefix, present in SUITE_TREE_PRESENT.items() if not present(root)]
    tree_state = f"absent trees: {', '.join(absent_trees) if absent_trees else 'none'}"
    if unexplained:
        return Result(
            "schema baseline",
            "FAIL",
            f"{n} errors ({tree_state}); {len(unexplained)} unexplained (first: {unexplained[0][:150]})",
        )
    # Raw count is informational only — the identity check above is the actual gate
    # and is machine-independent by construction (CLAUDE.md invariant 7).
    return Result("schema baseline", "WARN" if n else "PASS", f"{n} errors ({tree_state}), all trace to known phantom specs / absent trees")


def check_schema_baseline() -> Result:
    r = _run(_venv_python() + ["scripts/validate_schema.py", "--all"], timeout=120)
    return _judge_schema_baseline(r.returncode, r.stdout, r.stderr)


def check_spec_argc() -> Result:
    """Spec run args vs source argc checks (scripts/spec_tools/check_spec_argc.py)."""
    r = _run(_venv_python() + ["scripts/spec_tools/check_spec_argc.py", "--all"], timeout=120)
    summary = ""
    for line in r.stdout.splitlines():
        if line.startswith("checked="):
            summary = line
    if r.returncode == 0:
        return Result("spec argc vs source", "PASS", summary)
    return Result("spec argc vs source", "FAIL", summary or (r.stdout + r.stderr).strip()[-200:])


def check_contract_argc() -> Result:
    """Pair-contract translated_run.args vs the governing tree's parser (OD-3).

    Allowlist policy (2026-08-21 Codex architecture review, item d): the
    checker itself absorbs only entries pinned in its
    KNOWN_CONTRACT_ARGC_MISMATCHES allowlist (exact pair AND detail) and
    exits 0 for them - surfaced here as WARN with the ruling pending. Any
    NEW mismatch, stale allowlist entry, missing referenced spec, or
    spec-index problem exits 1 -> FAIL; a checker crash (unloadable
    registry, absent spec root) exits 2 -> FAIL. A clean corpus is PASS, so
    no policy switch is needed once the allowlisted entries are fixed.
    """
    r = _run(
        _venv_python() + ["scripts/spec_tools/check_spec_argc.py", "--contracts"],
        timeout=120,
    )
    summary = ""
    known: list[str] = []
    for line in r.stdout.splitlines():
        if line.startswith("contracts_checked="):
            summary = line
        elif line.startswith("MISMATCH (known) "):
            known.append(line[len("MISMATCH (known) "):].split(":", 1)[0])
    if r.returncode == 0 and not known:
        return Result("pair-contract argc vs governing parser", "PASS", summary)
    if r.returncode == 0:
        return Result(
            "pair-contract argc vs governing parser", "WARN",
            f"{summary} | {len(known)} allowlisted mismatch(es), owner "
            f"ruling pending: {'; '.join(known)}",
        )
    return Result(
        "pair-contract argc vs governing parser", "FAIL",
        (summary + " | " if summary else "") + (r.stdout + r.stderr).strip()[-200:],
    )


# --- Self-test (positive + negative controls) ---------------------------


def self_test() -> bool:
    ok = True

    def check(label: str, cond: bool):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] self-test: {label}")
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        good_py = tdp / "good.py"
        good_py.write_text("x = 1\n")
        bad_py = tdp / "bad.py"
        bad_py.write_text("def f(:\n")
        check("python compile — positive control (valid)", check_python_import(good_py).status == "PASS")
        check("python compile — negative control (syntax error)", check_python_import(bad_py).status == "FAIL")

        # argv-only invocation control: a filename containing a quote must not
        # crash or inject — it's just another argv element, never interpolated
        # into a `-c` source string
        quote_py = tdp / "quo'te.py"
        quote_py.write_text("x = 1\n")
        check("python compile — quoted-filename control (argv-safe, no injection)", check_python_import(quote_py).status == "PASS")

        good_sh = tdp / "good.sh"
        good_sh.write_text("#!/bin/bash\necho hi\n")
        bad_sh = tdp / "bad.sh"
        bad_sh.write_text("if [ 1 -eq 1\n  echo unterminated\n")
        check("shell syntax — positive control", check_shell_syntax(good_sh).status == "PASS")
        check("shell syntax — negative control", check_shell_syntax(bad_sh).status == "FAIL")

        good_agent = tdp / "good_agent.md"
        good_agent.write_text('---\nname: x\ndescription: "y"\ntools: Bash\nmodel: sonnet\n---\nbody\n')
        bad_agent = tdp / "bad_agent.md"
        bad_agent.write_text('---\nname: x\n---\nbody\n')
        check("agent frontmatter — positive control", check_agent_frontmatter(good_agent).status == "PASS")
        check("agent frontmatter — negative control (missing fields)", check_agent_frontmatter(bad_agent).status == "FAIL")

        opus_agent = tdp / "opus_agent.md"
        opus_agent.write_text('---\nname: x\ndescription: "y"\ntools: Bash\nmodel: opus\n---\nbody\n')
        check("agent frontmatter — bare 'opus' alias control (must FAIL)", check_agent_frontmatter(opus_agent).status == "FAIL")
        pinned_agent = tdp / "pinned_agent.md"
        pinned_agent.write_text('---\nname: x\ndescription: "y"\ntools: Bash\nmodel: claude-opus-4-8[1m]\n---\nbody\n')
        check("agent frontmatter — pinned opus id control (must PASS)", check_agent_frontmatter(pinned_agent).status == "PASS")

    # manifest categories: enum lives on manifest entries (ruling 34 fix 4)
    good_manifest = '{"kernel_name": "x", "category": "graph"}\n'
    bad_manifest = good_manifest + '{"kernel_name": "y", "category": "bioinformatics"}\n'
    check("manifest categories — positive control (enum value)",
          _judge_manifest_categories(good_manifest).status == "PASS")
    check("manifest categories — negative control (forbidden alias must FAIL)",
          _judge_manifest_categories(bad_manifest).status == "FAIL")

    # lint scoping: violation in a changed file -> FAIL; violation only outside the
    # diff -> WARN, never a hard block on pre-existing repo debt
    res_fail = _classify_lint(["foo/changed.py:1:1: F401 `x` imported but unused"], {"foo/changed.py"})
    check("lint scoping — positive control (violation in changed file -> FAIL)", res_fail.status == "FAIL")
    res_warn = _classify_lint(["bar/untouched.py:1:1: F401 `x` imported but unused"], {"foo/changed.py"})
    check("lint scoping — negative control (violation outside diff -> WARN, not FAIL)", res_warn.status == "WARN")

    # crashed validator: nonzero exit + zero parseable error lines must not read as PASS
    res_crash = _judge_schema_baseline(1, "", "Traceback (most recent call last):\nModuleNotFoundError: No module named 'jsonschema'")
    check("schema baseline — crashed-validator control (nonzero exit, 0 errors -> FAIL)", res_crash.status == "FAIL")

    # crashed ruff: nonzero exit + diagnostics only on stderr must not read as PASS
    res_lint_crash = _judge_lint(2, "", "ruff failed\n  Cause: could not read config", set())
    check("lint — crashed-tool control (nonzero exit, 0 violations -> FAIL)", res_lint_crash.status == "FAIL")

    # deleted-file filter: a deleted path must not be dispatched to a per-file check
    sample_status = "M\tfoo/changed.py\nD\tfoo/deleted.py\nR100\told_name.py\tfoo/new_name.py\n"
    parsed = _parse_changed_files(sample_status)
    check("changed-files filter — deleted path excluded", "foo/deleted.py" not in parsed)
    check(
        "changed-files filter — modified + renamed paths kept",
        "foo/changed.py" in parsed and "foo/new_name.py" in parsed and "old_name.py" not in parsed,
    )

    # git-diff command failure: must raise, never silently become an empty
    # changed-file list (which would make check_lint/check_changed_files fail open)
    raised = False
    try:
        _decide_changed_files(128, "", "fatal: bad revision 'HEAD'")
    except ChangedFilesUnavailable:
        raised = True
    check("changed-files - command-failure control (nonzero git exit raises, never empty list)", raised)

    # check_unit_tests actually RUNS the suite — --collect-only alone would miss
    # a genuinely failing test. Point it at a fixture, not the real suite.
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td) / "test_fixture.py"
        fixture.write_text("def test_fails():\n    assert 1 == 2\n")
        res_run_fail = check_unit_tests(str(fixture))
        check("unit tests — negative control (a real failing test is caught, not just collected)", res_run_fail.status == "FAIL")

        fixture_ok = Path(td) / "test_fixture_ok.py"
        fixture_ok.write_text("\n".join(f"def test_ok_{i}():\n    assert 1 == 1" for i in range(UNIT_TEST_MIN)) + "\n")
        res_run_ok = check_unit_tests(str(fixture_ok))
        check(
            f"unit tests — positive control ({UNIT_TEST_MIN} real passing tests -> PASS)",
            res_run_ok.status == "PASS" and f"{UNIT_TEST_MIN} passed" in res_run_ok.detail,
        )

    # classify_schema_errors: phantom-5 always classify clean, regardless of tree state
    phantom_line = "  • Line 181: [manifest] source_dir: 'rodinia/rodinia-src/openmp/gaussian' does not exist on disk"
    exp, unexp = classify_schema_errors([phantom_line])
    check("schema classify — negative control (15 phantom-derived errors classify clean)", len(exp) == 1 and len(unexp) == 0)

    # positive control: inject one unexplainable error name — a real (non-phantom)
    # spec's source_dir miss under a tree the probe reports as PRESENT — must FAIL
    with tempfile.TemporaryDirectory() as td:
        fake_root = Path(td)
        (fake_root / "rodinia" / "rodinia-src").mkdir(parents=True)
        (fake_root / "rodinia" / "rodinia-src" / ".git").write_text("gitdir: ../../.git/modules/rodinia\n")
        unexplainable_line = "  • Line 42: [manifest] source_dir: 'rodinia/rodinia-src/openmp/bfs' does not exist on disk"
        exp3, unexp3 = classify_schema_errors([unexplainable_line], project_root=fake_root)
        check(
            "schema classify — positive control (unexplainable error under a present tree, must FAIL)",
            len(unexp3) == 1 and len(exp3) == 0,
        )

    # tree-presence probe itself: a source_dir miss under an ABSENT tree classifies
    # as expected; the identical line under a PRESENT tree does not
    with tempfile.TemporaryDirectory() as td:
        absent_root = Path(td)  # no HeCBench-master/src created -> tree absent
        hec_line = "  • Line 197: [manifest] source_dir: 'HeCBench-master/src/jacobi-cuda' does not exist on disk"
        exp4, unexp4 = classify_schema_errors([hec_line], project_root=absent_root)
        check("schema classify — tree-absence control (HeCBench absent -> expected)", len(exp4) == 1 and len(unexp4) == 0)

    with tempfile.TemporaryDirectory() as td:
        present_root = Path(td)
        (present_root / "HeCBench-master" / "src").mkdir(parents=True)  # tree present
        exp5, unexp5 = classify_schema_errors([hec_line], project_root=present_root)
        check("schema classify — tree-presence control (HeCBench present -> unexplained)", len(unexp5) == 1 and len(exp5) == 0)

    return ok


# --- Main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run positive/negative controls, no repo checks")
    args = parser.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    results: list[Result] = []
    results.append(check_lint())
    results.append(check_test_collection())
    results.append(check_import_smoke())
    results.extend(check_spec_counts())
    results.append(check_unit_tests())
    results.append(check_key_infra_files())
    results.extend(check_manifest())
    results.extend(check_changed_files())
    results.append(check_schema_baseline())
    results.append(check_spec_argc())
    results.append(check_contract_argc())

    for res in results:
        print(res.line())

    failed = [r for r in results if r.status == "FAIL"]
    print(f"\nHEALTH CHECK: {'FAIL' if failed else 'PASS'} ({len(results)} checks, {len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
