"""Contract tests: all production callers of verify_run() must pass working_dir.

S1 introduced a `working_dir` kwarg on `verify_run()` so file_hash strategies can
SHA-256 output files relative to the run's working directory. Any production
caller that omits it will ERROR on every file_hash spec. These tests statically
check each caller's source file to ensure the kwarg is passed, so a future
regression (e.g. a refactor dropping the kwarg) is caught at test time.

The value check (`test_caller_does_not_access_raw_resolve_paths_working_dir`)
guards against the bug class where a caller passes
`working_dir=resolved["working_dir"]` when `resolve_paths()` actually returns
`{..., "_resolved": {"working_dir": ...}}`. That access pattern runs the
`working_dir=` substring check but raises KeyError at runtime. S5 discovered
two callers in this state even after S1.6 was marked complete.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Regex catches `verify_run(...)` calls and checks for `working_dir=` within
# the argument list (single- or multi-line).
_VERIFY_RUN_CALL = re.compile(r"verify_run\s*\((?P<args>.*?)\)", re.DOTALL)

# Whitelist of accepted `working_dir=<RHS>` patterns. Each alt covers one
# legitimate binding pattern observed across the 4 production callers:
#   1. `<ident>["_resolved"]["working_dir"]`    — direct, most common
#   2. `resolved["working_dir"]` where `resolved = resolve_paths(...)["_resolved"]`
#      was pre-unwrapped on the prior line (harness/cli.py:140-141)
#   3. `tempdir` — Path bound in augment_verify.py from `original_resolved["working_dir"]`
#      then re-rooted into a scratch copy; semantically correct, no spec dict access.
# Any RHS that does not match one of these three is rejected.
_VALID_WORKING_DIR = re.compile(
    r"""working_dir\s*=\s*"""
    r"""(?:"""
    r"""\w+\[\s*["']_resolved["']\s*\]\[\s*["']working_dir["']\s*\]"""
    r"""|resolved\[\s*["']working_dir["']\s*\]"""
    r"""|tempdir"""
    r""")"""
)

_CALLERS = [
    "harness/cli.py",
    "scripts/evaluation/llm_evaluate.py",
    "scripts/evaluation/reverify_pass_results.py",
    "scripts/augmentation/augment_verify.py",
]


def _caller_passes_working_dir(caller_path: Path) -> bool:
    """Return True iff every verify_run(...) call in `caller_path` passes a
    `working_dir=` kwarg."""
    source = caller_path.read_text()
    calls = list(_VERIFY_RUN_CALL.finditer(source))
    assert calls, f"no verify_run() call found in {caller_path}"
    return all("working_dir=" in m.group("args") for m in calls)


@pytest.mark.parametrize("rel_path", _CALLERS)
def test_caller_passes_working_dir(rel_path: str) -> None:
    caller = PROJECT_ROOT / rel_path
    assert _caller_passes_working_dir(caller), (
        f"{caller} calls verify_run() without working_dir=; "
        f"file_hash specs will ERROR under this caller"
    )


@pytest.mark.parametrize("rel_path", _CALLERS)
def test_caller_uses_resolved_working_dir(rel_path: str) -> None:
    """Each `working_dir=<RHS>` kwarg must match the accepted access-pattern
    whitelist. `resolve_paths(spec, project_root)` returns the spec dict with a
    `"_resolved"` sub-dict; `spec_or_resolved["working_dir"]` on the raw return
    raises KeyError at runtime. The substring check above does not distinguish
    buggy `<raw>["working_dir"]` from safe `<raw>["_resolved"]["working_dir"]`
    — this whitelist test does.
    """
    caller = PROJECT_ROOT / rel_path
    source = caller.read_text()
    for call in _VERIFY_RUN_CALL.finditer(source):
        args = call.group("args")
        if "working_dir=" not in args:
            continue
        assert _VALID_WORKING_DIR.search(args), (
            f"{caller}: verify_run(...) call uses an unrecognized `working_dir=` "
            f"access pattern. Accepted: `<ident>[\"_resolved\"][\"working_dir\"]`, "
            f"`resolved[\"working_dir\"]` (after pre-unwrap), or `tempdir`. "
            f"Call: {args.strip()!r}. This is the KeyError bug-class fixed in "
            f"S5 Batch 1 (harness/cli.py:140) and S5 review (llm_evaluate.py:1821 + "
            f"reverify_pass_results.py:214)."
        )
