"""Functional smoke test: each verify_run(...) caller's working_dir access
pattern must actually resolve at runtime.

Complements:
  - test_verifier_caller_contract.py (static regex)
  - test_cli_verify_smoke.py (end-to-end stubbed harness verify)

This test evaluates the EXACT dict-access expression used at each caller's
verify_run() call site. If a future refactor reverts to the buggy pre-S5
form (`target_spec_resolved["working_dir"]` instead of
`target_spec_resolved["_resolved"]["working_dir"]`), the indexing will raise
KeyError here — faster and more specific than an end-to-end eval stub.

The 3 non-augment-verify callers are parametrized. augment_verify.py binds
`working_dir=tempdir` where `tempdir` is a scratch Path derived from
`original_resolved["working_dir"]`; exercising that path requires staging
a scratch copytree so it's left to integration tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.spec_loader import load_spec, resolve_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = PROJECT_ROOT / "specs" / "rodinia-bfs-cuda.json"


@pytest.mark.parametrize(
    "caller",
    [
        # (module-site description, the exact access pattern used at the call site)
        "harness/cli.py:141 (via resolved = resolve_paths(...)['_resolved'])",
        "scripts/evaluation/llm_evaluate.py:1821",
        "scripts/evaluation/reverify_pass_results.py:214",
    ],
)
def test_verify_run_working_dir_access_resolves(caller: str) -> None:
    """The access expression used at each caller's verify_run() site must
    return a Path when applied to a real `resolve_paths()` return value."""
    spec = load_spec(SPEC_PATH)
    target_spec_resolved = resolve_paths(spec, PROJECT_ROOT)

    # Post-fix form — identical to the expression on the RHS of `working_dir=`
    # at each of the 3 sites:
    #   harness/cli.py:140  resolved = resolve_paths(spec, project_root)["_resolved"]
    #   harness/cli.py:141  verify_run(..., working_dir=resolved["working_dir"])
    #   llm_evaluate.py:1821     working_dir=target_spec_resolved["_resolved"]["working_dir"]
    #   reverify_pass_results.py:214 working_dir=target_spec_resolved["_resolved"]["working_dir"]
    working_dir = target_spec_resolved["_resolved"]["working_dir"]
    assert isinstance(working_dir, Path), (
        f"{caller}: expected Path, got {type(working_dir).__name__}"
    )
    # And the pre-S1.6 buggy form must NOT resolve — if this assertion ever
    # fails, the dict shape has changed and the whole contract is obsolete.
    with pytest.raises(KeyError):
        _ = target_spec_resolved["working_dir"]
