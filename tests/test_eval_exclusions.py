"""Task 1 (gate E-17): the canonical ten-spec exclusion set is automatic.

Every evaluation entry path — canonical/suite enumeration, --task-list,
resume, and retry — must apply ``resolve_excluded_specs()``, which is
``harness.constants.EXCLUDED_SPECS`` plus any caller-supplied extras.
The resolved set is recorded in every campaign manifest (batch summary),
and the copied nine-spec shell list in run_phase3.sh is gone.

Written against the POST-change contract; must fail on pre-change code
(which only excluded what the caller passed on the command line).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.constants import EXCLUDED_SPECS
from scripts.evaluation import run_eval_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_PHASE3 = PROJECT_ROOT / "scripts" / "batch" / "run_phase3.sh"

# The acceptance ledger's ten canonical exclusions, written out so this test
# fails loudly if the constant ever drifts from the documented set.
CANONICAL_TEN = frozenset({
    "rodinia-kmeans-cuda",
    "rodinia-mummergpu-cuda",
    "rodinia-mummergpu-omp",
    "rodinia-hybridsort-cuda",
    "rodinia-nn-opencl",
    "rodinia-kmeans-opencl",
    "rodinia-backprop-opencl",
    "hecbench-stencil1d-omp_target",
    "hecbench-scan-omp_target",
    "hecbench-lud-omp",
})


def _split_uid(uid: str) -> tuple[str, str, str]:
    """{suite}-{kernel}-{api}; kernel may itself contain dashes."""
    parts = uid.split("-")
    return parts[0], "-".join(parts[1:-1]), parts[-1]


def _write_project(tmp_path: Path) -> Path:
    """Fake project: for every canonical exclusion, a same-kernel partner spec
    with api 'otherapi', plus one non-excluded control pair."""
    (tmp_path / "specs").mkdir()
    entries = []

    def add(uid: str) -> None:
        suite, kernel, api = _split_uid(uid)
        entries.append({
            "source_suite": suite,
            "kernel_name": kernel,
            "parallel_api": api,
            "spec_file": f"specs/{uid}.json",
        })
        # Minimal identity/run blocks so the batch-level pair-contract gate
        # (Task 2) can cross-validate contracts against these fake specs.
        (tmp_path / "specs" / f"{uid}.json").write_text(json.dumps({
            "identity": {
                "kernel_name": kernel,
                "parallel_api": api,
                "unique_id": uid,
                "source_suite": suite,
            },
            "run": {"input_configurations": {"correctness": {"arguments": []}}},
        }))

    for uid in CANONICAL_TEN:
        suite, kernel, api = _split_uid(uid)
        add(uid)
        add(f"{suite}-{kernel}-otherapi")
    add("rodinia-control-cuda")
    add("rodinia-control-otherapi")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return manifest


# --------------------------------------------------------------------------- #
# resolve_excluded_specs()                                                     #
# --------------------------------------------------------------------------- #

def test_constants_match_the_documented_ten():
    assert EXCLUDED_SPECS == CANONICAL_TEN


def test_resolver_defaults_to_canonical_set():
    resolved = run_eval_batch.resolve_excluded_specs(None)
    assert isinstance(resolved, frozenset)
    assert resolved == EXCLUDED_SPECS


def test_resolver_with_empty_cli_list_is_canonical():
    assert run_eval_batch.resolve_excluded_specs([]) == EXCLUDED_SPECS


def test_resolver_adds_caller_extras_without_dropping_canonical():
    resolved = run_eval_batch.resolve_excluded_specs(["extra-spec-omp"])
    assert "extra-spec-omp" in resolved
    assert EXCLUDED_SPECS < resolved
    assert "hecbench-lud-omp" in resolved


# --------------------------------------------------------------------------- #
# Suite / canonical enumeration entry path                                     #
# --------------------------------------------------------------------------- #

def test_suite_enumeration_drops_all_ten_as_source_and_target(tmp_path):
    manifest = _write_project(tmp_path)
    resolved = run_eval_batch.resolve_excluded_specs(None)

    for uid in CANONICAL_TEN:
        suite, kernel, api = _split_uid(uid)
        for direction in (f"{api}-to-otherapi", f"otherapi-to-{api}"):
            tasks = run_eval_batch._build_tasks(
                manifest_path=manifest,
                project_root=tmp_path,
                direction=direction,
                suite=suite,
                kernels=[kernel],
                models=["m"],
                excluded_specs=resolved,
            )
            assert tasks == [], f"{uid} leaked through direction {direction}"

    control = run_eval_batch._build_tasks(
        manifest_path=manifest,
        project_root=tmp_path,
        direction="cuda-to-otherapi",
        suite="rodinia",
        kernels=["control"],
        models=["m"],
        excluded_specs=resolved,
    )
    assert len(control) == 1, "non-excluded control pair must survive"


# --------------------------------------------------------------------------- #
# --task-list entry path                                                       #
# --------------------------------------------------------------------------- #

def test_task_list_drops_all_ten_as_source_and_target(tmp_path):
    manifest = _write_project(tmp_path)
    resolved = run_eval_batch.resolve_excluded_specs(None)

    for uid in CANONICAL_TEN:
        suite, kernel, api = _split_uid(uid)
        partner = f"{suite}-{kernel}-otherapi"
        cases = [
            (f"{api}-to-otherapi", uid, partner),
            (f"otherapi-to-{api}", partner, uid),
        ]
        for direction, src, tgt in cases:
            passers = tmp_path / "passers.json"
            passers.write_text(json.dumps(
                [{"source_spec": src, "target_spec": tgt, "augment_level": 0}]
            ))
            tasks = run_eval_batch._build_tasks_from_task_list(
                task_list_path=passers,
                project_root=tmp_path,
                direction=direction,
                models=["m"],
                augment_levels=[0],
                num_samples=1,
                manifest_path=manifest,
                excluded_specs=resolved,
            )
            assert tasks == [], f"{uid} leaked through task-list direction {direction}"


# --------------------------------------------------------------------------- #
# main(): automatic exclusions, manifest recording, resume path                #
# --------------------------------------------------------------------------- #

def _run_main(tmp_path: Path, argv: list[str]):
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "overall_status": "PASS",
            "source_spec": kwargs["source_path"].stem,
            "target_spec": kwargs["target_path"].stem,
            "model": kwargs["model"],
        }

    # Pair-contract gate (Task 2, E-02): main() refuses a cross-API batch
    # without a registry covering every non-excluded pair, so the fake
    # project ships a resolved contract for its one surviving control pair.
    contracts = tmp_path / "pair_contracts.json"
    contracts.write_text(json.dumps({
        "contract_version": "parbench-final-v1",
        "pairs": [{
            "source_spec": "rodinia-control-cuda",
            "target_spec": "rodinia-control-otherapi",
            "shared_input": {"source_args": [], "target_args": []},
            "translated_run": {"args": [], "environment": {}},
            "target_oracle": {"strategies": [{"type": "exit_code", "expected": 0}]},
            "negative_checks": [],
            "comparator": {"type": "declared-strategy", "evidence": "tests/test_eval_exclusions.py"},
            "baseline_witness": {"status": "pass", "evidence": "tests/test_eval_exclusions.py"},
            "comparability": "eligible",
        }],
    }))

    out_prefix = tmp_path / "summary" / "batch"
    full_argv = argv + [
        "--project-root", str(tmp_path),
        "--out", str(out_prefix),
        "--pair-contracts", str(contracts),
        "--resume",
    ]
    with patch.object(run_eval_batch, "evaluate_translation", side_effect=fake_evaluate):
        with patch.object(run_eval_batch.sys, "argv", ["run_eval_batch.py"] + full_argv):
            run_eval_batch.main()
    summary = json.loads((tmp_path / "summary" / "batch.json").read_text())
    return calls, summary


def test_main_excludes_automatically_and_records_the_set(tmp_path):
    """No --excluded-specs on the command line: hecbench-lud-omp (and the
    other nine) must still be excluded, and the campaign manifest must
    record the resolved set."""
    _write_project(tmp_path)
    argv = ["--suite", "hecbench", "--direction", "omp-to-otherapi", "--models", "m"]

    with pytest.raises(SystemExit):
        # hecbench suite contains only excluded kernels in this fixture, so
        # zero tasks remain and main() exits 1 — the exclusion worked.
        _run_main(tmp_path, argv)


def test_main_records_resolved_set_in_campaign_manifest(tmp_path):
    _write_project(tmp_path)
    argv = ["--suite", "rodinia", "--direction", "cuda-to-otherapi",
            "--kernels", "control", "--models", "m"]
    calls, summary = _run_main(tmp_path, argv)

    assert len(calls) == 1
    recorded = set(summary["excluded_specs"])
    assert CANONICAL_TEN <= recorded
    assert "hecbench-lud-omp" in recorded


def test_resume_path_keeps_exclusions_and_manifest_record(tmp_path):
    """Second (resume) invocation over existing results: still zero excluded
    tasks, manifest still records the full resolved set.

    No --kernels filter: the whole rodinia cuda slice flows through, so the
    excluded kmeans/mummergpu/hybridsort cuda specs are genuine candidates
    on BOTH passes, not pre-filtered out by the test fixture.
    """
    _write_project(tmp_path)
    argv = ["--suite", "rodinia", "--direction", "cuda-to-otherapi",
            "--models", "m"]

    first_calls, first_summary = _run_main(tmp_path, argv)
    first_ran = {(c["source_path"].stem, c["target_path"].stem) for c in first_calls}
    assert first_ran == {("rodinia-control-cuda", "rodinia-control-otherapi")}, (
        "only the control pair may run; excluded cuda specs were candidates"
    )
    assert CANONICAL_TEN <= set(first_summary["excluded_specs"])

    resume_calls, summary = _run_main(tmp_path, argv)
    assert resume_calls == []  # existing PASS result skipped, nothing re-run
    assert CANONICAL_TEN <= set(summary["excluded_specs"])
    evaluated = {(r["source_spec"], r["target_spec"]) for r in summary["results"]}
    assert evaluated == {("rodinia-control-cuda", "rodinia-control-otherapi")}
    assert not any(
        s in CANONICAL_TEN or t in CANONICAL_TEN for s, t in evaluated
    )


def test_cli_extras_compose_with_canonical_set(tmp_path):
    """--excluded-specs adds to, never replaces, the canonical set — the
    retry path in run_phase3.sh used to rely on a copied nine-spec list."""
    _write_project(tmp_path)
    argv = ["--suite", "rodinia", "--direction", "cuda-to-otherapi",
            "--kernels", "control", "--models", "m",
            "--excluded-specs", "rodinia-control-cuda"]
    with pytest.raises(SystemExit):
        # The only surviving pair is now excluded by the CLI extra → 0 tasks.
        _run_main(tmp_path, argv)


# --------------------------------------------------------------------------- #
# run_phase3.sh: the copied shell list is gone                                 #
# --------------------------------------------------------------------------- #

def test_run_phase3_has_no_copied_exclusion_list():
    text = RUN_PHASE3.read_text()
    assert "KNOWN_FAIL_SPECS" not in text
    assert "--excluded-specs" not in text
    for uid in CANONICAL_TEN:
        assert uid not in text, f"hand-copied spec id {uid} still in run_phase3.sh"


def _extract_array_blocks(text: str, name: str) -> list[str]:
    """Return every shell 'NAME=( ... )' array-literal block from run_phase3.sh."""
    blocks = []
    pos = 0
    marker = f"{name}=("
    while (start := text.find(marker, pos)) != -1:
        end = text.index(")", start)
        blocks.append(text[start:end + 1])
        pos = end
    return blocks


def test_run_phase3_batch_and_retry_invocations_rely_on_automatic_exclusions():
    """Every invocation site — the first-pass CMD block and both retry-pass
    RETRY constructions — must carry --resume and must NOT hand-feed an
    exclusion list: the canonical set applies inside run_eval_batch.py."""
    text = RUN_PHASE3.read_text()
    runner_blocks = [
        b
        for name in ("CMD", "RETRY")
        for b in _extract_array_blocks(text, name)
        if "run_eval_batch.py" in b
    ]
    assert len(runner_blocks) == 3, "expected 1 CMD + 2 RETRY runner constructions"
    for block in runner_blocks:
        assert "--resume" in block, "runner invocation lost --resume"
        assert "--excluded-specs" not in block
        assert "KNOWN_FAIL" not in block
