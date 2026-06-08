"""Phase 2 / Plan 02-06 tests: --task-list flag (D-26).

Uses synthetic passer JSON + a stubbed manifest. Validates:
- runtime mutex enforces --task-list ⊥ {--suite, --kernels} (D-23, post 02-10 lift)
  while --suite and --kernels now compose (intersection filter).
- task matrix expansion: passers × augment_levels × num_samples (D-25).
- skip-on-missing-manifest behavior (D-26 case 2).
- --direction is still required even with --task-list (D-24).

Per F-12: SystemExit tests assert `exc_info.value.code == 2` only — never the
exact argparse error text (which drifts across Python versions). The 02-10
lift replaced argparse's `add_mutually_exclusive_group()` with a runtime
`parser.error()` call; both shapes raise SystemExit(2) so tests stay stable.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.evaluation import run_eval_batch


MANIFEST_ENTRIES = [
    {"source_suite": "rodinia", "kernel_name": "bfs", "parallel_api": "cuda",
     "spec_file": "specs/rodinia-bfs-cuda.json"},
    {"source_suite": "rodinia", "kernel_name": "bfs", "parallel_api": "omp",
     "spec_file": "specs/rodinia-bfs-omp.json"},
    {"source_suite": "xsbench", "kernel_name": "xsbench", "parallel_api": "cuda",
     "spec_file": "specs/xsbench-xsbench-cuda.json"},
    {"source_suite": "xsbench", "kernel_name": "xsbench", "parallel_api": "omp",
     "spec_file": "specs/xsbench-xsbench-omp.json"},
]


@pytest.fixture
def passer_json(tmp_path):
    passers = [
        {"source_spec": "rodinia-bfs-cuda", "target_spec": "rodinia-bfs-omp", "augment_level": 0},
        {"source_spec": "xsbench-xsbench-cuda", "target_spec": "xsbench-xsbench-omp", "augment_level": 0},
    ]
    p = tmp_path / "passers.json"
    p.write_text(json.dumps(passers))
    return p


@pytest.fixture
def fake_project(tmp_path):
    """Create a fake project root with the manifest spec files present on disk."""
    (tmp_path / "specs").mkdir()
    for entry in MANIFEST_ENTRIES:
        (tmp_path / entry["spec_file"]).write_text("{}")
    return tmp_path


# --------------------------------------------------------------------------- #
# Runtime mutex (D-23, post 02-10 lift) + compose + required-direction (D-24) #
# --------------------------------------------------------------------------- #

def test_parser_advertises_task_list_flag():
    """--help output must include --task-list."""
    parser = run_eval_batch._build_parser()
    help_text = parser.format_help()
    assert "--task-list" in help_text


def test_task_list_mutex_with_kernels(passer_json):
    """D-26 case 3: --task-list + --kernels → SystemExit(2) via runtime check.
    Per F-12: assert exit code only, NOT error text.

    Post 02-10 lift: argparse itself no longer rejects this combo (the 3-way
    mutex group was removed). The runtime `parser.error()` in `main()` does,
    with the same SystemExit(2) shape. This test exercises main() to hit it.
    """
    argv = [
        "--task-list", str(passer_json),
        "--kernels", "bfs",
        "--direction", "cuda-to-omp",
        "--models", "gpt-4o",
    ]
    with patch.object(run_eval_batch.sys, "argv", ["run_eval_batch.py", *argv]):
        with pytest.raises(SystemExit) as exc_info:
            run_eval_batch.main()
    assert exc_info.value.code == 2  # parser.error() → sys.exit(2)


def test_task_list_mutex_with_suite(passer_json):
    """D-26 case 3 (extended): --task-list + --suite → SystemExit(2) via runtime check."""
    argv = [
        "--task-list", str(passer_json),
        "--suite", "rodinia",
        "--direction", "cuda-to-omp",
        "--models", "gpt-4o",
    ]
    with patch.object(run_eval_batch.sys, "argv", ["run_eval_batch.py", *argv]):
        with pytest.raises(SystemExit) as exc_info:
            run_eval_batch.main()
    assert exc_info.value.code == 2


def test_suite_and_kernels_compose():
    """02-10 regression guard: --suite and --kernels MUST parse together.

    The 02-06 D-23 LOCK (3-way argparse-native mutex) blocked this combo,
    causing 12/12 Qwen real-API tests to fail at argparse on 2026-04-17
    with `argument --kernels: not allowed with argument --suite`. The lift
    restores composition semantics (narrow-to-suite then filter-by-kernel).
    """
    parser = run_eval_batch._build_parser()
    args = parser.parse_args([
        "--suite", "rodinia",
        "--kernels", "bfs", "hotspot",
        "--direction", "cuda-to-omp",
        "--models", "gpt-4o",
    ])
    assert args.suite == "rodinia"
    assert args.kernels == ["bfs", "hotspot"]
    assert args.task_list is None


def test_task_list_without_direction_fails(passer_json):
    """D-26 case 4: --task-list without --direction → SystemExit (existing required=True)."""
    parser = run_eval_batch._build_parser()
    argv = [
        "--task-list", str(passer_json),
        "--models", "gpt-4o",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# _build_tasks_from_task_list — matrix expansion + manifest filtering          #
# --------------------------------------------------------------------------- #

def test_task_list_expands_by_augment_levels(passer_json, fake_project):
    """D-26 case 1: 2 passers × 4 levels × 1 sample = 8 tasks."""
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        tasks = run_eval_batch._build_tasks_from_task_list(
            task_list_path=passer_json,
            project_root=fake_project,
            direction="cuda-to-omp",
            models=["gpt-4o"],
            augment_levels=[1, 2, 3, 4],
            num_samples=1,
            manifest_path=fake_project / "manifest.jsonl",
        )
    assert len(tasks) == 8
    assert {t["augment_level"] for t in tasks} == {1, 2, 3, 4}
    # Task dict shape matches existing _build_tasks contract.
    expected_keys = {"kernel", "src_spec", "tgt_spec", "model", "augment_level",
                     "sample_id", "num_samples", "src_id", "tgt_id"}
    assert all(expected_keys <= set(t.keys()) for t in tasks)


def test_task_list_expands_by_num_samples(passer_json, fake_project):
    """2 passers × 4 levels × 3 samples = 24 tasks."""
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        tasks = run_eval_batch._build_tasks_from_task_list(
            task_list_path=passer_json,
            project_root=fake_project,
            direction="cuda-to-omp",
            models=["gpt-4o"],
            augment_levels=[1, 2, 3, 4],
            num_samples=3,
            manifest_path=fake_project / "manifest.jsonl",
        )
    assert len(tasks) == 24
    assert {t["sample_id"] for t in tasks} == {0, 1, 2}


def test_task_list_expands_by_models(passer_json, fake_project):
    """2 passers × 1 level × 1 sample × 2 models = 4 tasks."""
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        tasks = run_eval_batch._build_tasks_from_task_list(
            task_list_path=passer_json,
            project_root=fake_project,
            direction="cuda-to-omp",
            models=["gpt-4o", "azure-gpt-5.4"],
            augment_levels=[0],
            num_samples=1,
            manifest_path=fake_project / "manifest.jsonl",
        )
    assert len(tasks) == 4
    assert {t["model"] for t in tasks} == {"gpt-4o", "azure-gpt-5.4"}


def test_task_list_skips_unknown_specs(tmp_path, fake_project, capsys):
    """D-26 case 2: passer entry with spec not in manifest → skip + stderr warning."""
    passers = [
        {"source_spec": "rodinia-bfs-cuda", "target_spec": "rodinia-bfs-omp", "augment_level": 0},
        {"source_spec": "nonexistent-kernel-cuda", "target_spec": "nonexistent-kernel-omp", "augment_level": 0},
    ]
    p = tmp_path / "passers.json"
    p.write_text(json.dumps(passers))

    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        tasks = run_eval_batch._build_tasks_from_task_list(
            task_list_path=p,
            project_root=fake_project,
            direction="cuda-to-omp",
            models=["gpt-4o"],
            augment_levels=[1],
            num_samples=1,
            manifest_path=fake_project / "manifest.jsonl",
        )
    # Only the known-manifest passer yields a task.
    assert len(tasks) == 1
    err = capsys.readouterr().err
    assert "nonexistent-kernel" in err
    assert "not in manifest" in err


def test_task_list_skips_direction_mismatch(tmp_path, fake_project, capsys):
    """Passer entry whose source/target APIs don't match --direction → skip + warning."""
    passers = [
        # Direction declared cuda-to-omp but entry is omp→cuda.
        {"source_spec": "rodinia-bfs-omp", "target_spec": "rodinia-bfs-cuda", "augment_level": 0},
    ]
    p = tmp_path / "passers.json"
    p.write_text(json.dumps(passers))

    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        tasks = run_eval_batch._build_tasks_from_task_list(
            task_list_path=p,
            project_root=fake_project,
            direction="cuda-to-omp",
            models=["gpt-4o"],
            augment_levels=[1],
            num_samples=1,
            manifest_path=fake_project / "manifest.jsonl",
        )
    assert tasks == []
    err = capsys.readouterr().err
    assert "do not match" in err.lower() or "not match" in err.lower()


def test_task_list_malformed_json_raises(tmp_path, fake_project):
    """Malformed JSON input → ValueError (clear error, not silent crash)."""
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        with pytest.raises(ValueError, match="not valid JSON"):
            run_eval_batch._build_tasks_from_task_list(
                task_list_path=p,
                project_root=fake_project,
                direction="cuda-to-omp",
                models=["gpt-4o"],
                augment_levels=[1],
                num_samples=1,
                manifest_path=fake_project / "manifest.jsonl",
            )


def test_task_list_non_list_json_raises(tmp_path, fake_project):
    """JSON that isn't a list → ValueError."""
    p = tmp_path / "dict.json"
    p.write_text(json.dumps({"not": "a list"}))
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        with pytest.raises(ValueError, match="must contain a JSON list"):
            run_eval_batch._build_tasks_from_task_list(
                task_list_path=p,
                project_root=fake_project,
                direction="cuda-to-omp",
                models=["gpt-4o"],
                augment_levels=[1],
                num_samples=1,
                manifest_path=fake_project / "manifest.jsonl",
            )


def test_task_list_empty_list_returns_empty(tmp_path, fake_project):
    """Empty passer list → empty task list (not an error)."""
    p = tmp_path / "empty.json"
    p.write_text("[]")
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        tasks = run_eval_batch._build_tasks_from_task_list(
            task_list_path=p,
            project_root=fake_project,
            direction="cuda-to-omp",
            models=["gpt-4o"],
            augment_levels=[1],
            num_samples=1,
            manifest_path=fake_project / "manifest.jsonl",
        )
    assert tasks == []


def test_task_list_missing_file_raises(tmp_path, fake_project):
    """Missing input file → FileNotFoundError with the path in the message."""
    missing = tmp_path / "does_not_exist.json"
    with patch.object(run_eval_batch, "load_manifest", return_value=MANIFEST_ENTRIES):
        with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
            run_eval_batch._build_tasks_from_task_list(
                task_list_path=missing,
                project_root=fake_project,
                direction="cuda-to-omp",
                models=["gpt-4o"],
                augment_levels=[1],
                num_samples=1,
                manifest_path=fake_project / "manifest.jsonl",
            )
