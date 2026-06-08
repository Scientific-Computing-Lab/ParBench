"""Unit tests for scripts.spec_tools._capture_baseline.

Private helper is single-shot and side-effecting (spawns builds and runs),
so these tests exercise:
  - Pure helpers (_snapshot_dir, _diff_snapshots, _sha256_file) with tmp_path.
  - capture_baseline() end-to-end via monkeypatched build_spec + run_spec so
    the tests never touch a real compiler.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.models import BuildResult, RunResult, Status  # noqa: E402
from scripts.spec_tools import _capture_baseline as cap  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_snapshot_diff_finds_new_file(tmp_path: Path):
    before = cap._snapshot_dir(tmp_path)
    (tmp_path / "result.txt").write_bytes(b"hi\n")
    after = cap._snapshot_dir(tmp_path)
    cands = cap._diff_snapshots(before, after, exclude_basenames=set())
    assert cands == ["result.txt"]


def test_snapshot_diff_excludes_prompt_payload_basename(tmp_path: Path):
    (tmp_path / "src.cpp").write_bytes(b"// input\n")
    before = cap._snapshot_dir(tmp_path)
    time.sleep(0.01)  # ensure mtime changes on modification
    (tmp_path / "src.cpp").write_bytes(b"// modified\n")
    (tmp_path / "output.bin").write_bytes(b"genuine output\n")
    after = cap._snapshot_dir(tmp_path)
    cands = cap._diff_snapshots(before, after, exclude_basenames={"src.cpp"})
    assert cands == ["output.bin"]


def test_sha256_file_matches_hashlib(tmp_path: Path):
    data = b"deterministic bytes\n" * 100
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert cap._sha256_file(p) == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# End-to-end tests (mocked build_spec + run_spec)
# ---------------------------------------------------------------------------


def _write_fake_spec(project_root: Path, working_dir: Path) -> Path:
    """Stage a project_root with config/paths.json + a spec whose resolve_paths
    maps working_dir to *working_dir*. Returns the spec path."""
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "config" / "paths.json").write_text(json.dumps({
        "project_root": str(project_root),
        "downloads_root": str(working_dir.parent),
        "hecbench_root": str(project_root),
    }))
    spec_dir = project_root / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / "fake-fake-omp.json"
    spec_path.write_text(json.dumps({
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": "fake",
            "parallel_api": "omp",
            "unique_id": "fake-fake-omp",
            "source_suite": "fake",
        },
        "provenance": {
            "repo_root": working_dir.name,
            "source_path": "",
            "repository": {},
        },
        "files": {
            "prompt_payload": ["src.cpp"],
            "support_files": [],
            "verification_only": [],
            "translation_targets": ["src.cpp"],
        },
        "build": {
            "working_directory": "",
            "commands": {"build": "true"},
            "outputs": {"executable": ""},
        },
        "run": {
            "executable": "./fake",
            "input_configurations": {
                "correctness": {
                    "arguments": [],
                    "description": "",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
    }))
    return spec_path


def test_missing_spec_file_errors(tmp_path: Path, capsys):
    rc = cap.capture_baseline(tmp_path / "does_not_exist.json", tmp_path)
    assert rc != 0
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "spec file not found" in combined


def test_build_fail_aborts(tmp_path: Path, monkeypatch, capsys):
    work = tmp_path / "kernel"
    work.mkdir()
    spec_path = _write_fake_spec(tmp_path, work)

    def fake_build(*args, **kwargs):
        return BuildResult(
            status=Status.FAIL,
            duration_seconds=0.1,
            stdout="",
            stderr="simulated compile error\n",
        )

    monkeypatch.setattr(cap, "build_spec", fake_build)
    rc = cap.capture_baseline(spec_path, tmp_path)
    assert rc != 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "BUILD_FAIL" in combined or "build failed" in combined.lower()


def _fake_build_pass(*args, **kwargs):
    return BuildResult(
        status=Status.PASS,
        duration_seconds=0.1,
        stdout="",
        stderr="",
        executable_path=None,
    )


def test_nondeterministic_output_triggers_loud_warning(
    tmp_path: Path, monkeypatch, capsys
):
    work = tmp_path / "kernel"
    work.mkdir()
    spec_path = _write_fake_spec(tmp_path, work)

    counter = {"n": 0}

    def fake_run(spec, project_root, configuration="correctness", **kwargs):
        counter["n"] += 1
        # Write different bytes each call → non-determinism.
        (work / "result.txt").write_bytes(
            f"run-{counter['n']}-{time.time_ns()}".encode()
        )
        return RunResult(
            status=Status.PASS,
            configuration=configuration,
            duration_seconds=0.1,
            exit_code=0,
            stdout=f"run {counter['n']} ok\n",
            stderr="",
        )

    monkeypatch.setattr(cap, "build_spec", _fake_build_pass)
    monkeypatch.setattr(cap, "run_spec", fake_run)

    rc = cap.capture_baseline(spec_path, tmp_path)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert rc == 0  # helper itself did not error; it just warned
    assert "NON-DETERMINISTIC" in combined
    assert "result.txt" in combined
    assert "oracle_strength" in combined.lower() and "weak" in combined.lower()
    assert counter["n"] == 2  # must run exactly twice


def test_mixed_determinism_suppresses_file_hash_snippet(
    tmp_path: Path, monkeypatch, capsys
):
    """M2 contract: when ANY candidate is non-deterministic, file_hash snippet
    must be withheld entirely — even for the deterministic subset."""
    work = tmp_path / "kernel"
    work.mkdir()
    spec_path = _write_fake_spec(tmp_path, work)

    stable_content = b"stable deterministic bytes\n"
    stable_sha = hashlib.sha256(stable_content).hexdigest()
    counter = {"n": 0}

    def fake_run(spec, project_root, configuration="correctness", **kwargs):
        counter["n"] += 1
        # stable.bin: same bytes every run (deterministic)
        (work / "stable.bin").write_bytes(stable_content)
        # volatile.bin: different bytes each run (non-deterministic)
        (work / "volatile.bin").write_bytes(
            f"volatile-{counter['n']}-{time.time_ns()}".encode()
        )
        return RunResult(
            status=Status.PASS,
            configuration=configuration,
            duration_seconds=0.05,
            exit_code=0,
            stdout="done\n",
            stderr="",
        )

    monkeypatch.setattr(cap, "build_spec", _fake_build_pass)
    monkeypatch.setattr(cap, "run_spec", fake_run)

    rc = cap.capture_baseline(spec_path, tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    # (a) non-det banner fires
    assert "NON-DETERMINISTIC" in combined
    # (b) volatile.bin named in the warning
    assert "volatile.bin" in combined
    # (c) explicit MIXED DETERMINISM message
    assert "MIXED DETERMINISM" in combined
    # (d) file_hash JSON snippet WITHHELD — the JSON type field must not appear in stdout
    assert '"type": "file_hash"' not in captured.out
    # (e) stable.bin SHA NOT in the snippet block (after the snippet header)
    snippet_section = captured.out.split("--- suggested JSON snippets")[-1]
    assert stable_sha not in snippet_section


def test_deterministic_output_emits_file_hash_snippet(
    tmp_path: Path, monkeypatch, capsys
):
    work = tmp_path / "kernel"
    work.mkdir()
    spec_path = _write_fake_spec(tmp_path, work)

    content = b"bit-for-bit deterministic output\n"
    expected_sha = hashlib.sha256(content).hexdigest()

    def fake_run(spec, project_root, configuration="correctness", **kwargs):
        (work / "result.txt").write_bytes(content)
        return RunResult(
            status=Status.PASS,
            configuration=configuration,
            duration_seconds=0.05,
            exit_code=0,
            stdout="Result stored in result.txt\n",
            stderr="",
        )

    monkeypatch.setattr(cap, "build_spec", _fake_build_pass)
    monkeypatch.setattr(cap, "run_spec", fake_run)

    rc = cap.capture_baseline(spec_path, tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out

    # Deterministic path: no non-det warning.
    assert "NON-DETERMINISTIC" not in (captured.out + captured.err)
    # Paste-ready snippet produced with correct hash + path.
    assert "file_hash" in out
    assert expected_sha in out
    assert "result.txt" in out
    assert "reference_files" in out
    assert "specs/references/fake/result.txt" in out
