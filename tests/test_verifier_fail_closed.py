"""Task 1 (gate E-18): declared verification checks fail closed.

Contract: a declared check that cannot execute — unknown strategy type,
malformed declared strategy (no type), empty required pattern, or an
unavailable declared check (custom_script) — must produce ERROR. It must
never become SKIP and disappear from the overall verdict.

file_diff is now implemented (byte-exact comparison against a reference
file), with missing-file and path-safety failures.

These tests are written against the POST-change contract and must fail on
the pre-change verifier (which turned all of the above into SKIP).
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.models import RunResult, Status
from harness.verifier import verify_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_SCHEMA = PROJECT_ROOT / "schema" / "spec_schema.json"


def _run_result(stdout: str = "ok\n", exit_code: int = 0) -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


def _spec(*strategies: dict) -> dict:
    return {"verification": {"strategies": list(strategies)}}


# --------------------------------------------------------------------------- #
# Unknown / malformed / unavailable declared checks → ERROR                    #
# --------------------------------------------------------------------------- #

def test_unknown_strategy_type_is_error():
    result = verify_run(_spec({"type": "no_such_strategy"}), _run_result())
    assert result.status == Status.ERROR


def test_strategy_without_type_is_error():
    result = verify_run(_spec({"pattern": "ok"}), _run_result())
    assert result.status == Status.ERROR


def test_custom_script_is_error_not_skip():
    result = verify_run(_spec({"type": "custom_script", "script": "x.sh"}), _run_result())
    assert result.status == Status.ERROR


def test_empty_stdout_pattern_is_error():
    result = verify_run(_spec({"type": "stdout_pattern", "pattern": ""}), _run_result())
    assert result.status == Status.ERROR


def test_missing_stdout_pattern_is_error():
    result = verify_run(_spec({"type": "stdout_pattern"}), _run_result())
    assert result.status == Status.ERROR


def test_empty_stdout_exclude_pattern_is_error():
    result = verify_run(_spec({"type": "stdout_exclude_pattern", "pattern": ""}), _run_result())
    assert result.status == Status.ERROR


def test_unknown_strategy_cannot_vanish_behind_passing_checks():
    """The exact mechanism Task 1 closes: a declared-but-unexecutable check
    must not be dropped from the conjunction while the other checks pass."""
    spec = _spec(
        {"type": "exit_code", "expected": 0},
        {"type": "no_such_strategy"},
        {"type": "stdout_pattern", "pattern": "ok"},
    )
    result = verify_run(spec, _run_result())
    assert result.status == Status.ERROR


def test_conjunction_is_preserved():
    """All declared positive and negative checks remain conjunctive."""
    spec = _spec(
        {"type": "stdout_pattern", "pattern": "ok"},
        {"type": "exit_code", "expected": 1},
    )
    result = verify_run(spec, _run_result(exit_code=0))
    assert result.status == Status.FAIL

    spec_all_pass = _spec(
        {"type": "stdout_pattern", "pattern": "ok"},
        {"type": "exit_code", "expected": 0},
        {"type": "stdout_exclude_pattern", "pattern": "FAILED"},
    )
    result = verify_run(spec_all_pass, _run_result())
    assert result.status == Status.PASS


# --------------------------------------------------------------------------- #
# file_diff — exact comparison, implemented (no longer SKIP)                   #
# --------------------------------------------------------------------------- #

def _file_diff_spec(path: str = "out.dat", reference_file: str = "ref.dat") -> dict:
    return _spec({"type": "file_diff", "path": path, "reference_file": reference_file})


def test_file_diff_identical_files_pass(tmp_path: Path):
    (tmp_path / "out.dat").write_bytes(b"payload\n")
    (tmp_path / "ref.dat").write_bytes(b"payload\n")
    result = verify_run(_file_diff_spec(), _run_result(), working_dir=tmp_path)
    assert result.status == Status.PASS


def test_file_diff_different_files_fail(tmp_path: Path):
    (tmp_path / "out.dat").write_bytes(b"payload\n")
    (tmp_path / "ref.dat").write_bytes(b"other\n")
    result = verify_run(_file_diff_spec(), _run_result(), working_dir=tmp_path)
    assert result.status == Status.FAIL


def test_file_diff_missing_output_file_fails(tmp_path: Path):
    (tmp_path / "ref.dat").write_bytes(b"payload\n")
    result = verify_run(_file_diff_spec(), _run_result(), working_dir=tmp_path)
    assert result.status == Status.FAIL


def test_file_diff_missing_reference_file_is_error(tmp_path: Path):
    """A reference the harness cannot read means the declared check cannot
    execute — ERROR, not a verdict about the program."""
    (tmp_path / "out.dat").write_bytes(b"payload\n")
    result = verify_run(_file_diff_spec(), _run_result(), working_dir=tmp_path)
    assert result.status == Status.ERROR


def test_file_diff_output_path_escape_is_error(tmp_path: Path):
    (tmp_path / "ref.dat").write_bytes(b"payload\n")
    outside = tmp_path.parent / "escape_target.dat"
    outside.write_bytes(b"payload\n")
    try:
        result = verify_run(
            _file_diff_spec(path="../escape_target.dat"),
            _run_result(),
            working_dir=tmp_path,
        )
    finally:
        outside.unlink(missing_ok=True)
    assert result.status == Status.ERROR


def test_file_diff_missing_fields_is_error(tmp_path: Path):
    result = verify_run(_spec({"type": "file_diff"}), _run_result(), working_dir=tmp_path)
    assert result.status == Status.ERROR


def test_file_diff_requires_working_dir():
    result = verify_run(_file_diff_spec(), _run_result(), working_dir=None)
    assert result.status == Status.ERROR


# --------------------------------------------------------------------------- #
# Malformed declared strategies → ERROR (Codex gate findings, 2026-08-10)      #
# --------------------------------------------------------------------------- #

def test_exit_code_missing_expected_is_error():
    """exit_code without 'expected' must not silently default to 0 and PASS."""
    result = verify_run(_spec({"type": "exit_code"}), _run_result(exit_code=0))
    assert result.status == Status.ERROR


def test_exit_code_non_integer_expected_is_error():
    """A string '0' is a malformed strategy (ERROR), not a run verdict (FAIL)."""
    result = verify_run(
        _spec({"type": "exit_code", "expected": "0"}), _run_result(exit_code=0)
    )
    assert result.status == Status.ERROR


def test_exit_code_integral_float_expected_is_accepted():
    """Draft-7 'integer' accepts 0.0; the runtime check must agree."""
    result = verify_run(
        _spec({"type": "exit_code", "expected": 0.0}), _run_result(exit_code=0)
    )
    assert result.status == Status.PASS

    result = verify_run(
        _spec({"type": "exit_code", "expected": 0.5}), _run_result(exit_code=0)
    )
    assert result.status == Status.ERROR


def test_non_dict_strategy_entry_is_error_not_crash():
    """A null/non-object entry in the strategies list must ERROR, not raise."""
    result = verify_run({"verification": {"strategies": [None]}}, _run_result())
    assert result.status == Status.ERROR

    result = verify_run(
        {"verification": {"strategies": ["exit_code"]}}, _run_result()
    )
    assert result.status == Status.ERROR


def test_numeric_comparison_missing_extract_regex_is_error():
    result = verify_run(
        _spec({"type": "numeric_comparison", "expected": 0.95}), _run_result()
    )
    assert result.status == Status.ERROR


def test_numeric_comparison_missing_expected_is_error():
    result = verify_run(
        _spec({"type": "numeric_comparison", "extract_regex": r"val=(\S+)"}),
        _run_result(),
    )
    assert result.status == Status.ERROR


def test_file_hash_missing_fields_is_error(tmp_path: Path):
    result = verify_run(
        _spec({"type": "file_hash", "path": "out.dat"}),
        _run_result(),
        working_dir=tmp_path,
    )
    assert result.status == Status.ERROR


def _chmod_unreadable_supported(p: Path) -> bool:
    import os

    p.chmod(0o000)
    readable = os.access(p, os.R_OK)
    return not readable


def test_file_hash_unreadable_output_is_error(tmp_path: Path):
    """An existing but unreadable output file must ERROR, not crash."""
    out = tmp_path / "out.dat"
    out.write_bytes(b"payload\n")
    if not _chmod_unreadable_supported(out):
        out.chmod(0o644)
        import pytest

        pytest.skip("cannot make file unreadable on this host (root?)")
    try:
        result = verify_run(
            _spec({
                "type": "file_hash",
                "path": "out.dat",
                "expected_sha256": "0" * 64,
            }),
            _run_result(),
            working_dir=tmp_path,
        )
    finally:
        out.chmod(0o644)
    assert result.status == Status.ERROR


def test_file_diff_unreadable_output_is_error(tmp_path: Path):
    """An existing but unreadable output file must ERROR, not crash."""
    (tmp_path / "ref.dat").write_bytes(b"payload\n")
    out = tmp_path / "out.dat"
    out.write_bytes(b"payload\n")
    if not _chmod_unreadable_supported(out):
        out.chmod(0o644)
        import pytest

        pytest.skip("cannot make file unreadable on this host (root?)")
    try:
        result = verify_run(_file_diff_spec(), _run_result(), working_dir=tmp_path)
    finally:
        out.chmod(0o644)
    assert result.status == Status.ERROR


# --------------------------------------------------------------------------- #
# Schema: custom_script and its method value are no longer accepted            #
# --------------------------------------------------------------------------- #

def _schema() -> dict:
    return json.loads(SPEC_SCHEMA.read_text())


def test_schema_rejects_custom_script_strategy():
    schema = _schema()
    strategy_types = (
        schema["properties"]["verification"]["properties"]["strategies"]["items"]
        ["properties"]["type"]["enum"]
    )
    assert "custom_script" not in strategy_types
    assert "file_diff" in strategy_types


def test_schema_rejects_external_script_method():
    schema = _schema()
    methods = schema["properties"]["verification"]["properties"]["method"]["enum"]
    assert "external_script" not in methods


def test_schema_requires_exit_code_expected():
    """All 206 specs already declare 'expected'; the schema now requires it."""
    schema = _schema()
    items = schema["properties"]["verification"]["properties"]["strategies"]["items"]
    exit_code_clauses = [
        clause
        for clause in items.get("allOf", [])
        if clause.get("if", {}).get("properties", {}).get("type", {}).get("const")
        == "exit_code"
    ]
    assert len(exit_code_clauses) == 1
    assert "expected" in exit_code_clauses[0]["then"]["required"]


def test_schema_requires_file_diff_fields():
    """file_diff must declare both its output path and its reference file."""
    schema = _schema()
    items = schema["properties"]["verification"]["properties"]["strategies"]["items"]
    file_diff_clauses = [
        clause
        for clause in items.get("allOf", [])
        if clause.get("if", {}).get("properties", {}).get("type", {}).get("const")
        == "file_diff"
    ]
    assert len(file_diff_clauses) == 1
    required = set(file_diff_clauses[0]["then"]["required"])
    assert {"path", "reference_file"} <= required
