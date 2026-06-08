"""Unit tests for the file_hash verifier strategy."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from harness.models import RunResult, Status
from harness.verifier import verify_run


def _make_run() -> RunResult:
    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=0,
        stdout="",
        stderr="",
    )


def _spec_with_file_hash(path: str, expected_sha256: str) -> dict:
    return {
        "verification": {
            "method": "reference_comparison",
            "strategies": [
                {
                    "type": "file_hash",
                    "path": path,
                    "expected_sha256": expected_sha256,
                }
            ],
        }
    }


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_file_hash_correct_sha_passes(tmp_path: Path):
    content = b"deterministic output\n"
    expected = _sha256_of(content)
    (tmp_path / "result.txt").write_bytes(content)

    spec = _spec_with_file_hash("result.txt", expected)
    result = verify_run(spec, _make_run(), working_dir=tmp_path)

    assert result.status == Status.PASS
    assert result.strategy_used == "file_hash"


def test_file_hash_wrong_sha_fails(tmp_path: Path):
    (tmp_path / "result.txt").write_bytes(b"actual content\n")
    wrong = _sha256_of(b"different content\n")

    spec = _spec_with_file_hash("result.txt", wrong)
    result = verify_run(spec, _make_run(), working_dir=tmp_path)

    assert result.status == Status.FAIL
    assert result.strategy_used == "file_hash"


def test_file_hash_missing_file_fails(tmp_path: Path):
    spec = _spec_with_file_hash("does_not_exist.txt", "0" * 64)
    result = verify_run(spec, _make_run(), working_dir=tmp_path)

    assert result.status == Status.FAIL
    assert result.strategy_used == "file_hash"


def test_file_hash_none_working_dir_errors(tmp_path: Path):
    spec = _spec_with_file_hash("result.txt", "0" * 64)
    result = verify_run(spec, _make_run(), working_dir=None)

    assert result.status == Status.ERROR
    assert result.strategy_used == "file_hash"


def test_file_hash_absolute_path_fails(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"leaked\n")
    try:
        spec = _spec_with_file_hash(str(outside), _sha256_of(b"leaked\n"))
        result = verify_run(spec, _make_run(), working_dir=tmp_path)
    finally:
        outside.unlink(missing_ok=True)

    assert result.status == Status.FAIL
    assert result.strategy_used == "file_hash"
    assert "escape" in result.details.lower() or "working_dir" in result.details.lower()


def test_file_hash_parent_traversal_fails(tmp_path: Path):
    outside = tmp_path.parent / "leak.txt"
    outside.write_bytes(b"traversal\n")
    try:
        spec = _spec_with_file_hash("../leak.txt", _sha256_of(b"traversal\n"))
        result = verify_run(spec, _make_run(), working_dir=tmp_path)
    finally:
        outside.unlink(missing_ok=True)

    assert result.status == Status.FAIL
    assert result.strategy_used == "file_hash"


def test_file_hash_symlink_outside_fails(tmp_path: Path):
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_bytes(b"symlinked\n")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    try:
        spec = _spec_with_file_hash("link.txt", _sha256_of(b"symlinked\n"))
        result = verify_run(spec, _make_run(), working_dir=tmp_path)
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)

    assert result.status == Status.FAIL
    assert result.strategy_used == "file_hash"


def test_file_hash_uppercase_sha_passes(tmp_path: Path):
    content = b"uppercase hex spec\n"
    expected_upper = _sha256_of(content).upper()
    (tmp_path / "result.txt").write_bytes(content)

    spec = _spec_with_file_hash("result.txt", expected_upper)
    result = verify_run(spec, _make_run(), working_dir=tmp_path)

    assert result.status == Status.PASS
    assert result.strategy_used == "file_hash"
