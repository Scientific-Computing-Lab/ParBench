"""Phase 2 / Plan 02-05 tests: derive_l0_passers filter semantics (D-22)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.derive_l0_passers import derive_passers, main


MODEL = "together-qwen-3.5-397b-a17b"


def _write_result(tmp_path: Path, filename: str, *, model: str = MODEL, augment_level: int = 0,
                  source_spec: str = "rodinia-bfs-cuda", target_spec: str = "rodinia-bfs-omp",
                  overall_status: str = "PASS") -> Path:
    obj = {
        "model": model,
        "augment_level": augment_level,
        "source_spec": source_spec,
        "target_spec": target_spec,
        "overall_status": overall_status,
    }
    p = tmp_path / filename
    p.write_text(json.dumps(obj))
    return p


def test_three_of_three_pass_included(tmp_path, capsys):
    """D-22 case 1: 3/3 PASS -> included, no warning."""
    for i in range(3):
        _write_result(tmp_path, f"cell-s{i}.json", overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL)
    assert len(passers) == 1
    assert passers[0] == {"source_spec": "rodinia-bfs-cuda", "target_spec": "rodinia-bfs-omp", "augment_level": 0}
    err = capsys.readouterr().err
    assert "incomplete" not in err
    assert "0 samples" not in err


def test_one_of_three_pass_included(tmp_path, capsys):
    """D-22 case 2: 1/3 PASS -> included, no warning."""
    _write_result(tmp_path, "cell-s0.json", overall_status="PASS")
    _write_result(tmp_path, "cell-s1.json", overall_status="BUILD_FAIL")
    _write_result(tmp_path, "cell-s2.json", overall_status="VERIFY_FAIL")
    passers = derive_passers(tmp_path, MODEL)
    assert len(passers) == 1
    err = capsys.readouterr().err
    assert "incomplete" not in err


def test_zero_of_three_pass_excluded(tmp_path, capsys):
    """D-22 case 3: 0/3 PASS -> excluded, no warning (samples complete)."""
    for i in range(3):
        _write_result(tmp_path, f"cell-s{i}.json", overall_status="BUILD_FAIL")
    passers = derive_passers(tmp_path, MODEL)
    assert passers == []
    err = capsys.readouterr().err
    assert "incomplete" not in err


def test_two_samples_one_pass_included_with_warning(tmp_path, capsys):
    """D-22 case 4: 2 samples, 1 PASS -> included + incomplete-sample warning."""
    _write_result(tmp_path, "cell-s0.json", overall_status="PASS")
    _write_result(tmp_path, "cell-s1.json", overall_status="BUILD_FAIL")
    passers = derive_passers(tmp_path, MODEL)
    assert len(passers) == 1
    err = capsys.readouterr().err
    assert "2/3" in err or "incomplete" in err


def test_zero_samples_vacuous(tmp_path, capsys):
    """D-22 case 5: no matching files -> empty passer list (0 samples means no cell discovered)."""
    # Put a file that does NOT match the model filter - should be ignored.
    _write_result(tmp_path, "other-model.json", model="azure-gpt-5.4", overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL)
    assert passers == []


def test_model_filter_excludes_other_models(tmp_path):
    """Only files with the target model id count."""
    _write_result(tmp_path, "wrong-model-pass.json", model="gpt-4o", overall_status="PASS")
    _write_result(tmp_path, "right-model-fail.json", model=MODEL, overall_status="BUILD_FAIL")
    passers = derive_passers(tmp_path, MODEL)
    assert passers == []


def test_augment_level_filter_excludes_non_L0(tmp_path):
    """Only augment_level==0 results count."""
    for lvl in (1, 2, 3, 4):
        _write_result(tmp_path, f"L{lvl}.json", augment_level=lvl, overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL)
    assert passers == []


def test_unparseable_json_is_skipped(tmp_path, capsys):
    """Unparseable JSON -> stderr warning, not a crash. Good file still processed."""
    (tmp_path / "broken.json").write_text("{not json")
    _write_result(tmp_path, "good.json", overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL)
    assert len(passers) == 1
    err = capsys.readouterr().err
    assert "failed to load" in err and "broken.json" in err


def test_main_writes_default_out_path(tmp_path, monkeypatch, capsys):
    """Default --out path = .planning/eval-selections/l0_passers_{model}.json."""
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_result(canonical, "good.json", overall_status="PASS")
    rc = main(["--canonical-dir", str(canonical), "--model", MODEL])
    assert rc == 0
    out = tmp_path / ".planning" / "eval-selections" / f"l0_passers_{MODEL}.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["augment_level"] == 0


def test_missing_canonical_dir_returns_1(tmp_path, capsys):
    """Non-existent dir -> exit code 1 with stderr error."""
    rc = main(["--canonical-dir", str(tmp_path / "nope"), "--model", MODEL])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


# --- S7c Stage 2: --direction filter (D-?? TBD) ---------------------------------

def test_direction_filter_keeps_matching_entries(tmp_path):
    """S7c: direction='cuda-to-omp' keeps cells whose source.api=cuda and target.api=omp."""
    _write_result(tmp_path, "c2o-pass.json",
                  source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                  overall_status="PASS")
    _write_result(tmp_path, "o2c-pass.json",
                  source_spec="rodinia-nw-omp", target_spec="rodinia-nw-cuda",
                  overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL, direction="cuda-to-omp")
    assert len(passers) == 1
    assert passers[0]["source_spec"] == "rodinia-bfs-cuda"
    assert passers[0]["target_spec"] == "rodinia-bfs-omp"


def test_direction_filter_drops_non_matching_entries(tmp_path):
    """S7c: direction='cuda-to-omp' drops omp-to-cuda cells even if they PASS."""
    _write_result(tmp_path, "o2c-pass.json",
                  source_spec="rodinia-nw-omp", target_spec="rodinia-nw-cuda",
                  overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL, direction="cuda-to-omp")
    assert passers == []


def test_direction_none_preserves_legacy_behavior(tmp_path):
    """S7c: direction=None keeps both directions (pre-S7c default behavior)."""
    _write_result(tmp_path, "c2o-pass.json",
                  source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                  overall_status="PASS")
    _write_result(tmp_path, "o2c-pass.json",
                  source_spec="rodinia-nw-omp", target_spec="rodinia-nw-cuda",
                  overall_status="PASS")
    passers = derive_passers(tmp_path, MODEL, direction=None)
    assert len(passers) == 2
