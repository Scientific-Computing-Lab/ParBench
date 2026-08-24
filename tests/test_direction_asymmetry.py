"""Task 6 (gate E-01): corrected statistical unit for direction asymmetry.

The submitted analysis keyed its pass lookup without ``sample_id``, so with
three stochastic L0 samples per task cell only the last-loaded sample
survived. The corrected rule collapses the three samples of one
(model, suite, kernel, direction) cell by any-sample success and pairs
directions within the same cell. Both analysis scripts must consume the one
shared helper (``scripts/analysis/paired_tasks.py``) and produce identical
paired cells, byte-for-byte, regardless of ``PYTHONHASHSEED``.

The fixture (tests/fixtures/direction_asymmetry/) encodes cells where
any-of-three and last-sample-wins disagree, so the table assertions below
fail on the pre-Task-6 code and pass after it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
FIXTURE_RESULTS = PROJECT_ROOT / "tests" / "fixtures" / "direction_asymmetry" / "results"

sys.path.insert(0, str(SCRIPTS / "analysis"))
import paired_tasks  # noqa: E402


def run(args: list[str], hashseed: str = "0") -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=300,
    )


def run_statistical(out_dir: Path, hashseed: str = "0") -> dict:
    r = run([
        str(SCRIPTS / "analysis" / "statistical_analysis.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(FIXTURE_RESULTS),
        "--output-dir", str(out_dir),
    ], hashseed=hashseed)
    assert r.returncode == 0, r.stderr
    return json.loads((out_dir / "statistical_analysis.json").read_text())


def run_quantitative(out_dir: Path, hashseed: str = "0") -> dict:
    r = run([
        str(SCRIPTS / "analysis" / "quantitative_findings.py"),
        "--project-root", str(PROJECT_ROOT),
        "--results-root", str(FIXTURE_RESULTS),
        "--output-dir", str(out_dir),
    ], hashseed=hashseed)
    assert r.returncode == 0, r.stderr
    return json.loads((out_dir / "quantitative_findings.json").read_text())


def _entry_for(stat: dict, d1: str, d2: str) -> dict:
    matches = [
        e for e in stat["direction_asymmetry"]
        if {e["forward_direction"], e["reverse_direction"]} == {d1, d2}
    ]
    assert len(matches) == 1, stat["direction_asymmetry"]
    return matches[0]


def _qf_entry_for(qf: dict, d1: str, d2: str) -> dict:
    asym = qf["canonical"]["direction_asymmetry"]
    matches = [v for k, v in asym.items() if d1 in k and d2 in k]
    assert len(matches) == 1, list(asym)
    return matches[0]["value"]


# ---------------------------------------------------------------------------
# Shared-helper unit semantics
# ---------------------------------------------------------------------------

def _rec(src, tgt, status, level=0, sample=0, model="m1"):
    return {
        "source_spec": src,
        "target_spec": tgt,
        "model": model,
        "direction": f"{src.rsplit('-', 1)[-1]}-to-{tgt.rsplit('-', 1)[-1]}",
        "overall_status": status,
        "augment_level": level,
        "sample_id": sample,
    }


def test_collapse_any_of_three_not_last_sample():
    """The synthetic three-sample case the overwrite semantics misclassifies:
    PASS on s0 followed by two failures must collapse to True."""
    records = [
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS", sample=0),
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "VERIFY_FAIL", sample=1),
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "VERIFY_FAIL", sample=2),
    ]
    collapsed = paired_tasks.collapse_l0_any_success(records)
    assert collapsed == {("m1", "rodinia", "bfs", "cuda-to-omp"): True}


def test_collapse_all_fail_is_false():
    records = [
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "VERIFY_FAIL", sample=s)
        for s in range(3)
    ]
    collapsed = paired_tasks.collapse_l0_any_success(records)
    assert collapsed == {("m1", "rodinia", "bfs", "cuda-to-omp"): False}


def test_collapse_excludes_augmented_levels():
    records = [
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "VERIFY_FAIL", level=0),
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS", level=2),
    ]
    collapsed = paired_tasks.collapse_l0_any_success(records)
    assert collapsed == {("m1", "rodinia", "bfs", "cuda-to-omp"): False}


def test_collapse_separates_suites_models():
    records = [
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "PASS", model="m1"),
        _rec("hecbench-bfs-cuda", "hecbench-bfs-omp", "VERIFY_FAIL", model="m1"),
        _rec("rodinia-bfs-cuda", "rodinia-bfs-omp", "VERIFY_FAIL", model="m2"),
    ]
    collapsed = paired_tasks.collapse_l0_any_success(records)
    assert collapsed[("m1", "rodinia", "bfs", "cuda-to-omp")] is True
    assert collapsed[("m1", "hecbench", "bfs", "cuda-to-omp")] is False
    assert collapsed[("m2", "rodinia", "bfs", "cuda-to-omp")] is False


def test_paired_cells_pair_only_within_cell():
    collapsed = {
        ("m1", "rodinia", "bfs", "cuda-to-omp"): True,
        ("m1", "rodinia", "bfs", "omp-to-cuda"): False,
        ("m1", "rodinia", "nw", "cuda-to-omp"): True,  # no reverse: unpaired
        ("m2", "rodinia", "bfs", "omp-to-cuda"): True,  # other model: unpaired
    }
    pairs = paired_tasks.paired_direction_cells(collapsed, "cuda-to-omp", "omp-to-cuda")
    assert pairs == [(True, False)]


def test_paired_cell_details_are_sorted_and_deterministic():
    collapsed = {
        ("m1", "rodinia", "nw", "cuda-to-omp"): True,
        ("m1", "rodinia", "nw", "omp-to-cuda"): True,
        ("m1", "rodinia", "bfs", "cuda-to-omp"): False,
        ("m1", "rodinia", "bfs", "omp-to-cuda"): True,
    }
    details = paired_tasks.paired_direction_cell_details(
        collapsed, "cuda-to-omp", "omp-to-cuda"
    )
    assert [d["kernel"] for d in details] == ["bfs", "nw"]


# ---------------------------------------------------------------------------
# Regression on the fixture: statistical_analysis.py
# ---------------------------------------------------------------------------

def test_statistical_regression_three_sample_cells(tmp_path):
    """Fails under the overwrite semantics (table would be both_fail=2)."""
    stat = run_statistical(tmp_path)
    entry = _entry_for(stat, "cuda-to-omp", "omp-to-cuda")
    assert entry["n_paired"] == 2
    assert entry["table"]["both_pass"] == 0
    assert entry["table"]["both_fail"] == 0
    assert entry["discordant"] == 2


def test_statistical_reports_five_pairs_and_paired_cells(tmp_path):
    stat = run_statistical(tmp_path)
    assert len(stat["direction_asymmetry"]) == 5
    for entry in stat["direction_asymmetry"]:
        cells = entry["paired_cells"]
        assert len(cells) == entry["n_paired"]
        keys = [(c["model"], c["suite"], c["kernel"]) for c in cells]
        assert keys == sorted(keys)
        for c in cells:
            assert set(c) == {"model", "suite", "kernel",
                              "forward_pass", "reverse_pass"}


# ---------------------------------------------------------------------------
# Regression on the fixture: quantitative_findings.py
# ---------------------------------------------------------------------------

def test_quantitative_regression_three_sample_cells(tmp_path):
    qf = run_quantitative(tmp_path)
    value = _qf_entry_for(qf, "cuda-to-omp", "omp-to-cuda")
    test_result = value["test_result"]
    assert test_result["n_paired"] == 2
    assert test_result["table"]["both_pass"] == 0
    assert test_result["table"]["both_fail"] == 0
    assert test_result["discordant"] == 2
    assert len(value["paired_cells"]) == 2


def test_quantitative_reports_five_pairs(tmp_path):
    qf = run_quantitative(tmp_path)
    assert len(qf["canonical"]["direction_asymmetry"]) == 5


# ---------------------------------------------------------------------------
# PYTHONHASHSEED byte-identity on the generated paired cells
# ---------------------------------------------------------------------------

def _asym_bytes_statistical(data: dict) -> bytes:
    return json.dumps(data["direction_asymmetry"], sort_keys=False).encode()


def _asym_bytes_quantitative(data: dict) -> bytes:
    return json.dumps(
        data["canonical"]["direction_asymmetry"], sort_keys=False
    ).encode()


def test_hashseed_byte_identity_statistical(tmp_path):
    a = run_statistical(tmp_path / "a", hashseed="0")
    b = run_statistical(tmp_path / "b", hashseed="424242")
    assert _asym_bytes_statistical(a) == _asym_bytes_statistical(b)


def test_hashseed_byte_identity_quantitative(tmp_path):
    a = run_quantitative(tmp_path / "a", hashseed="0")
    b = run_quantitative(tmp_path / "b", hashseed="424242")
    assert _asym_bytes_quantitative(a) == _asym_bytes_quantitative(b)


# ---------------------------------------------------------------------------
# Cross-path agreement via the verifier CLI
# ---------------------------------------------------------------------------

def test_verify_direction_pairs_cli(tmp_path):
    run_statistical(tmp_path / "statistical")
    run_quantitative(tmp_path / "quantitative")
    r = run([
        str(SCRIPTS / "analysis" / "verify_direction_pairs.py"),
        "--statistical", str(tmp_path / "statistical" / "statistical_analysis.json"),
        "--quantitative", str(tmp_path / "quantitative" / "quantitative_findings.json"),
        "--expected-pairs", "5",
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_verify_direction_pairs_detects_mismatch(tmp_path):
    stat = run_statistical(tmp_path / "statistical")
    run_quantitative(tmp_path / "quantitative")
    # Tamper one paired cell in the statistical output.
    entry = _entry_for(stat, "cuda-to-omp", "omp-to-cuda")
    entry["paired_cells"][0]["forward_pass"] = (
        not entry["paired_cells"][0]["forward_pass"]
    )
    tampered = tmp_path / "statistical" / "statistical_analysis.json"
    tampered.write_text(json.dumps(stat))
    r = run([
        str(SCRIPTS / "analysis" / "verify_direction_pairs.py"),
        "--statistical", str(tampered),
        "--quantitative", str(tmp_path / "quantitative" / "quantitative_findings.json"),
        "--expected-pairs", "5",
    ])
    assert r.returncode != 0
