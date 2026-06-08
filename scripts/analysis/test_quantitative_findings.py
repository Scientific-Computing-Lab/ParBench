#!/usr/bin/env python3
"""Tests for quantitative_findings.py — TDD RED phase.

Covers:
  Test 1: KNOWN_FAIL exclusion removes all 9 specs (7 Rodinia + 2 HeCBench)
  Test 2: Campaign split produces C1 (temp=0.0) and C2 (temp=0.7) with no overlap
  Test 3: Wilson CI returns dict with value, ci_lower, ci_upper, ci_level=0.95, n fields
  Test 4: Provenance wrapper includes value, source, files_matched, derivation fields
  Test 5: Suite extraction from source_spec correctly parses all 5 suites
  Test 6: Direction extraction handles omp_target correctly (not merged with omp)
  Test 7: Augment level extraction: no suffix->0, -L1->1, -L4->4, -s0->0 (seed, not level)
  Test 8: Error taxonomy classifies BUILD_FAIL, RUN_FAIL, VERIFY_FAIL, EXTRACTION_FAIL
  Test 9: McNemar test pairs on (suite, kernel) for forward/reverse directions
  Test 10: Cochran-Armitage computed per-direction AND aggregate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the analysis scripts directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import quantitative_findings as qf  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — synthetic result records
# ---------------------------------------------------------------------------

def _make_record(
    source_spec: str = "rodinia-bfs-cuda",
    target_spec: str = "rodinia-bfs-omp",
    overall_status: str = "PASS",
    temperature: float = 0.0,
    augment_level: int = 0,
    sample_id: int | None = None,
    direction: str | None = None,
    build_error_snippet: str | None = None,
    run_exit_code: int | None = None,
    run_stderr_snippet: str | None = None,
    run_stdout_snippet: str | None = None,
    error_message: str | None = None,
    verify_status: str | None = None,
    attempts: list | None = None,
) -> dict:
    """Create a synthetic result record for testing."""
    stem_parts = f"{source_spec}-to-{target_spec}"
    if augment_level > 0:
        stem_parts += f"-L{augment_level}"
    if sample_id is not None:
        stem_parts += f"-s{sample_id}"

    rec = {
        "source_spec": source_spec,
        "target_spec": target_spec,
        "overall_status": overall_status,
        "temperature": temperature,
        "augment_level": augment_level,
        "sample_id": sample_id,
        "build_error_snippet": build_error_snippet,
        "run_exit_code": run_exit_code,
        "run_stderr_snippet": run_stderr_snippet,
        "run_stdout_snippet": run_stdout_snippet,
        "error_message": error_message,
        "verify_status": verify_status,
        "attempts": attempts or [],
        "_filename": f"{stem_parts}.json",
        "_stem": stem_parts,
    }
    if direction is None:
        rec["direction"] = qf._direction_from_data(rec)
    else:
        rec["direction"] = direction
    rec["kernel"] = qf._kernel_from_spec(source_spec)
    rec["_suite"] = qf._suite_from_spec(source_spec)
    return rec


# ---------------------------------------------------------------------------
# Test 1: KNOWN_FAIL exclusion removes all 9 specs
# ---------------------------------------------------------------------------

def test_known_fail_exclusion():
    """All 9 KNOWN_FAIL specs (7 Rodinia + 2 HeCBench) are excluded."""
    records = [
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp"),
        _make_record(source_spec="rodinia-kmeans-cuda", target_spec="rodinia-kmeans-omp"),
        _make_record(source_spec="rodinia-mummergpu-cuda", target_spec="rodinia-mummergpu-omp"),
        _make_record(source_spec="rodinia-mummergpu-omp", target_spec="rodinia-mummergpu-cuda"),
        _make_record(source_spec="rodinia-hybridsort-cuda", target_spec="rodinia-hybridsort-omp"),
        _make_record(source_spec="rodinia-nn-opencl", target_spec="rodinia-nn-cuda"),
        _make_record(source_spec="rodinia-kmeans-opencl", target_spec="rodinia-kmeans-cuda"),
        _make_record(source_spec="rodinia-backprop-opencl", target_spec="rodinia-backprop-cuda"),
        _make_record(source_spec="hecbench-stencil1d-omp_target", target_spec="hecbench-stencil1d-cuda"),
        _make_record(source_spec="hecbench-scan-omp_target", target_spec="hecbench-scan-cuda"),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp"),
    ]
    filtered = qf.exclude_known_fail(records)
    # Only 2 records survive: bfs and hotspot
    assert len(filtered) == 2
    specs_remaining = {r["source_spec"] for r in filtered}
    assert "rodinia-bfs-cuda" in specs_remaining
    assert "rodinia-hotspot-cuda" in specs_remaining

    # Verify the constant has exactly 9 entries
    assert len(qf.EXCLUDED_SPECS) == 9
    assert "rodinia-backprop-opencl" in qf.EXCLUDED_SPECS
    assert "hecbench-stencil1d-omp_target" in qf.EXCLUDED_SPECS
    assert "hecbench-scan-omp_target" in qf.EXCLUDED_SPECS


# ---------------------------------------------------------------------------
# Test 2: Campaign split produces C1/C2 with no overlap
# ---------------------------------------------------------------------------

def test_campaign_split():
    """Campaign split: C1=temp 0.0, C2=temp 0.7, no overlap."""
    records = [
        _make_record(temperature=0.0),
        _make_record(temperature=0.0),
        _make_record(temperature=0.7),
    ]
    c1, c2 = qf.split_campaigns(records)
    assert len(c1) == 2
    assert len(c2) == 1
    assert all(r["temperature"] == 0.0 for r in c1)
    assert all(r["temperature"] > 0.0 for r in c2)


# ---------------------------------------------------------------------------
# Test 3: Wilson CI returns dict with required fields
# ---------------------------------------------------------------------------

def test_wilson_ci_fields():
    """Wilson CI dict has value, ci_lower, ci_upper, ci_level, n."""
    result = qf.wilson_ci(7, 10)
    assert "value" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "ci_level" in result
    assert "n" in result
    assert result["ci_level"] == 0.95
    assert result["n"] == 10
    assert 0 <= result["ci_lower"] <= result["value"] <= result["ci_upper"] <= 1.0


def test_wilson_ci_zero():
    """Wilson CI with zero total returns zeros."""
    result = qf.wilson_ci(0, 0)
    assert result["value"] == 0.0
    assert result["n"] == 0


# ---------------------------------------------------------------------------
# Test 4: Provenance wrapper
# ---------------------------------------------------------------------------

def test_make_finding_fields():
    """Provenance wrapper returns dict with required fields."""
    finding = qf.make_finding(
        value=0.362,
        source="paper_data.json",
        files_matched=710,
        derivation="PASS count / total primary campaign records",
    )
    assert finding["value"] == 0.362
    assert finding["source"] == "paper_data.json"
    assert finding["files_matched"] == 710
    assert finding["derivation"] == "PASS count / total primary campaign records"


def test_make_finding_with_ci():
    """Provenance wrapper with CI fields."""
    finding = qf.make_finding(
        value=0.362,
        source="computed",
        files_matched=710,
        derivation="wilson_ci(passes, total)",
        ci_lower=0.32,
        ci_upper=0.41,
        n=710,
    )
    assert finding["ci_lower"] == 0.32
    assert finding["ci_upper"] == 0.41
    assert finding["ci_level"] == 0.95
    assert finding["n"] == 710


# ---------------------------------------------------------------------------
# Test 5: Suite extraction
# ---------------------------------------------------------------------------

def test_suite_from_spec():
    """Suite extraction parses all 5 suites correctly."""
    assert qf._suite_from_spec("rodinia-bfs-cuda") == "rodinia"
    assert qf._suite_from_spec("hecbench-stencil1d-cuda") == "hecbench"
    assert qf._suite_from_spec("xsbench-xsbench-omp") == "xsbench"
    assert qf._suite_from_spec("rsbench-rsbench-cuda") == "rsbench"
    assert qf._suite_from_spec("mixbench-mixbench-cuda") == "mixbench"


# ---------------------------------------------------------------------------
# Test 6: Direction extraction with omp_target
# ---------------------------------------------------------------------------

def test_direction_from_data_omp_target():
    """omp_target directions are distinct from omp."""
    rec = {"source_spec": "rodinia-bfs-cuda", "target_spec": "hecbench-stencil1d-omp_target"}
    direction = qf._direction_from_data(rec)
    assert direction == "cuda-to-omp_target"
    assert "omp_target" in direction
    assert direction != "cuda-to-omp"


def test_direction_from_data_standard():
    """Standard directions parse correctly."""
    rec = {"source_spec": "rodinia-bfs-cuda", "target_spec": "rodinia-bfs-omp"}
    assert qf._direction_from_data(rec) == "cuda-to-omp"


# ---------------------------------------------------------------------------
# Test 7: Augment level extraction
# ---------------------------------------------------------------------------

def test_augment_level_from_filename():
    """Augment level parsing from filename stems."""
    assert qf._augment_level_from_filename("rodinia-bfs-cuda-to-rodinia-bfs-omp") == 0
    assert qf._augment_level_from_filename("rodinia-bfs-cuda-to-rodinia-bfs-omp-L1") == 1
    assert qf._augment_level_from_filename("rodinia-bfs-cuda-to-rodinia-bfs-omp-L4") == 4
    assert qf._augment_level_from_filename("rodinia-bfs-cuda-to-rodinia-bfs-omp-L2-s0") == 2
    # -s0 alone means sample_id, NOT augment level
    assert qf._augment_level_from_filename("rodinia-bfs-cuda-to-rodinia-bfs-omp-s0") == 0


# ---------------------------------------------------------------------------
# Test 8: Error taxonomy classification
# ---------------------------------------------------------------------------

def test_failure_taxonomy_classification():
    """Failure taxonomy counts BUILD_FAIL, RUN_FAIL, VERIFY_FAIL, EXTRACTION_FAIL."""
    records = [
        _make_record(overall_status="PASS"),
        _make_record(overall_status="BUILD_FAIL", build_error_snippet="undefined reference to foo"),
        _make_record(overall_status="BUILD_FAIL", build_error_snippet="cudaMalloc not found"),
        _make_record(overall_status="RUN_FAIL", run_exit_code=-11),
        _make_record(overall_status="VERIFY_FAIL", verify_status="fail"),
        _make_record(overall_status="EXTRACTION_FAIL", error_message="did not contain parseable code"),
    ]
    result = qf.compute_failure_taxonomy(records)
    assert result["total_failures"] == 5
    assert result["by_status"]["BUILD_FAIL"]["count"] == 2
    assert result["by_status"]["RUN_FAIL"]["count"] == 1
    assert result["by_status"]["VERIFY_FAIL"]["count"] == 1
    assert result["by_status"]["EXTRACTION_FAIL"]["count"] == 1
    # Subcategories for BUILD_FAIL
    assert "subcategories" in result["by_status"]["BUILD_FAIL"]


# ---------------------------------------------------------------------------
# Test 9: McNemar pairs on (suite, kernel) for forward/reverse
# ---------------------------------------------------------------------------

def test_direction_asymmetry_pairs_on_suite_kernel():
    """McNemar pairs on (suite, kernel) to avoid cross-suite false pairs."""
    # Same kernel name "bfs" in two suites — they should NOT be paired
    records = [
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-bfs-omp", target_spec="rodinia-bfs-cuda",
                     overall_status="FAIL", augment_level=0),
        _make_record(source_spec="hecbench-bfs-cuda", target_spec="hecbench-bfs-omp",
                     overall_status="FAIL", augment_level=0),
        _make_record(source_spec="hecbench-bfs-omp", target_spec="hecbench-bfs-cuda",
                     overall_status="PASS", augment_level=0),
    ]
    result = qf.compute_direction_asymmetry(records)
    # Should have a test for the cuda-to-omp / omp-to-cuda pair (key order may vary)
    matching_keys = [k for k in result if "cuda-to-omp" in k and "omp-to-cuda" in k]
    assert len(matching_keys) == 1, f"Expected 1 matching key, got {matching_keys}"
    key = matching_keys[0]
    # The result is a provenance-wrapped finding; extract test_result from value
    finding = result[key]
    test_result = finding["value"]["test_result"]
    # Two paired observations: (rodinia, bfs) and (hecbench, bfs) — separate suites
    assert test_result["n_paired"] == 2


# ---------------------------------------------------------------------------
# Test 10: Cochran-Armitage computed per-direction AND aggregate
# ---------------------------------------------------------------------------

def test_augmentation_trends_per_direction_and_aggregate():
    """Cochran-Armitage trend computed per-direction and aggregate."""
    records = []
    for level in range(5):  # L0-L4
        for i in range(10):  # 10 per level
            status = "PASS" if i < (5 + level) else "BUILD_FAIL"
            records.append(_make_record(
                source_spec="rodinia-bfs-cuda",
                target_spec="rodinia-bfs-omp",
                overall_status=status,
                augment_level=level,
            ))

    result = qf.compute_augmentation_trends(records)
    # Should have aggregate
    assert "aggregate" in result
    assert "cochran_armitage" in result["aggregate"]
    # Should have per_direction
    assert "per_direction" in result
    assert "cuda-to-omp" in result["per_direction"]
    assert "cochran_armitage" in result["per_direction"]["cuda-to-omp"]
    # Should have cohens_h between adjacent levels
    assert "cohens_h_adjacent" in result["aggregate"]


# ---------------------------------------------------------------------------
# QUANT-01: Aggregate pass rates (dimension-level test)
# ---------------------------------------------------------------------------

def test_aggregate_pass_rates():
    """compute_aggregate_pass_rates computes overall + per-suite rates with wilson_ci."""
    records = [
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS"),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="BUILD_FAIL"),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="PASS"),
        _make_record(source_spec="xsbench-xsbench-cuda", target_spec="xsbench-xsbench-omp",
                     overall_status="BUILD_FAIL"),
        _make_record(source_spec="xsbench-xsbench-cuda", target_spec="xsbench-xsbench-omp",
                     overall_status="PASS"),
    ]
    result = qf.compute_aggregate_pass_rates(records)

    # Overall: 3 PASS / 5 total = 0.6
    assert result["overall"]["value"] == 0.6
    assert result["overall"]["n"] == 5
    assert result["overall"]["ci_lower"] <= 0.6 <= result["overall"]["ci_upper"]

    # Per-suite: rodinia = 2/3, xsbench = 1/2
    assert "rodinia" in result["per_suite"]
    assert "xsbench" in result["per_suite"]
    rod = result["per_suite"]["rodinia"]
    assert rod["n"] == 3
    assert round(rod["value"], 4) == round(2 / 3, 4)
    xsb = result["per_suite"]["xsbench"]
    assert xsb["n"] == 2
    assert xsb["value"] == 0.5

    # Provenance fields present on overall
    assert "source" in result["overall"]
    assert "derivation" in result["overall"]
    assert "files_matched" in result["overall"]


# ---------------------------------------------------------------------------
# QUANT-02: Direction pass rates (dimension-level test)
# ---------------------------------------------------------------------------

def test_direction_pass_rates():
    """compute_direction_pass_rates filters L0, separates standard vs case_study."""
    records = [
        # L0 standard directions
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-bfs-omp", target_spec="rodinia-bfs-cuda",
                     overall_status="BUILD_FAIL", augment_level=0),
        # L0 case-study direction
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="hecbench-stencil1d-omp_target",
                     overall_status="PASS", augment_level=0,
                     direction="cuda-to-omp_target"),
        # L2 record -- should be EXCLUDED (L0 only)
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=2),
    ]
    result = qf.compute_direction_pass_rates(records)

    # Standard directions
    assert "cuda-to-omp" in result["standard"]
    assert result["standard"]["cuda-to-omp"]["value"] == 1.0
    assert result["standard"]["cuda-to-omp"]["n"] == 1
    assert "omp-to-cuda" in result["standard"]
    assert result["standard"]["omp-to-cuda"]["value"] == 0.0

    # Case-study directions
    assert "cuda-to-omp_target" in result["case_study"]
    assert result["case_study"]["cuda-to-omp_target"]["value"] == 1.0
    assert result["case_study"]["cuda-to-omp_target"]["n"] == 1

    # L2 record should not appear (L0 filtering)
    total_n = sum(
        v["n"] for v in result["standard"].values()
    ) + sum(
        v["n"] for v in result["case_study"].values()
    )
    assert total_n == 3  # Only the 3 L0 records


# ---------------------------------------------------------------------------
# QUANT-07: pass@k estimates
# ---------------------------------------------------------------------------

def test_pass_at_k():
    """pass@k uses Chen et al. (2021) unbiased estimator: 1 - C(n-c,k)/C(n,k).

    Task 1: c=3 → pass@1 = 3/3 = 1.0,   pass@3 = 1 - C(0,3)/C(3,3) = 1.0
    Task 2: c=1 → pass@1 = 1/3 = 0.333,  pass@3 = 1 - C(2,3)/C(3,3) = 1.0
    Task 3: c=0 → pass@1 = 0/3 = 0.0,    pass@3 = 1 - C(3,3)/C(3,3) = 0.0
    Macro-avg: pass@1 = (1 + 1/3 + 0)/3 = 4/9 ≈ 0.4444
               pass@3 = (1 + 1   + 0)/3 = 2/3 ≈ 0.6667
    """
    records = [
        # Task 1: all 3 seeds pass (c=3)
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, sample_id=0),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, sample_id=1),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, sample_id=2),
        # Task 2: 1 of 3 passes (c=1)
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="PASS", temperature=0.7, sample_id=0),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=1),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=2),
        # Task 3: 0 of 3 pass (c=0)
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=0),
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=1),
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=2),
    ]
    result = qf.compute_pass_at_k(records)

    # 3 tasks total
    assert result["total_tasks"]["value"] == 3

    # Chen formula: pass@1 = macro-avg(c_i/n_i) = (1 + 1/3 + 0)/3 = 4/9
    assert abs(result["pass_at_1"]["value"] - 4 / 9) < 0.001
    # Chen formula: pass@3 = macro-avg(1-C(n-c,3)/C(n,3)) = (1+1+0)/3 = 2/3
    assert abs(result["pass_at_3"]["value"] - 2 / 3) < 0.001

    # MONOTONICITY: pass@1 <= pass@3 (Chen et al. convention)
    assert result["pass_at_1"]["value"] <= result["pass_at_3"]["value"]

    # Task classification unchanged
    assert result["task_classification"]["always_pass"] == 1
    assert result["task_classification"]["hard_fail"] == 1
    assert result["task_classification"]["noisy_fail"] == 1

    # Per-direction and per-suite keys present
    assert "per_direction" in result
    assert "per_suite" in result


def test_pass_at_k_excludes_augmentation_levels():
    """compute_pass_at_k ignores augment_level>0 records (augmentation, not seeds)."""
    records = [
        # Task 1: 3 canonical seeds all PASS (augment_level=0)
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, sample_id=0, augment_level=0),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, sample_id=1, augment_level=0),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, sample_id=2, augment_level=0),
        # Task 1: 4 augmentation records (augment_level 1-4) — some FAIL
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, augment_level=1),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, augment_level=2),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, augment_level=3),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", temperature=0.7, augment_level=4),
        # Task 2: 3 canonical seeds, 1 passes (augment_level=0)
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="PASS", temperature=0.7, sample_id=0, augment_level=0),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=1, augment_level=0),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="BUILD_FAIL", temperature=0.7, sample_id=2, augment_level=0),
    ]
    result = qf.compute_pass_at_k(records)

    # Only 2 tasks (augmentation records excluded from task-set)
    assert result["total_tasks"]["value"] == 2

    # Chen formula (augmentation excluded, only canonical L0 seeds):
    # Task 1: c=3 → pass@1=1.0, pass@3=1.0
    # Task 2: c=1 → pass@1=1/3, pass@3=1.0
    # Macro-avg: pass@1 = (1 + 1/3)/2 = 2/3, pass@3 = (1+1)/2 = 1.0
    assert abs(result["pass_at_1"]["value"] - 2 / 3) < 0.001
    assert abs(result["pass_at_3"]["value"] - 1.0) < 0.001

    # Classification
    assert result["task_classification"]["always_pass"] == 1
    assert result["task_classification"]["noisy_fail"] == 1
    assert result["task_classification"]["hard_fail"] == 0

    # No warnings about unexpected seed counts
    assert len(result.get("warnings", [])) == 0


# ---------------------------------------------------------------------------
# QUANT-08: Per-kernel difficulty tiers
# ---------------------------------------------------------------------------

def test_per_kernel_tiers():
    """compute_per_kernel_tiers ranks kernels by pass rate with quartile tiers."""
    records = []
    # Create 8 kernels with varying pass rates at L0
    kernel_configs = [
        ("bfs", 6, 6),       # 100% pass
        ("hotspot", 5, 6),   # 83%
        ("srad", 4, 6),      # 67%
        ("nw", 3, 6),        # 50%
        ("pathfinder", 2, 6),  # 33%
        ("backprop", 1, 6),  # 17%
        ("lud", 0, 6),       # 0%
        ("cfd", 0, 6),       # 0%
    ]
    for kernel_name, passes, total in kernel_configs:
        for i in range(total):
            status = "PASS" if i < passes else "BUILD_FAIL"
            records.append(_make_record(
                source_spec=f"rodinia-{kernel_name}-cuda",
                target_spec=f"rodinia-{kernel_name}-omp",
                overall_status=status,
                augment_level=0,
            ))

    result = qf.compute_per_kernel_tiers(records)

    # 8 kernels
    assert result["n_kernels"] == 8

    # Top-5 easiest: bfs, hotspot, srad, nw, pathfinder (highest pass rates)
    top5_names = [k["kernel"] for k in result["top_5_easiest"]]
    assert top5_names[0] == "bfs"  # Highest pass rate

    # Bottom-5 hardest: last 5 in sorted order (lowest pass rates)
    bot5_names = [k["kernel"] for k in result["top_5_hardest"]]
    assert "lud" in bot5_names or "cfd" in bot5_names  # 0% kernels should be in bottom

    # Quartile boundaries exist
    assert "quartile_boundaries" in result
    assert "Q1_threshold" in result["quartile_boundaries"]
    assert "Q2_threshold" in result["quartile_boundaries"]
    assert "Q3_threshold" in result["quartile_boundaries"]

    # Kernels list has tier assignments
    for ks in result["kernels"]:
        assert "tier" in ks
        assert ks["tier"] in ("Q1_easiest", "Q2", "Q3", "Q4_hardest")


# ---------------------------------------------------------------------------
# QUANT-09: Translation complexity correlation
# ---------------------------------------------------------------------------

def test_complexity_correlation(tmp_path):
    """compute_complexity_correlation classifies 4 complexity classes from specs."""
    # Create mock spec files
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()

    # Single-file spec (1 translation target)
    (specs_dir / "rodinia-bfs-cuda.json").write_text(json.dumps({
        "files": {"translation_targets": ["bfs.cu"]}
    }))
    (specs_dir / "rodinia-bfs-omp.json").write_text(json.dumps({
        "files": {"translation_targets": ["bfs.cpp"]}
    }))

    # Multi-file spec (2 translation targets)
    (specs_dir / "rodinia-srad-cuda.json").write_text(json.dumps({
        "files": {"translation_targets": ["srad_kernel.cu", "srad_helper.cu"]}
    }))
    (specs_dir / "rodinia-srad-omp.json").write_text(json.dumps({
        "files": {"translation_targets": ["srad.cpp"]}
    }))

    records = [
        # single_file: 1->1
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS"),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="BUILD_FAIL"),
        # multi_to_single: 2->1
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-omp",
                     overall_status="BUILD_FAIL"),
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-omp",
                     overall_status="BUILD_FAIL"),
    ]

    result = qf.compute_complexity_correlation(records, tmp_path)

    # All 4 records classified
    assert result["classified"] == 4
    assert result["unclassified"] == 0

    # Per-class pass rates
    assert "single_file" in result["per_class"]
    assert result["per_class"]["single_file"]["total"] == 2
    assert result["per_class"]["single_file"]["passes"] == 1
    assert result["per_class"]["single_file"]["value"] == 0.5

    assert "multi_to_single" in result["per_class"]
    assert result["per_class"]["multi_to_single"]["total"] == 2
    assert result["per_class"]["multi_to_single"]["passes"] == 0
    assert result["per_class"]["multi_to_single"]["value"] == 0.0

    # Test result present
    assert "test_result" in result


# ---------------------------------------------------------------------------
# QUANT-10: Cross-suite comparison
# ---------------------------------------------------------------------------

def test_cross_suite(tmp_path):
    """compute_cross_suite aggregates per-suite pass rates at L0."""
    # Create minimal sloc_analysis.json
    analysis_dir = tmp_path / "results" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "sloc_analysis.json").write_text(json.dumps({
        "kernels": {
            "bfs": {"suite": "rodinia", "physical_sloc": 101},
            "xsbench": {"suite": "xsbench", "physical_sloc": 450},
        },
        "summary": {"total_kernels": 2}
    }))

    # Create minimal specs dir
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "rodinia-bfs-cuda.json").write_text(json.dumps({
        "identity": {"source_suite": "rodinia"},
        "files": {"translation_targets": ["bfs.cu"]},
    }))
    (specs_dir / "xsbench-xsbench-cuda.json").write_text(json.dumps({
        "identity": {"source_suite": "xsbench"},
        "files": {"translation_targets": ["xsbench.cu"]},
    }))

    records = [
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="BUILD_FAIL", augment_level=0),
        _make_record(source_spec="xsbench-xsbench-cuda", target_spec="xsbench-xsbench-omp",
                     overall_status="PASS", augment_level=0),
        # L2 record -- excluded from L0 comparison
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=2),
    ]

    result = qf.compute_cross_suite(records, tmp_path)

    # Per-suite pass rates at L0
    assert "rodinia" in result["per_suite_pass_rate_l0"]
    assert result["per_suite_pass_rate_l0"]["rodinia"]["value"] == 0.5
    assert result["per_suite_pass_rate_l0"]["rodinia"]["n"] == 2

    assert "xsbench" in result["per_suite_pass_rate_l0"]
    assert result["per_suite_pass_rate_l0"]["xsbench"]["value"] == 1.0
    assert result["per_suite_pass_rate_l0"]["xsbench"]["n"] == 1

    # SLoC characteristics loaded
    assert "sloc_characteristics" in result


# ---------------------------------------------------------------------------
# QUANT-11: Token cost analysis
# ---------------------------------------------------------------------------

def test_token_cost():
    """compute_token_cost sums tokens and computes cost with Together AI pricing."""
    records = [
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS"),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="BUILD_FAIL"),
    ]
    # Add token fields
    records[0]["prompt_tokens"] = 1_000_000
    records[0]["completion_tokens"] = 100_000
    records[1]["prompt_tokens"] = 500_000
    records[1]["completion_tokens"] = 50_000

    result = qf.compute_token_cost(records)

    # Total tokens
    assert result["total_input_tokens"]["value"] == 1_500_000
    assert result["total_output_tokens"]["value"] == 150_000

    # Total cost = 1.5M * 0.60/1M + 0.15M * 3.60/1M = 0.90 + 0.54 = 1.44
    expected_input_cost = 1_500_000 * 0.60 / 1_000_000  # 0.90
    expected_output_cost = 150_000 * 3.60 / 1_000_000    # 0.54
    expected_total = round(expected_input_cost + expected_output_cost, 4)
    assert result["total_cost"]["value"] == expected_total

    # Cost per task = 1.44 / 2
    assert result["cost_per_task"]["value"] == round(expected_total / 2, 4)

    # Cost per PASS = 1.44 / 1 (only 1 PASS record)
    assert result["cost_per_pass"]["value"] == round(expected_total / 1, 4)

    # 1 PASS total
    assert result["pass_count"] == 1
    assert result["tasks_with_tokens"] == 2

    # Per-suite breakdown
    assert "rodinia" in result["per_suite"]


# ---------------------------------------------------------------------------
# QUANT-12: SLoC correlation
# ---------------------------------------------------------------------------

def test_sloc_correlation(tmp_path):
    """compute_sloc_correlation computes Spearman and Pearson with rho/r and p_value."""
    # Create sloc_analysis.json with 4 kernels
    analysis_dir = tmp_path / "results" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "sloc_analysis.json").write_text(json.dumps({
        "kernels": {
            "bfs": {"physical_sloc": 100},
            "hotspot": {"physical_sloc": 200},
            "srad": {"physical_sloc": 300},
            "nw": {"physical_sloc": 400},
        }
    }))

    # Records at L0 with pass rates inversely correlated with SLoC
    records = [
        # bfs: 3/3 pass (SLoC 100)
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-bfs-omp", target_spec="rodinia-bfs-cuda",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-opencl",
                     overall_status="PASS", augment_level=0),
        # hotspot: 2/3 pass (SLoC 200)
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-hotspot-omp", target_spec="rodinia-hotspot-cuda",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-opencl",
                     overall_status="BUILD_FAIL", augment_level=0),
        # srad: 1/3 pass (SLoC 300)
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-omp",
                     overall_status="PASS", augment_level=0),
        _make_record(source_spec="rodinia-srad-omp", target_spec="rodinia-srad-cuda",
                     overall_status="BUILD_FAIL", augment_level=0),
        _make_record(source_spec="rodinia-srad-cuda", target_spec="rodinia-srad-opencl",
                     overall_status="BUILD_FAIL", augment_level=0),
        # nw: 0/3 pass (SLoC 400)
        _make_record(source_spec="rodinia-nw-cuda", target_spec="rodinia-nw-omp",
                     overall_status="BUILD_FAIL", augment_level=0),
        _make_record(source_spec="rodinia-nw-omp", target_spec="rodinia-nw-cuda",
                     overall_status="BUILD_FAIL", augment_level=0),
        _make_record(source_spec="rodinia-nw-cuda", target_spec="rodinia-nw-opencl",
                     overall_status="BUILD_FAIL", augment_level=0),
    ]

    result = qf.compute_sloc_correlation(records, tmp_path)

    # 4 kernels paired
    assert result["n_kernels"] == 4

    # Spearman result has rho and p_value fields
    assert "spearman" in result
    spearman = result["spearman"]["value"]
    assert "rho" in spearman
    assert "p_value" in spearman

    # Pearson result has r and p_value fields
    assert "pearson" in result
    pearson = result["pearson"]["value"]
    assert "r" in pearson
    assert "p_value" in pearson

    # Correlation should be negative (higher SLoC -> lower pass rate)
    assert spearman["rho"] < 0
    assert pearson["r"] < 0


# ---------------------------------------------------------------------------
# QUANT-13: OpenCL kernel-only effect
# ---------------------------------------------------------------------------

def test_opencl_kernel_only_effect():
    """compute_opencl_kernel_only_effect separates kernel-only vs full-program."""
    records = [
        # X-to-opencl (kernel-only): 3 records, 1 pass
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-opencl",
                     overall_status="PASS", augment_level=0,
                     direction="cuda-to-opencl"),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-opencl",
                     overall_status="BUILD_FAIL", augment_level=0,
                     direction="cuda-to-opencl"),
        _make_record(source_spec="rodinia-srad-omp", target_spec="rodinia-srad-opencl",
                     overall_status="BUILD_FAIL", augment_level=0,
                     direction="omp-to-opencl"),
        # X-to-omp (full program): 3 records, 2 pass
        _make_record(source_spec="rodinia-bfs-cuda", target_spec="rodinia-bfs-omp",
                     overall_status="PASS", augment_level=0,
                     direction="cuda-to-omp"),
        _make_record(source_spec="rodinia-hotspot-cuda", target_spec="rodinia-hotspot-omp",
                     overall_status="PASS", augment_level=0,
                     direction="cuda-to-omp"),
        _make_record(source_spec="rodinia-srad-opencl", target_spec="rodinia-srad-omp",
                     overall_status="BUILD_FAIL", augment_level=0,
                     direction="opencl-to-omp"),
        # L2 record -- should be excluded (L0 only)
        _make_record(source_spec="rodinia-nw-cuda", target_spec="rodinia-nw-opencl",
                     overall_status="PASS", augment_level=2,
                     direction="cuda-to-opencl"),
    ]

    result = qf.compute_opencl_kernel_only_effect(records)

    # X-to-opencl: 1/3 pass
    assert result["x_to_opencl"]["n"] == 3
    assert round(result["x_to_opencl"]["value"], 4) == round(1 / 3, 4)

    # X-to-omp: 2/3 pass
    assert result["x_to_omp"]["n"] == 3
    assert round(result["x_to_omp"]["value"], 4) == round(2 / 3, 4)

    # Fisher's exact test result
    assert result["test_result"]["method"] == "fisher_exact"
    assert "p_value" in result["test_result"]
    assert "odds_ratio" in result["test_result"]

    # CI bounds valid
    assert result["x_to_opencl"]["ci_lower"] <= result["x_to_opencl"]["value"]
    assert result["x_to_opencl"]["value"] <= result["x_to_opencl"]["ci_upper"]


# ---------------------------------------------------------------------------
# QUANT-SR: Self-repair removal verification
# ---------------------------------------------------------------------------

def test_no_compute_self_repair_function():
    """compute_self_repair should no longer exist after self-repair removal."""
    assert not hasattr(qf, "compute_self_repair"), \
        "compute_self_repair function should be deleted"


def test_no_self_repair_helpers():
    """Self-repair helper functions should no longer exist."""
    assert not hasattr(qf, "_classify_attempt_status"), \
        "_classify_attempt_status should be deleted"
    assert not hasattr(qf, "_classify_repair"), \
        "_classify_repair should be deleted"
    assert not hasattr(qf, "STATUS_ORDER"), \
        "STATUS_ORDER should be deleted"
