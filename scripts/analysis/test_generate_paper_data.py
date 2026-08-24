"""Tests for generate_paper_data.py — validates output against verified expected values.

All tests load the real paper_data.json produced by generate_paper_data.py.
Test 1 runs the script end-to-end via subprocess; all others validate the JSON structure.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DATA_PATH = PROJECT_ROOT / "results" / "analysis" / "paper_data.json"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analysis" / "generate_paper_data.py"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation" / "together-qwen-3.5-397b-a17b"


@pytest.fixture(scope="module")
def paper_data() -> dict:
    """Load the existing paper_data.json once for all tests."""
    assert PAPER_DATA_PATH.exists(), f"paper_data.json not found at {PAPER_DATA_PATH}"
    with open(PAPER_DATA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test 1: Script runs end-to-end and produces valid JSON
# ---------------------------------------------------------------------------

def test_script_runs_and_produces_json():
    """Run generate_paper_data.py via subprocess and verify it produces valid JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--results-dir", str(RESULTS_DIR),
                "--output", tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"Script exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[:2000]}"
        )

        # Verify output is valid JSON with expected top-level keys
        with open(tmp_path) as f:
            data = json.load(f)

        assert "primary_campaign" in data
        assert "passk_campaign" in data
        assert "file_counts" in data
        assert "excluded_specs" in data
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 2: Campaign separation — correct totals
# ---------------------------------------------------------------------------

def test_campaign_totals(paper_data):
    """Primary campaign = 710 tasks, pass@k campaign = 426 tasks (all suites)."""
    assert paper_data["file_counts"]["primary_campaign"] == 710
    assert paper_data["file_counts"]["passk_campaign"] == 426
    assert paper_data["primary_campaign"]["total"] == 710
    assert paper_data["passk_campaign"]["total"] == 426


# ---------------------------------------------------------------------------
# Test 3: Status breakdown sums to total
# ---------------------------------------------------------------------------

def test_status_breakdown_sums(paper_data):
    """Sum of all status counts must equal the total task count."""
    overall = paper_data["primary_campaign"]["overall"]
    status_sum = sum(overall["by_status"].values())
    assert status_sum == overall["total"], (
        f"Status sum {status_sum} != total {overall['total']}"
    )
    assert status_sum == 710


# ---------------------------------------------------------------------------
# Test 4: Individual status counts
# ---------------------------------------------------------------------------

def test_individual_status_counts(paper_data):
    """Each status category matches the independently verified count."""
    by_status = paper_data["primary_campaign"]["overall"]["by_status"]
    assert by_status["PASS"] == 272
    assert by_status["BUILD_FAIL"] == 241
    assert by_status["RUN_FAIL"] == 144
    assert by_status["VERIFY_FAIL"] == 51
    assert by_status["EXTRACTION_FAIL"] == 1
    assert by_status["ERROR"] == 1


# ---------------------------------------------------------------------------
# Test 5: Overall pass rate
# ---------------------------------------------------------------------------

def test_overall_pass_rate(paper_data):
    """Overall pass rate = 272/710 = 0.3831."""
    overall = paper_data["primary_campaign"]["overall"]
    assert overall["pass_rate"] == pytest.approx(0.3831, abs=0.0001)


# ---------------------------------------------------------------------------
# Test 6: Per-direction rates
# ---------------------------------------------------------------------------

def test_per_direction_rates(paper_data):
    """cuda-to-omp all-levels = 77/120; opencl-to-cuda L0 = 2/20."""
    by_dir = paper_data["primary_campaign"]["by_direction"]

    # cuda-to-omp: 77 PASS out of 120 total (all augmentation levels)
    cuda_to_omp = by_dir["cuda-to-omp"]
    assert cuda_to_omp["pass"] == 77
    assert cuda_to_omp["total"] == 120
    assert cuda_to_omp["pass_rate"] == pytest.approx(0.6417, abs=0.001)

    # opencl-to-cuda L0: check from augmentation per_direction_by_level
    aug = paper_data["primary_campaign"]["augmentation"]
    ocl_to_cuda_l0 = aug["per_direction_by_level"]["opencl-to-cuda"]["L0"]
    assert ocl_to_cuda_l0["pass"] == 2
    assert ocl_to_cuda_l0["total"] == 20
    assert ocl_to_cuda_l0["rate"] == pytest.approx(2 / 20, abs=0.001)


# ---------------------------------------------------------------------------
# Test 9: Cochran-Armitage trend test results
# ---------------------------------------------------------------------------

def test_cochran_armitage(paper_data):
    """z ~ -0.0, p ~ 1.0, significant=False."""
    ca = paper_data["primary_campaign"]["augmentation"]["cochran_armitage"]
    assert ca["z"] == pytest.approx(-0.0, abs=0.05)
    assert ca["p_value"] == pytest.approx(1.0, abs=0.05)
    assert ca["significant"] is False
    assert ca["trend_direction"] == "decreasing"


# ---------------------------------------------------------------------------
# Test 10: pass@k estimates
# ---------------------------------------------------------------------------

def test_passk_estimates(paper_data):
    """pass@1 macro avg ~ 0.197, pass@3 macro avg ~ 0.275 (all suites)."""
    agg = paper_data["passk_campaign"]["aggregate_passk"]
    assert agg["pass@1_macro_avg"] == pytest.approx(0.197, abs=0.01)
    assert agg["pass@3_macro_avg"] == pytest.approx(0.275, abs=0.01)


# ---------------------------------------------------------------------------
# Test 11: Wilson CIs present on all rate dicts
# ---------------------------------------------------------------------------

def test_wilson_cis_present(paper_data):
    """Every rate dict in primary_campaign overall and by_direction has CI keys."""
    overall = paper_data["primary_campaign"]["overall"]
    assert "ci_lower" in overall
    assert "ci_upper" in overall

    for direction, stats in paper_data["primary_campaign"]["by_direction"].items():
        assert "ci_lower" in stats, f"ci_lower missing in by_direction[{direction}]"
        assert "ci_upper" in stats, f"ci_upper missing in by_direction[{direction}]"

    for kernel, stats in paper_data["primary_campaign"]["by_kernel"].items():
        assert "ci_lower" in stats, f"ci_lower missing in by_kernel[{kernel}]"
        assert "ci_upper" in stats, f"ci_upper missing in by_kernel[{kernel}]"

    for level, stats in paper_data["primary_campaign"]["by_level"].items():
        assert "ci_lower" in stats, f"ci_lower missing in by_level[{level}]"
        assert "ci_upper" in stats, f"ci_upper missing in by_level[{level}]"


# ---------------------------------------------------------------------------
# Test 12: EXCLUDED_SPECS correctness
# ---------------------------------------------------------------------------

def test_excluded_specs(paper_data):
    """Exactly 6 specs excluded, all are known KNOWN_FAIL specs."""
    excluded = paper_data["excluded_specs"]
    assert len(excluded) == 6

    known_fail = {
        "rodinia-hybridsort-cuda",
        "rodinia-kmeans-cuda",
        "rodinia-kmeans-opencl",
        "rodinia-mummergpu-cuda",
        "rodinia-mummergpu-omp",
        "rodinia-nn-opencl",
    }
    assert set(excluded) == known_fail


# ---------------------------------------------------------------------------
# Test 13: Balanced subset for Cochran-Armitage
# ---------------------------------------------------------------------------

def test_balanced_subset_size(paper_data):
    """Cochran-Armitage uses 24 kernels, each with 5 levels = 120 total."""
    ca = paper_data["primary_campaign"]["augmentation"]["cochran_armitage"]
    assert ca["n_kernels"] == 24
    assert len(ca["levels"]) == 5
    assert len(ca["pass_counts"]) == 5
    assert len(ca["total_counts"]) == 5
    # Each level should have 24 tasks (one per kernel)
    for count in ca["total_counts"]:
        assert count == 24


# ---------------------------------------------------------------------------
# Test 14: No negative rates or CIs
# ---------------------------------------------------------------------------

def test_no_negative_rates(paper_data):
    """All rates and CI bounds must be in [0.0, 1.0]."""

    def check_rate_dict(d: dict, path: str):
        if "pass_rate" in d:
            assert 0.0 <= d["pass_rate"] <= 1.0, f"pass_rate out of range at {path}"
        if "rate" in d:
            assert 0.0 <= d["rate"] <= 1.0, f"rate out of range at {path}"
        if "ci_lower" in d:
            assert 0.0 <= d["ci_lower"] <= 1.0, f"ci_lower out of range at {path}"
        if "ci_upper" in d:
            assert 0.0 <= d["ci_upper"] <= 1.0, f"ci_upper out of range at {path}"
        if "ci_lower" in d and "ci_upper" in d:
            assert d["ci_lower"] <= d["ci_upper"], f"ci_lower > ci_upper at {path}"

    pc = paper_data["primary_campaign"]
    check_rate_dict(pc["overall"], "overall")

    for direction, stats in pc["by_direction"].items():
        check_rate_dict(stats, f"by_direction.{direction}")

    for kernel, stats in pc["by_kernel"].items():
        check_rate_dict(stats, f"by_kernel.{kernel}")

    for level, stats in pc["by_level"].items():
        check_rate_dict(stats, f"by_level.{level}")

    # Augmentation level rates
    for section_name in ["all_directions", "cuda_to_omp_balanced"]:
        section = pc["augmentation"][section_name]
        for level, stats in section.items():
            check_rate_dict(stats, f"augmentation.{section_name}.{level}")


# ---------------------------------------------------------------------------
# Test 15: L0 all-directions pass rate
# ---------------------------------------------------------------------------

def test_l0_all_directions_pass_rate(paper_data):
    """L0 all-directions: 57/142 = 0.4014..."""
    l0 = paper_data["primary_campaign"]["by_level"]["L0"]
    assert l0["total"] == 142
    assert l0["pass"] == 57
    assert l0["pass_rate"] == pytest.approx(57 / 142, abs=0.001)


# ---------------------------------------------------------------------------
# Test 16: Direction totals sum to campaign total
# ---------------------------------------------------------------------------

def test_direction_totals_sum(paper_data):
    """Sum of all direction totals must equal primary campaign total."""
    by_dir = paper_data["primary_campaign"]["by_direction"]
    direction_sum = sum(stats["total"] for stats in by_dir.values())
    assert direction_sum == 710


# ---------------------------------------------------------------------------
# Test 17: Level totals sum to campaign total
# ---------------------------------------------------------------------------

def test_level_totals_sum(paper_data):
    """Sum of all level totals must equal primary campaign total."""
    by_level = paper_data["primary_campaign"]["by_level"]
    level_sum = sum(stats["total"] for stats in by_level.values())
    assert level_sum == 710


# ---------------------------------------------------------------------------
# Test 18: Build fail subcategories sum to BUILD_FAIL total
# ---------------------------------------------------------------------------

def test_build_fail_subcategories_sum(paper_data):
    """Build fail subcategory counts must sum to total BUILD_FAIL."""
    bf = paper_data["primary_campaign"]["build_fail_subcategories"]
    sub_sum = sum(bf["subcategories"].values())
    assert sub_sum == bf["total"]
    assert bf["total"] == 241


# ---------------------------------------------------------------------------
# Test 19: Run fail subcategories sum to RUN_FAIL total
# ---------------------------------------------------------------------------

def test_run_fail_subcategories_sum(paper_data):
    """Run fail subcategory counts must sum to total RUN_FAIL."""
    rf = paper_data["primary_campaign"]["run_fail_subcategories"]
    sub_sum = sum(rf["subcategories"].values())
    assert sub_sum == rf["total"]
    assert rf["total"] == 144


# ---------------------------------------------------------------------------
# Test 20: Verify fail subcategories sum to VERIFY_FAIL total
# ---------------------------------------------------------------------------

def test_verify_fail_subcategories_sum(paper_data):
    """Verify fail subcategory counts must sum to total VERIFY_FAIL."""
    vf = paper_data["primary_campaign"]["verify_fail_subcategories"]
    sub_sum = sum(vf["subcategories"].values())
    assert sub_sum == vf["total"]
    assert vf["total"] == 51


# ---------------------------------------------------------------------------
# Test 21: pass@k individual estimates have valid structure
# ---------------------------------------------------------------------------

def test_passk_individual_estimates_structure(paper_data):
    """Each pass@k estimate entry has n, c, pass@1, pass@3 keys with valid values."""
    estimates = paper_data["passk_campaign"]["passk_estimates"]
    assert len(estimates) > 0

    for key, entry in estimates.items():
        assert ":" in key, f"pass@k key {key} missing kernel:direction separator"
        assert "n" in entry, f"Missing 'n' in {key}"
        assert "c" in entry, f"Missing 'c' in {key}"
        assert "pass@1" in entry, f"Missing 'pass@1' in {key}"
        assert "pass@3" in entry, f"Missing 'pass@3' in {key}"
        assert entry["c"] <= entry["n"], f"c > n in {key}"
        assert 0.0 <= entry["pass@1"] <= 1.0, f"pass@1 out of range in {key}"
        assert 0.0 <= entry["pass@3"] <= 1.0, f"pass@3 out of range in {key}"
        assert entry["pass@1"] <= entry["pass@3"] + 1e-9, (
            f"pass@1 > pass@3 in {key} (should be monotonically non-decreasing with k)"
        )


# ---------------------------------------------------------------------------
# Test 22: Direction asymmetry tests present and valid
# ---------------------------------------------------------------------------

def test_direction_asymmetry(paper_data):
    """Direction asymmetry section has 4 pairs with valid McNemar results."""
    asym = paper_data["primary_campaign"]["direction_asymmetry"]
    assert len(asym) == 4

    for pair_name, result in asym.items():
        assert "n_paired" in result
        assert "forward_pass_rate" in result
        assert "reverse_pass_rate" in result
        assert "p_value" in result
        assert "table" in result
        assert "significant" in result
        # Table should sum to n_paired
        t = result["table"]
        table_sum = t["both_pass"] + t["reverse_only"] + t["forward_only"] + t["both_fail"]
        assert table_sum == result["n_paired"], (
            f"Table sum {table_sum} != n_paired {result['n_paired']} for {pair_name}"
        )


# ---------------------------------------------------------------------------
# Test 23: Token metrics present and positive
# ---------------------------------------------------------------------------

def test_token_metrics(paper_data):
    """Token metrics contain expected keys with non-negative values."""
    tm = paper_data["primary_campaign"]["token_metrics"]
    for metric_type in ["prompt_tokens", "completion_tokens"]:
        m = tm[metric_type]
        assert m["mean"] > 0
        assert m["median"] > 0
        assert m["min"] >= 0
        assert m["max"] >= m["min"]
        assert m["total"] > 0
        assert m["p95"] >= m["median"]


# ---------------------------------------------------------------------------
# Test 24: Kernel totals sum to campaign total
# ---------------------------------------------------------------------------

def test_kernel_totals_sum(paper_data):
    """Sum of all kernel totals must equal primary campaign total."""
    by_kernel = paper_data["primary_campaign"]["by_kernel"]
    kernel_sum = sum(stats["total"] for stats in by_kernel.values())
    assert kernel_sum == 710


# ---------------------------------------------------------------------------
# Test 25: pass@k total samples matches campaign total
# ---------------------------------------------------------------------------

def test_passk_total_samples(paper_data):
    """pass@k aggregate total_samples matches passk campaign total (all suites)."""
    agg = paper_data["passk_campaign"]["aggregate_passk"]
    assert agg["total_samples"] == 426
    assert agg["n_tasks"] > 0


# ---------------------------------------------------------------------------
# Test 26–29: Helper extraction & module constants (tech-debt refactoring)
# ---------------------------------------------------------------------------

def _import_gpd():
    """Import generate_paper_data as a module (scripts/ has no __init__.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_paper_data", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_determine_first_attempt_status_pass():
    """Extracted helper returns PASS for a passing first attempt."""
    gpd = _import_gpd()
    att = {"build_status": "pass", "run_status": "pass", "verify_status": "pass"}
    assert gpd._determine_first_attempt_status(att) == "PASS"


def test_determine_first_attempt_status_build_fail():
    """Extracted helper returns BUILD_FAIL when build_status is 'fail'."""
    gpd = _import_gpd()
    att = {"build_status": "fail", "run_status": None, "verify_status": None}
    assert gpd._determine_first_attempt_status(att) == "BUILD_FAIL"


def test_constants_exist():
    """Module-level constants exist with correct values."""
    gpd = _import_gpd()
    assert gpd.MIN_L0_SAMPLE_SIZE == 10
    assert gpd.ALPHA_SIGNIFICANCE == 0.05
    assert gpd.COHEN_H_SMALL_THRESHOLD == 0.20


def test_passk_k_values():
    """PASSK_K_VALUES constant is [1, 3]."""
    gpd = _import_gpd()
    assert set(gpd.PASSK_K_VALUES) == {1, 3}
