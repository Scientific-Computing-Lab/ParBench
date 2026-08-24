#!/usr/bin/env python3
"""Tests for benchmark_characterization.py -- validates output against independently derived ground truth.

Tests verify each characterization metric by computing expected values directly
from raw data sources (manifest.jsonl, spec JSONs, source directories) and
comparing against the script's output. This catches bugs via independent computation.

Test organization by requirement:
  - Section 1:  End-to-end script execution
  - Section 1b: Independent raw-data ground truth (no benchmark_characterization.json needed)
  - Section 2:  CHAR-01 SLoC metrics
  - Section 3:  CHAR-02 Category distribution
  - Section 4:  CHAR-03 API coverage cross-tab
  - Section 5:  CHAR-04 Multi-file translation complexity
  - Section 6:  CHAR-05 Language feature tiers
  - Section 7:  CHAR-06 Language standard distribution
  - Section 8:  Cross-metric consistency
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analysis" / "benchmark_characterization.py"
OUTPUT_JSON = PROJECT_ROOT / "results" / "analysis" / "benchmark_characterization.json"
MANIFEST_PATH = PROJECT_ROOT / "manifest.jsonl"
SPECS_DIR = PROJECT_ROOT / "specs"

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — independent ground truth from raw data
# ---------------------------------------------------------------------------

def _load_valid_manifest_entries() -> list[dict]:
    """Load manifest.jsonl, filtering out phantom entries whose spec files don't exist."""
    entries = []
    for line in MANIFEST_PATH.read_text().strip().split("\n"):
        entry = json.loads(line)
        if Path(PROJECT_ROOT / entry["spec_file"]).exists():
            entries.append(entry)
    return entries


def _load_spec(spec_path: str) -> dict:
    """Load a spec JSON file relative to PROJECT_ROOT."""
    return json.loads((PROJECT_ROOT / spec_path).read_text())


def _get_corpus_kernels() -> list[tuple[str, str]]:
    """Import CORPUS_KERNELS from sloc_analysis.py."""
    from scripts.analysis.sloc_analysis import CORPUS_KERNELS  # noqa: E402
    return CORPUS_KERNELS


# ---------------------------------------------------------------------------
# Fixture — characterization output (skips gracefully if not generated yet)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def char_data() -> dict:
    """Load benchmark_characterization.json. Skip if not yet generated."""
    if not OUTPUT_JSON.exists():
        pytest.skip(
            "benchmark_characterization.json not yet generated -- "
            "run benchmark_characterization.py first"
        )
    return json.loads(OUTPUT_JSON.read_text())


# ---------------------------------------------------------------------------
# Section 1: End-to-end script execution
# ---------------------------------------------------------------------------

def test_script_runs_successfully(tmp_path):
    """Run benchmark_characterization.py via subprocess to a temp dir; assert exit 0 and valid JSON."""
    if not SCRIPT_PATH.exists():
        pytest.skip("benchmark_characterization.py not yet created -- plan 02-01 pending")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--project-root",
            str(PROJECT_ROOT),
            "--output-dir",
            str(tmp_path),
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
    tmp_json = tmp_path / "benchmark_characterization.json"
    assert tmp_json.exists(), "Output JSON not written after successful script run"
    # Verify valid JSON
    data = json.loads(tmp_json.read_text())
    assert isinstance(data, dict), "Output is not a JSON object"


# ---------------------------------------------------------------------------
# Section 1b: Independent raw-data ground truth tests
#   (NO dependency on benchmark_characterization.json — run even when it doesn't exist)
#   Category is NOT in spec JSONs — only in manifest.jsonl.
# ---------------------------------------------------------------------------

def test_manifest_has_12_categories():
    """manifest.jsonl (valid entries only) contains exactly 12 distinct categories."""
    valid = _load_valid_manifest_entries()
    categories = set(e["category"] for e in valid)
    expected = {
        "crypto", "financial", "graph", "image", "linear_algebra", "ml",
        "molecular_dynamics", "other", "physics", "reduction", "sort", "stencil",
    }
    assert categories == expected, (
        f"Category mismatch.\nExpected: {sorted(expected)}\nGot: {sorted(categories)}"
    )


def test_manifest_has_206_valid_entries():
    """manifest.jsonl has exactly 206 entries whose spec files exist on disk."""
    valid = _load_valid_manifest_entries()
    assert len(valid) == 206, f"Expected 206 valid entries, got {len(valid)}"


def test_manifest_has_211_total_entries():
    """manifest.jsonl has 211 total entries (206 valid + 5 phantom)."""
    all_entries = MANIFEST_PATH.read_text().strip().split("\n")
    assert len(all_entries) == 211, f"Expected 211 total entries, got {len(all_entries)}"


def test_corpus_kernels_all_have_cuda_specs():
    """Every one of the 35 CORPUS_KERNELS has a CUDA spec file on disk."""
    corpus = _get_corpus_kernels()
    assert len(corpus) == 35, f"Expected 35 corpus kernels, got {len(corpus)}"
    for suite, kernel in corpus:
        spec_path = SPECS_DIR / f"{suite}-{kernel}-cuda.json"
        assert spec_path.exists(), f"Missing CUDA spec: {spec_path.name}"


def test_hecbench_omp_spec_availability():
    """5 HeCBench curated kernels have omp_target but NOT omp specs.

    This documents the known gap affecting SLoC OMP ratio calculation.
    """
    omp_missing = ["convolution1d", "jacobi", "md", "nqueen", "page-rank"]
    for kernel in omp_missing:
        omp_path = SPECS_DIR / f"hecbench-{kernel}-omp.json"
        omp_target_path = SPECS_DIR / f"hecbench-{kernel}-omp_target.json"
        assert not omp_path.exists(), f"Expected NO OMP spec for {kernel}, but found {omp_path.name}"
        assert omp_target_path.exists(), f"Expected omp_target spec for {kernel}, but missing"


def test_specs_dir_has_206_files():
    """specs/ directory contains exactly 206 JSON files."""
    spec_count = len(list(SPECS_DIR.glob("*.json")))
    assert spec_count == 206, f"Expected 206 spec files, got {spec_count}"


def test_manifest_has_4_apis():
    """manifest.jsonl valid entries span exactly 4 parallel APIs."""
    valid = _load_valid_manifest_entries()
    apis = sorted(set(e["parallel_api"] for e in valid))
    assert apis == ["cuda", "omp", "omp_target", "opencl"], f"APIs: {apis}"


def test_manifest_has_5_suites():
    """manifest.jsonl valid entries span exactly 5 source suites."""
    valid = _load_valid_manifest_entries()
    suites = sorted(set(e["source_suite"] for e in valid))
    assert suites == ["hecbench", "mixbench", "rodinia", "rsbench", "xsbench"], (
        f"Suites: {suites}"
    )


# ---------------------------------------------------------------------------
# Section 2: CHAR-01 SLoC metrics
# ---------------------------------------------------------------------------

def test_sloc_covers_all_35_kernels(char_data):
    """SLoC per_kernel must have exactly 35 entries (one per corpus kernel)."""
    assert len(char_data["sloc"]["per_kernel"]) == 35


def test_sloc_has_required_fields(char_data):
    """Each kernel in sloc.per_kernel has required keys and positive SLoC."""
    required_keys = {"kernel", "suite", "cuda_sloc", "num_source_files", "num_target_files"}
    for name, info in char_data["sloc"]["per_kernel"].items():
        missing = required_keys - set(info.keys())
        assert not missing, f"Kernel '{name}' missing keys: {missing}"
        assert info["cuda_sloc"] > 0, f"Kernel '{name}' has zero SLoC"


def test_sloc_summary_statistics(char_data):
    """Summary statistics include min/max/mean/median with sensible ranges."""
    summary = char_data["sloc"]["summary"]
    for key in ("min_sloc", "max_sloc", "mean_sloc", "median_sloc"):
        assert key in summary, f"Missing summary key: {key}"
    assert summary["min_sloc"] > 0, "min_sloc must be positive"
    assert summary["max_sloc"] > summary["min_sloc"], "max_sloc must exceed min_sloc"
    assert summary["min_sloc"] <= summary["mean_sloc"] <= summary["max_sloc"]
    assert summary["min_sloc"] <= summary["median_sloc"] <= summary["max_sloc"]


def test_sloc_omp_ratio_present(char_data):
    """Kernels with OMP specs have positive omp_cuda_ratio; those without have null."""
    # 9 kernels lack OMP specs: 5 HeCBench (omp_target only) + 3 Rodinia phantoms + dwt2d (no omp variant)
    no_omp = {"convolution1d", "jacobi", "md", "nqueen", "page-rank", "dwt2d", "gaussian", "huffman", "hybridsort"}
    for name, info in char_data["sloc"]["per_kernel"].items():
        if name in no_omp:
            assert info.get("omp_cuda_ratio") is None, (
                f"Kernel '{name}' should have null omp_cuda_ratio (no OMP spec)"
            )
        else:
            ratio = info.get("omp_cuda_ratio")
            assert ratio is not None and ratio > 0, (
                f"Kernel '{name}' should have positive omp_cuda_ratio, got {ratio}"
            )


def test_sloc_kernel_names_match_corpus(char_data):
    """Set of kernel names in sloc output matches CORPUS_KERNELS names."""
    corpus = _get_corpus_kernels()
    expected_names = {k for _, k in corpus}
    actual_names = set(char_data["sloc"]["per_kernel"].keys())
    assert actual_names == expected_names, (
        f"Mismatch.\nMissing: {expected_names - actual_names}\n"
        f"Extra: {actual_names - expected_names}"
    )


# ---------------------------------------------------------------------------
# Section 3: CHAR-02 Category distribution
# ---------------------------------------------------------------------------

def test_categories_has_12_entries(char_data):
    """categories section has exactly 12 entries."""
    assert len(char_data["categories"]) == 12


def test_category_names_correct(char_data):
    """Category keys match the expected 12 names."""
    expected = {
        "crypto", "financial", "graph", "image", "linear_algebra", "ml",
        "molecular_dynamics", "other", "physics", "reduction", "sort", "stencil",
    }
    actual = set(char_data["categories"].keys())
    assert actual == expected, (
        f"Category name mismatch.\nExpected: {sorted(expected)}\nGot: {sorted(actual)}"
    )


def test_categories_have_suite_annotations(char_data):
    """Each category has 'suites' dict and positive 'kernel_count'."""
    for cat_name, cat_data in char_data["categories"].items():
        assert "suites" in cat_data, f"Category '{cat_name}' missing 'suites' key"
        assert isinstance(cat_data["suites"], dict), (
            f"Category '{cat_name}' suites is not a dict"
        )
        assert "kernel_count" in cat_data, f"Category '{cat_name}' missing 'kernel_count'"
        assert cat_data["kernel_count"] > 0, f"Category '{cat_name}' has zero kernel_count"


def test_category_counts_match_manifest(char_data):
    """Category kernel counts match independently computed values from manifest.jsonl.

    Uses manifest.jsonl as ground truth. Counts distinct (suite, kernel_name) pairs
    per category, deduplicating entries where the same kernel has multiple APIs.
    """
    valid = _load_valid_manifest_entries()
    # Build category -> set of distinct kernel_names (deduplicate across APIs and suites)
    cat_kernels: dict[str, set[str]] = defaultdict(set)
    for e in valid:
        cat_kernels[e["category"]].add(e["kernel_name"])
    expected_counts = {cat: len(kernels) for cat, kernels in cat_kernels.items()}

    for cat_name, cat_data in char_data["categories"].items():
        expected = expected_counts.get(cat_name, 0)
        actual = cat_data["kernel_count"]
        assert actual == expected, (
            f"Category '{cat_name}': expected {expected} kernels, got {actual}"
        )


# ---------------------------------------------------------------------------
# Section 4: CHAR-03 API coverage cross-tab
# ---------------------------------------------------------------------------

def test_api_coverage_has_5_suites(char_data):
    """API coverage section covers all 5 suites."""
    assert len(char_data["api_coverage"]["suites"]) == 5


def test_api_coverage_has_4_apis(char_data):
    """Each suite in api_coverage has entries for cuda, omp, opencl, omp_target."""
    expected_apis = {"cuda", "omp", "opencl", "omp_target"}
    for suite_name, suite_data in char_data["api_coverage"]["suites"].items():
        actual_apis = set(suite_data.keys())
        assert expected_apis.issubset(actual_apis), (
            f"Suite '{suite_name}' missing APIs: {expected_apis - actual_apis}"
        )


def test_api_coverage_matches_specs(char_data):
    """API coverage counts match independently computed spec counts from manifest.

    Counts distinct kernel_names per (suite, api) from manifest.jsonl (valid entries).
    """
    valid = _load_valid_manifest_entries()
    # Count distinct kernels per (suite, api)
    expected: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: dict[str, set[str]] = defaultdict(set)
    for e in valid:
        key = (e["source_suite"], e["parallel_api"], e["kernel_name"])
        combo = f"{e['source_suite']}:{e['parallel_api']}:{e['kernel_name']}"
        if combo not in seen[e["source_suite"]]:
            seen[e["source_suite"]].add(combo)
            expected[e["source_suite"]][e["parallel_api"]] += 1

    for suite_name, suite_data in char_data["api_coverage"]["suites"].items():
        for api in ("cuda", "omp", "opencl", "omp_target"):
            exp = expected.get(suite_name, {}).get(api, 0)
            act = suite_data.get(api, 0)
            assert act == exp, (
                f"API coverage {suite_name}/{api}: expected {exp}, got {act}"
            )


def test_api_coverage_totals_consistent(char_data):
    """Totals row equals column sums across all suites."""
    suites = char_data["api_coverage"]["suites"]
    totals = char_data["api_coverage"]["totals"]
    for api in ("cuda", "omp", "opencl", "omp_target"):
        col_sum = sum(s.get(api, 0) for s in suites.values())
        assert totals[api] == col_sum, (
            f"Total for '{api}': expected {col_sum}, got {totals[api]}"
        )


# ---------------------------------------------------------------------------
# Section 5: CHAR-04 Multi-file translation complexity
# ---------------------------------------------------------------------------

def test_multi_file_covers_all_kernels(char_data):
    """multi_file.by_kernel has 35 entries (one per corpus kernel)."""
    assert len(char_data["multi_file"]["by_kernel"]) == 35


def test_multi_file_has_both_counts(char_data):
    """Each kernel entry has headline payload and target counts."""
    for name, info in char_data["multi_file"]["by_kernel"].items():
        assert "headline_payload_count" in info, (
            f"Kernel '{name}' missing headline_payload_count"
        )
        assert "headline_target_count" in info, (
            f"Kernel '{name}' missing headline_target_count"
        )


def test_multi_file_classification_uses_targets(char_data):
    """headline_multi_file flag equals (headline_target_count > 1) per D-02."""
    for name, info in char_data["multi_file"]["by_kernel"].items():
        expected = info["headline_target_count"] > 1
        actual = info["headline_multi_file"]
        assert actual == expected, (
            f"Kernel '{name}': headline_multi_file={actual} but "
            f"headline_target_count={info['headline_target_count']}"
        )


def test_multi_file_aggregate_sums(char_data):
    """single_file_count + multi_file_count == 35."""
    agg = char_data["multi_file"]["aggregate"]
    total = agg["single_file_count"] + agg["multi_file_count"]
    assert total == 35, f"Aggregate: {agg['single_file_count']} + {agg['multi_file_count']} = {total}, expected 35"


def test_multi_file_matches_specs():
    """Independently verify multi-file counts for 3 known kernels against spec files.

    bfs: 3 translation_targets, nn: 1 target, xsbench: 2 targets.
    """
    test_cases = [
        ("specs/rodinia-bfs-cuda.json", 3),
        ("specs/rodinia-nn-cuda.json", 1),
        ("specs/xsbench-xsbench-cuda.json", 2),
    ]
    for spec_path, expected_count in test_cases:
        spec = _load_spec(spec_path)
        targets = spec["files"]["translation_targets"]
        assert len(targets) == expected_count, (
            f"{spec_path}: expected {expected_count} targets, got {len(targets)}"
        )


# ---------------------------------------------------------------------------
# Section 6: CHAR-05 Language feature tiers
# ---------------------------------------------------------------------------

def test_language_features_has_entries(char_data):
    """language_features.per_kernel has at least some entries."""
    assert "per_kernel" in char_data["language_features"]
    assert len(char_data["language_features"]["per_kernel"]) > 0


VALID_TIERS = {
    "cuda_basic", "cuda_library", "cuda_9plus",
    "omp_basic", "omp_4.5", "omp_target",
    "opencl_1x", "opencl_2x",
    "undetected",
}


def test_language_feature_tiers_valid(char_data):
    """Every kernel with features has an overall_tier from the expected set."""
    for name, info in char_data["language_features"]["per_kernel"].items():
        tier = info.get("overall_tier")
        assert tier in VALID_TIERS, (
            f"Kernel '{name}' has invalid overall_tier '{tier}'. "
            f"Expected one of: {sorted(VALID_TIERS)}"
        )


def test_bfs_has_cuda_basic_features(char_data):
    """BFS (CUDA) uses __global__ so its tier should be at least cuda_basic."""
    bfs = char_data["language_features"]["per_kernel"].get("bfs")
    if bfs is None:
        pytest.skip("bfs not in language_features per_kernel")
    cuda_tiers = {"cuda_basic", "cuda_library", "cuda_9plus"}
    bfs_cuda_tier = bfs.get("apis", {}).get("cuda", {}).get("tier")
    assert bfs_cuda_tier in cuda_tiers, (
        f"Expected bfs CUDA tier in {cuda_tiers}, got '{bfs_cuda_tier}'"
    )


def test_language_features_skip_missing_dirs(char_data):
    """Kernels with missing source dirs should not crash; check total count is reasonable."""
    per_kernel = char_data["language_features"]["per_kernel"]
    # At least some kernels should have features detected (features are under apis.{api}.features_found)
    with_features = [
        k for k, v in per_kernel.items()
        if any(
            api_data.get("features_found")
            for api_data in v.get("apis", {}).values()
        )
    ]
    assert len(with_features) > 0, "No kernels have detected features"


# ---------------------------------------------------------------------------
# Section 7: CHAR-06 Language standard distribution
# ---------------------------------------------------------------------------

def test_language_standards_distribution_sums_to_206(char_data):
    """Sum of all language_standard counts equals 206 (total spec count)."""
    dist = char_data["language_standards"]["distribution"]
    total = sum(dist.values())
    assert total == 206, f"Language standard distribution sums to {total}, expected 206"


def test_language_standards_known_values(char_data):
    """Known language standard counts match independently verified values from spec files."""
    dist = char_data["language_standards"]["distribution"]
    # Independently verified from spec files
    assert dist.get("C++17") == 128, f"C++17: expected 128, got {dist.get('C++17')}"
    assert dist.get("C++14") == 51, f"C++14: expected 51, got {dist.get('C++14')}"
    assert dist.get("C11") == 6, f"C11: expected 6, got {dist.get('C11')}"
    assert dist.get("C++11") == 2, f"C++11: expected 2, got {dist.get('C++11')}"
    # The remaining 19 are unspecified (None in spec -> some key like "unspecified" or "null")
    remaining = 206 - 128 - 51 - 6 - 2
    assert remaining == 19
    # Check that some key accounts for the 19 unspecified
    known_sum = 128 + 51 + 6 + 2
    unspecified_sum = sum(v for k, v in dist.items() if k not in ("C++17", "C++14", "C11", "C++11"))
    assert unspecified_sum == 19, (
        f"Unspecified standards sum: expected 19, got {unspecified_sum}"
    )


def test_language_standards_matches_specs():
    """Independently verify language standard distribution from spec files on disk."""
    standards: Counter[str] = Counter()
    for spec_file in sorted(SPECS_DIR.glob("*.json")):
        spec = json.loads(spec_file.read_text())
        std = (spec.get("implementation") or {}).get("language_standard")
        standards[std] += 1

    assert standards["C++17"] == 128
    assert standards["C++14"] == 51
    assert standards["C11"] == 6
    assert standards["C++11"] == 2
    assert standards[None] == 19
    assert sum(standards.values()) == 206


def test_language_standards_by_api_exists(char_data):
    """language_standards has a by_api breakdown."""
    assert "by_api" in char_data["language_standards"], "Missing 'by_api' in language_standards"
    by_api = char_data["language_standards"]["by_api"]
    assert len(by_api) > 0, "by_api is empty"


# ---------------------------------------------------------------------------
# Section 8: Cross-metric consistency and metadata
# ---------------------------------------------------------------------------

def test_total_kernels_consistent(char_data):
    """SLoC kernel count == multi_file kernel count == 35."""
    sloc_count = len(char_data["sloc"]["per_kernel"])
    mf_count = len(char_data["multi_file"]["by_kernel"])
    assert sloc_count == 35, f"SLoC kernel count: {sloc_count}"
    assert mf_count == 35, f"Multi-file kernel count: {mf_count}"


def test_metadata_present(char_data):
    """Output has metadata with 'generated' and 'script' keys."""
    assert "metadata" in char_data
    assert "generated" in char_data["metadata"]
    assert "script" in char_data["metadata"]


def test_md_output_exists():
    """Markdown output file exists alongside the JSON."""
    md_path = PROJECT_ROOT / "results" / "analysis" / "benchmark_characterization.md"
    if not OUTPUT_JSON.exists():
        pytest.skip("benchmark_characterization.json not yet generated")
    assert md_path.exists(), f"Markdown output not found at {md_path}"


def test_summary_section_present(char_data):
    """Output has a summary section with total_kernels and total_specs."""
    assert "summary" in char_data
    summary = char_data["summary"]
    assert summary.get("total_kernels") == 35, (
        f"summary.total_kernels: expected 35, got {summary.get('total_kernels')}"
    )
    assert summary.get("total_specs") == 206, (
        f"summary.total_specs: expected 206, got {summary.get('total_specs')}"
    )
