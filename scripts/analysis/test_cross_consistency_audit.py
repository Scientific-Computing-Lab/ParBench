#!/usr/bin/env python3
"""Tests for cross_consistency_audit.py.

TDD RED phase: These tests define the expected behavior of the audit script.
"""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tex_content():
    """Minimal paper.tex-like content with known numbers."""
    return textwrap.dedent(r"""
    \begin{abstract}
    % src: paper_data.json > primary_campaign > overall
    We observe an overall pass rate of 38.3\% [34.8\%, 41.9\%].
    BUILD\_FAIL accounts for 33.9\% of all tasks.
    VERIFY\_FAIL accounts for 7.2\%.
    Qwen achieves 64.2\% (66.7\% at L0).
    \end{abstract}
    % This is a pure comment with 99.9\%
    \section{Results}
    The campaign spans 710 tasks.
    \cite{paper2024} reports 80\% accuracy.
    LASSI reports 80--85\% pass rates.
    """)


@pytest.fixture
def sample_tex_path(tmp_path, sample_tex_content):
    """Write sample tex content to a temp file."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(sample_tex_content)
    return tex_file


@pytest.fixture
def sample_paper_data(tmp_path):
    """Create a minimal paper_data.json."""
    data = {
        "primary_campaign": {
            "total": 710,
            "overall": {
                "total": 710,
                "pass": 272,
                "pass_rate": 0.3831,
                "ci_lower": 0.3481,
                "ci_upper": 0.4194,
                "by_status": {
                    "BUILD_FAIL": 241,
                    "VERIFY_FAIL": 51,
                    "PASS": 272,
                    "RUN_FAIL": 144,
                    "EXTRACTION_FAIL": 1,
                    "ERROR": 1,
                },
            },
            "by_direction": {
                "cuda-to-omp": {
                    "total": 120,
                    "pass": 77,
                    "pass_rate": 0.6417,
                    "ci_lower": 0.5527,
                    "ci_upper": 0.7218,
                },
            },
            "by_kernel": {},
            "augmentation": {
                "cuda_to_omp_balanced": {
                    "L0": {"rate": 0.6667, "n": 24},
                },
                "cochran_armitage": {"z": -0.0, "p_value": 1.0, "n_kernels": 24},
            },
            "build_fail_subcategories": {
                "total": 241,
                "subcategories": {
                    "undeclared_identifier": 56,
                    "missing_header": 55,
                    "linker_error": 49,
                },
            },
            "verify_fail_subcategories": {
                "total": 51,
                "subcategories": {
                    "wrong_numerical_output": 46,
                    "missing_output": 5,
                },
            },
            "run_fail_subcategories": {
                "total": 144,
                "subcategories": {
                    "opencl_jit_error": 52,
                },
            },
        },
        "file_counts": {
            "primary_campaign": 710,
            "passk_campaign": 426,
        },
    }
    results_dir = tmp_path / "results" / "analysis"
    results_dir.mkdir(parents=True)
    (results_dir / "paper_data.json").write_text(json.dumps(data))
    return data


@pytest.fixture
def sample_qf(tmp_path):
    """Create a minimal quantitative_findings.json."""
    data = {
        "paper_claims": [
            {
                "claim_id": "overall_pass_rate_all_suite",
                "paper_location": "abstract",
                "json_path": "canonical.aggregate_pass_rates.overall.value",
                "scope": "all_suite",
                "current_value": 0.38,
                "display_value": "38.0%",
            },
        ],
    }
    results_dir = tmp_path / "results" / "analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "quantitative_findings.json").write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Test 1: extract_numbers_from_tex finds XX.X% patterns
# ---------------------------------------------------------------------------


def test_extract_percentages(sample_tex_path):
    """extract_numbers_from_tex finds all XX.X% patterns in a sample tex string."""
    from scripts.analysis.cross_consistency_audit import extract_numbers_from_tex

    results = extract_numbers_from_tex(sample_tex_path)
    pct_values = [r["value"] for r in results if r["type"] == "percentage"]
    # Should find: 38.3, 33.9, 7.2, 64.2, 66.7
    assert 38.3 in pct_values
    assert 33.9 in pct_values
    assert 7.2 in pct_values
    assert 64.2 in pct_values
    assert 66.7 in pct_values


# ---------------------------------------------------------------------------
# Test 2: extract_numbers_from_tex finds CI patterns
# ---------------------------------------------------------------------------


def test_extract_ci_patterns(sample_tex_path):
    """extract_numbers_from_tex finds CI patterns like [XX.X%, YY.Y%]."""
    from scripts.analysis.cross_consistency_audit import extract_numbers_from_tex

    results = extract_numbers_from_tex(sample_tex_path)
    ci_values = [r["value"] for r in results if r["type"] == "ci"]
    # Should find: 34.8, 41.9
    assert 34.8 in ci_values
    assert 41.9 in ci_values


# ---------------------------------------------------------------------------
# Test 3: load_ground_truth loads both JSON files
# ---------------------------------------------------------------------------


def test_load_ground_truth(tmp_path, sample_paper_data, sample_qf):
    """load_ground_truth loads paper_data.json and quantitative_findings.json."""
    from scripts.analysis.cross_consistency_audit import load_ground_truth

    gt = load_ground_truth(tmp_path)
    assert "paper_data" in gt
    assert "quantitative_findings" in gt
    assert gt["paper_data"]["primary_campaign"]["overall"]["pass"] == 272


# ---------------------------------------------------------------------------
# Test 4: match_claims correctly matches 38.3% to pass_rate
# ---------------------------------------------------------------------------


def test_match_claims_verified(tmp_path, sample_paper_data, sample_qf):
    """match_claims correctly matches '38.3%' to paper_data.json overall pass_rate."""
    from scripts.analysis.cross_consistency_audit import (
        build_known_values,
        build_whitelist,
        match_claims,
    )

    extracted = [
        {"line": 4, "raw": "38.3\\%", "value": 38.3, "type": "percentage", "context": "pass rate of 38.3%"},
    ]
    known = build_known_values({"paper_data": sample_paper_data})
    whitelist = build_whitelist()
    paper_claims = sample_qf.get("paper_claims", [])
    verified, unverified = match_claims(extracted, known, paper_claims, whitelist)
    assert len(verified) == 1
    assert verified[0]["status"] == "verified"
    assert len(unverified) == 0


# ---------------------------------------------------------------------------
# Test 5: match_claims flags mismatch when tex has wrong number
# ---------------------------------------------------------------------------


def test_match_claims_flags_mismatch(tmp_path, sample_paper_data, sample_qf):
    """match_claims flags a mismatch when tex contains wrong percentage."""
    from scripts.analysis.cross_consistency_audit import (
        build_known_values,
        build_whitelist,
        match_claims,
    )

    extracted = [
        {"line": 4, "raw": "40.0\\%", "value": 40.0, "type": "percentage", "context": "pass rate of 40.0%"},
    ]
    known = build_known_values({"paper_data": sample_paper_data})
    whitelist = build_whitelist()
    paper_claims = sample_qf.get("paper_claims", [])
    verified, unverified = match_claims(extracted, known, paper_claims, whitelist)
    # 40.0 should NOT match any known value (overall pass rate is 38.3%)
    assert len(unverified) == 1
    assert unverified[0]["status"] == "unverified"


# ---------------------------------------------------------------------------
# Test 6: Whitelisted numbers not flagged
# ---------------------------------------------------------------------------


def test_whitelisted_not_flagged(tmp_path, sample_paper_data, sample_qf):
    """Whitelisted numbers (external citations, params) are not flagged."""
    from scripts.analysis.cross_consistency_audit import (
        build_known_values,
        build_whitelist,
        match_claims,
    )

    extracted = [
        {"line": 10, "raw": "80\\%", "value": 80.0, "type": "percentage", "context": "LASSI reports 80%"},
        {"line": 11, "raw": "85\\%", "value": 85.0, "type": "percentage", "context": "LASSI reports 85%"},
    ]
    known = build_known_values({"paper_data": sample_paper_data})
    whitelist = build_whitelist()
    paper_claims = sample_qf.get("paper_claims", [])
    verified, unverified = match_claims(extracted, known, paper_claims, whitelist)
    # Whitelisted values should not appear in unverified
    assert len(unverified) == 0


# ---------------------------------------------------------------------------
# Test 7: Script exit codes
# ---------------------------------------------------------------------------


def test_script_exit_0_when_all_verified(tmp_path, sample_paper_data, sample_qf, sample_tex_path):
    """Script exits 0 when all critical claims match."""
    # This is an integration test — we check the main function logic
    # by verifying that verified claims produce exit 0
    from scripts.analysis.cross_consistency_audit import (
        build_known_values,
        build_whitelist,
        match_claims,
    )

    extracted = [
        {"line": 4, "raw": "38.3\\%", "value": 38.3, "type": "percentage", "context": "38.3%"},
    ]
    known = build_known_values({"paper_data": sample_paper_data})
    whitelist = build_whitelist()
    paper_claims = sample_qf.get("paper_claims", [])
    verified, unverified = match_claims(extracted, known, paper_claims, whitelist)
    critical_unverified = [u for u in unverified if u["type"] == "percentage"]
    # No critical unverified = would exit 0
    assert len(critical_unverified) == 0


def test_script_exit_1_when_mismatch(tmp_path, sample_paper_data, sample_qf):
    """Script exits 1 when mismatches found."""
    from scripts.analysis.cross_consistency_audit import (
        build_known_values,
        build_whitelist,
        match_claims,
    )

    extracted = [
        {"line": 4, "raw": "99.9\\%", "value": 99.9, "type": "percentage", "context": "99.9%"},
    ]
    known = build_known_values({"paper_data": sample_paper_data})
    whitelist = build_whitelist()
    paper_claims = sample_qf.get("paper_claims", [])
    verified, unverified = match_claims(extracted, known, paper_claims, whitelist)
    critical_unverified = [u for u in unverified if u["type"] == "percentage"]
    # 99.9% should be unverified and critical
    assert len(critical_unverified) == 1


# ---------------------------------------------------------------------------
# Test 8: check_provenance_comments detects broken JSON path references
# ---------------------------------------------------------------------------


def test_provenance_detects_broken_path(tmp_path, sample_paper_data, sample_qf):
    """check_provenance_comments flags a % src: reference to a nonexistent key."""
    from scripts.analysis.cross_consistency_audit import (
        check_provenance_comments,
        load_ground_truth,
    )

    tex_content = textwrap.dedent(r"""
    % src: paper_data.json > nonexistent_key > something
    Some text here 38.3\%.
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    gt = load_ground_truth(tmp_path)
    broken = check_provenance_comments(tex_file, gt)
    assert len(broken) == 1
    assert broken[0]["target_file"] == "paper_data"


# ---------------------------------------------------------------------------
# Test 9: check_provenance_comments resolves valid paths
# ---------------------------------------------------------------------------


def test_provenance_resolves_valid_path(tmp_path, sample_paper_data, sample_qf):
    """check_provenance_comments does NOT flag valid JSON path references."""
    from scripts.analysis.cross_consistency_audit import (
        check_provenance_comments,
        load_ground_truth,
    )

    tex_content = textwrap.dedent(r"""
    % src: paper_data.json > primary_campaign > overall
    Some text here 38.3\%.
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    gt = load_ground_truth(tmp_path)
    broken = check_provenance_comments(tex_file, gt)
    assert len(broken) == 0


# ---------------------------------------------------------------------------
# Test 10: check_provenance_comments skips non-JSON references
# ---------------------------------------------------------------------------


def test_provenance_skips_non_json(tmp_path, sample_paper_data, sample_qf):
    """check_provenance_comments ignores % src: lines without JSON filenames."""
    from scripts.analysis.cross_consistency_audit import (
        check_provenance_comments,
        load_ground_truth,
    )

    tex_content = textwrap.dedent(r"""
    % src: computed from 272/710
    Some text here.
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    gt = load_ground_truth(tmp_path)
    broken = check_provenance_comments(tex_file, gt)
    assert len(broken) == 0


# ---------------------------------------------------------------------------
# Test 11: extract_numbers_from_tex skips verbatim environments
# ---------------------------------------------------------------------------


def test_extract_skips_verbatim(tmp_path):
    """Numbers inside verbatim/lstlisting environments are not extracted."""
    from scripts.analysis.cross_consistency_audit import extract_numbers_from_tex

    tex_content = textwrap.dedent(r"""
    Real claim: 38.3\% pass rate.
    \begin{verbatim}
    This 99.9\% should be ignored.
    \end{verbatim}
    \begin{lstlisting}
    Also 77.7\% ignored.
    \end{lstlisting}
    Another real claim: 64.2\% direction rate.
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    results = extract_numbers_from_tex(tex_file)
    pct_values = [r["value"] for r in results if r["type"] == "percentage"]
    assert 38.3 in pct_values
    assert 64.2 in pct_values
    assert 99.9 not in pct_values
    assert 77.7 not in pct_values


# ---------------------------------------------------------------------------
# Test 12: extract_numbers_from_tex finds integer counts
# ---------------------------------------------------------------------------


def test_extract_counts(tmp_path):
    """extract_numbers_from_tex finds integer counts near data keywords."""
    from scripts.analysis.cross_consistency_audit import extract_numbers_from_tex

    tex_content = textwrap.dedent(r"""
    The campaign spans 710 tasks across 24 kernels.
    BUILD\_FAIL accounts for 241 failures.
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    results = extract_numbers_from_tex(tex_file)
    count_values = [r["value"] for r in results if r["type"] == "count"]
    assert 710 in count_values
    assert 24 in count_values
    assert 241 in count_values


# ---------------------------------------------------------------------------
# Test 13: CI values are NOT double-counted as percentages
# ---------------------------------------------------------------------------


def test_ci_not_double_counted(tmp_path):
    """CI bracket values are extracted as ci type only, not also as percentage."""
    from scripts.analysis.cross_consistency_audit import extract_numbers_from_tex

    tex_content = textwrap.dedent(r"""
    Overall pass rate is 38.3\% [34.8\%, 41.9\%].
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    results = extract_numbers_from_tex(tex_file)
    ci_values = [r for r in results if r["type"] == "ci"]
    pct_values = [r for r in results if r["type"] == "percentage"]
    ci_nums = {r["value"] for r in ci_values}
    pct_nums = {r["value"] for r in pct_values}
    assert 34.8 in ci_nums
    assert 41.9 in ci_nums
    # CI values should NOT also appear as percentages
    assert 34.8 not in pct_nums
    assert 41.9 not in pct_nums
    # But 38.3 (standalone percentage) should still be extracted
    assert 38.3 in pct_nums


# ---------------------------------------------------------------------------
# Test 14: check_provenance_comments uses primary_campaign fallback
# ---------------------------------------------------------------------------


def test_provenance_primary_campaign_fallback(tmp_path, sample_paper_data, sample_qf):
    """check_provenance_comments resolves paths via primary_campaign fallback."""
    from scripts.analysis.cross_consistency_audit import (
        check_provenance_comments,
        load_ground_truth,
    )

    # "overall" is inside primary_campaign, not at top level of paper_data.json
    # The fallback logic should find it
    tex_content = textwrap.dedent(r"""
    % src: paper_data.json > overall > by_status
    BUILD\_FAIL is 241.
    """)
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(tex_content)
    gt = load_ground_truth(tmp_path)
    broken = check_provenance_comments(tex_file, gt)
    assert len(broken) == 0
