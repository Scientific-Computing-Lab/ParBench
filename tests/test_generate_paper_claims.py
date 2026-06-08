"""TDD RED phase — tests for generate_paper_claims.py.

These tests define the contract: the script must produce a paper_claims.json
with >=25 claims, each traceable to source analysis JSONs, using canonical
terminology (never campaign_1/campaign_2).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAIMS_SCRIPT = PROJECT_ROOT / "scripts" / "analysis" / "generate_paper_claims.py"
CLAIMS_OUTPUT = PROJECT_ROOT / "results" / "analysis" / "paper_claims.json"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"

REQUIRED_KEYS = {
    "claim_id", "section", "claim_text", "value",
    "source_files", "json_path", "verification_cmd",
    "model", "terminology",
}

PATH_ALIAS = {}


def _resolve_json_path(data: dict, path: str) -> object:
    """Traverse a dotted path, mapping canonical -> campaign_2."""
    keys = path.split(".")
    node = data
    for k in keys:
        k = PATH_ALIAS.get(k, k)
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return None
    if isinstance(node, dict) and "value" in node:
        node = node["value"]
    return node


def _generate_claims() -> dict:
    """Run the script and return parsed output."""
    result = subprocess.run(
        [sys.executable, str(CLAIMS_SCRIPT),
         "--project-root", str(PROJECT_ROOT),
         "--output", str(CLAIMS_OUTPUT)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    return json.loads(CLAIMS_OUTPUT.read_text())


@pytest.fixture(scope="module")
def claims_data():
    return _generate_claims()


def test_claims_count_minimum(claims_data):
    """Output must contain at least 25 claims."""
    assert len(claims_data["claims"]) >= 25


def test_claim_required_keys(claims_data):
    """Every claim must have the required keys."""
    for claim in claims_data["claims"]:
        missing = REQUIRED_KEYS - set(claim.keys())
        assert not missing, f"Claim {claim.get('claim_id', '?')} missing keys: {missing}"


def test_claim_values_match_source(claims_data):
    """For each claim with a resolvable json_path, the value must match the source."""
    for claim in claims_data["claims"]:
        if not claim["source_files"] or not claim["json_path"]:
            continue
        src_path = PROJECT_ROOT / claim["source_files"][0]
        if not src_path.exists():
            continue
        src_data = json.loads(src_path.read_text())
        node = _resolve_json_path(src_data, claim["json_path"])
        if node is None:
            continue
        if isinstance(node, (int, float)) and isinstance(claim["value"], (int, float)):
            assert abs(node - claim["value"]) < 1e-6, (
                f"Claim {claim['claim_id']}: expected {node}, got {claim['value']}"
            )


def test_overall_pass_at_1(claims_data):
    """Specific claim 'overall_pass_at_1' must have value ~0.2394 (Qwen pass@1)."""
    matches = [c for c in claims_data["claims"] if c["claim_id"] == "overall_pass_at_1"]
    assert len(matches) == 1, "Expected exactly one overall_pass_at_1 claim"
    assert abs(matches[0]["value"] - 0.2394) < 0.001


def test_known_fail_count(claims_data):
    """Claim about spec counts must reference 9 KNOWN_FAIL."""
    matches = [c for c in claims_data["claims"] if c["claim_id"] == "spec_counts"]
    assert len(matches) == 1
    assert "9" in matches[0]["claim_text"]


def test_validate_mode_exit_code():
    """Running with --validate returns exit code 0 when claims are consistent."""
    _generate_claims()
    result = subprocess.run(
        [sys.executable, str(CLAIMS_SCRIPT),
         "--project-root", str(PROJECT_ROOT),
         "--validate"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Validate failed:\n{result.stdout}\n{result.stderr}"


def test_canonical_terminology(claims_data):
    """No claim should use legacy campaign_1/campaign_2 as its terminology."""
    for claim in claims_data["claims"]:
        assert claim["terminology"] != "campaign_1", (
            f"Claim {claim['claim_id']} uses legacy campaign_1 terminology"
        )
        assert claim["terminology"] != "campaign_2", (
            f"Claim {claim['claim_id']} uses legacy campaign_2 terminology"
        )


def test_verification_cmd_non_empty(claims_data):
    """Claims with source_files and json_path must have a non-empty verification_cmd."""
    for claim in claims_data["claims"]:
        if claim["source_files"] and claim["json_path"]:
            assert claim["verification_cmd"], (
                f"Claim {claim['claim_id']} has source_files and json_path "
                f"but empty verification_cmd"
            )
