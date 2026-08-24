"""TDD RED phase — tests for generate_paper_claims.py.

These tests define the contract: the script must produce a paper_claims.json
with >=25 claims, each traceable to source analysis JSONs, using canonical
terminology (never campaign_1/campaign_2).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAIMS_SCRIPT = PROJECT_ROOT / "scripts" / "analysis" / "generate_paper_claims.py"
# gate2 finding 8 (2026-08-12): write the test's regenerated claims to a temp
# dir, never the tracked results/analysis/paper_claims.json - regenerating that
# committed submitted-era file rewrote its timestamp and dirtied the tree.
_OUT_DIR = Path(tempfile.mkdtemp(prefix="paper_claims_test_"))
CLAIMS_OUTPUT = _OUT_DIR / "paper_claims.json"
# The canonical camera-ready build reads the final/ namespace only (CLAUDE.md:
# the top-level results/analysis/ per-model files are a superseded 2026-05-06
# build with a 9-spec exclusion set and wrong denominators). The fixture targets
# final/ so every assertion checks the numbers the paper actually ships.
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
ANALYSIS_FINAL_DIR = ANALYSIS_DIR / "final"
# D12 (ruling 2026-08-17): every model-scoped claim_id carries a model suffix,
# the primary model included; corpus-level claims stay bare.
PRIMARY_MODEL = "together-qwen-3.5-397b-a17b"

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
         "--analysis-root", str(ANALYSIS_FINAL_DIR),
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
    """The primary model's overall pass@1 claim carries the model-suffixed id
    (D12) and the canonical final-namespace value (~0.2034 =
    aggregate_passk.pass@1_macro_avg). There is NO bare 'overall_pass_at_1'."""
    assert not [c for c in claims_data["claims"]
                if c["claim_id"] == "overall_pass_at_1"], (
        "bare 'overall_pass_at_1' must not exist under the uniform suffix rule")
    cid = f"overall_pass_at_1_{PRIMARY_MODEL}"
    matches = [c for c in claims_data["claims"] if c["claim_id"] == cid]
    assert len(matches) == 1, f"Expected exactly one {cid} claim"
    assert matches[0]["model"] == PRIMARY_MODEL
    assert abs(matches[0]["value"] - 0.2034) < 0.001


def test_known_fail_count(claims_data):
    """The spec_counts claim must reference the canonical KNOWN_FAIL count,
    sourced from ground truth (harness.constants.EXCLUDED_SPECS = 10), never a
    literal. The submitted-era 9 is stale (pre-hecbench-lud-omp)."""
    from harness.constants import EXCLUDED_SPECS
    known_fail = len(EXCLUDED_SPECS)
    assert known_fail == 10, "EXCLUDED_SPECS drifted from the ratified 10"
    matches = [c for c in claims_data["claims"] if c["claim_id"] == "spec_counts"]
    assert len(matches) == 1
    assert matches[0]["value"] == known_fail
    assert str(known_fail) in matches[0]["claim_text"]


def test_validate_mode_exit_code():
    """Running with --validate returns exit code 0 when claims are consistent."""
    _generate_claims()
    # Validate the temp-generated file (via --analysis-root), so the tracked
    # committed paper_claims.json is never read or rewritten here.
    result = subprocess.run(
        [sys.executable, str(CLAIMS_SCRIPT),
         "--project-root", str(PROJECT_ROOT),
         "--analysis-root", str(_OUT_DIR),
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


def _minimal_analysis_dir(root: Path) -> None:
    """Write the smallest analysis set build_claims consumes."""
    model = "together-qwen-3.5-397b-a17b"
    root.mkdir(parents=True, exist_ok=True)

    def metric(v, n=100):
        return {"value": v, "ci_lower": v - 0.02, "ci_upper": v + 0.02, "n": n}
    qf = {
        "canonical": {
            "pass_at_k": {
                "pass_at_1": {**metric(0.4829), "n": 2160},
                "pass_at_3": {**metric(0.55), "n": 2160},
                "total_tasks": {"value": 2160},
                "per_suite": {"pass_at_1": {}},
            },
            "aggregate_pass_rates": {"overall": metric(0.4829, 2160)},
            "direction_pass_rates": {"standard": {
                "cuda-to-omp": metric(0.5, 360),
                "omp-to-cuda": metric(0.45, 360),
            }},
            "failure_taxonomy": {
                "total_records": 2160,
                "status_counts": {"BUILD_FAIL": 480, "RUN_FAIL": 305,
                                  "VERIFY_FAIL": 329},
                "by_status": {"BUILD_FAIL": {"subcategories": {"link": 10, "type": 5}}},
            },
        },
        "metadata": {"excluded_specs_count": 10,
                     "file_counts": {"total_on_disk": 2344, "valid_after_exclusion": 2160,
                                     "excluded_known_fail": 184}},
    }
    (root / f"quantitative_findings_{model}.json").write_text(json.dumps(qf))
    (root / "error_taxonomy.json").write_text(json.dumps({}))
    (root / "sloc_analysis.json").write_text(json.dumps(
        {"summary": {"sloc_vs_pass_rate_spearman": -0.5924}}))
    (root / "token_analysis.json").write_text(json.dumps({"grand_total_cost_usd": 12.5}))
    (root / "benchmark_characterization.json").write_text(json.dumps(
        {"summary": {"total_kernels": 35, "total_suites": 5, "total_specs": 206,
                     "multi_file_pct": 40.0}}))
    (root / "statistical_analysis.json").write_text(json.dumps({}))


def test_claim_sources_track_the_analysis_root(tmp_path):
    """gate1 finding 2 (2026-08-12): with a nested --analysis-root every claim's
    declared source_files must point INTO that root (not the submitted-era
    results/analysis/), so the declared provenance and the emitted value come
    from the same file. Validate-mode must then pass against that root."""
    import scripts.analysis.generate_paper_claims as gpc

    final_dir = tmp_path / "results" / "analysis" / "final"
    _minimal_analysis_dir(final_dir)
    data = gpc.build_claims(tmp_path, analysis_root=final_dir)

    for claim in data["claims"]:
        for sf in claim["source_files"]:
            assert sf.startswith("results/analysis/final/"), (
                f"Claim {claim['claim_id']} source {sf} does not track the "
                "analysis root"
            )
    # overall_pass_at_1 (model-suffixed, D12) value must equal the final-dir
    # source, not 0.2394.
    p1 = [c for c in data["claims"]
          if c["claim_id"] == f"overall_pass_at_1_{PRIMARY_MODEL}"][0]
    assert abs(p1["value"] - 0.4829) < 1e-9

    # Round-trip: written claims validate against their declared sources.
    (final_dir / "paper_claims.json").write_text(json.dumps(data))
    assert gpc.validate_claims(tmp_path, analysis_root=final_dir) == 0


def test_default_analysis_root_is_final_namespace(tmp_path):
    """batch-review 2026-08-17 HIGH finding 5: with NO --analysis-root the
    generator must read the canonical results/analysis/final build, never the
    top-level results/analysis (superseded per-model files, wrong denominators
    and the wrong GPT-5.4 id). Encode the documented default explicitly."""
    import scripts.analysis.generate_paper_claims as gpc

    assert gpc._default_analysis_root(tmp_path) == (
        tmp_path / "results" / "analysis" / "final"
    )

    # A top-level results/analysis populated with DIFFERENT numbers must be
    # ignored by the default; only the final/ tree is consumed.
    top_dir = tmp_path / "results" / "analysis"
    _minimal_analysis_dir(top_dir)
    top_qf = top_dir / "quantitative_findings_together-qwen-3.5-397b-a17b.json"
    poisoned = json.loads(top_qf.read_text())
    poisoned["canonical"]["pass_at_k"]["pass_at_1"]["value"] = 0.2394
    top_qf.write_text(json.dumps(poisoned))

    final_dir = top_dir / "final"
    _minimal_analysis_dir(final_dir)

    data = gpc.build_claims(tmp_path)  # no analysis_root -> default
    for claim in data["claims"]:
        for sf in claim["source_files"]:
            assert sf.startswith("results/analysis/final/"), (
                f"default run read {sf} outside the final/ namespace"
            )
    p1 = [c for c in data["claims"]
          if c["claim_id"] == f"overall_pass_at_1_{PRIMARY_MODEL}"][0]
    assert abs(p1["value"] - 0.4829) < 1e-9, (
        "default run picked up the poisoned top-level value"
    )


# --------------------------------------------------------------------------- #
# gate2 finding 4: derived claims resolve to scalars; validation fails on a    #
# container path that cannot resolve.                                          #
# --------------------------------------------------------------------------- #

def test_eval_derivation_ops():
    import scripts.analysis.generate_paper_claims as gpc
    doc = {
        "canonical": {
            "failure_taxonomy": {"by_status": {"BUILD_FAIL": {
                "subcategories": {"a": 1, "b": 2, "c": 3}}}},
            "augmentation_trends": {"aggregate": {"per_level": {
                "L0": {"value": 0.5}, "L4": {"value": 0.3}}}},
        },
        "direction_asymmetry": [
            {"forward_pass_rate": 0.6, "reverse_pass_rate": 0.2},
            {"forward_pass_rate": 0.5, "reverse_pass_rate": 0.45},
        ],
        "chi2_augmentation_by_model": [{"chi2": 7.5}, {"chi2": 1.0}],
    }
    def num(der: dict) -> float:
        """Resolve a derivation and narrow its object return to a float so the
        numeric asserts below are well-typed (the op returns object | None)."""
        v = gpc._eval_derivation(doc, der)
        assert isinstance(v, (int, float)), f"expected numeric, got {v!r}"
        return float(v)

    assert num({
        "op": "len",
        "path": "canonical.failure_taxonomy.by_status.BUILD_FAIL.subcategories"}) == 3
    assert abs(num({
        "op": "diff",
        "minuend": "canonical.augmentation_trends.aggregate.per_level.L4.value",
        "subtrahend": "canonical.augmentation_trends.aggregate.per_level.L0.value"})
        - (-0.2)) < 1e-9
    assert abs(num({
        "op": "max_abs_gap", "path": "direction_asymmetry",
        "forward": "forward_pass_rate", "reverse": "reverse_pass_rate"}) - 0.4) < 1e-9
    assert num({
        "op": "index_field", "path": "chi2_augmentation_by_model",
        "index": 0, "field": "chi2"}) == 7.5


def test_validate_fails_on_unresolved_container_claim(tmp_path):
    """A numeric claim whose json_path points at a container and carries NO
    derivation must FAIL validation, not be silently skipped (gate2 finding 4)."""
    import scripts.analysis.generate_paper_claims as gpc
    an = tmp_path / "an"
    an.mkdir()
    src = an / "src.json"
    src.write_text(json.dumps({"box": {"x": 1, "y": 2}}))
    claims = {"claims": [{
        "claim_id": "bad", "value": 2, "source_files": [str(src)],
        "json_path": "box", "section": "S", "claim_text": "t"}]}
    (an / "paper_claims.json").write_text(json.dumps(claims))
    assert gpc.validate_claims(tmp_path, analysis_root=an) == 1


def test_validate_passes_derived_claim_with_recipe(tmp_path):
    """The same container source resolves when the claim carries a derivation."""
    import scripts.analysis.generate_paper_claims as gpc
    an = tmp_path / "an"
    an.mkdir()
    src = an / "src.json"
    src.write_text(json.dumps({"box": {"x": 1, "y": 2, "z": 3}}))
    claims = {"claims": [{
        "claim_id": "good", "value": 3, "source_files": [str(src)],
        "json_path": "box", "section": "S", "claim_text": "t",
        "derivation": {"op": "len", "path": "box"}}]}
    (an / "paper_claims.json").write_text(json.dumps(claims))
    assert gpc.validate_claims(tmp_path, analysis_root=an) == 0


# --------------------------------------------------------------------------- #
# gate2 finding 5: no output may depend on a hand-written corpus constant; a   #
# missing key must fail rather than silently use a baked-in literal.           #
# --------------------------------------------------------------------------- #

def test_build_claims_fails_on_missing_required_key(tmp_path):
    import scripts.analysis.generate_paper_claims as gpc
    final_dir = tmp_path / "results" / "analysis" / "final"
    _minimal_analysis_dir(final_dir)
    qf_path = final_dir / "quantitative_findings_together-qwen-3.5-397b-a17b.json"
    qf = json.loads(qf_path.read_text())
    del qf["metadata"]["file_counts"]["total_on_disk"]
    qf_path.write_text(json.dumps(qf))
    with pytest.raises(SystemExit):
        gpc.build_claims(tmp_path, analysis_root=final_dir)


def test_no_hardcoded_corpus_fallback_literals():
    """The submitted-era fallback literals (708/206/35/5/626) must be gone from
    the claim builders - every count now comes from a generated artifact."""
    src = (PROJECT_ROOT / "scripts/analysis/generate_paper_claims.py").read_text()
    # Only inspect build_claims (the claim-emitting body), where the fallbacks lived.
    body = src[src.index("def build_claims"):src.index("def validate_claims")]
    for magic in (".get('total_on_disk', 708)", ", 708)", ", 626)",
                  "'total_kernels', 35", "'total_suites', 5", "'total_specs', 206"):
        assert magic not in body, f"hand-written fallback still present: {magic}"


def test_gate12_chi2_claim_has_distinct_id(claims_data):
    """gate12 finding 1: the chi-squared augmentation-trend claim must NOT reuse
    the claim_id `cochran_armitage_trend` - quantitative_findings.py already
    emits that id for the genuine Cochran-Armitage z-statistic, so the same key
    denoted two different statistics across the evidence package. The chi-squared
    claim is emitted under the distinct id `chi2_augmentation_trend`."""
    by_id = {}
    for c in claims_data["claims"]:
        by_id.setdefault(c["claim_id"], []).append(c)
    # The paper_claims chi2 claim is the one whose text names a chi-squared trend.
    chi2 = [c for c in claims_data["claims"]
            if "chi" in c.get("claim_text", "").lower()
            and "augmentation trend" in c.get("claim_text", "").lower()]
    assert chi2, "no chi-squared augmentation-trend claim emitted"
    # Under the uniform suffix rule the id is model-scoped (D12), so it is the
    # distinct base `chi2_augmentation_trend` plus a model suffix - never the
    # bare or the colliding `cochran_armitage_trend` base.
    for c in chi2:
        assert c["claim_id"].startswith("chi2_augmentation_trend_"), (
            f"chi-squared claim uses unexpected id {c['claim_id']!r}; expected "
            "base 'chi2_augmentation_trend' with a model suffix")
        assert not c["claim_id"].startswith("cochran_armitage_trend"), c["claim_id"]
    assert "cochran_armitage_trend" not in by_id, (
        "paper_claims.json must not emit `cochran_armitage_trend` - that id is "
        "reserved for the Cochran-Armitage z-statistic in quantitative_findings")


def test_gate16_direction_asymmetry_claim_is_pooled(claims_data):
    """gate16 finding 3: the direction_asymmetry statistic pools its paired cells
    across all three evaluated models (its paired_cells carry codex, gpt-5.4, and
    qwen rows), so the claim must be labelled model='pooled'. The default label
    (the headline Qwen model) would misattribute a cross-model statistic to a
    single model."""
    da = next(c for c in claims_data["claims"]
              if c["claim_id"] == "direction_asymmetry")
    assert da["model"] == "pooled", (
        f"direction_asymmetry pools all models but is labelled {da['model']!r}")


def test_gate14_every_compound_claim_number_is_verified(claims_data):
    """gate14 finding 3: a compound claim (spec_counts, suite_composition,
    total_eval_files) states several numbers in its claim_text but declares a
    value/source/json_path for only one, so the others are never resolved,
    validated, or compared old-vs-new. Every integer named in a compound
    claim's text must equal some emitted claim's numeric value, so each reported
    number is independently traceable to a generated artifact."""
    import re
    claims = claims_data["claims"]
    numeric_values = {
        c["value"] for c in claims
        if isinstance(c["value"], (int, float)) and not isinstance(c["value"], bool)
    }
    # total_eval_files is model-scoped (per-model file counts) so it is suffixed;
    # spec_counts and suite_composition are corpus-level and stay bare (D12).
    compound_ids = {"spec_counts", "suite_composition",
                    f"total_eval_files_{PRIMARY_MODEL}"}
    for c in claims:
        if c["claim_id"] not in compound_ids:
            continue
        # Every standalone integer in the text must be some claim's value.
        for tok in re.findall(r"\b\d+\b", c["claim_text"]):
            n = int(tok)
            assert n in numeric_values, (
                f"{c['claim_id']} text names {n} but no emitted claim carries it "
                f"as a verifiable value: {c['claim_text']!r}")


def test_gate14_sibling_count_claims_resolve_to_source(claims_data):
    """gate14 finding 3: the split-out sibling count claims must each resolve to
    their declared source path (not merely appear in prose)."""
    by_id = {c["claim_id"]: c for c in claims_data["claims"]}
    for cid, jpath in (
        # total_specs_count / total_suites_count are corpus-level (bare);
        # valid_eval_files_count is per-model (suffixed) under D12.
        ("total_specs_count", "summary.total_specs"),
        ("total_suites_count", "summary.total_suites"),
        (f"valid_eval_files_count_{PRIMARY_MODEL}",
         "metadata.file_counts.valid_after_exclusion"),
    ):
        assert cid in by_id, f"missing sibling count claim {cid!r}"
        c = by_id[cid]
        assert c["json_path"] == jpath, c
        src = PROJECT_ROOT / c["source_files"][0]
        node = _resolve_json_path(json.loads(src.read_text()), c["json_path"])
        assert node == c["value"], f"{cid}: source {node} != value {c['value']}"


# --------------------------------------------------------------------------- #
# D12 (ruling 2026-08-17): the generator emits claims for all three evaluated  #
# models under a UNIFORM suffix convention - EVERY model-scoped claim_id (the  #
# primary model included) carries a model suffix, and only corpus-level claims #
# stay bare. The two added models source from their paper_data passk_campaign. #
# --------------------------------------------------------------------------- #

_ADDED_MODELS = ("azure-gpt-5.4", "azure-gpt-5.3-codex")
# Corpus-level claims that are not model-specific and therefore stay bare.
_BARE_CORPUS_IDS = {
    "direction_asymmetry", "spec_counts", "total_specs_count",
    "suite_composition", "total_suites_count", "multi_file_fraction",
    "token_cost_total", "sloc_correlation",
}


def test_d12_three_models_present(claims_data):
    """The output declares all three evaluated models, and each added model
    contributes a model-suffixed per-model claim set."""
    models = set(claims_data.get("models", []))
    assert "together-qwen-3.5-397b-a17b" in models
    for m in _ADDED_MODELS:
        assert m in models, f"added model {m} missing from output 'models'"

    labelled = {c["model"] for c in claims_data["claims"]}
    for m in _ADDED_MODELS:
        assert m in labelled, f"no claim labelled model={m}"
        # Each added model must carry its overall pass@1, sourced from its own
        # paper_data file, with a model-suffixed id distinct from the bare one.
        cid = f"overall_pass_at_1_{m}"
        match = [c for c in claims_data["claims"] if c["claim_id"] == cid]
        assert len(match) == 1, f"expected exactly one {cid}"
        assert match[0]["source_files"][0].endswith(f"paper_data_{m}.json")
        assert match[0]["json_path"] == (
            "passk_campaign.aggregate_passk.pass@1_macro_avg")


def test_d12_uniform_suffix_convention(claims_data):
    """The uniform-suffix invariant: a claim's id is suffixed with its model IFF
    the claim is model-scoped (model != 'pooled'); corpus-level claims (model ==
    'pooled') stay bare. This closes the implicit-default ambiguity where a bare
    id silently meant the primary model."""
    ids = [c["claim_id"] for c in claims_data["claims"]]
    assert len(ids) == len(set(ids)), "claim_ids must be unique"
    for c in claims_data["claims"]:
        cid, model = c["claim_id"], c["model"]
        if model == "pooled":
            assert cid in _BARE_CORPUS_IDS, (
                f"pooled claim {cid!r} is not in the known bare-corpus set")
            assert not cid.endswith(PRIMARY_MODEL), (
                f"corpus-level {cid!r} must stay bare")
        else:
            assert cid.endswith(f"_{model}"), (
                f"model-scoped claim {cid!r} (model={model!r}) must be suffixed "
                "with its model id")


def test_d12_no_bare_model_scoped_ids(claims_data):
    """No bare (unsuffixed) form of a model-scoped base survives - e.g. a bare
    'overall_pass_at_1' would silently default to one model."""
    present = {c["claim_id"] for c in claims_data["claims"]}
    for base in ("overall_pass_at_1", "overall_pass_at_3", "aggregate_pass_rate",
                 "build_fail_count", "best_direction", "worst_direction"):
        assert base not in present, (
            f"bare model-scoped id {base!r} must not exist under the suffix rule")


def test_d12_qwen_two_sources_agree():
    """D12 licensing check: the whole justification for sourcing the two added
    models from paper_data (they have no quantitative_findings file) is that
    paper_data's aggregate_passk macro-averages ARE the same metric as the
    primary model's quantitative_findings pass_at_1/pass_at_3. Assert that
    equality on qwen (which has BOTH files) so a future schema divergence between
    the two producers fails here instead of silently making the claim family
    inconsistent across models."""
    model = "together-qwen-3.5-397b-a17b"
    qf = json.loads((ANALYSIS_FINAL_DIR /
                     f"quantitative_findings_{model}.json").read_text())["canonical"]
    pd = json.loads((ANALYSIS_FINAL_DIR /
                     f"paper_data_{model}.json").read_text())["passk_campaign"]
    agg = pd["aggregate_passk"]
    assert abs(qf["pass_at_k"]["pass_at_1"]["value"] - agg["pass@1_macro_avg"]) < 1e-9, (
        "quantitative_findings pass_at_1 != paper_data aggregate_passk pass@1_macro_avg")
    assert abs(qf["pass_at_k"]["pass_at_3"]["value"] - agg["pass@3_macro_avg"]) < 1e-9, (
        "quantitative_findings pass_at_3 != paper_data aggregate_passk pass@3_macro_avg")


def test_d12_glob_arity_guard(tmp_path):
    """D12: the per-model set is discovered from the paper_data_*.json glob, so a
    shrunk glob (a missing or renamed file) must fail loudly, not emit a silent
    subset. A non-empty glob whose size != the evaluated-model count aborts."""
    import scripts.analysis.generate_paper_claims as gpc
    an = tmp_path / "an"
    an.mkdir()
    # Only two of the three evaluated models present -> must abort.
    (an / "paper_data_together-qwen-3.5-397b-a17b.json").write_text("{}")
    (an / "paper_data_azure-gpt-5.4.json").write_text("{}")
    with pytest.raises(SystemExit):
        gpc._discover_additional_models(an)
    # Empty glob (minimal unit fixture) is allowed and yields no added models.
    assert gpc._discover_additional_models(tmp_path / "empty") == []


def test_per_model_eval_file_count_invariants(claims_data):
    """Per-model eval-file counts (total on disk / valid / excluded) are emitted
    for all three evaluated models and satisfy the canonical-build invariants:
    totals sum to 2,344, valid to 2,160, excluded to 184 (D10/D12). Qwen's valid
    count stays byte-identical at 604 (rename-not-revalue)."""
    by_id = {c["claim_id"]: c["value"] for c in claims_data["claims"]}

    def _sum(prefix):
        vals = {k: v for k, v in by_id.items() if k.startswith(prefix)}
        assert len(vals) == 3, f"expected 3 {prefix}* claims, got {sorted(vals)}"
        return sum(vals.values())

    assert _sum("total_eval_files_") == 2344
    assert _sum("valid_eval_files_count_") == 2160
    assert _sum("excluded_eval_files_count_") == 184
    # Per-model total = valid + excluded, for every model.
    for model in ("together-qwen-3.5-397b-a17b", "azure-gpt-5.4", "azure-gpt-5.3-codex"):
        t = by_id[f"total_eval_files_{model}"]
        v = by_id[f"valid_eval_files_count_{model}"]
        e = by_id[f"excluded_eval_files_count_{model}"]
        assert t == v + e, f"{model}: {t} != {v} + {e}"
    # Qwen valid stays byte-identical at 604.
    assert by_id["valid_eval_files_count_together-qwen-3.5-397b-a17b"] == 604
