"""Tests for extracted utility functions in generate_paper_figures.py."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_aggregate_status_counts_by_model():
    from scripts.generate_paper_figures import aggregate_status_counts
    records = [
        {"model": "claude", "overall_status": "PASS"},
        {"model": "claude", "overall_status": "FAIL"},
        {"model": "gpt", "overall_status": "PASS"},
    ]
    result = aggregate_status_counts(records, "model")
    assert result["claude"]["PASS"] == 1
    assert result["claude"]["FAIL"] == 1
    assert result["gpt"]["PASS"] == 1


def test_aggregate_status_counts_missing_status_defaults_unknown():
    from scripts.generate_paper_figures import aggregate_status_counts
    records = [{"model": "m", "overall_status": None}]
    result = aggregate_status_counts(records, "model")
    assert result["m"]["UNKNOWN"] == 1


def test_constants_exist():
    import scripts.generate_paper_figures as gpf
    assert gpf.FIGURE_DPI == 600
    assert gpf.PDF_FONTTYPE == 42
    assert isinstance(gpf.REPO_KERNEL_PAIRS, list)
    assert isinstance(gpf.HECBENCH_FUNNEL_STAGES, list)
    assert len(gpf.HECBENCH_FUNNEL_STAGES) == 5


def test_create_status_legend_returns_patches():
    from matplotlib.patches import Patch

    from scripts.generate_paper_figures import create_status_legend
    statuses = ["PASS", "BUILD_FAIL"]
    patches = create_status_legend(statuses)
    assert len(patches) == 2
    assert all(isinstance(p, Patch) for p in patches)


# ---------------------------------------------------------------------------
# G1: Model constants reflect the canonical 3-model layout
# (Qwen + Azure GPT-5.4 + Azure GPT-5.3-codex)
# ---------------------------------------------------------------------------

def _canonical_model_ids() -> list[str]:
    """Canonical model ids, read from the frozen per-model paper_data file set.

    Ground truth is the same the analysis chain uses: the stems of
    ``results/analysis/final/paper_data_*.json`` (canonical 2026-08-13 build),
    with the ``paper_data_`` prefix stripped. Never hard-code a model literal here.
    """
    final_dir = Path(__file__).resolve().parent.parent.parent / "results" / "analysis" / "final"
    ids = sorted(
        p.stem[len("paper_data_"):]
        for p in final_dir.glob("paper_data_*.json")
    )
    # Assert the EXACT canonical count, not merely non-empty: a glob that silently
    # resolves to fewer files (one renamed/removed/filtered) would let a widened
    # test quietly run over 2 models and still pass. Fail loud with the resolved set.
    assert len(ids) == 3, (
        f"expected exactly 3 canonical model ids from {final_dir}, got {len(ids)}: {ids}"
    )
    return ids


def test_model_constants_reflect_canonical_layout():
    """Old models are removed and every canonical model id is present in all 4 dicts."""
    import scripts.generate_paper_figures as gpf

    old_models = [
        "claude-sonnet-4-6",
        "groq-llama-3.3-70b-versatile",
        "gemini-2.5-flash-lite",
    ]
    canonical_models = _canonical_model_ids()

    for dct_name in ("MODEL_COLORS", "MODEL_DISPLAY", "MODEL_DISPLAY_SHORT", "MODEL_LINESTYLE"):
        dct = getattr(gpf, dct_name)
        # Old models must be absent
        for old in old_models:
            assert old not in dct, f"{old} still in {dct_name}"
        # Every canonical model (all three) must be present
        for model_id in canonical_models:
            assert model_id in dct, f"{model_id} missing from {dct_name}"


# ---------------------------------------------------------------------------
# G2: F3 canonical-L0 data path yields the same kernel set for every model
# ---------------------------------------------------------------------------

# Expected unique-kernel count in the F3 canonical-L0 data path, per model,
# after CORRECTNESS_INELIGIBLE_SPECS exclusion (enforced in load_eval_results).
# The corpus is frozen (results/evaluation is the immutable submitted corpus),
# so this is an EXACT invariant, not a floor: a change either way means the
# exclusion stopped being applied or the corpus was mutated.
#
# Why the same 30 for all three models: Qwen's batch INCLUDED the KNOWN_FAIL
# specs (excluded at analysis), while the GPT-5.4 and codex batches pre-excluded
# them (appendices_neurips.tex; benchmark-curation.tex:87). Before exclusion
# Qwen carried 34 kernels vs 31 for the others; the extra three -- hybridsort,
# kmeans, mummergpu -- exist for Qwen only through KNOWN_FAIL specs
# (rodinia-hybridsort-cuda, rodinia-kmeans-{cuda,opencl}, rodinia-mummergpu-{cuda,omp}).
# CORRECTNESS_INELIGIBLE_SPECS removes those, converging every model to 30.
EXPECTED_F3_KERNELS_PER_MODEL = 30
KNOWN_FAIL_ONLY_KERNELS = {"hybridsort", "kmeans", "mummergpu"}


def test_f3_heatmap_kernel_count():
    """F3 canonical-L0 path yields exactly 30 kernels for every canonical model.

    Uses the real primitives (load_eval_results, which now enforces the
    correctness-eligibility exclusion, and get_canonical_l0, the s0-fallback
    the real generate_f3_kernel_heatmap uses) rather than reimplementing the
    filter, so the test cannot drift from the code path it certifies.
    """
    from scripts.generate_paper_figures import (
        ALL_DIRECTIONS,
        get_canonical_l0,
        load_eval_results,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    records = load_eval_results(project_root)
    assert len(records) > 0, "No eval records found on disk"

    directions = list(ALL_DIRECTIONS)
    for model_id in _canonical_model_ids():
        # Per-model canonical L0 (s0 fallback for sample-only models like Qwen).
        model_records = [r for r in records if r.get("model") == model_id]
        l0_records = get_canonical_l0(model_records)
        std_records = [r for r in l0_records if r["direction"] in directions]

        kernel_suite = {r["kernel"]: r.get("suite", "other") for r in std_records}

        assert len(kernel_suite) == EXPECTED_F3_KERNELS_PER_MODEL, (
            f"Expected exactly {EXPECTED_F3_KERNELS_PER_MODEL} kernels in the F3 "
            f"canonical path for {model_id}, got {len(kernel_suite)}. "
            f"Kernels: {sorted(kernel_suite)}"
        )

        # The KNOWN_FAIL-only kernels must be gone (proves the exclusion fired).
        leaked = KNOWN_FAIL_ONLY_KERNELS & set(kernel_suite)
        assert not leaked, (
            f"KNOWN_FAIL-only kernels leaked into F3 for {model_id}: {sorted(leaked)}"
        )

        # Multiple suites represented (not just rodinia).
        suites_present = set(kernel_suite.values())
        assert len(suites_present) >= 3, (
            f"Expected kernels from at least 3 suites for {model_id}, got: {suites_present}"
        )


# ---------------------------------------------------------------------------
# G3: SUITE_ORDER contains all 5 suites
# ---------------------------------------------------------------------------

def test_suite_order_contains_all_five_suites():
    """Verify SUITE_ORDER lists exactly the 5 expected benchmark suites."""
    from scripts.generate_paper_figures import SUITE_ORDER

    expected = ["rodinia", "xsbench", "rsbench", "mixbench", "hecbench"]
    assert SUITE_ORDER == expected, (
        f"SUITE_ORDER mismatch.\n  Expected: {expected}\n  Got:      {SUITE_ORDER}"
    )


# ---------------------------------------------------------------------------
# G4: T2 table carries a row for every canonical model, all with computed stats
# ---------------------------------------------------------------------------

def test_t2_table_has_canonical_model_layout(tmp_path):
    """generate_t2_model_table produces a row per canonical model, all computed."""
    import scripts.generate_paper_figures as gpf
    from scripts.generate_paper_figures import (
        generate_t2_model_table,
        load_eval_results,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    records = load_eval_results(project_root)
    assert len(records) > 0, "No eval records found on disk"

    generate_t2_model_table(records, tmp_path, verbose=False)

    tex_path = tmp_path / "t2_model_comparison.tex"
    assert tex_path.exists(), "T2 LaTeX file was not created"

    content = tex_path.read_text()
    # Every canonical model must have a row, keyed off the display name the
    # table actually renders (MODEL_DISPLAY_SHORT) — no hard-coded literal.
    for model_id in _canonical_model_ids():
        display = gpf.MODEL_DISPLAY_SHORT[model_id]
        assert display in content, f"{display} row missing from T2 table"
    # Every row should have computed stats (no "pending" entries)
    assert "pending" not in content.lower(), (
        "T2 table still has 'pending' entries — all model stats should be computed"
    )
    assert "\\toprule" in content, "LaTeX table missing \\toprule"
    assert "\\bottomrule" in content, "LaTeX table missing \\bottomrule"


# ---------------------------------------------------------------------------
# G5: Every generated PDF has a matching PNG file
# ---------------------------------------------------------------------------

def test_pdf_png_parity():
    """Verify each figure PDF in docs/paper/figures/ has a corresponding PNG sibling."""
    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = project_root / "docs" / "paper" / "figures"

    # These are the actual output stems from _save_figure() calls in the script
    expected_stems = [
        "f2_repo_vs_kernel",
        "f3_kernel_model_heatmap_qwen",
        "f3_kernel_model_heatmap_gpt",
        "f4_failure_taxonomy_qwen",
        "f4_failure_taxonomy_gpt",
        "f5_pass_at_k_by_direction_qwen",
        "f5_pass_at_k_by_direction_gpt",
        "f6_cross_suite_comparison_qwen",
        "f6_cross_suite_comparison_gpt",
        "f7_augmentation_robustness",
        # C.1/C.2 (self-repair figures) are intentionally omitted: the canonical
        # sealed-replay data has zero multi-attempt records (pass@k sampling, not
        # iterative repair), so the generator correctly skips them and the manuscript
        # inputs neither. Do not re-add without self-repair data on disk.
        "c3_transform_frequency",
        "c4_selection_funnel",
    ]

    missing_pdf = []
    missing_png = []

    for stem in expected_stems:
        pdf_path = figures_dir / f"{stem}.pdf"
        png_path = figures_dir / f"{stem}.png"
        if not pdf_path.exists():
            missing_pdf.append(stem)
        if not png_path.exists():
            missing_png.append(stem)

    assert not missing_pdf, f"Missing PDFs for stems: {missing_pdf}"
    assert not missing_png, f"Missing PNGs for stems: {missing_png}"

    # Verify parity: every PDF has a matching PNG and vice versa
    actual_pdfs = {p.stem for p in figures_dir.glob("*.pdf")}
    actual_pngs = {p.stem for p in figures_dir.glob("*.png")}
    for stem in expected_stems:
        assert stem in actual_pdfs, f"PDF missing: {stem}.pdf"
        assert stem in actual_pngs, f"PNG missing: {stem}.png"


# ---------------------------------------------------------------------------
# G6: an eval-figure request with no evaluation data FAILS LOUDLY (finding 6)
# ---------------------------------------------------------------------------

def test_missing_eval_data_fails_loudly(tmp_path):
    """When eval figures are requested but --results-root has no records, the
    generator must exit nonzero instead of silently pruning and exiting 0.

    This is the artifact contract: reproduce.sh must never emit an incomplete
    figure set that still looks successful.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    empty_root = tmp_path / "empty_eval"
    empty_root.mkdir()
    out = tmp_path / "figs"

    r = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "generate_paper_figures.py"),
         "--project-root", str(project_root),
         "--results-root", str(empty_root),
         "--figure", "F3",
         "--output-dir", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, r.stdout + r.stderr
    assert "no evaluation data" in r.stderr.lower()
