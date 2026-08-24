#!/usr/bin/env bash
# ParBench — Reproduce all tables and figures from raw evaluation data
# Usage: ./reproduce.sh
# Expected runtime: ~10-15 minutes on a modern machine (no GPU required)
set -euo pipefail

# ── Resolve project root (works in Docker /app, repo checkout, and artifact tarball) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/manifest.jsonl" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../manifest.jsonl" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: Cannot find project root (looked for manifest.jsonl)." >&2
    echo "Run from the project directory or use Docker (see artifact/README.md)." >&2
    exit 1
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! python3 -c "import harness" 2>/dev/null; then
    echo "ERROR: 'harness' package not importable." >&2
    echo "Install dependencies first: pip install -r requirements-lock.txt && pip install -e ." >&2
    exit 1
fi

echo "========================================"
echo "ParBench Reproducibility Artifact"
echo "========================================"
echo ""

OUTPUT_DIR="${1:-$PROJECT_ROOT/output}"
mkdir -p "$OUTPUT_DIR"

# Reproduction is self-contained (2026-08-21): every regenerated aggregate
# lands under $OUTPUT_DIR, never in the checkout's results/analysis/. The
# committed canonical aggregates live in results/analysis/final/ (the sealed
# 2026-08-13 build the paper prints); this run's aggregates may differ where
# the sealed replay corrected verdicts.
ANALYSIS_DIR="$OUTPUT_DIR/analysis"
mkdir -p "$ANALYSIS_DIR"

# ── Step 1: Generate per-model paper_data JSONs ──────────────────────
echo "[Step 1/5] Generating per-model paper_data JSONs..."

python3 scripts/analysis/generate_paper_data.py \
    --results-dir results/evaluation/together-qwen-3.5-397b-a17b \
    --output "$ANALYSIS_DIR/paper_data_together-qwen-3.5-397b-a17b.json" -v

python3 scripts/analysis/generate_paper_data.py \
    --results-dir results/evaluation/azure-gpt-5.4 \
    --output "$ANALYSIS_DIR/paper_data_azure-gpt-5.4.json" -v

python3 scripts/analysis/generate_paper_data.py \
    --results-dir results/evaluation/azure-gpt-5.3-codex \
    --output "$ANALYSIS_DIR/paper_data_azure-gpt-5.3-codex.json" -v

echo "  Done."
echo ""

# ── Step 2: Generate per-model quantitative findings ─────────────────
echo "[Step 2/5] Generating per-model quantitative findings..."

python3 scripts/analysis/quantitative_findings.py \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$ANALYSIS_DIR" \
    --model-dir together-qwen-3.5-397b-a17b -v

python3 scripts/analysis/quantitative_findings.py \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$ANALYSIS_DIR" \
    --model-dir azure-gpt-5.4 -v

python3 scripts/analysis/quantitative_findings.py \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$ANALYSIS_DIR" \
    --model-dir azure-gpt-5.3-codex -v

echo "  Done."
echo ""

# ── Step 3: Run statistical analysis (all models) ────────────────────
echo "[Step 3/5] Running statistical analysis (cross-model)..."

python3 scripts/analysis/statistical_analysis.py \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$ANALYSIS_DIR" -v

echo "  Done."
echo ""

# ── Step 4: Cross-model comparisons (3 pairs) ────────────────────────
echo "[Step 4/5] Running cross-model comparisons..."

python3 scripts/analysis/cross_model_comparison.py \
    --model-a "$ANALYSIS_DIR/paper_data_together-qwen-3.5-397b-a17b.json" \
    --model-b "$ANALYSIS_DIR/paper_data_azure-gpt-5.4.json" \
    --output "$ANALYSIS_DIR/cross_model_comparison_qwen_vs_gpt54.json" -v

python3 scripts/analysis/cross_model_comparison.py \
    --model-a "$ANALYSIS_DIR/paper_data_together-qwen-3.5-397b-a17b.json" \
    --model-b "$ANALYSIS_DIR/paper_data_azure-gpt-5.3-codex.json" \
    --output "$ANALYSIS_DIR/cross_model_comparison_qwen_vs_codex.json" -v

python3 scripts/analysis/cross_model_comparison.py \
    --model-a "$ANALYSIS_DIR/paper_data_azure-gpt-5.4.json" \
    --model-b "$ANALYSIS_DIR/paper_data_azure-gpt-5.3-codex.json" \
    --output "$ANALYSIS_DIR/cross_model_comparison_gpt54_vs_codex.json" -v

echo "  Done."
echo ""

# ── Step 5: Generate all figures and tables ───────────────────────────
echo "[Step 5/5] Generating all figures (F2-F7, C.1-C.4) and tables (T1-T5)..."

# --results-root points the figure loader at the SAME evaluation namespace that
# Steps 1-4 read (results/evaluation/<model>), the layout both artifact builders
# stage. Without it the loader defaults to results/evaluation_final/parbench-final-v1,
# which the artifact does not contain, and figures would be silently incomplete.
# --findings-dir points the T1/T3/T4/T5 tables at THIS run's regenerated
# aggregates (Steps 1-3) instead of the committed canonical build in
# results/analysis/final/ - the reproduction stays end-to-end from raw data.
python3 scripts/generate_paper_figures.py \
    --project-root "$PROJECT_ROOT" \
    --results-root "$PROJECT_ROOT/results/evaluation" \
    --findings-dir "$ANALYSIS_DIR" \
    --figure all \
    --output-dir "$OUTPUT_DIR" -v

echo "  Done."
echo ""

# ── Summary ───────────────────────────────────────────────────────────
echo "========================================"
echo "Reproduction complete!"
echo "========================================"
TOTAL=$(find "$OUTPUT_DIR" -type f | wc -l)
echo "Output files: $TOTAL"
echo "Output directory: $OUTPUT_DIR"
echo ""

echo "Tables:"
for t in "$OUTPUT_DIR"/t*.tex; do
    [ -f "$t" ] && echo "  $(basename "$t")" || true
done
echo ""
echo "Figures:"
for f in "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.png; do
    [ -f "$f" ] && echo "  $(basename "$f")" || true
done
