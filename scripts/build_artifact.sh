#!/usr/bin/env bash
# Build the ParBench reproducibility artifact tarball.
# Usage: bash scripts/build_artifact.sh [--dry-run]
#   --dry-run  Stage files and run anonymization but skip Docker build/run
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="/tmp/parbench-artifact-staging/parbench-artifact"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

TOTAL_STEPS=5
STEP=0
step() { STEP=$((STEP + 1)); echo "[$STEP/$TOTAL_STEPS] $1"; }

echo "=== ParBench Artifact Builder ==="
echo "Project root: $PROJECT_ROOT"
echo "Staging dir:  $STAGING"
echo "Dry run:      $DRY_RUN"
echo ""

# ── 1. Clean and create staging directory ───────────────────────────
rm -rf /tmp/parbench-artifact-staging
mkdir -p "$STAGING"

# ── 2. Copy files ───────────────────────────────────────────────────
step "Copying files to staging..."

# Specs, manifest, schema
cp -r "$PROJECT_ROOT/specs" "$STAGING/specs"
cp "$PROJECT_ROOT/manifest.jsonl" "$STAGING/"
cp -r "$PROJECT_ROOT/schema" "$STAGING/schema"

# Results — JSON only (no .log, .marker)
for model_dir in together-qwen-3.5-397b-a17b azure-gpt-5.4 azure-gpt-5.3-codex; do
    src="$PROJECT_ROOT/results/evaluation/$model_dir"
    dst="$STAGING/results/evaluation/$model_dir"
    mkdir -p "$dst"
    find "$src" -maxdepth 1 -name "*.json" -exec cp {} "$dst/" \;
done

# Analysis summaries
mkdir -p "$STAGING/results/analysis"
cp "$PROJECT_ROOT"/results/analysis/*.json "$STAGING/results/analysis/"

# Scripts — analysis + evaluation dependency (BLOCK-3)
mkdir -p "$STAGING/scripts/analysis"
find "$PROJECT_ROOT/scripts/analysis" -maxdepth 1 -name "*.py" -exec cp {} "$STAGING/scripts/analysis/" \;

mkdir -p "$STAGING/scripts/evaluation"
cp "$PROJECT_ROOT/scripts/evaluation/analyze_eval.py" "$STAGING/scripts/evaluation/"
cp "$PROJECT_ROOT/scripts/evaluation/__init__.py" "$STAGING/scripts/evaluation/"

cp "$PROJECT_ROOT/scripts/generate_paper_figures.py" "$STAGING/scripts/"
cp "$PROJECT_ROOT/scripts/validate_schema.py" "$STAGING/scripts/"

# Harness + c_augmentation (needed for pip install -e .)
mkdir -p "$STAGING/harness" "$STAGING/c_augmentation"
find "$PROJECT_ROOT/harness" -maxdepth 1 -name "*.py" -exec cp {} "$STAGING/harness/" \;
find "$PROJECT_ROOT/c_augmentation" -maxdepth 1 -name "*.py" -exec cp {} "$STAGING/c_augmentation/" \;

# Config, build files
cp -r "$PROJECT_ROOT/config" "$STAGING/config"
cp "$PROJECT_ROOT/pyproject.toml" "$STAGING/"
cp "$PROJECT_ROOT/requirements-lock.txt" "$STAGING/"
cp "$PROJECT_ROOT/LICENSE" "$STAGING/"

# Artifact files (Dockerfile, reproduce.sh, README.md) → staging root
cp "$PROJECT_ROOT/artifact/Dockerfile" "$STAGING/"
cp "$PROJECT_ROOT/artifact/reproduce.sh" "$STAGING/"
cp "$PROJECT_ROOT/artifact/README.md" "$STAGING/"

# Paper LaTeX for traceability (BLOCK-5)
mkdir -p "$STAGING/paper/sections"
cp "$PROJECT_ROOT"/docs/paper/NeurIPS_ready_version/sections/*.tex "$STAGING/paper/sections/"
cp "$PROJECT_ROOT/docs/paper/NeurIPS_ready_version/appendices_neurips.tex" "$STAGING/paper/"

echo "  Done."

# ── 3. Anonymize paths (FLAG-1) ────────────────────────────────────
step "Anonymizing paths..."

# Two-pass scrub: first replace exact project path, then replace remaining
# /home/samyak occurrences (truncated paths, downloads dir) without consuming
# beyond the username to avoid breaking JSON escape sequences
find "$STAGING" -type f \( -name "*.json" -o -name "*.py" -o -name "*.tex" \) -exec \
    sed -i 's|/home/samyak/Desktop/parbench_sam|/app|g; s|/home/samyak|/anonymized|g' {} +

# Final leak check
if grep -rqi "samyak\|jhaveri\|/home/samyak\|/Users/samyak" "$STAGING/" 2>/dev/null; then
    echo "FAIL: De-anonymization leak detected!"
    grep -rni "samyak\|jhaveri\|/home/samyak\|/Users/samyak" "$STAGING/" || true
    exit 1
fi
echo "  Anonymization check passed."

# ── 4. Count staged files ──────────────────────────────────────────
SPEC_COUNT=$(find "$STAGING/specs" -name "*.json" | wc -l)
RESULT_COUNT=$(find "$STAGING/results/evaluation" -name "*.json" | wc -l)
echo "  Specs: $SPEC_COUNT, Result JSONs: $RESULT_COUNT"

# ── 5. Docker build + run (skip if --dry-run) ──────────────────────
if [ "$DRY_RUN" = true ]; then
    step "Skipping Docker build (--dry-run)"
    step "Skipping Docker run (--dry-run)"
else
    step "Building Docker image..."
    docker build -t parbench-artifact "$STAGING"

    step "Running reproduce.sh in Docker..."
    mkdir -p "$STAGING/expected_outputs"
    docker run --rm -v "$STAGING/expected_outputs:/app/output" parbench-artifact ./reproduce.sh
fi

# ── 6. Create tarball ──────────────────────────────────────────────
step "Creating tarball..."
TARBALL="$PROJECT_ROOT/parbench-artifact-v1.tar.gz"
tar czf "$TARBALL" -C /tmp/parbench-artifact-staging parbench-artifact

SIZE=$(du -sh "$TARBALL" | cut -f1)
FILE_COUNT=$(tar tzf "$TARBALL" | wc -l)
echo ""
echo "=== Artifact Built ==="
echo "  File: $TARBALL"
echo "  Size: $SIZE"
echo "  Files: $FILE_COUNT"
echo "  Specs: $SPEC_COUNT"
echo "  Results: $RESULT_COUNT"
