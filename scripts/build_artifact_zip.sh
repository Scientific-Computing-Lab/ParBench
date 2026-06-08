#!/usr/bin/env bash
set -euo pipefail
# Build anonymized artifact ZIP for NeurIPS supplementary upload

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR=$(mktemp -d)

echo "Building sanitized artifact in $ARTIFACT_DIR..."

# Copy only artifact-relevant files (no .git, no dev tools, no benchmark sources)
rsync -a \
  --exclude='.git' --exclude='.git*' --exclude='.gitmodules' \
  --exclude='env_parbench/' --exclude='.claude/' --exclude='docs/' \
  --exclude='node_modules/' --exclude='HeCBench-master/' \
  --exclude='rodinia/' --exclude='xsbench-src/' --exclude='rsbench-src/' \
  --exclude='mixbench-src/' --exclude='graphify-out/' --exclude='.planning/' \
  --exclude='*.pyc' --exclude='__pycache__/' \
  --exclude='Screenshot*' --exclude='*.png' --exclude='*.zip' \
  --exclude='HANDOFF*.md' --exclude='CLAUDE.md' --exclude='AGENTS.md' \
  --exclude='neeruipds*' \
  --exclude='.mcp.json' --exclude='meeting_notes/' \
  --exclude='analysis/' --exclude='.validation_passed' \
  --exclude='.codex_review_done' \
  --exclude='scripts/build_artifact*.sh' --exclude='scripts/survey/' \
  --exclude='scripts/archive/' --exclude='scripts/batch/' \
  "$PROJECT_ROOT/" "$ARTIFACT_DIR/parbench-artifact/"

# Sanitize config/paths.json
cat > "$ARTIFACT_DIR/parbench-artifact/config/paths.json" << 'EOF'
{
    "project_root": ".",
    "downloads_root": ".",
    "hecbench_root": "."
}
EOF

# Anonymize absolute paths in all text files (compiler output in result JSONs, script docstrings)
find "$ARTIFACT_DIR/parbench-artifact" -type f \( -name "*.json" -o -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.tex" -o -name "*.txt" \) -exec \
    sed -i 's|/home/samyak/Desktop/parbench_sam|.|g; s|/Users/samyakjhaveri/[^ "]*||g; s|/home/samyak[^ "]*||g; s|samyaknj@uci\.edu|anonymous@example.com|g' {} +

# Identity check (MUST pass)
if grep -rn "samyak\|jhaveri" "$ARTIFACT_DIR/parbench-artifact/" \
   --include="*.py" --include="*.json" --include="*.md" \
   --include="*.toml" --include="*.sh" --include="*.txt" \
   --include="*.tex" --include="*.cfg" 2>/dev/null; then
    echo "ERROR: Identity leak found! Aborting."
    rm -rf "$ARTIFACT_DIR"
    exit 1
fi

# Build ZIP (remove stale archive first — zip -r only adds/updates, never removes)
rm -f "$PROJECT_ROOT/parbench-artifact-neurips2026.zip"
cd "$ARTIFACT_DIR"
zip -r "$PROJECT_ROOT/parbench-artifact-neurips2026.zip" parbench-artifact/
SIZE=$(du -sh "$PROJECT_ROOT/parbench-artifact-neurips2026.zip" | cut -f1)
echo ""
echo "=== Artifact ZIP built: parbench-artifact-neurips2026.zip ($SIZE) ==="

# Size check (OpenReview limit: 100MB)
BYTES=$(stat -c%s "$PROJECT_ROOT/parbench-artifact-neurips2026.zip" 2>/dev/null || stat -f%z "$PROJECT_ROOT/parbench-artifact-neurips2026.zip")
if [ "$BYTES" -gt 104857600 ]; then
    echo "WARNING: ZIP exceeds 100MB ($SIZE)!"
    echo "Rebuilding without results/evaluation/ ..."
    rm -rf "$ARTIFACT_DIR/parbench-artifact/results/evaluation"
    mkdir -p "$ARTIFACT_DIR/parbench-artifact/results/evaluation"
    cat > "$ARTIFACT_DIR/parbench-artifact/results/evaluation/NOTE.md" << 'EOF'
# Evaluation Results

Full evaluation results (2,344 per-task JSONs across 3 models) are available at the
anonymous artifact repository: https://anonymous.4open.science/r/parbench-D5FD/

This ZIP contains only code and specifications due to OpenReview's 100MB size limit.
EOF
    cd "$ARTIFACT_DIR"
    rm -f "$PROJECT_ROOT/parbench-artifact-neurips2026.zip"
    zip -r "$PROJECT_ROOT/parbench-artifact-neurips2026.zip" parbench-artifact/
    SIZE=$(du -sh "$PROJECT_ROOT/parbench-artifact-neurips2026.zip" | cut -f1)
    echo "=== Rebuilt ZIP without results: $SIZE ==="
fi

rm -rf "$ARTIFACT_DIR"
