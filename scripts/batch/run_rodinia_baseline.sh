#!/usr/bin/env bash
# run_rodinia_baseline.sh
#
# Runs the full Rodinia baseline pipeline inside a tmux session so you can
# safely disconnect from the remote machine while it runs.
#
# Usage:
#   bash run_rodinia_baseline.sh          # launch in tmux (or reattach)
#   bash run_rodinia_baseline.sh --attach # attach to existing session
#
# Attach later with:
#   tmux attach -t rodinia_baseline
#
# The script:
#   1. Re-runs all 60 Rodinia specs through the harness (results/rodinia/logs/)
#   2. Populates baseline_results in each specs/rodinia-*.json
#   3. Commits and pushes the updated specs to GitHub

set -euo pipefail

SESSION="rodinia_baseline"
REPO="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$REPO/results/rodinia/baseline_pipeline.log"

# ── If called with --attach, just attach to existing session ─────────────────
if [[ "${1:-}" == "--attach" ]]; then
    exec tmux attach -t "$SESSION"
fi

# ── If we are NOT already inside tmux, launch ourselves in a new session ──────
if [[ -z "${TMUX:-}" ]]; then
    # Kill stale session if it exists
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    # Create a new detached session that runs this very script (now inside tmux)
    tmux new-session -d -s "$SESSION" -x 220 -y 50 \
        "bash \"$0\" --inside-tmux 2>&1 | tee \"$LOGFILE\"; echo ''; echo '=== PIPELINE FINISHED ==='; read -r -p 'Press Enter to close...'"

    echo ""
    echo "Launched tmux session '$SESSION'."
    echo ""
    echo "  Attach with:  tmux attach -t $SESSION"
    echo "  Log file:     $LOGFILE"
    echo ""
    echo "You can safely close this SSH connection — the session will keep running."
    exit 0
fi

# ── Inside tmux: do the real work ─────────────────────────────────────────────

echo "============================================================"
echo " Rodinia Baseline Pipeline"
echo " $(date)"
echo " Repo: $REPO"
echo "============================================================"
echo ""

# Activate venv
source "$REPO/env_parbench/bin/activate"
echo "[venv] activated: $(which python3)"
echo ""

# ── Step 1: Re-run the full Rodinia batch ─────────────────────────────────────
echo "------------------------------------------------------------"
echo "[1/3] Running Rodinia batch (60 specs)..."
echo "      Results will be written to results/rodinia/logs/"
echo "------------------------------------------------------------"

bash "$REPO/scripts/run_rodinia_batch.sh"

echo ""
echo "[1/3] Batch complete."
echo ""

# ── Step 2: Populate baseline_results in all specs ───────────────────────────
echo "------------------------------------------------------------"
echo "[2/3] Populating baseline_results in specs/rodinia-*.json..."
echo "------------------------------------------------------------"

python3 "$REPO/scripts/populate_baseline_results.py"

echo ""
echo "[2/3] Specs updated."
echo ""

# ── Step 3: Commit and push ───────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "[3/3] Committing and pushing updated specs..."
echo "------------------------------------------------------------"

cd "$REPO"

# Stage all updated Rodinia specs and the populate script
git add specs/rodinia-*.json
git add scripts/populate_baseline_results.py

# Only commit if there are staged changes
if git diff --cached --quiet; then
    echo "Nothing to commit — specs already up to date."
else
    git commit -m "$(cat <<'EOF'
Populate baseline_results in all Rodinia specs (RTX 4070, CUDA 12.3)

Runs all 60 Rodinia specs through the harness (correctness config) and
writes verified build/run/verify results into each spec's baseline_results
field. Reference platform: rtx4070-linux-x86_64 (NVIDIA GeForce RTX 4070,
AMD Ryzen 9 7900X, Linux x86_64, HPC SDK 24.3 / CUDA 12.3).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
    git push origin main
    echo ""
    echo "[3/3] Pushed to GitHub."
fi

echo ""
echo "============================================================"
echo " ALL DONE — $(date)"
echo " Check results/rodinia/baseline_pipeline.log for full output"
echo "============================================================"
