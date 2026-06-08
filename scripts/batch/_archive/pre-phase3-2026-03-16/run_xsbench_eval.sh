#!/usr/bin/env bash
# run_xsbench_eval.sh — Session 8: XSBench multi-API LLM translation eval
#
# Runs 180 evaluations: 12 directions × 3 models × 5 augmentation levels (L0-L4)
# Self-launches in a detached tmux session so you can safely disconnect SSH.
#
# Usage:
#   bash scripts/batch/run_xsbench_eval.sh          # launch in tmux (default)
#   bash scripts/batch/run_xsbench_eval.sh --attach  # attach to running session
#
# Attach later with:
#   tmux attach -t xsbench_eval
#
# Models: claude-sonnet-4-6, gemini-2.5-flash-lite, groq-llama-3.3-70b-versatile
# Directions: 12 (all API permutations of cuda/omp/opencl/omp_target)
# Levels: L0-L4 (5 augmentation levels per direction×model cell)
# Total: 180 evaluation tasks
#
# Results written to: results/evaluation/{model}/xsbench-xsbench-*-to-xsbench-xsbench-*.json
# Log file: results/evaluation/xsbench_eval.log
# Done marker: results/evaluation/xsbench_eval_done.marker

set -euo pipefail

SESSION="xsbench_eval"
PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
LOGFILE="$PROJECT_ROOT/results/evaluation/xsbench_eval.log"

# ── If called with --attach, just attach to existing session ─────────────────
if [[ "${1:-}" == "--attach" ]]; then
    exec tmux attach -t "$SESSION"
fi

# ── If we are NOT already inside tmux, launch ourselves in a new session ──────
if [[ -z "${TMUX:-}" ]]; then
    # Kill stale session if it exists
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    # Ensure log directory exists
    mkdir -p "$(dirname "$LOGFILE")"

    # Create a new detached session that runs this very script (now inside tmux)
    tmux new-session -d -s "$SESSION" -x 220 -y 50 \
        "bash \"$0\" --inside-tmux 2>&1 | tee \"$LOGFILE\"; echo ''; echo '=== SESSION FINISHED ==='; read -r -p 'Press Enter to close...'"

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

# Trap SIGINT/SIGTERM for clean shutdown
trap 'echo ""; echo "INTERRUPTED at $(date -Iseconds)"; exit 130' INT TERM

echo "============================================================"
echo " XSBench Multi-API LLM Translation Eval — Session 8"
echo " $(date)"
echo " Repo: $PROJECT_ROOT"
echo "============================================================"
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────────
source "$PROJECT_ROOT/env_parbench/bin/activate"
echo "[env] Python: $(python3 --version)"
echo ""

# ── Pre-flight: verify API keys ───────────────────────────────────────────────
echo "--- Pre-flight checks ---"
MISSING_KEYS=""
for VAR in ANTHROPIC_API_KEY GEMINI_API_KEY GROQ_API_KEY; do
    if [ -z "${!VAR:-}" ]; then
        echo "  FATAL: $VAR is not set."
        MISSING_KEYS="$MISSING_KEYS $VAR"
    else
        echo "  OK: $VAR is set"
    fi
done

if [ -n "$MISSING_KEYS" ]; then
    echo ""
    echo "FATAL: Missing API keys:$MISSING_KEYS — aborting."
    exit 1
fi

# ── System info ───────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
NVC_VER=$(/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/compilers/bin/nvc --version 2>&1 | head -1 || echo "N/A")
echo "  GPU: $GPU_NAME"
echo "  nvc: $NVC_VER"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS="claude-sonnet-4-6 gemini-2.5-flash-lite groq-llama-3.3-70b-versatile"
AUGMENT_LEVELS="0 1 2 3 4"
MAX_RETRIES=2
TASKS_PER_DIR=15   # 3 models × 5 levels

# All 12 directions in tier order
ALL_DIRS=(
    # Tier 1: Primary (matches Rodinia primary direction)
    cuda-to-omp
    omp-to-cuda
    # Tier 2: Cross-API (OpenCL file asymmetry: 6↔1)
    cuda-to-opencl
    opencl-to-cuda
    opencl-to-omp
    omp-to-opencl
    # Tier 3: OMP-target case study (requires nvc for *-to-omp_target)
    cuda-to-omp_target
    omp-to-omp_target
    opencl-to-omp_target
    omp_target-to-cuda
    omp_target-to-omp
    omp_target-to-opencl
)
TOTAL_DIRS=${#ALL_DIRS[@]}

echo "Configuration:"
echo "  Models: $MODELS"
echo "  Augment levels: $AUGMENT_LEVELS"
echo "  Directions: $TOTAL_DIRS"
echo "  Tasks per direction: $TASKS_PER_DIR"
echo "  Total tasks: $((TOTAL_DIRS * TASKS_PER_DIR))"
echo "  Max retries per task: $MAX_RETRIES"
echo ""

# ── Helper: run one direction ─────────────────────────────────────────────────
run_direction() {
    local DIR="$1"
    local DIR_IDX="$2"
    local TASK_START="$3"

    echo "============================================================"
    echo " [Dir $DIR_IDX/$TOTAL_DIRS] $DIR"
    echo "  Tasks: ${TASK_START}-$((TASK_START + TASKS_PER_DIR - 1)) of $((TOTAL_DIRS * TASKS_PER_DIR))"
    echo "  Started: $(date -Iseconds)"
    echo "============================================================"

    python3 scripts/evaluation/run_eval_batch.py \
        --suite xsbench \
        --direction "$DIR" \
        --models $MODELS \
        --augment-levels $AUGMENT_LEVELS \
        --max-retries $MAX_RETRIES \
        --resume -v \
        --project-root "$PROJECT_ROOT"
    return $?
}

# ── Pass 1: run all 12 directions ─────────────────────────────────────────────
DIR_IDX=0
FAILED_DIRS=()
TASK_OFFSET=1
START_TIME=$(date +%s)

cd "$PROJECT_ROOT"

for DIR in "${ALL_DIRS[@]}"; do
    DIR_IDX=$((DIR_IDX + 1))

    set +e
    run_direction "$DIR" "$DIR_IDX" "$TASK_OFFSET"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "  ⚠ Direction $DIR exited with code $EXIT_CODE — will retry in Pass 2"
        FAILED_DIRS+=("$DIR")
    fi

    echo "  Completed: $(date -Iseconds)"
    echo ""
    TASK_OFFSET=$((TASK_OFFSET + TASKS_PER_DIR))
done

# ── Pass 2: retry any failed directions ───────────────────────────────────────
if [ "${#FAILED_DIRS[@]}" -gt 0 ]; then
    echo "============================================================"
    echo " RETRY PASS: ${#FAILED_DIRS[@]} direction(s) need retry"
    echo " Directions: ${FAILED_DIRS[*]}"
    echo "============================================================"
    echo ""

    RETRY_STILL_FAILED=()
    for DIR in "${FAILED_DIRS[@]}"; do
        echo "--- Retrying: $DIR ---"
        set +e
        python3 scripts/evaluation/run_eval_batch.py \
            --suite xsbench \
            --direction "$DIR" \
            --models $MODELS \
            --augment-levels $AUGMENT_LEVELS \
            --max-retries $MAX_RETRIES \
            --resume -v \
            --project-root "$PROJECT_ROOT"
        EXIT_CODE=$?
        set -e
        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "  ✗ Still failing: $DIR"
            RETRY_STILL_FAILED+=("$DIR")
        else
            echo "  ✓ Retry succeeded: $DIR"
        fi
        echo ""
    done
    FAILED_DIRS=("${RETRY_STILL_FAILED[@]}")
fi

# ── Completeness check ────────────────────────────────────────────────────────
echo "============================================================"
echo " COMPLETENESS CHECK"
echo "============================================================"

TOTAL_FILES=0
for MODEL in claude-sonnet-4-6 gemini-2.5-flash-lite groq-llama-3.3-70b-versatile; do
    COUNT=$(ls "$PROJECT_ROOT/results/evaluation/$MODEL"/xsbench-xsbench-*-to-xsbench-xsbench-*.json 2>/dev/null | wc -l || echo 0)
    echo "  $MODEL: $COUNT / 60"
    TOTAL_FILES=$((TOTAL_FILES + COUNT))
done
echo "  Total: $TOTAL_FILES / 180"
echo ""

# ── Regenerate analysis ───────────────────────────────────────────────────────
echo "============================================================"
echo " REGENERATING ANALYSIS"
echo "============================================================"

set +e
python3 scripts/evaluation/analyze_eval.py \
    --project-root "$PROJECT_ROOT" \
    --write-dashboard
ANALYZE_EXIT=$?
set -e

if [ "$ANALYZE_EXIT" -ne 0 ]; then
    echo "  ⚠ analyze_eval.py exited with code $ANALYZE_EXIT — analysis may be incomplete"
fi
echo ""

# ── Summary matrix ────────────────────────────────────────────────────────────
echo "============================================================"
echo " RESULTS SUMMARY"
echo "============================================================"
echo ""

python3 << 'PYEOF'
import json, glob, os, sys

BASE = "/home/samyak/Desktop/parbench_sam"
models = ['claude-sonnet-4-6', 'gemini-2.5-flash-lite', 'groq-llama-3.3-70b-versatile']
short  = ['claude', 'gemini', 'groq']
directions = [
    'cuda-to-omp',         'omp-to-cuda',
    'cuda-to-opencl',      'opencl-to-cuda',
    'opencl-to-omp',       'omp-to-opencl',
    'cuda-to-omp_target',  'omp-to-omp_target',   'opencl-to-omp_target',
    'omp_target-to-cuda',  'omp_target-to-omp',   'omp_target-to-opencl',
]

def load_result(model, src, tgt, tag=""):
    fp = f"{BASE}/results/evaluation/{model}/xsbench-xsbench-{src}-to-xsbench-xsbench-{tgt}{tag}.json"
    try:
        return json.loads(open(fp).read())
    except Exception:
        return None

# L0 status matrix
print("=== L0 Status Matrix ===")
header = f"{'Direction':30}  " + "  ".join(f"{s:>12}" for s in short)
print(header)
print("-" * len(header))
for d in directions:
    src, tgt = d.split("-to-")
    cells = []
    for m in models:
        r = load_result(m, src, tgt)
        cells.append(f"{r.get('overall_status', '?') if r else 'MISSING':>12}")
    print(f"{d:30}  " + "  ".join(cells))

# Aggregate per model (all levels)
print("\n=== Aggregate (L0-L4, 60 tasks per model) ===")
for m, s in zip(models, short):
    files = glob.glob(f"{BASE}/results/evaluation/{m}/xsbench-xsbench-*-to-xsbench-xsbench-*.json")
    results = []
    for f in files:
        try: results.append(json.loads(open(f).read()))
        except: pass
    passes = sum(1 for r in results if r.get("overall_status") == "PASS")
    total = len(results)
    pct = 100*passes/total if total else 0
    print(f"  {s:8} ({m}): {passes}/{total} PASS  ({pct:.1f}%)")

# Augmentation robustness table
print("\n=== Augmentation Robustness (PASS / 3 models per level) ===")
hdr = f"{'Direction':30}  {'L0':>6}  {'L1':>6}  {'L2':>6}  {'L3':>6}  {'L4':>6}"
print(hdr)
print("-" * len(hdr))
for d in directions:
    src, tgt = d.split("-to-")
    cols = []
    for level in range(5):
        tag = f"-L{level}" if level > 0 else ""
        n = 0
        for m in models:
            r = load_result(m, src, tgt, tag)
            if r and r.get("overall_status") == "PASS":
                n += 1
        cols.append(f"{n}/3")
    print(f"{d:30}  " + "  ".join(f"{c:>6}" for c in cols))

# Status breakdown
print("\n=== Status Breakdown (all 180 tasks) ===")
status_counts = {}
for m in models:
    files = glob.glob(f"{BASE}/results/evaluation/{m}/xsbench-xsbench-*-to-xsbench-xsbench-*.json")
    for f in files:
        try:
            r = json.loads(open(f).read())
            s = r.get("overall_status", "UNKNOWN")
            status_counts[s] = status_counts.get(s, 0) + 1
        except: pass
for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")
PYEOF

# ── Write done marker ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

DONE_MARKER="$PROJECT_ROOT/results/evaluation/xsbench_eval_done.marker"
{
    echo "COMPLETED $(date -Iseconds)"
    echo "FILES=$TOTAL_FILES/180"
    echo "ELAPSED=${ELAPSED}m"
    if [ "${#FAILED_DIRS[@]}" -gt 0 ]; then
        echo "FAILED_DIRS=${FAILED_DIRS[*]}"
    else
        echo "FAILED_DIRS=none"
    fi
} > "$DONE_MARKER"

# ── Final banner ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " XSBENCH EVAL COMPLETE — $(date)"
echo " Total results: $TOTAL_FILES / 180"
echo " Elapsed: ${ELAPSED} minutes"
if [ "${#FAILED_DIRS[@]}" -gt 0 ]; then
    echo " ⚠ Directions still failed after retry: ${FAILED_DIRS[*]}"
    echo " Re-run:  bash scripts/batch/run_xsbench_eval.sh"
    echo " (--resume will skip completed tasks)"
else
    echo " All directions completed."
fi
echo " Log:  results/evaluation/xsbench_eval.log"
echo " Done: results/evaluation/xsbench_eval_done.marker"
echo "============================================================"
