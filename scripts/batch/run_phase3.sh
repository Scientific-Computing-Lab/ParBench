#!/usr/bin/env bash
# Phase 3 evaluation — canonical + ablation for NeurIPS 2026
#
# Usage:
#   bash scripts/batch/run_phase3.sh [canonical|derive|ablation] [MODEL]
#
# Phase A (canonical):  pass@3, L0, temp=0.7, thinking=ON, self-repair=OFF
# Phase B (derive):     derive_l0_passers.py → .planning/eval-selections/l0_passers_*.json
# Phase C (ablation):   pass@1, L1-L4, temp=0.7, thinking=ON, --task-list from Phase B
#
# Defaults: canonical / together-qwen-3.5-397b-a17b
#
# Examples:
#   bash scripts/batch/run_phase3.sh canonical
#   bash scripts/batch/run_phase3.sh canonical azure-gpt-5.4
#   bash scripts/batch/run_phase3.sh derive
#   bash scripts/batch/run_phase3.sh ablation

set -euo pipefail

PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
PHASE="${1:-canonical}"
MODEL="${2:-together-qwen-3.5-397b-a17b}"
INSIDE_TMUX="${3:-}"

MODEL_SHORT=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9_-]/_/g' | cut -c1-30)
SESSION="phase3-${PHASE}"
LOGFILE="$PROJECT_ROOT/results/evaluation/${MODEL_SHORT}_${PHASE}.log"
DONE_MARKER="$PROJECT_ROOT/results/evaluation/${MODEL_SHORT}_${PHASE}_done.marker"

# ── API key check ────────────────────────────────────────────────────────────
check_api_key() {
    case "$MODEL" in
        together-*)
            [[ -n "${TOGETHER_API_KEY:-}" ]] \
                || { echo "FATAL: TOGETHER_API_KEY is not set."; exit 1; }
            echo "  OK: TOGETHER_API_KEY set (${#TOGETHER_API_KEY} chars)"
            ;;
        azure-*codex*)
            [[ -n "${AZURE_OPENAI_API_KEY_CODEX:-}" ]] \
                || { echo "FATAL: AZURE_OPENAI_API_KEY_CODEX is not set."; exit 1; }
            echo "  OK: AZURE_OPENAI_API_KEY_CODEX set"
            ;;
        azure-*)
            [[ -n "${AZURE_OPENAI_API_KEY:-}" ]] \
                || { echo "FATAL: AZURE_OPENAI_API_KEY is not set."; exit 1; }
            echo "  OK: AZURE_OPENAI_API_KEY set"
            ;;
        claude-*)
            [[ -n "${ANTHROPIC_API_KEY:-}" ]] \
                || { echo "FATAL: ANTHROPIC_API_KEY is not set."; exit 1; }
            echo "  OK: ANTHROPIC_API_KEY set"
            ;;
        *)
            echo "  WARNING: Unknown model prefix — API key not verified."
            ;;
    esac
}

# ── Self-launch in tmux ──────────────────────────────────────────────────────
if [[ "$INSIDE_TMUX" != "--inside-tmux" ]]; then
    echo "========================================================"
    echo " Phase 3 ${PHASE^^} — Pre-launch checks"
    echo " Model: $MODEL"
    echo "========================================================"

    # API key
    check_api_key

    # Rodinia submodule
    if [[ -z "$(ls -A "$PROJECT_ROOT/rodinia/rodinia-src/cuda" 2>/dev/null)" ]]; then
        echo "FATAL: Rodinia submodule empty. Run: git submodule update --init"
        exit 1
    fi
    echo "  OK: Rodinia submodule populated"

    # Not in a worktree
    if [[ -f "$PROJECT_ROOT/.git" ]]; then
        echo "FATAL: Running inside a git worktree — submodules are empty there."
        exit 1
    fi
    echo "  OK: Not in a worktree"

    # GPU (warn only)
    GPU=$(nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    echo "  GPU: $GPU"

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    mkdir -p "$(dirname "$LOGFILE")"

    tmux new-session -d -s "$SESSION" -x 220 -y 50 \
        "bash \"$0\" \"$PHASE\" \"$MODEL\" --inside-tmux 2>&1 | tee \"$LOGFILE\"; \
         echo ''; echo '=== SESSION FINISHED ==='; \
         read -r -p 'Press Enter to close...'"

    echo ""
    echo "Launched tmux session: $SESSION"
    echo "  Attach:  tmux attach -t $SESSION"
    echo "  Log:     $LOGFILE"
    echo "  Done:    $DONE_MARKER"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# Inside tmux
# ════════════════════════════════════════════════════════════════════════════
trap 'echo ""; echo "INTERRUPTED $(date -Iseconds)"; exit 130' INT TERM

source ~/.bashrc 2>/dev/null || true
source "$PROJECT_ROOT/env_parbench/bin/activate"
cd "$PROJECT_ROOT"

echo "========================================================"
echo " Phase 3 ${PHASE^^} — $(date)"
echo " Model:   $MODEL"
echo " Repo:    $PROJECT_ROOT"
echo " Python:  $(python3 --version)"
echo "========================================================"
echo ""

# ── Phase B: derive ──────────────────────────────────────────────────────────
if [[ "$PHASE" == "derive" ]]; then
    CANONICAL_DIR="$PROJECT_ROOT/results/evaluation/$MODEL"
    SLUG=$(echo "$MODEL_SHORT" | tr '-' '_')
    OUT_PATH="$PROJECT_ROOT/.planning/eval-selections/l0_passers_${SLUG}.json"
    mkdir -p "$PROJECT_ROOT/.planning/eval-selections"

    echo "Deriving L0 passer set..."
    echo "  Input:  $CANONICAL_DIR"
    echo "  Output: $OUT_PATH"
    echo ""

    python3 -m scripts.evaluation.derive_l0_passers \
        --canonical-dir "$CANONICAL_DIR" \
        --model "$MODEL" \
        --out "$OUT_PATH"

    COUNT=$(python3 -c "import json; print(len(json.load(open('$OUT_PATH'))))")
    echo ""
    echo "L0 passers: $COUNT cells (out of ~522 possible)"
    echo "$(date -Iseconds) DONE derive $COUNT cells" > "$DONE_MARKER"
    exit 0
fi

# ── Phase config ─────────────────────────────────────────────────────────────
if [[ "$PHASE" == "canonical" ]]; then
    AUGMENT_LEVELS="0"
    TEMPERATURE=0.7
    NUM_SAMPLES=3
    MAX_RETRIES=1
    TASK_LIST_ARGS=()
    TOTAL_BATCHES=30  # 6 rodinia + 6 xsbench + 6 rsbench + 6 mixbench + 6 hecbench
elif [[ "$PHASE" == "ablation" ]]; then
    AUGMENT_LEVELS="1 2 3 4"
    TEMPERATURE=0.7
    NUM_SAMPLES=1
    MAX_RETRIES=1
    SLUG=$(echo "$MODEL_SHORT" | tr '-' '_')
    PASSERS_FILE="$PROJECT_ROOT/.planning/eval-selections/l0_passers_${SLUG}.json"
    [[ -f "$PASSERS_FILE" ]] \
        || { echo "FATAL: $PASSERS_FILE not found. Run Phase B (derive) first."; exit 1; }
    TASK_LIST_ARGS=(--task-list "$PASSERS_FILE")
    PASSERS_COUNT=$(python3 -c "import json; print(len(json.load(open('$PASSERS_FILE'))))")
    TOTAL_BATCHES=30
    echo "Ablation kernel pool: $PASSERS_COUNT L0-passer cells"
else
    echo "FATAL: Unknown phase '$PHASE'. Use: canonical | derive | ablation"
    exit 1
fi

# KNOWN_FAIL specs — excluded at run time to avoid wasting API calls
KNOWN_FAIL_SPECS=(
    rodinia-kmeans-cuda
    rodinia-mummergpu-cuda
    rodinia-mummergpu-omp
    rodinia-hybridsort-cuda
    rodinia-nn-opencl
    rodinia-kmeans-opencl
    rodinia-backprop-opencl
    hecbench-stencil1d-omp_target
    hecbench-scan-omp_target
)

echo "Configuration:"
echo "  Phase:        $PHASE"
echo "  Augment:      $AUGMENT_LEVELS"
echo "  Temperature:  $TEMPERATURE"
echo "  Samples:      $NUM_SAMPLES"
echo "  Thinking:     ON"
echo "  Max retries:  $MAX_RETRIES (self-repair OFF)"
[[ "$PHASE" == "ablation" ]] && echo "  Task list:    ${TASK_LIST_ARGS[*]}"
echo ""
echo "Coverage:"
echo "  Rodinia:  6 standard dirs (KNOWN_FAIL included, excluded in analysis)"
echo "  XSBench:  6 standard dirs"
echo "  RSBench:  6 standard dirs"
echo "  mixbench: 6 standard dirs"
echo "  HeCBench: 6 dirs (cuda↔omp: 5 kernels, cuda↔omp_target: 10 kernels, omp↔omp_target: 5 kernels)"
echo "  NOTE: HeCBench has no opencl specs; omp_target promoted to standard direction"
echo ""

# ── HeCBench curated kernel groups ───────────────────────────────────────────
# HeCBench has 60+ kernels in manifest — MUST filter with --kernels.
# 10 curated kernels, grouped by API availability:
#   5 have cuda+omp+omp_target, 5 have cuda+omp_target only.
HECBENCH_OMP=(stencil1d heat2d floydwarshall scan iso2dfd)
HECBENCH_OMP_TARGET=(stencil1d heat2d floydwarshall scan iso2dfd convolution1d jacobi md nqueen page-rank)
HECBENCH_OMP_X_OMP_TARGET=(stencil1d heat2d floydwarshall scan iso2dfd)

# ── Batch runner ─────────────────────────────────────────────────────────────
FAILED_BATCHES=()
BATCH_IDX=0
START_TIME=$(date +%s)

run_batch() {
    local SUITE="$1" DIR="$2"
    shift 2
    local KERNEL_ARGS=("$@")
    BATCH_IDX=$((BATCH_IDX + 1))

    echo "============================================================"
    echo " [$BATCH_IDX/$TOTAL_BATCHES] $SUITE $DIR  $(date -Iseconds)"
    [[ ${#KERNEL_ARGS[@]} -gt 0 ]] && echo "  Kernels: ${KERNEL_ARGS[*]}"
    echo "============================================================"

    local CMD=(python3 scripts/evaluation/run_eval_batch.py
        --direction "$DIR"
        --models "$MODEL"
        --augment-levels $AUGMENT_LEVELS
        --temperature $TEMPERATURE
        --num-samples $NUM_SAMPLES
        --max-retries $MAX_RETRIES
        --thinking on
        --resume -v
        --project-root "$PROJECT_ROOT")
    if [[ ${#TASK_LIST_ARGS[@]} -gt 0 ]]; then
        CMD+=("${TASK_LIST_ARGS[@]}")
    else
        CMD+=(--suite "$SUITE")
        [[ ${#KERNEL_ARGS[@]} -gt 0 ]] && CMD+=(--kernels "${KERNEL_ARGS[@]}")
        CMD+=(--excluded-specs "${KNOWN_FAIL_SPECS[@]}")
    fi

    set +e
    "${CMD[@]}"
    local RC=$?
    set -e

    if [[ $RC -ne 0 ]]; then
        echo "  WARNING: exited $RC — will retry"
        FAILED_BATCHES+=("$SUITE|$DIR|${KERNEL_ARGS[*]}")
    fi
    echo "  Done: $(date -Iseconds)"
    echo ""
}

# ── Rodinia: 6 standard directions ───────────────────────────────────────────
for DIR in cuda-to-omp omp-to-cuda cuda-to-opencl opencl-to-cuda omp-to-opencl opencl-to-omp; do
    run_batch rodinia "$DIR"
done

# ── Small suites: 6 standard directions each ─────────────────────────────────
for SUITE in xsbench rsbench mixbench; do
    for DIR in cuda-to-omp omp-to-cuda cuda-to-opencl opencl-to-cuda omp-to-opencl opencl-to-omp; do
        run_batch "$SUITE" "$DIR"
    done
done

# ── HeCBench: cuda↔omp (5 kernels) ──────────────────────────────────────────
for DIR in cuda-to-omp omp-to-cuda; do
    run_batch hecbench "$DIR" "${HECBENCH_OMP[@]}"
done

# ── HeCBench: cuda↔omp_target (all 10 curated) ─────────────────────────────
for DIR in cuda-to-omp_target omp_target-to-cuda; do
    run_batch hecbench "$DIR" "${HECBENCH_OMP_TARGET[@]}"
done

# ── HeCBench: omp↔omp_target (5 kernels with both) ─────────────────────────
for DIR in omp-to-omp_target omp_target-to-omp; do
    run_batch hecbench "$DIR" "${HECBENCH_OMP_X_OMP_TARGET[@]}"
done

# ── Retry pass ───────────────────────────────────────────────────────────────
if [[ ${#FAILED_BATCHES[@]} -gt 0 ]]; then
    echo "============================================================"
    echo " RETRY PASS: ${#FAILED_BATCHES[@]} batch(es)"
    echo "============================================================"
    STILL_FAILED=()
    for ENTRY in "${FAILED_BATCHES[@]}"; do
        IFS='|' read -r SUITE DIR KERNELS <<< "$ENTRY"
        read -r -a KERNEL_ARGS <<< "$KERNELS"
        echo "--- Retrying: $SUITE $DIR ---"
        set +e
        if [[ ${#TASK_LIST_ARGS[@]} -gt 0 ]]; then
            # Ablation phase: --task-list is mutually exclusive with --suite/--kernels
            RETRY=(python3 scripts/evaluation/run_eval_batch.py
                --direction "$DIR"
                --models "$MODEL"
                --augment-levels $AUGMENT_LEVELS
                --temperature $TEMPERATURE
                --num-samples $NUM_SAMPLES
                --max-retries $MAX_RETRIES
                --thinking on
                --resume -v
                --project-root "$PROJECT_ROOT"
                "${TASK_LIST_ARGS[@]}")
        else
            # Canonical phase: --suite is required, no --task-list
            RETRY=(python3 scripts/evaluation/run_eval_batch.py
                --suite "$SUITE" --direction "$DIR"
                --models "$MODEL"
                --augment-levels $AUGMENT_LEVELS
                --temperature $TEMPERATURE
                --num-samples $NUM_SAMPLES
                --max-retries $MAX_RETRIES
                --thinking on
                --resume -v
                --project-root "$PROJECT_ROOT")
            [[ ${#KERNEL_ARGS[@]} -gt 0 ]] && RETRY+=(--kernels "${KERNEL_ARGS[@]}")
            RETRY+=(--excluded-specs "${KNOWN_FAIL_SPECS[@]}")
        fi
        "${RETRY[@]}"
        [[ $? -ne 0 ]] && STILL_FAILED+=("$ENTRY") || echo "  OK: retry succeeded"
        set -e
        echo ""
    done
    FAILED_BATCHES=("${STILL_FAILED[@]}")
fi

# ── Post-run analysis ─────────────────────────────────────────────────────────
echo "============================================================"
echo " POST-RUN ANALYSIS"
echo "============================================================"
set +e
python3 scripts/evaluation/analyze_eval.py \
    --project-root "$PROJECT_ROOT" \
    --write-dashboard
ANALYZE_RC=$?
set -e
[[ $ANALYZE_RC -ne 0 ]] && echo "  WARNING: analyze_eval.py exited $ANALYZE_RC"

# ── Result summary ────────────────────────────────────────────────────────────
TOTAL_FILES=$(ls "$PROJECT_ROOT/results/evaluation/$MODEL/"*.json 2>/dev/null | wc -l || echo 0)
ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

python3 - "$MODEL" "$PROJECT_ROOT" << 'PYEOF'
import json, glob, sys
model, root = sys.argv[1], sys.argv[2]
files = glob.glob(f"{root}/results/evaluation/{model}/*.json")
counts = {}
suite_counts = {}
for f in files:
    try:
        d = json.loads(open(f).read())
        s = d.get("overall_status", "?")
        src = d.get("source_spec", "")
        suite = src.split("-")[0] if src else "?"
    except Exception:
        s, suite = "ERROR", "?"
    counts[s] = counts.get(s, 0) + 1
    if suite not in suite_counts:
        suite_counts[suite] = {"total": 0, "pass": 0}
    suite_counts[suite]["total"] += 1
    if s == "PASS":
        suite_counts[suite]["pass"] += 1
total = sum(counts.values())
passed = counts.get("PASS", 0)
print(f"Results: {total} | PASS: {passed}/{total} ({100*passed/total:.1f}%)" if total else "No results")
print("Status breakdown:")
for s, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")
print("Per-suite:")
for suite in sorted(suite_counts):
    sc = suite_counts[suite]
    pct = 100*sc['pass']/sc['total'] if sc['total'] else 0
    print(f"  {suite}: {sc['pass']}/{sc['total']} ({pct:.1f}%)")
PYEOF

# ── Done marker ───────────────────────────────────────────────────────────────
{
    echo "COMPLETED $(date -Iseconds)"
    echo "MODEL=$MODEL"
    echo "PHASE=$PHASE"
    echo "FILES=$TOTAL_FILES"
    echo "ELAPSED=${ELAPSED}m"
    echo "FAILED=${#FAILED_BATCHES[@]}"
} > "$DONE_MARKER"

echo ""
echo "============================================================"
echo " ${PHASE^^} COMPLETE — $(date)"
echo " Model:    $MODEL"
echo " Results:  $TOTAL_FILES"
echo " Elapsed:  ${ELAPSED}m"
[[ ${#FAILED_BATCHES[@]} -gt 0 ]] && echo " WARNING:  ${#FAILED_BATCHES[@]} batches still failing"
echo " Log:      $LOGFILE"
echo " Done:     $DONE_MARKER"
echo ""
if [[ "$PHASE" == "canonical" ]]; then
    echo " Next step — Phase B (derive):"
    echo "   bash scripts/batch/run_phase3.sh derive $MODEL"
fi
if [[ "$PHASE" == "derive" ]]; then
    echo " Next step — Phase C (ablation):"
    echo "   bash scripts/batch/run_phase3.sh ablation $MODEL"
fi
echo "============================================================"
