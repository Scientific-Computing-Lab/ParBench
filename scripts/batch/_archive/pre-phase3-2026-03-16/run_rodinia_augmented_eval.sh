#!/usr/bin/env bash
# run_rodinia_augmented_eval.sh — Session 7: Rodinia cuda-to-omp augmented eval L1-L4
#
# Runs 204 evaluations: 17 kernels × 3 models × 4 augmentation levels (L1-L4)
# L0 baselines already exist — --resume skips them automatically.
# Self-launches in a detached tmux session so you can safely disconnect SSH.
#
# Usage:
#   bash scripts/batch/run_rodinia_augmented_eval.sh          # launch in tmux (default)
#   bash scripts/batch/run_rodinia_augmented_eval.sh --attach  # attach to running session
#
# Attach later with:
#   tmux attach -t rodinia_aug_eval
#
# Models: claude-sonnet-4-6, gemini-2.5-flash-lite, groq-llama-3.3-70b-versatile
# Direction: cuda-to-omp (single direction, 17 kernels)
# Levels: L1-L4 (4 augmentation levels; L0 pre-exists)
# Total: 204 new evaluation tasks (split into two passes: L1+L2, then L3+L4)
#
# Results written to: results/evaluation/{model}/rodinia-*-cuda-to-rodinia-*-omp-L{1..4}.json
# Log file: results/evaluation/s7_rodinia_aug.log
# Done marker: results/evaluation/rodinia_aug_eval_done.marker

set -euo pipefail

SESSION="rodinia_aug_eval"
PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
LOGFILE="$PROJECT_ROOT/results/evaluation/s7_rodinia_aug.log"

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
echo " Rodinia Augmented Eval L1-L4 (cuda-to-omp) — Session 7"
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
NVCC_VER=$(nvcc --version 2>&1 | grep "release" | head -1 || echo "N/A")
echo "  GPU: $GPU_NAME"
echo "  nvcc: $NVCC_VER"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS="claude-sonnet-4-6 gemini-2.5-flash-lite groq-llama-3.3-70b-versatile"
KERNELS="backprop bfs bptree cfd heartwall hotspot hotspot3d kmeans lavamd lud myocyte nn nw particlefilter pathfinder srad streamcluster"
MAX_RETRIES=2
EXPECTED_PER_LEVEL=17   # 17 kernels per model per level

echo "Configuration:"
echo "  Models: $MODELS"
echo "  Kernels: $KERNELS"
echo "  Direction: cuda-to-omp"
echo "  Augment levels: L1, L2, L3, L4 (L0 pre-exists, skipped by --resume)"
echo "  Max retries per task: $MAX_RETRIES"
echo "  Expected new files: $((EXPECTED_PER_LEVEL * 3 * 4)) (17 × 3 models × 4 levels)"
echo ""

# ── Pre-flight: verify L0 baselines exist ────────────────────────────────────
echo "--- L0 baseline check ---"
L0_OK=1
for MODEL in claude-sonnet-4-6 gemini-2.5-flash-lite groq-llama-3.3-70b-versatile; do
    COUNT=$(ls "$PROJECT_ROOT/results/evaluation/$MODEL"/rodinia-*-cuda-to-rodinia-*-omp.json 2>/dev/null | wc -l || echo 0)
    if [ "$COUNT" -lt 17 ]; then
        echo "  FATAL: $MODEL has only $COUNT L0 results (expected 17)"
        L0_OK=0
    else
        echo "  OK: $MODEL has $COUNT L0 results"
    fi
done

if [ "$L0_OK" -eq 0 ]; then
    echo ""
    echo "FATAL: L0 baselines incomplete — run Sessions 2+3 first."
    exit 1
fi
echo ""

START_TIME=$(date +%s)
cd "$PROJECT_ROOT"

# ── Pass 1: L1+L2 (102 tasks) ─────────────────────────────────────────────────
echo "============================================================"
echo " PASS 1: Augmentation Levels L1 + L2 (102 tasks)"
echo " Started: $(date -Iseconds)"
echo "============================================================"
echo ""

set +e
python3 scripts/evaluation/run_eval_batch.py \
    --suite rodinia \
    --direction cuda-to-omp \
    --models $MODELS \
    --kernels $KERNELS \
    --augment-levels 1 2 \
    --max-retries $MAX_RETRIES \
    --resume -v \
    --project-root "$PROJECT_ROOT"
PASS1_EXIT=$?
set -e

echo ""
echo "  Pass 1 finished: $(date -Iseconds)  (exit code: $PASS1_EXIT)"
echo ""

# ── Pass 2: L3+L4 (102 tasks) ─────────────────────────────────────────────────
echo "============================================================"
echo " PASS 2: Augmentation Levels L3 + L4 (102 tasks)"
echo " Started: $(date -Iseconds)"
echo "============================================================"
echo ""

set +e
python3 scripts/evaluation/run_eval_batch.py \
    --suite rodinia \
    --direction cuda-to-omp \
    --models $MODELS \
    --kernels $KERNELS \
    --augment-levels 3 4 \
    --max-retries $MAX_RETRIES \
    --resume -v \
    --project-root "$PROJECT_ROOT"
PASS2_EXIT=$?
set -e

echo ""
echo "  Pass 2 finished: $(date -Iseconds)  (exit code: $PASS2_EXIT)"
echo ""

# ── Pass 3: Retry if either pass failed ───────────────────────────────────────
if [ "$PASS1_EXIT" -ne 0 ] || [ "$PASS2_EXIT" -ne 0 ]; then
    echo "============================================================"
    echo " RETRY PASS: One or more passes exited non-zero — retrying"
    echo " (--resume skips completed tasks)"
    echo "============================================================"
    echo ""

    set +e
    if [ "$PASS1_EXIT" -ne 0 ]; then
        echo "--- Retrying L1+L2 ---"
        python3 scripts/evaluation/run_eval_batch.py \
            --suite rodinia \
            --direction cuda-to-omp \
            --models $MODELS \
            --kernels $KERNELS \
            --augment-levels 1 2 \
            --max-retries $MAX_RETRIES \
            --resume -v \
            --project-root "$PROJECT_ROOT"
        PASS1_EXIT=$?
    fi

    if [ "$PASS2_EXIT" -ne 0 ]; then
        echo "--- Retrying L3+L4 ---"
        python3 scripts/evaluation/run_eval_batch.py \
            --suite rodinia \
            --direction cuda-to-omp \
            --models $MODELS \
            --kernels $KERNELS \
            --augment-levels 3 4 \
            --max-retries $MAX_RETRIES \
            --resume -v \
            --project-root "$PROJECT_ROOT"
        PASS2_EXIT=$?
    fi
    set -e

    echo ""
    echo "  Retry finished: $(date -Iseconds)"
    echo "  L1+L2 exit: $PASS1_EXIT  |  L3+L4 exit: $PASS2_EXIT"
    echo ""
fi

# ── Completeness check ────────────────────────────────────────────────────────
echo "============================================================"
echo " COMPLETENESS CHECK"
echo "============================================================"

TOTAL_NEW_FILES=0
ALL_COMPLETE=1
for MODEL in claude-sonnet-4-6 gemini-2.5-flash-lite groq-llama-3.3-70b-versatile; do
    echo "  Model: $MODEL"
    for LVL in 1 2 3 4; do
        COUNT=$(ls "$PROJECT_ROOT/results/evaluation/$MODEL"/rodinia-*-cuda-to-rodinia-*-omp-L${LVL}.json 2>/dev/null | wc -l || echo 0)
        STATUS="OK"
        if [ "$COUNT" -lt "$EXPECTED_PER_LEVEL" ]; then
            STATUS="INCOMPLETE"
            ALL_COMPLETE=0
        fi
        echo "    L$LVL: $COUNT / $EXPECTED_PER_LEVEL  [$STATUS]"
        TOTAL_NEW_FILES=$((TOTAL_NEW_FILES + COUNT))
    done
done
echo "  Total new files: $TOTAL_NEW_FILES / 204"

if [ "$ALL_COMPLETE" -eq 0 ]; then
    echo "  ⚠ Some levels are incomplete — re-run this script to fill gaps (--resume skips existing)"
fi
echo ""

# ── Regenerate analysis ───────────────────────────────────────────────────────
echo "============================================================"
echo " REGENERATING ANALYSIS"
echo "============================================================"

set +e
python3 scripts/evaluation/analyze_eval.py \
    --project-root "$PROJECT_ROOT" \
    --write-dashboard \
    --show-gaps \
    --expected-models claude-sonnet-4-6 gemini-2.5-flash-lite groq-llama-3.3-70b-versatile \
    --expected-directions cuda-to-omp \
    --expected-levels 0 1 2 3 4
ANALYZE_EXIT=$?
set -e

if [ "$ANALYZE_EXIT" -ne 0 ]; then
    echo "  ⚠ analyze_eval.py exited with code $ANALYZE_EXIT — analysis may be incomplete"
fi
echo ""

# ── Rodinia-specific summary (paper-ready) ────────────────────────────────────
echo "============================================================"
echo " RODINIA AUGMENTATION RESULTS SUMMARY"
echo "============================================================"
echo ""

python3 << 'PYEOF'
import json, glob, os, sys, csv

BASE = "/home/samyak/Desktop/parbench_sam"
models = ['claude-sonnet-4-6', 'gemini-2.5-flash-lite', 'groq-llama-3.3-70b-versatile']
short  = ['claude', 'gemini', 'groq']
kernels = [
    'backprop', 'bfs', 'bptree', 'cfd', 'heartwall',
    'hotspot', 'hotspot3d', 'kmeans', 'lavamd', 'lud',
    'myocyte', 'nn', 'nw', 'particlefilter', 'pathfinder',
    'srad', 'streamcluster',
]

def load_result(model, kernel, level):
    tag = f"-L{level}" if level > 0 else ""
    fp = f"{BASE}/results/evaluation/{model}/rodinia-{kernel}-cuda-to-rodinia-{kernel}-omp{tag}.json"
    try:
        return json.loads(open(fp).read())
    except Exception:
        return None

# ─── Translation mode verification ──────────────────────────────────────────
print("=== Translation Mode Verification (all L1-L4 files) ===")
mode_ok = 0
mode_fail = 0
for m in models:
    for lvl in range(1, 5):
        files = glob.glob(f"{BASE}/results/evaluation/{m}/rodinia-*-cuda-to-rodinia-*-omp-L{lvl}.json")
        for f in files:
            try:
                r = json.loads(open(f).read())
                if r.get("translation_mode") == "kernel_centric":
                    mode_ok += 1
                else:
                    mode_fail += 1
                    print(f"  FAIL: {f} has translation_mode={r.get('translation_mode')}")
            except Exception as e:
                print(f"  ERROR reading {f}: {e}")
if mode_fail == 0:
    print(f"  All {mode_ok} new result files have translation_mode='kernel_centric'  ✓")
else:
    print(f"  WARNING: {mode_fail} files have wrong translation_mode!")
print()

# ─── Model × Level pass rate matrix ─────────────────────────────────────────
print("=== Model × Level Pass Rate Matrix (Rodinia cuda-to-omp, 17 kernels) ===")
header = f"{'Model':35}  {'L0':>8}  {'L1':>8}  {'L2':>8}  {'L3':>8}  {'L4':>8}"
print(header)
print("-" * len(header))
for m, s in zip(models, short):
    row = []
    for lvl in range(5):
        n_pass = 0
        n_total = 0
        for k in kernels:
            r = load_result(m, k, lvl)
            if r is not None:
                n_total += 1
                if r.get("overall_status") == "PASS":
                    n_pass += 1
        pct = f"{100*n_pass/n_total:.0f}%" if n_total > 0 else "N/A"
        row.append(f"{n_pass}/{n_total} ({pct})")
    print(f"{m:35}  {'  '.join(f'{c:>8}' for c in row)}")
print()

# ─── Augmentation fragility analysis ────────────────────────────────────────
print("=== Augmentation Fragility Analysis ===")
print("(Shows cases where L0 PASS flips to FAIL at any augmented level)")
print()

fragile = []    # (model, kernel, level, l0_status, ln_status)
healed  = []    # (model, kernel, level) — L0 FAIL but L{n} PASS
stable_pass = []
stable_fail = []

for m in models:
    for k in kernels:
        r0 = load_result(m, k, 0)
        if r0 is None:
            continue
        s0 = r0.get("overall_status", "MISSING")

        level_statuses = []
        for lvl in range(1, 5):
            rn = load_result(m, k, lvl)
            sn = rn.get("overall_status", "MISSING") if rn else "MISSING"
            level_statuses.append((lvl, sn))
            if s0 == "PASS" and sn != "PASS" and sn != "MISSING":
                fragile.append((m, k, lvl, s0, sn))
            elif s0 != "PASS" and sn == "PASS":
                healed.append((m, k, lvl, s0, sn))

        all_statuses = [s0] + [s for _, s in level_statuses]
        non_missing = [s for s in all_statuses if s != "MISSING"]
        if all(s == "PASS" for s in non_missing):
            stable_pass.append((m, k))
        elif all(s != "PASS" for s in non_missing):
            stable_fail.append((m, k))

if fragile:
    print(f"Fragile translations (L0 PASS → L{{N}} FAIL): {len(fragile)}")
    for m, k, lvl, s0, sn in sorted(fragile):
        ms = m.split('-')[0]
        print(f"  {ms:8} {k:20} L0={s0:10} → L{lvl}={sn}")
else:
    print("Fragile translations (L0 PASS → L{N} FAIL): 0  (all L0 PASSes held)")
print()

if healed:
    print(f"Healed translations (L0 FAIL → L{{N}} PASS): {len(healed)}")
    for m, k, lvl, s0, sn in sorted(healed):
        ms = m.split('-')[0]
        print(f"  {ms:8} {k:20} L0={s0:10} → L{lvl}={sn}")
    print()

print(f"Stable PASS at all levels: {len(stable_pass)}")
for m, k in sorted(stable_pass):
    ms = m.split('-')[0]
    print(f"  {ms:8} {k}")
print()

print(f"Stable FAIL at all levels: {len(stable_fail)}")
for m, k in sorted(stable_fail):
    ms = m.split('-')[0]
    print(f"  {ms:8} {k}")
print()

# ─── Complexity × Level breakdown ────────────────────────────────────────────
print("=== Pass Rate by Translation Complexity × Level (Rodinia cuda-to-omp) ===")

# Load complexity for the 17 kernels (cuda-to-omp direction)
complexity = {}
csv_path = f"{BASE}/results/evaluation/translation_complexity.csv"
try:
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if (row['suite'] == 'rodinia' and
                row['source_api'] == 'cuda' and
                row['target_api'] == 'omp'):
                complexity[row['kernel']] = row['complexity_class']
except Exception as e:
    print(f"  WARNING: Could not load complexity CSV: {e}")
    complexity = {}

classes = ['single_file', 'multi_to_single', 'multi_to_multi']
header = f"{'Complexity Class':22}  {'L0':>12}  {'L1':>12}  {'L2':>12}  {'L3':>12}  {'L4':>12}"
print(header)
print("-" * len(header))
for cls in classes:
    cls_kernels = [k for k in kernels if complexity.get(k) == cls]
    if not cls_kernels:
        continue
    row = []
    for lvl in range(5):
        n_pass = n_total = 0
        for m in models:
            for k in cls_kernels:
                r = load_result(m, k, lvl)
                if r is not None:
                    n_total += 1
                    if r.get("overall_status") == "PASS":
                        n_pass += 1
        pct = f"{100*n_pass/n_total:.0f}%" if n_total > 0 else "N/A"
        row.append(f"{n_pass}/{n_total} ({pct})")
    print(f"{cls:22}  {'  '.join(f'{c:>12}' for c in row)}")

# Also print total row
print("-" * len(header))
total_row = []
for lvl in range(5):
    n_pass = n_total = 0
    for m in models:
        for k in kernels:
            r = load_result(m, k, lvl)
            if r is not None:
                n_total += 1
                if r.get("overall_status") == "PASS":
                    n_pass += 1
    pct = f"{100*n_pass/n_total:.0f}%" if n_total > 0 else "N/A"
    total_row.append(f"{n_pass}/{n_total} ({pct})")
print(f"{'TOTAL (all classes)':22}  {'  '.join(f'{c:>12}' for c in total_row)}")
print()

# ─── Status breakdown ─────────────────────────────────────────────────────────
print("=== Status Breakdown (all 204 new L1-L4 tasks) ===")
status_counts = {}
for m in models:
    for lvl in range(1, 5):
        for k in kernels:
            r = load_result(m, k, lvl)
            if r is not None:
                s = r.get("overall_status", "UNKNOWN")
                status_counts[s] = status_counts.get(s, 0) + 1
for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")
total = sum(status_counts.values())
print(f"  Total files found: {total}")
PYEOF

# ── Write done marker ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

DONE_MARKER="$PROJECT_ROOT/results/evaluation/rodinia_aug_eval_done.marker"
{
    echo "COMPLETED $(date -Iseconds)"
    echo "TOTAL_NEW_FILES=$TOTAL_NEW_FILES/204"
    echo "ELAPSED=${ELAPSED}m"
    echo "PASS1_EXIT=$PASS1_EXIT"
    echo "PASS2_EXIT=$PASS2_EXIT"
    if [ "$ALL_COMPLETE" -eq 1 ]; then
        echo "STATUS=COMPLETE"
    else
        echo "STATUS=INCOMPLETE"
    fi
} > "$DONE_MARKER"

# ── Final banner ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " RODINIA AUGMENTED EVAL COMPLETE — $(date)"
echo " New result files: $TOTAL_NEW_FILES / 204"
echo " Elapsed: ${ELAPSED} minutes"
if [ "$ALL_COMPLETE" -eq 0 ]; then
    echo " ⚠ Some levels are incomplete."
    echo " Re-run:  bash scripts/batch/run_rodinia_augmented_eval.sh"
    echo " (--resume will skip completed tasks)"
else
    echo " All 204 tasks completed."
fi
echo " Log:  results/evaluation/s7_rodinia_aug.log"
echo " Done: results/evaluation/rodinia_aug_eval_done.marker"
echo "============================================================"
