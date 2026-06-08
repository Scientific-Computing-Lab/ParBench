#!/usr/bin/env bash
# Re-run the 7 fixed CUDA kernels through the full harness pipeline
set -euo pipefail

PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
LOG_DIR="$PROJECT_ROOT/results/phase3/cuda_batch3_logs"

cd "$PROJECT_ROOT"
source env_parbench/bin/activate

KERNELS=(sobol stencil1d triad crc64 mandelbrot murmurhash3 feynman-kac)
TOTAL=${#KERNELS[@]}

echo "========================================"
echo "Re-running $TOTAL fixed CUDA kernels"
echo "Time: $(date -Iseconds)"
echo "========================================"

IDX=0
PASS_COUNT=0
FAIL_COUNT=0

for KERNEL in "${KERNELS[@]}"; do
  IDX=$((IDX + 1))
  SPEC="specs/hecbench-${KERNEL}-cuda.json"
  LOGFILE="$LOG_DIR/${KERNEL}.log"
  JSONFILE="$LOG_DIR/${KERNEL}.json"

  echo ""
  echo "[$IDX/$TOTAL] Re-running: $KERNEL ..."
  echo "  Time: $(date -Iseconds)"

  set +e
  OUTPUT=$(python -m harness --json -v verify "$SPEC" --config correctness 2>&1)
  EXIT_CODE=$?
  set -e

  echo "$OUTPUT" > "$LOGFILE"

  JSON_BLOCK=$(echo "$OUTPUT" | sed -n '/^{/,/^}/p')
  if [ -n "$JSON_BLOCK" ]; then
    echo "$JSON_BLOCK" > "$JSONFILE"
  fi

  SUMMARY_LINE=$(echo "$OUTPUT" | grep '^\[hecbench-' | head -1)
  VERIFY_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'VERIFY: \K\w+' || echo "UNKNOWN")

  if [ "$VERIFY_STATUS" = "PASS" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "  ✅ PASS — $SUMMARY_LINE"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "  ❌ $VERIFY_STATUS — $SUMMARY_LINE"
  fi
done

echo ""
echo "========================================"
echo "RERUN DONE: $(date -Iseconds)"
echo "  Total:  $TOTAL"
echo "  PASS:   $PASS_COUNT"
echo "  FAIL:   $FAIL_COUNT"
echo "========================================"

echo "RERUN_COMPLETED $(date -Iseconds) PASS=$PASS_COUNT FAIL=$FAIL_COUNT TOTAL=$TOTAL" > "$LOG_DIR/_rerun_done.marker"
