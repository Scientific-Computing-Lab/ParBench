#!/usr/bin/env bash
# Re-run all recently fixed specs

set -uo pipefail

PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
LOG_DIR="$PROJECT_ROOT/results/rodinia/logs"

cd "$PROJECT_ROOT"
source env_parbench/bin/activate

SPECS=(
  # CUDA arch fixes (via spec build command override or Makefile patches)
  specs/rodinia-bptree-cuda.json
  specs/rodinia-cfd-cuda.json
  specs/rodinia-dwt2d-cuda.json
  specs/rodinia-huffman-cuda.json
  specs/rodinia-hybridsort-cuda.json
  specs/rodinia-lavamd-cuda.json
  specs/rodinia-lud-cuda.json
  specs/rodinia-hotspot3d-cuda.json
  specs/rodinia-particlefilter-cuda.json
  # OpenCL Makefile fixes
  specs/rodinia-heartwall-opencl.json
  specs/rodinia-hybridsort-opencl.json
  specs/rodinia-lavamd-opencl.json
  specs/rodinia-myocyte-opencl.json
  specs/rodinia-hotspot3d-opencl.json
)

TOTAL=${#SPECS[@]}
IDX=0
PASS_COUNT=0
FAIL_COUNT=0

echo "========================================"
echo "Fixable specs: $TOTAL"
echo "Time: $(date -Iseconds)"
echo "========================================"

for SPEC in "${SPECS[@]}"; do
  IDX=$((IDX + 1))
  BASENAME=$(basename "$SPEC" .json)
  LOGFILE="$LOG_DIR/${BASENAME}.log"
  JSONFILE="$LOG_DIR/${BASENAME}.json"

  echo ""
  echo "[$IDX/$TOTAL] $BASENAME"

  set +e
  OUTPUT=$(python3 -m harness --json -v verify "$SPEC" --config correctness 2>&1)
  EXIT_CODE=$?
  set -e

  echo "$OUTPUT" > "$LOGFILE"

  JSON_BLOCK=$(echo "$OUTPUT" | python3 -c "
import sys
lines = sys.stdin.read()
start = lines.find('{')
end = lines.rfind('}')
if start >= 0 and end >= 0:
    print(lines[start:end+1])
" 2>/dev/null)
  if [ -n "$JSON_BLOCK" ]; then
    echo "$JSON_BLOCK" > "$JSONFILE"
  fi

  SUMMARY_LINE=$(echo "$OUTPUT" | grep "^\[rodinia-" | head -1)
  VERIFY_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'VERIFY: \K\w+' || echo "UNKNOWN")
  BUILD_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'BUILD: \K\w+' || echo "UNKNOWN")
  RUN_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'RUN\([^)]+\): \K\w+' || echo "UNKNOWN")
  BUILD_TIME=$(echo "$SUMMARY_LINE" | grep -oP 'BUILD: \w+ \(\K[0-9.]+' || echo "0")
  RUN_TIME=$(echo "$SUMMARY_LINE" | grep -oP 'RUN\([^)]+\): \w+ \(\K[0-9.]+' || echo "0")
  VERIFY_STRATEGY=$(echo "$SUMMARY_LINE" | grep -oP 'VERIFY: \w+ \(\K[^)]+' || echo "unknown")

  if [ "$VERIFY_STATUS" = "PASS" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "  PASS (build=${BUILD_TIME}s, run=${RUN_TIME}s, verify=${VERIFY_STRATEGY})"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "  FAIL: ${VERIFY_STATUS} (build=${BUILD_STATUS}, run=${RUN_STATUS})"
  fi
done

echo ""
echo "========================================"
echo "DONE: $(date -Iseconds)"
echo "  Ran: $TOTAL"
echo "  PASS: $PASS_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo "========================================"
