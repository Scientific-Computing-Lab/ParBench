#!/usr/bin/env bash
# Re-run only the specs that failed in the first batch run.

set -euo pipefail

PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
LOG_DIR="$PROJECT_ROOT/results/rodinia/logs"
SUMMARY="$LOG_DIR/_summary.tsv"

cd "$PROJECT_ROOT"
source env_parbench/bin/activate

# Specs that had fixable issues and should be retried
RETRY_SPECS=(
  # Build command fixes
  specs/rodinia-backprop-cuda.json
  specs/rodinia-cfd-opencl.json
  specs/rodinia-streamcluster-opencl.json
  # Executable name fixes
  specs/rodinia-backprop-opencl.json
  specs/rodinia-gaussian-opencl.json
  specs/rodinia-hotspot-opencl.json
  specs/rodinia-kmeans-opencl.json
  specs/rodinia-nn-opencl.json
  specs/rodinia-srad-opencl.json
  specs/rodinia-dwt2d-opencl.json
  specs/rodinia-nw-opencl.json
  specs/rodinia-lud-opencl.json
  specs/rodinia-lud-omp.json
  specs/rodinia-particlefilter-omp.json
  specs/rodinia-myocyte-omp.json
  # Run argument fixes
  specs/rodinia-hotspot-cuda.json
  specs/rodinia-srad-cuda.json
  specs/rodinia-srad-omp.json
  specs/rodinia-nw-omp.json
  specs/rodinia-heartwall-omp.json
  specs/rodinia-nn-cuda.json
  specs/rodinia-nn-omp.json
)

TOTAL=${#RETRY_SPECS[@]}

echo "========================================"
echo "Retry run: $TOTAL specs"
echo "Time: $(date -Iseconds)"
echo "========================================"

IDX=0
PASS_COUNT=0
FAIL_COUNT=0

for SPEC in "${RETRY_SPECS[@]}"; do
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

  JSON_BLOCK=$(echo "$OUTPUT" | sed -n '/^{/,/^}/p')
  if [ -n "$JSON_BLOCK" ]; then
    echo "$JSON_BLOCK" > "$JSONFILE"
  fi

  SUMMARY_LINE=$(echo "$OUTPUT" | grep '^\[rodinia-' | head -1)
  BUILD_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'BUILD: \K\w+' || echo "UNKNOWN")
  BUILD_TIME=$(echo "$SUMMARY_LINE" | grep -oP 'BUILD: \w+ \(\K[0-9.]+' || echo "0")
  RUN_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'RUN\([^)]+\): \K\w+' || echo "UNKNOWN")
  RUN_TIME=$(echo "$SUMMARY_LINE" | grep -oP 'RUN\([^)]+\): \w+ \(\K[0-9.]+' || echo "0")
  VERIFY_STATUS=$(echo "$SUMMARY_LINE" | grep -oP 'VERIFY: \K\w+' || echo "UNKNOWN")
  VERIFY_STRATEGY=$(echo "$SUMMARY_LINE" | grep -oP 'VERIFY: \w+ \(\K[^)]+' || echo "unknown")

  # Update summary TSV (replace existing line for this spec)
  ESCAPED=$(echo "$BASENAME" | sed 's/[\/&]/\\&/g')
  STDOUT_FIRST5=$(echo "$JSON_BLOCK" | python3 -c "
import sys, json
d = json.load(sys.stdin)
stdout = d.get('runs',{}).get('correctness',{}).get('stdout','')
lines = stdout.strip().split('\n')[:5]
print(' | '.join(lines))
" 2>/dev/null || echo "(no stdout)")

  RUN_EXIT=$(echo "$JSON_BLOCK" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('runs',{}).get('correctness',{}).get('exit_code','?'))" 2>/dev/null || echo "?")

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
echo "RETRY DONE: $(date -Iseconds)"
echo "  Retried: $TOTAL"
echo "  PASS:    $PASS_COUNT"
echo "  FAIL:    $FAIL_COUNT"
echo "========================================"
