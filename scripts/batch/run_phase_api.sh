#!/usr/bin/env bash
# Augmentation stream: Phase 3 smoke → Phase 4 Rodinia → Phase 5 all specs
# Usage: run_phase_api.sh <cuda|omp|opencl> [smoke_spec]
#
# Default smoke specs:
#   cuda    → rodinia-bfs-cuda
#   omp     → rodinia-hotspot-omp
#   opencl  → rodinia-bfs-opencl
set -e

API="${1:?Usage: run_phase_api.sh <cuda|omp|opencl> [smoke_spec]}"
API_UPPER="$(echo "$API" | tr '[:lower:]' '[:upper:]')"

# Default smoke spec per API
case "$API" in
    omp)     DEFAULT_SMOKE="rodinia-hotspot-omp" ;;
    *)       DEFAULT_SMOKE="rodinia-bfs-${API}" ;;
esac
SMOKE_SPEC="${2:-$DEFAULT_SMOKE}"

PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
cd "$PROJECT_ROOT"
source env_parbench/bin/activate

echo "========================================================"
echo "  ${API_UPPER} STREAM START: $(date)"
echo "========================================================"

echo ""
echo "===== Phase 3 ${API_UPPER}: ${SMOKE_SPEC} × L1,L2,L4 ====="
python3 scripts/augmentation/run_augment_batch.py \
    "specs/${SMOKE_SPEC}.json" \
    --levels 1 2 4 --seed 42 \
    --out "results/augmentation/phase3_${API}" \
    --title "Phase 3 Smoke Test: ${API_UPPER} (${SMOKE_SPEC})"

echo ""
echo "===== Phase 4 ${API_UPPER}: All Rodinia ${API_UPPER} specs × L2 ====="
python3 scripts/augmentation/run_augment_batch.py \
    "specs/rodinia-*-${API}.json" \
    --levels 2 --seed 42 \
    --out "results/augmentation/phase4_${API}" \
    --title "Phase 4: All Rodinia ${API_UPPER} Specs (L2)"

echo ""
echo "===== Phase 5 ${API_UPPER}: All ${API_UPPER} specs × L1,L2,L4 ====="
python3 scripts/augmentation/run_augment_batch.py \
    "specs/*-${API}.json" \
    --levels 1 2 4 --seed 42 \
    --out "results/augmentation/phase5_${API}" \
    --title "Phase 5: All ${API_UPPER} Specs (L1+L2+L4)"

echo ""
echo "========================================================"
echo "  ${API_UPPER} STREAM DONE: $(date)"
echo "========================================================"
touch "results/augmentation/.${API}_done"
