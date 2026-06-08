#!/usr/bin/env bash
# Qwen 3.5 397B OpenCL re-evaluation — all 4 OpenCL-involving directions
# Post kernel-only fix (commit 820e492)
# Created: 2026-03-30
#
# SAFETY: --resume ensures existing CUDA↔OMP results are never touched
# Directions 1 & 2 use the new kernel-only path (_is_kernel_only_translation)
# Directions 3 & 4 use the existing full-program path (always correct)

set -euo pipefail

cd /home/samyak/Desktop/parbench_sam
source env_parbench/bin/activate

PROJECT_ROOT="/home/samyak/Desktop/parbench_sam"
MODEL="together-qwen-3.5-397b-a17b"
COMMON_ARGS="--suite rodinia --models ${MODEL} --augment-levels 0 1 2 3 4 --max-retries 3 --temperature 0.0 --resume -v --project-root ${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/results/evaluation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/qwen_opencl_redo_${TIMESTAMP}.log"

echo "========================================" | tee "$LOG_FILE"
echo "Qwen 3.5 OpenCL Re-evaluation" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Pre-flight: count existing files
EXISTING=$(ls ${LOG_DIR}/${MODEL}/ | wc -l)
echo "Pre-flight: ${EXISTING} existing result files" | tee -a "$LOG_FILE"

# Direction 1: CUDA → OpenCL (kernel-only translation)
echo "" | tee -a "$LOG_FILE"
echo "===== Direction 1/4: cuda-to-opencl (kernel-only) =====" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
python3 scripts/evaluation/run_eval_batch.py \
  --direction cuda-to-opencl \
  ${COMMON_ARGS} 2>&1 | tee -a "$LOG_FILE"
D1_COUNT=$(ls ${LOG_DIR}/${MODEL}/ | wc -l)
echo "After Direction 1: ${D1_COUNT} files (added $((D1_COUNT - EXISTING)))" | tee -a "$LOG_FILE"

# Direction 2: OMP → OpenCL (kernel-only translation)
echo "" | tee -a "$LOG_FILE"
echo "===== Direction 2/4: omp-to-opencl (kernel-only) =====" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
python3 scripts/evaluation/run_eval_batch.py \
  --direction omp-to-opencl \
  ${COMMON_ARGS} 2>&1 | tee -a "$LOG_FILE"
D2_COUNT=$(ls ${LOG_DIR}/${MODEL}/ | wc -l)
echo "After Direction 2: ${D2_COUNT} files (added $((D2_COUNT - D1_COUNT)))" | tee -a "$LOG_FILE"

# Direction 3: OpenCL → CUDA (full-program translation)
echo "" | tee -a "$LOG_FILE"
echo "===== Direction 3/4: opencl-to-cuda (full-program) =====" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
python3 scripts/evaluation/run_eval_batch.py \
  --direction opencl-to-cuda \
  ${COMMON_ARGS} 2>&1 | tee -a "$LOG_FILE"
D3_COUNT=$(ls ${LOG_DIR}/${MODEL}/ | wc -l)
echo "After Direction 3: ${D3_COUNT} files (added $((D3_COUNT - D2_COUNT)))" | tee -a "$LOG_FILE"

# Direction 4: OpenCL → OMP (full-program translation)
echo "" | tee -a "$LOG_FILE"
echo "===== Direction 4/4: opencl-to-omp (full-program) =====" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
python3 scripts/evaluation/run_eval_batch.py \
  --direction opencl-to-omp \
  ${COMMON_ARGS} 2>&1 | tee -a "$LOG_FILE"
D4_COUNT=$(ls ${LOG_DIR}/${MODEL}/ | wc -l)
echo "After Direction 4: ${D4_COUNT} files (added $((D4_COUNT - D3_COUNT)))" | tee -a "$LOG_FILE"

# Post-run analysis
echo "" | tee -a "$LOG_FILE"
echo "===== Running analyze_eval.py =====" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
python3 scripts/evaluation/analyze_eval.py \
  --project-root "${PROJECT_ROOT}" \
  --output-dir results/evaluation 2>&1 | tee -a "$LOG_FILE"

# Final summary
FINAL_COUNT=$(ls ${LOG_DIR}/${MODEL}/ | wc -l)
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "COMPLETE: $(date)" | tee -a "$LOG_FILE"
echo "Total files: ${FINAL_COUNT} (started with ${EXISTING}, added $((FINAL_COUNT - EXISTING)))" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Verify CUDA↔OMP results are untouched
CUDA_OMP_COUNT=$(ls ${LOG_DIR}/${MODEL}/ | grep -E "(cuda-to-.*-omp|omp-to-.*-cuda)" | grep -v opencl | wc -l)
echo "SAFETY CHECK: CUDA↔OMP files = ${CUDA_OMP_COUNT} (expected: 180)" | tee -a "$LOG_FILE"
if [ "$CUDA_OMP_COUNT" -ne 180 ]; then
  echo "WARNING: CUDA↔OMP file count changed! Expected 180, got ${CUDA_OMP_COUNT}" | tee -a "$LOG_FILE"
fi
