"""Shared constants for the ParBench harness and analysis scripts.

EXCLUDED_SPECS: The 10 KNOWN_FAIL specs excluded from evaluation batches
and statistics. See .claude/rules/known-issues.md for WHY each spec fails.

PERFORMANCE_ONLY_SPECS: Specs retained in the artifact and benchmark
inventory but excluded from correctness denominators (Task 4, gate E-04).
Mirrors the spec JSONs' ``verification.scope == "performance_only"`` markers;
tests/test_mixbench_scope.py pins the two in lockstep. Evidence:
results/analysis/source_baseline_dispositions/mixbench.json.
"""
from __future__ import annotations

EXCLUDED_SPECS: frozenset[str] = frozenset({
    "rodinia-kmeans-cuda",
    "rodinia-mummergpu-cuda",
    "rodinia-mummergpu-omp",
    "rodinia-hybridsort-cuda",
    "rodinia-nn-opencl",
    "rodinia-kmeans-opencl",
    "rodinia-backprop-opencl",
    "hecbench-stencil1d-omp_target",
    "hecbench-scan-omp_target",
    "hecbench-lud-omp",  # uninitialized-read UB under thread_limit under-provisioning (P0.2; corrected 2026-07-27, see results/response_2026/lud/CORRECTION_2026-07-27.md)
})

# Task 4 (E-04): mixbench has no extractable computed value (kernel results are
# discarded behind a guard designed never to execute; the host frees the output
# buffer unread), so no independent comparator exists and a seeded wrong-kernel
# control PASSes the declared oracle. performance_only is the recorded verdict.
PERFORMANCE_ONLY_SPECS: frozenset[str] = frozenset({
    "mixbench-mixbench-cuda",
    "mixbench-mixbench-omp",
    "mixbench-mixbench-opencl",
})

# Union used by every correctness-denominator consumer: KNOWN_FAIL specs are
# excluded because their baselines are broken; performance_only specs because
# no oracle can gate semantic correctness for them.
CORRECTNESS_INELIGIBLE_SPECS: frozenset[str] = (
    EXCLUDED_SPECS | PERFORMANCE_ONLY_SPECS
)
