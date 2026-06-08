"""Shared constants for the ParBench harness and analysis scripts.

EXCLUDED_SPECS: The 9 KNOWN_FAIL specs excluded from evaluation batches
and statistics. See .claude/rules/known-issues.md for WHY each spec fails.
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
})
