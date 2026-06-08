# XSBench Augmentation Smoke Test (L2, seed=42)
**Date:** 2026-03-24  |  **Seed:** 42

## Results Matrix

| Spec | L2 | Transforms Applied |
|------|---|---|
| xsbench-xsbench-cuda | ✓ PASS | SwapCondition |
| xsbench-xsbench-omp | ✓ PASS | ChangeNames, SwapCondition |
| xsbench-xsbench-opencl | ✓ PASS | SwapCondition |

## Summary Statistics

**Level 2:** 3/3 PASS (100%) | BUILD_FAIL=0 | FAIL=0 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 3 |
| ChangeNames | 1 |
