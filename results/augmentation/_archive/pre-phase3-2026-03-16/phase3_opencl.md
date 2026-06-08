# Phase 3 Smoke Test: OpenCL (rodinia-bfs-opencl)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L4 | Transforms Applied |
|------|---|---|---|---|
| rodinia-bfs-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |

## Summary Statistics

**Level 1:** 1/1 PASS (100%) | BUILD_FAIL=0 | FAIL=0 | ERROR=0
**Level 2:** 1/1 PASS (100%) | BUILD_FAIL=0 | FAIL=0 | ERROR=0
**Level 4:** 0/1 PASS (0%) | BUILD_FAIL=1 | FAIL=0 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| ArithmeticTransform | 1 |
| ChangeNames | 1 |
| PointerArithmeticToArrayIndex | 1 |
| SwapCondition | 1 |
