# Phase 3: Smoke Test (3 Specs × L1,L2,L4)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L4 | Transforms Applied |
|------|---|---|---|---|
| rodinia-bfs-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bfs-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |

## Summary Statistics

**Level 1:** 3/3 PASS (100%) | BUILD_FAIL=0 | FAIL=0 | ERROR=0
**Level 2:** 3/3 PASS (100%) | BUILD_FAIL=0 | FAIL=0 | ERROR=0
**Level 4:** 0/3 PASS (0%) | BUILD_FAIL=3 | FAIL=0 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| ChangeNames | 3 |
| PointerArithmeticToArrayIndex | 3 |
| SwapCondition | 3 |
| TypedefExpansion | 3 |
| ArithmeticTransform | 1 |
