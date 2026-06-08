# Phase 5: All OpenCL Specs (L1+L2+L4)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L4 | Transforms Applied |
|------|---|---|---|---|
| rodinia-backprop-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bfs-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bptree-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-cfd-opencl | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-dwt2d-opencl | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-gaussian-opencl | ✗ BUILD_FAIL | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot3d-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hybridsort-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-opencl | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-lud-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-myocyte-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-opencl | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nw-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-particlefilter-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-pathfinder-opencl | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-srad-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-streamcluster-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |

## Summary Statistics

**Level 1:** 15/20 PASS (75%) | BUILD_FAIL=3 | FAIL=2 | ERROR=0
**Level 2:** 15/20 PASS (75%) | BUILD_FAIL=3 | FAIL=2 | ERROR=0
**Level 4:** 0/20 PASS (0%) | BUILD_FAIL=20 | FAIL=0 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 42 |
| PointerArithmeticToArrayIndex | 26 |
| ArithmeticTransform | 25 |
| ChangeNames | 25 |
| TypedefExpansion | 7 |
