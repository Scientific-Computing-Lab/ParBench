# Post-Fix Eval: Rodinia OpenCL
**Date:** 2026-03-11  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L3 | L4 | Transforms Applied |
|------|---|---|---|---|---|
| rodinia-backprop-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bfs-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bptree-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-cfd-opencl | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-dwt2d-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-gaussian-opencl | ✗ BUILD_FAIL | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot3d-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hybridsort-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-opencl | ✗ FAIL | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-lud-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-myocyte-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-opencl | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nw-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✗ FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-particlefilter-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-pathfinder-opencl | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-srad-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-streamcluster-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |

## Summary Statistics

**Level 1:** 15/20 PASS (75%) | BUILD_FAIL=3 | FAIL=2 | ERROR=0
**Level 2:** 16/20 PASS (80%) | BUILD_FAIL=2 | FAIL=2 | ERROR=0
**Level 3:** 7/20 PASS (35%) | BUILD_FAIL=12 | FAIL=1 | ERROR=0
**Level 4:** 4/20 PASS (20%) | BUILD_FAIL=15 | FAIL=1 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 61 |
| PointerArithmeticToArrayIndex | 46 |
| ChangeNames | 43 |
| ArithmeticTransform | 40 |
| TypedefExpansion | 9 |
