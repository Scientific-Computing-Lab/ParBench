# Post-Fix Eval: Rodinia OMP
**Date:** 2026-03-11  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L3 | L4 | Transforms Applied |
|------|---|---|---|---|---|
| rodinia-backprop-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames |
| rodinia-bfs-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bptree-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-cfd-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-hotspot3d-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-omp | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lud-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-mummergpu-omp | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-myocyte-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-omp | ✗ FAIL | ✗ FAIL | ✗ FAIL | ✗ FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nw-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-particlefilter-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-pathfinder-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-srad-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-streamcluster-omp | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |

## Summary Statistics

**Level 1:** 15/18 PASS (83%) | BUILD_FAIL=2 | FAIL=1 | ERROR=0
**Level 2:** 14/18 PASS (77%) | BUILD_FAIL=3 | FAIL=1 | ERROR=0
**Level 3:** 9/18 PASS (50%) | BUILD_FAIL=8 | FAIL=1 | ERROR=0
**Level 4:** 7/18 PASS (38%) | BUILD_FAIL=10 | FAIL=1 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 44 |
| PointerArithmeticToArrayIndex | 39 |
| ChangeNames | 31 |
| ArithmeticTransform | 23 |
| TypedefExpansion | 12 |
