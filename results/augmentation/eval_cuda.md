# Post-Fix Eval: Rodinia CUDA
**Date:** 2026-03-11  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L3 | L4 | Transforms Applied |
|------|---|---|---|---|---|
| rodinia-backprop-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bfs-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bptree-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-cfd-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-dwt2d-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-gaussian-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot3d-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-huffman-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hybridsort-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lud-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-mummergpu-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-myocyte-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-cuda | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-nw-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-particlefilter-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-pathfinder-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✗ FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-srad-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-streamcluster-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |

## Summary Statistics

**Level 1:** 15/22 PASS (68%) | BUILD_FAIL=6 | FAIL=1 | ERROR=0
**Level 2:** 15/22 PASS (68%) | BUILD_FAIL=6 | FAIL=1 | ERROR=0
**Level 3:** 13/22 PASS (59%) | BUILD_FAIL=9 | FAIL=0 | ERROR=0
**Level 4:** 11/22 PASS (50%) | BUILD_FAIL=10 | FAIL=1 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 55 |
| PointerArithmeticToArrayIndex | 47 |
| ChangeNames | 36 |
| ArithmeticTransform | 31 |
| TypedefExpansion | 10 |
