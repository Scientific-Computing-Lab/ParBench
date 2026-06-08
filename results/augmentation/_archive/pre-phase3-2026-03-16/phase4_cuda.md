# Phase 4: All Rodinia CUDA Specs (L2)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L2 | Transforms Applied |
|------|---|---|
| rodinia-backprop-cuda | ✓ PASS | *(none)* |
| rodinia-bfs-cuda | ✓ PASS | *(none)* |
| rodinia-bptree-cuda | ✓ PASS | SwapCondition |
| rodinia-cfd-cuda | ✗ BUILD_FAIL | SwapCondition |
| rodinia-dwt2d-cuda | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-gaussian-cuda | ✓ PASS | *(none)* |
| rodinia-heartwall-cuda | ✓ PASS | *(none)* |
| rodinia-hotspot-cuda | ✓ PASS | *(none)* |
| rodinia-hotspot3d-cuda | ✓ PASS | *(none)* |
| rodinia-huffman-cuda | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-hybridsort-cuda | ✗ BUILD_FAIL | SwapCondition |
| rodinia-kmeans-cuda | ✗ BUILD_FAIL | *(none)* |
| rodinia-lavamd-cuda | ✓ PASS | *(none)* |
| rodinia-lud-cuda | ✓ PASS | *(none)* |
| rodinia-mummergpu-cuda | ✗ BUILD_FAIL | *(none)* |
| rodinia-myocyte-cuda | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-cuda | ✗ FAIL | TypedefExpansion |
| rodinia-nw-cuda | ✓ PASS | *(none)* |
| rodinia-particlefilter-cuda | ✓ PASS | *(none)* |
| rodinia-pathfinder-cuda | ✓ PASS | *(none)* |
| rodinia-srad-cuda | ✓ PASS | *(none)* |
| rodinia-streamcluster-cuda | ✗ BUILD_FAIL | TypedefExpansion |

## Summary Statistics

**Level 2:** 15/22 PASS (68%) | BUILD_FAIL=6 | FAIL=1 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 6 |
| ArithmeticTransform | 2 |
| ChangeNames | 2 |
| TypedefExpansion | 2 |
| PointerArithmeticToArrayIndex | 1 |
