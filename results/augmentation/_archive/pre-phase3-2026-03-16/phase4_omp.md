# Phase 4: All Rodinia OMP Specs (L2)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L2 | Transforms Applied |
|------|---|---|
| rodinia-backprop-omp | ✓ PASS | *(none)* |
| rodinia-bfs-omp | ✓ PASS | *(none)* |
| rodinia-bptree-omp | ✓ PASS | *(none)* |
| rodinia-cfd-omp | ✓ PASS | SwapCondition |
| rodinia-heartwall-omp | ✓ PASS | *(none)* |
| rodinia-hotspot-omp | ✓ PASS | TypedefExpansion |
| rodinia-hotspot3d-omp | ✓ PASS | *(none)* |
| rodinia-kmeans-omp | ✗ BUILD_FAIL | PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-omp | ✓ PASS | *(none)* |
| rodinia-lud-omp | ✗ BUILD_FAIL | ArithmeticTransform, SwapCondition |
| rodinia-mummergpu-omp | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-myocyte-omp | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-omp | ✗ FAIL | *(none)* |
| rodinia-nw-omp | ✓ PASS | *(none)* |
| rodinia-particlefilter-omp | ✓ PASS | *(none)* |
| rodinia-pathfinder-omp | ✓ PASS | *(none)* |
| rodinia-srad-omp | ✓ PASS | *(none)* |
| rodinia-streamcluster-omp | ✗ BUILD_FAIL | TypedefExpansion |

## Summary Statistics

**Level 2:** 13/18 PASS (72%) | BUILD_FAIL=4 | FAIL=1 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 5 |
| ArithmeticTransform | 3 |
| TypedefExpansion | 2 |
| PointerArithmeticToArrayIndex | 2 |
| ChangeNames | 2 |
