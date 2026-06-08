# Phase 4: Rodinia Batch — All Specs at L2
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L2 | Transforms Applied |
|------|---|---|
| rodinia-backprop-cuda | ✓ PASS | *(none)* |
| rodinia-backprop-omp | ✓ PASS | *(none)* |
| rodinia-backprop-opencl | ✓ PASS | SwapCondition |
| rodinia-bfs-cuda | ✓ PASS | *(none)* |
| rodinia-bfs-omp | ✓ PASS | *(none)* |
| rodinia-bfs-opencl | ✓ PASS | *(none)* |
| rodinia-bptree-cuda | ✓ PASS | SwapCondition |
| rodinia-bptree-omp | ✓ PASS | *(none)* |
| rodinia-bptree-opencl | ✓ PASS | SwapCondition |
| rodinia-cfd-cuda | ✗ BUILD_FAIL | SwapCondition |
| rodinia-cfd-omp | ✓ PASS | SwapCondition |
| rodinia-cfd-opencl | ✗ BUILD_FAIL | TypedefExpansion |
| rodinia-dwt2d-cuda | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-dwt2d-opencl | ✗ BUILD_FAIL | SwapCondition |
| rodinia-gaussian-cuda | ✓ PASS | *(none)* |
| rodinia-gaussian-opencl | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-heartwall-cuda | ✓ PASS | *(none)* |
| rodinia-heartwall-omp | ✓ PASS | *(none)* |
| rodinia-heartwall-opencl | ✓ PASS | *(none)* |
| rodinia-hotspot-cuda | ✓ PASS | *(none)* |
| rodinia-hotspot-omp | ✓ PASS | TypedefExpansion |
| rodinia-hotspot-opencl | ✓ PASS | *(none)* |
| rodinia-hotspot3d-cuda | ✓ PASS | *(none)* |
| rodinia-hotspot3d-omp | ✓ PASS | *(none)* |
| rodinia-hotspot3d-opencl | ✓ PASS | *(none)* |
| rodinia-huffman-cuda | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-hybridsort-cuda | ✗ BUILD_FAIL | SwapCondition |
| rodinia-hybridsort-opencl | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-kmeans-cuda | ✗ BUILD_FAIL | *(none)* |
| rodinia-kmeans-omp | ✗ BUILD_FAIL | PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-opencl | ✗ FAIL | ChangeNames, SwapCondition |
| rodinia-lavamd-cuda | ✓ PASS | *(none)* |
| rodinia-lavamd-omp | ✓ PASS | *(none)* |
| rodinia-lavamd-opencl | ✓ PASS | TypedefExpansion |
| rodinia-lud-cuda | ✓ PASS | *(none)* |
| rodinia-lud-omp | ✗ BUILD_FAIL | ArithmeticTransform, SwapCondition |
| rodinia-lud-opencl | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-mummergpu-cuda | ✗ BUILD_FAIL | *(none)* |
| rodinia-mummergpu-omp | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-myocyte-cuda | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-myocyte-omp | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-myocyte-opencl | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-nn-cuda | ✗ FAIL | TypedefExpansion |
| rodinia-nn-omp | ✗ FAIL | *(none)* |
| rodinia-nn-opencl | ✗ FAIL | SwapCondition |
| rodinia-nw-cuda | ✓ PASS | *(none)* |
| rodinia-nw-omp | ✓ PASS | *(none)* |
| rodinia-nw-opencl | ✓ PASS | *(none)* |
| rodinia-particlefilter-cuda | ✓ PASS | *(none)* |
| rodinia-particlefilter-omp | ✓ PASS | *(none)* |
| rodinia-particlefilter-opencl | ✓ PASS | SwapCondition |
| rodinia-pathfinder-cuda | ✓ PASS | *(none)* |
| rodinia-pathfinder-omp | ✓ PASS | *(none)* |
| rodinia-pathfinder-opencl | ✗ BUILD_FAIL | *(none)* |
| rodinia-srad-cuda | ✓ PASS | *(none)* |
| rodinia-srad-omp | ✓ PASS | *(none)* |
| rodinia-srad-opencl | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-streamcluster-cuda | ✗ BUILD_FAIL | TypedefExpansion |
| rodinia-streamcluster-omp | ✗ BUILD_FAIL | TypedefExpansion |
| rodinia-streamcluster-opencl | ✓ PASS | *(none)* |

## Summary Statistics

**Level 2:** 43/60 PASS (71%) | BUILD_FAIL=13 | FAIL=4 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 22 |
| ArithmeticTransform | 9 |
| ChangeNames | 7 |
| TypedefExpansion | 6 |
| PointerArithmeticToArrayIndex | 3 |
