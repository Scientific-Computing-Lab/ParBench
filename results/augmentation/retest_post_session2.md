# Post-Session 2 Full Retest: 60 Rodinia specs x L1-L4 (seed=42)
**Date:** 2026-03-20  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L3 | L4 | Transforms Applied |
|------|---|---|---|---|---|
| rodinia-backprop-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-backprop-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames |
| rodinia-backprop-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-bfs-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-bfs-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | SwapCondition |
| rodinia-bfs-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-bptree-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-bptree-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-bptree-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-cfd-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-cfd-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-cfd-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-dwt2d-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-dwt2d-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-gaussian-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-gaussian-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-heartwall-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | SwapCondition |
| rodinia-heartwall-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-hotspot-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-hotspot-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | SwapCondition, TypedefExpansion |
| rodinia-hotspot-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | SwapCondition |
| rodinia-hotspot3d-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-hotspot3d-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-hotspot3d-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-huffman-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeFunctionNames, ChangeNames, SwapCondition |
| rodinia-hybridsort-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-hybridsort-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-kmeans-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, SwapCondition |
| rodinia-kmeans-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-kmeans-opencl | ✗ FAIL | ✗ FAIL | ✗ FAIL | ✗ FAIL | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-lavamd-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-lavamd-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-lavamd-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-lud-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-lud-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition, TypedefExpansion |
| rodinia-lud-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition, TypedefExpansion |
| rodinia-mummergpu-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, SwapCondition |
| rodinia-mummergpu-omp | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-myocyte-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-myocyte-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-myocyte-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-nn-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | SwapCondition |
| rodinia-nn-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-nn-opencl | ✗ FAIL | ✗ FAIL | ✗ FAIL | ✗ FAIL | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-nw-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-nw-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-nw-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-particlefilter-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-particlefilter-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-particlefilter-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-pathfinder-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-pathfinder-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | SwapCondition |
| rodinia-pathfinder-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-srad-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-srad-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-srad-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-streamcluster-cuda | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-streamcluster-omp | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, ChangeFunctionNames, ChangeNames, SwapCondition |
| rodinia-streamcluster-opencl | ✓ PASS | ✓ PASS | ✓ PASS | ✓ PASS | ArithmeticTransform, SwapCondition |

## Summary Statistics

**Level 1:** 54/60 PASS (90%) | BUILD_FAIL=4 | FAIL=2 | ERROR=0
**Level 2:** 54/60 PASS (90%) | BUILD_FAIL=4 | FAIL=2 | ERROR=0
**Level 3:** 54/60 PASS (90%) | BUILD_FAIL=4 | FAIL=2 | ERROR=0
**Level 4:** 54/60 PASS (90%) | BUILD_FAIL=4 | FAIL=2 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 162 |
| ArithmeticTransform | 69 |
| ChangeNames | 55 |
| TypedefExpansion | 7 |
| PointerArithmeticToArrayIndex | 6 |
| ChangeFunctionNames | 2 |
