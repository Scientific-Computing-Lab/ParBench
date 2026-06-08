# Phase 4: All Rodinia OpenCL Specs (L2)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L2 | Transforms Applied |
|------|---|---|
| rodinia-backprop-opencl | ✓ PASS | SwapCondition |
| rodinia-bfs-opencl | ✓ PASS | *(none)* |
| rodinia-bptree-opencl | ✓ PASS | SwapCondition |
| rodinia-cfd-opencl | ✗ BUILD_FAIL | TypedefExpansion |
| rodinia-dwt2d-opencl | ✗ BUILD_FAIL | SwapCondition |
| rodinia-gaussian-opencl | ✓ PASS | ChangeNames, SwapCondition |
| rodinia-heartwall-opencl | ✓ PASS | *(none)* |
| rodinia-hotspot-opencl | ✓ PASS | *(none)* |
| rodinia-hotspot3d-opencl | ✓ PASS | *(none)* |
| rodinia-hybridsort-opencl | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-kmeans-opencl | ✗ FAIL | ChangeNames, SwapCondition |
| rodinia-lavamd-opencl | ✓ PASS | TypedefExpansion |
| rodinia-lud-opencl | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-myocyte-opencl | ✓ PASS | ArithmeticTransform, SwapCondition |
| rodinia-nn-opencl | ✗ FAIL | SwapCondition |
| rodinia-nw-opencl | ✓ PASS | *(none)* |
| rodinia-particlefilter-opencl | ✓ PASS | SwapCondition |
| rodinia-pathfinder-opencl | ✗ BUILD_FAIL | *(none)* |
| rodinia-srad-opencl | ✓ PASS | ArithmeticTransform, ChangeNames, SwapCondition |
| rodinia-streamcluster-opencl | ✓ PASS | *(none)* |

## Summary Statistics

**Level 2:** 15/20 PASS (75%) | BUILD_FAIL=3 | FAIL=2 | ERROR=0

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 11 |
| ArithmeticTransform | 4 |
| ChangeNames | 3 |
| TypedefExpansion | 2 |
