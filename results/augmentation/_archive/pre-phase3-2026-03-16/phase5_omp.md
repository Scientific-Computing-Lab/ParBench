# Phase 5: All OMP Specs (L1+L2+L4)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L4 | Transforms Applied |
|------|---|---|---|---|
| hecbench-aes-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-babelstream-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-backprop-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-bezier-surface-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-bilateral-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-binomial-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-ccsd-trpdrv-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-chacha20-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-chi2-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-convolution3d-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-convolutionseparable-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-crc64-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-dct8x8-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-deredundancy-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-eigenvalue-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-feynman-kac-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-fft-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-floydwarshall-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-fpc-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-fwt-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-ga-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-gaussian-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-geglu-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-heat2d-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-ising-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-iso2dfd-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-jenkins-hash-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-keccaktreehash-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-knn-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-laplace3d-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-lud-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-lulesh-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-mandelbrot-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-maxpool3d-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-md5hash-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-merge-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-mis-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-murmurhash3-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-myocyte-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-nbody-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-nn-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-nw-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-particle-diffusion-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-pathfinder-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-perplexity-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-popcount-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-pso-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-radixsort-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-rmsnorm-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-scan-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-secp256k1-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-simplespmv-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-sobel-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-sobol-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-softmax-online-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-stencil1d-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-thomas-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-tissue-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-triad-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-tsp-omp | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| rodinia-backprop-omp | ✓ PASS | ✓ PASS | ✓ PASS | ChangeNames |
| rodinia-bfs-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bptree-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-cfd-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-hotspot3d-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-omp | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lud-omp | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-mummergpu-omp | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-myocyte-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-omp | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nw-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-particlefilter-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-pathfinder-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-srad-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-streamcluster-omp | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |

## Summary Statistics

**Level 1:** 14/78 PASS (17%) | BUILD_FAIL=3 | FAIL=1 | ERROR=60
**Level 2:** 13/78 PASS (16%) | BUILD_FAIL=4 | FAIL=1 | ERROR=60
**Level 4:** 1/78 PASS (1%) | BUILD_FAIL=17 | FAIL=0 | ERROR=60

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 27 |
| ChangeNames | 22 |
| PointerArithmeticToArrayIndex | 22 |
| ArithmeticTransform | 16 |
| TypedefExpansion | 8 |
