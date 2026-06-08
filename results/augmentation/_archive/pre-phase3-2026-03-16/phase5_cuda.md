# Phase 5: All CUDA Specs (L1+L2+L4)
**Date:** 2026-03-10  |  **Seed:** 42

## Results Matrix

| Spec | L1 | L2 | L4 | Transforms Applied |
|------|---|---|---|---|
| hecbench-aes-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-babelstream-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-backprop-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-bezier-surface-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-bilateral-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-binomial-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-ccsd-trpdrv-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-chacha20-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-chi2-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-convolution3d-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-convolutionseparable-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-crc64-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-dct8x8-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-deredundancy-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-eigenvalue-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-feynman-kac-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-fft-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-floydwarshall-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-fpc-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-fwt-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-ga-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-gaussian-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-geglu-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-heat2d-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-ising-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-iso2dfd-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-jenkins-hash-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-keccaktreehash-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-knn-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-laplace3d-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-lud-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-lulesh-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-mandelbrot-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-maxpool3d-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-md5hash-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-merge-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-mis-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-murmurhash3-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-myocyte-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-nbody-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-nn-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-nw-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-particle-diffusion-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-pathfinder-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-perplexity-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-popcount-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-pso-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-radixsort-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-rmsnorm-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-scan-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-secp256k1-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-simplespmv-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-sobel-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-sobol-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-softmax-online-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-stencil1d-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-thomas-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-tissue-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-triad-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| hecbench-tsp-cuda | ! ERROR | ! ERROR | ! ERROR | *(none)* |
| rodinia-backprop-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bfs-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-bptree-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-cfd-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-dwt2d-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-gaussian-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-heartwall-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hotspot3d-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-huffman-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-hybridsort-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-kmeans-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lavamd-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-lud-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-mummergpu-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-myocyte-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-nn-cuda | ✗ FAIL | ✗ FAIL | ✗ BUILD_FAIL | ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |
| rodinia-nw-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-particlefilter-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-pathfinder-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-srad-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition |
| rodinia-streamcluster-cuda | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArithmeticToArrayIndex, SwapCondition, TypedefExpansion |

## Summary Statistics

**Level 1:** 15/82 PASS (18%) | BUILD_FAIL=6 | FAIL=1 | ERROR=60
**Level 2:** 15/82 PASS (18%) | BUILD_FAIL=6 | FAIL=1 | ERROR=60
**Level 4:** 0/82 PASS (0%) | BUILD_FAIL=22 | FAIL=0 | ERROR=60

## Transform Frequency

| Transform | Times Applied |
|-----------|--------------|
| SwapCondition | 34 |
| ChangeNames | 26 |
| PointerArithmeticToArrayIndex | 25 |
| ArithmeticTransform | 21 |
| TypedefExpansion | 7 |
