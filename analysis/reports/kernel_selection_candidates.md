# Kernel Selection Candidates — Phase 1b

**Date**: 2026-02-12
**Pool**: 327 kernels with all 4 API variants found in HeCBench
**Filtered**: 325 with Makefiles, 242 with self-checking patterns
**Selected**: 15 new kernels (+ 5 existing = 20 total)

## Existing 5 Pilot Kernels

| # | Kernel | Domain |
|---|--------|--------|
| 1 | nn | Nearest neighbor / graph search |
| 2 | scan | Parallel prefix sum / reduction |
| 3 | binomial | Finance / option pricing |
| 4 | particle-diffusion | Physics simulation |
| 5 | radixsort | Sorting |

## Proposed 15 New Kernels

| # | Kernel | Domain | Files (CUDA) | Self-checking? | Why it's a good pick |
|---|--------|--------|:---:|---|---|
| 6 | aes | Cryptography (block cipher) | 5 | Yes — memcmp verify | AES encryption is a classic GPU crypto workload with clear correctness check |
| 7 | chacha20 | Cryptography (stream cipher) | 2 | Yes — `PASS`/`FAIL` | Compact stream cipher with explicit PASS/FAIL output |
| 8 | lud | Linear algebra (LU decomposition) | 4 | Yes — `lud_verify()` | Classic dense linear algebra decomposition with built-in verification |
| 9 | eigenvalue | Linear algebra (eigenvalue) | 5 | Yes — reference comparison | Eigenvalue computation for tridiagonal symmetric matrices |
| 10 | nbody | N-body simulation | 7 | Yes — `PASS`/`FAIL` | Gravitational N-body simulation, fundamental physics benchmark |
| 11 | simpleSpmv | Sparse computation (SpMV) | 4 | Yes — error rate | Sparse matrix-vector multiply, core of many scientific codes |
| 12 | sobel | Image processing (edge detection) | 4 | Yes — norm error | Sobel edge detection filter with CPU reference comparison |
| 13 | dct8x8 | Signal processing (DCT) | 4 | Yes — `PASS`/`FAIL` | 8×8 Discrete Cosine Transform, fundamental to JPEG/video compression |
| 14 | convolutionSeparable | Signal processing (convolution) | 4 | Yes — `PASS`/`FAIL` via L2 norm | Separable 2D convolution, fundamental image/signal processing kernel |
| 15 | fft | Signal processing (FFT) | 4 | Yes — error check | Fast Fourier Transform, a foundational signal processing algorithm |
| 16 | ising | Statistical physics (Ising model) | 3 | Yes — `PASS`/`FAIL` | 2D Ising model Monte Carlo simulation with verification |
| 17 | bilateral | Image processing (bilateral filter) | 3 | Yes — reference comparison | Bilateral filter for edge-preserving smoothing, classic image processing kernel |
| 18 | chi2 | Statistics (chi-squared) | 3 | Yes — reference comparison | Chi-squared distance computation, statistical analysis primitive |
| 19 | merge | Parallel primitives (merge) | 2 | Yes — `PASS`/`FAIL` | Parallel merge of sorted arrays, fundamental parallel primitive |
| 20 | fwt | Signal processing (Walsh transform) | 4 | Yes — `PASS`/`FAIL` | Fast Walsh Transform, fundamental signal processing algorithm |

## Domain Coverage Summary

| Domain | Kernels |
|--------|---------|
| Sorting / parallel primitives | radixsort, merge |
| Reduction / prefix sum | scan |
| Graph / search | nn |
| Finance | binomial |
| Physics simulation | particle-diffusion, nbody |
| Statistical physics | ising |
| Cryptography | aes, chacha20 |
| Linear algebra | lud, eigenvalue |
| Sparse computation | simpleSpmv |
| Image processing | sobel, bilateral |
| Signal processing | dct8x8, convolutionSeparable, fft, fwt |
| Statistics | chi2 |

## Selection Methodology

1. Started with 327 kernels that have all 4 API variants (cuda, hip, sycl, omp)
2. Filtered to 325 with Makefiles present in CUDA variant
3. Filtered to 242 with self-checking patterns (PASS/FAIL/verify/correct)
4. Excluded existing 5 pilot kernels
5. Excluded kernels with > 15 source files (too complex) or < 2 files (too trivial)
6. Excluded kernels requiring external input data files (e.g., hotspot3D)
7. Selected for maximum domain diversity across computational science
8. All 15 compile successfully with `make ARCH=sm_89` on this machine (RTX 4070)
