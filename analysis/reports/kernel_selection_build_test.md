# CUDA Build Test Results — Phase 1b

**Date**: 2026-02-12
**Machine**: RTX 4070, sm_89, CUDA 12.3, nvcc V12.3.103
**Build command**: `make ARCH=sm_89` in each `<kernel>-cuda/` directory

## Results

| # | Kernel | Build Result | Notes |
|---|--------|:---:|---|
| 1 | aes | PASS | Clean build, 2 object files |
| 2 | chacha20 | PASS | Clean build |
| 3 | lud | PASS | Warnings: unused result from `fscanf` (harmless) |
| 4 | eigenvalue | PASS | Clean build, 2 object files |
| 5 | nbody | PASS | Clean build, 2 object files |
| 6 | simpleSpmv | PASS | Clean build, 3 object files (mixed .cu/.cpp) |
| 7 | sobel | PASS | Clean build, 2 object files |
| 8 | dct8x8 | PASS | Clean build, 3 object files |
| 9 | convolutionSeparable | PASS | Clean build, 3 object files |
| 10 | fft | PASS | Clean build |
| 11 | ising | PASS | Uses `-Xcompiler -fopenmp` (host-side OpenMP for reference) |
| 12 | histogram | PASS | Warnings: unused result from `fread` (harmless) |
| 13 | bwt | PASS | Clean build, mixed .cu/.cpp |
| 14 | merge | PASS | Clean build |
| 15 | heat2d | PASS | Uses `-Xcompiler -fopenmp` (host-side OpenMP for reference) |

## Summary

**15/15 candidates compile successfully.** No replacements needed.

### Notes
- `ising` and `heat2d` use host-side OpenMP (`-Xcompiler -fopenmp`) for CPU reference computation — this is fine since g++ supports OpenMP on this machine.
- `lud` and `histogram` have minor compiler warnings (unused return values) that don't affect correctness.
- All builds used the standard HeCBench Makefile infrastructure with `ARCH=sm_89`.
