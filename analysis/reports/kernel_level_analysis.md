# Kernel-Level API Co-occurrence Analysis for Rosetta Stone

## Executive Summary

This analysis identifies **kernel-level** translation pairs across parallel computing benchmarks.
Unlike repository-level analysis (which counted benchmarks), this counts **individual kernels**
that have verified equivalent implementations across multiple APIs.

## Key Finding: Repository vs Kernel Level

| Metric | Repository-Level | Kernel-Level |
|--------|-----------------|--------------|
| CUDA + OpenMP | 21 repos | **472 kernels** |
| CUDA + HIP | 10 repos | **633 kernels** |
| CUDA + SYCL | 9 repos | **616 kernels** |

**The kernel-level count is ~20-60x higher** because single benchmarks like HeCBench contain hundreds of kernels.

## Top Translation Pair Opportunities

| API Pair | # Kernels | Best Sources |
|----------|-----------|--------------|
| CUDA ↔ HIP | 633 | HeCBench (504), RAJAPerf (106), CloverLeaf (16) |
| CUDA ↔ SYCL | 616 | HeCBench (487), RAJAPerf (106), CloverLeaf (16) |
| HIP ↔ SYCL | 615 | HeCBench (486), RAJAPerf (106), CloverLeaf (16) |
| CUDA ↔ OpenMP | 472 | HeCBench (325), RAJAPerf (106), Rodinia (19) |
| HIP ↔ OpenMP | 453 | HeCBench (324), RAJAPerf (106), CloverLeaf (16) |
| OpenMP ↔ SYCL | 453 | HeCBench (324), RAJAPerf (106), BabelStream (5) |

## Benchmark Contributions

### 1. HeCBench (506 kernels)
- **Structure**: Each kernel in separate directory with `-cuda`, `-hip`, `-sycl`, `-omp` variants
- **Coverage**: 325 kernels have all 4 APIs (CUDA, HIP, SYCL, OpenMP)
- **Ideal for**: Large-scale translation evaluation

### 2. RAJAPerf (106 kernels)
- **Structure**: Each kernel has 6 implementations (CUDA, HIP, OpenMP, OpenMP Target, SYCL, Sequential)
- **Categories**: Basic, Apps, LCALS, PolyBench, Stream
- **Ideal for**: Comprehensive API coverage including OpenMP Target

### 3. Rodinia (22 kernels)
- **Structure**: Separate directories for CUDA, OpenCL, OpenMP
- **Coverage**: 19 kernels with CUDA+OpenCL+OpenMP
- **Ideal for**: Classic GPU benchmarks with OpenCL

### 4. CloverLeaf (16 kernels)
- **Structure**: 8+ API implementations per kernel
- **Coverage**: CUDA, HIP, SYCL, OpenMP, OpenACC, Kokkos, TBB
- **Ideal for**: Portability layer comparison (Kokkos vs RAJA vs native)

### 5. BabelStream (5 kernels)
- **Structure**: 11+ API implementations of same 5 memory operations
- **Coverage**: Most APIs including stdpar, Thrust
- **Ideal for**: Simple, clean translation pairs

### 6. miniBUDE (1 kernel)
- **Structure**: 13 API implementations of molecular docking kernel
- **Coverage**: Widest API coverage
- **Ideal for**: Complex kernel translation

## Recommended Rosetta Stone Subsets

### Tier 1: High Priority (>100 kernels, 4+ APIs)
1. **HeCBench 4-API subset**: 325 kernels × 4 APIs = 1300 implementations
2. **RAJAPerf full suite**: 106 kernels × 6 APIs = 636 implementations

### Tier 2: Medium Priority (Clean multi-API)
3. **CloverLeaf**: 16 kernels × 8 APIs = 128 implementations
4. **Rodinia 3-API subset**: 19 kernels × 3 APIs = 57 implementations

### Tier 3: Reference (Canonical translations)
5. **BabelStream**: 5 kernels × 11 APIs = 55 implementations
6. **miniBUDE**: 1 kernel × 13 APIs = 13 implementations

## Total Translation Pairs Available

| Translation Type | Count |
|-----------------|-------|
| Total unique kernels | 656 |
| Total kernel-API implementations | 2,717 |
| CUDA→HIP translation pairs | 633 |
| CUDA→SYCL translation pairs | 616 |
| CUDA→OpenMP translation pairs | 472 |
| Any GPU→CPU translation pairs | ~600 |

