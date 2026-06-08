# ParBench Benchmark Characterization

Generated: 2026-05-04T19:49:31.429678+00:00

## 1. Source Lines of Code (SLoC)

Physical SLoC of CUDA source files (prompt_payload) per corpus kernel.

| Kernel | Suite | CUDA SLoC | OMP SLoC | Ratio | Src Files | Tgt Files |
|--------|-------|----------:|---------:|------:|----------:|----------:|
| myocyte | rodinia | 3,304 | 1806 | 0.55 | 16 | 2 |
| mummergpu | rodinia | 2,773 | 5325 | 1.92 | 3 | 2 |
| cfd | rodinia | 1,955 | 400 | 0.2 | 4 | 1 |
| xsbench | xsbench | 1,390 | 1238 | 0.89 | 6 | 2 |
| dwt2d | rodinia | 1,238 | N/A | N/A | 8 | 7 |
| heartwall | rodinia | 1,046 | 837 | 0.8 | 3 | 2 |
| particlefilter | rodinia | 1,023 | 400 | 0.39 | 2 | 1 |
| rsbench | rsbench | 1,016 | 1092 | 1.07 | 6 | 2 |
| huffman | rodinia | 686 | N/A | N/A | 7 | 6 |
| hybridsort | rodinia | 650 | N/A | N/A | 6 | 6 |
| srad | rodinia | 391 | 173 | 0.44 | 2 | 2 |
| streamcluster | rodinia | 372 | 981 | 2.64 | 2 | 2 |
| bptree | rodinia | 338 | 1721 | 5.09 | 5 | 4 |
| gaussian | rodinia | 329 | N/A | N/A | 1 | 1 |
| nw | rodinia | 319 | 291 | 0.91 | 2 | 2 |
| mixbench | mixbench | 312 | 297 | 0.95 | 4 | 1 |
| kmeans | rodinia | 299 | 1048 | 3.51 | 2 | 2 |
| lud | rodinia | 271 | 400 | 1.48 | 2 | 2 |
| nn | rodinia | 259 | 111 | 0.43 | 1 | 1 |
| hotspot3d | rodinia | 246 | 206 | 0.84 | 2 | 2 |
| hotspot | rodinia | 243 | 262 | 1.08 | 1 | 1 |
| bfs | rodinia | 242 | 144 | 0.6 | 3 | 3 |
| page-rank | hecbench | 235 | N/A | N/A | 2 | 1 |
| nqueen | hecbench | 209 | N/A | N/A | 1 | 1 |
| scan | hecbench | 206 | 189 | 0.92 | 1 | 1 |
| lavamd | rodinia | 200 | 258 | 1.29 | 3 | 2 |
| convolution1d | hecbench | 200 | N/A | N/A | 1 | 1 |
| backprop | rodinia | 197 | 449 | 2.28 | 2 | 2 |
| iso2dfd | hecbench | 196 | 170 | 0.87 | 1 | 1 |
| pathfinder | rodinia | 195 | 103 | 0.53 | 1 | 1 |
| floydwarshall | hecbench | 164 | 153 | 0.93 | 1 | 1 |
| jacobi | hecbench | 129 | N/A | N/A | 1 | 1 |
| heat2d | hecbench | 125 | 122 | 0.98 | 1 | 1 |
| md | hecbench | 107 | N/A | N/A | 1 | 1 |
| stencil1d | hecbench | 80 | 71 | 0.89 | 1 | 1 |

### SLoC Summary Statistics

| Metric | Value |
|--------|------:|
| Total Kernels | 35 |
| Min SLoC | 80 |
| Max SLoC | 3,304 |
| Mean SLoC | 598.4 |
| Median SLoC | 271 |
| Std Dev | 736.0 |
| Total SLoC | 20,945 |

### SLoC Distribution

| Range | Count |
|-------|------:|
| <100 | 1 |
| 100-500 | 24 |
| 500-1000 | 2 |
| >1000 | 8 |

### OMP/CUDA SLoC Ratio

Computed over 26 kernels with both CUDA and OMP specs.

| Metric | Value |
|--------|------:|
| Min Ratio | 0.2 |
| Max Ratio | 5.09 |
| Median Ratio | 0.92 |
| Mean Ratio | 1.25 |

## 2. Domain Category Distribution

| Category | Kernels | Suites |
|----------|--------:|--------|
| crypto | 5 | hecbench(5) |
| financial | 2 | hecbench(2) |
| graph | 6 | hecbench(5), rodinia(1) |
| image | 4 | hecbench(2), rodinia(2) |
| linear_algebra | 5 | hecbench(5), rodinia(2) |
| ml | 8 | hecbench(7), rodinia(2) |
| molecular_dynamics | 2 | hecbench(1), rodinia(1) |
| other | 30 | hecbench(23), mixbench(1), rodinia(7) |
| physics | 15 | hecbench(7), rodinia(6), rsbench(1), xsbench(1) |
| reduction | 1 | hecbench(1) |
| sort | 3 | hecbench(2), rodinia(1) |
| stencil | 6 | hecbench(6) |
| **Total** | **87** | |

## 3. API Coverage Cross-Tab

Distinct kernel count per (suite, API) cell.

| Suite | cuda | omp | opencl | omp_target | Total |
|-------|------:|------:|------:|------:|------:|
| rodinia | 22 | 18 | 20 | 0 | 60 |
| hecbench | 65 | 60 | 0 | 10 | 135 |
| xsbench | 1 | 1 | 1 | 1 | 4 |
| rsbench | 1 | 1 | 1 | 1 | 4 |
| mixbench | 1 | 1 | 1 | 0 | 3 |
| **Total** | 90 | 81 | 23 | 12 | 206 |

## 4. Multi-File Translation Breakdown

Classification based on `translation_targets` count (>1 = multi-file).

- **Single-file:** 17 kernels
- **Multi-file:** 18 kernels (51.4%)

### Per-Kernel Detail

| Kernel | Suite | Payload Files | Target Files | Multi-File |
|--------|-------|-------------:|-------------:|:----------:|
| dwt2d | rodinia | 8 | 7 | Yes |
| huffman | rodinia | 7 | 6 | Yes |
| hybridsort | rodinia | 6 | 6 | Yes |
| bptree | rodinia | 5 | 4 | Yes |
| bfs | rodinia | 3 | 3 | Yes |
| backprop | rodinia | 2 | 2 | Yes |
| heartwall | rodinia | 3 | 2 | Yes |
| hotspot3d | rodinia | 2 | 2 | Yes |
| kmeans | rodinia | 2 | 2 | Yes |
| lavamd | rodinia | 3 | 2 | Yes |
| lud | rodinia | 2 | 2 | Yes |
| mummergpu | rodinia | 3 | 2 | Yes |
| myocyte | rodinia | 16 | 2 | Yes |
| nw | rodinia | 2 | 2 | Yes |
| srad | rodinia | 2 | 2 | Yes |
| streamcluster | rodinia | 2 | 2 | Yes |
| xsbench | xsbench | 6 | 2 | Yes |
| rsbench | rsbench | 6 | 2 | Yes |
| cfd | rodinia | 4 | 1 | No |
| gaussian | rodinia | 1 | 1 | No |
| hotspot | rodinia | 1 | 1 | No |
| nn | rodinia | 1 | 1 | No |
| particlefilter | rodinia | 2 | 1 | No |
| pathfinder | rodinia | 1 | 1 | No |
| convolution1d | hecbench | 1 | 1 | No |
| floydwarshall | hecbench | 1 | 1 | No |
| heat2d | hecbench | 1 | 1 | No |
| iso2dfd | hecbench | 1 | 1 | No |
| jacobi | hecbench | 1 | 1 | No |
| md | hecbench | 1 | 1 | No |
| nqueen | hecbench | 1 | 1 | No |
| page-rank | hecbench | 2 | 1 | No |
| scan | hecbench | 1 | 1 | No |
| stencil1d | hecbench | 1 | 1 | No |
| mixbench | mixbench | 4 | 1 | No |

### By Suite

| Suite | Single | Multi |
|-------|-------:|------:|
| hecbench | 10 | 0 |
| mixbench | 1 | 0 |
| rodinia | 6 | 16 |
| rsbench | 0 | 1 |
| xsbench | 0 | 1 |

## 5. Language Feature Tiers

Features detected via regex grep of source directories.

| Kernel | Suite | CUDA Tier | OMP Tier | OpenCL Tier |
|--------|-------|-----------|----------|-------------|
| backprop | rodinia | cuda_basic | omp_basic | opencl_1x |
| bfs | rodinia | cuda_basic | omp_target | opencl_1x |
| bptree | rodinia | cuda_basic | omp_basic | opencl_1x |
| cfd | rodinia | cuda_basic | omp_target | opencl_1x |
| convolution1d | hecbench | cuda_basic | omp_target | N/A |
| dwt2d | rodinia | cuda_basic | N/A | opencl_1x |
| floydwarshall | hecbench | cuda_basic | omp_target | N/A |
| gaussian | rodinia | cuda_basic | N/A | opencl_1x |
| heartwall | rodinia | cuda_basic | omp_basic | opencl_1x |
| heat2d | hecbench | cuda_basic | omp_target | N/A |
| hotspot | rodinia | cuda_basic | omp_target | opencl_1x |
| hotspot3d | rodinia | cuda_basic | omp_basic | opencl_1x |
| huffman | rodinia | cuda_basic | N/A | N/A |
| hybridsort | rodinia | cuda_basic | N/A | opencl_1x |
| iso2dfd | hecbench | cuda_basic | omp_target | N/A |
| jacobi | hecbench | cuda_basic | omp_target | N/A |
| kmeans | rodinia | cuda_basic | omp_basic | opencl_1x |
| lavamd | rodinia | cuda_basic | omp_basic | opencl_1x |
| lud | rodinia | cuda_basic | omp_target | opencl_1x |
| md | hecbench | cuda_basic | omp_target | N/A |
| mixbench | mixbench | cuda_basic | omp_4.5 | opencl_1x |
| mummergpu | rodinia | cuda_basic | omp_basic | N/A |
| myocyte | rodinia | cuda_basic | omp_basic | opencl_1x |
| nn | rodinia | cuda_basic | omp_basic | opencl_1x |
| nqueen | hecbench | cuda_basic | omp_target | N/A |
| nw | rodinia | cuda_basic | omp_target | opencl_1x |
| page-rank | hecbench | cuda_basic | omp_target | N/A |
| particlefilter | rodinia | cuda_basic | omp_basic | opencl_1x |
| pathfinder | rodinia | cuda_basic | omp_basic | opencl_1x |
| rsbench | rsbench | cuda_library | omp_basic | opencl_1x |
| scan | hecbench | cuda_basic | omp_target | N/A |
| srad | rodinia | cuda_basic | omp_basic | opencl_1x |
| stencil1d | hecbench | cuda_basic | omp_target | N/A |
| streamcluster | rodinia | cuda_basic | omp_basic | opencl_1x |
| xsbench | xsbench | cuda_library | omp_basic | opencl_1x |

### Tier Distribution (across all API variants)

| Tier | Count |
|------|------:|
| cuda_basic | 33 |
| cuda_library | 2 |
| omp_4.5 | 1 |
| omp_basic | 15 |
| omp_target | 22 |
| opencl_1x | 23 |

## 6. Language Standard Distribution

Extracted from `implementation.language_standard` across all spec files.

| Standard | Count | Percentage |
|----------|------:|-----------:|
| C++11 | 2 | 1.0% |
| C++14 | 51 | 24.8% |
| C++17 | 128 | 62.1% |
| C11 | 6 | 2.9% |
| unspecified | 19 | 9.2% |
| **Total** | **206** | **100.0%** |

### By API

**cuda:**
  - C++11: 1
  - C++14: 24
  - C++17: 65

**omp:**
  - C++14: 8
  - C++17: 61
  - C11: 2
  - unspecified: 10

**omp_target:**
  - C++14: 8
  - C++17: 2
  - C11: 2

**opencl:**
  - C++11: 1
  - C++14: 11
  - C11: 2
  - unspecified: 9

## Summary

- **total_kernels:** 35
- **total_specs:** 206
- **total_categories:** 12
- **total_suites:** 5
- **total_apis:** 4
- **sloc_range:** 80-3304
- **sloc_median:** 271
- **multi_file_pct:** 51.4
