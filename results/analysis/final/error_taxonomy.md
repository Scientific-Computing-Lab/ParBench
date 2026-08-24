# Error Taxonomy — ParBench LLM Evaluation Results

**Total results:** 2344  
**PASS:** 1043 (44.5%)  
**Failures:** 1301 (55.5%)

## Table 1: Overall Status Distribution

| Status | Count | % of Total |
|--------|------:|----------:|
| PASS | 1043 | 44.5% |
| BUILD_FAIL | 480 | 20.5% |
| RUN_FAIL | 305 | 13.0% |
| VERIFY_FAIL | 329 | 14.0% |
| EXTRACTION_FAIL | 0 | 0.0% |
| ERROR | 184 | 7.8% |
| NOT_REPLAYABLE | 3 | 0.1% |
| **Total** | **2344** | **100.0%** |

## Table 2: BUILD_FAIL Root Cause Categories

*480 total BUILD_FAIL results classified into 9 categories.*

| # | Category | Count | % of BUILD_FAIL | Description |
|--:|----------|------:|----------------:|-------------|
| 1 | `linker_error` | 152 | 31.7% | Compilation succeeded but linking failed (undefined references, etc.) |
| 2 | `missing_header` | 138 | 28.7% | Missing or wrong #include directive — file not found |
| 3 | `undeclared_identifier` | 85 | 17.7% | Function, variable, or type not declared/defined in scope |
| 4 | `other_build` | 75 | 15.6% | Build failures not matching any specific pattern |
| 5 | `syntax_error` | 15 | 3.1% | Parse errors, malformed code, unexpected tokens |
| 6 | `retained_cuda_types` | 6 | 1.2% | LLM retained CUDA-specific types (float3, dim3, etc.) in non-CUDA target |
| 7 | `type_mismatch` | 5 | 1.0% | Type mismatches, wrong function signatures, incompatible conversions |
| 8 | `redefinition` | 3 | 0.6% | Duplicate or conflicting definitions of types/functions/variables |
| 9 | `retained_opencl_api` | 1 | 0.2% | LLM retained OpenCL API calls in non-OpenCL target code |
| | **Total** | **480** | **100.0%** | |

## Table 3: RUN_FAIL Root Cause Categories

*305 total RUN_FAIL results classified into 8 categories.*

| # | Category | Count | % of RUN_FAIL | Description |
|--:|----------|------:|-------------:|-------------|
| 1 | `wrong_exit_code` | 135 | 44.3% | Non-zero exit code without crash signal or other specific error pattern |
| 2 | `wrong_args` | 59 | 19.3% | Translated binary expects different arguments than spec provides |
| 3 | `simulation_mode_unsupported` | 47 | 15.4% | XSBench translated code used wrong simulation mode (history vs event) |
| 4 | `opencl_jit_error` | 35 | 11.5% | OpenCL runtime kernel compilation failed (clBuildProgram errors) |
| 5 | `segfault` | 19 | 6.2% | Process killed by SIGSEGV (signal 11) — null pointer or buffer overflow |
| 6 | `wrong_checksum` | 6 | 2.0% | Program ran to completion but output checksum was incorrect |
| 7 | `abort` | 3 | 1.0% | Process killed by SIGABRT — stack smashing, assertion failure, or abort() |
| 8 | `gpu_memory_error` | 1 | 0.3% | CUDA illegal memory access during kernel execution |
| | **Total** | **305** | **100.0%** | |

## Table 4: EXTRACTION_FAIL Root Cause Categories

*0 total EXTRACTION_FAIL results classified into 0 categories.*

| # | Category | Count | % of EXTRACTION_FAIL | Description |
|--:|----------|------:|--------------------:|-------------|
| | **Total** | **0** | **100.0%** | |

## Table 4b: VERIFY_FAIL Root Cause Categories

*329 total VERIFY_FAIL results classified into 2 categories.*

| # | Category | Count | % of VERIFY_FAIL | Description |
|--:|----------|------:|----------------:|-------------|
| 1 | `wrong_numerical_output` | 291 | 88.4% | Program produced output that did not match the expected stdout pattern |
| 2 | `missing_output` | 38 | 11.6% | Program produced no stdout output (empty or whitespace-only) |
| | **Total** | **329** | **100.0%** | |

## Table 5: BUILD_FAIL Categories by Model

| Category | azure-gpt-5.3-codex | azure-gpt-5.4 | together-qwen-3.5-397b-a17b | Total |
|----------|------:|------:|------:|------:|
| `linker_error` | 58 | 56 | 38 | 152 |
| `missing_header` | 46 | 36 | 56 | 138 |
| `undeclared_identifier` | 13 | 11 | 61 | 85 |
| `other_build` | 8 | 12 | 55 | 75 |
| `syntax_error` | 3 | 0 | 12 | 15 |
| `retained_cuda_types` | 1 | 0 | 5 | 6 |
| `type_mismatch` | 1 | 0 | 4 | 5 |
| `redefinition` | 0 | 0 | 3 | 3 |
| `retained_opencl_api` | 0 | 0 | 1 | 1 |
| **Total** | **130** | **115** | **235** | **480** |

## Table 6: BUILD_FAIL Categories by Direction

| Category | opencl→cuda | omp→cuda | opencl→omp | cuda→omp | cuda→omp_target | omp_target→cuda | omp→omp_target | omp_target→omp | Total |
|----------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| `linker_error` | 52 | 41 | 25 | 34 | 0 | 0 | 0 | 0 | 152 |
| `missing_header` | 47 | 34 | 30 | 27 | 0 | 0 | 0 | 0 | 138 |
| `undeclared_identifier` | 21 | 29 | 17 | 15 | 2 | 1 | 0 | 0 | 85 |
| `other_build` | 11 | 19 | 7 | 6 | 21 | 8 | 2 | 1 | 75 |
| `syntax_error` | 2 | 2 | 0 | 0 | 3 | 7 | 1 | 0 | 15 |
| `retained_cuda_types` | 0 | 0 | 4 | 1 | 1 | 0 | 0 | 0 | 6 |
| `type_mismatch` | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| `redefinition` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `retained_opencl_api` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Total** | **133** | **128** | **89** | **83** | **27** | **16** | **3** | **1** | **480** |

## Table 7: RUN_FAIL Categories by Model

| Category | azure-gpt-5.3-codex | azure-gpt-5.4 | together-qwen-3.5-397b-a17b | Total |
|----------|------:|------:|------:|------:|
| `wrong_exit_code` | 52 | 47 | 36 | 135 |
| `wrong_args` | 21 | 21 | 17 | 59 |
| `simulation_mode_unsupported` | 17 | 20 | 10 | 47 |
| `opencl_jit_error` | 5 | 4 | 26 | 35 |
| `segfault` | 2 | 2 | 15 | 19 |
| `wrong_checksum` | 0 | 0 | 6 | 6 |
| `abort` | 1 | 1 | 1 | 3 |
| `gpu_memory_error` | 1 | 0 | 0 | 1 |
| **Total** | **99** | **95** | **111** | **305** |

## Table 8: Complete Status Distribution by Model

| Model | Total | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL | ERROR | Pass Rate |
|-------|------:|-----:|----------:|--------:|----------:|--------------:|------:|----------:|
| azure-gpt-5.3-codex | 814 | 412 | 130 | 99 | 131 | 0 | 42 | 50.6% |
| azure-gpt-5.4 | 822 | 440 | 115 | 95 | 131 | 0 | 41 | 53.5% |
| together-qwen-3.5-397b-a17b | 708 | 191 | 235 | 111 | 67 | 0 | 104 | 27.0% |

## Table 9: Complete Status Distribution by Direction

| Direction | Total | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL | ERROR | Pass Rate |
|-----------|------:|-----:|----------:|--------:|----------:|--------------:|------:|----------:|
| cuda→omp | 438 | 273 | 83 | 7 | 50 | 0 | 25 | 62.3% |
| cuda→omp_target | 142 | 99 | 27 | 3 | 7 | 0 | 6 | 69.7% |
| cuda→opencl | 295 | 89 | 0 | 122 | 55 | 0 | 29 | 30.2% |
| omp→cuda | 374 | 170 | 128 | 9 | 44 | 0 | 23 | 45.5% |
| omp→omp_target | 69 | 47 | 3 | 6 | 7 | 0 | 6 | 68.1% |
| omp→opencl | 314 | 41 | 0 | 138 | 101 | 0 | 34 | 13.1% |
| omp_target→cuda | 174 | 149 | 16 | 1 | 2 | 0 | 6 | 85.6% |
| omp_target→omp | 69 | 62 | 1 | 0 | 0 | 0 | 6 | 89.9% |
| opencl→cuda | 223 | 39 | 133 | 4 | 17 | 0 | 30 | 17.5% |
| opencl→omp | 246 | 74 | 89 | 15 | 46 | 0 | 22 | 30.1% |

## Table 10: Per-Kernel Pass Rate and Primary Failure Mode

| Kernel | Total | PASS | Pass Rate | Primary Failure Mode | Primary Failure Count |
|--------|------:|-----:|----------:|---------------------|---------------------:|
| backprop | 38 | 7 | 18.4% | `api_error` | 16 |
| bfs | 110 | 53 | 48.2% | `wrong_numerical_output` | 35 |
| bptree | 74 | 28 | 37.8% | `missing_header` | 33 |
| cfd | 90 | 61 | 67.8% | `wrong_numerical_output` | 18 |
| convolution1d | 38 | 21 | 55.3% | `wrong_numerical_output` | 9 |
| dwt2d | 26 | 0 | 0.0% | `wrong_exit_code` | 17 |
| floydwarshall | 122 | 110 | 90.2% | `other_build` | 4 |
| gaussian | 18 | 0 | 0.0% | `wrong_exit_code` | 8 |
| heartwall | 62 | 9 | 14.5% | `wrong_numerical_output` | 18 |
| heat2d | 122 | 108 | 88.5% | `segfault` | 5 |
| hotspot | 98 | 52 | 53.1% | `wrong_exit_code` | 21 |
| hotspot3d | 110 | 47 | 42.7% | `wrong_args` | 42 |
| hybridsort | 6 | 0 | 0.0% | `api_error` | 6 |
| iso2dfd | 122 | 116 | 95.1% | `other_build` | 4 |
| jacobi | 38 | 31 | 81.6% | `syntax_error` | 5 |
| kmeans | 18 | 0 | 0.0% | `api_error` | 18 |
| lavamd | 66 | 2 | 3.0% | `missing_header` | 35 |
| lud | 98 | 69 | 70.4% | `linker_error` | 11 |
| md | 38 | 31 | 81.6% | `other_build` | 3 |
| mixbench | 102 | 0 | 0.0% | `api_error` | 102 |
| mummergpu | 6 | 0 | 0.0% | `api_error` | 6 |
| myocyte | 62 | 0 | 0.0% | `missing_header` | 32 |
| nn | 46 | 0 | 0.0% | `wrong_numerical_output` | 28 |
| nqueen | 38 | 33 | 86.8% | `other_build` | 4 |
| nw | 102 | 25 | 24.5% | `wrong_exit_code` | 30 |
| page-rank | 38 | 31 | 81.6% | `other_build` | 6 |
| particlefilter | 106 | 53 | 50.0% | `wrong_numerical_output` | 36 |
| pathfinder | 98 | 0 | 0.0% | `missing_output` | 38 |
| rsbench | 94 | 35 | 37.2% | `linker_error` | 25 |
| scan | 46 | 28 | 60.9% | `api_error` | 12 |
| srad | 98 | 25 | 25.5% | `wrong_exit_code` | 28 |
| stencil1d | 54 | 37 | 68.5% | `api_error` | 12 |
| streamcluster | 66 | 0 | 0.0% | `wrong_exit_code` | 22 |
| xsbench | 94 | 31 | 33.0% | `linker_error` | 29 |

## Key Findings

### Top BUILD_FAIL Root Causes

1. **`linker_error`** — 152 (31.7% of BUILD_FAIL)
2. **`missing_header`** — 138 (28.7% of BUILD_FAIL)
3. **`undeclared_identifier`** — 85 (17.7% of BUILD_FAIL)

### Top RUN_FAIL Root Causes

1. **`wrong_exit_code`** — 135 (44.3% of RUN_FAIL)
2. **`wrong_args`** — 59 (19.3% of RUN_FAIL)
3. **`simulation_mode_unsupported`** — 47 (15.4% of RUN_FAIL)

### Top VERIFY_FAIL Root Causes

1. **`wrong_numerical_output`** — 291 (88.4% of VERIFY_FAIL)
2. **`missing_output`** — 38 (11.6% of VERIFY_FAIL)

### Direction Asymmetry

- cuda→omp: 273/438 PASS (62.3%)
- cuda→omp_target: 99/142 PASS (69.7%)
- cuda→opencl: 89/295 PASS (30.2%)
- omp→cuda: 170/374 PASS (45.5%)
- omp→omp_target: 47/69 PASS (68.1%)
- omp→opencl: 41/314 PASS (13.1%)
- omp_target→cuda: 149/174 PASS (85.6%)
- omp_target→omp: 62/69 PASS (89.9%)
- opencl→cuda: 39/223 PASS (17.5%)
- opencl→omp: 74/246 PASS (30.1%)

### Compound Failures (Multiple Root Causes)

**149** results (11.5% of failures) have secondary error categories.

| Primary + Secondary | Count |
|---------------------|------:|
| missing_header + implicit_declaration | 68 |
| missing_header + linker_error | 30 |
| missing_header + undeclared_identifier | 24 |
| undeclared_identifier + redefinition | 15 |
| undeclared_identifier + syntax_error | 13 |
| linker_error + implicit_declaration | 8 |
| wrong_checksum + opencl_jit_error | 6 |
| missing_header + syntax_error | 4 |
| retained_cuda_types + type_mismatch | 3 |
| undeclared_identifier + type_mismatch | 3 |

---

## Canonical Taxonomy (L0)

**Total:** 1356  
**PASS:** 456 (33.6%)  
**Failures:** 900 (66.4%)

| Status | Count | % |
|--------|------:|--:|
| PASS | 456 | 33.6% |
| BUILD_FAIL | 399 | 29.4% |
| RUN_FAIL | 172 | 12.7% |
| VERIFY_FAIL | 194 | 14.3% |
| ERROR | 132 | 9.7% |
| NOT_REPLAYABLE | 3 | 0.2% |

## Augmentation Taxonomy (L1-L4)

**Total:** 988  
**PASS:** 587 (59.4%)  
**Failures:** 401 (40.6%)

| Level | Total | PASS | Pass Rate |
|-------|------:|-----:|----------:|
| L1 | 247 | 149 | 60.3% |
| L2 | 247 | 150 | 60.7% |
| L3 | 247 | 146 | 59.1% |
| L4 | 247 | 142 | 57.5% |

## Comparison: Canonical vs Augmentation

- Canonical pass rate: 33.6%
- Augmentation pass rate: 59.4%
- Delta: +25.8 pp
- *Augmentation runs only on tasks that passed canonical, so higher pass rate is expected*
