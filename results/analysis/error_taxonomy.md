# Error Taxonomy — ParBench LLM Evaluation Results

**Total results:** 2344  
**PASS:** 1468 (62.6%)  
**Failures:** 876 (37.4%)

## Table 1: Overall Status Distribution

| Status | Count | % of Total |
|--------|------:|----------:|
| PASS | 1468 | 62.6% |
| BUILD_FAIL | 552 | 23.5% |
| RUN_FAIL | 218 | 9.3% |
| VERIFY_FAIL | 102 | 4.4% |
| EXTRACTION_FAIL | 4 | 0.2% |
| ERROR | 0 | 0.0% |
| **Total** | **2344** | **100.0%** |

## Table 2: BUILD_FAIL Root Cause Categories

*552 total BUILD_FAIL results classified into 11 categories.*

| # | Category | Count | % of BUILD_FAIL | Description |
|--:|----------|------:|----------------:|-------------|
| 1 | `linker_error` | 150 | 27.2% | Compilation succeeded but linking failed (undefined references, etc.) |
| 2 | `missing_header` | 149 | 27.0% | Missing or wrong #include directive — file not found |
| 3 | `other_build` | 97 | 17.6% | Build failures not matching any specific pattern |
| 4 | `undeclared_identifier` | 95 | 17.2% | Function, variable, or type not declared/defined in scope |
| 5 | `missing_target_api` | 21 | 3.8% | LLM translation contains no target API constructs (no pragmas/kernels for target language) |
| 6 | `syntax_error` | 19 | 3.4% | Parse errors, malformed code, unexpected tokens |
| 7 | `type_mismatch` | 8 | 1.4% | Type mismatches, wrong function signatures, incompatible conversions |
| 8 | `retained_cuda_types` | 6 | 1.1% | LLM retained CUDA-specific types (float3, dim3, etc.) in non-CUDA target |
| 9 | `redefinition` | 4 | 0.7% | Duplicate or conflicting definitions of types/functions/variables |
| 10 | `implicit_declaration` | 2 | 0.4% | Missing function prototypes — implicit declaration warnings promoted to errors |
| 11 | `retained_opencl_api` | 1 | 0.2% | LLM retained OpenCL API calls in non-OpenCL target code |
| | **Total** | **552** | **100.0%** | |

## Table 3: RUN_FAIL Root Cause Categories

*218 total RUN_FAIL results classified into 7 categories.*

| # | Category | Count | % of RUN_FAIL | Description |
|--:|----------|------:|-------------:|-------------|
| 1 | `wrong_exit_code` | 81 | 37.2% | Non-zero exit code without crash signal or other specific error pattern |
| 2 | `opencl_jit_error` | 81 | 37.2% | OpenCL runtime kernel compilation failed (clBuildProgram errors) |
| 3 | `segfault` | 27 | 12.4% | Process killed by SIGSEGV (signal 11) — null pointer or buffer overflow |
| 4 | `wrong_checksum` | 16 | 7.3% | Program ran to completion but output checksum was incorrect |
| 5 | `simulation_mode_unsupported` | 9 | 4.1% | XSBench translated code used wrong simulation mode (history vs event) |
| 6 | `abort` | 3 | 1.4% | Process killed by SIGABRT — stack smashing, assertion failure, or abort() |
| 7 | `gpu_memory_error` | 1 | 0.5% | CUDA illegal memory access during kernel execution |
| | **Total** | **218** | **100.0%** | |

## Table 4: EXTRACTION_FAIL Root Cause Categories

*4 total EXTRACTION_FAIL results classified into 2 categories.*

| # | Category | Count | % of EXTRACTION_FAIL | Description |
|--:|----------|------:|--------------------:|-------------|
| 1 | `no_code_blocks` | 3 | 75.0% | LLM response did not contain parseable code for expected target files |
| 2 | `missing_files` | 1 | 25.0% | Expected output files not found in LLM response |
| | **Total** | **4** | **100.0%** | |

## Table 4b: VERIFY_FAIL Root Cause Categories

*102 total VERIFY_FAIL results classified into 2 categories.*

| # | Category | Count | % of VERIFY_FAIL | Description |
|--:|----------|------:|----------------:|-------------|
| 1 | `wrong_numerical_output` | 97 | 95.1% | Program produced output that did not match the expected stdout pattern |
| 2 | `pass_overall_mislabel` | 5 | 4.9% | Verify passed but overall_status is VERIFY_FAIL (pipeline labeling bug) |
| | **Total** | **102** | **100.0%** | |

## Table 5: BUILD_FAIL Categories by Model

| Category | azure-gpt-5.3-codex | azure-gpt-5.4 | together-qwen-3.5-397b-a17b | Total |
|----------|------:|------:|------:|------:|
| `linker_error` | 57 | 57 | 36 | 150 |
| `missing_header` | 49 | 35 | 65 | 149 |
| `other_build` | 9 | 14 | 74 | 97 |
| `undeclared_identifier` | 13 | 16 | 66 | 95 |
| `missing_target_api` | 6 | 0 | 15 | 21 |
| `syntax_error` | 2 | 1 | 16 | 19 |
| `type_mismatch` | 2 | 0 | 6 | 8 |
| `retained_cuda_types` | 1 | 0 | 5 | 6 |
| `redefinition` | 0 | 0 | 4 | 4 |
| `implicit_declaration` | 0 | 0 | 2 | 2 |
| `retained_opencl_api` | 0 | 0 | 1 | 1 |
| **Total** | **139** | **123** | **290** | **552** |

## Table 6: BUILD_FAIL Categories by Direction

| Category | opencl→cuda | omp→cuda | opencl→omp | cuda→omp | cuda→omp_target | omp_target→cuda | omp→omp_target | omp_target→omp | Total |
|----------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| `linker_error` | 50 | 39 | 28 | 33 | 0 | 0 | 0 | 0 | 150 |
| `missing_header` | 54 | 33 | 31 | 31 | 0 | 0 | 0 | 0 | 149 |
| `other_build` | 15 | 25 | 8 | 7 | 23 | 10 | 6 | 3 | 97 |
| `undeclared_identifier` | 25 | 24 | 27 | 15 | 3 | 1 | 0 | 0 | 95 |
| `missing_target_api` | 7 | 9 | 3 | 2 | 0 | 0 | 0 | 0 | 21 |
| `syntax_error` | 1 | 2 | 0 | 3 | 5 | 7 | 1 | 0 | 19 |
| `type_mismatch` | 0 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 8 |
| `retained_cuda_types` | 0 | 0 | 4 | 1 | 1 | 0 | 0 | 0 | 6 |
| `redefinition` | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| `implicit_declaration` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `retained_opencl_api` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Total** | **153** | **137** | **109** | **93** | **32** | **18** | **7** | **3** | **552** |

## Table 7: RUN_FAIL Categories by Model

| Category | azure-gpt-5.3-codex | azure-gpt-5.4 | together-qwen-3.5-397b-a17b | Total |
|----------|------:|------:|------:|------:|
| `wrong_exit_code` | 24 | 21 | 36 | 81 |
| `opencl_jit_error` | 11 | 13 | 57 | 81 |
| `segfault` | 3 | 2 | 22 | 27 |
| `wrong_checksum` | 1 | 0 | 15 | 16 |
| `simulation_mode_unsupported` | 3 | 6 | 0 | 9 |
| `abort` | 1 | 1 | 1 | 3 |
| `gpu_memory_error` | 1 | 0 | 0 | 1 |
| **Total** | **44** | **43** | **131** | **218** |

## Table 8: Complete Status Distribution by Model

| Model | Total | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL | ERROR | Pass Rate |
|-------|------:|-----:|----------:|--------:|----------:|--------------:|------:|----------:|
| azure-gpt-5.3-codex | 814 | 604 | 139 | 44 | 27 | 0 | 0 | 74.2% |
| azure-gpt-5.4 | 822 | 621 | 123 | 43 | 32 | 3 | 0 | 75.5% |
| together-qwen-3.5-397b-a17b | 708 | 243 | 290 | 131 | 43 | 1 | 0 | 34.3% |

## Table 9: Complete Status Distribution by Direction

| Direction | Total | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL | ERROR | Pass Rate |
|-----------|------:|-----:|----------:|--------:|----------:|--------------:|------:|----------:|
| cuda→omp | 438 | 330 | 93 | 7 | 6 | 2 | 0 | 75.3% |
| cuda→omp_target | 142 | 106 | 32 | 4 | 0 | 0 | 0 | 74.6% |
| cuda→opencl | 295 | 170 | 0 | 99 | 26 | 0 | 0 | 57.6% |
| omp→cuda | 374 | 226 | 137 | 9 | 1 | 1 | 0 | 60.4% |
| omp→omp_target | 69 | 49 | 7 | 6 | 7 | 0 | 0 | 71.0% |
| omp→opencl | 314 | 215 | 0 | 72 | 27 | 0 | 0 | 68.5% |
| omp_target→cuda | 174 | 154 | 18 | 1 | 1 | 0 | 0 | 88.5% |
| omp_target→omp | 69 | 64 | 3 | 0 | 2 | 0 | 0 | 92.8% |
| opencl→cuda | 223 | 51 | 153 | 5 | 13 | 1 | 0 | 22.9% |
| opencl→omp | 246 | 103 | 109 | 15 | 19 | 0 | 0 | 41.9% |

## Table 10: Per-Kernel Pass Rate and Primary Failure Mode

| Kernel | Total | PASS | Pass Rate | Primary Failure Mode | Primary Failure Count |
|--------|------:|-----:|----------:|---------------------|---------------------:|
| backprop | 38 | 12 | 31.6% | `linker_error` | 13 |
| bfs | 110 | 85 | 77.3% | `linker_error` | 9 |
| bptree | 74 | 26 | 35.1% | `missing_header` | 33 |
| cfd | 90 | 61 | 67.8% | `wrong_numerical_output` | 18 |
| convolution1d | 38 | 30 | 78.9% | `other_build` | 8 |
| dwt2d | 26 | 13 | 50.0% | `missing_header` | 8 |
| floydwarshall | 122 | 110 | 90.2% | `other_build` | 4 |
| gaussian | 18 | 0 | 0.0% | `wrong_exit_code` | 10 |
| heartwall | 62 | 9 | 14.5% | `linker_error` | 14 |
| heat2d | 122 | 108 | 88.5% | `segfault` | 5 |
| hotspot | 98 | 70 | 71.4% | `wrong_numerical_output` | 14 |
| hotspot3d | 110 | 79 | 71.8% | `linker_error` | 15 |
| hybridsort | 6 | 0 | 0.0% | `missing_header` | 3 |
| iso2dfd | 122 | 116 | 95.1% | `other_build` | 4 |
| jacobi | 38 | 31 | 81.6% | `syntax_error` | 5 |
| kmeans | 18 | 0 | 0.0% | `wrong_numerical_output` | 6 |
| lavamd | 66 | 14 | 21.2% | `missing_header` | 35 |
| lud | 98 | 69 | 70.4% | `linker_error` | 11 |
| md | 38 | 31 | 81.6% | `other_build` | 3 |
| mixbench | 102 | 61 | 59.8% | `wrong_exit_code` | 11 |
| mummergpu | 6 | 0 | 0.0% | `other_build` | 4 |
| myocyte | 62 | 12 | 19.4% | `missing_header` | 29 |
| nn | 46 | 29 | 63.0% | `missing_header` | 6 |
| nqueen | 38 | 33 | 86.8% | `other_build` | 4 |
| nw | 102 | 60 | 58.8% | `linker_error` | 18 |
| page-rank | 38 | 31 | 81.6% | `other_build` | 6 |
| particlefilter | 106 | 71 | 67.0% | `wrong_numerical_output` | 18 |
| pathfinder | 98 | 63 | 64.3% | `other_build` | 12 |
| rsbench | 94 | 49 | 52.1% | `linker_error` | 20 |
| scan | 46 | 30 | 65.2% | `other_build` | 8 |
| srad | 98 | 63 | 64.3% | `linker_error` | 18 |
| stencil1d | 54 | 42 | 77.8% | `other_build` | 6 |
| streamcluster | 66 | 15 | 22.7% | `wrong_exit_code` | 18 |
| xsbench | 94 | 45 | 47.9% | `linker_error` | 25 |

## Key Findings

### Top BUILD_FAIL Root Causes

1. **`linker_error`** — 150 (27.2% of BUILD_FAIL)
2. **`missing_header`** — 149 (27.0% of BUILD_FAIL)
3. **`other_build`** — 97 (17.6% of BUILD_FAIL)

### Top RUN_FAIL Root Causes

1. **`wrong_exit_code`** — 81 (37.2% of RUN_FAIL)
2. **`opencl_jit_error`** — 81 (37.2% of RUN_FAIL)
3. **`segfault`** — 27 (12.4% of RUN_FAIL)

### Top VERIFY_FAIL Root Causes

1. **`wrong_numerical_output`** — 97 (95.1% of VERIFY_FAIL)
2. **`pass_overall_mislabel`** — 5 (4.9% of VERIFY_FAIL)

### Direction Asymmetry

- cuda→omp: 330/438 PASS (75.3%)
- cuda→omp_target: 106/142 PASS (74.6%)
- cuda→opencl: 170/295 PASS (57.6%)
- omp→cuda: 226/374 PASS (60.4%)
- omp→omp_target: 49/69 PASS (71.0%)
- omp→opencl: 215/314 PASS (68.5%)
- omp_target→cuda: 154/174 PASS (88.5%)
- omp_target→omp: 64/69 PASS (92.8%)
- opencl→cuda: 51/223 PASS (22.9%)
- opencl→omp: 103/246 PASS (41.9%)

### Compound Failures (Multiple Root Causes)

**192** results (21.9% of failures) have secondary error categories.

| Primary + Secondary | Count |
|---------------------|------:|
| missing_header + implicit_declaration | 72 |
| missing_header + linker_error | 34 |
| missing_header + undeclared_identifier | 23 |
| wrong_checksum + opencl_jit_error | 15 |
| undeclared_identifier + redefinition | 14 |
| undeclared_identifier + syntax_error | 13 |
| missing_target_api + undeclared_identifier | 10 |
| missing_target_api + linker_error | 10 |
| linker_error + implicit_declaration | 9 |
| opencl_jit_error + data_file_missing | 8 |

---

## Canonical Taxonomy (L0)

**Total:** 1356  
**PASS:** 646 (47.6%)  
**Failures:** 710 (52.4%)

| Status | Count | % |
|--------|------:|--:|
| PASS | 646 | 47.6% |
| BUILD_FAIL | 462 | 34.1% |
| RUN_FAIL | 161 | 11.9% |
| VERIFY_FAIL | 83 | 6.1% |
| EXTRACTION_FAIL | 4 | 0.3% |

## Augmentation Taxonomy (L1-L4)

**Total:** 988  
**PASS:** 822 (83.2%)  
**Failures:** 166 (16.8%)

| Level | Total | PASS | Pass Rate |
|-------|------:|-----:|----------:|
| L1 | 247 | 210 | 85.0% |
| L2 | 247 | 209 | 84.6% |
| L3 | 247 | 202 | 81.8% |
| L4 | 247 | 201 | 81.4% |

## Comparison: Canonical vs Augmentation

- Canonical pass rate: 47.6%
- Augmentation pass rate: 83.2%
- Delta: +35.6 pp
- *Augmentation runs only on tasks that passed canonical, so higher pass rate is expected*
