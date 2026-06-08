# ParBench Evaluation Summary
**Generated:** 2026-04-30 19:40  |  **Total tasks:** 2262

## Pass Rates by Model

| Model | PASS | Total | Rate | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | EXTRACTION_FAIL |
|-------|-----:|------:|-----:|----------:|--------:|------------:|---------------:|
| azure-gpt-5.3-codex | 604 | 814 | 74.2% | 139 | 44 | 27 | 0 |
| azure-gpt-5.4 | 621 | 822 | 75.5% | 123 | 43 | 32 | 3 |
| together-qwen-3.5-397b-a17b | 230 | 626 | 36.7% | 245 | 121 | 29 | 1 |

## Pass Rates by Translation Direction

| Direction | PASS | Total | Rate |
|-----------|-----:|------:|-----:|
| cuda-to-omp | 330 | 432 | 76.4% |
| cuda-to-omp_target | 106 | 136 | 77.9% |
| cuda-to-opencl | 170 | 283 | 60.1% |
| omp-to-cuda | 226 | 368 | 61.4% |
| omp-to-omp_target | 47 | 63 | 74.6% |
| omp-to-opencl | 210 | 301 | 69.8% |
| omp_target-to-cuda | 151 | 168 | 89.9% |
| omp_target-to-omp | 62 | 63 | 98.4% |
| opencl-to-cuda | 50 | 211 | 23.7% |
| opencl-to-omp | 103 | 237 | 43.5% |

## Pass Rates by Augmentation Level

| Level | PASS | Total | Rate |
|-------|-----:|------:|-----:|
| L0 | 636 | 1278 | 49.8% |
| L1 | 209 | 246 | 85.0% |
| L2 | 208 | 246 | 84.5% |
| L3 | 201 | 246 | 81.7% |
| L4 | 201 | 246 | 81.7% |

> **Scope note (level-invariance claim).** L1–L4 results reflect only kernels that passed L0 (conditional ablation — `derive_l0_passers.py` filter at `:79` uses pass@1-of-any on the canonical L0 cells). Any level-invariance observation here is scoped to the **L0-passer subset** by construction; it is NOT an unconditional statement about augmentation robustness on kernels that fail L0. See `.planning/_archive/phase-02-llm-eval-testing/02-THREATS-TO-VALIDITY.md` §4 and the Bucket D D1 unconditional probe for context.

## pass@1 Metrics (Dual Reporting)

Cells are `(kernel, direction, augment_level, model)` tuples. `of_any` = fraction of cells with ≥1 PASS across all samples. `mean` = average single-draw pass rate across cells (passes / samples, averaged over cells). Both metrics coincide when every cell is at 0 / k or k / k passes; they diverge structurally at any interior rate.

| Level | Cells | pass@1_of_any | pass@1_mean | divergence |
|-------|------:|--------------:|------------:|-----------:|
| L0 | 426 | 57.8% | 49.8% | +0.0798 |
| L1 | 246 | 85.0% | 85.0% | +0.0000 |
| L2 | 246 | 84.5% | 84.5% | +0.0000 |
| L3 | 246 | 81.7% | 81.7% | +0.0000 |
| L4 | 246 | 81.7% | 81.7% | +0.0000 |

## Pass Rates by Translation Complexity

| Complexity Class | PASS | Total | Rate | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL |
|-----------------|-----:|------:|-----:|----------:|--------:|------------:|
| multi_to_multi | 29 | 118 | 24.6% | 70 | 14 | 3 |
| multi_to_single | 301 | 438 | 68.7% | 50 | 85 | 2 |
| single_file | 1052 | 1404 | 74.9% | 184 | 104 | 64 |
| single_to_multi | 73 | 302 | 24.2% | 203 | 5 | 19 |

### Model × Complexity Cross-Tab

| Complexity Class | azure-gpt-5.3-codex | azure-gpt-5.4 | together-qwen-3.5-397b-a17b |
|-----------------|---:|---:|---:|
| multi_to_multi | 14/46 (30.4%) | 15/42 (35.7%) | 0/30 (0.0%) |
| multi_to_single | 123/158 (77.8%) | 141/166 (84.9%) | 37/114 (32.5%) |
| single_file | 437/508 (86.0%) | 427/500 (85.4%) | 188/396 (47.5%) |
| single_to_multi | 30/102 (29.4%) | 38/114 (33.3%) | 5/86 (5.8%) |

## Kernel × Model Matrix (cuda→omp, L0)

| Kernel | azure-gpt-5.3-codex | azure-gpt-5.4 | together-qwen-3.5-397b-a17b |
|--------|---|---|---|
| backprop | ✗ BUILD_FAIL | ✓ PASS | ✗ BUILD_FAIL |
| bfs | ✓ PASS | ✓ PASS | ✓ PASS |
| bptree | ✗ BUILD_FAIL | ✗ BUILD_FAIL | ✗ BUILD_FAIL |
| cfd | ✓ PASS | ✓ PASS | ✓ PASS |
| convolution1d | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| floydwarshall | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| heartwall | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| heat2d | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| hotspot | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| hotspot3d | ✓ PASS | ✓ PASS | ✓ PASS |
| iso2dfd | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| jacobi | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| lavamd | ✗ VERIFY_FAIL | ✓ PASS | ✗ BUILD_FAIL |
| lud | ✓ PASS | ✓ PASS | ✓ PASS |
| md | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| mixbench | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| myocyte | ✗ BUILD_FAIL | ✗ EXTRACTION_FAIL | ✗ BUILD_FAIL |
| nn | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| nqueen | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| nw | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| page-rank | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL |
| particlefilter | ✓ PASS | ✓ PASS | ✓ PASS |
| pathfinder | ✓ PASS | ✓ PASS | ✓ PASS |
| rsbench | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL |
| scan | ✓ PASS | ✓ PASS | ✗ RUN_FAIL |
| srad | ✓ PASS | ✓ PASS | ✓ PASS |
| stencil1d | ✓ PASS | ✓ PASS | ✓ PASS |
| streamcluster | ✓ PASS | ✓ PASS | ✗ RUN_FAIL |
| xsbench | ✓ PASS | ✗ BUILD_FAIL | ✗ BUILD_FAIL |

## Failure Taxonomy

| Status | Count |
|--------|------:|
| BUILD_FAIL | 507 |
| RUN_FAIL | 208 |
| VERIFY_FAIL | 88 |
| EXTRACTION_FAIL | 4 |

## Self-Repair (Iterative Retry) Effectiveness

- Tasks with recorded attempts: **2262**
- Passed on first attempt: **1455**
- Repaired by retry: **0**

