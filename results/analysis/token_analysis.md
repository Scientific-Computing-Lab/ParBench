# Token Usage Analysis — ParBench Evaluation

**626 results** across 1 models, 31 kernels, 10 directions.
Grand total (from result JSONs): **11,481,716 tokens** (7,394,516 input + 4,087,200 output), estimated cost: **$19.15**.

### Actual Billing (Together AI, Mar 27 – Apr 2 2026)

- **Input tokens:** 96,241,738 (96.2M)
- **Output tokens:** 24,324,474 (24.3M)
- **Total tokens:** 120,566,212 (120.6M)
- **Total cost:** $145.37
- **Requests:** ~4,600

Result JSONs capture ~46% of billed tokens. The gap reflects system prompt overhead, HTTP-level retries, and JSON-mode formatting tokens billed by the provider.

## Table 1: Per-Model Token Statistics

| Model | N | Pass% | Prompt (mean) | Completion (mean) | tok/s (mean) | Total Cost | Cost/PASS |
|-------|--:|------:|--------------:|------------------:|-------------:|-----------:|----------:|
| Qwen 3.5 397B (Together) | 626 | 36.7% | 11,812 | 6,529 | 110 | $19.15 | $0.0235 |

## Table 2: Per-Kernel Token Statistics (sorted by prompt size)

| Kernel | N | Pass% | Prompt (mean) | Completion (mean) |
|--------|--:|------:|--------------:|------------------:|
| myocyte | 18 | 0.0% | 45,667 | 20,750 |
| heartwall | 18 | 0.0% | 37,232 | 13,027 |
| cfd | 22 | 27.3% | 28,710 | 8,036 |
| xsbench | 22 | 4.5% | 27,738 | 6,315 |
| dwt2d | 6 | 0.0% | 27,408 | 11,110 |
| rsbench | 18 | 0.0% | 25,959 | 6,029 |
| particlefilter | 26 | 34.6% | 22,718 | 8,208 |
| streamcluster | 22 | 4.5% | 22,065 | 7,712 |
| bptree | 18 | 0.0% | 20,297 | 6,516 |
| gaussian | 6 | 0.0% | 15,472 | 4,942 |
| lud | 30 | 46.7% | 11,051 | 5,219 |
| srad | 26 | 34.6% | 10,949 | 5,233 |
| mixbench | 22 | 9.1% | 9,221 | 5,662 |
| lavamd | 18 | 0.0% | 8,945 | 4,618 |
| bfs | 34 | 44.1% | 8,596 | 4,283 |
| nw | 30 | 36.7% | 8,355 | 7,913 |
| backprop | 6 | 0.0% | 6,777 | 4,936 |
| hotspot3d | 34 | 58.8% | 6,315 | 5,542 |
| hotspot | 30 | 46.7% | 6,139 | 6,182 |
| pathfinder | 30 | 26.7% | 3,733 | 5,675 |
| md | 10 | 40.0% | 3,376 | 5,163 |
| iso2dfd | 38 | 84.2% | 2,928 | 5,411 |
| convolution1d | 10 | 20.0% | 2,770 | 6,357 |
| nn | 6 | 0.0% | 2,722 | 4,876 |
| page-rank | 10 | 60.0% | 2,420 | 4,798 |
| nqueen | 10 | 60.0% | 2,420 | 5,931 |
| scan | 6 | 0.0% | 2,348 | 6,894 |
| floydwarshall | 38 | 76.3% | 1,876 | 4,576 |
| heat2d | 38 | 76.3% | 1,786 | 4,136 |
| jacobi | 10 | 30.0% | 1,423 | 4,574 |
| stencil1d | 14 | 64.3% | 1,074 | 8,976 |

## Table 3: Per-Direction Token Statistics

| Direction | N | Pass% | Prompt (mean) | Completion (mean) |
|-----------|--:|------:|--------------:|------------------:|
| cuda-to-omp | 120 | 55.8% | 9,562 | 7,253 |
| omp-to-cuda | 104 | 37.5% | 7,634 | 7,574 |
| omp-to-opencl | 87 | 39.1% | 16,092 | 4,418 |
| cuda-to-opencl | 69 | 11.6% | 18,514 | 4,498 |
| opencl-to-omp | 67 | 16.4% | 19,720 | 8,838 |
| opencl-to-cuda | 57 | 0.0% | 20,581 | 9,357 |
| omp_target-to-cuda | 56 | 71.4% | 2,315 | 5,369 |
| cuda-to-omp_target | 24 | 0.0% | 2,528 | 5,309 |
| omp-to-omp_target | 21 | 52.4% | 2,185 | 4,207 |
| omp_target-to-omp | 21 | 95.2% | 2,151 | 4,406 |

## Table 4: Per-Augmentation-Level Statistics

| Level | N | Pass% | Prompt (mean) | Completion (mean) |
|-------|--:|------:|--------------:|------------------:|
| L0 | 426 | 23.9% | 14,026 | 7,113 |
| L1 | 50 | 74.0% | 7,053 | 5,186 |
| L2 | 50 | 64.0% | 7,053 | 5,797 |
| L3 | 50 | 62.0% | 7,063 | 5,294 |
| L4 | 50 | 56.0% | 7,220 | 4,864 |

## Table 5: Cost Analysis

| Model | Total Cost | Cost on PASS | Cost on FAIL | Cost/PASS | Cost/Task |
|-------|----------:|-----------:|-----------:|----------:|----------:|
| Qwen 3.5 397B (Together) | $19.15 | $5.41 | $13.74 | $0.0235 | $0.0306 |
| **TOTAL** | **$19.15** | | | | |

## Correlations

- **Kernel-level (prompt)**: Spearman(mean prompt tokens, pass rate) = **-0.5786**
- **Result-level (prompt)**: Mean prompt tokens for PASS = **5,692**, for FAIL = **15,367**
- **Result-level (completion)**: Spearman(completion tokens, pass) = **-0.0207**; Mean completion for PASS = **5,589**, for FAIL = **7,075**

