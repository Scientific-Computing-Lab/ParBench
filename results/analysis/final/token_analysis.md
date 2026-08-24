# Token Usage Analysis — ParBench Evaluation

**2160 results** across 3 models, 30 kernels, 10 directions.
Grand total (from result JSONs): **38,042,087 tokens** (25,137,349 input + 12,904,738 output), estimated cost (priced models only, EXCLUDES unpriced azure-gpt-5.3-codex): **$119.41**; a complete pooled cost is not reported.

## Table 1: Per-Model Token Statistics

| Model | N | Pass% | Prompt (mean) | Completion (mean) | tok/s (mean) | Total Cost | Cost/PASS |
|-------|--:|------:|--------------:|------------------:|-------------:|-----------:|----------:|
| azure-gpt-5.3-codex | 772 | 53.4% | 11,529 | 4,829 | 92 | N/A | N/A |
| Azure GPT-5.4 | 784 | 56.1% | 11,537 | 6,651 | 57 | $100.83 | $0.0989 |
| Qwen 3.5 397B (Together) | 604 | 31.6% | 11,907 | 6,561 | 110 | $18.58 | $0.0240 |

## Table 2: Per-Kernel Token Statistics (sorted by prompt size)

| Kernel | N | Pass% | Prompt (mean) | Completion (mean) |
|--------|--:|------:|--------------:|------------------:|
| myocyte | 62 | 0.0% | 40,417 | 20,258 |
| heartwall | 62 | 14.5% | 34,618 | 13,777 |
| dwt2d | 26 | 0.0% | 27,495 | 11,094 |
| cfd | 90 | 67.8% | 27,469 | 8,070 |
| xsbench | 94 | 33.0% | 25,918 | 6,842 |
| rsbench | 94 | 37.2% | 23,801 | 6,809 |
| particlefilter | 106 | 50.0% | 21,281 | 8,434 |
| streamcluster | 66 | 0.0% | 18,830 | 8,800 |
| bptree | 74 | 37.8% | 18,506 | 7,188 |
| gaussian | 18 | 0.0% | 14,656 | 4,743 |
| lud | 98 | 70.4% | 10,572 | 4,947 |
| srad | 98 | 25.5% | 10,441 | 5,250 |
| lavamd | 66 | 3.0% | 8,516 | 3,837 |
| nw | 102 | 24.5% | 8,367 | 6,311 |
| bfs | 110 | 48.2% | 8,353 | 3,613 |
| backprop | 22 | 31.8% | 6,616 | 6,240 |
| hotspot3d | 110 | 42.7% | 5,763 | 4,813 |
| hotspot | 98 | 53.1% | 5,648 | 4,976 |
| pathfinder | 98 | 0.0% | 3,601 | 3,937 |
| md | 38 | 81.6% | 3,227 | 3,972 |
| iso2dfd | 122 | 95.1% | 2,767 | 4,228 |
| convolution1d | 38 | 55.3% | 2,558 | 5,157 |
| nn | 34 | 0.0% | 2,526 | 3,701 |
| page-rank | 38 | 81.6% | 2,313 | 4,244 |
| nqueen | 38 | 86.8% | 2,299 | 4,183 |
| scan | 34 | 82.3% | 2,185 | 4,564 |
| floydwarshall | 122 | 90.2% | 1,774 | 3,451 |
| heat2d | 122 | 88.5% | 1,691 | 3,135 |
| jacobi | 38 | 81.6% | 1,426 | 3,266 |
| stencil1d | 42 | 88.1% | 1,027 | 4,260 |

## Table 3: Per-Direction Token Statistics

| Direction | N | Pass% | Prompt (mean) | Completion (mean) |
|-----------|--:|------:|--------------:|------------------:|
| cuda-to-omp | 415 | 65.8% | 9,713 | 6,029 |
| omp-to-cuda | 351 | 48.4% | 7,520 | 6,845 |
| omp-to-opencl | 280 | 14.6% | 15,723 | 4,892 |
| cuda-to-opencl | 266 | 33.5% | 17,712 | 4,275 |
| opencl-to-omp | 224 | 33.0% | 18,917 | 8,026 |
| opencl-to-cuda | 194 | 20.1% | 21,497 | 10,516 |
| omp_target-to-cuda | 168 | 88.7% | 2,179 | 4,175 |
| cuda-to-omp_target | 136 | 72.8% | 2,369 | 4,035 |
| omp-to-omp_target | 63 | 74.6% | 2,056 | 3,419 |
| omp_target-to-omp | 63 | 98.4% | 2,026 | 3,010 |

## Table 4: Per-Augmentation-Level Statistics

| Level | N | Pass% | Prompt (mean) | Completion (mean) |
|-------|--:|------:|--------------:|------------------:|
| L0 | 1224 | 37.2% | 13,431 | 6,756 |
| L1 | 234 | 63.7% | 9,266 | 4,878 |
| L2 | 234 | 64.1% | 9,267 | 5,026 |
| L3 | 234 | 62.4% | 9,275 | 4,974 |
| L4 | 234 | 60.7% | 9,364 | 4,934 |

## Table 5: Cost Analysis

| Model | Total Cost | Cost on PASS | Cost on FAIL | Cost/PASS | Cost/Task |
|-------|----------:|-----------:|-----------:|----------:|----------:|
| azure-gpt-5.3-codex | N/A | N/A | N/A | N/A | N/A |
| Azure GPT-5.4 | $100.83 | $43.53 | $57.29 | $0.0989 | $0.1286 |
| Qwen 3.5 397B (Together) | $18.58 | $4.59 | $13.99 | $0.0240 | $0.0308 |
| **TOTAL** | **$119.41** (priced models only; EXCLUDES azure-gpt-5.3-codex) | | | | |

## Correlations

- **Kernel-level (prompt)**: Spearman(mean prompt tokens, pass rate) = **-0.6536**
- **Result-level (prompt)**: Mean prompt tokens for PASS = **7,790**, for FAIL = **15,231**
- **Result-level (completion)**: Spearman(completion tokens, pass) = **-0.1942**; Mean completion for PASS = **4,736**, for FAIL = **7,131**

