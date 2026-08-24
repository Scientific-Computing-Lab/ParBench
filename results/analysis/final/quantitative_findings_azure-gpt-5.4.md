# Quantitative Findings — NeurIPS 2026 ParBench

Generated: 2026-08-17T17:27:44.945018+00:00
Git hash: 6f3c9776

## File Counts

- Total on disk: 822
- Excluded (KNOWN_FAIL, 10 specs): 38
- Valid after exclusion: 784
- Canonical (temp=0.7): 784

---

## Legacy (temperature=0.0) - No Data

Not run: no temperature=0.0 records exist. All quantitative dimensions are reported under the Canonical Evaluation below.

---

## Canonical Evaluation (temperature=0.7)

**Overall:** 56.1% [52.6%, 59.6%] (n=784)

### Dimension 7: pass@k Estimates

**Total tasks:** 136
- **pass@1** (single-sample success rate): 46.3% [38.4%, 54.2%]
- **pass@3** (at least 1 of 3 passes): 52.2% [43.8%, 60.6%]

**Task classification:** 53 always pass, 18 noisy fail, 65 hard fail

**Per-direction pass@k:**

| Direction | pass@1 | pass@3 | n |
|-----------|--------|--------|---|
| cuda-to-omp | 69.6% | 78.3% | 23 |
| cuda-to-omp_target | 91.7% | 100.0% | 8 |
| cuda-to-opencl | 35.2% | 38.9% | 18 |
| omp-to-cuda | 40.6% | 47.8% | 23 |
| omp-to-omp_target | 100.0% | 100.0% | 3 |
| omp-to-opencl | 14.6% | 18.8% | 16 |
| omp_target-to-cuda | 95.8% | 100.0% | 8 |
| omp_target-to-omp | 100.0% | 100.0% | 3 |
| opencl-to-cuda | 13.0% | 16.7% | 18 |
| opencl-to-omp | 35.4% | 43.8% | 16 |

**Per-suite pass@k:**

| Suite | pass@1 | pass@3 | n |
|-------|--------|--------|---|
| hecbench | 96.9% | 100.0% | 32 |
| rodinia | 29.0% | 33.7% | 92 |
| rsbench | 44.4% | 66.7% | 6 |
| xsbench | 44.4% | 66.7% | 6 |

### Dimension 1: Aggregate Pass Rates

**Overall:** 56.1% [52.6%, 59.6%] (n=784)

| Suite | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| hecbench | 95.5% | [92.0%, 97.6%] | 224 |
| rodinia | 38.4% | [34.2%, 42.8%] | 484 |
| rsbench | 57.9% | [42.2%, 72.2%] | 38 |
| xsbench | 47.4% | [32.5%, 62.7%] | 38 |

### Dimension 2: Per-Direction Pass Rates (L0 only)

**Standard directions:**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp | 69.6% | [57.9%, 79.1%] | 69 |
| cuda-to-opencl | 35.2% | [23.8%, 48.5%] | 54 |
| omp-to-cuda | 40.6% | [29.8%, 52.4%] | 69 |
| omp-to-omp_target | 100.0% | [70.1%, 100.0%] | 9 |
| omp-to-opencl | 14.6% | [7.2%, 27.2%] | 48 |
| omp_target-to-omp | 100.0% | [70.1%, 100.0%] | 9 |
| opencl-to-cuda | 13.0% | [6.4%, 24.4%] | 54 |
| opencl-to-omp | 35.4% | [23.4%, 49.6%] | 48 |

**Case study directions (omp_target):**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp_target | 91.7% | [74.2%, 97.7%] | 24 |
| omp_target-to-cuda | 95.8% | [79.8%, 99.3%] | 24 |

*Observation: omp-to-omp_target has 7.7x higher pass rate than opencl-to-cuda.*

### Dimension 3: Direction Asymmetry (McNemar, L0)

| Pair | Fwd Rate | Rev Rate | p-value | Cohen's h | Effect | Sig? |
|------|----------|----------|---------|-----------|--------|------|
| cuda-to-omp vs omp-to-cuda | 78.3% | 47.8% | 0.015625 | 0.6442 | medium | No |
| omp_target-to-cuda vs cuda-to-omp_target | 100.0% | 100.0% | 1.0 | 0.0 | negligible | No |
| omp_target-to-omp vs omp-to-omp_target | 100.0% | 100.0% | 1.0 | 0.0 | negligible | No |
| opencl-to-cuda vs cuda-to-opencl | 16.7% | 38.9% | 0.21875 | -0.5056 | medium | No |
| opencl-to-omp vs omp-to-opencl | 43.8% | 18.8% | 0.125 | 0.5498 | medium | No |

### Dimension 4: Augmentation Trends

**Aggregate Cochran-Armitage:** z=-0.9813, p=0.326429, trend=decreasing, significant=No

| Level | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| L0 | 74.5% | [64.8%, 82.2%] | 94 |
| L1 | 66.0% | [55.9%, 74.7%] | 94 |
| L2 | 68.1% | [58.1%, 76.6%] | 94 |
| L3 | 66.0% | [55.9%, 74.7%] | 94 |
| L4 | 67.0% | [57.0%, 75.7%] | 94 |

**Cohen's h (adjacent levels):**
- L0_to_L1: h=-0.1865
- L1_to_L2: h=0.0453
- L2_to_L3: h=-0.0453
- L3_to_L4: h=0.0225

### Dimension 5: Failure Taxonomy

| Status | Count | % |
|--------|-------|---|
| PASS | 440 | 56.1% |
| BUILD_FAIL | 115 | 14.7% |
| RUN_FAIL | 95 | 12.1% |
| VERIFY_FAIL | 131 | 16.7% |
| EXTRACTION_FAIL | 0 | 0.0% |
| ERROR | 0 | 0.0% |

**Top-3 BUILD_FAIL subcategories:**

| Subcategory | Count |
|-------------|-------|
| linker_error | 56 |
| missing_header | 36 |
| other_build | 12 |

### Dimension 8: Per-Kernel Difficulty Tiers (L0)

**Total kernels:** 30

| Rank | Kernel | Suite | Pass Rate | 95% CI | Passes/Total | Tier |
|------|--------|-------|-----------|--------|--------------|------|
| 1 | floydwarshall | hecbench | 100.0% | [82.4%, 100.0%] | 18/18 | Q1_easiest |
| 2 | heat2d | hecbench | 100.0% | [82.4%, 100.0%] | 18/18 | Q1_easiest |
| 3 | iso2dfd | hecbench | 100.0% | [82.4%, 100.0%] | 18/18 | Q1_easiest |
| 4 | jacobi | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 5 | nqueen | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 6 | scan | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 7 | stencil1d | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 8 | convolution1d | hecbench | 83.3% | [43.6%, 97.0%] | 5/6 | Q2 |
| 9 | md | hecbench | 83.3% | [43.6%, 97.0%] | 5/6 | Q2 |
| 10 | page-rank | hecbench | 83.3% | [43.6%, 97.0%] | 5/6 | Q2 |
| 11 | cfd | rodinia | 66.7% | [43.8%, 83.7%] | 12/18 | Q2 |
| 12 | lud | rodinia | 66.7% | [43.8%, 83.7%] | 12/18 | Q2 |
| 13 | backprop | rodinia | 50.0% | [18.8%, 81.2%] | 3/6 | Q2 |
| 14 | bfs | rodinia | 50.0% | [29.0%, 71.0%] | 9/18 | Q2 |
| 15 | hotspot | rodinia | 50.0% | [29.0%, 71.0%] | 9/18 | Q2 |
| 16 | particlefilter | rodinia | 50.0% | [29.0%, 71.0%] | 9/18 | Q3 |
| 17 | hotspot3d | rodinia | 44.4% | [24.6%, 66.3%] | 8/18 | Q3 |
| 18 | rsbench | rsbench | 44.4% | [24.6%, 66.3%] | 8/18 | Q3 |
| 19 | xsbench | xsbench | 44.4% | [24.6%, 66.3%] | 8/18 | Q3 |
| 20 | bptree | rodinia | 33.3% | [16.3%, 56.2%] | 6/18 | Q3 |
| 21 | nw | rodinia | 22.2% | [9.0%, 45.2%] | 4/18 | Q3 |
| 22 | srad | rodinia | 22.2% | [9.0%, 45.2%] | 4/18 | Q3 |
| 23 | heartwall | rodinia | 11.1% | [3.1%, 32.8%] | 2/18 | Q4_hardest |
| 24 | lavamd | rodinia | 11.1% | [3.1%, 32.8%] | 2/18 | Q4_hardest |
| 25 | dwt2d | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 26 | gaussian | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 27 | myocyte | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 28 | nn | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 29 | pathfinder | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 30 | streamcluster | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |

**Top-5 easiest:** floydwarshall (100.0%), heat2d (100.0%), iso2dfd (100.0%), jacobi (100.0%), nqueen (100.0%)
**Top-5 hardest:** gaussian (0.0%), myocyte (0.0%), nn (0.0%), pathfinder (0.0%), streamcluster (0.0%)

**Direction anomalies (>50pp gap):**

- cfd: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- lud: cuda-to-omp=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- backprop: cuda-to-omp=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- bfs: cuda-to-omp=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- hotspot: cuda-to-omp=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- particlefilter: cuda-to-omp=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- hotspot3d: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- rsbench: cuda-to-opencl=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- xsbench: cuda-to-opencl=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- bptree: cuda-to-opencl=100.0% vs cuda-to-omp=0.0% (100.0pp gap)
- nw: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- srad: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- heartwall: cuda-to-omp=66.7% vs cuda-to-opencl=0.0% (66.7pp gap)
- lavamd: cuda-to-omp=66.7% vs cuda-to-opencl=0.0% (66.7pp gap)

### Dimension 9: Translation Complexity Correlation

| Complexity Class | Pass Rate | 95% CI | Passes/Total |
|------------------|-----------|--------|--------------|
| single_file | 64.1% | [59.6%, 68.3%] | 296/462 |
| multi_to_single | 54.2% | [46.6%, 61.6%] | 90/166 |
| single_to_multi | 33.3% | [25.4%, 42.4%] | 38/114 |
| multi_to_multi | 38.1% | [25.0%, 53.2%] | 16/42 |

**Statistical test:** chi_squared, p=0.0, significant=Yes

### Dimension 10: Cross-Suite Comparison (L0)

| Suite | Pass Rate (L0) | 95% CI | n | Mean SLoC | Multi-File % |
|-------|----------------|--------|---|-----------|-------------|
| hecbench | 96.9% | [91.2%, 98.9%] | 96 | 165.1 | 38.5% |
| rodinia | 29.0% | [23.9%, 34.6%] | 276 | 753.5 | 35.0% |
| rsbench | 44.4% | [24.6%, 66.3%] | 18 | 1016.0 | 25.0% |
| xsbench | 44.4% | [24.6%, 66.3%] | 18 | 1390.0 | 50.0% |

### Dimension 11: Token Cost Analysis

- **Total cost:** $100.83
- **Cost per task:** $0.1286
- **Cost per PASS:** $0.2291
- **Tasks with tokens:** 784

| Suite | Input Tokens | Output Tokens | Cost | Tasks | Cost/Task |
|-------|-------------|---------------|------|-------|-----------|
| hecbench | 461,721 | 837,723 | $13.72 | 224 | $0.0613 |
| rodinia | 6,737,038 | 3,761,327 | $73.26 | 484 | $0.1514 |
| rsbench | 885,016 | 307,761 | $6.83 | 38 | $0.1797 |
| xsbench | 961,220 | 307,356 | $7.01 | 38 | $0.1846 |

### Dimension 12: SLoC Correlation

- **Spearman:** rho=-0.6132, p=0.000315 (significant)
- **Pearson:** r=-0.358, p=0.052078 (not significant)
- **Interpretation:** significant_negative
- **n kernels:** 30

### Dimension 13: OpenCL Kernel-Only Effect (L0)

- **X-to-OpenCL (kernel-only):** 25.5% [18.0%, 34.7%] (n=102)
- **X-to-OMP (full program):** 58.7% [50.0%, 66.9%] (n=126)
- **Fisher's exact:** p=1e-06, OR=0.2404, Cohen's h=-0.6878, significant=Yes

---

## Paper Claims Mapping

| # | Claim ID | Scope | Display Value | Paper Location |
|---|----------|-------|---------------|----------------|
| 1 | overall_pass_rate_rodinia | rodinia_only | 38.4% [34.2%, 42.8%] | abstract/line~71, S6.1/line~707 |
| 2 | primary_campaign_task_counts | all_suite | 484 Rodinia, 784 all-suite | abstract/line~61, S5.2/line~630 |
| 3 | passk_task_count | all_suite | 136 pass@k tasks | S1/line~106, S5.5/line~689 |
| 4 | build_fail_percentage | all_suite | 115/784 = 14.7% | abstract/line~66, S1/line~107, S6.2/line~714 |
| 5 | verify_fail_percentage | all_suite | 131/784 = 16.7% | abstract/line~66, S6.2/line~714 |
| 6 | cuda_to_omp_pass_rate | all_suite | 69.6% | S6.1/line~909, S7.1/line~1041 |
| 7 | l0_pass_rate | all_suite | 74.5% | S6.4/line~899 |
| 8 | cochran_armitage_trend | all_suite | z=-0.9813, p=0.326429 | abstract/line~71, S6.4/line~899, S6.8/line~1003 |
| 9 | cohens_h_range | all_suite | h range [-0.1865, 0.0453] | S6.4/implied |
| 10 | spec_count | all_suite | 206 specs on disk | abstract/line~60, S3.2/line~297, S4.3/line~511 |
| 11 | opencl_to_cuda_pass_rate | all_suite | 13.0% | S6.1/line~941 |
| 12 | multi_file_percentage | all_suite | 76/206 = 36.9% | S1/implied, S4/implied |
| 13 | overall_pass_rate_all_suite | all_suite | 56.1% [52.6%, 59.6%] | S6.1 (all-suite scope) |
| 14 | pass_at_k_rates | all_suite | pass@1=46.3%, pass@3=52.2% | S6.5/line~955 |
| 15 | token_cost | all_suite | $100.83 | S5.2/implied |
| 16 | sloc_correlation | all_suite | rho=-0.6132, p=0.000315 | S7/implied |

---

## Cross-Check Results

Checks run: 5
Status: pass

- INFO: paper_data.json total_on_disk=708, our total=822 (expected match)
