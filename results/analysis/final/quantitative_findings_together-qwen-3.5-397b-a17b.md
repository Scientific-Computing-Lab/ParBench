# Quantitative Findings — NeurIPS 2026 ParBench

Generated: 2026-08-13T18:11:10.481555+00:00
Git hash: 767b91b7

## File Counts

- Total on disk: 708
- Excluded (KNOWN_FAIL, 10 specs): 104
- Valid after exclusion: 604
- Canonical (temp=0.7): 604

---

## Legacy (temperature=0.0) - No Data

Not run: no temperature=0.0 records exist. All quantitative dimensions are reported under the Canonical Evaluation below.

---

## Canonical Evaluation (temperature=0.7)

**Overall:** 31.6% [28.0%, 35.4%] (n=604)

### Dimension 7: pass@k Estimates

**Total tasks:** 136
- **pass@1** (single-sample success rate): 20.3% [14.3%, 26.4%]
- **pass@3** (at least 1 of 3 passes): 27.9% [20.4%, 35.5%]

**Task classification:** 18 always pass, 20 noisy fail, 98 hard fail

**Per-direction pass@k:**

| Direction | pass@1 | pass@3 | n |
|-----------|--------|--------|---|
| cuda-to-omp | 40.6% | 47.8% | 23 |
| cuda-to-omp_target | 0.0% | 0.0% | 8 |
| cuda-to-opencl | 3.7% | 11.1% | 18 |
| omp-to-cuda | 24.6% | 30.4% | 23 |
| omp-to-omp_target | 44.4% | 100.0% | 3 |
| omp-to-opencl | 6.2% | 6.2% | 16 |
| omp_target-to-cuda | 62.5% | 87.5% | 8 |
| omp_target-to-omp | 100.0% | 100.0% | 3 |
| opencl-to-cuda | 0.0% | 0.0% | 18 |
| opencl-to-omp | 10.4% | 25.0% | 16 |

**Per-suite pass@k:**

| Suite | pass@1 | pass@3 | n |
|-------|--------|--------|---|
| hecbench | 51.0% | 65.6% | 32 |
| rodinia | 12.3% | 18.5% | 92 |
| rsbench | 0.0% | 0.0% | 6 |
| xsbench | 0.0% | 0.0% | 6 |

### Dimension 1: Aggregate Pass Rates

**Overall:** 31.6% [28.0%, 35.4%] (n=604)

| Suite | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| hecbench | 64.1% | [57.0%, 70.7%] | 184 |
| rodinia | 19.2% | [15.6%, 23.5%] | 380 |
| rsbench | 0.0% | [0.0%, 17.6%] | 18 |
| xsbench | 0.0% | [0.0%, 14.9%] | 22 |

### Dimension 2: Per-Direction Pass Rates (L0 only)

**Standard directions:**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp | 40.6% | [29.8%, 52.4%] | 69 |
| cuda-to-opencl | 3.7% | [1.0%, 12.5%] | 54 |
| omp-to-cuda | 24.6% | [16.0%, 36.0%] | 69 |
| omp-to-omp_target | 44.4% | [18.9%, 73.3%] | 9 |
| omp-to-opencl | 6.2% | [2.1%, 16.8%] | 48 |
| omp_target-to-omp | 100.0% | [70.1%, 100.0%] | 9 |
| opencl-to-cuda | 0.0% | [0.0%, 6.6%] | 54 |
| opencl-to-omp | 10.4% | [4.5%, 22.2%] | 48 |

**Case study directions (omp_target):**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp_target | 0.0% | [0.0%, 13.8%] | 24 |
| omp_target-to-cuda | 62.5% | [42.7%, 78.8%] | 24 |

*Observation: omp_target-to-omp has infx higher pass rate than opencl-to-cuda.*

### Dimension 3: Direction Asymmetry (McNemar, L0)

| Pair | Fwd Rate | Rev Rate | p-value | Cohen's h | Effect | Sig? |
|------|----------|----------|---------|-----------|--------|------|
| cuda-to-omp vs omp-to-cuda | 47.8% | 30.4% | 0.21875 | 0.3586 | small | No |
| omp_target-to-cuda vs cuda-to-omp_target | 87.5% | 0.0% | 0.015625 | 2.4189 | large | No |
| omp_target-to-omp vs omp-to-omp_target | 100.0% | 100.0% | 1.0 | 0.0 | negligible | No |
| opencl-to-cuda vs cuda-to-opencl | 0.0% | 11.1% | 0.5 | -0.6797 | medium | No |
| opencl-to-omp vs omp-to-opencl | 25.0% | 6.2% | 0.25 | 0.5418 | medium | No |

### Dimension 4: Augmentation Trends

**Aggregate Cochran-Armitage:** z=-2.7574, p=0.005826, trend=decreasing, significant=Yes

| Level | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| L0 | 75.5% | [61.9%, 85.4%] | 49 |
| L1 | 61.2% | [47.2%, 73.6%] | 49 |
| L2 | 55.1% | [41.3%, 68.2%] | 49 |
| L3 | 57.1% | [43.3%, 70.0%] | 49 |
| L4 | 46.9% | [33.7%, 60.6%] | 49 |

**Cohen's h (adjacent levels):**
- L0_to_L1: h=-0.309
- L1_to_L2: h=-0.1242
- L2_to_L3: h=0.0411
- L3_to_L4: h=-0.2046

### Dimension 5: Failure Taxonomy

| Status | Count | % |
|--------|-------|---|
| PASS | 191 | 31.6% |
| BUILD_FAIL | 235 | 38.9% |
| RUN_FAIL | 111 | 18.4% |
| VERIFY_FAIL | 67 | 11.1% |
| EXTRACTION_FAIL | 0 | 0.0% |
| ERROR | 0 | 0.0% |

**Top-3 BUILD_FAIL subcategories:**

| Subcategory | Count |
|-------------|-------|
| undeclared_identifier | 61 |
| missing_header | 56 |
| other_build | 55 |

### Dimension 8: Per-Kernel Difficulty Tiers (L0)

**Total kernels:** 30

| Rank | Kernel | Suite | Pass Rate | 95% CI | Passes/Total | Tier |
|------|--------|-------|-----------|--------|--------------|------|
| 1 | floydwarshall | hecbench | 72.2% | [49.1%, 87.5%] | 13/18 | Q1_easiest |
| 2 | iso2dfd | hecbench | 72.2% | [49.1%, 87.5%] | 13/18 | Q1_easiest |
| 3 | heat2d | hecbench | 66.7% | [43.8%, 83.7%] | 12/18 | Q1_easiest |
| 4 | stencil1d | hecbench | 66.7% | [30.0%, 90.3%] | 4/6 | Q1_easiest |
| 5 | page-rank | hecbench | 50.0% | [18.8%, 81.2%] | 3/6 | Q1_easiest |
| 6 | lud | rodinia | 38.9% | [20.3%, 61.4%] | 7/18 | Q1_easiest |
| 7 | bfs | rodinia | 33.3% | [16.3%, 56.2%] | 6/18 | Q1_easiest |
| 8 | nqueen | hecbench | 33.3% | [9.7%, 70.0%] | 2/6 | Q2 |
| 9 | hotspot | rodinia | 22.2% | [9.0%, 45.2%] | 4/18 | Q2 |
| 10 | hotspot3d | rodinia | 22.2% | [9.0%, 45.2%] | 4/18 | Q2 |
| 11 | particlefilter | rodinia | 22.2% | [9.0%, 45.2%] | 4/18 | Q2 |
| 12 | cfd | rodinia | 16.7% | [5.8%, 39.2%] | 3/18 | Q2 |
| 13 | jacobi | hecbench | 16.7% | [3.0%, 56.4%] | 1/6 | Q2 |
| 14 | md | hecbench | 16.7% | [3.0%, 56.4%] | 1/6 | Q2 |
| 15 | srad | rodinia | 16.7% | [5.8%, 39.2%] | 3/18 | Q2 |
| 16 | nw | rodinia | 11.1% | [3.1%, 32.8%] | 2/18 | Q3 |
| 17 | bptree | rodinia | 5.6% | [1.0%, 25.8%] | 1/18 | Q3 |
| 18 | backprop | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q3 |
| 19 | convolution1d | hecbench | 0.0% | [0.0%, 39.0%] | 0/6 | Q3 |
| 20 | dwt2d | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q3 |
| 21 | gaussian | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q3 |
| 22 | heartwall | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q3 |
| 23 | lavamd | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 24 | myocyte | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 25 | nn | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 26 | pathfinder | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 27 | rsbench | rsbench | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 28 | scan | hecbench | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 29 | streamcluster | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 30 | xsbench | xsbench | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |

**Top-5 easiest:** floydwarshall (72.2%), iso2dfd (72.2%), heat2d (66.7%), stencil1d (66.7%), page-rank (50.0%)
**Top-5 hardest:** pathfinder (0.0%), rsbench (0.0%), scan (0.0%), streamcluster (0.0%), xsbench (0.0%)

**Direction anomalies (>50pp gap):**

- floydwarshall: cuda-to-omp=100.0% vs cuda-to-omp_target=0.0% (100.0pp gap)
- iso2dfd: cuda-to-omp=100.0% vs cuda-to-omp_target=0.0% (100.0pp gap)
- heat2d: omp-to-cuda=100.0% vs cuda-to-omp_target=0.0% (100.0pp gap)
- page-rank: omp_target-to-cuda=100.0% vs cuda-to-omp_target=0.0% (100.0pp gap)
- lud: omp-to-opencl=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- bfs: cuda-to-omp=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- nqueen: omp_target-to-cuda=66.7% vs cuda-to-omp_target=0.0% (66.7pp gap)
- hotspot: omp-to-cuda=100.0% vs cuda-to-omp=0.0% (100.0pp gap)
- hotspot3d: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- particlefilter: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- cfd: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- srad: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)

### Dimension 9: Translation Complexity Correlation

| Complexity Class | Pass Rate | 95% CI | Passes/Total |
|------------------|-----------|--------|--------------|
| single_file | 40.9% | [36.0%, 46.0%] | 153/374 |
| multi_to_single | 28.1% | [20.6%, 36.9%] | 32/114 |
| single_to_multi | 5.8% | [2.5%, 12.9%] | 5/86 |
| multi_to_multi | 3.3% | [0.6%, 16.7%] | 1/30 |

**Statistical test:** chi_squared, p=0.0, significant=Yes

### Dimension 10: Cross-Suite Comparison (L0)

| Suite | Pass Rate (L0) | 95% CI | n | Mean SLoC | Multi-File % |
|-------|----------------|--------|---|-----------|-------------|
| hecbench | 51.0% | [41.2%, 60.8%] | 96 | 165.1 | 38.5% |
| rodinia | 12.3% | [8.9%, 16.7%] | 276 | 753.5 | 35.0% |
| rsbench | 0.0% | [0.0%, 17.6%] | 18 | 1016.0 | 25.0% |
| xsbench | 0.0% | [0.0%, 17.6%] | 18 | 1390.0 | 50.0% |

### Dimension 11: Token Cost Analysis

- **Total cost:** $18.58
- **Cost per task:** $0.0308
- **Cost per PASS:** $0.0973
- **Tasks with tokens:** 604

| Suite | Input Tokens | Output Tokens | Cost | Tasks | Cost/Task |
|-------|-------------|---------------|------|-------|-----------|
| hecbench | 403,649 | 971,965 | $3.74 | 184 | $0.0203 |
| rodinia | 5,710,505 | 2,743,226 | $13.30 | 380 | $0.0350 |
| rsbench | 467,262 | 108,518 | $0.67 | 18 | $0.0373 |
| xsbench | 610,246 | 138,928 | $0.87 | 22 | $0.0394 |

### Dimension 12: SLoC Correlation

- **Spearman:** rho=-0.4226, p=0.019998 (significant)
- **Pearson:** r=-0.3216, p=0.083054 (not significant)
- **Interpretation:** significant_negative
- **n kernels:** 30

### Dimension 13: OpenCL Kernel-Only Effect (L0)

- **X-to-OpenCL (kernel-only):** 4.9% [2.1%, 11.0%] (n=102)
- **X-to-OMP (full program):** 33.3% [25.7%, 41.9%] (n=126)
- **Fisher's exact:** p=0.0, OR=0.1031, Cohen's h=-0.7845, significant=Yes

---

## Paper Claims Mapping

| # | Claim ID | Scope | Display Value | Paper Location |
|---|----------|-------|---------------|----------------|
| 1 | overall_pass_rate_rodinia | rodinia_only | 19.2% [15.6%, 23.5%] | abstract/line~71, S6.1/line~707 |
| 2 | primary_campaign_task_counts | all_suite | 380 Rodinia, 604 all-suite | abstract/line~61, S5.2/line~630 |
| 3 | passk_task_count | all_suite | 136 pass@k tasks | S1/line~106, S5.5/line~689 |
| 4 | build_fail_percentage | all_suite | 235/604 = 38.9% | abstract/line~66, S1/line~107, S6.2/line~714 |
| 5 | verify_fail_percentage | all_suite | 67/604 = 11.1% | abstract/line~66, S6.2/line~714 |
| 6 | cuda_to_omp_pass_rate | all_suite | 40.6% | S6.1/line~909, S7.1/line~1041 |
| 7 | l0_pass_rate | all_suite | 75.5% | S6.4/line~899 |
| 8 | cochran_armitage_trend | all_suite | z=-2.7574, p=0.005826 | abstract/line~71, S6.4/line~899, S6.8/line~1003 |
| 9 | cohens_h_range | all_suite | h range [-0.3090, 0.0411] | S6.4/implied |
| 10 | spec_count | all_suite | 206 specs on disk | abstract/line~60, S3.2/line~297, S4.3/line~511 |
| 11 | opencl_to_cuda_pass_rate | all_suite | 0.0% | S6.1/line~941 |
| 12 | multi_file_percentage | all_suite | 76/206 = 36.9% | S1/implied, S4/implied |
| 13 | overall_pass_rate_all_suite | all_suite | 31.6% [28.0%, 35.4%] | S6.1 (all-suite scope) |
| 14 | pass_at_k_rates | all_suite | pass@1=20.3%, pass@3=27.9% | S6.5/line~955 |
| 15 | token_cost | all_suite | $18.58 | S5.2/implied |
| 16 | sloc_correlation | all_suite | rho=-0.4226, p=0.019998 | S7/implied |

---

## Cross-Check Results

Checks run: 3
Status: pass
