# Quantitative Findings — NeurIPS 2026 ParBench

Generated: 2026-08-13T18:11:09.822206+00:00
Git hash: 767b91b7

## File Counts

- Total on disk: 2344
- Excluded (KNOWN_FAIL, 10 specs): 184
- Valid after exclusion: 2160
- Canonical (temp=0.7): 2160

---

## Legacy (temperature=0.0) - No Data

Not run: no temperature=0.0 records exist. All quantitative dimensions are reported under the Canonical Evaluation below.

---

## Canonical Evaluation (temperature=0.7)

**Overall:** 48.3% [46.2%, 50.4%] (n=2160)

### Dimension 7: pass@k Estimates

**Total tasks:** 136
- **pass@1** (single-sample success rate): 37.2% [30.6%, 44.0%]
- **pass@3** (at least 1 of 3 passes): 49.2% [41.2%, 57.2%]

**Task classification:** 18 always pass, 55 noisy fail, 63 hard fail

**Per-direction pass@k:**

| Direction | pass@1 | pass@3 | n |
|-----------|--------|--------|---|
| cuda-to-omp | 57.5% | 71.2% | 23 |
| cuda-to-omp_target | 62.5% | 97.0% | 8 |
| cuda-to-opencl | 23.5% | 34.7% | 18 |
| omp-to-cuda | 35.3% | 44.7% | 23 |
| omp-to-omp_target | 81.5% | 100.0% | 3 |
| omp-to-opencl | 13.2% | 18.1% | 16 |
| omp_target-to-cuda | 86.1% | 99.7% | 8 |
| omp_target-to-omp | 100.0% | 100.0% | 3 |
| opencl-to-cuda | 10.5% | 19.5% | 18 |
| opencl-to-omp | 23.6% | 36.9% | 16 |

**Per-suite pass@k:**

| Suite | pass@1 | pass@3 | n |
|-------|--------|--------|---|
| hecbench | 82.3% | 99.1% | 32 |
| rodinia | 22.9% | 31.5% | 92 |
| rsbench | 27.8% | 53.6% | 6 |
| xsbench | 25.9% | 50.6% | 6 |

### Dimension 1: Aggregate Pass Rates

**Overall:** 48.3% [46.2%, 50.4%] (n=2160)

| Suite | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| hecbench | 86.4% | [83.5%, 88.8%] | 632 |
| rodinia | 32.2% | [29.7%, 34.7%] | 1340 |
| rsbench | 37.2% | [28.1%, 47.3%] | 94 |
| xsbench | 33.0% | [24.3%, 43.0%] | 94 |

### Dimension 2: Per-Direction Pass Rates (L0 only)

**Standard directions:**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp | 57.5% | [50.7%, 64.0%] | 207 |
| cuda-to-opencl | 23.5% | [17.6%, 30.6%] | 162 |
| omp-to-cuda | 35.3% | [29.1%, 42.0%] | 207 |
| omp-to-omp_target | 81.5% | [63.3%, 91.8%] | 27 |
| omp-to-opencl | 13.2% | [8.6%, 19.7%] | 144 |
| omp_target-to-omp | 100.0% | [87.5%, 100.0%] | 27 |
| opencl-to-cuda | 10.5% | [6.7%, 16.2%] | 162 |
| opencl-to-omp | 23.6% | [17.4%, 31.2%] | 144 |

**Case study directions (omp_target):**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp_target | 62.5% | [50.9%, 72.8%] | 72 |
| omp_target-to-cuda | 86.1% | [76.3%, 92.3%] | 72 |

*Observation: omp_target-to-omp has 9.5x higher pass rate than opencl-to-cuda.*

### Dimension 3: Direction Asymmetry (McNemar, L0)

| Pair | Fwd Rate | Rev Rate | p-value | Cohen's h | Effect | Sig? |
|------|----------|----------|---------|-----------|--------|------|
| cuda-to-omp vs omp-to-cuda | 65.2% | 42.0% | 0.000402 | 0.4694 | small | Yes |
| omp_target-to-cuda vs cuda-to-omp_target | 95.8% | 66.7% | 0.015625 | 0.8198 | large | No |
| omp_target-to-omp vs omp-to-omp_target | 100.0% | 100.0% | 1.0 | 0.0 | negligible | No |
| opencl-to-cuda vs cuda-to-opencl | 13.0% | 27.8% | 0.057373 | -0.3736 | small | No |
| opencl-to-omp vs omp-to-opencl | 33.3% | 14.6% | 0.011719 | 0.4473 | small | No |

### Dimension 4: Augmentation Trends

**Aggregate Cochran-Armitage:** z=-0.969, p=0.332527, trend=decreasing, significant=No

| Level | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| L0 | 74.2% | [64.7%, 81.9%] | 97 |
| L1 | 66.0% | [56.1%, 74.6%] | 97 |
| L2 | 70.1% | [60.4%, 78.3%] | 97 |
| L3 | 66.0% | [56.1%, 74.6%] | 97 |
| L4 | 67.0% | [57.2%, 75.6%] | 97 |

**Cohen's h (adjacent levels):**
- L0_to_L1: h=-0.1805
- L1_to_L2: h=0.0885
- L2_to_L3: h=-0.0885
- L3_to_L4: h=0.0218

### Dimension 5: Failure Taxonomy

| Status | Count | % |
|--------|-------|---|
| PASS | 1043 | 48.3% |
| BUILD_FAIL | 480 | 22.2% |
| RUN_FAIL | 305 | 14.1% |
| VERIFY_FAIL | 329 | 15.2% |
| EXTRACTION_FAIL | 0 | 0.0% |
| ERROR | 0 | 0.0% |

**Top-3 BUILD_FAIL subcategories:**

| Subcategory | Count |
|-------------|-------|
| linker_error | 152 |
| missing_header | 138 |
| undeclared_identifier | 85 |

### Dimension 8: Per-Kernel Difficulty Tiers (L0)

**Total kernels:** 30

| Rank | Kernel | Suite | Pass Rate | 95% CI | Passes/Total | Tier |
|------|--------|-------|-----------|--------|--------------|------|
| 1 | floydwarshall | hecbench | 90.7% | [80.1%, 96.0%] | 49/54 | Q1_easiest |
| 2 | iso2dfd | hecbench | 90.7% | [80.1%, 96.0%] | 49/54 | Q1_easiest |
| 3 | heat2d | hecbench | 88.9% | [77.8%, 94.8%] | 48/54 | Q1_easiest |
| 4 | stencil1d | hecbench | 88.9% | [67.2%, 96.9%] | 16/18 | Q1_easiest |
| 5 | nqueen | hecbench | 77.8% | [54.8%, 91.0%] | 14/18 | Q1_easiest |
| 6 | page-rank | hecbench | 77.8% | [54.8%, 91.0%] | 14/18 | Q1_easiest |
| 7 | jacobi | hecbench | 72.2% | [49.1%, 87.5%] | 13/18 | Q1_easiest |
| 8 | md | hecbench | 66.7% | [43.8%, 83.7%] | 12/18 | Q2 |
| 9 | scan | hecbench | 66.7% | [43.8%, 83.7%] | 12/18 | Q2 |
| 10 | lud | rodinia | 57.4% | [44.2%, 69.7%] | 31/54 | Q2 |
| 11 | convolution1d | hecbench | 55.6% | [33.7%, 75.4%] | 10/18 | Q2 |
| 12 | cfd | rodinia | 50.0% | [37.1%, 62.9%] | 27/54 | Q2 |
| 13 | bfs | rodinia | 44.4% | [32.0%, 57.6%] | 24/54 | Q2 |
| 14 | particlefilter | rodinia | 44.4% | [32.0%, 57.6%] | 24/54 | Q2 |
| 15 | hotspot | rodinia | 40.7% | [28.7%, 54.0%] | 22/54 | Q2 |
| 16 | hotspot3d | rodinia | 35.2% | [23.8%, 48.5%] | 19/54 | Q3 |
| 17 | rsbench | rsbench | 27.8% | [17.6%, 40.9%] | 15/54 | Q3 |
| 18 | xsbench | xsbench | 25.9% | [16.1%, 38.9%] | 14/54 | Q3 |
| 19 | bptree | rodinia | 24.1% | [14.6%, 37.0%] | 13/54 | Q3 |
| 20 | nw | rodinia | 20.4% | [11.8%, 32.9%] | 11/54 | Q3 |
| 21 | srad | rodinia | 18.5% | [10.4%, 30.8%] | 10/54 | Q3 |
| 22 | backprop | rodinia | 16.7% | [5.8%, 39.2%] | 3/18 | Q3 |
| 23 | heartwall | rodinia | 7.4% | [2.9%, 17.5%] | 4/54 | Q4_hardest |
| 24 | lavamd | rodinia | 3.7% | [1.0%, 12.5%] | 2/54 | Q4_hardest |
| 25 | dwt2d | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 26 | gaussian | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 27 | myocyte | rodinia | 0.0% | [0.0%, 6.6%] | 0/54 | Q4_hardest |
| 28 | nn | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 29 | pathfinder | rodinia | 0.0% | [0.0%, 6.6%] | 0/54 | Q4_hardest |
| 30 | streamcluster | rodinia | 0.0% | [0.0%, 6.6%] | 0/54 | Q4_hardest |

**Top-5 easiest:** floydwarshall (90.7%), iso2dfd (90.7%), heat2d (88.9%), stencil1d (88.9%), nqueen (77.8%)
**Top-5 hardest:** gaussian (0.0%), myocyte (0.0%), nn (0.0%), pathfinder (0.0%), streamcluster (0.0%)

**Direction anomalies (>50pp gap):**

- lud: omp-to-opencl=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- cfd: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- bfs: cuda-to-omp=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- particlefilter: cuda-to-omp=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- hotspot: omp-to-cuda=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- hotspot3d: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- rsbench: cuda-to-opencl=66.7% vs omp-to-cuda=0.0% (66.7pp gap)
- xsbench: cuda-to-opencl=66.7% vs omp-to-cuda=0.0% (66.7pp gap)
- bptree: cuda-to-opencl=55.6% vs cuda-to-omp=0.0% (55.6pp gap)
- nw: cuda-to-omp=77.8% vs cuda-to-opencl=0.0% (77.8pp gap)
- srad: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)

### Dimension 9: Translation Complexity Correlation

| Complexity Class | Pass Rate | 95% CI | Passes/Total |
|------------------|-----------|--------|--------------|
| single_file | 56.8% | [54.1%, 59.5%] | 740/1302 |
| multi_to_single | 45.4% | [40.8%, 50.1%] | 199/438 |
| single_to_multi | 24.2% | [19.7%, 29.3%] | 73/302 |
| multi_to_multi | 26.3% | [19.2%, 34.9%] | 31/118 |

**Statistical test:** chi_squared, p=0.0, significant=Yes

### Dimension 10: Cross-Suite Comparison (L0)

| Suite | Pass Rate (L0) | 95% CI | n | Mean SLoC | Multi-File % |
|-------|----------------|--------|---|-----------|-------------|
| hecbench | 82.3% | [77.5%, 86.3%] | 288 | 165.1 | 38.5% |
| rodinia | 22.9% | [20.2%, 25.9%] | 828 | 753.5 | 35.0% |
| rsbench | 27.8% | [17.6%, 40.9%] | 54 | 1016.0 | 25.0% |
| xsbench | 25.9% | [16.1%, 38.9%] | 54 | 1390.0 | 50.0% |

### Dimension 11: Token Cost Analysis

- **Total cost:** not reported (unpriced model(s): azure-gpt-5.3-codex)
- **Cost per task:** not reported (unpriced model(s): azure-gpt-5.3-codex)
- **Cost per PASS:** not reported (unpriced model(s): azure-gpt-5.3-codex)
- **Tasks with tokens:** 2160

| Suite | Input Tokens | Output Tokens | Cost | Tasks | Cost/Task |
|-------|-------------|---------------|------|-------|-----------|
| hecbench | 1,327,091 | 2,444,752 | N/A | 632 | N/A |
| rodinia | 19,136,647 | 9,176,808 | N/A | 1340 | N/A |
| rsbench | 2,237,294 | 640,078 | N/A | 94 | N/A |
| xsbench | 2,436,317 | 643,100 | N/A | 94 | N/A |

### Dimension 12: SLoC Correlation

- **Spearman:** rho=-0.5683, p=0.001051 (significant)
- **Pearson:** r=-0.3595, p=0.051006 (not significant)
- **Interpretation:** significant_negative
- **n kernels:** 30

### Dimension 13: OpenCL Kernel-Only Effect (L0)

- **X-to-OpenCL (kernel-only):** 18.6% [14.7%, 23.4%] (n=306)
- **X-to-OMP (full program):** 47.6% [42.6%, 52.6%] (n=378)
- **Fisher's exact:** p=0.0, OR=0.2518, Cohen's h=-0.6306, significant=Yes

---

## Paper Claims Mapping

| # | Claim ID | Scope | Display Value | Paper Location |
|---|----------|-------|---------------|----------------|
| 1 | overall_pass_rate_rodinia | rodinia_only | 32.2% [29.7%, 34.7%] | abstract/line~71, S6.1/line~707 |
| 2 | primary_campaign_task_counts | all_suite | 1340 Rodinia, 2160 all-suite | abstract/line~61, S5.2/line~630 |
| 3 | passk_task_count | all_suite | 136 pass@k tasks | S1/line~106, S5.5/line~689 |
| 4 | build_fail_percentage | all_suite | 480/2160 = 22.2% | abstract/line~66, S1/line~107, S6.2/line~714 |
| 5 | verify_fail_percentage | all_suite | 329/2160 = 15.2% | abstract/line~66, S6.2/line~714 |
| 6 | cuda_to_omp_pass_rate | all_suite | 57.5% | S6.1/line~909, S7.1/line~1041 |
| 7 | l0_pass_rate | all_suite | 74.2% | S6.4/line~899 |
| 8 | cochran_armitage_trend | all_suite | z=-0.969, p=0.332527 | abstract/line~71, S6.4/line~899, S6.8/line~1003 |
| 9 | cohens_h_range | all_suite | h range [-0.1805, 0.0885] | S6.4/implied |
| 10 | spec_count | all_suite | 206 specs on disk | abstract/line~60, S3.2/line~297, S4.3/line~511 |
| 11 | opencl_to_cuda_pass_rate | all_suite | 10.5% | S6.1/line~941 |
| 12 | multi_file_percentage | all_suite | 76/206 = 36.9% | S1/implied, S4/implied |
| 13 | overall_pass_rate_all_suite | all_suite | 48.3% [46.2%, 50.4%] | S6.1 (all-suite scope) |
| 14 | pass_at_k_rates | all_suite | pass@1=37.2%, pass@3=49.2% | S6.5/line~955 |
| 15 | token_cost | all_suite | N/A | S5.2/implied |
| 16 | sloc_correlation | all_suite | rho=-0.5683, p=0.001051 | S7/implied |

---

## Cross-Check Results

Checks run: 3
Status: pass
