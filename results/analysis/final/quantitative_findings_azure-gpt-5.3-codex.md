# Quantitative Findings — NeurIPS 2026 ParBench

Generated: 2026-08-17T17:27:45.967510+00:00
Git hash: 6f3c9776

## File Counts

- Total on disk: 814
- Excluded (KNOWN_FAIL, 10 specs): 42
- Valid after exclusion: 772
- Canonical (temp=0.7): 772

---

## Legacy (temperature=0.0) - No Data

Not run: no temperature=0.0 records exist. All quantitative dimensions are reported under the Canonical Evaluation below.

---

## Canonical Evaluation (temperature=0.7)

**Overall:** 53.4% [49.8%, 56.9%] (n=772)

### Dimension 7: pass@k Estimates

**Total tasks:** 136
- **pass@1** (single-sample success rate): 45.1% [37.0%, 53.1%]
- **pass@3** (at least 1 of 3 passes): 49.3% [40.8%, 57.7%]

**Task classification:** 55 always pass, 12 noisy fail, 69 hard fail

**Per-direction pass@k:**

| Direction | pass@1 | pass@3 | n |
|-----------|--------|--------|---|
| cuda-to-omp | 62.3% | 69.6% | 23 |
| cuda-to-omp_target | 95.8% | 100.0% | 8 |
| cuda-to-opencl | 31.5% | 33.3% | 18 |
| omp-to-cuda | 40.6% | 47.8% | 23 |
| omp-to-omp_target | 100.0% | 100.0% | 3 |
| omp-to-opencl | 18.8% | 18.8% | 16 |
| omp_target-to-cuda | 100.0% | 100.0% | 8 |
| omp_target-to-omp | 100.0% | 100.0% | 3 |
| opencl-to-cuda | 18.5% | 22.2% | 18 |
| opencl-to-omp | 25.0% | 31.2% | 16 |

**Per-suite pass@k:**

| Suite | pass@1 | pass@3 | n |
|-------|--------|--------|---|
| hecbench | 99.0% | 100.0% | 32 |
| rodinia | 27.5% | 30.4% | 92 |
| rsbench | 38.9% | 66.7% | 6 |
| xsbench | 33.3% | 50.0% | 6 |

### Dimension 1: Aggregate Pass Rates

**Overall:** 53.4% [49.8%, 56.9%] (n=772)

| Suite | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| hecbench | 95.5% | [92.0%, 97.6%] | 224 |
| rodinia | 36.1% | [31.9%, 40.6%] | 476 |
| rsbench | 34.2% | [21.2%, 50.1%] | 38 |
| xsbench | 38.2% | [23.9%, 55.0%] | 34 |

### Dimension 2: Per-Direction Pass Rates (L0 only)

**Standard directions:**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp | 62.3% | [50.5%, 72.8%] | 69 |
| cuda-to-opencl | 31.5% | [20.7%, 44.7%] | 54 |
| omp-to-cuda | 40.6% | [29.8%, 52.4%] | 69 |
| omp-to-omp_target | 100.0% | [70.1%, 100.0%] | 9 |
| omp-to-opencl | 18.8% | [10.2%, 31.9%] | 48 |
| omp_target-to-omp | 100.0% | [70.1%, 100.0%] | 9 |
| opencl-to-cuda | 18.5% | [10.4%, 30.8%] | 54 |
| opencl-to-omp | 25.0% | [14.9%, 38.8%] | 48 |

**Case study directions (omp_target):**

| Direction | Pass Rate | 95% CI | n |
|-----------|-----------|--------|---|
| cuda-to-omp_target | 95.8% | [79.8%, 99.3%] | 24 |
| omp_target-to-cuda | 100.0% | [86.2%, 100.0%] | 24 |

*Observation: omp-to-omp_target has 5.4x higher pass rate than opencl-to-cuda.*

### Dimension 3: Direction Asymmetry (McNemar, L0)

| Pair | Fwd Rate | Rev Rate | p-value | Cohen's h | Effect | Sig? |
|------|----------|----------|---------|-----------|--------|------|
| cuda-to-omp vs omp-to-cuda | 69.6% | 47.8% | 0.125 | 0.4455 | small | No |
| omp_target-to-cuda vs cuda-to-omp_target | 100.0% | 100.0% | 1.0 | 0.0 | negligible | No |
| omp_target-to-omp vs omp-to-omp_target | 100.0% | 100.0% | 1.0 | 0.0 | negligible | No |
| opencl-to-cuda vs cuda-to-opencl | 22.2% | 33.3% | 0.6875 | -0.2492 | small | No |
| opencl-to-omp vs omp-to-opencl | 31.2% | 18.8% | 0.625 | 0.2907 | small | No |

### Dimension 4: Augmentation Trends

**Aggregate Cochran-Armitage:** z=-1.5968, p=0.110313, trend=decreasing, significant=No

| Level | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| L0 | 73.6% | [63.7%, 81.6%] | 91 |
| L1 | 62.6% | [52.4%, 71.9%] | 91 |
| L2 | 64.8% | [54.6%, 73.9%] | 91 |
| L3 | 61.5% | [51.3%, 70.9%] | 91 |
| L4 | 61.5% | [51.3%, 70.9%] | 91 |

**Cohen's h (adjacent levels):**
- L0_to_L1: h=-0.2366
- L1_to_L2: h=0.0457
- L2_to_L3: h=-0.0684
- L3_to_L4: h=0.0

### Dimension 5: Failure Taxonomy

| Status | Count | % |
|--------|-------|---|
| PASS | 412 | 53.4% |
| BUILD_FAIL | 130 | 16.8% |
| RUN_FAIL | 99 | 12.8% |
| VERIFY_FAIL | 131 | 17.0% |
| EXTRACTION_FAIL | 0 | 0.0% |
| ERROR | 0 | 0.0% |

**Top-3 BUILD_FAIL subcategories:**

| Subcategory | Count |
|-------------|-------|
| linker_error | 58 |
| missing_header | 46 |
| undeclared_identifier | 13 |

### Dimension 8: Per-Kernel Difficulty Tiers (L0)

**Total kernels:** 30

| Rank | Kernel | Suite | Pass Rate | 95% CI | Passes/Total | Tier |
|------|--------|-------|-----------|--------|--------------|------|
| 1 | floydwarshall | hecbench | 100.0% | [82.4%, 100.0%] | 18/18 | Q1_easiest |
| 2 | heat2d | hecbench | 100.0% | [82.4%, 100.0%] | 18/18 | Q1_easiest |
| 3 | iso2dfd | hecbench | 100.0% | [82.4%, 100.0%] | 18/18 | Q1_easiest |
| 4 | jacobi | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 5 | md | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 6 | nqueen | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 7 | page-rank | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q1_easiest |
| 8 | scan | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q2 |
| 9 | stencil1d | hecbench | 100.0% | [61.0%, 100.0%] | 6/6 | Q2 |
| 10 | convolution1d | hecbench | 83.3% | [43.6%, 97.0%] | 5/6 | Q2 |
| 11 | cfd | rodinia | 66.7% | [43.8%, 83.7%] | 12/18 | Q2 |
| 12 | lud | rodinia | 66.7% | [43.8%, 83.7%] | 12/18 | Q2 |
| 13 | particlefilter | rodinia | 61.1% | [38.6%, 79.7%] | 11/18 | Q2 |
| 14 | bfs | rodinia | 50.0% | [29.0%, 71.0%] | 9/18 | Q2 |
| 15 | hotspot | rodinia | 50.0% | [29.0%, 71.0%] | 9/18 | Q2 |
| 16 | hotspot3d | rodinia | 38.9% | [20.3%, 61.4%] | 7/18 | Q3 |
| 17 | rsbench | rsbench | 38.9% | [20.3%, 61.4%] | 7/18 | Q3 |
| 18 | bptree | rodinia | 33.3% | [16.3%, 56.2%] | 6/18 | Q3 |
| 19 | xsbench | xsbench | 33.3% | [16.3%, 56.2%] | 6/18 | Q3 |
| 20 | nw | rodinia | 27.8% | [12.5%, 50.9%] | 5/18 | Q3 |
| 21 | srad | rodinia | 16.7% | [5.8%, 39.2%] | 3/18 | Q3 |
| 22 | heartwall | rodinia | 11.1% | [3.1%, 32.8%] | 2/18 | Q3 |
| 23 | backprop | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 24 | dwt2d | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 25 | gaussian | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 26 | lavamd | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 27 | myocyte | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 28 | nn | rodinia | 0.0% | [0.0%, 39.0%] | 0/6 | Q4_hardest |
| 29 | pathfinder | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |
| 30 | streamcluster | rodinia | 0.0% | [0.0%, 17.6%] | 0/18 | Q4_hardest |

**Top-5 easiest:** floydwarshall (100.0%), heat2d (100.0%), iso2dfd (100.0%), jacobi (100.0%), md (100.0%)
**Top-5 hardest:** lavamd (0.0%), myocyte (0.0%), nn (0.0%), pathfinder (0.0%), streamcluster (0.0%)

**Direction anomalies (>50pp gap):**

- cfd: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- lud: cuda-to-omp=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- particlefilter: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- bfs: cuda-to-omp=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- hotspot: cuda-to-omp=100.0% vs omp-to-opencl=0.0% (100.0pp gap)
- hotspot3d: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- rsbench: cuda-to-opencl=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- bptree: omp-to-opencl=100.0% vs cuda-to-omp=0.0% (100.0pp gap)
- xsbench: cuda-to-opencl=100.0% vs omp-to-cuda=0.0% (100.0pp gap)
- nw: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- srad: cuda-to-omp=100.0% vs cuda-to-opencl=0.0% (100.0pp gap)
- heartwall: cuda-to-omp=66.7% vs cuda-to-opencl=0.0% (66.7pp gap)

### Dimension 9: Translation Complexity Correlation

| Complexity Class | Pass Rate | 95% CI | Passes/Total |
|------------------|-----------|--------|--------------|
| single_file | 62.5% | [58.0%, 66.7%] | 291/466 |
| multi_to_single | 48.7% | [41.1%, 56.5%] | 77/158 |
| single_to_multi | 29.4% | [21.4%, 38.9%] | 30/102 |
| multi_to_multi | 30.4% | [19.1%, 44.8%] | 14/46 |

**Statistical test:** chi_squared, p=0.0, significant=Yes

### Dimension 10: Cross-Suite Comparison (L0)

| Suite | Pass Rate (L0) | 95% CI | n | Mean SLoC | Multi-File % |
|-------|----------------|--------|---|-----------|-------------|
| hecbench | 99.0% | [94.3%, 99.8%] | 96 | 165.1 | 38.5% |
| rodinia | 27.5% | [22.6%, 33.1%] | 276 | 753.5 | 35.0% |
| rsbench | 38.9% | [20.3%, 61.4%] | 18 | 1016.0 | 25.0% |
| xsbench | 33.3% | [16.3%, 56.2%] | 18 | 1390.0 | 50.0% |

### Dimension 11: Token Cost Analysis

- **Total cost:** not reported (unpriced model(s): azure-gpt-5.3-codex)
- **Cost per task:** not reported (unpriced model(s): azure-gpt-5.3-codex)
- **Cost per PASS:** not reported (unpriced model(s): azure-gpt-5.3-codex)
- **Tasks with tokens:** 772

| Suite | Input Tokens | Output Tokens | Cost | Tasks | Cost/Task |
|-------|-------------|---------------|------|-------|-----------|
| hecbench | 461,721 | 635,064 | N/A | 224 | N/A |
| rodinia | 6,689,104 | 2,672,255 | N/A | 476 | N/A |
| rsbench | 885,016 | 223,799 | N/A | 38 | N/A |
| xsbench | 864,851 | 196,816 | N/A | 34 | N/A |

### Dimension 12: SLoC Correlation

- **Spearman:** rho=-0.5294, p=0.002629 (significant)
- **Pearson:** r=-0.3276, p=0.077167 (not significant)
- **Interpretation:** significant_negative
- **n kernels:** 30

### Dimension 13: OpenCL Kernel-Only Effect (L0)

- **X-to-OpenCL (kernel-only):** 25.5% [18.0%, 34.7%] (n=102)
- **X-to-OMP (full program):** 50.8% [42.2%, 59.4%] (n=126)
- **Fisher's exact:** p=0.000126, OR=0.3314, Cohen's h=-0.5282, significant=Yes

---

## Paper Claims Mapping

| # | Claim ID | Scope | Display Value | Paper Location |
|---|----------|-------|---------------|----------------|
| 1 | overall_pass_rate_rodinia | rodinia_only | 36.1% [31.9%, 40.6%] | abstract/line~71, S6.1/line~707 |
| 2 | primary_campaign_task_counts | all_suite | 476 Rodinia, 772 all-suite | abstract/line~61, S5.2/line~630 |
| 3 | passk_task_count | all_suite | 136 pass@k tasks | S1/line~106, S5.5/line~689 |
| 4 | build_fail_percentage | all_suite | 130/772 = 16.8% | abstract/line~66, S1/line~107, S6.2/line~714 |
| 5 | verify_fail_percentage | all_suite | 131/772 = 17.0% | abstract/line~66, S6.2/line~714 |
| 6 | cuda_to_omp_pass_rate | all_suite | 62.3% | S6.1/line~909, S7.1/line~1041 |
| 7 | l0_pass_rate | all_suite | 73.6% | S6.4/line~899 |
| 8 | cochran_armitage_trend | all_suite | z=-1.5968, p=0.110313 | abstract/line~71, S6.4/line~899, S6.8/line~1003 |
| 9 | cohens_h_range | all_suite | h range [-0.2366, 0.0457] | S6.4/implied |
| 10 | spec_count | all_suite | 206 specs on disk | abstract/line~60, S3.2/line~297, S4.3/line~511 |
| 11 | opencl_to_cuda_pass_rate | all_suite | 18.5% | S6.1/line~941 |
| 12 | multi_file_percentage | all_suite | 76/206 = 36.9% | S1/implied, S4/implied |
| 13 | overall_pass_rate_all_suite | all_suite | 53.4% [49.8%, 56.9%] | S6.1 (all-suite scope) |
| 14 | pass_at_k_rates | all_suite | pass@1=45.1%, pass@3=49.3% | S6.5/line~955 |
| 15 | token_cost | all_suite | N/A | S5.2/implied |
| 16 | sloc_correlation | all_suite | rho=-0.5294, p=0.002629 | S7/implied |

---

## Cross-Check Results

Checks run: 5
Status: pass

- INFO: paper_data.json total_on_disk=708, our total=814 (expected match)
