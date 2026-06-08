# Quantitative Findings — NeurIPS 2026 ParBench

Generated: 2026-05-06T03:37:02.713401+00:00
Git hash: 10a1901

## File Counts

- Total on disk: 708
- Excluded (KNOWN_FAIL, 9 specs): 82
- Valid after exclusion: 626
- Canonical (temp=0.7): 626

---

## Legacy (temperature=0.0) — No Data

### Dimension 1: Aggregate Pass Rates

**Overall:** ? [?, ?] (n=?)

### Dimension 2: Per-Direction Pass Rates (L0 only)

### Dimension 3: Direction Asymmetry (McNemar, L0)

No direction pairs found for McNemar test.

### Dimension 4: Augmentation Trends

### Dimension 5: Failure Taxonomy

### Dimension 8: Per-Kernel Difficulty Tiers (L0)

**Total kernels:** ?


### Dimension 9: Translation Complexity Correlation

### Dimension 10: Cross-Suite Comparison (L0)

### Dimension 11: Token Cost Analysis

- **Total cost:** ?
- **Cost per task:** ?
- **Cost per PASS:** ?
- **Tasks with tokens:** ?

### Dimension 12: SLoC Correlation

- **Spearman:** rho=?, p=? (not significant)
- **Pearson:** r=?, p=? (not significant)
- **Interpretation:** ?
- **n kernels:** ?

### Dimension 13: OpenCL Kernel-Only Effect (L0)

- **X-to-OpenCL (kernel-only):** ? [?, ?] (n=?)
- **X-to-OMP (full program):** ? [?, ?] (n=?)

---

## Canonical Evaluation (temperature=0.7)

**Overall:** 36.7% [33.1%, 40.6%] (n=626)

### Dimension 7: pass@k Estimates

**Total tasks:** 142
- **pass@1** (single-sample success rate): 23.9% [17.9%, 30.0%]
- **pass@3** (at least 1 of 3 passes): 35.2% [27.3%, 43.1%]

**Task classification:** 19 always pass, 31 noisy fail, 92 hard fail

**Per-direction pass@k:**

| Direction | pass@1 | pass@3 | n |
|-----------|--------|--------|---|
| cuda-to-omp | 40.3% | 50.0% | 24 |
| cuda-to-omp_target | 0.0% | 0.0% | 8 |
| cuda-to-opencl | 7.0% | 15.8% | 19 |
| omp-to-cuda | 25.0% | 33.3% | 24 |
| omp-to-omp_target | 44.4% | 100.0% | 3 |
| omp-to-opencl | 33.3% | 52.9% | 17 |
| omp_target-to-cuda | 66.7% | 100.0% | 8 |
| omp_target-to-omp | 100.0% | 100.0% | 3 |
| opencl-to-cuda | 0.0% | 0.0% | 19 |
| opencl-to-omp | 9.8% | 23.5% | 17 |

**Per-suite pass@k:**

| Suite | pass@1 | pass@3 | n |
|-------|--------|--------|---|
| hecbench | 52.1% | 68.8% | 32 |
| mixbench | 5.6% | 16.7% | 6 |
| rodinia | 18.1% | 28.3% | 92 |
| rsbench | 0.0% | 0.0% | 6 |
| xsbench | 5.6% | 16.7% | 6 |

---

## Paper Claims Mapping

| # | Claim ID | Scope | Display Value | Paper Location |
|---|----------|-------|---------------|----------------|
| 1 | overall_pass_rate_rodinia | rodinia_only | 0.0% [0.0%, 0.0%] | abstract/line~71, S6.1/line~707 |
| 2 | primary_campaign_task_counts | all_suite | 0 Rodinia, 0 all-suite | abstract/line~61, S5.2/line~630 |
| 3 | passk_task_count | all_suite | 142 pass@k tasks | S1/line~106, S5.5/line~689 |
| 4 | build_fail_percentage | all_suite | 0/0 = 0.0% | abstract/line~66, S1/line~107, S6.2/line~714 |
| 5 | verify_fail_percentage | all_suite | 0/0 = 0.0% | abstract/line~66, S6.2/line~714 |
| 6 | cuda_to_omp_pass_rate | all_suite | ? | S6.1/line~909, S7.1/line~1041 |
| 7 | l0_pass_rate | all_suite | ? | S6.4/line~899 |
| 8 | cochran_armitage_trend | all_suite | z=None, p=None | abstract/line~71, S6.4/line~899, S6.8/line~1003 |
| 9 | cohens_h_range | all_suite | h range [0.0000, 0.0000] | S6.4/implied |
| 10 | spec_count | all_suite | 96 specs (60+25+4+4+3) | abstract/line~60, S3.2/line~297, S4.3/line~511 |
| 11 | opencl_to_cuda_pass_rate | all_suite | ? | S6.1/line~941 |
| 12 | multi_file_percentage | all_suite | 76/206 = 36.9% | S1/implied, S4/implied |
| 13 | overall_pass_rate_all_suite | all_suite | 0.0% [0.0%, 0.0%] | S6.1 (all-suite scope) |
| 14 | pass_at_k_rates | all_suite | pass@1=23.9%, pass@3=35.2% | S6.5/line~955 |
| 15 | token_cost | all_suite | $0.00 | S5.2/implied |
| 16 | sloc_correlation | all_suite | None | S7/implied |

---

## Cross-Check Results

Checks run: 3
Status: pass
