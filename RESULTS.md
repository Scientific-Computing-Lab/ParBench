# ParBench — Evaluation Results

ParBench evaluates LLM-based parallel code translation across **5 benchmark suites** (Rodinia, HeCBench, XSBench, RSBench, mixbench), **206 kernel specs**, and **4 parallel APIs** (CUDA, OpenMP, OpenCL, OpenMP Target). Three models were evaluated on **2,262 total translation tasks**.

## Overall Pass Rates

| Model | PASS | BUILD | RUN | VERIFY | EXTRACT | Total | Rate [95% CI] |
|-------|-----:|------:|----:|-------:|--------:|------:|---------------|
| Qwen 3.5 397B-A17B | 230 | 245 | 121 | 29 | 1 | 626 | 36.7% [33.1, 40.6] |
| GPT-5.4 | 621 | 123 | 43 | 32 | 3 | 822 | 75.5% [72.5, 78.4] |
| GPT-5.3-codex | 604 | 139 | 44 | 27 | 0 | 814 | 74.2% [71.1, 77.1] |

Wilson 95% confidence intervals. Full corpus: Qwen 626, GPT-5.4 822, Codex 814 tasks (totals vary by model due to L0-conditional augmentation filtering).

## Per-Direction Pass Rates

| Model | Overall | CUDA→OMP | OMP→CUDA | CUDA→OCL | OCL→CUDA | OMP→OCL | OCL→OMP |
|-------|--------:|---------:|---------:|---------:|---------:|--------:|--------:|
| Qwen 3.5 397B-A17B | 21/138 (15.2%) | 9/26 (34.6%) | 5/26 (19.2%) | 1/23 (4.3%) | 0/23 (0.0%) | 5/20 (25.0%) | 1/20 (5.0%) |
| GPT-5.4 | 69/120 (57.5%) | 21/24 (87.5%) | 14/24 (58.3%) | 11/19 (57.9%) | 5/19 (26.3%) | 12/17 (70.6%) | 6/17 (35.3%) |
| GPT-5.3-codex | 67/120 (55.8%) | 18/24 (75.0%) | 14/24 (58.3%) | 11/19 (57.9%) | 3/19 (15.8%) | 14/17 (82.4%) | 7/17 (41.2%) |

L0 canonical, first sample only (sample\_id=0), 6 standard translation directions. Totals differ from the overall table, which includes all augmentation levels and samples. OpenMP Target directions (4 additional pairs, 19% of total tasks) are available in the full dataset but excluded from this summary table.

## Failure Taxonomy

| Failure Mode | Count | % of Failures |
|-------------|------:|--------------:|
| BUILD\_FAIL | 507 | 62.8% |
| RUN\_FAIL | 208 | 25.8% |
| VERIFY\_FAIL | 88 | 10.9% |
| EXTRACTION\_FAIL | 4 | 0.5% |

807 total failures across 2,262 tasks (1,455 passed).

## Key Findings

1. **Compilation is the primary bottleneck.** BUILD\_FAIL accounts for 507/807 (62.8%) of all failures. Models struggle more with producing syntactically valid cross-API code than with semantic correctness.

2. **Strong direction asymmetry.** CUDA→OMP achieves 76.4% pass rate while OpenCL→CUDA achieves only 23.7% (aggregate across all models and augmentation levels), a 3.2x gap driven by OpenCL's distinct memory model and host-device API surface.

3. **GPT-5.4 and Codex are statistically indistinguishable.** Odds ratio = 0.931 (95% CI [0.745, 1.164]), Bonferroni-corrected p = 1.000, Cohen's h = -0.031.

4. **Augmentation is robust for GPT-5.4 and Codex.** Pass rates across L1-L4 AST augmentation levels show no significant degradation (Bonferroni-corrected chi-squared p = 1.000 for both). Qwen shows sensitivity (p = 0.005).

5. **Systematic failures dominate.** Hard-fail tasks (fail on every sample): Qwen 92/142, GPT-5.4 43/142, Codex 45/142. These represent inherently difficult translation pairs that no retry resolves.

## Reproduction

```bash
cd artifact/
docker build -t parbench .
docker run --rm -v $(pwd)/output:/app/output parbench ./reproduce.sh
```

See [`artifact/README.md`](artifact/README.md) for the full reproduction guide, including non-Docker setup and model API configuration.

<details>
<summary><strong>Detailed Analysis</strong> (pass@k, augmentation, statistics, complexity, per-kernel heatmap)</summary>

### pass@k and Task Stability

| Model | Tasks | pass@1 | pass@3 | Always-pass | Noisy | Hard-fail |
|-------|------:|-------:|-------:|------------:|------:|----------:|
| Qwen 3.5 397B-A17B | 142 | 23.9% | 35.2% | 19 | 31 | 92 |
| GPT-5.4 | 142 | 62.7% | 69.7% | 76 | 23 | 43 |
| GPT-5.3-codex | 142 | 62.7% | 68.3% | 81 | 16 | 45 |

Always-pass = all samples PASS. Noisy = mixed results across samples. Hard-fail = all samples fail.

### Augmentation Robustness (L0-L4)

| Model | L0 | L1 | L2 | L3 | L4 | chi-squared p (Bonf.) |
|-------|---:|---:|---:|---:|---:|----------------------:|
| Qwen 3.5 397B-A17B | 23.9% | 74.0% | 64.0% | 62.0% | 56.0% | 0.005 |
| GPT-5.4 | 62.7% | 88.9% | 90.9% | 86.9% | 90.9% | 1.000 |
| GPT-5.3-codex | 62.7% | 86.6% | 88.7% | 86.6% | 85.6% | 1.000 |

L1-L4 are conditional on L0 passers only (pass@1-of-any filter). L0 uses all tasks. Augmented levels test robustness to AST-level code transforms (variable renaming, arithmetic rewriting, condition swapping, typedef expansion).

### Statistical Comparisons

| Pair | OR | 95% CI | p (Bonf.) | Cohen's h | Effect |
|------|---:|-------:|----------:|----------:|--------|
| GPT-5.3-codex vs GPT-5.4 | 0.931 | [0.745, 1.164] | 1.000 | -0.031 | small |
| GPT-5.3-codex vs Qwen 3.5 | 4.952 | [3.950, 6.207] | <0.001 | 0.774 | medium |
| GPT-5.4 vs Qwen 3.5 | 5.319 | [4.237, 6.678] | <0.001 | 0.805 | large |

Omnibus chi-squared = 287.27, p < 0.001, Cramer's V = 0.356 (medium effect).

### Translation Complexity

| Complexity Class | PASS | Total | Rate |
|-----------------|-----:|------:|-----:|
| single\_file | 1,052 | 1,404 | 74.9% |
| multi\_to\_single | 301 | 438 | 68.7% |
| multi\_to\_multi | 29 | 118 | 24.6% |
| single\_to\_multi | 73 | 302 | 24.2% |

### Per-Kernel Heatmap (CUDA→OMP, L0)

| Kernel | GPT-5.3-codex | GPT-5.4 | Qwen 3.5 |
|--------|---|---|---|
| backprop | BUILD | PASS | BUILD |
| bfs | PASS | PASS | PASS |
| bptree | BUILD | BUILD | BUILD |
| cfd | PASS | PASS | PASS |
| convolution1d | PASS | PASS | BUILD |
| floydwarshall | PASS | PASS | BUILD |
| heartwall | PASS | PASS | BUILD |
| heat2d | PASS | PASS | BUILD |
| hotspot | PASS | PASS | BUILD |
| hotspot3d | PASS | PASS | PASS |
| iso2dfd | PASS | PASS | BUILD |
| jacobi | PASS | PASS | BUILD |
| lavamd | VERIFY | PASS | BUILD |
| lud | PASS | PASS | PASS |
| md | PASS | PASS | BUILD |
| mixbench | PASS | PASS | BUILD |
| myocyte | BUILD | EXTRACT | BUILD |
| nn | PASS | PASS | BUILD |
| nqueen | PASS | PASS | BUILD |
| nw | PASS | PASS | BUILD |
| page-rank | PASS | PASS | BUILD |
| particlefilter | PASS | PASS | PASS |
| pathfinder | PASS | PASS | PASS |
| rsbench | PASS | BUILD | BUILD |
| scan | PASS | PASS | RUN |
| srad | PASS | PASS | PASS |
| stencil1d | PASS | PASS | PASS |
| streamcluster | PASS | PASS | RUN |
| xsbench | PASS | BUILD | BUILD |

29 kernels, 3 models. Single sample (sample\_id=0), L0 augmentation level.

</details>
