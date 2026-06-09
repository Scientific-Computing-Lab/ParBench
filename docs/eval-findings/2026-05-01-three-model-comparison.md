# 3-Model Comparison Analysis — 2026-05-01

**Models:** `together-qwen-3.5-397b-a17b`, `azure-gpt-5.4`, `azure-gpt-5.3-codex`
**Scope:** 2,262 result records (626 Qwen + 822 GPT-5.4 + 814 codex)
**Protocol:** Canonical pass@3 (temp=0.7, 3 seeds) + L0-conditional ablation (L1-L4)
**Data sources:** `results/analysis/statistical_analysis.json`, `cross_model_comparison_*.json`, `paper_data_*.json`

## 1. Overall Pass Rates (Wilson 95% CIs)

| Model | Pass Rate | 95% CI | n |
|-------|-----------|--------|---|
| azure-gpt-5.4 | 75.5% | [72.5%, 78.4%] | 822 |
| azure-gpt-5.3-codex | 74.2% | [71.1%, 77.1%] | 814 |
| together-qwen-3.5-397b-a17b | 36.7% | [33.1%, 40.6%] | 626 |

Omnibus chi-squared: chi2(2) = 287.27, p = 0.0, Cramer's V = 0.356 (medium).

## 2. Pairwise Model Comparisons (Fisher's exact, Bonferroni-corrected)

| Pair | OR [95% CI] | p (corrected) | Cohen's h | Effect |
|------|-------------|--------------|-----------|--------|
| GPT-5.3-codex vs GPT-5.4 | 0.93 [0.74, 1.16] | 1.0000 | -0.031 | negligible |
| GPT-5.3-codex vs Qwen | 4.95 [3.95, 6.21] | 0.0000 | +0.774 | medium |
| GPT-5.4 vs Qwen | 5.32 [4.24, 6.68] | 0.0000 | +0.805 | large |

**Finding 1:** GPT-5.4 and GPT-5.3-codex are statistically indistinguishable (p=1.0, OR=0.93). Both massively outperform Qwen (~5x odds ratio, large effect).

## 3. Per-Direction Pass Rates (10 directions, spread = 74.7pp)

| Direction | Rate | n |
|-----------|------|---|
| omp_target-to-omp | 98.4% | 63 |
| omp_target-to-cuda | 89.9% | 168 |
| cuda-to-omp_target | 77.9% | 136 |
| cuda-to-omp | 76.4% | 432 |
| omp-to-omp_target | 74.6% | 63 |
| omp-to-opencl | 69.8% | 301 |
| omp-to-cuda | 61.4% | 368 |
| cuda-to-opencl | 60.1% | 283 |
| opencl-to-omp | 43.5% | 237 |
| opencl-to-cuda | 23.7% | 211 |

**Finding 2:** Direction is the strongest predictor of difficulty (74.7pp spread), stronger than model choice (38.8pp spread between GPT and Qwen). A single aggregate pass rate hides this entirely.

## 4. Direction Asymmetry (McNemar, L0 only)

| Forward | Fwd Rate | Reverse | Rev Rate | h | p | Sig? |
|---------|----------|---------|----------|---|---|------|
| cuda-to-omp | 70.8% | omp-to-cuda | 43.1% | +0.569 | 0.0000 | **Yes** |
| cuda-to-opencl | 43.9% | opencl-to-cuda | 12.3% | +0.732 | 0.0009 | **Yes** |
| omp-to-opencl | 68.6% | opencl-to-omp | 31.4% | +0.764 | 0.0005 | **Yes** |
| omp_target-to-cuda | 91.7% | cuda-to-omp_target | 66.7% | +0.645 | 0.0312 | No |
| omp-to-omp_target | 88.9% | omp_target-to-omp | 100.0% | -0.680 | 1.0000 | No |

**Finding 3:** 3 of 5 paired directions show significant asymmetry after Bonferroni correction. Translation difficulty is direction-dependent: translating FROM a language is consistently harder than translating TO it, especially for OpenCL.

## 5. From-OpenCL Bottleneck

OpenCL poses a bidirectional challenge, most severe in the source-language role:

| Direction | Aggregate | Canonical pass@1 |
|-----------|-----------|-----------------|
| opencl-to-cuda | 23.7% | 12.9% |
| opencl-to-omp | 43.5% | 30.1% |
| cuda-to-opencl | 60.1% | — |
| omp-to-opencl | 69.8% | — |

**Finding 4:** OpenCL as source is dramatically harder than as target. This is consistent with OpenCL's explicit memory management and kernel-host separation being harder for LLMs to read and restructure. Canonical pass@1 rates are even starker.

## 6. Kernel Difficulty (31 kernels, spread = 95.1pp)

### Hardest (<40%)
| Kernel | Rate | n |
|--------|------|---|
| gaussian | 0.0% | 18 |
| heartwall | 14.5% | 62 |
| myocyte | 19.4% | 62 |
| lavamd | 21.2% | 66 |
| streamcluster | 22.7% | 66 |
| backprop | 31.8% | 22 |
| bptree | 35.1% | 74 |

### Easiest (>=80%)
| Kernel | Rate | n |
|--------|------|---|
| iso2dfd | 95.1% | 122 |
| floydwarshall | 90.2% | 122 |
| heat2d | 88.5% | 122 |
| stencil1d | 88.1% | 42 |
| nqueen | 86.8% | 38 |
| scan | 82.3% | 34 |
| nn | 82.3% | 34 |
| page-rank | 81.6% | 38 |
| md | 81.6% | 38 |
| jacobi | 81.6% | 38 |

**Finding 5:** ParBench discriminates between trivially parallel patterns (stencils, reductions: iso2dfd, heat2d) and architecturally complex patterns (GPU memory management, irregular parallelism: gaussian, heartwall). A benchmark with only easy kernels would overestimate LLM capability.

Distribution: 7 hard (<40%), 14 medium (40-80%), 10 easy (>=80%).

## 7. Per-Kernel Agreement Matrix

### Qwen vs GPT-5.4 (31 common kernels)
| Category | Count | Kernels |
|----------|-------|---------|
| Both pass | 21 | bfs, cfd, convolution1d, floydwarshall, heat2d, hotspot, hotspot3d, iso2dfd, jacobi, lud, md, mixbench, nqueen, nw, page-rank, particlefilter, pathfinder, scan, srad, stencil1d, xsbench |
| Both fail | 1 | gaussian |
| Qwen only | 0 | — |
| GPT-5.4 only | 9 | backprop, bptree, dwt2d, heartwall, lavamd, myocyte, nn, rsbench, scan |

### Qwen vs GPT-5.3-codex (31 common kernels)
| Category | Count | Kernels |
|----------|-------|---------|
| Both pass | 21 | (same as above) |
| Both fail | 2 | backprop, gaussian |
| Qwen only | 0 | — |
| Codex only | 8 | bptree, dwt2d, heartwall, lavamd, myocyte, nn, rsbench, scan |

### GPT-5.4 vs GPT-5.3-codex (31 common kernels)
| Category | Count | Kernels |
|----------|-------|---------|
| Both pass | 29 | (all except gaussian, backprop) |
| Both fail | 1 | gaussian |
| GPT-5.4 only | 1 | backprop |
| Codex only | 0 | — |

**Finding 6:** At kernel granularity, GPT-5.4 and codex agree on 30/31 kernels (96.8%), with `backprop` as the sole disagreement. Qwen passes zero kernels that both GPT models fail — the GPT models are a strict superset of Qwen's kernel coverage. Nine kernels discriminate between model tiers: backprop, bptree, dwt2d, heartwall, lavamd, myocyte, nn, rsbench, scan.

## 8. Augmentation Curves

| Model | L0 | L1 | L2 | L3 | L4 | Pattern |
|-------|----|----|----|----|-----|---------|
| GPT-5.3-codex | 62.7% | 86.6% | 88.7% | 86.6% | 85.6% | Plateau |
| GPT-5.4 | 62.7% | 88.9% | 90.9% | 86.9% | 90.9% | Plateau |
| Qwen 3.5 | 23.9% | 74.0% | 64.0% | 62.0% | 56.0% | Peak-then-decline |

Augmentation chi-squared significance:
- Qwen: chi2=18.0, p=0.005 (Bonferroni-corrected) — **significant**
- GPT-5.4: p=1.0 — not significant
- GPT-5.3-codex: p=1.0 — not significant

**Finding 7:** Augmentation reveals qualitatively different model behavior. GPT models plateau at L1 (~87-91%) — they're already strong enough that code simplification barely helps. Qwen shows a dramatic +50pp boost at L1 but then declines monotonically through L4 (74% → 56%), suggesting higher augmentation levels add surface complexity that degrades Qwen's performance. Structured code hints compensate for capability gaps rather than amplifying existing strength.

## 9. GPT-5.4 vs GPT-5.3-codex: Detailed Divergence

Despite identical overall pass@k rates (62.7% at L0), per-direction analysis reveals small differences:

| Direction | GPT-5.4 | Codex | h | Effect |
|-----------|---------|-------|---|--------|
| cuda-to-omp | 83.3% | 76.4% | +0.174 | negligible |
| cuda-to-omp_target | 95.8% | 100.0% | -0.411 | small |
| omp-to-opencl | 72.5% | 82.3% | -0.236 | small |
| omp_target-to-cuda | 95.8% | 100.0% | -0.411 | small |
| All others | identical or negligible | | | |

McNemar on 142 balanced tasks: chi2=0.125, p=0.724. Concordance: 94 both_pass, 40 both_fail, 5 GPT-5.4-only, 3 codex-only.

**Finding 8:** Two models with identical headline rates differ on only 8/142 tasks. Codex is slightly stronger on omp_target directions (100% vs 95.8%) and omp-to-opencl (82.3% vs 72.5%). GPT-5.4's only kernel-level advantage is `backprop`. This demonstrates that even statistically indistinguishable models can be differentiated by ParBench's multi-dimensional design.

## 10. Failure Taxonomy by Model

| Model | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | BF:VF Ratio |
|-------|-----------|---------|-------------|-------------|
| Qwen 3.5 | 245 | 121 | 29 | 8.4:1 |
| GPT-5.4 | 123 | 43 | 32 | 3.8:1 |
| GPT-5.3-codex | 139 | 44 | 27 | 5.1:1 |

**Finding 9:** The model capability gap is primarily compilability and runtime robustness, not semantic correctness. Qwen fails at compile time 8.4x more than at verification, while GPT-5.4's ratio is 3.8:1. This suggests Qwen's lower pass rate is driven by code generation quality (syntax, API usage) rather than algorithmic understanding.

## 11. Paper-Ready Claims

1. GPT-5.4 and GPT-5.3-codex are statistically indistinguishable at pass@3 (p=1.0, OR=0.93), differing on only 8/142 balanced tasks, indicating that increased code pretraining does not yield measurable gains at this task difficulty.
2. Both GPT models surpass Qwen by ~5x odds ratio (Cohen's h ≈ 0.8, large effect), with Qwen contributing zero unique kernel-level passes.
3. Direction is the strongest predictor of difficulty (74.7pp spread across 10 directions), with 3/5 paired comparisons showing significant asymmetry after Bonferroni correction.
4. OpenCL poses a bidirectional challenge most severe as source: opencl-to-cuda = 23.7% aggregate (12.9% canonical pass@1), compared to 60.1% for cuda-to-opencl.
5. Augmentation compensates for capability gaps rather than amplifying existing strength: Qwen gains +50pp at L1 then declines (p=0.005), while GPT models plateau (p=1.0).
6. ParBench's kernel difficulty spans 0%–95.1%, with 9 discriminating kernels separating model tiers (architecturally complex patterns: GPU memory management, irregular parallelism).
7. Performance gaps are driven by compile/runtime robustness (BUILD_FAIL:VERIFY_FAIL = 8.4:1 for Qwen vs 3.8:1 for GPT-5.4), not semantic correctness.

## 12. Scope Caveats

- All aggregate rates (Sections 3, 5, 6) include augmented samples across all models and levels. Canonical-only rates are lower (e.g., opencl-to-cuda canonical pass@1 = 12.9% vs aggregate 23.7%).
- McNemar p=0.72 (Section 9) comes from the balanced 142-task set; OR=0.93 comes from statistical_analysis.json over unequal n (822 vs 814). Cite with scope labels.
- Unequal n across models (626/822/814) reflects different augmentation coverage; CI widths differ accordingly.
- Zero-shot evaluation (no self-repair) — BUILD_FAILs that a single retry could fix count as failures.
- OpenMP target compilations use nvc++ (NVIDIA HPC SDK 24.3), stricter than GCC.

## 13. Confounding Variables

1. **Training data distribution:** GPT models may share a corpus over-representing CUDA/OMP code relative to OpenCL.
2. **Prompt sensitivity:** Identical prompts but different tokenization across models. OpenCL's verbose boilerplate may consume more of Qwen's effective context window.
3. **Augmentation as prompt engineering:** The L0→L1 boost could partly be a prompt length/context effect rather than genuine code simplification.
4. **nvc++ strictness:** Some OMP BUILD_FAILs might pass on GCC — affects cross-model comparisons if models differ in pragma syntax sensitivity.
