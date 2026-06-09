# Qwen 3.5 397B Eval Pipeline Audit — 2026-04-24

**Model:** `together-qwen-3.5-397b-a17b`
**Scope:** 708 result JSONs (504 canonical + 204 ablation)
**Verdict:** Pipeline structurally sound. No harness bugs. Data trustworthy for analysis.

## 1. Overall Status Distribution

| Status | Count | % |
|--------|-------|---|
| PASS | 243 | 34.3% |
| BUILD_FAIL | 290 | 41.0% |
| RUN_FAIL | 131 | 18.5% |
| VERIFY_FAIL | 43 | 6.1% |
| EXTRACTION_FAIL | 1 | 0.1% |

Canonical (504): 22.2% PASS | Ablation (204): 64.2% PASS (pre-filtered on L0 pass@1-of-any).

## 2. Experiment Configuration (confirmed)

- **Zero-shot (no self-repair):** `max_retries=1` = 1 total attempt, zero retries. Intentional.
- **Temperature:** 0.7 across all 708 results.
- **Thinking:** Enabled for all results (`thinking_enabled=true`).
- **Seeds:** 3 distinct seeds per canonical pair, deterministically derived.
- **Ablation:** L1-L4 on 51 pairs where ≥1 canonical sample passed.

## 3. BUILD_FAIL Taxonomy (290 total)

### Model Translation Errors: 119 (41%)

| Category | Count | Description |
|----------|-------|-------------|
| cuda_qualifier_error | 25 | `__global__`/`__device__` used incorrectly in CUDA targets |
| not_in_scope | 22 | Variables/functions not declared in scope |
| identifier_undefined | 18 | Undefined identifiers |
| cuda_launch_syntax | 17 | `<<<>>>` kernel launch syntax errors |
| syntax_error | 15 | General syntax (`expected a ")"`, etc.) |
| implicit_declaration | 10 | C implicit function declarations |
| undeclared_identifier | 7 | `use of undeclared identifier` |
| type_conversion | 3 | Invalid type conversions |
| unknown_type | 2 | Unknown type names |

### OMP-Specific Errors: 32 (11%)

| Category | Count | Description |
|----------|-------|-------------|
| omp_invalid_pragma_syntax | 22 | Invalid OpenMP pragma text (nvc++ strict) |
| omp_nesting_violation | 5 | Illegal nesting of OMP constructs |
| omp_pragma_error_other | 5 | Other OMP pragma issues |

### Linker Errors: 46 (16%)

| Category | Count | Description |
|----------|-------|-------------|
| linker_undefined_ref | 45 | Undefined references at link time |
| linker_multiple_def | 1 | Multiple symbol definitions |

### Header/File Confusion: 37 (13%)

| Category | Count | Description |
|----------|-------|-------------|
| header_confusion | 18 | Model `#include`d source headers instead of inlining (lcutil.h, common.h, timing.h) |
| missing_generated_file | 13 | Model referenced files it should have produced (translated_0.cu, kernel.cu) |
| wrong_lang_header | 6 | C++ headers (`<vector>`) in `.c` files |

### Remaining: 56 (19%)

| Category | Count | Description |
|----------|-------|-------------|
| other_compilation_error | 22 | Miscellaneous compiler errors |
| errors_truncated | 13 | Error snippet truncated — only warnings visible |
| type_incompatible | 5 | Type incompatibility in function arguments |
| cuda_builtin_shadow | 3 | Model redeclared CUDA builtins (blockDim, gridDim) |
| nvc++_canonical_form | 3 | NVC++ "OMP for loop not in canonical form" |
| no_member | 2 | Struct/class member access errors |
| truly_unknown | 8 | Require manual inspection |

### Header Confusion — Reviewer Risk

**Exact count: 18 BUILD_FAILs** (6.2% of 290 BUILD_FAILs, 2.5% of 708 total) are
directly caused by "header confusion" — the model saw header content in the prompt
but `#include`d it instead of inlining definitions as instructed.

The broader "header/file confusion" category totals **37 BUILD_FAILs** (12.8% of
BUILD_FAILs) when including models that referenced files they should have generated
themselves (13) or used C++ headers in `.c` files (6). But only the 18 header-confusion
cases involve the prompt design tension.

The prompt (`llm_evaluate.py:719-723`) explicitly says:
> _"These files exist in the **source** build directory but may NOT exist in the
> target directory. If your translated code needs definitions from them, inline the
> definitions directly rather than using `#include`."_

Affected headers: lcutil.h (mixbench, 4 results), common.h (bptree, 6+ results),
timing.h (7 results), util/timer/timer.h (3 results).

**Paper defense (chosen approach):** Kernel-centric translation is a deliberate design
choice that mirrors realistic deployment, where target build environments differ from
source environments. In production HPC code migration, a developer cannot assume that
source-specific utility headers will exist in the target project. The prompt instruction
is explicit, and models that follow it succeed. The 18 header-confused BUILD_FAILs
(2.5% of total results) confirm this is a minor effect, not a systematic bias.

## 4. KNOWN_FAIL PASS Results (Interesting Signal)

8 results from KNOWN_FAIL-involved pairs achieved PASS. These are excluded from
reported statistics by `analyze_eval.py` (line 89-92, EXCLUDED_SPECS check).

| Pair | Samples | KNOWN_FAIL Spec | Significance |
|------|---------|----------------|--------------|
| hecbench-stencil1d-omp_target → cuda | 3/3 PASS | Source (stencil1d-omp_target: BUILD_FAIL) | LLM "fixed" broken source during translation |
| hecbench-scan-omp → omp_target | 2/3 PASS | Target (scan-omp_target: BUILD_FAIL) | LLM produced compilable omp_target from omp source |
| rodinia-nn-opencl → cuda | 1/3 PASS | Source (nn-opencl: TIMEOUT/SIGSEGV) | LLM translated from buggy source, result passed |

**Interpretation:** The LLM can sometimes "fix" broken code during translation. The
stencil1d case is strongest — the omp_target source can't even compile (BUILD_FAIL),
yet the LLM read it and produced working CUDA code in all 3 samples. This suggests
LLMs don't just mechanically port syntax; they can recover algorithmic intent from
broken implementations.

**Status:** Excluded from pass rate statistics. Worth a discussion paragraph in paper.
To be confirmed with re-analysis.

## 5. Ablation Pass Rate by Level

| Level | PASS | BUILD_FAIL | RUN_FAIL | VERIFY_FAIL | Pass Rate |
|-------|------|-----------|----------|-------------|-----------|
| L1 | 38 | 8 | 5 | 0 | 74.5% |
| L2 | 33 | 6 | 9 | 3 | 64.7% |
| L3 | 32 | 8 | 10 | 1 | 62.7% |
| L4 | 28 | 10 | 10 | 3 | 54.9% |

Monotonic decline confirms augmentation progressively degrades translation quality.

## 6. Other Verified Facts

- **5 false-positive overrides:** backprop→opencl results where `verify_status=pass`
  but `overall_status=VERIFY_FAIL` — harness correctly detected hidden
  `clBuildProgram() => -11` in stdout.
- **1 EXTRACTION_FAIL:** rodinia-bfs-omp→cuda-s2. Model merged 3 expected files
  into 1. Other 2 samples PASS. Model variance, not harness bug.
- **15 run timeouts:** All legitimate hangs (infinite loops in translated code).
  6/15 target nn-opencl (KNOWN_FAIL target).
- **2 numeric_comparison VERIFY_FAILs:** Both hotspot3d-opencl→omp. Actual=0.037
  vs expected=0.00004 (~900x off). Medium-strength oracle correctly caught what
  weak oracle would have missed.

## 7. Open Items for Next Analysis Session

- [ ] Re-run analysis with `analyze_eval.py` to get updated eval_summary
- [ ] Spot-check 8 "truly unknown" BUILD_FAIL results
- [ ] Quantify pass@1-of-any vs pass@1 for canonical results
- [ ] Cross-direction analysis (which directions are hardest?)
- [ ] Per-kernel difficulty ranking
- [ ] Hypothesis development on failure mode patterns
