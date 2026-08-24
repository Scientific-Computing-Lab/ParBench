# Running LLM evaluations

The batch runner `scripts/evaluation/run_eval_batch.py` enumerates translation pairs from
the manifest, sends each source kernel to a model for translation, and grades every
translation through the harness (build → run → verify). One JSON result file per task lands
under `results/evaluation/<model>/`.

## A first batch

```bash
source env_parbench/bin/activate

python3 scripts/evaluation/run_eval_batch.py \
  --suite rodinia \
  --direction cuda-to-omp \
  --models <model-id> \
  --pair-contracts config/pair_contracts_v2.json \
  --project-root . \
  --resume -v
```

- `--direction` is `SRC-to-TGT` over the APIs `cuda`, `omp`, `opencl`, `omp_target`.
- `--suite` filters to one suite. **Always pass it**: without it, `--kernels nw` matches
  every suite that has a kernel named `nw`. `--kernels` composes with `--suite`.
- `--models` takes one or more model IDs from `MODEL_REGISTRY` in
  `scripts/evaluation/llm_evaluate.py`; add your own entry there to evaluate a new model
  or provider (Anthropic, OpenAI/Azure, Together, and OpenAI-compatible endpoints are
  supported).
- `--resume` (the default) skips tasks whose result file already exists, so an interrupted
  batch continues where it stopped. `--resume` also re-attempts tasks that ended in
  `ERROR` or `EXTRACTION_FAIL`; use `--no-resume` to re-run everything.
- `--pair-contracts` names the pair-contract registry. Any batch containing cross-API
  tasks fails closed before the first model call if the registry is missing or incomplete.
  The contract declares the oracle and the run arguments for each source→target pair:
  verification always uses the target API's own oracle; the translated program's run
  arguments follow the program that actually runs (target-native arguments for
  kernel-only pairs, where only `.cl` kernel files are rewritten and the target host
  binary is untouched; source arguments for full-program pairs, where the model keeps
  source argument parsing). Use `config/pair_contracts_v2.json` (current) for new runs;
  `config/final_pair_contracts.json` is the frozen registry behind the sealed NeurIPS
  evidence package and is kept only for provenance.

Useful variations: `--augment-levels 0 1 2` evaluates augmented source variants
(L0 is the unmodified original), `--max-retries 3` enables iterative repair with build-error
feedback, and `--temperature 0.7 --num-samples 3` draws independent samples per task for
pass@k.

## Exclusion semantics

Two spec lists in `harness/constants.py` govern eligibility, and both are enforced by code
rather than by convention:

- `EXCLUDED_SPECS` (10 specs): kernels whose baseline is broken in the current toolchain
  (KNOWN_FAIL). The batch runner's `resolve_excluded_specs`
  (`scripts/evaluation/run_eval_batch.py:78`) removes them from every batch on entry;
  the CLI flag `--excluded-specs` can add exclusions but never remove one. Rationale:
  source-side, an unverified baseline makes a PASS meaningless; target-side, broken
  infrastructure makes the evaluation unfair.
- `CORRECTNESS_INELIGIBLE_SPECS` (13 specs): the 10 above plus the 3 mixbench specs,
  which run but expose no extractable computed value and therefore carry no correctness
  oracle. The analysis scripts drop any record whose source or target spec is in this
  list from every correctness denominator; mixbench records remain usable for
  performance-only analysis.

## How pass@k is computed

With `--num-samples n` at a sampling temperature above zero, each task gets `n` independent
translations. The analysis uses the unbiased estimator of Chen et al. (2021)
(`pass_at_k` in `scripts/analysis/statistical_analysis.py:901`):

    pass@k = 1 - C(n-c, k) / C(n, k)

where `c` of the `n` samples passed. For n = k it reduces to any-of-k task success. The
paper's headline configuration is pass@3: three samples per task at temperature 0.7 with
model reasoning enabled.

## Reading results

Each result JSON's authoritative verdict is `overall_status`: `PASS`, `BUILD_FAIL`,
`RUN_FAIL`, `VERIFY_FAIL`, `EXTRACTION_FAIL`, or `ERROR`. The `attempts[]` array is the
per-attempt record (top-level `run_status` can be stale on multi-attempt tasks; ignore it).
Aggregate a finished batch with:

```bash
python3 scripts/evaluation/analyze_eval.py --project-root . --results-dir results/evaluation
```

which writes `eval_summary.json` and `eval_summary.md`, including pass rates by direction,
model, and translation complexity.
