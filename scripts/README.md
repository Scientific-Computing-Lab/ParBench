# scripts/ — Guide to Subdirectory Organization

`scripts/validate_schema.py` stays at the top level (most-used, referenced everywhere).
All other scripts are organized into functional subdirectories.

## Subdirectories

| Directory | Contents | Usage |
|-----------|----------|-------|
| `generators/` | Spec generation scripts for each benchmark suite | `python3 scripts/generators/generate_rodinia_specs.py` |
| `survey/` | Codebase surveying and kernel inspection scripts | `python3 scripts/survey/survey_rodinia.py` |
| `analysis/` | Results analysis and report generation | `python3 scripts/analysis/generate_report.py` |
| `baselines/` | Baseline population scripts | `python3 scripts/baselines/populate_baselines.py` |
| `augmentation/` | Augment→build→run→verify pipeline | `python3 scripts/augmentation/augment_verify.py specs/rodinia-bfs-cuda.json` |
| `batch/` | Shell batch runners for multi-spec testing | `bash scripts/batch/run_rodinia_batch.sh` |
| `archive/` | One-time fix scripts (kept for reference, not for re-use) | — |

## Key Scripts

- `validate_schema.py` — JSON schema + cross-cutting validator (path unchanged from docs)
- `augmentation/augment_verify.py` — single-spec augment pipeline
- `augmentation/run_augment_batch.py` — batch runner: spec list × augment levels → JSON + MD
- `augmentation/combine_aug_results.py` — merge multi-stream batch results
- `generators/generate_rodinia_specs.py` — Rodinia spec generator
- `batch/run_phase_cuda.sh`, `run_phase_omp.sh`, `run_phase_opencl.sh` — full augmentation test runs
