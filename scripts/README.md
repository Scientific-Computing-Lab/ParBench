# scripts/ — Guide to Subdirectory Organization

`scripts/validate_schema.py` stays at the top level (most-used, referenced everywhere).
All other scripts are organized into functional subdirectories.

## Subdirectories

| Directory | Contents | Usage |
|-----------|----------|-------|
| `survey/` | Codebase surveying scripts | `python3 scripts/survey/survey_rodinia.py` |
| `analysis/` | Results analysis and report generation | `python3 scripts/analysis/statistical_analysis.py` |
| `baselines/` | Baseline population scripts | `python3 scripts/baselines/populate_baselines.py` |
| `augmentation/` | Augment→build→run→verify pipeline | `python3 scripts/augmentation/augment_verify.py specs/rodinia-bfs-cuda.json` |
| `batch/` | Shell batch runners for multi-spec testing | `bash scripts/batch/run_phase3.sh` |

One-time spec generators and dead fix scripts were retired to
`docs/archive/2026-08-dead-scripts/` (2026-08-20). They are history, not tools.

## Key Scripts

- `validate_schema.py` — JSON schema + cross-cutting validator (path unchanged from docs)
- `augmentation/augment_verify.py` — single-spec augment pipeline
- `augmentation/run_augment_batch.py` — batch runner: spec list × augment levels → JSON + MD
- `augmentation/combine_aug_results.py` — merge multi-stream batch results
