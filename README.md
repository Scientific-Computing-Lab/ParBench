# ParBench — A Kernel-Centric Benchmark for Evaluating LLM-Based Parallel Code Translation

Translating parallel code between programming APIs (CUDA, OpenMP, OpenCL, OpenMP target
offload) is a long-standing porting burden in high-performance computing, and large language
models now attempt it with uneven, hard-to-compare results. ParBench is an **evaluation
harness** for measuring exactly that: you bring a parallel benchmark suite, describe each
kernel in a machine-readable spec, and the harness builds, runs, and verifies every
LLM-produced translation so that results are comparable across models, suites, and
translation directions.

ParBench is **not a translator**. It does not translate code itself and it ships no model.
It is the measurement instrument: prompt construction, code extraction, build → run → verify,
and a declared correctness oracle per kernel, with every outcome recorded as a JSON verdict.

<p align="center">
  <img src="docs/figures/parbench_architecture.png" alt="ParBench Architecture" width="100%">
</p>
<p align="center"><em>ParBench end-to-end pipeline: API selection → benchmark corpus → augmentation → evaluation → harness → results.</em></p>

## Bring your own suite

The five bundled suites below are a starting corpus, not a boundary. Any parallel benchmark
can join the harness by writing one spec file per kernel variant against
`schema/spec_schema.json` and appending one line to `manifest.jsonl`. Translation pairs,
prompt payloads, and verification then work for the new suite exactly as they do for the
bundled ones. See **[docs/adding-a-suite.md](docs/adding-a-suite.md)** for the full path.

## Benchmark corpus

ParBench bundles 206 spec files covering 90 unique kernels from five HPC suites, across four
parallel APIs (CUDA, OpenMP, OpenCL, OpenMP target offload). The paper's evaluation uses a
curated subset: 96 curated specs, of which 87 are eval-eligible, forming 136 eligible
translation pairs.

| Suite | Kernels | Spec files | APIs | Source |
|-------|---------|------------|------|--------|
| [Rodinia](https://rodinia.cs.virginia.edu/) | 22 | 60 | CUDA, OpenMP, OpenCL | Git submodule (`rodinia/rodinia-src/`, commit `9c10d3ea`) |
| [HeCBench](https://github.com/zjin-lcf/HeCBench) | 65 | 135 | CUDA, OpenMP, OpenMP target | Fetched by SHA (`HeCBench-master/`, gitignored) |
| [XSBench](https://github.com/ANL-CESAR/XSBench) | 1 | 4 | CUDA, OpenMP, OpenCL, OpenMP target | Fetched by SHA (`xsbench/xsbench-src/`) |
| [RSBench](https://github.com/ANL-CESAR/RSBench) | 1 | 4 | CUDA, OpenMP, OpenCL, OpenMP target | Fetched by SHA (`rsbench/rsbench-src/`) |
| [mixbench](https://github.com/ekondis/mixbench) | 1 | 3 | CUDA, OpenMP, OpenCL | Fetched by SHA (`mixbench/mixbench-src/`) |

Only `rodinia` is a git submodule. The other four trees are fetched at pinned commits;
**[docs/benchmark-tree-pins.md](docs/benchmark-tree-pins.md)** records the SHAs and the
fetch recipes. The append-only `manifest.jsonl` (211 entries) indexes all spec files and
enables automatic discovery of translation pairs across APIs.

## Installation

Python 3.12 or later is required.

```bash
git clone <repository-url>
cd ParBench

python3 -m venv env_parbench
source env_parbench/bin/activate

# Core dependencies (harness, schema validation, augmentation)
python3 -m pip install -r requirements.txt

# Or for exact pinned versions (reproducible environment)
python3 -m pip install -r requirements-lock.txt
```

Optional dependency groups from `pyproject.toml`:

```bash
python3 -m pip install ".[eval]"      # LLM evaluation pipeline (anthropic, openai clients)
python3 -m pip install ".[analysis]"  # Analysis and figure generation (matplotlib, numpy)
python3 -m pip install ".[dev]"       # Development tools (pytest, ruff)
python3 -m pip install ".[all]"       # Everything
```

Building and running kernels additionally requires compilers for the target APIs (`nvcc` for
CUDA, `g++` with `-fopenmp` for OpenMP, OpenCL headers and runtime for OpenCL). Tested
versions are listed in `config/compiler_inventory.txt`.

## Quick start

```bash
source env_parbench/bin/activate

# 1. Fetch the Rodinia sources (the only submodule; ~101 MB)
git submodule update --init rodinia

# 2. Validate the manifest and all specs
python3 scripts/validate_schema.py --all

# 3. Build, run, and verify one kernel (OpenMP — needs only a multi-core CPU;
#    nw generates its own input matrix, so no data download is required)
python3 -m harness verify specs/rodinia-nw-omp.json

# 4. List all valid translation pairs (e.g., CUDA to OpenMP)
python3 -m harness pairs
```

Two platform notes. (1) OpenMP kernels need a compiler with `-fopenmp` (GNU g++; Apple's
clang on macOS does not support it — use Linux, the tested platform). (2) Rodinia kernels
that read input files (bfs, hotspot, srad, ...) additionally need the separate
[Rodinia data package](https://rodinia.cs.virginia.edu/), unpacked to
`rodinia/rodinia-src/data/`; kernels with self-generated inputs (nw, lud, pathfinder,
backprop, ...) run without it.

To evaluate an LLM on translation pairs, see
**[docs/running-evals.md](docs/running-evals.md)**.

## Results data

The repository ships the paper's aggregate analysis outputs in `results/analysis/final/`
(canonical build of 2026-08-13: 2,344 evaluation records, of which 184 are excluded by the
pre-registered eligibility rules, leaving 2,160 valid records over 136 translation pairs and
three models). These JSON and CSV files carry every number in the paper: pass@k tables,
direction asymmetry, augmentation trends, the failure taxonomy, and per-suite results.

Raw per-task records are not tracked in this repository. The self-contained reproducibility
artifact (Docker image recipe, raw records, and `reproduce.sh`) is published as a release
asset; **[docs/reproducing-paper.md](docs/reproducing-paper.md)** explains both reproduction
paths.

## Project structure

```
ParBench/
├── README.md                       # This file
├── manifest.jsonl                  # Master index (append-only, one JSON object per line)
├── schema/                         # JSON Schemas (draft-07) for manifest and specs
├── specs/                          # One spec file per kernel variant
├── templates/spec_template.json    # Blank spec with all fields
├── harness/                        # Build, run, verify automation (python3 -m harness)
├── c_augmentation/                 # AST-driven, semantics-preserving code transforms
├── scripts/                        # Validation, evaluation, analysis utilities
├── docs/                           # Guides: adding a suite, running evals, reproducing
├── results/analysis/final/         # Canonical aggregate results (the paper's numbers)
├── expected_outputs/               # Reference outputs for bit-exact table verification
├── artifact/                       # Reproducibility artifact (Dockerfile, reproduce.sh)
└── config/                         # Machine-specific config (git-ignored paths.json)
```

## Spec anatomy

Each kernel variant is described by one JSON spec (draft-07 schema in
`schema/spec_schema.json`). Key sections:

- **identity** — unique_id, kernel name, API, source suite
- **provenance** — repository URL, pinned commit, license
- **files** — `prompt_payload` (LLM sees), `support_files` (build only), `verification_only` (never shown to LLM)
- **build / run** — environment, commands, executable, arguments, timeout
- **verification** — method, strategies, floating-point tolerance
- **hardware** — target device, requirements, reference platform

### File security model

The `files` section protects evaluation integrity:

- `prompt_payload` — files the LLM receives for translation. **Only** these enter the prompt.
- `support_files` — Makefiles and shared headers needed for compilation, **not** sent to the LLM.
- `verification_only` — reference implementations and test harnesses, **never** shown to the LLM.

No file may appear in both `prompt_payload` and `verification_only`; the validator enforces
this.

## Validation

```bash
python3 scripts/validate_schema.py --manifest manifest.jsonl   # manifest only
python3 scripts/validate_schema.py --spec specs/rodinia-bfs-cuda.json   # one spec
python3 scripts/validate_schema.py --all                       # everything
```

The validator checks schema conformance, `unique_id` naming and format, API consistency, that
every listed file exists on disk, and the prompt/verification separation above. Two classes
of errors are expected and explained by the validator: (1) specs whose benchmark tree you
have not fetched yet report missing source files (fetch the tree, per the table above, and
they clear); (2) five phantom manifest entries remain permanently (the manifest is
append-only and retains entries whose spec files were deleted).

## Requirements

- **Reproducing paper tables and figures:** any x86_64 machine, no GPU (~4 GB RAM).
- **Running CUDA/OpenCL specs:** NVIDIA GPU (compute capability ≥ 7.0, e.g., RTX 3060+).
- **OpenMP-only specs:** any multi-core x86_64 CPU.
- **Tested platform:** NVIDIA RTX 4070, AMD Ryzen 9 7900X, Ubuntu 24.04, NVIDIA HPC SDK 24.3.

## Testing

```bash
python3 -m pytest c_augmentation/test_transforms.py -v   # augmentation transform tests
python3 scripts/validate_schema.py --all                 # schema validation
```

## License

MIT (see `LICENSE`). The bundled benchmark suites are fetched from their upstream
repositories and keep their own upstream licenses; ParBench does not vendor them.

## Citation

The ParBench paper — "ParBench: A Kernel-Centric Benchmark for Evaluating LLM-Based Parallel
Code Translation" (NeurIPS 2026) — describes the benchmark design and the evaluation results.
We will add the citation entry and the arXiv link to this section when the camera-ready
version is published.
