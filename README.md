# ParBench — Meta-Benchmark for LLM-Based Parallel Code Translation

> **[Evaluation Results](RESULTS.md)** — Full results for three LLMs across 2,262 parallel code translation tasks, including pass rates, direction analysis, failure taxonomy, and per-kernel breakdowns.

ParBench is a curated meta-benchmark for evaluating how well large language models
translate parallel source code across different programming APIs (CUDA, HIP, SYCL,
OpenMP, etc.). It aggregates kernels from multiple existing benchmark suites and
wraps each one in a machine-readable specification that drives automated build,
run, verification, and LLM evaluation workflows.

## Project Structure

```
parbench/
├── README.md                       # This file
├── GUIDE.md                        # Complete guide: pipeline, commands, adding benchmarks
├── manifest.jsonl                  # Level 1: Master index (one JSON object per line)
│
├── schema/                         # JSON Schemas (draft-07)
│   ├── manifest_schema.json        #   Schema for manifest.jsonl entries
│   ├── spec_schema.json            #   Schema for Level 2 kernel spec files
│   └── reference_platform.json     #   Shared hardware reference platform
│
├── specs/                          # Level 2 spec files (one per kernel variant)
│   └── <suite>-<kernel>-<api>.json
│
├── templates/
│   └── spec_template.json          # Blank spec with all fields & placeholder values
│
├── harness/                        # Python harness: build, run, verify automation
│   ├── cli.py                      #   Command-line interface
│   ├── builder.py                  #   Compilation logic
│   ├── runner.py                   #   Execution logic
│   ├── verifier.py                 #   Verification strategies
│   ├── spec_loader.py              #   JSON loading & path resolution
│   ├── reporter.py                 #   Output formatting
│   └── models.py                   #   Result data classes
│
├── scripts/                        # Utility scripts
│   ├── validate_schema.py          #   Schema + cross-cutting validator
│   ├── generate_paper_figures.py   #   Generate publication figures
│   └── generate_viz_data.py        #   Generate visualization data
│
├── examples/                       # Reference data
│   └── example_178_kernels.json    #   178 kernels in single-file format
│
├── analysis/                       # All analysis outputs
│   ├── visualizations/             #   PNG charts and network graphs
│   ├── reports/                    #   Markdown reports, presentations
│   └── data/                       #   CSV matrices, Excel workbooks
│
└── config/                         # Machine-specific config (git-ignored)
    └── paths.json                  #   Maps downloads_root to local path
```

> **See [GUIDE.md](GUIDE.md) for the complete pipeline walkthrough, all commands, and instructions for adding new benchmarks.**

## Schemas

All schemas use **JSON Schema draft-07**.

### manifest.jsonl

Each line is a self-contained JSON object that indexes one kernel variant:

| Field          | Type   | Required | Description |
|----------------|--------|----------|-------------|
| `kernel_name`  | string | ✓        | Logical kernel name (pairing key) |
| `parallel_api` | enum   | ✓        | API: serial, cuda, hip, sycl, … |
| `source_suite` | string | ✓        | Origin benchmark suite (lowercase) |
| `spec_file`    | string | ✓        | Relative path to Level 2 spec JSON |
| `source_dir`   | string | ✓        | Relative path to kernel source files |
| `category`     | enum   | —        | Domain: ml, graph, physics, … |

### Level 2 Spec (`specs/*.json`)

A comprehensive specification that drives the full evaluation pipeline.
Key sections:

- **identity** — unique_id, kernel name, API, source suite
- **provenance** — repository URL, pinned commit, license
- **files** — `prompt_payload` (LLM sees), `support_files` (build only),
  `verification_only` (never shown to LLM)
- **implementation** — language, API details
- **build** — environment, build system, commands, outputs
- **run** — executable, arguments, input configurations, timeout
- **verification** — method, strategies, floating-point tolerance
- **performance** — optional metric extraction
- **hardware** — target device, requirements, reference platform
- **baseline_results** — populated after benchmarking (starts null)
- **metadata** — description, domain, tags

### File Security Model

The `files` section is **critical** for LLM evaluation integrity:

- `prompt_payload` → Files the LLM receives for translation. **ONLY** these go
  in the prompt.
- `support_files` → Makefiles, shared headers needed for compilation but
  **NOT** sent to the LLM.
- `verification_only` → Reference implementations and test harnesses.
  **NEVER** shown to the LLM.

No file may appear in both `prompt_payload` and `verification_only`.

## Validation

```bash
# Install dependency
python3 -m pip install jsonschema

# Validate the manifest
python3 scripts/validate_schema.py --manifest manifest.jsonl

# Validate a single spec
python3 scripts/validate_schema.py --spec specs/rodinia-bfs-cuda.json

# Validate everything (manifest + all specs)
python3 scripts/validate_schema.py --all
```

The validator checks:
1. JSON Schema conformance (draft-07)
2. `unique_id` matches the spec filename
3. `unique_id` format matches `{source_suite}-{kernel_name}-{api}`
4. `implementation.api` matches `identity.parallel_api`
5. All files listed in `files.*` exist on disk
6. No file appears in both `prompt_payload` and `verification_only`

## Paths

- All paths in specs and the manifest are **relative to `parbench/`** (the project root).
- `downloads_root` equals `project_root` — downloaded benchmark repos live inside the project directory itself.
- Machine-specific path configuration lives in `config/paths.json` (git-ignored).

## Requirements

- Python 3.12+
- `jsonschema` (`python3 -m pip install jsonschema`)

### Hardware

- **For reproducing paper tables/figures (Tier 2):** Any x86_64 machine, no GPU needed (~4 GB RAM)
- **For running CUDA/OpenCL specs (Tier 3):** NVIDIA GPU (compute capability ≥ 7.0, e.g., RTX 3060+)
- **For OpenMP-only specs:** Any multi-core x86_64 CPU (no GPU required)
- **Tested platform:** NVIDIA RTX 4070, AMD Ryzen 9 7900X, Ubuntu 24.04, NVIDIA HPC SDK 24.3

## Installation

Python 3.12 or later is required. Clone the repository, create a virtual environment, and install dependencies:

```bash
git clone <repository-url>
cd parbench

python3 -m venv env_parbench
source env_parbench/bin/activate

# Core dependencies (harness, schema validation, augmentation)
python3 -m pip install -r requirements.txt

# Or for exact pinned versions (reproducible environment)
python3 -m pip install -r requirements-lock.txt
```

Optional dependency groups can be installed via `pyproject.toml`:

```bash
# LLM evaluation pipeline (anthropic, openai clients)
python3 -m pip install ".[eval]"

# Analysis and figure generation (matplotlib, numpy)
python3 -m pip install ".[analysis]"

# Development tools (pytest, ruff)
python3 -m pip install ".[dev]"

# Everything
python3 -m pip install ".[all]"
```

The build-run-verify harness also requires compilers for the target parallel APIs (e.g., `nvcc` for CUDA, `g++` with `-fopenmp` for OpenMP, OpenCL headers and runtime libraries for OpenCL). See `config/compiler_inventory.txt` for the tested compiler versions.

## Quick start

1. Activate the virtual environment:

   ```bash
   source env_parbench/bin/activate
   ```

2. Validate the spec and manifest schemas:

   ```bash
   python3 scripts/validate_schema.py --all
   ```

3. Run the full build-run-verify pipeline on a single kernel spec:

   ```bash
   python3 -m harness verify specs/rodinia-bfs-cuda.json
   ```

4. List all available translation pairs (e.g., CUDA to OpenMP):

   ```bash
   python3 -m harness pairs
   ```

## Reproducing Paper Results

Results reproduction is tiered by what you want to verify:

### Tier 1: Pipeline Verification (free, ~5 min, no GPU needed)

Confirm the harness and analysis pipeline work on your machine:

```bash
source env_parbench/bin/activate
python3 scripts/validate_schema.py --all             # Schema validation (expect ~15 known errors from phantom specs)
python3 -m harness info specs/rodinia-bfs-cuda.json  # Inspect a spec
```

### Tier 2: Reproduce Tables & Figures from Bundled Results (free, ~15 min, no GPU needed)

All 2,344 per-task result JSONs are included in `results/evaluation/`. Regenerate every
table and figure in the paper from these raw results:

```bash
# Docker (recommended — exact environment):
cd artifact && docker build -t parbench . && docker run --rm -v $(pwd)/../output:/app/output parbench ./reproduce.sh

# Or without Docker:
bash artifact/reproduce.sh
```

Output lands in `output/` — 5 LaTeX tables (T1–T5) and 16 figures (F2–F7, C.3–C.4, per-model variants).
Deterministic table values can be diffed against `expected_outputs/` for bit-exact verification.
See `artifact/README.md` for full details.

### Tier 3: Full Re-Evaluation (paid, ~8 hours, requires API keys + NVIDIA GPU)

Re-run all 2,262 LLM evaluations from scratch. Requires:
- NVIDIA GPU (CUDA 12.x) for build-run-verify of CUDA/OpenCL specs
- API keys: Together AI (Qwen), Azure OpenAI (GPT-5.4, GPT-5.3-Codex)
- Estimated cost: ~$150–200 in API credits (as of May 2026)

```bash
# Example: re-run Qwen on CUDA-to-OpenMP direction
python3 scripts/evaluation/run_eval_batch.py \
  --suite rodinia --direction cuda-to-omp \
  --models together-qwen-3.5-397b-a17b \
  --project-root . --resume -v
```

See the paper's Appendix J for full campaign configuration and cost details.

## Evaluation Results

Pre-computed evaluation results for all three models are in `results/evaluation/`:

```
results/evaluation/
├── together-qwen-3.5-397b-a17b/   # 708 result JSONs
├── azure-gpt-5.4/                  # 822 result JSONs
└── azure-gpt-5.3-codex/            # 814 result JSONs
```

Each JSON file represents one translation task and contains:
- `overall_status`: PASS, BUILD_FAIL, RUN_FAIL, VERIFY_FAIL, or EXTRACTION_FAIL
- `translation_code`: The LLM-generated translated source code
- `build_result`, `run_result`, `verification_result`: Per-stage outcomes with stdout/stderr
- `model`, `direction`, `source_spec`, `target_spec`: Task metadata
- `augmentation_level`, `sample_id`: Experiment design coordinates

After KNOWN_FAIL exclusion, 2,262 records are eval-eligible (the denominator for all paper statistics).

## Usage examples

**Inspect a spec without building or running:**

```bash
python3 -m harness info specs/rodinia-hotspot-omp.json
```

**View the LLM prompt payload (the source files an LLM would receive for translation):**

```bash
python3 -m harness prompt specs/rodinia-nw-cuda.json
```

**Build and run a kernel with verbose output:**

```bash
python3 -m harness -v verify specs/rodinia-bfs-cuda.json
```

**Run an LLM evaluation batch (CUDA to OpenMP, with resume support):**

```bash
python3 scripts/evaluation/run_eval_batch.py \
  --suite rodinia \
  --direction cuda-to-omp \
  --models <model-name> \
  --project-root /path/to/parbench \
  --resume -v
```

**Analyze evaluation results and generate a summary:**

```bash
python3 scripts/evaluation/analyze_eval.py \
  --project-root /path/to/parbench \
  --results-dir results/evaluation
```

## Benchmark Suites

ParBench aggregates kernels from five HPC benchmark suites, covering 90 unique kernels
across 206 spec files and four parallel APIs (CUDA, OpenMP, OpenCL, OpenMP target offload).

| Suite | Kernels | Spec Files | APIs | Source |
|-------|---------|------------|------|--------|
| [Rodinia](https://rodinia.cs.virginia.edu/) | 22 | 60 | CUDA, OpenMP, OpenCL | Git submodule (`rodinia/rodinia-src/`, commit `9c10d3ea`) |
| [HeCBench](https://github.com/zjin-lcf/HeCBench) | 65 | 135 | CUDA, OpenMP, OpenMP target | Cloned locally (`HeCBench-master/`, gitignored) |
| [XSBench](https://github.com/ANL-CESAR/XSBench) | 1 | 4 | CUDA, OpenMP, OpenCL, OpenMP target | Git submodule (`xsbench-src/`) |
| [RSBench](https://github.com/ANL-CESAR/RSBench) | 1 | 4 | CUDA, OpenMP, OpenCL, OpenMP target | Git submodule (`rsbench-src/`) |
| [mixbench](https://github.com/ekondis/mixbench) | 1 | 3 | CUDA, OpenMP, OpenCL | Git submodule (`mixbench-src/`) |

Each kernel variant is fully described by a JSON spec file in `specs/` that drives the
build-run-verify pipeline. The append-only manifest (`manifest.jsonl`, 211 entries) indexes
all spec files and enables automatic discovery of translation pairs across APIs.

## Code Augmentation

The `c_augmentation/` package provides AST-driven, semantics-preserving code transforms
powered by libclang. These transforms create diverse LLM input variants to evaluate
translation robustness -- the transformed code compiles and runs identically to the original.

**Transforms available:**

| Transform | Description |
|-----------|-------------|
| `ArithmeticTransform` | Expands compound operators (e.g., `x += 1` to `x = x + 1`) |
| `SwapCondition` | Flips comparison operands (e.g., `x < y` to `y > x`) |
| `PointerArithmeticToArrayIndex` | Converts pointer arithmetic to array indexing (e.g., `*(arr + i)` to `arr[i]`) |
| `TypedefExpansion` | Inlines typedef aliases with their underlying types |
| `ChangeNames` | Renames local variables to neutral identifiers |
| `ChangeFunctionNames` | Renames non-entry-point functions |

**Augmentation levels (L1--L4)** control the fraction of eligible candidates each transform
modifies, from a single candidate (L1) to all candidates (L4). Level L0 is the unaugmented
original source.

**Run augmentation on a single spec:**

```bash
python3 scripts/augmentation/augment_verify.py specs/rodinia-bfs-cuda.json \
  --augment_level 2 --seed 42 --project-root /path/to/parbench
```

**Run augmentation unit tests (15 tests, all must pass before commit):**

```bash
python3 -m pytest c_augmentation/test_transforms.py -v
```

## Testing

ParBench uses pytest for its test suite. The primary tests cover the code augmentation
transforms:

```bash
# Run all augmentation transform tests
python3 -m pytest c_augmentation/test_transforms.py -v

# Run schema validation across all specs and the manifest
python3 scripts/validate_schema.py --all
```

Approximately 15 schema validation errors are expected from phantom spec entries in the
append-only manifest -- these are not bugs. See `GUIDE.md` for details.

## Citation

ParBench is under review at NeurIPS 2026 (Evaluations & Datasets Track). Citation
guidance will be added upon publication.
