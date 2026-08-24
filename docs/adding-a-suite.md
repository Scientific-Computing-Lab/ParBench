# Adding your own benchmark suite

ParBench is a bring-your-own-suite harness: any parallel benchmark joins the evaluation by
describing each kernel variant in one JSON spec and registering it in the manifest. This
guide walks the full path: source layout, spec authoring, validation, manifest registration,
and a first verified run. The bundled Rodinia specs (`specs/rodinia-*.json`) are working
examples for every pattern described here.

## 1. Lay out the source

Place the suite's source tree inside the project root (the harness resolves all paths
relative to it). A kernel variant is one implementation of one kernel in one parallel API,
for example a CUDA and an OpenMP implementation of the same stencil are two variants.

## 2. Write one spec per kernel variant

Copy `templates/spec_template.json` and fill it in. The schema is
`schema/spec_schema.json` (JSON Schema draft-07). The naming rule is strict:

- `unique_id` = `{source_suite}-{kernel_name}-{parallel_api}`, e.g. `mysuite-stencil-cuda`
- the spec file is `specs/{unique_id}.json`
- `kernel_name` is a lowercase slug (no uppercase, no `+`)

The sections that matter most:

| Section | What it drives |
|---|---|
| `identity` | The unique_id, kernel pairing key, API |
| `provenance` | Upstream repository URL, pinned commit, license |
| `files` | What the LLM sees vs. what only the build sees (below) |
| `build` | Build commands, run from the kernel's source directory |
| `run` | Executable, arguments, input configurations, timeout |
| `verification` | The correctness oracle (below) |

### The file security model

Three lists in `files` protect evaluation integrity:

- `prompt_payload` — files the LLM receives. Only these enter the prompt.
- `support_files` — Makefiles and headers needed to compile, never sent to the LLM.
- `verification_only` — reference implementations and expected outputs, never shown to the LLM.

No file may appear in both `prompt_payload` and `verification_only`; the validator rejects it.
`files.translation_targets` names the subset of the payload the LLM must actually produce
(kernel-centric translation): for OpenCL targets that is the `.cl` kernel files only; for
CUDA/OpenMP targets it is the curated kernel files.

### Run arguments: read the source, not the docs

The single most common spec bug is wrong run arguments, and its cause is trusting a README
over the code. **Read the source's argument parser before writing `run.args`**
(`grep -n argc <source-file>`), and count what it actually checks. Two bundled kernels were
silently broken for weeks by documentation-derived arguments: `rodinia-nw-omp`
(`needle.cpp:249` checks `argc==4`) and `rodinia-hotspot-omp` (`hotspot_openmp.cpp:282`
checks `argc!=8`). Flag-parsed programs (`-boxes1d`, `-c/-r/-h`, ...) need the same
treatment: read the parser, then run the reference binary once to confirm.
`python3 scripts/spec_tools/check_spec_argc.py --all` compares every spec's arguments
against the source's literal `argc` checks.

## 3. Design the correctness oracle

`verification.strategies` lists checks that are evaluated as a conjunction: every declared
strategy must pass, and an unknown or malformed strategy is an error, never a silent skip.
Available strategies: `exit_code`, `stdout_pattern`, `stdout_exclude_pattern`,
`numeric_comparison`, `file_hash`, `file_diff`.

Oracle strength is worth deliberate effort, because it bounds what a PASS means:

- **Strong** (`file_hash` / `file_diff` on a computed output): the translated program
  reproduced the reference output exactly. Use when the kernel writes a deterministic
  output file and the APIs agree bit-for-bit.
- **Medium** (`numeric_comparison`): a computed scalar matches within a declared tolerance.
  Use for kernels that print a checksum or an aggregate.
- **Weak** (`stdout_pattern` + `exit_code`): the program ran to completion and printed the
  expected completion marker. This proves execution, not numerical correctness.

Prefer the strongest oracle the benchmark supports, and check cross-API agreement before
committing to a strong one: several Rodinia kernel pairs diverge between their own CUDA and
OpenMP reference implementations (different hardcoded parameters, different output
behavior), which makes a shared `file_hash` oracle unsatisfiable for a faithful translation.
When the references disagree, declare the weak oracle and record why. Beware
trivially-satisfiable patterns: a program that prints both `PASS` and `FAIL` lines satisfies
a bare `stdout_pattern: "PASS"`; add a `stdout_exclude_pattern` for the failure marker.

## 4. Register and validate

Append one line per spec to `manifest.jsonl` (`kernel_name`, `parallel_api`,
`source_suite`, `spec_file`, `source_dir`). The manifest is append-only. Then validate:

```bash
python3 scripts/validate_schema.py --spec specs/<unique_id>.json
python3 scripts/validate_schema.py --all
```

The validator enforces schema conformance, the naming rules, file existence, and the
prompt/verification separation.

## 5. Verify the baseline

```bash
python3 -m harness verify specs/<unique_id>.json
```

A spec whose pristine reference does not pass its own oracle must not enter an evaluation:
a PASS against an unverified baseline proves nothing. Once every variant of a kernel
verifies, `python3 -m harness pairs` discovers its translation pairs automatically, and the
suite participates in evaluations like any bundled one (see
[running-evals.md](running-evals.md)). Cross-API pairs additionally need a pair contract
(`schema/pair_contract_schema.json`) declaring the target-side run arguments and oracle;
see the bundled registry `config/final_pair_contracts.json` for examples.
