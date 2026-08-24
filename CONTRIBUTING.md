# Contributing to ParBench

ParBench is a bring-your-own-suite harness, so the most valuable contribution is a new
benchmark suite. Bug reports on existing specs are the next most valuable. Both are welcome.

## Proposing a new suite

Read [docs/adding-a-suite.md](docs/adding-a-suite.md) first: it walks the full path from
source layout to a first verified run. Open a
[new suite issue](https://github.com/Scientific-Computing-Lab/ParBench/issues/new?template=new-suite.yml)
before writing code, so the pinning and licensing questions are settled early.

What a suite contribution must satisfy:

- **One spec per kernel variant**, validating against `schema/spec_schema.json`. The naming
  rule is `unique_id` = `{source_suite}-{kernel_name}-{parallel_api}`, and the file is
  `specs/{unique_id}.json`.
- **A pinned upstream source.** `provenance` records the repository URL, the exact commit,
  and the license. ParBench does not vendor benchmark source; it fetches it at a pinned
  commit, and suites keep their upstream licenses.
- **Run arguments read from the source, not the docs.** Read the argument parser
  (`grep -n argc <source-file>`) before writing `run.args`. Two bundled kernels were silently
  broken for weeks by documentation-derived arguments. Check your specs with
  `python3 scripts/spec_tools/check_spec_argc.py --all`.
- **The strongest correctness oracle the benchmark supports.** `file_hash` or `file_diff` on
  a computed output is strong, `numeric_comparison` against a declared tolerance is medium,
  and `stdout_pattern` plus `exit_code` is weak: it proves execution, not numerical
  correctness. Declare the strength in `verification.oracle_strength`. A weak oracle is
  acceptable when the upstream references genuinely disagree across APIs, but say so in the
  spec, and guard trivially satisfiable patterns with a `stdout_exclude_pattern`.
- **A verified baseline.** `python3 -m harness verify specs/<unique_id>.json` must pass on
  the pristine reference before the spec enters an evaluation. A PASS against an unverified
  baseline proves nothing.
- **One appended line per spec in `manifest.jsonl`.** The manifest is append-only: add lines,
  never edit or remove existing ones. If a spec is retired, its manifest entry stays.
- **A pair contract** for cross-API pairs (`schema/pair_contract_schema.json`), declaring the
  target-side run arguments and oracle. See `config/final_pair_contracts.json` for examples.

## Reporting a spec or verify failure

Open a
[bug report](https://github.com/Scientific-Computing-Lab/ParBench/issues/new?template=bug-report.yml)
with the spec's `unique_id`, your
platform and compiler versions, the exact command you ran, and its full output. Build and
verify failures are usually toolchain-specific, so the compiler version and OS matter as much
as the error text.

Some failures are already known and are excluded from evaluation batches by code
(`EXCLUDED_SPECS` in `harness/constants.py`). Check that list before filing.

## Before opening a pull request

```bash
bash scripts/run_public_tests.sh                               # the test suite scoped to this release
python3 scripts/validate_schema.py --spec specs/<your-spec>.json   # exits 0 when valid
python3 -m harness verify specs/<your-spec>.json                # baseline verification
```

`python3 scripts/validate_schema.py --all` is the full-corpus check. It exits nonzero on any
checkout that does not have every benchmark tree fetched, and it always reports three errors
for each of the five retired manifest entries, so read its error list rather than its exit
status.

Pull requests target `main` and are merged after review by the maintainers. Use the
[pull request template](.github/PULL_REQUEST_TEMPLATE.md) and keep the diff scoped to one
suite or one fix. Do not include machine-specific paths, credentials, or local
`config/paths.json` values in a contribution.
