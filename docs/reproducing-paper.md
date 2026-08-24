# Reproducing the paper's results

Every number in the ParBench paper comes from the canonical analysis build in
`results/analysis/final/` (built 2026-08-13: 2,344 evaluation records, 184 excluded by the
pre-registered eligibility rules, 2,160 valid records over 136 translation pairs and three
models). `paper_claims.json` in that directory maps each claim in the paper to the analysis
file and JSON path that produces it, so any published value can be traced to its source
without re-running anything.

Three reproduction depths:

1. **Trace a number (no compute).** Open `results/analysis/final/paper_claims.json`, find
   the claim, and read the value at the recorded path in the recorded file.
2. **Regenerate all tables and figures from raw records.** The raw per-task records are not
   tracked in this repository; they ship inside the self-contained reproducibility artifact
   (a ~43 MB zip attached to the GitHub Release), together with a Dockerfile and
   `reproduce.sh`. Follow `artifact/README.md` — it carries both the Docker path (exact
   environment) and the no-Docker path, expected runtime ~10-15 minutes, no GPU needed.
   Deterministic table values can be diffed against `expected_outputs/` for bit-exact
   verification.
3. **Re-run the LLM evaluations from scratch.** Requires API keys and an NVIDIA GPU for
   the CUDA/OpenCL specs; see [running-evals.md](running-evals.md) for the batch runner
   and the paper's sampling configuration (pass@3 at temperature 0.7). Benchmark source
   trees are fetched at pinned commits — [benchmark-tree-pins.md](benchmark-tree-pins.md)
   records the SHA and fetch recipe for each suite (only `rodinia` is a git submodule).
