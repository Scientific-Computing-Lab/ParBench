# Active TikZ fragment edit-path note (figure slice 3, ticket #27)

Per-fragment manual edit path for the active, number-carrying TikZ fragments the
manuscript `\input`s from this directory. This is the map for the Phase 2 hand
edits (ticket #27) and any future by-hand number correction.

**Rules (advisor D9 ruling):**

- Hand edits only. There is **no** manifest-to-fragment producer and one must not
  be built. The only tool is the reverse direction (fragment-to-manifest) in slice 4 (#28).
- Target values for the hand edits come from **slice 4's named-cell diff** (#28); the
  acceptance oracle is `results/analysis/final_evidence_manifest.json`, never the
  committed fragment bytes (which predate the sealed replay and are what D7 suspects).
- These are the fragments under
  `docs/paper/NeurIPS_ready_version/figures/figures_tek_version/`. **Never** edit the
  frozen mirror under `parbench_paper_neurips_final_submitted/`.

**Half-dead hazard:** `f3` and `f4` each carry a full commented-out prior revision
*above* the live block. Editing a number in the dead range changes nothing rendered and
silently desyncs the two copies. The LIVE line ranges below are exact; edit only inside them.

---

## f3_kernel_model_heatmap_unified.tex  (1885 lines)

- **DEAD:** lines 1-958 (commented prior revision; dead panel titles at 28 / 331 / 634;
  840 commented `\HeatCell{}` invocations plus 1 commented `\newcommand` definition).
- **LIVE:** lines 959-1885. `\begingroup` at 959, `\newcommand{\HeatCell}[3]` at 960-967,
  then three side-by-side `minipage` panels with **LIVE** titles: Qwen at **983**,
  GPT-5.4 at **1284**, GPT-5.3-codex at **1585**.
- **Number carrier:** `\HeatCell{col}{row}{STATUS}` (840 live `\HeatCell{}` invocations, plus the
  `\newcommand{\HeatCell}[3]` definition at 960 which is not a cell). `col` = direction
  index 0-7 (`C--O, O--C, C--OC, OC--C, O--OC, OC--O, C--OT, OT--C`); `row` = kernel index
  0-34 (`y dir=reverse`); `STATUS` ∈ `{P, BF, RF, VF, EF, --}` (pass / build-fail / run-fail /
  verify-fail / extraction-fail / N-A).
- **Edit only at/after line 959.** The mirror cells at 28/331/634 and their `\HeatCell` stay commented.

## f4_failure_taxonomy_unified.tex  (251 lines)

- **DEAD:** lines 1-151 (commented; dead `\ffourpanel` def at 63, dead invocations 87 / 95 / 103).
- **LIVE:** lines 152-251. `\begingroup` at 152, `\newcommand{\ffourpanel}[7]` at 192, then
  invocations: Qwen at **217**, GPT-5.4 at **225**, GPT-5.3-codex at **233**.
- **Number carrier:** the `\ffourpanel` arguments. Arg 1 = title (model), arg 2 = axis options,
  **args 3-7 = five failure-status coordinate lists**, each `{(dir,count)}` for `dir` 0-5
  (the 6 canonical directions; OMP-target omitted for cross-model comparability). The five lists
  are the stacked failure-taxonomy categories in the fragment's legend order.
- **Edit only the invocations at 217 / 225 / 233.**

## f5_pass_at_k_by_direction_{qwen,gpt_5-4,gpt_5-3_codex}.tex  (77 / 65 / 65 lines)

- No dead block (2-4 header comment lines only). One model per file (named in `title`).
- **Number carrier:** the inline `\pgfplotstableread{ dir p1 p3 ... }\fvpass<model>` table.
  8 rows, `dir` 0-7 (`CUDA--OMP, OMP--CUDA, CUDA--OCL, OCL--CUDA, OMP--OCL, OCL--OMP,
  CUDA--OMP-T, OMP-T--CUDA`); columns `p1` = pass@1 (%), `p3` = pass@3 (%).

## f6_cross_suite_comparison_{qwen,gpt_5-4,gpt_5-3_codex}.tex  (41 lines each)

- No dead block (1 header comment). One model per file.
- **Number carrier:** the inline `\pgfplotstableread{ x y errm errp }\crosssuite` table.
  5 rows, `x` = suite index 0-4 (`Rodinia, XSBench, RSBench, mixbench, HeCBench`);
  `y` = pass rate (fraction 0-1); `errm` / `errp` = asymmetric CI lower/upper half-widths.

## f7_augmentation_robustness.tex  (114 lines)

- No dead render block, but comment lines **4-9 restate the same numbers** as
  `L0=x [lo,hi]` per model - a documentation mirror that must be updated in lockstep.
- **Number carrier:** the inline
  `\pgfplotstableread{ lv qw qw_lo qw_hi g54 g54_lo g54_hi cdx cdx_lo cdx_hi }\fviidata` table
  (lines 13-18). 5 rows, `lv` = level 0-4 (L0-L4); per model a value plus lower/upper CI
  half-widths. This single fragment carries all three models.
- **Edit the `\fviidata` rows AND the 4-9 comment block together.**

## aug_heatmap.tex  (149 lines)

- Single panel: CUDA$\to$OMP, Qwen 3.5. Per-kernel rows x augmentation level columns (L0-L4, 5 cols).
- **Two synchronized carriers - a cell edit must change BOTH:**
  - (a) `\fill[augPass|augBF|augEF] (axis cs:col-0.5,row-0.5) rectangle (axis cs:col+0.5,row+0.5)`
    - the cell background color (status).
  - (b) `\node[...] at (axis cs:col,row) {P|BF|EF}` - the letter label (48 node lines).
- **Hazard:** changing only the fill or only the letter desyncs color from label.
