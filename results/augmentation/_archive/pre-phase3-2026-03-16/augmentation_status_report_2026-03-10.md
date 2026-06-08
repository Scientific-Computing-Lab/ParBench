# Augmentation Integration — Status Report
**Date:** 2026-03-10
**Branch:** `main` (post `Merge branch 'feature/aug'`, commit `8efe990`)
**Author:** Automated via Claude Code + subagent batch runs
**Seed:** 42 for all tests

---

## Executive Summary

The augmentation pipeline (`feature/aug`) is **merged into `main`** and the end-to-end
integration (`augment → build → run → verify`) works correctly at levels 1 and 2.
Of the 49 Rodinia specs confirmed passing at baseline, **43/49 (88%) pass augmentation
at level 2**. The 6 failures are all caused by fixable bugs in the transform implementations.
**Level 4 is essentially unusable** (1/49 PASS) due to three known bugs that combine at
higher aggressiveness. No HeCBench specs could be tested because their source is not on disk.

---

## Git / Branch Status

| Question | Answer |
|----------|--------|
| Is `feature/aug` merged into `main`? | **YES** — commit `8efe990 "Merge branch 'feature/aug'"` |
| Current branch | `main` |
| What the merge brought in | `c_augmentation/` module (5 AST transforms + unit tests), `harness/spec_loader.py` augmentation wiring, `harness/cli.py --augment_level` flag, `scripts/augment_verify.py` |
| Any outstanding changes to merge? | None — branch is fully integrated |

---

## Phase Completion Status

| Phase | Description | Status | Deliverable |
|-------|-------------|--------|-------------|
| 0 | Pre-flight (pytest + prompt diff) | ✓ COMPLETE | 8/8 unit tests PASS |
| 1 | Build dir analysis | ✓ COMPLETE | Confirmed file structure |
| 2 | Implement `augment_verify.py` | ✓ COMPLETE | `scripts/augment_verify.py` |
| 3 | Smoke test (3 specs × L1,L2,L4) | ✓ COMPLETE | 6/9 PASS — see matrix below |
| 4 | Rodinia batch (60 specs × L2) | ✓ COMPLETE | `rodinia_aug_results.md` (43/60 PASS) |
| 5 | Full batch (all specs × L1+L2+L4) | ✓ COMPLETE† | `full_aug_results.md` |
| 6 | Wire into harness pairs workflow | ✓ Already works by design | No action needed |

† **Phase 5 caveat:** All 120 HeCBench specs produce `ERROR` (source directory not on disk).
Phase 5 effectively covers only the 60 Rodinia specs. The "152 passing specs" target from the
plan assumes HeCBench source is available, which it is not on this machine. Actual tested scope:
60 Rodinia × 3 levels = 180 runs.

---

## Phase 3 — Smoke Test Matrix

| Spec | L1 | L2 | L4 | Transforms at L4 |
|------|----|----|----|----|
| rodinia-bfs-cuda | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArith, SwapCondition |
| rodinia-hotspot-omp | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ChangeNames, PointerArith, SwapCondition, Typedef |
| rodinia-bfs-opencl | ✓ PASS | ✓ PASS | ✗ BUILD_FAIL | ArithmeticTransform, ChangeNames, PointerArith, SwapCondition |

**L1: 3/3 PASS | L2: 3/3 PASS | L4: 0/3 PASS**

Notable: `hotspot-omp` fires `TypedefExpansion` at both L1 and L2 (and passes) — these levels
do test real augmentation for this kernel, contrary to the earlier observation that "L1/L2 apply
no transforms." TypedefExpansion is the lightest transform and can fire even at low levels.

---

## Phase 4 — Rodinia Batch at L2 (60 Specs)

**43/60 PASS (71%)** | BUILD_FAIL=13 | FAIL=4 | ERROR=0

### Corrected Baseline: 49 "should-pass" specs (not 51)

The integration plan stated "51 passing Rodinia specs" but baseline harness tests reveal:
- `rodinia-nn-cuda`: FAIL at baseline (SIGSEGV, exit_code=-11) — should be in exclusion list
- `rodinia-nn-omp`: FAIL at baseline (exit_code=1) — should be in exclusion list
- All 4 OpenCL exclusions (nn, kmeans, cfd, pathfinder) and both OMP/CUDA exclusions are correctly identified

**Corrected count: 49 should-pass specs (17 CUDA, 16 OMP, 16 OpenCL)**

### Performance Against Corrected 49-Spec Baseline

| Metric | Value |
|--------|-------|
| Should-pass specs tested | 49 |
| PASS at L2 | **43/49 (88%)** |
| BUILD_FAIL at L2 (augmentation-caused) | **6/49 (12%)** |
| Pre-existing failures (known bad + newly discovered) | 11 |

### L2 Augmentation-Caused Failures (6 specs that should pass)

| Spec | Transforms Applied | Root Cause |
|------|-------------------|------------|
| `rodinia-lud-omp` | SwapCondition | **Bug C**: `while (-1 != opt = getopt_long(...))` → swapped to non-lvalue |
| `rodinia-kmeans-omp` | PointerArithmeticToArrayIndex, SwapCondition | **Bug A**: overlapping candidate byte-offset corruption |
| `rodinia-myocyte-cuda` | ChangeNames, PointerArith, SwapCondition | **Bug A+B**: overlapping subscripts / struct member precedence |
| `rodinia-dwt2d-opencl` | SwapCondition | **Bug A (SwapCondition)**: string literal corrupted (`r95wh` injected into getopt option string) |
| `rodinia-streamcluster-cuda` | TypedefExpansion | **Bug D (NEW)**: TypedefExpansion expands struct pointer type → breaks `.weight` member access |
| `rodinia-streamcluster-omp` | TypedefExpansion | **Bug D (NEW)**: same as above |

---

## Phase 5 — Full Batch at L1+L2+L4

### Rodinia Results by Level

| Level | PASS | BUILD_FAIL | FAIL | ERROR (HeCBench) |
|-------|------|-----------|------|------|
| L1 | 44/180 | 12 | 4 | 120 |
| L2 | 43/180 | 13 | 4 | 120 |
| L4 | 1/180 | 59 | 0 | 120 |

**Rodinia-only pass rate (excl. ERROR):**
- L1: 44/60 (73%)
- L2: 43/60 (71%)
- L4: 1/60 (2%) — only `rodinia-backprop-omp` passes (with `ChangeNames` applied)

### Transform Frequency (all Rodinia runs across L1+L2+L4)

| Transform | Times Applied | Times Caused BUILD_FAIL |
|-----------|--------------|------------------------|
| SwapCondition | 103 | many (Bug C + Bug A) |
| ChangeNames | 73 | few (only at L4) |
| PointerArithmeticToArrayIndex | 73 | many (Bug A + Bug B) |
| ArithmeticTransform | 62 | few |
| TypedefExpansion | 22 | 2+ (Bug D) |

---

## Issue Classification

### BUG-A: Multi-Candidate Byte-Offset Corruption (CRITICAL — blocks L4, affects L2)

**Severity:** CRITICAL
**Affects:** PointerArithmeticToArrayIndex (confirmed) + SwapCondition (likely, see dwt2d-opencl)
**Levels:** L2 (3 specs), L4 (virtually all specs)

**Root cause:** `AstTransform.apply()` collects all candidates from a transform, selects
`N = max(1, int(num_candidates × fraction))` of them at random, then applies them as a
combined `RewriteCandidate` via `_apply_candidate`. All edits are applied in a single pass
in reverse byte-offset order. If two selected candidates have **overlapping or adjacent
byte ranges** (e.g., a nested subscript `iS[i]` inside its parent `J[iS[i]*cols+j]`),
the inner edit shifts bytes before the outer edit's offsets are consumed, producing garbled code.

**Evidence:**
```
// Original (srad-omp):
dS[k] = J[iS[i] * cols + j] - Jc;
// After Bug A (inner + outer both selected):
*((dS) + (k)) = *((J) + (iS[i] * cols + j))ols + j] - Jc;
//                                                 ^^^^ stale bytes from second edit
```
```
// Original (dwt2d-opencl):
getopt_long(argc, argv, "d:p:c:b:l:i:D:f", longopts, &optindex)
// After Bug A in SwapCondition (adjacent candidates corrupt getopt string):
"d:p:c:b:l:i:D:fr95wh"   ← garbled
```

**Fix:** Before applying, filter selected candidates to a non-overlapping set using a standard
interval selection algorithm (sort by start offset, greedily pick non-overlapping). Add
`_validate_rewrite()` call on the combined candidate before applying.

**File:** `c_augmentation/augment_dataset.py` — `AstTransform.apply()`

---

### BUG-B: Struct Member Access Precedence in PointerArithmeticToArrayIndex (HIGH)

**Severity:** HIGH
**Affects:** PointerArithmeticToArrayIndex
**Levels:** L3+ (observed), L2 (in combination with Bug A for myocyte-cuda)

**Root cause:** `arr[i].member` is transformed to `*((arr)+(i)).member`. In C/C++, `.`
has higher precedence than unary `*`, so this parses as `*(arr+i).member` → compiler
interprets as `*((arr+i).member)`, requiring `arr+i` to be a struct (it's a pointer).

**Evidence (bfs-cuda L3):**
```c
// Original:
g_graph_nodes[id].starting
// After Bug B:
*((g_graph_nodes) + (id)).starting
// Error: expression must have class type but it has type "Node *"
```

**Fix:** In `PointerArithmeticToArrayIndex._find_candidates()`, detect when the
`ARRAY_SUBSCRIPT_EXPR` parent is a `MEMBER_REF_EXPR`. Emit `(*((arr)+(i))).member`
instead of `*((arr)+(i)).member`. Simplest safe fix: always emit `(*((arr)+(i)))`
instead of `*((arr)+(i))` to ensure correct precedence in all contexts.

**File:** `c_augmentation/augment_dataset.py` — `PointerArithmeticToArrayIndex`

---

### BUG-C: SwapCondition with Assignment-in-Condition (HIGH)

**Severity:** HIGH
**Affects:** SwapCondition
**Levels:** L2+ (confirmed at L2 for lud-omp), L3+ (hotspot3d-cuda)

**Root cause:** The expression `x = f() == 0` is parsed by C/C++ as `x = (f() == 0)`
(since `==` binds tighter than `=`). The libclang AST shows a `BINARY_OPERATOR ==`
with LHS=`x = f()` and RHS=`0`. `SwapCondition` swaps to `0 == x = f()`, which parses
as `(0 == x) = f()` — assigning to a non-lvalue.

**Evidence (lud-omp L2):**
```c
// Original:
while (-1 != opt = getopt_long(argc, argv, "::vs:n:i:", longopts, NULL))
// After SwapCondition:
while (opt = getopt_long(argc, argv, "::vs:n:i:", longopts, NULL) != -1)
// Error: lvalue required as left operand of assignment
```

**Also confirmed:** `hotspot3d-cuda` at L3+: `if( fp = fopen(file, "r") == 0 )` → broken.

**Fix:** In `SwapCondition._find_candidates()`, skip any `BINARY_OPERATOR ==` node
where either operand's extent contains a `BINARY_OPERATOR =` (assignment) cursor.
Check recursively: if either child subtree has any `=`, `+=`, `-=`, etc. operator,
skip this `==` node.

**File:** `c_augmentation/augment_dataset.py` — `SwapCondition._find_candidates()`

---

### BUG-D: TypedefExpansion Breaks Struct Member Access (NEW — HIGH)

**Severity:** HIGH
**Affects:** TypedefExpansion
**Levels:** L2+ (confirmed at L2 for streamcluster-{cuda,omp})

**Root cause:** TypedefExpansion replaces typedef names with their underlying types.
When the typedef refers to a **struct pointer type** (e.g., `Point *p`), expanding
the typedef can change the apparent type of the pointer, breaking subsequent member
access (`.weight`, `.coord`, etc.).

**Evidence (streamcluster-cuda L2):**
```
// After TypedefExpansion:
streamcluster_cuda_cpu.cpp:802:19: error:
  request for member 'weight' in '*(points.Points::p + ((sizetype)(...)))',
  which is of non-class type 'int'
  802 |       points.p[i].weight = 1.0;
```
TypedefExpansion also causes format-string errors: `size_t numRead` → `unsigned long numRead`,
making `%d` format mismatches compiler errors in kernels with `-Werror=format`.

**Affected specs:** `rodinia-streamcluster-cuda`, `rodinia-streamcluster-omp`
Possibly also: `rodinia-cfd-opencl` (TypedefExpansion applied, BUILD_FAIL — but also pre-existing)

**Fix:** TypedefExpansion should skip typedefs whose underlying type is:
1. A struct, union, or class (i.e., `CursorKind.STRUCT_DECL`, `UNION_DECL`, `CLASS_DECL`)
2. A pointer to a struct/class type
3. `size_t` / `ptrdiff_t` and other C standard integer typedefs that appear in format strings

Safer alternative: Only expand typedefs to primitive numeric types (int, float, double, etc.),
never to pointer types or struct types.

**File:** `c_augmentation/augment_dataset.py` — `TypedefExpansion._find_candidates()`

---

### DATA-1: "51 Passing Rodinia Specs" List is Incorrect (2 overcounts)

**Severity:** MEDIUM (affects planning and metric interpretation)

The integration plan's "51 passing" baseline is overcounted by 2. These specs fail at
baseline even without augmentation:

| Spec | Failure Mode | Confirmed By |
|------|-------------|-------------|
| `rodinia-nn-cuda` | SIGSEGV (exit_code=-11) at runtime | `python3 -m harness verify` → RUN: FAIL |
| `rodinia-nn-omp` | exit_code=1 at runtime | `python3 -m harness verify` → RUN: FAIL |

**Corrected baseline: 49 passing Rodinia specs** (17 CUDA + 16 OMP + 16 OpenCL)

These specs should be added to the per-API exclusion lists or their specs/baseline_results
should be debugged and fixed separately.

---

### DATA-2: HeCBench Source Not On Disk — Phase 5 HeCBench Coverage = 0%

**Severity:** LOW (pre-existing, expected)

All 120 HeCBench specs produce `ERROR` (working directory not found). Phase 5 effectively
only covers Rodinia. To enable HeCBench augmentation testing, clone HeCBench source:
```
git clone https://github.com/zjin-lcf/HeCBench.git HeCBench-master/
```
Then re-run Phase 5 batch. This is the pre-existing known limitation in `CLAUDE.md`.

---

### INT-1: `.cl` Files Not Augmented by `harness prompt` (Known, Unresolved)

**Severity:** MEDIUM
**File:** `harness/spec_loader.py`, line 195, `get_prompt_payload()`

`harness prompt --augment_level N` does NOT augment OpenCL `.cl` files. Only:
```python
[".c", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".dp.cpp"]
```
`.cl` is missing. `augment_verify.py` correctly includes `.cl` in `AUGMENTABLE_SUFFIXES`.

**Impact:** When using `harness prompt` to generate LLM input for OpenCL kernels, the
kernel source file is NOT augmented even with `--augment_level 4`. The LLM receives
un-augmented OpenCL. (The C host code IS augmented.)

**Fix:** Add `".cl"` to the suffix list in `spec_loader.py:get_prompt_payload()` (line ~195).

---

## What Works Well

- **L1 + L2 augmentation:** 43/49 (88%) Rodinia specs pass with meaningful transforms applied
- **Pipeline integration:** `augment_verify.py` correctly orchestrates augment → copy → build → run → verify
- **`harness prompt --augment_level`:** Works correctly for `.c/.cu/.cpp` files; LLM input is visibly transformed
- **Unit tests:** 8/8 pass (ArithmeticTransform, SwapCondition positive/negative, PointerArithmetic, TypedefExpansion, ChangeNames, real kernel pipeline)
- **ArithmeticTransform:** Reliably safe at all tested levels (no failures attributed to it alone)
- **ChangeNames:** Reliably safe when used alone (backprop-omp passes L4 with ChangeNames)
- **SwapCondition:** Works for 22+ specs at L2; breaks only when assignment-in-condition pattern (Bug C) or overlapping candidates (Bug A) present
- **TypedefExpansion:** Works for hotspot-omp (L1+L2), lavamd-opencl (L2); breaks only with struct pointer typedefs (Bug D)

---

## Recommended Priority Order for Next Session

### P0 (Blocking L2+ reliability — fix now)

1. **Fix Bug C** (SwapCondition assignment-in-condition) — simple predicate check
   - File: `c_augmentation/augment_dataset.py` — `SwapCondition._find_candidates()`
   - Expected impact: +2 specs (lud-omp, possibly others) at L2

2. **Fix Bug A** (overlapping candidate byte-offset corruption)
   - File: `c_augmentation/augment_dataset.py` — `AstTransform.apply()`
   - Expected impact: +3 specs (kmeans-omp, myocyte-cuda, dwt2d-opencl) at L2; massive improvement at L4

### P1 (Enables L3+, fixes most remaining L4 failures)

3. **Fix Bug B** (struct member access precedence in PointerArithmeticToArrayIndex)
   - File: same — `PointerArithmeticToArrayIndex._find_candidates()` or rewrite emit
   - Expected impact: +2-3 specs at L3

4. **Fix Bug D** (TypedefExpansion struct pointer expansion)
   - File: same — `TypedefExpansion._find_candidates()`
   - Expected impact: +2 specs at L2 (streamcluster-{cuda,omp})

### P2 (Data quality / harness consistency)

5. **Fix `.cl` augmentation in `harness prompt`**
   - File: `harness/spec_loader.py:get_prompt_payload()` — add `".cl"` to suffix list
   - 1-line fix

6. **Investigate nn-cuda and nn-omp baseline failures**
   - These specs need debugged or their baseline_results corrected
   - Likely data path issue (nn needs specific input files)

7. **Update exclusion lists** to match reality (add nn-cuda, nn-omp to "known-failing")

### P3 (Scale)

8. **Clone HeCBench source** to enable Phase 5 real coverage
9. **Re-run Phase 5** after Bugs A+B+C+D are fixed to get a clean full-scale result

---

## Output Files

All under `results/augmentation/`:

| File | Contents |
|------|---------|
| `phase3_smoke_results.md` | Phase 3 matrix: 3 specs × L1,L2,L4 |
| `rodinia_aug_results.md` | Phase 4 full matrix: 60 Rodinia specs × L2 |
| `full_aug_results.md` | Phase 5 full matrix: all 180 specs × L1,L2,L4 |
| `rodinia_aug_results.json` | Phase 4 machine-readable JSON |
| `full_aug_results.json` | Phase 5 machine-readable JSON (540 entries) |
| `phase3_{cuda,omp,opencl}.json` | Per-API Phase 3 raw results |
| `phase4_{cuda,omp,opencl}.json` | Per-API Phase 4 raw results |
| `phase5_{cuda,omp,opencl}.json` | Per-API Phase 5 raw results |
| `phase_{cuda,omp,opencl}.log` | Full execution logs |

Scripts added this session:
- `scripts/run_augment_batch.py` — batch runner (spec list × levels → JSON + MD)
- `scripts/run_phase_{cuda,omp,opencl}.sh` — per-API stream scripts (P3→P4→P5 chained)
- `scripts/combine_aug_results.py` — merges 3-stream outputs into aggregate reports

---

*Report generated: 2026-03-10 | Next session: fix Bugs A+B+C+D, re-run batch*
