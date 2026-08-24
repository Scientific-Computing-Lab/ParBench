# Benchmark tree pins and fetch recipes

> Moved here 2026-08-16 from a retired onboarding doc (full text in git history).
> This is the standing home for the pinned SHAs and the two fetch recipes.

## Pinned SHAs

| tree | pinned SHA | upstream |
|---|---|---|
| `rodinia` | `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4` | `github.com/yuhc/gpu-rodinia` |
| `mixbench/mixbench-src` | `32edeca98bdd63b32769e3c7460676b9fd567f06` | `github.com/ekondis/mixbench` |
| `xsbench/xsbench-src` | `ba08e5221af6106252b866e50ea123c69d31a4e2` | `github.com/ANL-CESAR/XSBench` |
| `rsbench/rsbench-src` | `34b644787ea9af4fb188e1253da72e09bbed9989` | `github.com/ANL-CESAR/RSBench` |
| `HeCBench-master` | `22785cdd708de5dca56525a277b31fe119171fd1` | `github.com/zjin-lcf/HeCBench` |

Known deviation (accepted 2026-08-16): the Mac's `HeCBench-master` working copy is at
`accf77fde427525ea9a270a7be58a9b56eb36169`, not the pin; the Linux experiment machine
is at the pin. Owner ruling: leave the Mac copy as is.

## Recipe 1: attach history to a tree whose FILES are already present

`--mixed` rewrites only the index, never the working tree. Use for a copied tree that
arrived without `.git`. Guard with `[ -e "$1/.git" ]`, never `git rev-parse --git-dir`
(rev-parse walks UP into the parbench repo and the guard silently no-ops).

```bash
attach() {  # attach <dir> <url> <sha>
  [ -e "$1/.git" ] && { echo "$1 already has .git"; return; }
  ( cd "$1" && git init -q . && git remote add origin "$2" \
    && git fetch -q --depth 1 origin "$3" && git reset -q --mixed FETCH_HEAD ) \
    || { echo "attach FAILED for $1 - fix before continuing"; return 1; }
  echo "$1 -> $(git --git-dir="$1/.git" rev-parse HEAD)"
}
```

## Recipe 2: materialize a tree that is ABSENT

A checkout, not a mixed reset. Used on the Mac 2026-08-16 for the three small suites.

```bash
mkdir -p <dir> && cd <dir>
git init -q . && git remote add origin <url>
git fetch -q --depth 1 origin <sha>
git checkout -q FETCH_HEAD
```

Expect `rodinia` to show as dirty afterwards on build hosts: it carries the documented
toolchain patches (`docs/rodinia_toolchain_patches.diff`; re-apply with
`scripts/spec_tools/apply_rodinia_patches.sh`) plus some checked-in binaries.
The other four trees are gitignored working copies; rodinia is the one submodule
(its dirt is suppressed by `submodule.rodinia.ignore=dirty`). A fetch at the pinned
SHAs dirties nothing in the parbench repo.
