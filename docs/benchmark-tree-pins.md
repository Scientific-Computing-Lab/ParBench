# Benchmark tree pins and fetch recipes

ParBench does not vendor benchmark source.
Each suite is fetched at a pinned commit, so every spec's `provenance.repo_root` resolves to the same bytes the paper measured.
Only `rodinia` is a git submodule; the other four trees are gitignored working copies that the recipes below create.

## Pinned commits

| Suite | Directory | Pinned commit | Upstream |
|---|---|---|---|
| Rodinia | `rodinia/rodinia-src` | `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4` | `github.com/yuhc/gpu-rodinia` |
| HeCBench | `HeCBench-master` | `22785cdd708de5dca56525a277b31fe119171fd1` | `github.com/zjin-lcf/HeCBench` |
| mixbench | `mixbench/mixbench-src` | `32edeca98bdd63b32769e3c7460676b9fd567f06` | `github.com/ekondis/mixbench` |
| XSBench | `xsbench/xsbench-src` | `ba08e5221af6106252b866e50ea123c69d31a4e2` | `github.com/ANL-CESAR/XSBench` |
| RSBench | `rsbench/rsbench-src` | `34b644787ea9af4fb188e1253da72e09bbed9989` | `github.com/ANL-CESAR/RSBench` |

Rodinia, mixbench, XSBench, and RSBench specs record the same commit under `provenance.repository.commit`, and the two agree.
HeCBench specs mostly record an `archive-download` provenance instead (129 of 135; the other 6 carry the pinned commit) - the pin in this table is the authoritative one.
`git rev-parse HEAD` inside a fetched tree is the verification step: it must print the commit in this table.

Sizes, so you can pick what to fetch: Rodinia ~101 MB, HeCBench ~1.2 GB, and the three small suites a few MB each.
A Rodinia-only checkout is enough for the 60 Rodinia specs and for all analysis work.

## Rodinia (the one submodule)

Run from the repository root.

```bash
git submodule update --init rodinia
git -C rodinia rev-parse HEAD    # must print 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4

# Specs address Rodinia as rodinia/rodinia-src (provenance.repo_root), but the
# submodule checks out at rodinia/. This symlink bridges the two; without it no
# Rodinia spec resolves its source files.
ln -sfn . rodinia/rodinia-src
```

Then apply the two shipped patches.
Both are tracked under `patches/` and both apply cleanly onto the pinned checkout.

```bash
# Dry run first: --check changes nothing and reports whether the patch applies.
git -C rodinia apply --check -p1 "$PWD/patches/rodinia-build-fixes.patch"
git -C rodinia apply --check -p1 "$PWD/patches/rodinia-hotspot-correctness.patch"

git -C rodinia apply -p1 "$PWD/patches/rodinia-build-fixes.patch"
git -C rodinia apply -p1 "$PWD/patches/rodinia-hotspot-correctness.patch"
```

`rodinia-build-fixes.patch` carries toolchain fixes for nine files (`common/make.config`, four CUDA makefiles, four OpenCL makefiles).
One fix is toolchain-specific: it sets `CUDA_DIR = /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3` in `common/make.config`, the reference machine's NVIDIA HPC SDK path.
If your CUDA lives elsewhere, apply the patch and then edit that one variable to your CUDA root; nothing else in the patch is machine-specific.
CPU-only work (OpenMP kernels such as `rodinia-nw-omp`) never reads `CUDA_DIR`, so the default is harmless there.
`rodinia-hotspot-correctness.patch` restores a missing `else` branch in `openmp/hotspot/hotspot_openmp.cpp` so `delta` is never reused stale.

`git apply --check` doubles as the idempotence test: on an already-patched tree the forward check fails and `--check --reverse` succeeds.
If neither direction applies, the tree has diverged and needs hand inspection - do not force the patch.

Expect `rodinia` to show as dirty on a build host afterwards: it carries these patches plus some checked-in binaries.
The submodule is configured with `ignore = dirty`, so that dirt does not surface in the parent repository's `git status`.

## The other four suites

These are gitignored working copies, so no git command in the parent repository brings them in.
Fetch each at its pinned commit, from the repository root.

```bash
fetch_pinned() {  # fetch_pinned <dir> <url> <sha>
  mkdir -p "$1"
  git -C "$1" init -q .
  git -C "$1" remote add origin "$2" 2>/dev/null || true
  git -C "$1" fetch -q --depth 1 origin "$3"
  git -C "$1" checkout -q FETCH_HEAD
  echo "$1 -> $(git -C "$1" rev-parse HEAD)"
}

fetch_pinned mixbench/mixbench-src https://github.com/ekondis/mixbench  32edeca98bdd63b32769e3c7460676b9fd567f06
fetch_pinned xsbench/xsbench-src   https://github.com/ANL-CESAR/XSBench ba08e5221af6106252b866e50ea123c69d31a4e2
fetch_pinned rsbench/rsbench-src   https://github.com/ANL-CESAR/RSBench 34b644787ea9af4fb188e1253da72e09bbed9989

# HeCBench keeps its *.tar.bz datasets in Git LFS, and the upstream repository is
# currently over its GitHub LFS budget, so the smudge filter fails and the
# checkout aborts partway. Skipping the smudge leaves those files as pointer
# stubs and lets the checkout finish. No ParBench spec reads a .tar.bz, so the
# stubs cost nothing.
GIT_LFS_SKIP_SMUDGE=1 fetch_pinned HeCBench-master https://github.com/zjin-lcf/HeCBench 22785cdd708de5dca56525a277b31fe119171fd1
```

Each call echoes the resolved `HEAD`; check it against the table above.
Fetching an arbitrary commit by SHA works because GitHub serves reachable non-tip commits; `--depth 1` keeps the download to one commit.

Fetching at the pinned SHAs dirties nothing in the ParBench repository itself.

## Attaching history to a tree whose files are already present

Use this when a tree arrived as a plain copy, with source files but no `.git`.
`git reset --mixed` rewrites only the index, never the working tree, so the copied files survive.

```bash
attach() {  # attach <dir> <url> <sha>
  [ -e "$1/.git" ] && { echo "$1 already has .git"; return; }
  ( cd "$1" && git init -q . && git remote add origin "$2" \
    && git fetch -q --depth 1 origin "$3" && git reset -q --mixed FETCH_HEAD ) \
    || { echo "attach FAILED for $1 - fix before continuing"; return 1; }
  echo "$1 -> $(git --git-dir="$1/.git" rev-parse HEAD)"
}
```

Guard the check with `[ -e "$dir/.git" ]`, never `git -C "$dir" rev-parse --git-dir`.
`rev-parse` walks up the directory tree, finds the enclosing ParBench repository, and returns 0 for a tree that has no history of its own, which makes the guard silently skip every call.
Use `-e` rather than `-d`: a submodule's `.git` is a file, not a directory.
