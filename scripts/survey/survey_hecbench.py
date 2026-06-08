#!/usr/bin/env python3
"""
Scan the HeCBench src/ directory and produce:
  1. hecbench_full_kernel_survey.csv   — every kernel with metadata
  2. hecbench_candidate_kernels.csv     — filtered candidates for ParBench
  3. Printed summary statistics
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
HECBENCH_SRC = Path("/home/samyak/Downloads/HeCBench-master/src")
OUTPUT_DIR = Path("/home/samyak/Desktop/parbench_sam/analysis")
FULL_CSV = OUTPUT_DIR / "hecbench_full_kernel_survey.csv"
FILTERED_CSV = OUTPUT_DIR / "hecbench_candidate_kernels.csv"

API_SUFFIXES = ["cuda", "hip", "sycl", "omp"]
SOURCE_EXTS = {".cu", ".cpp", ".h", ".hpp", ".c"}

# Self-checking patterns (case-insensitive)
VERIFY_PATTERNS = re.compile(
    r"PASS|FAIL|pass|fail|verify|correct|error|mismatch|memcmp|PASSED|FAILED",
    re.IGNORECASE,
)

EXISTING_KERNELS = {
    "nn",
    "scan",
    "particle-diffusion",
    "binomial",
    "radixsort",
    "aes",
    "chacha20",
    "lud",
    "eigenvalue",
    "nbody",
    "simplespmv",
    "sobel",
    "dct8x8",
    "convolutionseparable",
    "fft",
    "ising",
    "bilateral",
    "chi2",
    "merge",
    "fwt",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def discover_kernels(src_dir: Path) -> dict[str, set[str]]:
    """Return {kernel_name: {api_suffix, ...}}."""
    kernels: dict[str, set[str]] = defaultdict(set)
    for entry in sorted(src_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        for api in API_SUFFIXES:
            suffix = f"-{api}"
            if name.endswith(suffix):
                kernel = name[: -len(suffix)]
                kernels[kernel].add(api)
                break
    return dict(kernels)


def count_source_files(kernel_dir: Path) -> int:
    """Count source files in a kernel directory."""
    count = 0
    if not kernel_dir.is_dir():
        return 0
    for f in kernel_dir.iterdir():
        if f.is_file() and f.suffix in SOURCE_EXTS:
            count += 1
    return count


def has_makefile(kernel_dir: Path) -> bool:
    """Check if a Makefile exists."""
    return (kernel_dir / "Makefile").is_file() or (kernel_dir / "makefile").is_file()


def has_self_check(kernel_dir: Path) -> bool:
    """Scan source files for self-checking patterns."""
    if not kernel_dir.is_dir():
        return False
    for f in kernel_dir.iterdir():
        if f.is_file() and f.suffix in SOURCE_EXTS:
            try:
                text = f.read_text(errors="ignore")
                if VERIFY_PATTERNS.search(text):
                    return True
            except Exception:
                pass
    return False


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    if not HECBENCH_SRC.is_dir():
        print(f"ERROR: HeCBench src/ not found at {HECBENCH_SRC}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering kernels …")
    kernels = discover_kernels(HECBENCH_SRC)
    print(f"  Found {len(kernels)} unique kernel names.\n")

    # ── Build full survey rows ────────────────────────────────────────────────
    rows = []
    for kname in sorted(kernels):
        apis = kernels[kname]
        cuda_dir = HECBENCH_SRC / f"{kname}-cuda"
        has_all4 = all(api in apis for api in API_SUFFIXES)
        n_src = count_source_files(cuda_dir)
        makefile = has_makefile(cuda_dir)
        selfcheck = has_self_check(cuda_dir)
        api_str = ",".join(sorted(apis))
        rows.append(
            {
                "kernel": kname,
                "apis": api_str,
                "all_4_apis": has_all4,
                "cuda_source_files": n_src,
                "has_makefile": makefile,
                "has_self_check": selfcheck,
            }
        )

    # ── Write full CSV ────────────────────────────────────────────────────────
    fieldnames = [
        "kernel",
        "apis",
        "all_4_apis",
        "cuda_source_files",
        "has_makefile",
        "has_self_check",
    ]
    with open(FULL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Full survey written to {FULL_CSV}  ({len(rows)} kernels)")

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = [
        r
        for r in rows
        if r["all_4_apis"]
        and r["has_makefile"]
        and r["has_self_check"]
        and r["kernel"] not in EXISTING_KERNELS
        and 1 <= r["cuda_source_files"] <= 15
    ]

    with open(FILTERED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)
    print(f"Filtered candidates written to {FILTERED_CSV}  ({len(filtered)} kernels)\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(rows)
    all4_count = sum(1 for r in rows if r["all_4_apis"])
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total unique kernels found        : {total}")
    print(f"  Kernels with all 4 API variants   : {all4_count}")
    print(f"  Kernels passing ALL filters       : {len(filtered)}")
    print(f"    (all 4 APIs, Makefile, self-check, not existing, 1-15 src files)")
    print()

    # Source-file distribution of filtered candidates
    print("Filtered candidates — source-file count distribution:")
    buckets = defaultdict(int)
    for r in filtered:
        n = r["cuda_source_files"]
        if n <= 3:
            buckets["1-3 files"] += 1
        elif n <= 6:
            buckets["4-6 files"] += 1
        elif n <= 10:
            buckets["7-10 files"] += 1
        else:
            buckets["11-15 files"] += 1
    for bucket in ["1-3 files", "4-6 files", "7-10 files", "11-15 files"]:
        print(f"  {bucket:>12s} : {buckets.get(bucket, 0)}")
    print()

    # Rough domain inference from kernel name
    domain_keywords = {
        "Linear Algebra": [
            "gemm",
            "spmv",
            "lu",
            "cholesky",
            "qr",
            "svd",
            "blas",
            "matrix",
            "sparse",
            "tridiagonal",
            "lanczos",
            "csr",
            "spmm",
            "tensor",
        ],
        "Image / Signal": [
            "bilateral",
            "sobel",
            "conv",
            "dct",
            "fft",
            "filter",
            "image",
            "blur",
            "median",
            "bm3d",
            "warp",
            "resize",
            "hist",
            "color",
            "jpeg",
            "rgb",
        ],
        "Sorting / Scan": [
            "sort",
            "scan",
            "merge",
            "radix",
            "bitonic",
            "prefix",
            "compact",
        ],
        "Graph / Tree": [
            "bfs",
            "sssp",
            "page",
            "graph",
            "scc",
            "cc",
            "mst",
            "tree",
            "cluster",
        ],
        "Physics / Sim": [
            "nbody",
            "particle",
            "md",
            "lbm",
            "ising",
            "fluid",
            "heat",
            "diffus",
            "cfd",
            "stencil",
            "jacobi",
            "laplace",
            "poisson",
            "wave",
        ],
        "Crypto": [
            "aes",
            "sha",
            "md5",
            "chacha",
            "encrypt",
            "hash",
            "cipher",
            "rng",
            "random",
        ],
        "ML / DNN": [
            "softmax",
            "attention",
            "adam",
            "layer",
            "norm",
            "dropout",
            "relu",
            "gelu",
            "embed",
            "bert",
            "transform",
            "conv2d",
            "pool",
            "batch",
        ],
        "Reduction": ["reduce", "dot", "sum", "min", "max", "norm", "accumul"],
        "Finance": ["binomial", "black", "monte", "option", "price"],
    }

    domain_counts = defaultdict(int)
    for r in filtered:
        kname_lower = r["kernel"].lower()
        matched = False
        for domain, kws in domain_keywords.items():
            if any(kw in kname_lower for kw in kws):
                domain_counts[domain] += 1
                matched = True
                break
        if not matched:
            domain_counts["Other / Unknown"] += 1

    print("Approximate domain distribution of filtered candidates:")
    for domain in sorted(domain_counts, key=lambda d: -domain_counts[d]):
        print(f"  {domain:<20s} : {domain_counts[domain]}")
    print()

    # Print filtered kernel names for quick review
    print("Filtered candidate kernel names:")
    for i, r in enumerate(filtered, 1):
        print(f"  {i:3d}. {r['kernel']:<40s}  (src_files={r['cuda_source_files']})")


if __name__ == "__main__":
    main()
