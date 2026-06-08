#!/usr/bin/env python3
"""Generate visualizations/results_data.js and visualizations/build_results_data.js
from the canonical JSON result files produced by the harness and batch runners.

DATA SOURCES
============
results_data.js (window.RESULTS_DATA)
    Merged from:
      results/augmentation/eval_cuda.json    -- 22 Rodinia CUDA specs × L1-L4
      results/augmentation/eval_omp.json     -- 18 Rodinia OMP specs × L1-L4
      results/augmentation/eval_opencl.json  -- 20 Rodinia OpenCL specs × L1-L4

    Each file is a JSON array of augmentation run records with fields:
      spec_id, augment_level, seed, config, transforms_applied,
      transforms_summary, files_changed, build_status, run_status,
      verify_status, overall_status, verify_details, wall_time_seconds,
      exit_code, tempdir, details

build_results_data.js (window.BUILD_DATA)
    Three sections:
    hecbench_phase3  -- 60 HeCBench kernels × CUDA + OMP
        sources:
          results/phase3/cuda_batch3_results.json  -- B2+B3 CUDA results
          results/phase3/omp_batch_results.json    -- B2+B3 OMP results
          (B1 data is hardcoded in this script; it was manually recorded
           from the original results_matrix.md before JSON logging existed)

    hecbench_phase4  -- 20 HeCBench kernels × CUDA/HIP/OMP/SYCL
        source:
          results/phase4/full_results_matrix.md   -- parsed markdown table

    rodinia          -- 60 Rodinia specs baseline (no augmentation)
        source:
          results/rodinia/rodinia_results.json    -- full batch run JSON

USAGE
=====
    source env_parbench/bin/activate
    python3 scripts/generate_viz_data.py [--project-root PATH] [--dry-run]

OPTIONS
=======
    --project-root PATH   Root of parbench repo (default: auto-detected from
                          this script's location: two levels up from scripts/)
    --output-dir PATH     Directory to write JS files into
                          (default: <project-root>/visualizations)
    --only-results        Only regenerate results_data.js
    --only-build          Only regenerate build_results_data.js
    --dry-run             Print what would be written without writing
    -v, --verbose         Print extra progress info
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Hardcoded Batch 1 data for hecbench_phase3
# This data was manually recorded from results/phase3/results_matrix.md before
# JSON logging was added. It covers the 20 B1 kernels for both CUDA and OMP.
# Keys: (kernel) -> {cuda_build, cuda_run, cuda_verify, cuda_build_time,
#                    cuda_run_time, omp_build, omp_run, omp_verify,
#                    omp_build_time, omp_run_time}
# Times are in seconds; null means not applicable (SKIP/FAIL with no time)
# ---------------------------------------------------------------------------
BATCH1_KERNELS = [
    "aes", "bilateral", "binomial", "chacha20", "chi2",
    "convolutionseparable", "dct8x8", "eigenvalue", "fft", "fwt",
    "ising", "lud", "merge", "nbody", "nn",
    "particle-diffusion", "radixsort", "scan", "simplespmv", "sobel",
]

BATCH2_KERNELS = [
    "babelstream", "backprop", "ccsd-trpdrv", "deredundancy", "feynman-kac",
    "fpc", "ga", "keccaktreehash", "laplace3d", "lulesh",
    "maxpool3d", "md5hash", "pathfinder", "pso", "rmsnorm",
    "secp256k1", "softmax-online", "thomas", "tissue", "tsp",
]

BATCH3_KERNELS = [
    "bezier-surface", "convolution3d", "crc64", "floydwarshall", "gaussian",
    "geglu", "heat2d", "iso2dfd", "jenkins-hash", "knn",
    "mandelbrot", "mis", "murmurhash3", "myocyte", "nw",
    "perplexity", "popcount", "sobol", "stencil1d", "triad",
]

# fmt: off
BATCH1_DATA = {
    # kernel: (cuda_build, cuda_run, cuda_verify, cuda_bt, cuda_rt,
    #          omp_build, omp_run, omp_verify, omp_bt, omp_rt)
    "aes":                {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.29,"cuda_run_time":0.07,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.61,"omp_run_time":1.93},
    "bilateral":          {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.74,"cuda_run_time":0.14,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.25,"omp_run_time":2.35},
    "binomial":           {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.58,"cuda_run_time":0.11,  "omp_build":"PASS","omp_run":"TIMEOUT","omp_verify":"FAIL","omp_build_time":0.36,"omp_run_time":600.1},
    "chacha20":           {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.38,"cuda_run_time":0.02,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.16,"omp_run_time":0.001},
    "chi2":               {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.13,"cuda_run_time":12.99, "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.35,"omp_run_time":13.66},
    "convolutionseparable":{"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.16,"cuda_run_time":0.04, "omp_build":"FAIL","omp_run":None,"omp_verify":None,"omp_build_time":None,"omp_run_time":None},
    "dct8x8":             {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.27,"cuda_run_time":0.07,  "omp_build":"PASS","omp_run":"FAIL","omp_verify":"FAIL","omp_build_time":0.33,"omp_run_time":None},
    "eigenvalue":         {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.30,"cuda_run_time":0.42,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.44,"omp_run_time":0.35},
    "fft":                {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.73,"cuda_run_time":0.08,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"FAIL","omp_build_time":0.46,"omp_run_time":2.17},
    "fwt":                {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.21,"cuda_run_time":0.07,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"FAIL","omp_build_time":0.23,"omp_run_time":1.14},
    "ising":              {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.09,"cuda_run_time":0.05,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.35,"omp_run_time":0.19},
    "lud":                {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.07,"cuda_run_time":0.08,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.37,"omp_run_time":0.06},
    "merge":              {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.31,"cuda_run_time":3.20,  "omp_build":"PASS","omp_run":"TIMEOUT","omp_verify":"FAIL","omp_build_time":0.90,"omp_run_time":300.1},
    "nbody":              {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.37,"cuda_run_time":0.31,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.74,"omp_run_time":6.78},
    "nn":                 {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.48,"cuda_run_time":0.08,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.40,"omp_run_time":0.008},
    "particle-diffusion": {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.34,"cuda_run_time":0.21,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"FAIL","omp_build_time":0.36,"omp_run_time":0.64},
    "radixsort":          {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.62,"cuda_run_time":0.05,  "omp_build":"PASS","omp_run":"FAIL","omp_verify":"FAIL","omp_build_time":0.24,"omp_run_time":None},
    "scan":               {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":2.51,"cuda_run_time":0.56,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.89,"omp_run_time":6.07},
    "simplespmv":         {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":3.75,"cuda_run_time":0.12,  "omp_build":"SKIP","omp_run":None,"omp_verify":None,"omp_build_time":None,"omp_run_time":None},
    "sobel":              {"cuda_build":"PASS","cuda_run":"PASS","cuda_verify":"PASS","cuda_build_time":1.45,"cuda_run_time":0.08,  "omp_build":"PASS","omp_run":"PASS","omp_verify":"PASS","omp_build_time":0.52,"omp_run_time":0.01},
}
# fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def js_val(v):
    """Format a Python value as a JS literal (null for None, string or number)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)  # properly escaped JS string
    if isinstance(v, (int, float)):
        return repr(v)
    # fallback: JSON-encode as string
    return json.dumps(str(v))


def kernel_batch(kernel):
    if kernel in BATCH1_KERNELS:
        return "B1"
    if kernel in BATCH2_KERNELS:
        return "B2"
    return "B3"


# ---------------------------------------------------------------------------
# results_data.js generator
# ---------------------------------------------------------------------------

def generate_results_data(project_root: Path, verbose: bool) -> str:
    """Read eval_cuda/omp/opencl.json, merge, and return the JS file content."""
    eval_files = {
        "cuda":   project_root / "results" / "augmentation" / "eval_cuda.json",
        "omp":    project_root / "results" / "augmentation" / "eval_omp.json",
        "opencl": project_root / "results" / "augmentation" / "eval_opencl.json",
    }

    all_records = []
    for api, path in eval_files.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {api}", file=sys.stderr)
            continue
        records = json.loads(path.read_text())
        if verbose:
            print(f"  Loaded {len(records):3d} records from {path.name}")
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No augmentation eval records found — nothing to generate.")

    # Sort by (spec_id, augment_level) for deterministic output
    all_records.sort(key=lambda r: (r["spec_id"], r["augment_level"]))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"// ParBench Augmentation Results Data",
        f"// Generated by: scripts/generate_viz_data.py",
        f"// Generated at: {now}",
        f"//",
        f"// Source files (results/augmentation/):",
        f"//   eval_cuda.json, eval_omp.json, eval_opencl.json",
        f"//",
        f"// Total records: {len(all_records)}",
        f"// Consumed by:  visualizations/results.html  (window.RESULTS_DATA)",
        f"",
        f"window.RESULTS_DATA = {json.dumps(all_records, separators=(',', ':'))};",
        f"",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# build_results_data.js generator — hecbench_phase3 section
# ---------------------------------------------------------------------------

def _load_phase3_hecbench(project_root: Path, verbose: bool) -> list:
    """
    Build the hecbench_phase3 rows.
    B1 data comes from BATCH1_DATA (hardcoded from pre-JSON era).
    B2+B3 CUDA comes from results/phase3/cuda_batch3_results.json.
    B2+B3 OMP comes from results/phase3/omp_batch_results.json.
    convolution3d OMP is hardcoded SKIP (no OMP source directory exists).
    """
    cuda_json = project_root / "results" / "phase3" / "cuda_batch3_results.json"
    omp_json  = project_root / "results" / "phase3" / "omp_batch_results.json"

    cuda_by_kernel = {}
    omp_by_kernel  = {}

    if cuda_json.exists():
        data = json.loads(cuda_json.read_text())
        cuda_by_kernel = {r["kernel"]: r for r in data["results"]}
        if verbose:
            print(f"  Loaded {len(cuda_by_kernel)} CUDA phase3 results from {cuda_json.name}")
    else:
        print(f"  WARNING: {cuda_json} not found", file=sys.stderr)

    if omp_json.exists():
        data = json.loads(omp_json.read_text())
        omp_by_kernel = {r["kernel"]: r for r in data["results"]}
        if verbose:
            print(f"  Loaded {len(omp_by_kernel)} OMP phase3 results from {omp_json.name}")
    else:
        print(f"  WARNING: {omp_json} not found", file=sys.stderr)

    all_kernels = sorted(BATCH1_KERNELS + BATCH2_KERNELS + BATCH3_KERNELS)
    rows = []

    for kernel in all_kernels:
        batch = kernel_batch(kernel)
        row = {"kernel": kernel, "batch": batch}

        # CUDA
        if kernel in BATCH1_DATA:
            b1 = BATCH1_DATA[kernel]
            row["cuda_build"]      = b1["cuda_build"]
            row["cuda_run"]        = b1["cuda_run"]
            row["cuda_verify"]     = b1["cuda_verify"]
            row["cuda_build_time"] = b1["cuda_build_time"]
            row["cuda_run_time"]   = b1["cuda_run_time"]
        elif kernel in cuda_by_kernel:
            r = cuda_by_kernel[kernel]
            row["cuda_build"]      = r["build_status"]
            row["cuda_run"]        = r["run_status"]
            row["cuda_verify"]     = r["verify_status"]
            row["cuda_build_time"] = round(r["build_time"], 2)
            row["cuda_run_time"]   = round(r["run_time"], 2) if r.get("run_time") is not None else None
        else:
            row["cuda_build"] = row["cuda_run"] = row["cuda_verify"] = "UNKNOWN"
            row["cuda_build_time"] = row["cuda_run_time"] = None

        # OMP — convolution3d has no OMP target
        if kernel == "convolution3d":
            row["omp_build"] = "SKIP"
            row["omp_run"] = row["omp_verify"] = None
            row["omp_build_time"] = row["omp_run_time"] = None
        elif kernel in BATCH1_DATA:
            b1 = BATCH1_DATA[kernel]
            row["omp_build"]      = b1["omp_build"]
            row["omp_run"]        = b1["omp_run"]
            row["omp_verify"]     = b1["omp_verify"]
            row["omp_build_time"] = b1["omp_build_time"]
            row["omp_run_time"]   = b1["omp_run_time"]
        elif kernel in omp_by_kernel:
            r = omp_by_kernel[kernel]
            row["omp_build"]      = r["build_status"]
            row["omp_run"]        = r["run_status"]
            row["omp_verify"]     = r["verify_status"]
            row["omp_build_time"] = round(r["build_time"], 2)
            row["omp_run_time"]   = round(r["run_time"], 2) if r.get("run_time") is not None else None
        else:
            row["omp_build"] = row["omp_run"] = row["omp_verify"] = "UNKNOWN"
            row["omp_build_time"] = row["omp_run_time"] = None

        rows.append(row)

    return rows


def _fmt_phase3_row(row: dict) -> str:
    def _s(v):
        if v is None:
            return "null"
        return json.dumps(v)

    def _n(v):
        if v is None:
            return "null"
        return str(v)

    k = row["kernel"]
    b = row["batch"]
    return (
        f'  {{ kernel:{json.dumps(k):<28}, batch:{json.dumps(b)}, '
        f'cuda_build:{_s(row["cuda_build"])}, cuda_run:{_s(row["cuda_run"])}, '
        f'cuda_verify:{_s(row["cuda_verify"])}, cuda_build_time:{_n(row["cuda_build_time"])}, '
        f'cuda_run_time:{_n(row["cuda_run_time"])},  '
        f'omp_build:{_s(row["omp_build"])}, omp_run:{_s(row["omp_run"])}, '
        f'omp_verify:{_s(row["omp_verify"])}, omp_build_time:{_n(row["omp_build_time"])}, '
        f'omp_run_time:{_n(row["omp_run_time"])} }}'
    )


# ---------------------------------------------------------------------------
# build_results_data.js generator — hecbench_phase4 section
# ---------------------------------------------------------------------------

def _load_phase4_hecbench(project_root: Path, verbose: bool) -> list:
    """
    Parse results/phase4/full_results_matrix.md markdown table.
    The table has columns: Kernel | API | Build | Run | Verify
    We pivot to one row per kernel with columns: cuda, hip, omp, sycl (verify status).
    """
    md_path = project_root / "results" / "phase4" / "full_results_matrix.md"
    if not md_path.exists():
        print(f"  WARNING: {md_path} not found — phase4 section will be empty", file=sys.stderr)
        return []

    # Parse markdown table: skip header and separator rows
    by_kernel: dict[str, dict] = {}
    for line in md_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| Kernel") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 5:
            continue
        kernel, api, build, run, verify = parts[0], parts[1], parts[2], parts[3], parts[4]
        if kernel not in by_kernel:
            by_kernel[kernel] = {}
        # Use the overall verify status as the representative result for the cell
        # (matches what build_results_data.js shows)
        by_kernel[kernel][api] = verify

    rows = []
    for kernel in sorted(by_kernel.keys()):
        apis = by_kernel[kernel]
        rows.append({
            "kernel": kernel,
            "cuda":   apis.get("cuda", "UNKNOWN"),
            "hip":    apis.get("hip",  "UNKNOWN"),
            "omp":    apis.get("omp",  "UNKNOWN"),
            "sycl":   apis.get("sycl", "UNKNOWN"),
        })

    if verbose:
        print(f"  Loaded {len(rows)} kernels from {md_path.name}")
    return rows


def _fmt_phase4_row(row: dict) -> str:
    k = row["kernel"]
    return (
        f'  {{ kernel:{json.dumps(k):<28}, '
        f'cuda:{json.dumps(row["cuda"])}, hip:{json.dumps(row["hip"])}, '
        f'omp:{json.dumps(row["omp"])},  sycl:{json.dumps(row["sycl"])} }}'
    )


# ---------------------------------------------------------------------------
# build_results_data.js generator — rodinia section
# ---------------------------------------------------------------------------

def _load_rodinia_baseline(project_root: Path, verbose: bool) -> list:
    """
    Read results/rodinia/rodinia_results.json.
    Returns one row per spec with fields: spec_id, slug, api, build, run, verify.
    """
    json_path = project_root / "results" / "rodinia" / "rodinia_results.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found — rodinia section will be empty", file=sys.stderr)
        return []

    data = json.loads(json_path.read_text())
    results = data.get("results", data) if isinstance(data, dict) else data

    rows = []
    for r in results:
        rows.append({
            "spec_id": r["spec_id"],
            "slug":    r["slug"],
            "api":     r["api"],
            "build":   r["build_status"],
            "run":     r["run_status"],
            "verify":  r["verify_status"],
        })

    # Sort by (slug, api) for a clean output
    api_order = {"cuda": 0, "omp": 1, "opencl": 2}
    rows.sort(key=lambda r: (r["slug"], api_order.get(r["api"], 99)))

    if verbose:
        print(f"  Loaded {len(rows)} Rodinia baseline results from {json_path.name}")
    return rows


def _fmt_rodinia_row(row: dict) -> str:
    spec_id = row["spec_id"]
    slug    = row["slug"]
    api     = row["api"]
    return (
        f'  {{ spec_id:{json.dumps(spec_id):<44}, '
        f'slug:{json.dumps(slug):<20}, '
        f'api:{json.dumps(api):<10}, '
        f'build:{json.dumps(row["build"]):<10}, '
        f'run:{json.dumps(row["run"])}, verify:{json.dumps(row["verify"])} }}'
    )


# ---------------------------------------------------------------------------
# build_results_data.js generator — top level
# ---------------------------------------------------------------------------

def generate_build_data(project_root: Path, verbose: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        print("  Loading hecbench_phase3 data...")
    phase3_rows = _load_phase3_hecbench(project_root, verbose)

    if verbose:
        print("  Loading hecbench_phase4 data...")
    phase4_rows = _load_phase4_hecbench(project_root, verbose)

    if verbose:
        print("  Loading rodinia baseline data...")
    rodinia_rows = _load_rodinia_baseline(project_root, verbose)

    lines = [
        f"// ParBench Build Results Data",
        f"// Generated by: scripts/generate_viz_data.py",
        f"// Generated at: {now}",
        f"//",
        f"// Source files:",
        f"//   results/phase3/cuda_batch3_results.json  (B2+B3 CUDA, hecbench_phase3)",
        f"//   results/phase3/omp_batch_results.json    (B2+B3 OMP, hecbench_phase3)",
        f"//   results/phase4/full_results_matrix.md    (hecbench_phase4)",
        f"//   results/rodinia/rodinia_results.json     (rodinia baseline)",
        f"//   BATCH1_DATA (hardcoded in script)        (B1 CUDA+OMP, hecbench_phase3)",
        f"//",
        f"// Consumed by: visualizations/build_results.html (window.BUILD_DATA)",
        f"",
        f"window.BUILD_DATA = {{",
        f"",
    ]

    # hecbench_phase3
    cuda_pass = sum(1 for r in phase3_rows if r.get("cuda_build") == "PASS" and r.get("cuda_run") == "PASS" and r.get("cuda_verify") == "PASS")
    omp_pass  = sum(1 for r in phase3_rows if r.get("omp_build") == "PASS" and r.get("omp_run") == "PASS" and r.get("omp_verify") == "PASS")
    lines += [
        f"// --- HeCBench Phase 3: {len(phase3_rows)} kernels x CUDA + OMP ---",
        f"// Source: results/phase3/cuda_batch3_results.json + omp_batch_results.json",
        f"// CUDA: {cuda_pass}/{len(phase3_rows)} PASS | OMP: {omp_pass}/{len(phase3_rows)} PASS",
        f"hecbench_phase3: [",
    ]
    for i, row in enumerate(phase3_rows):
        comma = "," if i < len(phase3_rows) - 1 else ""
        lines.append(_fmt_phase3_row(row) + comma)
    lines += ["],", ""]

    # hecbench_phase4
    phase4_pass = sum(1 for r in phase4_rows if r.get("cuda") == "PASS")
    lines += [
        f"// --- HeCBench Phase 4: {len(phase4_rows)} kernels x 4 APIs (CUDA/HIP/OMP/SYCL) ---",
        f"// Source: results/phase4/full_results_matrix.md",
        f"// CUDA: {phase4_pass}/{len(phase4_rows)} PASS",
        f"hecbench_phase4: [",
    ]
    for i, row in enumerate(phase4_rows):
        comma = "," if i < len(phase4_rows) - 1 else ""
        lines.append(_fmt_phase4_row(row) + comma)
    lines += ["],", ""]

    # rodinia
    rod_pass = sum(1 for r in rodinia_rows if r.get("build") == "PASS" and r.get("run") == "PASS" and r.get("verify") == "PASS")
    lines += [
        f"// --- Rodinia Baseline: {len(rodinia_rows)} specs ---",
        f"// Source: results/rodinia/rodinia_results.json",
        f"// {rod_pass}/{len(rodinia_rows)} full PASS",
        f"rodinia: [",
    ]
    for i, row in enumerate(rodinia_rows):
        comma = "," if i < len(rodinia_rows) - 1 else ""
        lines.append(_fmt_rodinia_row(row) + comma)
    lines += ["]", ""]

    lines.append("}; // end window.BUILD_DATA")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root of the parbench repo. Defaults to two levels above this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write JS files. Defaults to <project-root>/visualizations/",
    )
    parser.add_argument(
        "--only-results",
        action="store_true",
        help="Only regenerate results_data.js",
    )
    parser.add_argument(
        "--only-build",
        action="store_true",
        help="Only regenerate build_results_data.js",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing any files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extra progress info",
    )
    args = parser.parse_args()

    # Resolve project root
    if args.project_root is not None:
        project_root = args.project_root.resolve()
    else:
        # This script lives at <project_root>/scripts/generate_viz_data.py
        project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")

    # Resolve output dir
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "visualizations"
    print(f"Output dir:   {output_dir}")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    generate_results = not args.only_build
    generate_build   = not args.only_results

    # --- results_data.js ---
    if generate_results:
        print("\nGenerating results_data.js ...")
        content = generate_results_data(project_root, args.verbose)
        out_path = output_dir / "results_data.js"
        if args.dry_run:
            print(f"  [dry-run] Would write {len(content):,} bytes to {out_path}")
        else:
            out_path.write_text(content)
            size_kb = out_path.stat().st_size / 1024
            print(f"  Wrote {size_kb:.1f} KB -> {out_path}")

    # --- build_results_data.js ---
    if generate_build:
        print("\nGenerating build_results_data.js ...")
        content = generate_build_data(project_root, args.verbose)
        out_path = output_dir / "build_results_data.js"
        if args.dry_run:
            print(f"  [dry-run] Would write {len(content):,} bytes to {out_path}")
        else:
            out_path.write_text(content)
            size_kb = out_path.stat().st_size / 1024
            print(f"  Wrote {size_kb:.1f} KB -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
