#!/usr/bin/env python3
"""
Generate ParBench spec files for the XSBench benchmark suite.

XSBench is a Monte Carlo neutron cross-section lookup mini-app from ANL-CESAR.
Single kernel, multiple API variants: openmp-threading, cuda, opencl, openmp-offload.

Writes: specs/xsbench-xsbench-{api}.json  (one per API variant)
        manifest.jsonl                      (entries appended)

Usage:
    python3 scripts/generators/generate_xsbench_specs.py [--dry-run] [--force]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
MANIFEST_FILE = PROJECT_ROOT / "manifest.jsonl"

XSBENCH_REPO_URL = "https://github.com/ANL-CESAR/XSBench"
XSBENCH_COMMIT = "ba08e5221af6106252b866e50ea123c69d31a4e2"
XSBENCH_BRANCH = "master"
XSBENCH_REPO_ROOT = "xsbench/xsbench-src"
XSBENCH_LICENSE = "MIT"

REFERENCE_PLATFORM = {
    "platform_id": "rtx4070-r9-7900x",
    "gpu": {
        "model": "NVIDIA GeForce RTX 4070",
        "architecture": "Ada Lovelace",
        "compute_capability": "sm_89",
        "vram_gb": 12,
        "memory_bandwidth_gbps": 504,
        "cuda_cores": 5888,
    },
    "cpu": {
        "model": "AMD Ryzen 9 7900X",
        "cores": 12,
        "threads": 24,
        "base_clock_ghz": 4.7,
    },
    "software": {
        "os": "Ubuntu 22.04 LTS",
        "cuda_toolkit": "12.x",
        "gcc": "11+",
        "driver": "525.x+",
    },
}

# ---------------------------------------------------------------------------
# Per-variant configuration
# ---------------------------------------------------------------------------
# Each variant specifies:
#   source_path:    subdirectory within xsbench-src/
#   parallel_api:   ParBench parallel_api enum value
#   source_files:   .c/.cu files for prompt_payload
#   header_files:   .h/.cuh files for support_files (LLM context, not translation targets)
#   opencl_files:   .cl device kernel files (OpenCL only)
#   language:       C or C++
#   language_std:   language standard
#   build_cmd:      make command with necessary overrides
#   clean_cmd:      make clean command
#   build_vars:     build.variables dict or None
#   hw_target:      gpu or cpu
#   hw_requirements: hardware.requirements dict
#   dependencies:   system dependency list
#   correctness_args: run arguments for correctness config
#   performance_args: run arguments for performance config
#   corr_description: description for correctness config
#   perf_description: description for performance config

VARIANTS = {
    "omp": {
        "source_path": "openmp-threading",
        "parallel_api": "omp",
        "source_files": [
            "Main.c", "io.c", "Simulation.c",
            "GridInit.c", "XSutils.c", "Materials.c",
        ],
        "header_files": ["XSbench_header.h"],
        "opencl_files": [],
        "language": "C",
        "language_std": "C11",
        "build_cmd": "make",
        "clean_cmd": "make clean",
        "build_vars": None,
        "hw_target": "cpu",
        "hw_requirements": None,
        "dependencies": ["gcc>=9.0"],
        "correctness_args": ["-s", "small"],
        "performance_args": ["-s", "large"],
        "corr_description":
            "Small input for correctness verification of xsbench (OpenMP threading)",
        "perf_description":
            "Large input for performance measurement of xsbench (OpenMP threading)",
    },
    "cuda": {
        "source_path": "cuda",
        "parallel_api": "cuda",
        "source_files": [
            "Main.cu", "io.cu", "Simulation.cu",
            "GridInit.cu", "XSutils.cu", "Materials.cu",
        ],
        "header_files": ["XSbench_header.cuh"],
        "opencl_files": [],
        "language": "C++",
        "language_std": "C++14",
        "build_cmd": "make SM_VERSION=89",
        "clean_cmd": "make clean",
        "build_vars": {
            "SM_VERSION": {
                "description":
                    "CUDA compute capability without sm_ prefix (89 for RTX 4070)",
                "default": "89",
                "detection":
                    "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
                    " | head -1 | tr -d '.'",
            }
        },
        "hw_target": "gpu",
        "hw_requirements": {
            "gpu": {
                "vendor": "NVIDIA",
                "min_compute_capability": "sm_60",
                "min_memory_gb": 4,
            }
        },
        "dependencies": ["cuda-toolkit>=11.0", "gcc>=9.0"],
        # CUDA variant does NOT support history-based simulation — must use -m event
        "correctness_args": ["-m", "event", "-s", "small"],
        "performance_args": ["-m", "event", "-s", "large"],
        "corr_description":
            "Small event-based input for correctness verification of xsbench (CUDA)",
        "perf_description":
            "Large event-based input for performance measurement of xsbench (CUDA)",
    },
    "opencl": {
        "source_path": "opencl",
        "parallel_api": "opencl",
        "source_files": [
            "Main.c", "io.c", "Simulation.c", "GridInit.c",
            "XSutils.c", "CLutils.c", "Materials.c",
        ],
        "header_files": ["XSbench_header.h", "CLutils.h"],
        "opencl_files": ["kernel.cl"],
        "language": "C",
        "language_std": "C11",
        "build_cmd": (
            "make"
            " CFLAGS='-std=gnu99 -Wall -O3"
            " -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/include'"
            " LDFLAGS='-lm -lOpenCL"
            " -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/lib64'"
        ),
        "clean_cmd": "make clean",
        "build_vars": None,
        "hw_target": "gpu",
        "hw_requirements": {
            "gpu": {
                "vendor": "NVIDIA",
                "min_compute_capability": "sm_60",
                "min_memory_gb": 4,
            }
        },
        "dependencies": ["opencl-headers", "opencl-icd-loader", "gcc>=9.0"],
        # OpenCL does NOT support history-based — must use -m event.
        # Use -G nuclide to avoid exceeding OpenCL max allocation size on
        # the unionized grid (default). nuclide grid is much smaller.
        "correctness_args": ["-m", "event", "-s", "small", "-G", "nuclide"],
        "performance_args": ["-m", "event", "-s", "large", "-G", "nuclide"],
        "corr_description":
            "Small event-based input for correctness verification of xsbench (OpenCL)."
            " Uses nuclide grid to avoid exceeding device max allocation size.",
        "perf_description":
            "Large event-based input for performance measurement of xsbench (OpenCL)."
            " Uses nuclide grid to avoid exceeding device max allocation size.",
    },
    "omp_target": {
        "source_path": "openmp-offload",
        "parallel_api": "omp_target",
        "source_files": [
            "Main.c", "io.c", "Simulation.c",
            "GridInit.c", "XSutils.c", "Materials.c",
        ],
        "header_files": ["XSbench_header.h"],
        "opencl_files": [],
        "language": "C",
        "language_std": "C11",
        "build_cmd": (
            "make COMPILER=nvidia"
            " CC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/compilers/bin/nvc"
            " CFLAGS='-std=gnu99 -Wall -mp=gpu -Minfo=mp -gpu=cc89 -O3'"
        ),
        "clean_cmd": "make clean",
        "build_vars": None,
        "hw_target": "gpu",
        "hw_requirements": {
            "gpu": {
                "vendor": "NVIDIA",
                "min_compute_capability": "sm_60",
                "min_memory_gb": 4,
            }
        },
        "dependencies": ["nvidia-hpc-sdk (nvc with OpenMP target offload)"],
        # omp_target does NOT support history-based — must use -m event
        "correctness_args": ["-m", "event", "-s", "small"],
        "performance_args": ["-m", "event", "-s", "large"],
        "corr_description":
            "Small event-based input for correctness verification of xsbench"
            " (OpenMP target offload)",
        "perf_description":
            "Large event-based input for performance measurement of xsbench"
            " (OpenMP target offload)",
    },
}

# Verification pattern shared by all variants.
# XSBench prints "Verification checksum: <N> (Valid)" on success.
# On failure it prints "(WARNING - INAVALID CHECKSUM!)" (typo in source).
# The regex matches only the valid case.
VERIFICATION_STRATEGIES = [
    {
        "type": "exit_code",
        "expected": 0,
        "description":
            "Process exits with code 0 on valid checksum, 1 on invalid",
    },
    {
        "type": "stdout_pattern",
        "pattern": "Verification checksum: \\d+ \\(Valid\\)",
        "description": "XSBench self-verification reports valid checksum",
    },
]


def make_spec(api_key: str) -> dict:
    """Build a complete spec dict for one XSBench API variant."""
    v = VARIANTS[api_key]

    # prompt_payload: source files + OpenCL device files (all translatable code)
    prompt_payload = list(v["source_files"]) + list(v["opencl_files"])

    # support_files: headers + Makefile (context for LLM, not translation targets)
    support_files = ["Makefile"] + list(v["header_files"])

    spec = {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": "xsbench",
            "parallel_api": v["parallel_api"],
            "unique_id": f"xsbench-xsbench-{v['parallel_api']}",
            "source_suite": "xsbench",
        },
        "provenance": {
            "repository": {
                "url": XSBENCH_REPO_URL,
                "commit": XSBENCH_COMMIT,
                "branch": XSBENCH_BRANCH,
            },
            "repo_root": XSBENCH_REPO_ROOT,
            "source_path": v["source_path"],
            "license": XSBENCH_LICENSE,
        },
        "files": {
            "prompt_payload": prompt_payload,
            "support_files": support_files,
            "verification_only": [],
        },
        "implementation": {
            "api": v["parallel_api"],
            "api_version": None,
            "language": v["language"],
            "language_standard": v["language_std"],
        },
        "build": {
            "environment": {
                "preferred": "system",
                "conda": None,
                "system": {"dependencies": v["dependencies"]},
            },
            "build_system": "make",
            "working_directory": v["source_path"],
            "commands": {
                "configure": None,
                "build": v["build_cmd"],
                "clean": v["clean_cmd"],
            },
            "variables": v["build_vars"],
            "outputs": {"executable": "./XSBench"},
        },
        "run": {
            "executable": "./XSBench",
            "default_arguments": v["correctness_args"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": v["correctness_args"],
                    "description": v["corr_description"],
                    "input_files": [],
                    "expected_results": {
                        "stdout_pattern":
                            "Verification checksum: \\d+ \\(Valid\\)",
                    },
                },
                "performance": {
                    "arguments": v["performance_args"],
                    "description": v["perf_description"],
                    "input_files": [],
                    "expected_results": {
                        "stdout_pattern":
                            "Verification checksum: \\d+ \\(Valid\\)",
                    },
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": VERIFICATION_STRATEGIES,
            "floating_point": None,
        },
        "performance": None,
        "hardware": {
            "target": v["hw_target"],
            "requirements": v["hw_requirements"] if v["hw_requirements"] else {},
            "reference_platform": REFERENCE_PLATFORM,
        },
        "baseline_results": None,
        "metadata": {
            "description":
                "Monte Carlo neutron cross-section lookup (XSBench)",
            "domain": "nuclear physics",
            "complexity": "Monte Carlo cross-section lookup with energy grid search",
            "tags": [
                v["parallel_api"],
                "monte-carlo",
                "nuclear",
                "physics",
                "xsbench",
            ],
            "multi_file": True,
            # translation_complexity is set by standardize_specs.py (not here)
        },
    }
    return spec


def make_manifest_entry(api_key: str) -> dict:
    """Build a manifest.jsonl entry for one XSBench API variant."""
    v = VARIANTS[api_key]
    return {
        "kernel_name": "xsbench",
        "parallel_api": v["parallel_api"],
        "source_suite": "xsbench",
        "spec_file": f"specs/xsbench-xsbench-{v['parallel_api']}.json",
        "source_dir": f"{XSBENCH_REPO_ROOT}/{v['source_path']}",
        "category": "physics",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate XSBench ParBench spec files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing spec files",
    )
    args = parser.parse_args()

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect manifest entries to append at the end
    manifest_entries: list[dict] = []
    written = 0

    for api_key in ["omp", "cuda", "opencl", "omp_target"]:
        spec = make_spec(api_key)
        uid = spec["identity"]["unique_id"]
        spec_path = SPECS_DIR / f"{uid}.json"

        if spec_path.exists() and not args.force:
            print(f"SKIP {spec_path.name} (exists, use --force to overwrite)")
            continue

        spec_json = json.dumps(spec, indent=4, ensure_ascii=False) + "\n"

        if args.dry_run:
            print(f"DRY-RUN: would write {spec_path.name}")
            print(f"  prompt_payload: {spec['files']['prompt_payload']}")
            print(f"  build: {spec['build']['commands']['build']}")
        else:
            spec_path.write_text(spec_json)
            print(f"WROTE {spec_path.name}")

        manifest_entries.append(make_manifest_entry(api_key))
        written += 1

    # Append manifest entries
    if manifest_entries:
        if args.dry_run:
            print(f"\nDRY-RUN: would append {len(manifest_entries)} entries"
                  f" to manifest.jsonl")
            for entry in manifest_entries:
                print(f"  {json.dumps(entry, separators=(',', ':'))}")
        else:
            with open(MANIFEST_FILE, "a") as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            print(f"\nAppended {len(manifest_entries)} entries to manifest.jsonl")

    print(f"\nDone: {written} specs {'would be ' if args.dry_run else ''}written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
