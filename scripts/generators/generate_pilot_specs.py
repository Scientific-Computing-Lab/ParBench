#!/usr/bin/env python3
"""
generate_pilot_specs.py — Generate 20 Level 2 spec files for the HeCBench
pilot kernels (5 kernels × 4 APIs) and write manifest.jsonl entries.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
MANIFEST_PATH = PROJECT_ROOT / "manifest.jsonl"

REPO_ROOT = "../ParBench/downloads/HeCBench_extracted/HeCBench-master"

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

APIS = ["cuda", "hip", "sycl", "omp"]

API_BUILD_DEPS = {
    "cuda": ["CUDA Toolkit >= 11.0", "GCC >= 9.0"],
    "hip": ["ROCm >= 5.0", "hipcc"],
    "sycl": ["oneAPI >= 2023.0", "icpx"],
    "omp": ["GCC >= 9.0"],
}

API_HW_TARGET = {
    "cuda": "gpu",
    "hip": "gpu",
    "sycl": "gpu",
    "omp": "cpu",
}

API_HW_REQS = {
    "cuda": {
        "gpu": {
            "vendor": "NVIDIA",
            "min_compute_capability": "sm_60",
            "min_memory_gb": 2,
        }
    },
    "hip": {"gpu": {"vendor": "AMD", "min_memory_gb": 2}},
    "sycl": {"gpu": {"min_memory_gb": 2}},
    "omp": {"cpu": {"min_cores": 4, "min_memory_gb": 4}},
}


def cu_or_cpp(api):
    return ".cu" if api in ("cuda", "hip") else ".cpp"


def extra_makefiles(api):
    if api == "hip":
        return ["Makefile.hipcl"]
    elif api == "omp":
        return ["Makefile.aomp", "Makefile.nvc"]
    return []


def build_cmd(api):
    if api == "cuda":
        return "make ARCH=sm_89"
    return "make"


def make_spec(kernel_name, api, kernel_cfg):
    ext = cu_or_cpp(api)
    dir_name = f"{kernel_cfg.get('dir_name', kernel_name)}-{api}"
    source_path = f"src/{dir_name}"
    unique_id = f"hecbench-{kernel_name}-{api}"

    # Files
    variant_files = kernel_cfg["variants"][api]
    prompt_payload = variant_files["prompt_payload"]
    support_files_base = ["Makefile"] + extra_makefiles(api) + ["CMakeLists.txt"]
    support_files = support_files_base + variant_files.get("extra_support", [])
    verification_only = variant_files.get("verification_only", [])

    multi_file = len(prompt_payload) > 1

    spec = {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": kernel_name,
            "parallel_api": api,
            "unique_id": unique_id,
            "source_suite": "hecbench",
        },
        "provenance": {
            "repository": {
                "url": "https://github.com/zjin-lcf/HeCBench",
                "commit": "archive-download",
                "branch": "master",
            },
            "repo_root": REPO_ROOT,
            "source_path": source_path,
            "license": "MIT",
        },
        "files": {
            "prompt_payload": prompt_payload,
            "support_files": support_files,
            "verification_only": verification_only,
        },
        "implementation": {
            "api": api,
            "api_version": None,
            "language": "C++",
            "language_standard": "C++17",
        },
        "build": {
            "environment": {
                "preferred": "system",
                "conda": None,
                "system": {"dependencies": API_BUILD_DEPS[api]},
            },
            "build_system": "make",
            "working_directory": source_path,
            "commands": {
                "configure": None,
                "build": build_cmd(api),
                "clean": "make clean",
            },
            "variables": None,
            "outputs": {"executable": "main"},
        },
        "run": {
            "executable": "./main",
            "default_arguments": kernel_cfg["default_arguments"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": kernel_cfg["input_configurations"],
        },
        "verification": kernel_cfg["verification"],
        "performance": kernel_cfg["performance"],
        "hardware": {
            "target": API_HW_TARGET[api],
            "requirements": API_HW_REQS[api],
            "reference_platform": REFERENCE_PLATFORM,
        },
        "baseline_results": None,
        "metadata": {
            "description": kernel_cfg["description"],
            "domain": kernel_cfg["domain"],
            "complexity": kernel_cfg.get("complexity", None),
            "tags": kernel_cfg["tags"],
            "multi_file": multi_file,
        },
    }

    return spec


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

KERNELS = {}

# ---- nn (nearest neighbor) ----
KERNELS["nn"] = {
    "dir_name": "nn",
    "description": "Nearest neighbor search: computes Euclidean distances from a query point (lat,lng) to every record in a geographic database and finds the top-N closest.",
    "domain": "nearest neighbor search",
    "category": "graph",
    "complexity": "O(n)",
    "tags": ["nearest-neighbor", "euclidean-distance", "geographic", "search"],
    "default_arguments": [
        "filelist.txt",
        "-r",
        "5",
        "-lat",
        "30",
        "-lng",
        "90",
        "-i",
        "10",
    ],
    "input_configurations": {
        "correctness": {
            "arguments": [
                "filelist.txt",
                "-r",
                "5",
                "-lat",
                "30",
                "-lng",
                "90",
                "-i",
                "1",
            ],
            "description": "Single-iteration run for quick correctness validation",
            "input_files": ["filelist.txt"],
            "expected_results": None,
        },
        "performance": {
            "arguments": [
                "filelist.txt",
                "-r",
                "5",
                "-lat",
                "30",
                "-lng",
                "90",
                "-i",
                "10000",
                "-t",
            ],
            "description": "10000-iteration run for performance measurement with timing output",
            "input_files": ["filelist.txt"],
            "expected_results": None,
        },
    },
    "verification": {
        "method": "self_checking",
        "strategies": [
            {
                "type": "exit_code",
                "expected": 0,
                "description": "Check for successful execution (no PASS/FAIL output for this kernel)",
            }
        ],
        "floating_point": None,
    },
    "performance": {
        "metrics": [
            {
                "name": "kernel_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Average kernel execution time:\\s+([\\d.]+)\\s+\\(us\\)",
                    "capture_group": 1,
                },
                "unit": "us",
            },
            {
                "name": "offloading_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Device offloading time\\s+([\\d.]+)\\s+\\(s\\)",
                    "capture_group": 1,
                },
                "unit": "s",
            },
        ],
        "warmup_runs": 1,
        "measurement_runs": 5,
    },
    "variants": {
        "cuda": {
            "prompt_payload": ["nearestNeighbor.cu", "nearestNeighbor.h"],
            "extra_support": ["utils.cu", "utils.h", "filelist.txt"],
            "verification_only": [],
        },
        "hip": {
            "prompt_payload": ["nearestNeighbor.cu", "nearestNeighbor.h"],
            "extra_support": ["utils.cu", "utils.h", "filelist.txt"],
            "verification_only": [],
        },
        "sycl": {
            "prompt_payload": ["nearestNeighbor.cpp", "nearestNeighbor.h"],
            "extra_support": ["utils.cpp", "utils.h", "filelist.txt"],
            "verification_only": [],
        },
        "omp": {
            "prompt_payload": ["nearestNeighbor.cpp", "nearestNeighbor.h"],
            "extra_support": ["utils.cpp", "utils.h", "filelist.txt"],
            "verification_only": [],
        },
    },
}

# ---- scan (prefix sum) ----
KERNELS["scan"] = {
    "dir_name": "scan",
    "description": "Work-efficient exclusive prefix sum (scan) using the Blelloch up-sweep/down-sweep algorithm with shared memory, benchmarking both bank-conflict and bank-conflict-avoidance-optimized variants.",
    "domain": "parallel primitives",
    "category": "reduction",
    "complexity": "O(n)",
    "tags": ["prefix-sum", "scan", "blelloch", "shared-memory", "bank-conflicts"],
    "default_arguments": ["268435456", "100"],
    "input_configurations": {
        "correctness": {
            "arguments": ["1048576", "1"],
            "description": "Small array (1M elements), single iteration for correctness check",
            "input_files": [],
            "expected_results": {"stdout_pattern": "PASS"},
        },
        "performance": {
            "arguments": ["268435456", "100"],
            "description": "Large array (256M elements), 100 iterations for performance measurement",
            "input_files": [],
            "expected_results": {"stdout_pattern": "PASS"},
        },
    },
    "verification": {
        "method": "self_checking",
        "strategies": [
            {
                "type": "exit_code",
                "expected": 0,
                "description": "Process exits cleanly",
            },
            {
                "type": "stdout_pattern",
                "pattern": "PASS",
                "description": "CPU-vs-GPU memcmp prints PASS for each type/block-size combination",
            },
        ],
        "floating_point": None,
    },
    "performance": {
        "metrics": [
            {
                "name": "scan_with_bank_conflicts",
                "extraction": {
                    "type": "regex",
                    "pattern": "Average execution time of scan \\(w/\\s+bank conflicts\\):\\s+([\\d.]+)\\s+\\(us\\)",
                    "capture_group": 1,
                },
                "unit": "us",
            },
            {
                "name": "scan_without_bank_conflicts",
                "extraction": {
                    "type": "regex",
                    "pattern": "Average execution time of scan \\(w/o bank conflicts\\):\\s+([\\d.]+)\\s+\\(us\\)",
                    "capture_group": 1,
                },
                "unit": "us",
            },
        ],
        "warmup_runs": 1,
        "measurement_runs": 5,
    },
    "variants": {
        "cuda": {
            "prompt_payload": ["main.cu"],
            "extra_support": [],
            "verification_only": [],
        },
        "hip": {
            "prompt_payload": ["main.cu"],
            "extra_support": [],
            "verification_only": [],
        },
        "sycl": {
            "prompt_payload": ["main.cpp"],
            "extra_support": [],
            "verification_only": [],
        },
        "omp": {
            "prompt_payload": ["main.cpp"],
            "extra_support": [],
            "verification_only": [],
        },
    },
}

# ---- particle-diffusion ----
KERNELS["particle-diffusion"] = {
    "dir_name": "particle-diffusion",
    "description": "Monte Carlo simulation of water molecule diffusion in tissue. Particles undergo random walks on a 2D grid; a per-particle map accumulates cell-hit counts, then reduces to a global grid.",
    "domain": "Monte Carlo simulation",
    "category": "physics",
    "complexity": "O(particles × iterations)",
    "tags": ["monte-carlo", "diffusion", "random-walk", "simulation", "particle"],
    "default_arguments": ["2000", "100"],
    "input_configurations": {
        "correctness": {
            "arguments": ["100", "1"],
            "description": "Short simulation (100 iterations, 1 repeat) for correctness validation",
            "input_files": [],
            "expected_results": {"stdout_pattern": "PASS"},
        },
        "performance": {
            "arguments": ["2000", "100"],
            "description": "Full simulation (2000 iterations, 100 repeats) for performance",
            "input_files": [],
            "expected_results": {"stdout_pattern": "PASS"},
        },
    },
    "verification": {
        "method": "reference_comparison",
        "strategies": [
            {
                "type": "exit_code",
                "expected": 0,
                "description": "Process exits cleanly",
            },
            {
                "type": "stdout_pattern",
                "pattern": "PASS",
                "description": "GPU map compared element-wise to CPU reference (<=2 mismatches tolerated)",
            },
        ],
        "floating_point": {
            "tolerance": 2,
            "tolerance_type": "absolute",
            "note": "Integer map comparison allows up to 2 element-level mismatches due to floating-point non-determinism in random walk",
        },
    },
    "performance": {
        "metrics": [
            {
                "name": "kernel_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(s\\)",
                    "capture_group": 1,
                },
                "unit": "s",
            },
            {
                "name": "simulation_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Simulation time:\\s+([\\d.eE+-]+)\\s+\\(s\\)",
                    "capture_group": 1,
                },
                "unit": "s",
            },
        ],
        "warmup_runs": 1,
        "measurement_runs": 5,
    },
    "variants": {
        "cuda": {
            "prompt_payload": ["motionsim.cu"],
            "extra_support": [],
            "verification_only": ["reference.h"],
        },
        "hip": {
            "prompt_payload": ["motionsim.cu"],
            "extra_support": [],
            "verification_only": ["../particle-diffusion-cuda/reference.h"],
        },
        "sycl": {
            "prompt_payload": ["motionsim.cpp"],
            "extra_support": [],
            "verification_only": ["../particle-diffusion-cuda/reference.h"],
        },
        "omp": {
            "prompt_payload": ["motionsim.cpp"],
            "extra_support": [],
            "verification_only": ["../particle-diffusion-cuda/reference.h"],
        },
    },
}

# ---- binomial (binomial options pricing) ----
KERNELS["binomial"] = {
    "dir_name": "binomial",
    "description": "Prices European call options using the Cox-Ross-Rubinstein binomial tree on GPU. 1024 options over 2048 time steps with shared-memory tree traversal, verified against CPU binomial and Black-Scholes.",
    "domain": "computational finance",
    "category": "financial",
    "complexity": "O(options × timesteps)",
    "tags": [
        "binomial-tree",
        "options-pricing",
        "black-scholes",
        "finance",
        "shared-memory",
    ],
    "default_arguments": [],
    "input_configurations": {
        "correctness": {
            "arguments": [],
            "description": "Default run (no arguments): 1024 options, 1000 iterations, self-verified",
            "input_files": [],
            "expected_results": {"stdout_pattern": "Test passed"},
        }
    },
    "verification": {
        "method": "self_checking",
        "strategies": [
            {
                "type": "exit_code",
                "expected": 0,
                "description": "Exit code 0 on pass, 1 on failure",
            },
            {
                "type": "stdout_pattern",
                "pattern": "Test passed",
                "description": "L1 norm of GPU-vs-CPU binomial tree results must be < 5e-4",
            },
        ],
        "floating_point": {
            "tolerance": 5e-4,
            "tolerance_type": "relative",
            "note": "L1 norm comparison between GPU binomial, CPU binomial, and Black-Scholes closed-form",
        },
    },
    "performance": {
        "metrics": [
            {
                "name": "kernel_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Average kernel execution time\\s*:\\s*([\\d.eE+-]+)\\s*\\(us\\)",
                    "capture_group": 1,
                },
                "unit": "us",
            },
            {
                "name": "total_gpu_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Total binomialOptionsGPU\\(\\) time:\\s*([\\d.eE+-]+)\\s*msec",
                    "capture_group": 1,
                },
                "unit": "ms",
            },
        ],
        "warmup_runs": 1,
        "measurement_runs": 5,
    },
    "variants": {
        "cuda": {
            "prompt_payload": ["kernel.cu", "main.cu"],
            "extra_support": ["binomialOptions.h", "realtype.h"],
            "verification_only": ["reference.cu"],
        },
        "hip": {
            "prompt_payload": ["kernel.cu", "main.cu"],
            "extra_support": ["binomialOptions.h", "realtype.h"],
            "verification_only": ["reference.cu"],
        },
        "sycl": {
            "prompt_payload": ["kernel.cpp", "main.cpp"],
            "extra_support": ["binomialOptions.h", "realtype.h"],
            "verification_only": ["reference.cpp"],
        },
        "omp": {
            "prompt_payload": ["kernel.cpp", "main.cpp"],
            "extra_support": ["binomialOptions.h", "realtype.h"],
            "verification_only": ["reference.cpp"],
        },
    },
}

# ---- radixsort ----
KERNELS["radixsort"] = {
    "dir_name": "radixsort",
    "description": "Parallel radix sort (Satish-Harris-Garland) for unsigned 32-bit integers. 4-bit radix (16 buckets), 8 passes, with block-level sort, offset finding, prefix scan, and global reorder kernels.",
    "domain": "parallel sorting",
    "category": "sort",
    "complexity": "O(n × passes)",
    "tags": ["radix-sort", "sorting", "parallel-primitives", "prefix-scan"],
    "default_arguments": ["1000"],
    "input_configurations": {
        "correctness": {
            "arguments": ["1"],
            "description": "Single iteration for correctness validation (sorts 4M uint32 elements)",
            "input_files": [],
            "expected_results": {"stdout_pattern": "PASS"},
        },
        "performance": {
            "arguments": ["1000"],
            "description": "1000 iterations for performance measurement",
            "input_files": [],
            "expected_results": {"stdout_pattern": "PASS"},
        },
    },
    "verification": {
        "method": "self_checking",
        "strategies": [
            {
                "type": "exit_code",
                "expected": 0,
                "description": "Process exits cleanly",
            },
            {
                "type": "stdout_pattern",
                "pattern": "PASS",
                "description": "Verifies sorted array is monotonically non-decreasing",
            },
        ],
        "floating_point": None,
    },
    "performance": {
        "metrics": [
            {
                "name": "sort_time",
                "extraction": {
                    "type": "regex",
                    "pattern": "Average execution time of radixsort:\\s+([\\d.]+)\\s+\\(s\\)",
                    "capture_group": 1,
                },
                "unit": "s",
            }
        ],
        "warmup_runs": 1,
        "measurement_runs": 5,
    },
    "variants": {
        "cuda": {
            "prompt_payload": [
                "main.cu",
                "RadixSort.cu",
                "RadixSort_kernels.cu",
                "Scan.cu",
                "Scan_kernels.cu",
            ],
            "extra_support": ["RadixSort.h", "Scan.h"],
            "verification_only": [],
        },
        "hip": {
            "prompt_payload": [
                "main.cu",
                "RadixSort.cu",
                "RadixSort_kernels.cu",
                "Scan.cu",
                "Scan_kernels.cu",
            ],
            "extra_support": ["RadixSort.h", "Scan.h"],
            "verification_only": [],
        },
        "sycl": {
            "prompt_payload": [
                "main.cpp",
                "RadixSort.cpp",
                "RadixSort_kernels.cpp",
                "Scan.cpp",
                "Scan_kernels.cpp",
            ],
            "extra_support": ["RadixSort.h", "Scan.h"],
            "verification_only": [],
        },
        "omp": {
            "prompt_payload": [
                "main.cpp",
                "RadixSort.cpp",
                "RadixSort_kernels.cpp",
                "Scan.cpp",
                "Scan_kernels.cpp",
            ],
            "extra_support": ["RadixSort.h", "Scan.h"],
            "verification_only": [],
        },
    },
}


def main():
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lines = []

    for kernel_name, cfg in KERNELS.items():
        for api in APIS:
            spec = make_spec(kernel_name, api, cfg)
            unique_id = spec["identity"]["unique_id"]
            spec_filename = f"{unique_id}.json"
            spec_path = SPECS_DIR / spec_filename

            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
                f.write("\n")
            print(f"  ✓ Created {spec_path.relative_to(PROJECT_ROOT)}")

            # Manifest entry
            dir_name = f"{cfg.get('dir_name', kernel_name)}-{api}"
            manifest_entry = {
                "kernel_name": kernel_name,
                "parallel_api": api,
                "source_suite": "hecbench",
                "category": cfg["category"],
                "spec_file": f"specs/{spec_filename}",
                "source_dir": f"{REPO_ROOT}/src/{dir_name}",
            }
            manifest_lines.append(json.dumps(manifest_entry, separators=(",", ":")))

    # Write manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(line + "\n")
    print(
        f"\n  ✓ Wrote {len(manifest_lines)} entries to {MANIFEST_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
