#!/usr/bin/env python3
"""
Phase 3: Generate 80 JSON spec files (20 kernels × 4 APIs) and update manifest.jsonl.
Batch 2 kernels (from hecbench_batch2_20_selection.csv).
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
MANIFEST_FILE = PROJECT_ROOT / "manifest.jsonl"

# ─── Hardware templates ──────────────────────────────────────────────────────

GPU_HARDWARE = {
    "target": "gpu",
    "requirements": {
        "gpu": {
            "vendor": "NVIDIA",
            "min_compute_capability": "sm_60",
            "min_memory_gb": 2,
        }
    },
    "reference_platform": {
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
    },
}

CPU_HARDWARE = {
    "target": "cpu",
    "requirements": None,
    "reference_platform": {
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
    },
}

# ─── Build config by API ─────────────────────────────────────────────────────

BUILD_BY_API = {
    "cuda": {
        "build_cmd": "make ARCH=sm_89",
        "clean_cmd": "make clean",
        "deps": ["CUDA Toolkit >= 11.0", "GCC >= 9.0"],
    },
    "hip": {
        "build_cmd": "make ARCH=sm_89",
        "clean_cmd": "make clean",
        "deps": ["ROCm HIP >= 5.0", "GCC >= 9.0"],
    },
    "sycl": {
        "build_cmd": "make",
        "clean_cmd": "make clean",
        "deps": ["Intel oneAPI DPC++/C++ Compiler", "GCC >= 9.0"],
    },
    "omp": {
        "build_cmd": "make",
        "clean_cmd": "make clean",
        "deps": ["GCC >= 9.0 with OpenMP support"],
    },
}

# ─── Kernel definitions ──────────────────────────────────────────────────────
# Each kernel specifies per-API file classifications and shared run/verify info.
# Batch 2: 20 kernels from hecbench_batch2_20_selection.csv

KERNELS = [
    # ── 1. floydwarshall ─────────────────────────────────────────────────────
    {
        "kernel_name": "floydwarshall",
        "category": "graph",
        "description": "Floyd-Warshall all-pairs shortest-paths algorithm on GPU. Iteratively relaxes distance matrix entries in parallel, verified against CPU reference via memcmp.",
        "domain": "graph algorithms",
        "complexity": "O(V^3)",
        "tags": [
            "floyd-warshall",
            "shortest-path",
            "all-pairs",
            "graph",
            "dynamic-programming",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1024", "100", "16"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1024", "1", "16"],
                    "description": "Single iteration for correctness; memcmp of GPU vs CPU shortest-path matrix",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["1024", "100", "16"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "memcmp of GPU distance matrix vs CPU Floyd-Warshall reference",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 2. nw ────────────────────────────────────────────────────────────────
    {
        "kernel_name": "nw",
        "category": "other",
        "description": "Needleman-Wunsch global sequence alignment algorithm on GPU. Fills scoring matrix in anti-diagonal wavefront fashion, verified via memcmp against CPU reference implementation.",
        "domain": "bioinformatics",
        "complexity": "O(n^2)",
        "tags": [
            "needleman-wunsch",
            "sequence-alignment",
            "bioinformatics",
            "dynamic-programming",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["nw.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["nw.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["nw.cpp"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "kernel1.sycl",
                    "kernel2.sycl",
                ],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["nw.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["16384", "10"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["16384", "1"],
                    "description": "Single-iteration run; memcmp of GPU alignment matrix vs CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["16384", "10"],
                    "description": "10 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "memcmp of GPU alignment result vs CPU Needleman-Wunsch reference",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Total kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(s\\)",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 3. myocyte ───────────────────────────────────────────────────────────
    {
        "kernel_name": "myocyte",
        "category": "other",
        "description": "Cardiac myocyte ODE simulation using embedded Fehlberg 7-8th order Runge-Kutta solver. Multiple GPU kernels model calcium, electrical conduction, and force generation; verified against output file.",
        "domain": "biomedical simulation",
        "complexity": "O(n × steps)",
        "tags": ["myocyte", "ode-solver", "runge-kutta", "cardiac", "biomedical"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": [
                    "kernel.cu",
                    "kernel_cam.cu",
                    "kernel_ecc.cu",
                    "main.cu",
                ],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "define.h",
                    "embedded_fehlberg_7_8.cu",
                    "file.c",
                    "kernel_fin.cu",
                    "master.cu",
                    "solver.cu",
                    "work.cu",
                ],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": [
                    "kernel.cu",
                    "kernel_cam.cu",
                    "kernel_ecc.cu",
                    "main.cu",
                ],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "define.h",
                    "embedded_fehlberg_7_8.cu",
                    "file.c",
                    "kernel_fin.cu",
                    "master.cu",
                    "solver.cu",
                    "work.cu",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": [
                    "kernel.cpp",
                    "kernel_cam.cpp",
                    "kernel_ecc.cpp",
                    "main.cpp",
                ],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "define.h",
                    "embedded_fehlberg_7_8.cpp",
                    "file.c",
                    "kernel_fin.cpp",
                    "master.cpp",
                    "solver.cpp",
                    "work.cpp",
                ],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.c"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "common.h",
                    "kernel/",
                    "util/",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["10"],
                    "description": "10 time-steps for correctness; writes output.txt and verifies solver convergence",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["100"],
                    "description": "100 time-steps for performance measurement",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Exit code 0 indicates successful solver convergence",
                },
                {
                    "type": "stdout_pattern",
                    "pattern": "Total kernel execution time",
                    "description": "Presence of timing output indicates successful ODE solver completion",
                },
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "ODE solver uses adaptive step-size control with internal tolerance checks",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Total kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 4. geglu ─────────────────────────────────────────────────────────────
    {
        "kernel_name": "geglu",
        "category": "ml",
        "description": "GELU-Gated Linear Unit (GeGLU) activation function on GPU. Applies element-wise GeGLU activation, verified against CPU reference implementation via tolerance comparison.",
        "domain": "machine learning",
        "complexity": "O(n)",
        "tags": [
            "geglu",
            "activation-function",
            "gelu",
            "machine-learning",
            "neural-network",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1"],
                    "description": "Single iteration for correctness; element-wise comparison of GPU vs CPU GeGLU",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU GeGLU vs CPU reference",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference of GeGLU activation outputs",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average execution time of GeGLU kernel:\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 5. perplexity ────────────────────────────────────────────────────────
    {
        "kernel_name": "perplexity",
        "category": "ml",
        "description": "Perplexity computation for language model evaluation on GPU. Computes log-likelihood-based perplexity scores in parallel, verified against CPU reference implementation.",
        "domain": "machine learning / NLP",
        "complexity": "O(n × d)",
        "tags": [
            "perplexity",
            "language-model",
            "nlp",
            "machine-learning",
            "evaluation",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.cpp"],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["10000", "50", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["10000", "50", "1"],
                    "description": "Single iteration for correctness; verifies perplexity scores against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["10000", "50", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Comparison of GPU perplexity scores vs CPU reference",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "relative",
                "note": "Relative comparison of perplexity scores",
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
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 6. knn ───────────────────────────────────────────────────────────────
    {
        "kernel_name": "knn",
        "category": "ml",
        "description": "k-Nearest Neighbours classification on GPU. Computes pairwise distances, selects k nearest neighbours, and performs majority vote; verified via precision accuracy comparison.",
        "domain": "machine learning",
        "complexity": "O(n × m × k)",
        "tags": [
            "knn",
            "k-nearest-neighbours",
            "classification",
            "machine-learning",
            "pairwise-distance",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1"],
                    "description": "Single iteration for correctness; verifies precision accuracy == 1.0",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Precision accuracy == 1.0 comparing GPU vs CPU kNN classification",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "total_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "done in\\s+([\\d.eE+-]+)\\s+s for",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 7. iso2dfd ───────────────────────────────────────────────────────────
    {
        "kernel_name": "iso2dfd",
        "category": "stencil",
        "description": "2D isotropic finite-difference wave propagation stencil on GPU. Applies 2nd-order time-stepping to a 2D grid, verified against CPU reference via Euclidean norm comparison.",
        "domain": "seismic / stencil computation",
        "complexity": "O(Lx × Ly × niter)",
        "tags": [
            "iso2dfd",
            "finite-difference",
            "wave-propagation",
            "stencil",
            "seismic",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["iso2dfd.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "iso2dfd.h"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["iso2dfd.cu"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "iso2dfd.h",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["iso2dfd.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "iso2dfd.h"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["iso2dfd.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "iso2dfd.h",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["2048", "2048", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["2048", "2048", "10"],
                    "description": "10 time-steps for correctness; Euclidean norm comparison of GPU vs CPU wavefield",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["2048", "2048", "1000"],
                    "description": "1000 time-steps for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Euclidean norm of GPU vs CPU wavefield below threshold",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "absolute",
                "note": "Euclidean norm comparison between GPU and CPU wavefield snapshots",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 8. heat2d ────────────────────────────────────────────────────────────
    {
        "kernel_name": "heat2d",
        "category": "stencil",
        "description": "2D heat equation solver using explicit finite-difference stencil on GPU. Iteratively updates temperature grid with Jacobi-style 5-point stencil; verified against CPU reference.",
        "domain": "heat conduction / stencil computation",
        "complexity": "O(Lx × Ly × niter)",
        "tags": ["heat-equation", "stencil", "finite-difference", "jacobi", "2d-grid"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["4096", "4096", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["4096", "4096", "10"],
                    "description": "10 iterations for correctness; element-wise comparison of GPU vs CPU heat grid",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["4096", "4096", "1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU vs CPU temperature grid",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference of temperature values",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "throughput",
                    "extraction": {
                        "type": "regex",
                        "pattern": "BW =\\s+([\\d.eE+-]+)\\s+GB/s",
                        "capture_group": 1,
                    },
                    "unit": "GB/s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 9. stencil1d ─────────────────────────────────────────────────────────
    {
        "kernel_name": "stencil1d",
        "category": "stencil",
        "description": "1D stencil computation on GPU using shared memory with halo regions. Applies a 7-point stencil kernel, verified against CPU reference via element-wise comparison.",
        "domain": "stencil computation",
        "complexity": "O(n × iterations)",
        "tags": ["stencil", "1d", "shared-memory", "halo", "finite-difference"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["stencil_1d.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["stencil_1d.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["stencil_1d.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["stencil_1d.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["134217728", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["134217728", "1"],
                    "description": "Single iteration for correctness; element-wise comparison of GPU vs CPU stencil output",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["134217728", "1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU vs CPU 1D stencil results",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference comparison",
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
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 10. murmurhash3 ──────────────────────────────────────────────────────
    {
        "kernel_name": "murmurhash3",
        "category": "other",
        "description": "MurmurHash3 non-cryptographic hash function on GPU. Hashes large arrays in parallel, verified via memcmp against CPU MurmurHash3 reference implementation.",
        "domain": "hashing",
        "complexity": "O(n)",
        "tags": ["murmurhash3", "hashing", "non-cryptographic", "hash-function"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["murmurhash3.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["murmurhash3.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["murmurhash3.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["murmurhash3.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100000", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["100000", "1"],
                    "description": "Single iteration for correctness; memcmp of GPU vs CPU MurmurHash3 results",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["100000", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "FAIL",
                    "description": "Absence of 'FAIL' indicates memcmp passed; program prints FAIL only on mismatch",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 11. crc64 ────────────────────────────────────────────────────────────
    {
        "kernel_name": "crc64",
        "category": "other",
        "description": "CRC-64 cyclic redundancy check computation on GPU. Processes data blocks in parallel using lookup tables, verified against CPU CRC-64 reference via PASS/FAIL comparison.",
        "domain": "error detection / hashing",
        "complexity": "O(n)",
        "tags": ["crc64", "cyclic-redundancy-check", "error-detection", "checksum"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["CRC64.cu", "CRC64Test.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "CRC64.h"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["CRC64.cu", "CRC64Test.cu"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "CRC64.h",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["CRC64.cpp", "CRC64Test.cpp"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "CRC64.h",
                    "crc64_table.h",
                ],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["CRC64.cpp", "CRC64Test.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "CRC64.h",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["10", "5", "33554432"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1", "1", "33554432"],
                    "description": "Single iteration for correctness; verifies CRC-64 checksums match CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["10", "5", "33554432"],
                    "description": "10 iterations, 5 passes for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "CRC-64 checksum matches CPU reference",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "throughput",
                    "extraction": {
                        "type": "regex",
                        "pattern": "([\\d.eE+-]+)\\s+MB/s",
                        "capture_group": 1,
                    },
                    "unit": "MB/s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 12. jenkins-hash ─────────────────────────────────────────────────────
    {
        "kernel_name": "jenkins-hash",
        "category": "other",
        "description": "Jenkins one-at-a-time hash function on GPU. Hashes arrays of keys in parallel, verified against CPU Jenkins hash reference via element-wise comparison.",
        "domain": "hashing",
        "complexity": "O(n × key_length)",
        "tags": ["jenkins-hash", "hash-function", "non-cryptographic", "hashing"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["256", "16777216", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["256", "16777216", "1"],
                    "description": "Single iteration for correctness; compares GPU hash results vs CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["256", "16777216", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU vs CPU Jenkins hash values",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time\\s*:\\s*([\\d.eE+-]+)\\s+\\(s\\)",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 13. gaussian ─────────────────────────────────────────────────────────
    {
        "kernel_name": "gaussian",
        "category": "linear_algebra",
        "description": "Gaussian elimination with partial pivoting on GPU. Forward elimination and back substitution kernels solve a dense linear system, verified against CPU reference via element-wise comparison.",
        "domain": "dense linear algebra",
        "complexity": "O(n^3)",
        "tags": [
            "gaussian-elimination",
            "linear-algebra",
            "pivoting",
            "dense-matrix",
            "solver",
        ],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["gaussianElim.cu"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "gaussianElim.h",
                    "utils.cu",
                    "utils.h",
                ],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["gaussianElim.cu"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "gaussianElim.h",
                    "utils.cu",
                    "utils.h",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["gaussianElim.cpp"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "gaussianElim.h",
                    "utils.cpp",
                    "utils.h",
                ],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["gaussianElim.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "gaussianElim.h",
                    "utils.cpp",
                    "utils.h",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["-q", "-t", "-s", "4096"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["-q", "-t", "-s", "256"],
                    "description": "Small matrix (256) with timing; prints PASS/FAIL based on element-wise comparison",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["-q", "-t", "-s", "4096"],
                    "description": "Large matrix (4096) for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU vs CPU Gaussian elimination results",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference comparison between GPU and CPU solver results",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Total kernel execution time\\s+([\\d]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 14. triad ────────────────────────────────────────────────────────────
    {
        "kernel_name": "triad",
        "category": "other",
        "description": "STREAM Triad memory bandwidth benchmark on GPU. Performs a[i] = b[i] + scalar*c[i] to measure sustained memory throughput, verified via element-wise comparison with CPU reference.",
        "domain": "memory bandwidth",
        "complexity": "O(n)",
        "tags": ["triad", "stream", "memory-bandwidth", "benchmark", "throughput"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cpp", "triad.cu"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "LICENSE",
                    "Option.cpp",
                    "Option.h",
                    "OptionParser.cpp",
                    "OptionParser.h",
                    "Timer.cpp",
                    "Timer.h",
                    "Utility.h",
                    "config.h",
                ],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cpp", "triad.cu"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "LICENSE",
                    "Option.cpp",
                    "Option.h",
                    "OptionParser.cpp",
                    "OptionParser.h",
                    "Timer.cpp",
                    "Timer.h",
                    "Utility.h",
                    "config.h",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "triad.cpp", "triad2.cpp"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "LICENSE",
                    "Option.cpp",
                    "Option.h",
                    "OptionParser.cpp",
                    "OptionParser.h",
                    "Timer.cpp",
                    "Timer.h",
                    "Utility.h",
                    "config.h",
                ],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "triad.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "LICENSE",
                    "Option.cpp",
                    "Option.h",
                    "OptionParser.cpp",
                    "OptionParser.h",
                    "Timer.cpp",
                    "Timer.h",
                    "Utility.h",
                    "config.h",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["--passes", "100", "-v"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["--passes", "1", "-v"],
                    "description": "Single pass for correctness; verifies element-wise triad results",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["--passes", "100", "-v"],
                    "description": "100 passes for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU triad result vs CPU reference",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "Element-wise comparison of floating-point triad results",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "triad_gflops",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average TriadFlops\\s+([\\d.eE+-]+)\\s+GFLOPS/s",
                        "capture_group": 1,
                    },
                    "unit": "GFLOPS/s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 15. popcount ─────────────────────────────────────────────────────────
    {
        "kernel_name": "popcount",
        "category": "other",
        "description": "Population count (Hamming weight) computation on GPU using multiple algorithm variants. Counts set bits in large integer arrays, verified by comparing 5 GPU implementations against each other.",
        "domain": "bit manipulation",
        "complexity": "O(n)",
        "tags": ["popcount", "hamming-weight", "bit-manipulation", "population-count"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["16777216", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["16777216", "1"],
                    "description": "Single iteration for correctness; verifies 5 popcount variants agree",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["16777216", "1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Exit code 0 indicates all 5 popcount variants produced matching results",
                }
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time_pc1",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time \\(pc1\\):\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                },
                {
                    "name": "kernel_time_pc2",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time \\(pc2\\):\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                },
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 16. sobol ────────────────────────────────────────────────────────────
    {
        "kernel_name": "sobol",
        "category": "financial",
        "description": "Sobol quasi-random number sequence generator on GPU. Generates low-discrepancy sequences using direction vectors for Monte Carlo integration, verified against CPU Sobol implementation.",
        "domain": "quasi-random number generation",
        "complexity": "O(n × dimensions)",
        "tags": ["sobol", "quasi-random", "low-discrepancy", "monte-carlo", "qrng"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["sobol.cu", "sobol_gpu.cu"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "sobol.h",
                    "sobol_gpu.h",
                    "sobol_primitives.cu",
                    "sobol_primitives.h",
                ],
                "verification_only": ["sobol_gold.cu", "sobol_gold.h"],
            },
            "hip": {
                "prompt_payload": ["sobol.cu", "sobol_gpu.cu"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "sobol.h",
                    "sobol_gpu.h",
                    "sobol_primitives.cu",
                    "sobol_primitives.h",
                ],
                "verification_only": ["sobol_gold.cu", "sobol_gold.h"],
            },
            "sycl": {
                "prompt_payload": ["sobol.cpp", "sobol_gpu.cpp"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "sobol.h",
                    "sobol_gpu.h",
                    "sobol_primitives.cpp",
                    "sobol_primitives.h",
                ],
                "verification_only": ["sobol_gold.cpp", "sobol_gold.h"],
            },
            "omp": {
                "prompt_payload": ["sobol.cpp", "sobol_gpu.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "sobol.h",
                    "sobol_gpu.h",
                    "sobol_primitives.cpp",
                    "sobol_primitives.h",
                ],
                "verification_only": ["sobol_gold.cpp", "sobol_gold.h"],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1000000", "1000", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1000000", "1000", "1"],
                    "description": "Single iteration for correctness; compares GPU Sobol output vs CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["1000000", "1000", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Comparison of GPU Sobol sequences vs CPU gold reference implementation",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
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
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 17. convolution3D ────────────────────────────────────────────────────
    {
        "kernel_name": "convolution3D",
        "category": "other",
        "description": "3D convolution using multiple algorithmic strategies (direct, shared memory, register tiling) on GPU. Processes 3D feature maps with learned filters, verified against CPU reference.",
        "domain": "3D signal processing / deep learning",
        "complexity": "O(N × C × H × W × K^3)",
        "tags": [
            "convolution3d",
            "3d-convolution",
            "deep-learning",
            "feature-map",
            "tiling",
        ],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "conv3d_s4.cu"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "conv3d_s4.cu"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "conv3d_s4.cpp",
                    "onednn_utils.hpp",
                ],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["32", "1", "6", "32", "32", "5", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["32", "1", "6", "32", "32", "5", "1"],
                    "description": "Single iteration for correctness; element-wise comparison of GPU vs CPU 3D convolution",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["32", "1", "6", "32", "32", "5", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU vs CPU 3D convolution results",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference comparison between GPU and CPU convolution output",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time_s1",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time of conv3d_s1 kernel:\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                },
                {
                    "name": "kernel_time_s2",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time of conv3d_s2 kernel:\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                },
                {
                    "name": "kernel_time_s3",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time of conv3d_s3 kernel:\\s+([\\d.eE+-]+)\\s+\\(us\\)",
                        "capture_group": 1,
                    },
                    "unit": "us",
                },
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 18. mandelbrot ───────────────────────────────────────────────────────
    {
        "kernel_name": "mandelbrot",
        "category": "other",
        "description": "Mandelbrot set fractal computation on GPU. Evaluates escape-time iteration counts for each pixel in the complex plane, verified against CPU serial implementation via matrix comparison.",
        "domain": "fractal / computational mathematics",
        "complexity": "O(width × height × max_iter)",
        "tags": [
            "mandelbrot",
            "fractal",
            "escape-time",
            "complex-plane",
            "parallel-pixels",
        ],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "mandel.hpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "common.hpp"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu", "mandel.hpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "common.hpp",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "mandel.hpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "util.hpp"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "mandel.hpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "util.hpp",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1"],
                    "description": "Single iteration for correctness; matrix comparison of GPU vs CPU Mandelbrot output",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "Pass verification"},
                },
                "performance": {
                    "arguments": ["1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "Pass verification"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "Pass verification",
                    "description": "Matrix comparison of GPU vs CPU Mandelbrot iteration counts",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)",
                        "capture_group": 1,
                    },
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 19. mis ──────────────────────────────────────────────────────────────
    {
        "kernel_name": "mis",
        "category": "graph",
        "description": "Maximal Independent Set computation on GPU using Luby's parallel algorithm. Iteratively selects vertices into independent set, verified via graph structure validation on stderr.",
        "domain": "graph algorithms",
        "complexity": "O(V + E)",
        "tags": ["mis", "maximal-independent-set", "luby", "graph", "combinatorial"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": [
                    "Makefile",
                    "CMakeLists.txt",
                    "graph.h",
                    "internet.egr",
                ],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["internet.egr", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["internet.egr", "1"],
                    "description": "Single iteration for correctness; validates MIS properties on graph",
                    "input_files": ["internet.egr"],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["internet.egr", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": ["internet.egr"],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Exit code 0 indicates valid MIS computed without errors",
                },
                {
                    "type": "stdout_pattern",
                    "pattern": "compute time",
                    "description": "Presence of timing output indicates successful MIS computation",
                },
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "compute_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "compute time:\\s+([\\d.eE+-]+)\\s+s",
                        "capture_group": 1,
                    },
                    "unit": "s",
                },
                {
                    "name": "node_throughput",
                    "extraction": {
                        "type": "regex",
                        "pattern": "throughput:\\s+([\\d.eE+-]+)\\s+Mnodes/s",
                        "capture_group": 1,
                    },
                    "unit": "Mnodes/s",
                },
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
    # ── 20. bezier-surface ───────────────────────────────────────────────────
    {
        "kernel_name": "bezier-surface",
        "category": "other",
        "description": "Bezier surface evaluation on GPU. Computes surface points from control points using tensor-product Bernstein polynomials, verified against CPU reference via element-wise comparison.",
        "domain": "computational geometry",
        "complexity": "O(n^2 × p^2)",
        "tags": [
            "bezier-surface",
            "bernstein",
            "computational-geometry",
            "tensor-product",
            "surface-evaluation",
        ],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "input"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": [
                    "Makefile",
                    "Makefile.hipcl",
                    "CMakeLists.txt",
                    "input",
                ],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "input"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": [
                    "Makefile",
                    "Makefile.aomp",
                    "Makefile.nvc",
                    "CMakeLists.txt",
                    "input",
                ],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["-n", "8192"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["-n", "256"],
                    "description": "Small surface (256) for correctness; element-wise comparison of GPU vs CPU Bezier output",
                    "input_files": ["input/control.txt"],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["-n", "8192"],
                    "description": "Large surface (8192) for performance measurement",
                    "input_files": ["input/control.txt"],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {
                    "type": "stdout_pattern",
                    "pattern": "PASS",
                    "description": "Element-wise comparison of GPU vs CPU Bezier surface points",
                },
                {
                    "type": "exit_code",
                    "expected": 0,
                    "description": "Process exits cleanly",
                },
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference of surface point coordinates",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {
                        "type": "regex",
                        "pattern": "kernel execution time:\\s+([\\d.eE+-]+)\\s+ms",
                        "capture_group": 1,
                    },
                    "unit": "ms",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
]


def make_spec(kernel_data: dict, api: str) -> dict:
    """Generate a single spec JSON for a kernel+API combination."""
    k = kernel_data["kernel_name"]
    files = kernel_data["files"][api]
    build_cfg = BUILD_BY_API[api]
    hw = CPU_HARDWARE if api == "omp" else GPU_HARDWARE

    spec = {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": k,
            "parallel_api": api,
            "unique_id": f"hecbench-{k}-{api}",
            "source_suite": "hecbench",
        },
        "provenance": {
            "repository": {
                "url": "https://github.com/zjin-lcf/HeCBench",
                "commit": "archive-download",
                "branch": "master",
            },
            "repo_root": "HeCBench-master/",
            "source_path": f"src/{k}-{api}",
            "license": "MIT",
        },
        "files": {
            "prompt_payload": files["prompt_payload"],
            "support_files": files.get("support_files", []),
            "verification_only": files.get("verification_only", []),
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
                "system": {"dependencies": build_cfg["deps"]},
            },
            "build_system": "make",
            "working_directory": f"src/{k}-{api}",
            "commands": {
                "configure": None,
                "build": build_cfg["build_cmd"],
                "clean": build_cfg["clean_cmd"],
            },
            "variables": None,
            "outputs": {"executable": "main"},
        },
        "run": kernel_data["run"],
        "verification": kernel_data["verification"],
        "performance": kernel_data["performance"],
        "hardware": hw,
        "baseline_results": None,
        "metadata": {
            "description": kernel_data["description"],
            "domain": kernel_data["domain"],
            "complexity": kernel_data["complexity"],
            "tags": kernel_data["tags"],
            "multi_file": kernel_data["multi_file"],
        },
    }
    return spec


def make_manifest_entry(kernel_data: dict, api: str) -> dict:
    """Generate a manifest.jsonl entry."""
    k = kernel_data["kernel_name"]
    return {
        "kernel_name": k,
        "parallel_api": api,
        "source_suite": "hecbench",
        "category": kernel_data["category"],
        "spec_file": f"specs/hecbench-{k}-{api}.json",
        "source_dir": f"HeCBench-master/src/{k}-{api}",
    }


def main():
    apis = ["cuda", "hip", "sycl", "omp"]
    created = 0
    manifest_entries = []

    for kernel_data in KERNELS:
        k = kernel_data["kernel_name"]
        for api in apis:
            spec = make_spec(kernel_data, api)
            spec_file = SPECS_DIR / f"hecbench-{k}-{api}.json"

            if spec_file.exists():
                print(f"SKIP (exists): {spec_file.name}")
                continue

            with open(spec_file, "w") as f:
                json.dump(spec, f, indent=2)
                f.write("\n")
            print(f"CREATED: {spec_file.name}")
            created += 1

            manifest_entries.append(make_manifest_entry(kernel_data, api))

    # Append to manifest.jsonl
    if manifest_entries:
        with open(MANIFEST_FILE, "a") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        print(f"\nAppended {len(manifest_entries)} entries to manifest.jsonl")

    print(f"\nTotal specs created: {created}")
    print(f"Total manifest entries added: {len(manifest_entries)}")


if __name__ == "__main__":
    main()
