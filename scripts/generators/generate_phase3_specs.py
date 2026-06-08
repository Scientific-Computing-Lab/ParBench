#!/usr/bin/env python3
"""
Phase 3: Generate 80 JSON spec files (20 kernels × 4 APIs) and update manifest.jsonl.
Batch 3 kernels:
  pathfinder, deredundancy, softmax-online, backprop, rmsnorm, laplace3d,
  tissue, lulesh, thomas, keccaktreehash, md5hash, ccsd-trpdrv,
  babelstream, fpc, feynman-kac, maxpool3d, secp256k1, tsp, pso, ga
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
# Batch 3: 20 kernels selected in phase 1

KERNELS = [
    # ── 1. pathfinder ─────────────────────────────────────────────────────────
    {
        "kernel_name": "pathfinder",
        "category": "graph",
        "description": "Dynamic-programming pathfinder on a 2D grid. Finds minimum cost path from top row to bottom row using parallel wavefront propagation.",
        "domain": "dynamic programming",
        "complexity": "O(rows × cols)",
        "tags": ["pathfinder", "dynamic-programming", "wavefront", "grid", "shortest-path"],
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
                "prompt_payload": ["main.cpp", "kernel.sycl"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100000", "1000", "5"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["100", "10", "5"],
                    "description": "Small grid (100×10) for quick correctness run",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["100000", "1000", "5"],
                    "description": "Large grid (100000×1000) for performance measurement",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly with code 0"},
                {"type": "stdout_pattern", "pattern": "Total kernel execution time", "description": "Timing output confirms successful computation"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Total kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 2. deredundancy ───────────────────────────────────────────────────────
    {
        "kernel_name": "deredundancy",
        "category": "other",
        "description": "GPU-accelerated sequence dereplication (clustering) for bioinformatics. Reads FASTA-format DNA/protein sequences, clusters them by similarity threshold using GPU-parallelized pairwise comparisons.",
        "domain": "bioinformatics",
        "complexity": "O(n^2 × L)",
        "tags": ["bioinformatics", "clustering", "sequence-alignment", "fasta", "dereplication"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "kernels.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "utils.h", "LICENSE"],
                "verification_only": ["utils.cu"],
            },
            "hip": {
                "prompt_payload": ["main.cu", "kernels.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "utils.h", "LICENSE"],
                "verification_only": ["utils.cu"],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "kernels.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "utils.h", "LICENSE"],
                "verification_only": ["utils.cpp"],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "kernels.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "utils.h", "LICENSE"],
                "verification_only": ["utils.cpp"],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["i", "../deredundancy-sycl/testData.fasta"],
            "timeout_seconds": 600,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["i", "../deredundancy-sycl/testData.fasta"],
                    "description": "Cluster FASTA sequences at default 0.95 threshold; verify cluster count output",
                    "input_files": ["../deredundancy-sycl/testData.fasta"],
                    "expected_results": {"stdout_pattern": "cluster count:\\s+\\d+"},
                },
                "performance": {
                    "arguments": ["i", "../deredundancy-sycl/testData.fasta"],
                    "description": "Full clustering run for timing measurement",
                    "input_files": ["../deredundancy-sycl/testData.fasta"],
                    "expected_results": {"stdout_pattern": "cluster count:\\s+\\d+"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "cluster count:\\s+\\d+", "description": "Program outputs cluster count on successful completion"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "offload_time",
                    "extraction": {"type": "regex", "pattern": "Device offload time\\s+([\\d.eE+-]+)\\s+secs", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 0,
            "measurement_runs": 3,
        },
    },

    # ── 3. softmax-online ─────────────────────────────────────────────────────
    {
        "kernel_name": "softmax-online",
        "category": "ml",
        "description": "Online numerically-stable softmax forward pass optimized for GPU. Uses online normalization trick to compute softmax in a single pass, avoiding the two-pass max-subtract-exp-sum approach.",
        "domain": "machine learning",
        "complexity": "O(n)",
        "tags": ["softmax", "online-softmax", "attention", "ml", "transformer", "warp-reduction"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "common.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu", "common.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "common.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "common.h"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "LICENSE"],
                "verification_only": ["reference.h"],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1"],
                    "description": "Run online softmax kernel variant 1; validates GPU results against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "All results match"},
                },
                "performance": {
                    "arguments": ["1"],
                    "description": "Online softmax with performance benchmarks at multiple block sizes",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "All results match"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "All results match", "description": "validate_result() compares GPU output against CPU softmax; prints 'All results match' on success"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly; non-zero on mismatch"},
            ],
            "floating_point": {
                "tolerance": 1e-4,
                "tolerance_type": "absolute",
                "note": "validate_result uses element-wise tolerance check with epsilon scaling",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "block_size\\s+\\d+\\s+\\|\\s+time\\s+([\\d.eE+-]+)\\s+ms", "capture_group": 1},
                    "unit": "ms",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 4. backprop ───────────────────────────────────────────────────────────
    {
        "kernel_name": "backprop",
        "category": "ml",
        "description": "Backpropagation training for a simple feed-forward neural network. GPU kernels accelerate the layer-forward pass and weight-adjustment step. Verification compares GPU-trained weights against CPU reference.",
        "domain": "machine learning",
        "complexity": "O(n × hidden)",
        "tags": ["backpropagation", "neural-network", "ml", "training", "shared-memory"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "backprop.cu", "facetrain.cu", "imagenet.cu", "bpnn_layerforward.h", "bpnn_adjust_weights.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "backprop.h"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu", "backprop.cu", "facetrain.cu", "imagenet.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "backprop.cpp", "facetrain.cpp", "imagenet.cpp", "bpnn_layerforward.sycl", "bpnn_adjust_weights.sycl"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "backprop.cpp", "facetrain.cpp", "imagenet.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["65536"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["65536"],
                    "description": "Train neural network with 65536 input nodes; verify GPU weights against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["65536"],
                    "description": "Training run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Compares GPU-trained weights against CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Weight comparison between GPU and CPU backpropagation results",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Device offloading time\\s*=\\s*([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 5. rmsnorm ────────────────────────────────────────────────────────────
    {
        "kernel_name": "rmsnorm",
        "category": "ml",
        "description": "Root Mean Square Layer Normalization (RMSNorm) for transformer models. GPU-accelerated forward pass with multiple block-size variants using warp-level reductions. Used in LLaMA and other modern transformers.",
        "domain": "machine learning",
        "complexity": "O(n)",
        "tags": ["rmsnorm", "layer-normalization", "transformer", "llama", "ml", "warp-reduction"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "common.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "reduce.cuh", "utils.cuh"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu", "common.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "reduce.cuh", "utils.cuh"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "common.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "reduce.h", "utils.h"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "common.h"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1", "12288", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1", "12288", "1"],
                    "description": "Single iteration RMSNorm with hidden_dim=12288; validates against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "All results match"},
                },
                "performance": {
                    "arguments": ["1", "12288", "1000"],
                    "description": "1000-iteration RMSNorm for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "All results match"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "All results match", "description": "validate_result() compares GPU output vs CPU RMSNorm; prints 'All results match' or exits on mismatch"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly; non-zero on mismatch"},
            ],
            "floating_point": {
                "tolerance": 1e-5,
                "tolerance_type": "absolute",
                "note": "Element-wise comparison with epsilon-scaled tolerance",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "block_size\\s+\\d+\\s+\\|\\s+time\\s+([\\d.eE+-]+)\\s+ms", "capture_group": 1},
                    "unit": "ms",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 6. laplace3d ──────────────────────────────────────────────────────────
    {
        "kernel_name": "laplace3d",
        "category": "stencil",
        "description": "6-point stencil 3D Laplace equation solver on a regular grid. Iteratively applies the Laplace operator (average of 6 neighbors) with Dirichlet boundary conditions.",
        "domain": "PDE solving",
        "complexity": "O(NX × NY × NZ × iterations)",
        "tags": ["laplace", "stencil", "3d", "pde", "iterative-solver", "dirichlet"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "kernel.h"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "kernel.h"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "kernel.h"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["128", "128", "128", "100", "1"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["128", "128", "128", "100", "1"],
                    "description": "128^3 grid, 100 iterations with verification enabled (last arg=1)",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["512", "512", "512", "100", "0"],
                    "description": "512^3 grid, 100 iterations, verification disabled for timing",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "RMS error comparison of GPU stencil output vs CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "RMS error between GPU and CPU Laplace solutions must be near zero",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 7. tissue ─────────────────────────────────────────────────────────────
    {
        "kernel_name": "tissue",
        "category": "physics",
        "description": "GPU-accelerated cardiac tissue electrophysiology simulation using the ten Tusscher-Panfilov model. Computes ionic current through multiple channels and updates membrane potential via ODE integration.",
        "domain": "medical simulation",
        "complexity": "O(n × timesteps)",
        "tags": ["tissue", "cardiac", "electrophysiology", "medical-imaging", "ode-solver"],
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
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": ["reference.cpp"],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["32", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["32", "100"],
                    "description": "32 tissue points, 100 iterations; verified against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["32", "100"],
                    "description": "Default run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Compares GPU tissue simulation output against CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Floating-point comparison of membrane potential values",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 8. lulesh ─────────────────────────────────────────────────────────────
    {
        "kernel_name": "lulesh",
        "category": "physics",
        "description": "Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics (LULESH) mini-app. Solves Sedov blast wave problem on an unstructured hex mesh with GPU-accelerated force and kinematic calculations.",
        "domain": "hydrodynamics",
        "complexity": "O(elements × iterations)",
        "tags": ["lulesh", "hydrodynamics", "sedov-blast", "unstructured-mesh", "shock", "mini-app"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["lulesh.cu", "lulesh-init.cu", "lulesh-util.cu", "lulesh-viz.cu", "lulesh.h"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["lulesh.cu", "lulesh-init.cu", "lulesh-util.cu", "lulesh-viz.cu", "lulesh.h"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["lulesh.cc", "lulesh-comm.cc", "lulesh-init.cc", "lulesh-util.cc", "lulesh-viz.cc", "lulesh.h", "lulesh_tuple.h"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["lulesh.cc", "lulesh-init.cc", "lulesh-util.cc", "lulesh-viz.cc", "lulesh.h"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["-i", "100", "-s", "128", "-r", "11", "-b", "1", "-c", "1"],
            "timeout_seconds": 600,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["-i", "100", "-s", "45", "-r", "11", "-b", "1", "-c", "1"],
                    "description": "Small mesh (45^3 elements), 100 iterations; verify via TotalAbsDiff metric",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["-i", "100", "-s", "128", "-r", "11", "-b", "1", "-c", "1"],
                    "description": "Large mesh (128^3 elements), 100 iterations for timing",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "TotalAbsDiff\\s*=\\s*[\\d.eE+-]+", "description": "LULESH outputs TotalAbsDiff metric; small values indicate correct simulation"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "absolute",
                "note": "TotalAbsDiff should be near zero for correct Sedov blast simulation",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "elapsed_time",
                    "extraction": {"type": "regex", "pattern": "Elapsed time\\s+=\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                },
                {
                    "name": "grind_time",
                    "extraction": {"type": "regex", "pattern": "Grind time \\(us/z/c\\)\\s+=\\s+([\\d.eE+-]+)", "capture_group": 1},
                    "unit": "us/z/c",
                },
            ],
            "warmup_runs": 0,
            "measurement_runs": 3,
        },
    },

    # ── 9. thomas ─────────────────────────────────────────────────────────────
    {
        "kernel_name": "thomas",
        "category": "linear_algebra",
        "description": "Batched tridiagonal solver using the Thomas algorithm on GPU. Solves many independent tridiagonal linear systems in parallel, suitable for ADI methods and implicit PDE schemes.",
        "domain": "numerical linear algebra",
        "complexity": "O(n × batch_size)",
        "tags": ["thomas-algorithm", "tridiagonal", "batched-solver", "pcr", "linear-algebra"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "cuThomasBatch.cu", "cuThomasBatch.h", "ThomasMatrix.hpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE", "utils.hpp"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu", "cuThomasBatch.cu", "cuThomasBatch.h", "ThomasMatrix.hpp"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "LICENSE", "utils.hpp"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "ThomasMatrix.hpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "utils.hpp"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "ThomasMatrix.hpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "utils.hpp"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1024", "16384", "64", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1024", "16384", "64", "1"],
                    "description": "Single iteration with system_size=1024, num_systems=16384; checks max error",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["1024", "16384", "64", "100"],
                    "description": "100-iteration run for performance measurement",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "Maximum error:", "description": "Prints max L-infinity error of GPU solution vs analytical; small values indicate correctness"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Maximum error compares GPU Thomas solution against known analytical solution",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(ms\\)", "capture_group": 1},
                    "unit": "ms",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 10. keccaktreehash ────────────────────────────────────────────────────
    {
        "kernel_name": "keccaktreehash",
        "category": "crypto",
        "description": "GPU-accelerated Keccak (SHA-3) tree hashing. Computes Keccak-f[1600] permutation in parallel across tree nodes on GPU, then compares final hash state against sequential CPU reference.",
        "domain": "cryptographic hashing",
        "complexity": "O(tree_nodes × rounds)",
        "tags": ["keccak", "sha3", "tree-hash", "cryptography", "permutation"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "KeccakTreeGPU.cu", "KeccakF.cu", "Test.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "KeccakTree.h", "KeccakTypes.h", "KeccakF.h", "KeccakTreeGPU.h", "Test.h"],
                "verification_only": ["KeccakTreeCPU.cu", "KeccakTreeCPU.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu", "KeccakTreeGPU.cu", "KeccakF.cu", "Test.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "KeccakTree.h", "KeccakTypes.h", "KeccakF.h", "KeccakTreeGPU.h", "Test.h"],
                "verification_only": ["KeccakTreeCPU.cu", "KeccakTreeCPU.h"],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "KeccakTreeGPU.cpp", "KeccakF.cpp", "Test.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "KeccakTree.h", "KeccakTypes.h", "KeccakF.h", "KeccakTreeGPU.h", "Test.h"],
                "verification_only": ["KeccakTreeCPU.cpp", "KeccakTreeCPU.h"],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "KeccakTreeGPU.cpp", "KeccakF.cpp", "Test.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "KeccakTree.h", "KeccakTypes.h", "KeccakF.h", "KeccakTreeGPU.h", "Test.h"],
                "verification_only": ["KeccakTreeCPU.cpp", "KeccakTreeCPU.h"],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": [],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": [],
                    "description": "Run Keccak tree hash; compares GPU hash state against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": [],
                    "description": "Default run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "isEqual_KS() compares GPU Keccak state vs CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Total kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 11. md5hash ───────────────────────────────────────────────────────────
    {
        "kernel_name": "md5hash",
        "category": "crypto",
        "description": "GPU-accelerated MD5 hash cracking by brute-force search. Generates candidate strings on GPU, computes MD5 hashes in parallel, and compares against target digest.",
        "domain": "cryptographic hashing",
        "complexity": "O(search_space)",
        "tags": ["md5", "hash-cracking", "brute-force", "cryptography", "parallel-search"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["MD5Hash.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["MD5Hash.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["MD5Hash.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["MD5Hash.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1", "4"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1", "4"],
                    "description": "Search for MD5 match with string length 4; PASS if hash found",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["1", "4"],
                    "description": "Default run for hash throughput measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Reports PASS if MD5 search completes and rate is valid; FAIL otherwise"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Total kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 12. ccsd-trpdrv ───────────────────────────────────────────────────────
    {
        "kernel_name": "ccsd-trpdrv",
        "category": "other",
        "description": "CCSD(T) triples driver from computational chemistry (NWChem). GPU-accelerated tensor contraction kernels for coupled-cluster perturbative triples energy computation.",
        "domain": "computational chemistry",
        "complexity": "O(n^3)",
        "tags": ["ccsd", "coupled-cluster", "triples", "tensor-contraction", "nwchem", "quantum-chemistry"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "ccsd_trpdrv.cu", "ccsd_tengy.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "README"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["ccsd_tengy.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "README"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "ccsd_trpdrv.cpp", "ccsd_tengy.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "README"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp", "ccsd_trpdrv.cpp", "ccsd_tengy.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "README"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["160", "400", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["160", "400", "1"],
                    "description": "Single iteration; verifies GPU checksum against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["160", "400", "100"],
                    "description": "100-iteration run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Checksum comparison of GPU tensor contraction output vs CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "absolute",
                "note": "Checksum comparison between GPU and CPU tensor contraction results",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 13. babelstream ───────────────────────────────────────────────────────
    {
        "kernel_name": "babelstream",
        "category": "other",
        "description": "BabelStream memory bandwidth benchmark. Measures achievable memory bandwidth using Copy, Mul, Add, Triad, Dot, and Nstream kernels. Based on the STREAM benchmark methodology.",
        "domain": "memory bandwidth",
        "complexity": "O(n)",
        "tags": ["babelstream", "stream", "memory-bandwidth", "benchmark", "copy", "triad", "dot-product"],
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
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": [],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": [],
                    "description": "Default BabelStream run with internal validation of Copy/Mul/Add/Triad/Dot/Nstream",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "Function"},
                },
                "performance": {
                    "arguments": [],
                    "description": "Default run; reports bandwidth in MBytes/sec for each kernel",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "Function"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "Function", "description": "Output includes Function/MBytes/sec table header, indicating successful kernel execution"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly; internal validation aborts on error"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "triad_bandwidth",
                    "extraction": {"type": "regex", "pattern": "Triad\\s+([\\d.eE+-]+)", "capture_group": 1},
                    "unit": "MB/s",
                },
                {
                    "name": "copy_bandwidth",
                    "extraction": {"type": "regex", "pattern": "Copy\\s+([\\d.eE+-]+)", "capture_group": 1},
                    "unit": "MB/s",
                },
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 14. fpc ───────────────────────────────────────────────────────────────
    {
        "kernel_name": "fpc",
        "category": "other",
        "description": "Floating-Point Compressor (FPC) on GPU. Implements lossless compression of double-precision floating-point data using FCM and DFCM predictors with parallel encoding/decoding.",
        "domain": "data compression",
        "complexity": "O(n)",
        "tags": ["fpc", "compression", "floating-point", "lossless", "fcm", "dfcm"],
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
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["256", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["256", "1"],
                    "description": "Single iteration FPC compress/decompress; verifies round-trip correctness",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["256", "100"],
                    "description": "100-iteration run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Round-trip compress/decompress verification; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 15. feynman-kac ───────────────────────────────────────────────────────
    {
        "kernel_name": "feynman-kac",
        "category": "physics",
        "description": "Feynman-Kac PDE solver using GPU-accelerated Monte Carlo stochastic path integration. Solves elliptic PDE boundary value problems by simulating random walks and averaging boundary values.",
        "domain": "Monte Carlo / PDE solving",
        "complexity": "O(n_paths × path_length)",
        "tags": ["feynman-kac", "monte-carlo", "pde", "stochastic", "random-walk", "boundary-value"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "kernel.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "util.h"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "kernel.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "util.h"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "util.h"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["10"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["10"],
                    "description": "10-repeat Feynman-Kac stochastic solve; reports RMS error against analytical solution",
                    "input_files": [],
                    "expected_results": None,
                },
                "performance": {
                    "arguments": ["10"],
                    "description": "Default run for timing measurement",
                    "input_files": [],
                    "expected_results": None,
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "RMS absolute error in solution", "description": "Reports RMS absolute error; small values indicate convergence to analytical solution"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 0.1,
                "tolerance_type": "absolute",
                "note": "Monte Carlo method has statistical variance; RMS error should be small but not zero",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel time:\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 16. maxpool3d ─────────────────────────────────────────────────────────
    {
        "kernel_name": "maxpool3d",
        "category": "ml",
        "description": "3D max pooling layer for deep learning on GPU. Applies max pooling with configurable kernel/stride/padding over 3D input tensors. Verified against CPU reference implementation.",
        "domain": "machine learning",
        "complexity": "O(n × kernel_volume)",
        "tags": ["maxpool3d", "pooling", "3d", "deep-learning", "cnn", "volumetric"],
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
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["2048", "2048", "96", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["2048", "2048", "96", "1"],
                    "description": "Single iteration with width=2048, height=2048, depth=96; compared against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["2048", "2048", "96", "100"],
                    "description": "100-iteration run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Element-wise comparison of GPU maxpool3d output vs CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 17. secp256k1 ─────────────────────────────────────────────────────────
    {
        "kernel_name": "secp256k1",
        "category": "crypto",
        "description": "GPU-accelerated elliptic curve cryptography operations on the secp256k1 curve (used in Bitcoin/Ethereum). Performs parallel point multiplication and signature verification.",
        "domain": "cryptography",
        "complexity": "O(n × scalar_bits)",
        "tags": ["secp256k1", "elliptic-curve", "ecc", "bitcoin", "cryptography", "point-multiplication"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.clang", "Makefile.nvc", "CMakeLists.txt", "LICENSE"],
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
                    "description": "Single-iteration ECC operations; verifies against known test vectors",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["100"],
                    "description": "100-iteration run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Verifies GPU ECC results against known test vectors; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 18. tsp ───────────────────────────────────────────────────────────────
    {
        "kernel_name": "tsp",
        "category": "other",
        "description": "GPU-accelerated Travelling Salesman Problem solver using 2-opt local search. Reads city coordinates from .tsp file, performs parallel 2-opt evaluation on GPU, iteratively improves tour distance.",
        "domain": "combinatorial optimization",
        "complexity": "O(n^2 × iterations)",
        "tags": ["tsp", "travelling-salesman", "2-opt", "combinatorial-optimization", "local-search"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "d493.tsp"],
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
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["../tsp-cuda/d493.tsp", "24", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["../tsp-cuda/d493.tsp", "24", "1"],
                    "description": "Single iteration TSP solve on d493 (493 cities); verifies tour validity",
                    "input_files": ["../tsp-cuda/d493.tsp"],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["../tsp-cuda/d493.tsp", "24", "100"],
                    "description": "100-iteration 2-opt search for performance measurement",
                    "input_files": ["../tsp-cuda/d493.tsp"],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Verifies GPU 2-opt tour against reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 19. pso ───────────────────────────────────────────────────────────────
    {
        "kernel_name": "pso",
        "category": "other",
        "description": "Particle Swarm Optimization (PSO) on GPU. Parallelizes swarm evaluation across GPU threads, updating particle positions and velocities to minimize a fitness function (Rastrigin).",
        "domain": "metaheuristic optimization",
        "complexity": "O(particles × dimensions × iterations)",
        "tags": ["pso", "particle-swarm", "optimization", "metaheuristic", "rastrigin"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cpp", "kernel_gpu.cu", "kernel.h"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["kernel_cpu.cpp"],
            },
            "hip": {
                "prompt_payload": ["kernel_gpu.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["kernel_gpu.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["kernel_gpu.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["30", "10000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["30", "100"],
                    "description": "30 dimensions, 100 iterations; verifies GPU swarm result against CPU PSO",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["30", "10000"],
                    "description": "30 dimensions, 10000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Compares GPU PSO best fitness against CPU PSO reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "PSO is stochastic; fitness comparison uses tolerance",
            },
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Device offloading time\\s*=?\\s*([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },

    # ── 20. ga ────────────────────────────────────────────────────────────────
    {
        "kernel_name": "ga",
        "category": "other",
        "description": "GPU-Accelerated Genomic Alignment using coarse-grained plus fine-grained approach. Performs parallel sequence alignment with coarse matching followed by fine-grained scoring on GPU.",
        "domain": "bioinformatics / sequence alignment",
        "complexity": "O(target × query)",
        "tags": ["genomic-alignment", "sequence-alignment", "coarse-matching", "bioinformatics", "ga"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": ["reference.h"],
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": [],
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": [],
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": [],
            },
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["1000000", "1000", "11", "1"],
            "timeout_seconds": 600,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["10000", "100", "11", "1"],
                    "description": "Small sequences (target=10000, query=100) for quick correctness verification",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
                "performance": {
                    "arguments": ["1000000", "1000", "11", "1"],
                    "description": "Full-size alignment (target=1M, query=1000) for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"},
                },
            },
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Compares GPU alignment results against CPU reference; prints PASS or FAIL"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"},
            ],
            "floating_point": None,
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Total kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s",
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
    },
]


# ─── Spec and manifest generation ────────────────────────────────────────────

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
    skipped = 0
    manifest_entries = []
    warnings = []

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    for kernel_data in KERNELS:
        k = kernel_data["kernel_name"]
        for api in apis:
            if api not in kernel_data["files"]:
                warnings.append(f"WARNING: {k}-{api} has no file definition, skipping")
                continue

            spec = make_spec(kernel_data, api)
            spec_file = SPECS_DIR / f"hecbench-{k}-{api}.json"

            if spec_file.exists():
                print(f"SKIP (exists): {spec_file.name}")
                skipped += 1
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

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")

    print(f"\nTotal specs created: {created}")
    print(f"Total specs skipped (already exist): {skipped}")
    print(f"Total manifest entries added: {len(manifest_entries)}")


if __name__ == "__main__":
    main()
