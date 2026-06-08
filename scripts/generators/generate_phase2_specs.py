#!/usr/bin/env python3
"""
Phase 2: Generate 60 JSON spec files (15 kernels × 4 APIs) and update manifest.jsonl.
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
            "min_memory_gb": 2
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
            "cuda_cores": 5888
        },
        "cpu": {
            "model": "AMD Ryzen 9 7900X",
            "cores": 12,
            "threads": 24,
            "base_clock_ghz": 4.7
        },
        "software": {
            "os": "Ubuntu 22.04 LTS",
            "cuda_toolkit": "12.x",
            "gcc": "11+",
            "driver": "525.x+"
        }
    }
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
            "cuda_cores": 5888
        },
        "cpu": {
            "model": "AMD Ryzen 9 7900X",
            "cores": 12,
            "threads": 24,
            "base_clock_ghz": 4.7
        },
        "software": {
            "os": "Ubuntu 22.04 LTS",
            "cuda_toolkit": "12.x",
            "gcc": "11+",
            "driver": "525.x+"
        }
    }
}

# ─── Build config by API ─────────────────────────────────────────────────────

BUILD_BY_API = {
    "cuda": {
        "build_cmd": "make ARCH=sm_89",
        "clean_cmd": "make clean",
        "deps": ["CUDA Toolkit >= 11.0", "GCC >= 9.0"]
    },
    "hip": {
        "build_cmd": "make ARCH=sm_89",
        "clean_cmd": "make clean",
        "deps": ["ROCm HIP >= 5.0", "GCC >= 9.0"]
    },
    "sycl": {
        "build_cmd": "make",
        "clean_cmd": "make clean",
        "deps": ["Intel oneAPI DPC++/C++ Compiler", "GCC >= 9.0"]
    },
    "omp": {
        "build_cmd": "make",
        "clean_cmd": "make clean",
        "deps": ["GCC >= 9.0 with OpenMP support"]
    }
}

# ─── Kernel definitions ──────────────────────────────────────────────────────
# Each kernel specifies per-API file classifications and shared run/verify info.

KERNELS = [
    # ── 1. aes ────────────────────────────────────────────────────────────────
    {
        "kernel_name": "aes",
        "category": "crypto",
        "description": "AES-128 ECB encryption/decryption of grayscale image data on GPU. Processes blocks in parallel with shared-memory S-box lookups, verified against CPU reference AES implementation.",
        "domain": "cryptography",
        "complexity": "O(n × rounds)",
        "tags": ["aes", "encryption", "decryption", "block-cipher", "cryptography"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "aes.h", "COPYRIGHT", "README"],
                "verification_only": ["reference.cu"]
            },
            "hip": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "aes.h", "COPYRIGHT", "README"],
                "verification_only": ["reference.cu"]
            },
            "sycl": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "aes.h", "COPYRIGHT", "README"],
                "verification_only": ["reference.cpp"]
            },
            "omp": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "aes.h", "COPYRIGHT", "README"],
                "verification_only": ["reference.cpp"]
            }
        },
        # aes needs an external BMP input file
        "run": {
            "executable": "./main",
            "default_arguments": ["100", "0", "../urng-sycl/URNG_Input.bmp"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["10", "0", "../urng-sycl/URNG_Input.bmp"],
                    "description": "10-iteration AES encrypt with URNG input bitmap; verified via memcmp against CPU reference",
                    "input_files": ["../urng-sycl/URNG_Input.bmp"],
                    "expected_results": {"stdout_pattern": "Pass"}
                },
                "performance": {
                    "arguments": ["100", "0", "../urng-sycl/URNG_Input.bmp"],
                    "description": "100-iteration AES encrypt for timing measurement",
                    "input_files": ["../urng-sycl/URNG_Input.bmp"],
                    "expected_results": {"stdout_pattern": "Pass"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "Pass", "description": "memcmp of GPU output vs CPU AES reference; prints 'Pass' or 'Fail'"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": None
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 2. chacha20 ──────────────────────────────────────────────────────────
    {
        "kernel_name": "chacha20",
        "category": "crypto",
        "description": "ChaCha20 stream cipher keystream generation on GPU. Generates keystreams in parallel and verifies against known RFC 7539 test vectors via memcmp.",
        "domain": "cryptography",
        "complexity": "O(n × rounds)",
        "tags": ["chacha20", "stream-cipher", "encryption", "cryptography"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "chacha20.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "chacha20.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["main.cpp", "chacha20.h"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1"],
                    "description": "Single-iteration run for correctness validation against RFC test vector",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["100000"],
                    "description": "100000-iteration run for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "memcmp of GPU keystream vs RFC 7539 test vector"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": None
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average execution time of kernels:\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 3. lud ────────────────────────────────────────────────────────────────
    {
        "kernel_name": "lud",
        "category": "linear_algebra",
        "description": "LU Decomposition of a dense matrix using a blocked (tiled) algorithm with GPU-accelerated diagonal, perimeter, and internal update kernels using shared memory.",
        "domain": "dense linear algebra",
        "complexity": "O(n^3)",
        "tags": ["lu-decomposition", "linear-algebra", "dense-matrix", "tiled", "shared-memory"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["lud.cu", "lud_kernels.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "common/common.h"],
                "verification_only": ["common/common.cpp"]
            },
            "hip": {
                "prompt_payload": ["lud.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["lud.cpp", "kernel_lud_diagonal.sycl", "kernel_lud_perimeter.sycl", "kernel_lud_internal.sycl"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["lud.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["-s", "8192"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["-v", "-s", "256"],
                    "description": "Small matrix (256) with verification enabled; checks for element-wise mismatches",
                    "input_files": [],
                    "expected_results": None
                },
                "performance": {
                    "arguments": ["-s", "8192"],
                    "description": "Large matrix (8192) for performance measurement",
                    "input_files": [],
                    "expected_results": None
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "exit_code", "expected": 0, "description": "Exit code 0 indicates successful completion"},
                {"type": "stdout_pattern", "pattern": "Total kernel execution time", "description": "Presence of timing output indicates successful computation (verification via -v flag prints mismatch lines if incorrect)"}
            ],
            "floating_point": {
                "tolerance": 0.01,
                "tolerance_type": "absolute",
                "note": "Element-wise comparison in lud_verify(); mismatches printed to stdout"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Total kernel execution time\\s*:\\s*([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 4. eigenvalue ─────────────────────────────────────────────────────────
    {
        "kernel_name": "eigenvalue",
        "category": "linear_algebra",
        "description": "Eigenvalue computation of tridiagonal symmetric matrices using the bisection method with Gerschgorin intervals. GPU kernels compute eigenvalue counts per interval and recursively refine.",
        "domain": "numerical linear algebra",
        "complexity": "O(n × log(n))",
        "tags": ["eigenvalue", "bisection", "tridiagonal", "linear-algebra", "gerschgorin"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.cu", "reference.h", "utils.cu"]
            },
            "hip": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.cu", "reference.h", "utils.cu"]
            },
            "sycl": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.cpp", "reference.h", "utils.cpp"]
            },
            "omp": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": ["reference.cpp", "reference.h", "utils.cpp"]
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["2048", "10000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["2048", "1"],
                    "description": "Single iteration for correctness validation (2048-dim tridiagonal matrix)",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["2048", "10000"],
                    "description": "10000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "L2 norm comparison of GPU eigenvalues vs CPU reference"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "L2 norm comparison between GPU and CPU eigenvalue results"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 5. nbody ──────────────────────────────────────────────────────────────
    {
        "kernel_name": "nbody",
        "category": "physics",
        "description": "Gravitational N-body simulation computing all-pairs particle interactions. GPU kernels accelerate force computation, position/velocity updates, and kinetic energy accumulation. Verified against CPU reference.",
        "domain": "physics simulation",
        "complexity": "O(n^2 × steps)",
        "tags": ["n-body", "gravity", "particle-simulation", "physics", "all-pairs"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["GSimulation.cu", "GSimulationKernels.hpp", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "GSimulation.hpp", "Particle.hpp", "type.hpp", "License.txt", "README.md"],
                "verification_only": ["GSimulationReference.hpp"]
            },
            "hip": {
                "prompt_payload": ["GSimulation.cu", "GSimulationKernels.hpp", "main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "GSimulation.hpp", "Particle.hpp", "type.hpp", "License.txt", "README.md"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["GSimulation.cpp", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "GSimulation.hpp", "Particle.hpp", "type.hpp", "License.txt", "README.md"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["GSimulation.cpp", "main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "GSimulation.hpp", "Particle.hpp", "type.hpp", "License.txt"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["16000", "10"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["16000", "10"],
                    "description": "16000 particles, 10 steps; verifies kinetic energy against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["16000", "10"],
                    "description": "16000 particles, 10 steps for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Kinetic energy comparison: |GPU - CPU| < 1e-3"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Kinetic energy comparison between GPU and CPU reference"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "total_time",
                    "extraction": {"type": "regex", "pattern": "# Total Time \\(s\\)\\s*:\\s*([\\d.eE+-]+)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 6. simpleSpmv ────────────────────────────────────────────────────────
    {
        "kernel_name": "simpleSpmv",
        "category": "linear_algebra",
        "description": "Sparse matrix-vector multiplication (SpMV) comparing dense, CSR, and vector-CSR formats with varying thread block sizes. Verified by comparing GPU results against CPU serial SpMV via error rate.",
        "domain": "sparse linear algebra",
        "complexity": "O(nnz)",
        "tags": ["spmv", "sparse-matrix", "csr", "linear-algebra"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["kernels.cu", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "mv.h", "LICENSE"],
                "verification_only": ["utils.cpp"]
            },
            "hip": {
                "prompt_payload": ["kernels.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["kernels.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["kernels.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["16777216", "10240", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["16777216", "10240", "1"],
                    "description": "Single iteration for correctness; prints error rates near 0.0",
                    "input_files": [],
                    "expected_results": None
                },
                "performance": {
                    "arguments": ["16777216", "10240", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": None
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "exit_code", "expected": 0, "description": "Exit code 0 indicates successful execution; error rates printed to stdout should be near 0.0"}
            ],
            "floating_point": {
                "tolerance": 0.01,
                "tolerance_type": "relative",
                "note": "Error rate comparison between GPU and CPU serial SpMV"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average dense, sparse, and vector sparse kernel execution time \\(ms\\):\\s+([\\d.eE+-]+)", "capture_group": 1},
                    "unit": "ms"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 7. sobel ──────────────────────────────────────────────────────────────
    {
        "kernel_name": "sobel",
        "category": "image",
        "description": "Sobel edge detection filter on bitmap image data. Applies 3x3 gradient operators (Gx, Gy) to compute edge magnitude, verified against CPU reference via L2 norm comparison.",
        "domain": "image processing",
        "complexity": "O(width × height)",
        "tags": ["sobel", "edge-detection", "image-processing", "convolution", "gradient"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "sobel.h", "README"],
                "verification_only": ["reference.cu"]
            },
            "hip": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "sobel.h", "README"],
                "verification_only": ["reference.cu"]
            },
            "sycl": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "data.tar.gz"],
                "verification_only": ["reference.cpp"]
            },
            "omp": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "sobel.h", "README"],
                "verification_only": ["reference.cpp"]
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["../sobel-sycl/SobelFilter_Input.bmp", "100000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["../sobel-sycl/SobelFilter_Input.bmp", "1"],
                    "description": "Single-iteration Sobel filter on input bitmap; verified via L2 norm against CPU reference",
                    "input_files": ["../sobel-sycl/SobelFilter_Input.bmp"],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["../sobel-sycl/SobelFilter_Input.bmp", "100000"],
                    "description": "100000 iterations for performance measurement",
                    "input_files": ["../sobel-sycl/SobelFilter_Input.bmp"],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "L2 norm of GPU vs CPU Sobel output; PASS if within tolerance"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "L2 norm comparison between GPU and CPU Sobel filter output"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 8. dct8x8 ────────────────────────────────────────────────────────────
    {
        "kernel_name": "dct8x8",
        "category": "other",
        "description": "Forward and inverse 8x8 Discrete Cosine Transform (DCT/IDCT) on 2D image data. Used in JPEG-like compression. GPU-accelerated with shared memory, verified against CPU gold standard via L2 norm.",
        "domain": "signal processing",
        "complexity": "O(width × height)",
        "tags": ["dct", "idct", "jpeg", "compression", "signal-processing", "shared-memory"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "DCT8x8.h"],
                "verification_only": ["DCT8x8_gold.cu"]
            },
            "hip": {
                "prompt_payload": ["kernels.cu", "main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "DCT8x8.h"],
                "verification_only": ["DCT8x8_gold.cu"]
            },
            "sycl": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "DCT8x8.h"],
                "verification_only": ["DCT8x8_gold.cpp"]
            },
            "omp": {
                "prompt_payload": ["kernels.cpp", "main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "DCT8x8.h"],
                "verification_only": ["DCT8x8_gold.cpp"]
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["8192", "8192", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["8192", "8192", "1"],
                    "description": "Single iteration for correctness; verifies both DCT and IDCT via L2 norm",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["8192", "8192", "100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "L2 norm < 1e-6 for both DCT and IDCT; prints PASS/FAIL for each"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "Relative L2 norm comparison between GPU DCT/IDCT and CPU gold standard"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "dct_time",
                    "extraction": {"type": "regex", "pattern": "Average DCT8x8 kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                },
                {
                    "name": "idct_time",
                    "extraction": {"type": "regex", "pattern": "Average IDCT8x8 kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 9. convolutionSeparable ──────────────────────────────────────────────
    {
        "kernel_name": "convolutionSeparable",
        "category": "other",
        "description": "Separable 2D convolution decomposed into row-pass and column-pass with kernel radius 8. Uses shared-memory tiling for efficient data reuse. Verified against CPU reference via L2 norm.",
        "domain": "signal processing",
        "complexity": "O(width × height × kernel_radius)",
        "tags": ["convolution", "separable", "image-processing", "signal-processing", "shared-memory"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["conv.cu", "main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "conv.h"],
                "verification_only": ["conv_gold.cu"]
            },
            "hip": {
                "prompt_payload": ["conv.cu", "main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "conv.h"],
                "verification_only": ["conv_gold.cu"]
            },
            "sycl": {
                "prompt_payload": ["conv.cpp", "main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt", "conv.h"],
                "verification_only": ["conv_gold.cpp"]
            },
            "omp": {
                "prompt_payload": ["conv.cpp", "main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "conv.h"],
                "verification_only": ["conv_gold.cpp"]
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["8192", "8192", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["8192", "8192", "1"],
                    "description": "Single iteration for correctness; L2 norm compared against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["8192", "8192", "1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "L2 norm < 1e-6 compared against CPU convolution reference"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "L2 norm comparison between GPU and CPU separable convolution"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 10. fft ───────────────────────────────────────────────────────────────
    {
        "kernel_name": "fft",
        "category": "other",
        "description": "1D FFT and inverse FFT of 512-point complex sequences using radix-8 decomposition with shared memory transposes. Verified element-wise against CPU reference implementation.",
        "domain": "signal processing",
        "complexity": "O(n log n)",
        "tags": ["fft", "ifft", "fourier-transform", "signal-processing", "radix-8", "shared-memory"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "fft1D_512.h", "ifft1D_512.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": ["reference.h"]
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "fft1D_512.sycl", "ifft1D_512.sycl"],
                "support_files": ["Makefile", "CMakeLists.txt", "LICENSE"],
                "verification_only": ["reference.h"]
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "LICENSE"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["3", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["3", "1"],
                    "description": "Single pass for correctness; verifies both FFT and iFFT",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["3", "100"],
                    "description": "100 passes for performance measurement (problem size 3 = large)",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Element-wise comparison of GPU FFT/iFFT vs CPU reference; prints 'FFT PASS' and 'iFFT PASS'"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-5,
                "tolerance_type": "absolute",
                "note": "Element-wise comparison with tolerance for FFT/iFFT results"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 11. ising ─────────────────────────────────────────────────────────────
    {
        "kernel_name": "ising",
        "category": "physics",
        "description": "2D Ising model Monte Carlo simulation using checkerboard (black/white) lattice decomposition with Metropolis-Hastings spin flipping. Verified by comparing GPU lattice state against CPU reference.",
        "domain": "statistical physics",
        "complexity": "O(nx × ny × iterations)",
        "tags": ["ising-model", "monte-carlo", "metropolis", "statistical-physics", "lattice"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "cudamacro.h"],
                "verification_only": ["reference.h"]
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "Makefile.hipcl", "CMakeLists.txt", "hipmacro.h"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["-x", "5120", "-y", "5120", "-w", "10", "-n", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["-x", "512", "-y", "512", "-w", "10", "-n", "100"],
                    "description": "Small lattice for quick correctness validation",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["-x", "5120", "-y", "5120", "-w", "10", "-n", "1000"],
                    "description": "Large lattice with 1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Element-wise lattice comparison between GPU and CPU reference"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": None
        },
        "performance": {
            "metrics": [
                {
                    "name": "elapsed_time",
                    "extraction": {"type": "regex", "pattern": "elapsed time:\\s+([\\d.eE+-]+)\\s+sec", "capture_group": 1},
                    "unit": "s"
                },
                {
                    "name": "updates_per_ns",
                    "extraction": {"type": "regex", "pattern": "updates per ns:\\s+([\\d.eE+-]+)", "capture_group": 1},
                    "unit": "updates/ns"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 12. merge ─────────────────────────────────────────────────────────────
    {
        "kernel_name": "merge",
        "category": "sort",
        "description": "GPU merge of two sorted arrays using the merge path (diagonal intersection) algorithm. Tested across 4 data types (uint32, float, uint64, double). Verified by checking output is monotonically non-decreasing.",
        "domain": "parallel primitives",
        "complexity": "O(n)",
        "tags": ["merge", "merge-path", "sorting", "parallel-primitives"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "kernels.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": []
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "kernels.h"],
                "support_files": ["Makefile", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["main.cpp", "kernels.h"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt", "COPYRIGHT"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100000", "100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["100000", "1"],
                    "description": "Single run for correctness; checks merged arrays are monotonically sorted",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["100000", "100"],
                    "description": "100 runs for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Checks merged output is monotonically non-decreasing for 4 data types"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": None
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time:\\s+([\\d.eE+-]+)\\s+\\(us\\)", "capture_group": 1},
                    "unit": "us"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 13. fwt ───────────────────────────────────────────────────────────────
    {
        "kernel_name": "fwt",
        "category": "other",
        "description": "Fast Walsh-Hadamard Transform for dyadic convolution. Applies batched FWT to data and kernel arrays, modulates in transform domain, then inverse-transforms. Verified against CPU dyadic convolution via L2 norm.",
        "domain": "signal processing",
        "complexity": "O(n log n)",
        "tags": ["walsh-hadamard", "fwt", "dyadic-convolution", "signal-processing"],
        "multi_file": True,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu", "kernels.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.cu"]
            },
            "hip": {
                "prompt_payload": ["main.cu", "kernels.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["main.cpp", "kernels.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.cpp"]
            },
            "omp": {
                "prompt_payload": ["main.cpp", "kernels.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": ["reference.cpp"]
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["100"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["1"],
                    "description": "Single iteration for correctness; verifies FWT-based dyadic convolution via L2 norm",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["100"],
                    "description": "100 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "L2 norm < 1e-6 comparing GPU FWT dyadic convolution vs CPU reference"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-6,
                "tolerance_type": "relative",
                "note": "L2 norm comparison between GPU FWT result and CPU dyadic convolution"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average device execution time\\s+([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 14. chi2 ──────────────────────────────────────────────────────────────
    {
        "kernel_name": "chi2",
        "category": "other",
        "description": "Chi-squared independence test on SNP genotype data for genome-wide association studies (GWAS). Counts genotype occurrences in case/control groups and computes chi-squared statistics in parallel.",
        "domain": "bioinformatics / statistical genetics",
        "complexity": "O(rows × cols)",
        "tags": ["chi-squared", "statistics", "gwas", "bioinformatics", "genetics"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["chi2.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.h"]
            },
            "hip": {
                "prompt_payload": ["chi2.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["chi2.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["chi2.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["4000", "400000", "2000", "2000", "256", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["4000", "400000", "2000", "2000", "256", "1"],
                    "description": "Single iteration; verifies chi2 statistics against CPU reference with tolerance 1e-4",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["4000", "400000", "2000", "2000", "256", "1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Element-wise comparison of GPU vs CPU chi-squared statistics; tolerance 1e-4"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-4,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference comparison between GPU and CPU results"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time",
                    "extraction": {"type": "regex", "pattern": "Average (?:chi_)?kernel execution time\\s*=?\\s*([\\d.eE+-]+)\\s+\\(s\\)", "capture_group": 1},
                    "unit": "s"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
    },

    # ── 15. bilateral ────────────────────────────────────────────────────────
    {
        "kernel_name": "bilateral",
        "category": "image",
        "description": "Bilateral filter for edge-preserving image smoothing. Uses Gaussian kernels for both spatial proximity and intensity similarity. Runs 3 filter sizes (3x3, 6x6, 9x9), each verified against CPU reference.",
        "domain": "image processing",
        "complexity": "O(width × height × R^2)",
        "tags": ["bilateral-filter", "image-processing", "edge-preserving", "smoothing", "gaussian"],
        "multi_file": False,
        "files": {
            "cuda": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": ["reference.h"]
            },
            "hip": {
                "prompt_payload": ["main.cu"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "sycl": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "CMakeLists.txt"],
                "verification_only": []
            },
            "omp": {
                "prompt_payload": ["main.cpp"],
                "support_files": ["Makefile", "Makefile.aomp", "Makefile.nvc", "CMakeLists.txt"],
                "verification_only": []
            }
        },
        "run": {
            "executable": "./main",
            "default_arguments": ["2960", "1440", "0.5", "0.5", "1000"],
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": {
                "correctness": {
                    "arguments": ["2960", "1440", "0.5", "0.5", "1"],
                    "description": "Single iteration for correctness; verifies all 3 filter sizes against CPU reference",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                },
                "performance": {
                    "arguments": ["2960", "1440", "0.5", "0.5", "1000"],
                    "description": "1000 iterations for performance measurement",
                    "input_files": [],
                    "expected_results": {"stdout_pattern": "PASS"}
                }
            }
        },
        "verification": {
            "method": "self_checking",
            "strategies": [
                {"type": "stdout_pattern", "pattern": "PASS", "description": "Element-wise comparison with tolerance 1e-3 for all 3 filter sizes"},
                {"type": "exit_code", "expected": 0, "description": "Process exits cleanly"}
            ],
            "floating_point": {
                "tolerance": 1e-3,
                "tolerance_type": "absolute",
                "note": "Element-wise absolute difference comparison for 3x3, 6x6, and 9x9 filter sizes"
            }
        },
        "performance": {
            "metrics": [
                {
                    "name": "kernel_time_3x3",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time \\(3x3\\)\\s+([\\d.eE+-]+)\\s+\\(ms\\)", "capture_group": 1},
                    "unit": "ms"
                },
                {
                    "name": "kernel_time_6x6",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time \\(6x6\\)\\s+([\\d.eE+-]+)\\s+\\(ms\\)", "capture_group": 1},
                    "unit": "ms"
                },
                {
                    "name": "kernel_time_9x9",
                    "extraction": {"type": "regex", "pattern": "Average kernel execution time \\(9x9\\)\\s+([\\d.eE+-]+)\\s+\\(ms\\)", "capture_group": 1},
                    "unit": "ms"
                }
            ],
            "warmup_runs": 1,
            "measurement_runs": 5
        }
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
            "unique_id": f"hecbench-{k.lower()}-{api}",
            "source_suite": "hecbench"
        },
        "provenance": {
            "repository": {
                "url": "https://github.com/zjin-lcf/HeCBench",
                "commit": "archive-download",
                "branch": "master"
            },
            "repo_root": "HeCBench-master/",
            "source_path": f"src/{k}-{api}",
            "license": "MIT"
        },
        "files": {
            "prompt_payload": files["prompt_payload"],
            "support_files": files.get("support_files", []),
            "verification_only": files.get("verification_only", [])
        },
        "implementation": {
            "api": api,
            "api_version": None,
            "language": "C++",
            "language_standard": "C++17"
        },
        "build": {
            "environment": {
                "preferred": "system",
                "conda": None,
                "system": {
                    "dependencies": build_cfg["deps"]
                }
            },
            "build_system": "make",
            "working_directory": f"src/{k}-{api}",
            "commands": {
                "configure": None,
                "build": build_cfg["build_cmd"],
                "clean": build_cfg["clean_cmd"]
            },
            "variables": None,
            "outputs": {
                "executable": "main"
            }
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
            "multi_file": kernel_data["multi_file"]
        }
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
        "spec_file": f"specs/hecbench-{k.lower()}-{api}.json",
        "source_dir": f"HeCBench-master/src/{k}-{api}"
    }


def main():
    apis = ["cuda", "hip", "sycl", "omp"]
    created = 0
    manifest_entries = []

    for kernel_data in KERNELS:
        k = kernel_data["kernel_name"]
        for api in apis:
            spec = make_spec(kernel_data, api)
            spec_file = SPECS_DIR / f"hecbench-{k.lower()}-{api}.json"

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
