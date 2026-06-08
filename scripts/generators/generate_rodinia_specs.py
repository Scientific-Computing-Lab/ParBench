#!/usr/bin/env python3
"""
Generate ParBench spec files for the Rodinia benchmark suite.

Reads:  analysis/rodinia_survey.json
Writes: specs/rodinia-{kernel}-{api}.json  (one per kernel-API variant)
        manifest.jsonl                      (entries appended)
        analysis/rodinia_api_gaps.md        (API gaps report)

Usage:
    python scripts/generate_rodinia_specs.py [--dry-run] [--force]
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SURVEY_JSON = PROJECT_ROOT / "analysis" / "rodinia_survey.json"
SPECS_DIR = PROJECT_ROOT / "specs"
MANIFEST_FILE = PROJECT_ROOT / "manifest.jsonl"
GAPS_REPORT = PROJECT_ROOT / "analysis" / "rodinia_api_gaps.md"

RODINIA_REPO_URL = "https://github.com/yuhc/gpu-rodinia"
RODINIA_COMMIT = "9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4"
RODINIA_REPO_ROOT = "rodinia/rodinia-src"
RODINIA_LICENSE = "NCSA/Illinois Open Source License"

# APIs we support (maps survey key → schema enum value)
SUPPORTED_APIS = {"cuda": "cuda", "opencl": "opencl", "omp": "omp"}

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

# --------------------------------------------------------------------------
# Kernel-specific knowledge table
# Keys: kernel_name (matching rodinia_survey.json)
# Each entry has:
#   exe:       {api: executable_path_relative_to_working_dir}
#   args_sm:   small/correctness run arguments (list of strings)
#   args_lg:   large/performance run arguments (list of strings)
#   data_files:{config_name: [file paths relative to working_dir]}
#   desc:      description string
#   domain:    domain string
#   tags:      list of tag strings
# --------------------------------------------------------------------------
KERNEL_DATA = {
    "backprop": {
        "exe": {"cuda": "./backprop", "opencl": "./backprop", "omp": "./backprop"},
        "args_sm": ["65536"],
        "args_lg": ["1048576"],
        "data_files": {},
        "desc": "Backpropagation neural network training on GPU",
        "domain": "machine learning",
        "tags": ["neural-network", "backpropagation", "machine-learning"],
    },
    "bfs": {
        "exe": {"cuda": "./bfs.out", "opencl": "./bfs.out", "omp": "./bfs"},
        "args_sm": ["../../data/bfs/graph1MW_6.txt"],
        "args_lg": ["../../data/bfs/graph1MW_6.txt"],
        "data_files": {
            "correctness": ["../../data/bfs/graph1MW_6.txt"],
            "performance": ["../../data/bfs/graph1MW_6.txt"],
        },
        "desc": "Breadth-first search on a large sparse graph",
        "domain": "graph traversal",
        "tags": ["bfs", "graph", "sparse"],
    },
    "b+tree": {
        "exe": {
            "cuda": "./b+tree.out",
            "opencl": "./b+tree.out",
            "omp": "./b+tree.out",
        },
        "args_sm": [
            "file", "../../data/b+tree/mil.txt",
            "command", "../../data/b+tree/command.txt",
        ],
        "args_lg": [
            "file", "../../data/b+tree/mil.txt",
            "command", "../../data/b+tree/command.txt",
        ],
        "data_files": {
            "correctness": [
                "../../data/b+tree/mil.txt",
                "../../data/b+tree/command.txt",
            ],
            "performance": [
                "../../data/b+tree/mil.txt",
                "../../data/b+tree/command.txt",
            ],
        },
        "desc": "B+ tree search and range queries on GPU",
        "domain": "data structures",
        "tags": ["b-plus-tree", "search", "data-structures"],
    },
    "cfd": {
        "exe": {
            "cuda": "./euler3d",
            "opencl": "./euler3d",
            "omp": "./euler3d_cpu",
        },
        "args_sm": ["../../data/cfd/fvcorr.domn.097K"],
        "args_lg": ["../../data/cfd/missile.domn.0.2M"],
        "data_files": {
            "correctness": ["../../data/cfd/fvcorr.domn.097K"],
            "performance": ["../../data/cfd/missile.domn.0.2M"],
        },
        "desc": "Computational fluid dynamics Euler solver on GPU",
        "domain": "computational fluid dynamics",
        "tags": ["cfd", "euler", "fluid-dynamics"],
    },
    "dwt2d": {
        "exe": {"cuda": "./dwt2d", "opencl": "./dwt2d"},
        "args_sm": [
            "../../data/dwt2d/rgb.bmp", "-d", "1024x1024", "-f", "-5", "-l", "3",
        ],
        "args_lg": [
            "../../data/dwt2d/rgb.bmp", "-d", "1024x1024", "-f", "-5", "-l", "3",
        ],
        "data_files": {
            "correctness": ["../../data/dwt2d/rgb.bmp"],
            "performance": ["../../data/dwt2d/rgb.bmp"],
        },
        "desc": "2D discrete wavelet transform (DWT-5/3 and DWT-9/7) on GPU",
        "domain": "signal processing",
        "tags": ["dwt", "wavelet", "image-processing"],
    },
    "gaussian": {
        "exe": {"cuda": "./gaussian", "opencl": "./gaussian"},
        "args_sm": ["-s", "16"],
        "args_lg": ["-f", "../../data/gaussian/matrix1024.txt"],
        "data_files": {
            "performance": ["../../data/gaussian/matrix1024.txt"],
        },
        "desc": "Gaussian elimination with partial pivoting on GPU",
        "domain": "linear algebra",
        "tags": ["gaussian-elimination", "linear-algebra"],
    },
    "heartwall": {
        "exe": {
            "cuda": "./heartwall",
            "opencl": "./heartwall",
            "omp": "./heartwall",
        },
        "args_sm": ["../../data/heartwall/test.avi", "20"],
        "args_lg": ["../../data/heartwall/test.avi", "104"],
        "data_files": {
            "correctness": ["../../data/heartwall/test.avi"],
            "performance": ["../../data/heartwall/test.avi"],
        },
        "desc": "Heart wall tracking in ultrasound video frames on GPU",
        "domain": "image processing",
        "tags": ["heart-wall", "tracking", "image-processing", "ultrasound"],
    },
    "hotspot": {
        "exe": {"cuda": "./hotspot", "opencl": "./hotspot", "omp": "./hotspot"},
        "args_sm": [
            "512", "512", "2", "4",
            "../../data/hotspot/power_512",
            "../../data/hotspot/temp_512",
            "output.out",
        ],
        "args_lg": [
            "1024", "1024", "2", "4",
            "../../data/hotspot/power_1024",
            "../../data/hotspot/temp_1024",
            "output.out",
        ],
        "data_files": {
            "correctness": [
                "../../data/hotspot/power_512",
                "../../data/hotspot/temp_512",
            ],
            "performance": [
                "../../data/hotspot/power_1024",
                "../../data/hotspot/temp_1024",
            ],
        },
        "desc": "2D thermal simulation stencil computation on GPU",
        "domain": "simulation",
        "tags": ["hotspot", "stencil", "thermal", "simulation"],
    },
    "hotspot3D": {
        "exe": {"cuda": "./3D", "opencl": "./3D", "omp": "./3D"},
        "args_sm": [
            "512", "8", "100",
            "../../data/hotspot3D/power_512x8",
            "../../data/hotspot3D/temp_512x8",
            "output.out",
        ],
        "args_lg": [
            "512", "8", "100",
            "../../data/hotspot3D/power_512x8",
            "../../data/hotspot3D/temp_512x8",
            "output.out",
        ],
        "data_files": {
            "correctness": [
                "../../data/hotspot3D/power_512x8",
                "../../data/hotspot3D/temp_512x8",
            ],
            "performance": [
                "../../data/hotspot3D/power_512x8",
                "../../data/hotspot3D/temp_512x8",
            ],
        },
        "desc": "3D thermal simulation stencil computation on GPU",
        "domain": "simulation",
        "tags": ["hotspot3d", "stencil", "thermal", "simulation", "3d"],
    },
    "huffman": {
        "exe": {"cuda": "./pavle"},
        "args_sm": [],
        "args_lg": [],
        "data_files": {},
        "desc": "Huffman encoding and decoding on GPU",
        "domain": "data compression",
        "tags": ["huffman", "compression", "encoding"],
    },
    "hybridsort": {
        "exe": {"cuda": "./hybridsort", "opencl": "./hybridsort"},
        "args_sm": [],
        "args_lg": [],
        "data_files": {},
        "desc": "Hybrid sorting combining merge sort and bucket sort on GPU",
        "domain": "sorting",
        "tags": ["sort", "merge-sort", "bucket-sort", "hybrid"],
    },
    "kmeans": {
        "exe": {
            "cuda": "./kmeans",
            "opencl": "./kmeans",
            "omp": "./kmeans_openmp/kmeans",
        },
        "args_sm": ["-i", "../../data/kmeans/kdd_cup"],
        "args_lg": ["-i", "../../data/kmeans/kdd_cup"],
        "data_files": {
            "correctness": ["../../data/kmeans/kdd_cup"],
            "performance": ["../../data/kmeans/kdd_cup"],
        },
        "desc": "K-means clustering on GPU",
        "domain": "machine learning",
        "tags": ["k-means", "clustering", "machine-learning"],
    },
    "lavaMD": {
        "exe": {"cuda": "./lavaMD", "opencl": "./lavaMD", "omp": "./lavaMD"},
        "args_sm": ["-cores", "1", "-boxes", "10"],
        "args_lg": ["-cores", "1", "-boxes", "30"],
        "data_files": {},
        "desc": "Molecular dynamics simulation using Lennard-Jones potential on GPU",
        "domain": "molecular dynamics",
        "tags": ["molecular-dynamics", "lennard-jones", "simulation"],
    },
    "leukocyte": {
        "exe": {
            "cuda": "./CUDA/leukocyte",
            "opencl": "./OpenCL/leukocyte.out",
            "omp": "./OpenMP/leukocyte",
        },
        "args_sm": ["../../data/leukocyte/testfile.avi", "5"],
        "args_lg": ["../../data/leukocyte/testfile.avi", "5"],
        "data_files": {
            "correctness": ["../../data/leukocyte/testfile.avi"],
            "performance": ["../../data/leukocyte/testfile.avi"],
        },
        "desc": "Leukocyte tracking in video microscopy frames on GPU",
        "domain": "image processing",
        "tags": ["leukocyte", "tracking", "image-processing", "microscopy"],
    },
    "lud": {
        "exe": {
            "cuda": "./cuda/lud_cuda",
            "opencl": "./ocl/lud",
            "omp": "./omp/lud",
        },
        "args_sm": ["-s", "256"],
        "args_lg": ["-f", "../../data/lud/256.dat"],
        "data_files": {
            "performance": ["../../data/lud/256.dat"],
        },
        "desc": "LU decomposition on GPU",
        "domain": "linear algebra",
        "tags": ["lud", "lu-decomposition", "linear-algebra"],
    },
    "mummergpu": {
        "exe": {"cuda": "./src/mummergpu", "omp": "./src/mummergpu"},
        "args_sm": ["data/shortref.fa", "data/shortqry.fa"],
        "args_lg": ["data/shortref.fa", "data/shortqry.fa"],
        "data_files": {
            "correctness": ["data/shortref.fa", "data/shortqry.fa"],
            "performance": ["data/shortref.fa", "data/shortqry.fa"],
        },
        "desc": "Exact DNA sequence alignment using suffix trees on GPU",
        "domain": "bioinformatics",
        "tags": ["dna", "sequence-alignment", "suffix-tree", "bioinformatics"],
    },
    "myocyte": {
        "exe": {
            "cuda": "./myocyte.out",
            "opencl": "./myocyte",
            "omp": "./myocyte",
        },
        "args_sm": ["100", "1", "0"],
        "args_lg": ["1000", "1", "0"],
        "data_files": {},
        "desc": "Cardiac myocyte electrophysiology simulation (ODE solver) on GPU",
        "domain": "scientific simulation",
        "tags": ["myocyte", "cardiac", "ode", "simulation"],
    },
    "nn": {
        "exe": {"cuda": "./nn", "opencl": "./nn", "omp": "./nn"},
        "args_sm": ["filelist.txt", "-r", "5", "-lat", "30", "-lng", "90"],
        "args_lg": ["filelist.txt", "-r", "5", "-lat", "30", "-lng", "90"],
        "data_files": {
            "correctness": ["filelist.txt"],
            "performance": ["filelist.txt"],
        },
        "desc": "Nearest neighbor search on GPU",
        "domain": "data mining",
        "tags": ["nearest-neighbor", "knn", "search", "data-mining"],
    },
    "nw": {
        "exe": {"cuda": "./needle", "opencl": "./needle", "omp": "./needle"},
        "args_sm": ["2048", "10"],
        "args_lg": ["8192", "10"],
        "data_files": {},
        "desc": "Needleman-Wunsch global sequence alignment on GPU",
        "domain": "bioinformatics",
        "tags": ["needleman-wunsch", "sequence-alignment", "dynamic-programming"],
    },
    "particlefilter": {
        "exe": {
            "cuda": "./particlefilter_float",
            "opencl": "./OCL_particlefilter_single.out",
            "omp": "./particlefilter",
        },
        "args_sm": ["-x", "128", "-y", "128", "-z", "10", "-np", "1000"],
        "args_lg": ["-x", "128", "-y", "128", "-z", "10", "-np", "10000"],
        "data_files": {},
        "desc": "Particle filter for Bayesian state estimation on GPU",
        "domain": "signal processing",
        "tags": ["particle-filter", "bayesian", "state-estimation"],
    },
    "pathfinder": {
        "exe": {
            "cuda": "./pathfinder",
            "opencl": "./pathfinder",
            "omp": "./pathfinder",
        },
        "args_sm": ["100000", "100", "20"],
        "args_lg": ["100000", "1000", "20"],
        "data_files": {},
        "desc": "Dynamic programming shortest-path computation on GPU",
        "domain": "dynamic programming",
        "tags": ["pathfinder", "dynamic-programming", "shortest-path"],
    },
    "srad": {
        "exe": {
            "cuda": "./srad_v1/srad",
            "opencl": "./srad",
            "omp": "./srad_v1/srad",
        },
        "args_sm": [
            "502", "458", "0", "127", "0", "127", "2", "0.5", "2",
        ],
        "args_lg": [
            "502", "458", "0", "127", "0", "127", "100", "0.5", "2",
        ],
        "data_files": {},
        "desc": "Speckle reducing anisotropic diffusion (SRAD) image filter on GPU",
        "domain": "image processing",
        "tags": ["srad", "anisotropic-diffusion", "image-processing", "denoising"],
    },
    "streamcluster": {
        "exe": {
            "cuda": "./sc_gpu",
            "opencl": "./sc_gpu",
            "omp": "./sc_omp",
        },
        "args_sm": [
            "10", "20", "256", "65536", "65536", "1000",
            "none", "output.txt", "1",
        ],
        "args_lg": [
            "10", "20", "256", "1048576", "1048576", "1000",
            "none", "output.txt", "1",
        ],
        "data_files": {},
        "desc": "Online stream clustering algorithm on GPU",
        "domain": "data mining",
        "tags": ["stream-clustering", "online-learning", "data-mining"],
    },
}

# Domains → category mapping for manifest (must match manifest_schema.json enum)
# Allowed: ml, graph, physics, linear_algebra, stencil, reduction, sort,
#          molecular_dynamics, image, crypto, financial, other
DOMAIN_TO_CATEGORY = {
    "machine learning": "ml",
    "graph traversal": "graph",
    "data structures": "other",
    "computational fluid dynamics": "physics",
    "signal processing": "physics",
    "linear algebra": "linear_algebra",
    "image processing": "image",
    "simulation": "physics",
    "molecular dynamics": "molecular_dynamics",
    "data compression": "other",
    "sorting": "sort",
    "bioinformatics": "other",
    "scientific simulation": "physics",
    "data mining": "other",
    "dynamic programming": "other",
}

# Utility directory patterns — files in these dirs go to support_files
UTILITY_DIR_PATTERNS = [
    "meschach_lib/",
    "AVI/",
    "util/",
    "experiments/",
    "DOC/",
]


def slugify(kernel_name):
    """Convert kernel name to URL-safe slug for filenames and unique_id."""
    slug = kernel_name.lower()
    slug = re.sub(r"[^a-z0-9_-]", "", slug.replace("+", ""))
    return slug


def is_utility_file(path):
    """Return True if path is in a utility/support library directory."""
    return any(
        path.startswith(pat) or f"/{pat}" in path
        for pat in UTILITY_DIR_PATTERNS
    )


def select_files(api, source_files, header_files, opencl_files, build_files):
    """Split files into prompt_payload and support_files based on API."""
    prompt_payload = []
    support_files = []

    if api == "cuda":
        # Prompt: .cu files (all GPU-specific code)
        # Support: .c files, headers, Makefiles
        for f in source_files:
            if f.endswith(".cu"):
                prompt_payload.append(f)
            else:
                support_files.append(f)
    elif api == "opencl":
        # Prompt: .cl kernel files + non-utility host files
        prompt_payload.extend(opencl_files)
        for f in source_files:
            if not is_utility_file(f):
                prompt_payload.append(f)
            else:
                support_files.append(f)
    elif api == "omp":
        # Prompt: non-utility source files (contain OMP pragmas)
        for f in source_files:
            if not is_utility_file(f):
                prompt_payload.append(f)
            else:
                support_files.append(f)

    # Headers and build files always in support
    support_files.extend(header_files)
    support_files.extend(build_files)

    # Deduplicate while preserving order
    seen = set()
    prompt_payload = [f for f in prompt_payload if not (f in seen or seen.add(f))]
    seen = set()
    support_files = [f for f in support_files if not (f in seen or seen.add(f))]

    return prompt_payload, support_files


def detect_language(source_files, api):
    """Detect primary source language."""
    if api == "cuda":
        return "C++"
    has_cpp = any(f.endswith((".cpp", ".cxx", ".cc")) for f in source_files)
    has_c = any(f.endswith(".c") for f in source_files)
    if has_cpp:
        return "C++"
    if has_c:
        return "C"
    return "C++"


def build_environment(api, includes_common_config, makefile_env_vars):
    """Build the environment section of the spec."""
    deps = []
    if api == "cuda":
        deps = ["cuda-toolkit>=11.0", "gcc>=9.0"]
    elif api == "opencl":
        deps = ["opencl>=2.0", "gcc>=9.0"]
    elif api == "omp":
        deps = ["gcc>=9.0 (with OpenMP support)"]

    env = {
        "preferred": "system",
        "conda": None,
        "system": {"dependencies": deps},
    }
    return env


def build_commands(api, includes_common_config, makefile_env_vars):
    """Construct build commands."""
    if api == "cuda":
        if includes_common_config:
            build_cmd = "make CUDA_DIR=/usr/local/cuda"
        else:
            build_cmd = "make"
    elif api == "opencl":
        if includes_common_config:
            build_cmd = (
                "make OPENCL_INC=/usr/local/cuda/include "
                "OPENCL_LIB=/usr/local/cuda/lib64"
            )
        else:
            build_cmd = "make"
    else:  # omp
        build_cmd = "make"

    return {
        "configure": None,
        "build": build_cmd,
        "clean": "make clean",
    }


def build_variables(api, includes_common_config, makefile_env_vars):
    """Build template variables for build commands."""
    vars_map = {}
    if api == "cuda" and includes_common_config:
        vars_map["CUDA_DIR"] = {
            "description": "Path to CUDA toolkit installation",
            "default": "/usr/local/cuda",
            "detection": "dirname $(dirname $(which nvcc))",
        }
    elif api == "opencl" and includes_common_config:
        vars_map["OPENCL_INC"] = {
            "description": "Path to OpenCL headers",
            "default": "/usr/local/cuda/include",
            "detection": "find /usr -name 'cl.h' -path '*/CL/cl.h' | head -1 | xargs dirname | xargs dirname",
        }
        vars_map["OPENCL_LIB"] = {
            "description": "Path to OpenCL libraries",
            "default": "/usr/local/cuda/lib64",
            "detection": "find /usr -name 'libOpenCL.so' | head -1 | xargs dirname",
        }
    return vars_map if vars_map else None


def map_verification_method(verification_method_str):
    """Map survey verification string to schema method."""
    vm = verification_method_str.lower()
    if "pass/fail" in vm or "golden reference" in vm or "result comparison" in vm:
        return "self_checking"
    if "numerical error" in vm:
        return "self_checking"
    return "self_checking"


def build_verification_strategies(verification_method_str):
    """Build verification strategies from the survey's verification string."""
    strategies = []
    vm = verification_method_str.lower()

    # Always check exit code
    strategies.append({
        "type": "exit_code",
        "expected": 0,
        "description": "Process exits cleanly",
    })

    if "pass/fail" in vm:
        strategies.append({
            "type": "stdout_pattern",
            "pattern": "(?i)(PASS|passed|SUCCESS|correct)",
            "description": "Output indicates correctness",
        })
    if "numerical error" in vm:
        strategies.append({
            "type": "numeric_comparison",
            "tolerance": 0.001,
            "tolerance_type": "relative",
            "description": "Numerical results within tolerance",
        })
    if "result comparison" in vm or "golden reference" in vm:
        strategies.append({
            "type": "stdout_pattern",
            "pattern": r"(?i)(pass|correct|match|verified)",
            "description": "Results match reference",
        })

    return strategies


def build_input_configs(kernel_name, api, kdata):
    """Build run input_configurations."""
    if not kdata:
        return {
            "default": {
                "arguments": [],
                "description": "Default run configuration",
                "input_files": [],
                "expected_results": None,
            }
        }

    data_files_sm = kdata["data_files"].get("correctness", [])
    data_files_lg = kdata["data_files"].get("performance", [])

    configs = {}

    # Correctness configuration (small)
    configs["correctness"] = {
        "arguments": kdata["args_sm"],
        "description": f"Small input for correctness verification of {kernel_name} ({api.upper()})",
        "input_files": data_files_sm,
        "expected_results": None,
    }

    # Performance configuration (large)
    if kdata["args_lg"] != kdata["args_sm"]:
        configs["performance"] = {
            "arguments": kdata["args_lg"],
            "description": f"Large input for performance measurement of {kernel_name} ({api.upper()})",
            "input_files": data_files_lg,
            "expected_results": None,
        }

    return configs


def make_spec(kernel_entry, api_key, api_info, kdata, commit):
    """Generate a single spec dict for one kernel-API variant."""
    kernel_name = kernel_entry["kernel_name"]
    slug = slugify(kernel_name)
    parallel_api = SUPPORTED_APIS[api_key]
    unique_id = f"rodinia-{slug}-{parallel_api}"

    # Files
    source_files = api_info.get("source_files", [])
    header_files = api_info.get("header_files", [])
    opencl_files = api_info.get("opencl_files", [])
    build_files_list = api_info.get("build_files", ["Makefile"])

    prompt_payload, support_files = select_files(
        api_key, source_files, header_files, opencl_files, build_files_list
    )

    # Fallback: if prompt_payload is empty, use all source files
    if not prompt_payload:
        prompt_payload = source_files[:] or opencl_files[:] or ["Makefile"]

    language = detect_language(source_files, api_key)
    language_std = "C++14" if language == "C++" else None

    # Build
    source_dir = api_info["source_dir"]
    working_dir = source_dir
    includes_common = api_info.get("includes_common_config", False)
    makefile_env_vars = api_info.get("makefile_env_vars", [])

    # Executable
    exe_path = "./placeholder"
    if kdata and api_key in kdata.get("exe", {}):
        exe_path = kdata["exe"][api_key]
    exe_base = exe_path.lstrip("./")

    # Hardware target
    hw_target = "cpu" if api_key == "omp" else "gpu"

    # Verification
    vm_str = api_info.get("verification_method", "prints results")
    verification_method = map_verification_method(vm_str)
    strategies = build_verification_strategies(vm_str)

    # Input configurations
    input_configs = build_input_configs(kernel_name, api_key, kdata)

    # Tags and metadata
    tags = (kdata["tags"] if kdata else []) + [kernel_name, api_key, "rodinia"]
    domain = kdata["domain"] if kdata else kernel_entry.get("estimated_complexity", "")
    description = (
        kdata["desc"] if kdata
        else f"{kernel_name} ({api_key.upper()}) from Rodinia benchmark suite"
    )
    complexity = kernel_entry.get("estimated_complexity", None)
    is_multifile = len(source_files) > 1

    spec = {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": slug,  # slugified: lowercase, no special chars
            "parallel_api": parallel_api,
            "unique_id": unique_id,
            "source_suite": "rodinia",
        },
        "provenance": {
            "repository": {
                "url": RODINIA_REPO_URL,
                "commit": commit,
                "branch": "master",
            },
            "repo_root": RODINIA_REPO_ROOT,
            "source_path": source_dir,
            "license": RODINIA_LICENSE,
        },
        "files": {
            "prompt_payload": prompt_payload,
            "support_files": support_files,
            "verification_only": [],
        },
        "implementation": {
            "api": parallel_api,
            "api_version": None,
            "language": language,
            "language_standard": language_std,
        },
        "build": {
            "environment": build_environment(api_key, includes_common, makefile_env_vars),
            "build_system": "make",
            "working_directory": working_dir,
            "commands": build_commands(api_key, includes_common, makefile_env_vars),
            "variables": build_variables(api_key, includes_common, makefile_env_vars),
            "outputs": {"executable": exe_path},
        },
        "run": {
            "executable": exe_path,
            "default_arguments": (kdata["args_sm"] if kdata else []),
            "timeout_seconds": 300,
            "environment_variables": None,
            "input_configurations": input_configs,
        },
        "verification": {
            "method": verification_method,
            "strategies": strategies,
            "floating_point": None,
        },
        "performance": None,
        "hardware": {
            "target": hw_target,
            "requirements": (
                {
                    "gpu": {
                        "vendor": "NVIDIA",
                        "min_compute_capability": "sm_60",
                        "min_memory_gb": 4,
                    }
                }
                if hw_target == "gpu"
                else {
                    "cpu": {
                        "min_cores": 4,
                        "min_memory_gb": 8,
                    }
                }
            ),
            "reference_platform": REFERENCE_PLATFORM,
        },
        "baseline_results": None,
        "metadata": {
            "description": description,
            "domain": domain,
            "complexity": complexity,
            "tags": sorted(set(tags)),
            "multi_file": is_multifile,
        },
    }
    return spec, unique_id


def make_manifest_entry(kernel_name, api_key, spec_filename, source_dir, domain):
    """Build one manifest.jsonl line."""
    category = DOMAIN_TO_CATEGORY.get(domain, "other")
    return {
        "kernel_name": kernel_name,
        "parallel_api": SUPPORTED_APIS[api_key],
        "source_suite": "rodinia",
        "category": category,
        "spec_file": f"specs/{spec_filename}",
        "source_dir": f"{RODINIA_REPO_ROOT}/{source_dir}",
    }


def load_existing_manifest_ids(manifest_path):
    """Return set of unique_ids already in manifest.jsonl."""
    ids = set()
    if not manifest_path.exists():
        return ids
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Build the unique_id from manifest fields
                uid = (
                    f"{entry.get('source_suite', '')}-"
                    f"{entry.get('kernel_name', '')}-"
                    f"{entry.get('parallel_api', '')}"
                )
                ids.add(uid)
            except json.JSONDecodeError:
                pass
    return ids


def generate_api_gaps_report(survey_data, generated):
    """Generate markdown API gaps report."""
    all_apis = ["cuda", "opencl", "omp"]
    kernels = survey_data["kernels"]

    lines = ["# Rodinia ParBench — API Gaps Report\n"]
    lines.append(f"**Generated**: 2026-03-03  ")
    lines.append(f"**Rodinia commit**: {RODINIA_COMMIT}\n")

    lines.append("## Summary\n")
    total_specs = len(generated)
    lines.append(f"- **Total specs generated**: {total_specs}")
    for api in all_apis:
        count = sum(1 for (k, a, _) in generated if a == api)
        lines.append(f"- **{api.upper()}** variants: {count}")
    lines.append("")

    lines.append("## API Coverage Matrix\n")
    header = "| Kernel | CUDA | OpenCL | OMP | HeCBench Overlap |"
    sep    = "|--------|------|--------|-----|-----------------|"
    lines.append(header)
    lines.append(sep)

    for entry in kernels:
        kname = entry["kernel_name"]
        apis = entry["apis"]
        row = f"| {kname} |"
        for api in all_apis:
            if api in apis and api in SUPPORTED_APIS:
                row += " ✅ |"
            else:
                row += " ❌ |"
        overlap = "⚠️ Yes" if entry.get("overlaps_hecbench") else ""
        row += f" {overlap} |"
        lines.append(row)
    lines.append("")

    lines.append("## Gap Analysis\n")
    lines.append("### Kernels missing OpenCL\n")
    for entry in kernels:
        kname = entry["kernel_name"]
        apis = entry["apis"]
        if "cuda" in apis and "opencl" not in apis:
            available = ", ".join(sorted(apis.keys()))
            lines.append(f"- **{kname}**: has {available}, missing OpenCL")
    lines.append("")

    lines.append("### Kernels missing OpenMP\n")
    for entry in kernels:
        kname = entry["kernel_name"]
        apis = entry["apis"]
        if "cuda" in apis and "omp" not in apis:
            available = ", ".join(sorted(apis.keys()))
            lines.append(f"- **{kname}**: has {available}, missing OpenMP")
    lines.append("")

    lines.append("### Skipped kernels (unsupported API only)\n")
    for entry in kernels:
        kname = entry["kernel_name"]
        apis = entry["apis"]
        supported = [a for a in apis if a in SUPPORTED_APIS]
        skipped = [a for a in apis if a not in SUPPORTED_APIS]
        if not supported and skipped:
            lines.append(f"- **{kname}**: only has {', '.join(skipped)} (not in schema)")
    lines.append("")

    lines.append("## HeCBench Overlaps\n")
    lines.append(
        "These Rodinia kernels overlap with existing HeCBench specs. "
        "Rodinia specs are distinct entries with different source paths.\n"
    )
    for entry in kernels:
        if entry.get("overlaps_hecbench"):
            apis = ", ".join(sorted(a for a in entry["apis"] if a in SUPPORTED_APIS))
            lines.append(f"- **{entry['kernel_name']}**: {apis}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Rodinia ParBench specs")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing spec files"
    )
    args = parser.parse_args()

    print(f"Reading survey: {SURVEY_JSON}")
    with open(SURVEY_JSON) as f:
        survey = json.load(f)

    kernels = survey["kernels"]
    commit = RODINIA_COMMIT

    existing_ids = load_existing_manifest_ids(MANIFEST_FILE)
    print(f"Existing manifest entries: {len(existing_ids)}")

    generated = []       # list of (kernel_name, api_key, spec_filename)
    skipped = []         # already exist
    new_manifest_lines = []

    for kernel_entry in kernels:
        kernel_name = kernel_entry["kernel_name"]
        slug = slugify(kernel_name)
        kdata = KERNEL_DATA.get(kernel_name)

        for api_key, api_info in kernel_entry["apis"].items():
            if api_key not in SUPPORTED_APIS:
                print(f"  SKIP {kernel_name}/{api_key}: not in supported APIs")
                continue

            parallel_api = SUPPORTED_APIS[api_key]
            unique_id = f"rodinia-{slug}-{parallel_api}"
            spec_filename = f"rodinia-{slug}-{parallel_api}.json"
            spec_path = SPECS_DIR / spec_filename

            # Check if already in manifest
            if unique_id in existing_ids:
                print(f"  SKIP {unique_id}: already in manifest")
                skipped.append(unique_id)
                continue

            # Check if spec file exists
            if spec_path.exists() and not args.force:
                print(f"  SKIP {spec_filename}: file exists (use --force to overwrite)")
                skipped.append(unique_id)
                continue

            # Generate spec
            spec, uid = make_spec(kernel_entry, api_key, api_info, kdata, commit)

            domain = kdata["domain"] if kdata else kernel_entry.get("estimated_complexity", "other")
            manifest_entry = make_manifest_entry(
                slug, api_key, spec_filename,
                api_info["source_dir"], domain
            )

            if args.dry_run:
                print(f"  [DRY RUN] Would write: {spec_path}")
                print(f"  [DRY RUN] Manifest: {json.dumps(manifest_entry)}")
            else:
                with open(spec_path, "w") as f:
                    json.dump(spec, f, indent=2)
                    f.write("\n")
                print(f"  WROTE {spec_path.name}")

            generated.append((kernel_name, api_key, spec_filename))
            new_manifest_lines.append(manifest_entry)

    # Append manifest entries
    if new_manifest_lines and not args.dry_run:
        with open(MANIFEST_FILE, "a") as f:
            for entry in new_manifest_lines:
                f.write(json.dumps(entry) + "\n")
        print(f"\nAppended {len(new_manifest_lines)} entries to manifest.jsonl")

    # Write API gaps report
    gaps_md = generate_api_gaps_report(survey, generated)
    if not args.dry_run:
        with open(GAPS_REPORT, "w") as f:
            f.write(gaps_md)
        print(f"Wrote API gaps report: {GAPS_REPORT}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Specs written:  {len(generated)}")
    print(f"Specs skipped:  {len(skipped)}")
    print(f"Manifest lines: {len(new_manifest_lines)}")
    if args.dry_run:
        print("(DRY RUN — nothing was written)")


if __name__ == "__main__":
    main()
