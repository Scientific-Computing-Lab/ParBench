#!/usr/bin/env python3
"""
Deep-inspect 25 HeCBench kernels (20 primary + 5 backup) for ParBench Phase 2.
Produces a detailed JSON report for each kernel covering:
  - CUDA source files (classified by role)
  - Verification method
  - Run configuration (executable, args, problem sizes)
  - External data dependencies
  - OMP build notes
"""

import json
import os
import re
import sys
from pathlib import Path

HECBENCH_SRC = Path("/Users/samyakjhaveri/Downloads/HeCBench-master/src")
OUTPUT = Path(
    "/Users/samyakjhaveri/Desktop/parbench_sam/analysis/kernel_deep_inspection.json"
)

KERNELS = [
    # (name, domain, status)
    ("pathfinder", "graph", "primary"),
    ("b+tree", "graph", "primary"),
    ("heartwall", "bio/medical", "primary"),
    ("myocyte", "bio/medical", "primary"),
    ("geglu", "ML", "primary"),
    ("perplexity", "ML", "primary"),
    ("iso2dfd", "stencil", "primary"),
    ("heat2d", "stencil", "primary"),
    ("srad", "medical_imaging", "primary"),
    ("cfd", "iterative_solver", "primary"),
    ("crc64", "hashing", "primary"),
    ("jenkins-hash", "hashing", "primary"),
    ("gaussian", "linear_algebra", "primary"),
    ("triad", "memory_bandwidth", "primary"),
    ("popcount", "bit_manipulation", "primary"),
    ("sobol", "quasi_random", "primary"),
    ("convolution3D", "3D_processing", "primary"),
    ("mandelbrot", "fractal_math", "primary"),
    ("mis", "graph_combinatorial", "primary"),
    ("bezier-surface", "geometric_3D", "primary"),
    ("murmurhash3", "hashing", "backup"),
    ("stencil1d", "stencil", "backup"),
    ("nw", "bioinformatics", "backup"),
    ("knn", "ML", "backup"),
    ("floydwarshall", "graph", "backup"),
]

SOURCE_EXTS = {".cu", ".cpp", ".c", ".h", ".hpp"}
VERIFY_PATTERNS = {
    "PASS": re.compile(r"\bPASS(?:ED)?\b", re.IGNORECASE),
    "FAIL": re.compile(r"\bFAIL(?:ED)?\b", re.IGNORECASE),
    "verify": re.compile(r"\bverif(?:y|ied|ication)\b", re.IGNORECASE),
    "correct": re.compile(r"\bcorrect\b", re.IGNORECASE),
    "mismatch": re.compile(r"\bmismatch\b", re.IGNORECASE),
    "memcmp": re.compile(r"\bmemcmp\b", re.IGNORECASE),
    "error": re.compile(r"\berror\b", re.IGNORECASE),
}

REFERENCE_PATTERNS = re.compile(
    r"reference|_host\.|_cpu\.|_ref\.|_gold\.|sequential|serial", re.IGNORECASE
)


def classify_source_file(filepath: Path, all_text: str) -> str:
    """Classify a source file by role."""
    name = filepath.name.lower()
    stem = filepath.stem.lower()

    if name in ("makefile", "cmakelists.txt"):
        return "support_files"

    if any(
        kw in stem
        for kw in [
            "reference",
            "_host",
            "_cpu",
            "_ref",
            "_gold",
            "serial",
            "sequential",
        ]
    ):
        return "verification_only"

    if filepath.suffix in {".h", ".hpp"}:
        # Check if it contains kernel logic (__global__, __device__) or just declarations
        if re.search(r"__global__|__device__|__kernel__", all_text):
            return "prompt_payload"
        return "support_files"

    # .cu, .cpp, .c files with kernel logic
    if re.search(r"__global__|__device__|__kernel__|int\s+main|void\s+main", all_text):
        return "prompt_payload"

    return "support_files"


def extract_verification_method(cuda_dir: Path) -> dict:
    """Analyze source files to determine verification method."""
    result = {
        "method": "unknown",
        "patterns_found": [],
        "verification_strings": [],
    }

    for f in sorted(cuda_dir.iterdir()):
        if f.is_file() and f.suffix in SOURCE_EXTS:
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue

            for pname, pat in VERIFY_PATTERNS.items():
                matches = pat.findall(text)
                if matches:
                    result["patterns_found"].append(pname)
                    # Extract the actual print context
                    for m in re.finditer(
                        r"(printf|cout|fprintf|puts).*(" + pat.pattern + r").*", text
                    ):
                        line = m.group(0).strip()[:120]
                        if line not in result["verification_strings"]:
                            result["verification_strings"].append(line)

    result["patterns_found"] = list(set(result["patterns_found"]))

    if "PASS" in result["patterns_found"] or "FAIL" in result["patterns_found"]:
        result["method"] = "stdout_pattern"
    elif "verify" in result["patterns_found"] or "correct" in result["patterns_found"]:
        result["method"] = "stdout_pattern"
    elif "memcmp" in result["patterns_found"] or "mismatch" in result["patterns_found"]:
        result["method"] = "stdout_pattern"
    elif "error" in result["patterns_found"]:
        result["method"] = "stdout_pattern"

    return result


def extract_makefile_info(makefile_path: Path) -> dict:
    """Extract build target, run command, and data dependencies from Makefile."""
    result = {
        "executable": None,
        "run_command": None,
        "run_args": None,
        "data_dependencies": [],
        "compiler": None,
        "makefile_exists": False,
    }

    if not makefile_path.is_file():
        return result

    result["makefile_exists"] = True
    try:
        text = makefile_path.read_text(errors="ignore")
    except Exception:
        return result

    # Find executable name
    # Common patterns: $(CC) ... -o <name>, or target line
    exe_match = re.search(r"-o\s+(\S+)", text)
    if exe_match:
        result["executable"] = exe_match.group(1)

    # Find run target
    run_match = re.search(r"^run:.*\n((?:\t.*\n)*)", text, re.MULTILINE)
    if run_match:
        run_lines = run_match.group(1).strip()
        result["run_command"] = run_lines[:200]

        # Extract args from run command
        exe_run = re.search(r"\./(\S+)\s*(.*)", run_lines)
        if exe_run:
            if not result["executable"]:
                result["executable"] = exe_run.group(1)
            result["run_args"] = (
                exe_run.group(2).strip()[:200] if exe_run.group(2).strip() else None
            )

    # Check for data dependencies (../data, input files, etc.)
    data_deps = re.findall(r"(\.\./[^\s)]+|/data/[^\s)]+|\$\(DATA\)[^\s)]*)", text)
    if data_deps:
        result["data_dependencies"] = list(set(data_deps))

    # Check for compiler used
    if "nvcc" in text:
        result["compiler"] = "nvcc"

    return result


def extract_omp_makefile_info(omp_dir: Path) -> dict:
    """Quick inspection of OMP variant's Makefile."""
    result = {
        "makefile_exists": False,
        "uses_icpx": False,
        "uses_gcc": False,
        "uses_clang": False,
        "compiler_line": None,
        "has_aomp_makefile": False,
        "openmp_flags": None,
    }

    makefile = omp_dir / "Makefile"
    if not makefile.is_file():
        return result

    result["makefile_exists"] = True
    try:
        text = makefile.read_text(errors="ignore")
    except Exception:
        return result

    if "icpx" in text or "ICPX" in text:
        result["uses_icpx"] = True
    if "gcc" in text.lower() or "g++" in text:
        result["uses_gcc"] = True
    if "clang" in text:
        result["uses_clang"] = True

    # Find compiler definition line
    cc_match = re.search(r"^(CC\s*[:?]?=\s*.*)$", text, re.MULTILINE)
    if cc_match:
        result["compiler_line"] = cc_match.group(1).strip()

    # Check for -fopenmp
    omp_match = re.search(r"(-fopenmp\S*|-qopenmp|-mp)", text)
    if omp_match:
        result["openmp_flags"] = omp_match.group(0)

    # Check for Makefile.aomp or aomp variant
    if (omp_dir / "Makefile.aomp").is_file():
        result["has_aomp_makefile"] = True

    return result


def inspect_kernel(name: str, domain: str, status: str) -> dict:
    """Full inspection of a single kernel."""
    cuda_dir = HECBENCH_SRC / f"{name}-cuda"
    omp_dir = HECBENCH_SRC / f"{name}-omp"

    info = {
        "kernel": name,
        "domain": domain,
        "status": status,
        "cuda_dir_exists": cuda_dir.is_dir(),
        "omp_dir_exists": omp_dir.is_dir(),
    }

    if not cuda_dir.is_dir():
        info["error"] = f"CUDA directory not found: {cuda_dir}"
        return info

    # List and classify source files
    source_files = {}
    for f in sorted(cuda_dir.iterdir()):
        if f.is_file() and f.suffix in SOURCE_EXTS:
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                text = ""
            role = classify_source_file(f, text)
            source_files[f.name] = {
                "role": role,
                "lines": text.count("\n") + 1,
                "size_bytes": f.stat().st_size,
            }

    info["cuda_source_files"] = source_files
    info["cuda_file_count"] = len(source_files)
    info["prompt_payload_files"] = [
        fn for fn, meta in source_files.items() if meta["role"] == "prompt_payload"
    ]
    info["support_files"] = [
        fn for fn, meta in source_files.items() if meta["role"] == "support_files"
    ]
    info["verification_only_files"] = [
        fn for fn, meta in source_files.items() if meta["role"] == "verification_only"
    ]

    # Verification method
    info["verification"] = extract_verification_method(cuda_dir)

    # Makefile analysis
    makefile_path = cuda_dir / "Makefile"
    if not makefile_path.is_file():
        makefile_path = cuda_dir / "makefile"
    info["makefile_info"] = extract_makefile_info(makefile_path)

    # Check for external data
    needs_external = bool(info["makefile_info"]["data_dependencies"])
    # Also scan source for file open patterns
    for f in sorted(cuda_dir.iterdir()):
        if f.is_file() and f.suffix in SOURCE_EXTS:
            try:
                text = f.read_text(errors="ignore")
                if re.search(r"fopen|ifstream|open\s*\(", text):
                    # Check if it's opening external data files
                    file_opens = re.findall(r'(?:fopen|open)\s*\(\s*"([^"]+)"', text)
                    file_opens += re.findall(r"(?:fopen|open)\s*\(\s*argv\[", text)
                    if file_opens:
                        info.setdefault("file_io_patterns", []).extend(
                            [
                                fo[:80]
                                for fo in file_opens
                                if fo not in info.get("file_io_patterns", [])
                            ]
                        )
                        if any(
                            ".dat" in fo or ".txt" in fo or ".bin" in fo or "/" in fo
                            for fo in file_opens
                        ):
                            needs_external = True
            except Exception:
                pass

    info["needs_external_data"] = needs_external

    # OMP variant inspection
    info["omp_info"] = extract_omp_makefile_info(omp_dir)

    return info


def main():
    print(f"Inspecting {len(KERNELS)} kernels...\n")
    results = []
    for name, domain, status in KERNELS:
        print(f"  Inspecting: {name} ({domain}) ...", end="", flush=True)
        info = inspect_kernel(name, domain, status)
        results.append(info)
        print(
            f"  {info.get('cuda_file_count', '?')} files, verify={info.get('verification', {}).get('method', '?')}"
        )

    # Save detailed JSON
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed inspection saved to {OUTPUT}")

    # Print summary table
    print("\n" + "=" * 140)
    print(
        f"{'#':>2} {'Kernel':<20} {'Domain':<20} {'Status':<8} {'Files':>5} {'Verify':<16} {'ExtData':<8} {'OMP Compiler':<20} {'Conf':<6}"
    )
    print("-" * 140)

    for i, info in enumerate(results, 1):
        name = info["kernel"]
        domain = info["domain"]
        st = info["status"]
        nfiles = info.get("cuda_file_count", "?")
        verify = info.get("verification", {}).get("method", "?")
        ext_data = "YES" if info.get("needs_external_data") else "no"

        omp = info.get("omp_info", {})
        if omp.get("uses_icpx"):
            omp_comp = "icpx (Intel)"
        elif omp.get("uses_gcc"):
            omp_comp = "gcc/g++"
        elif omp.get("uses_clang"):
            omp_comp = "clang"
        else:
            omp_comp = (
                omp.get("compiler_line", "unknown")[:20]
                if omp.get("compiler_line")
                else "unknown"
            )

        # Confidence
        confidence = "High"
        if ext_data == "YES":
            confidence = "Medium"
        if nfiles and isinstance(nfiles, int) and nfiles > 10:
            confidence = "Low" if confidence == "Medium" else "Medium"
        if verify == "unknown":
            confidence = "Low"

        print(
            f"{i:>2} {name:<20} {domain:<20} {st:<8} {nfiles:>5} {verify:<16} {ext_data:<8} {omp_comp:<20} {confidence:<6}"
        )

    print("=" * 140)

    # Print detailed findings for each kernel
    print("\n\n" + "=" * 80)
    print("DETAILED FINDINGS PER KERNEL")
    print("=" * 80)

    for info in results:
        name = info["kernel"]
        print(f"\n--- {name} ({info['domain']}) [{info['status']}] ---")
        print(f"  CUDA files ({info.get('cuda_file_count', 0)}):")
        for fn, meta in info.get("cuda_source_files", {}).items():
            print(f"    {fn:<30} role={meta['role']:<20} lines={meta['lines']}")

        print(f"  Prompt payload: {info.get('prompt_payload_files', [])}")
        print(f"  Support files:  {info.get('support_files', [])}")
        print(f"  Verify-only:    {info.get('verification_only_files', [])}")

        v = info.get("verification", {})
        print(
            f"  Verification:   method={v.get('method')} patterns={v.get('patterns_found')}"
        )
        for vs in v.get("verification_strings", [])[:5]:
            print(f"    → {vs}")

        mk = info.get("makefile_info", {})
        print(f"  Executable:     {mk.get('executable')}")
        print(f"  Run command:    {mk.get('run_command', 'N/A')[:100]}")
        print(f"  Run args:       {mk.get('run_args')}")
        print(f"  Data deps:      {mk.get('data_dependencies')}")
        print(f"  External data:  {info.get('needs_external_data')}")

        if info.get("file_io_patterns"):
            print(f"  File I/O:       {info['file_io_patterns'][:5]}")

        omp = info.get("omp_info", {})
        print(f"  OMP compiler:   {omp.get('compiler_line', 'N/A')}")
        print(f"  OMP flags:      {omp.get('openmp_flags', 'N/A')}")
        if omp.get("uses_icpx"):
            print(f"  ⚠ OMP uses icpx — may need Makefile modification for g++")


if __name__ == "__main__":
    main()
