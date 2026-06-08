#!/usr/bin/env python3
"""
Survey the Rodinia GPU benchmark suite and produce structured output.
Catalogs every kernel-API variant with source files, build files, data files,
build system details, and verification methods.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

RODINIA_ROOT = Path("/home/samyak/Desktop/parbench_sam/rodinia")
OUTPUT_JSON = Path("/home/samyak/Desktop/parbench_sam/analysis/rodinia_survey.json")
OUTPUT_MD = Path("/home/samyak/Desktop/parbench_sam/analysis/rodinia_survey.md")

# Top-level API directories to scan
API_DIRS = {
    "cuda": "cuda",
    "opencl": "opencl",
    "omp": "openmp",
}

# Additional directories to check
OTHER_DIRS = ["others"]

# HeCBench overlaps
HECBENCH_OVERLAPS = {"nw", "lud", "pathfinder", "gaussian", "backprop", "nn", "knn"}

# Source file extensions
SOURCE_EXTS = {".cu", ".cpp", ".c", ".cc", ".cxx"}
HEADER_EXTS = {".h", ".hpp", ".hh", ".hxx", ".cuh"}
OPENCL_EXTS = {".cl"}
BUILD_FILES = {"Makefile", "makefile", "GNUmakefile", "CMakeLists.txt", "cmake", "build.sh", "compile.sh"}
DATA_EXTS = {".dat", ".txt", ".csv", ".graph", ".mtx", ".ppm", ".pgm", ".bmp", ".img", ".bin", ".input", ".matrix", ".data"}

# Known complexity descriptions
KERNEL_COMPLEXITY = {
    "backprop": "neural network training (backpropagation)",
    "bfs": "graph traversal (breadth-first search)",
    "b+tree": "B+ tree search and range queries",
    "cfd": "computational fluid dynamics (Euler solver)",
    "dwt2d": "2D discrete wavelet transform",
    "gaussian": "Gaussian elimination",
    "heartwall": "heart wall tracking (image processing)",
    "hotspot": "thermal simulation (2D stencil)",
    "hotspot3D": "thermal simulation (3D stencil)",
    "huffman": "Huffman encoding/decoding",
    "hybridsort": "hybrid sorting (merge + bucket sort)",
    "kmeans": "k-means clustering",
    "lavaMD": "molecular dynamics (Lennard-Jones potential)",
    "leukocyte": "leukocyte tracking (image processing)",
    "lud": "LU decomposition",
    "mummergpu": "DNA sequence alignment",
    "myocyte": "cardiac myocyte simulation (ODE solver)",
    "nn": "nearest neighbor search",
    "nw": "Needleman-Wunsch sequence alignment",
    "particlefilter": "particle filter (Bayesian estimation)",
    "pathfinder": "dynamic programming path finding",
    "srad": "speckle reducing anisotropic diffusion",
    "streamcluster": "online stream clustering",
    "rng": "random number generation",
}


def find_files_recursive(directory):
    """Recursively find all files in a directory."""
    files = []
    if not directory.exists():
        return files
    for root, dirs, filenames in os.walk(directory):
        # skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in filenames:
            files.append(Path(root) / f)
    return files


def categorize_files(base_dir):
    """Categorize all files in a kernel directory."""
    all_files = find_files_recursive(base_dir)
    
    source_files = []
    header_files = []
    opencl_files = []
    build_files = []
    data_files = []
    other_files = []
    
    for f in all_files:
        rel = f.relative_to(base_dir)
        ext = f.suffix.lower()
        name = f.name
        
        if ext in SOURCE_EXTS:
            source_files.append(str(rel))
        elif ext in HEADER_EXTS:
            header_files.append(str(rel))
        elif ext in OPENCL_EXTS:
            opencl_files.append(str(rel))
        elif name in BUILD_FILES or name.lower().endswith('.mk'):
            build_files.append(str(rel))
        elif ext in DATA_EXTS or 'data' in str(rel).lower():
            data_files.append(str(rel))
        else:
            other_files.append(str(rel))
    
    return {
        "source_files": sorted(source_files),
        "header_files": sorted(header_files),
        "opencl_files": sorted(opencl_files),
        "build_files": sorted(build_files),
        "data_files": sorted(data_files),
        "other_files": sorted(other_files),
    }


def detect_build_system(base_dir, build_files_list):
    """Detect the build system used."""
    systems = []
    for bf in build_files_list:
        name = Path(bf).name
        if name in ("Makefile", "makefile", "GNUmakefile") or name.endswith('.mk'):
            systems.append("make")
        elif name == "CMakeLists.txt":
            systems.append("cmake")
        elif name.endswith('.sh'):
            systems.append("shell script")
    systems = sorted(set(systems))
    return ", ".join(systems) if systems else "none detected"


def detect_verification(base_dir):
    """Check source files for verification patterns."""
    methods = set()
    all_files = find_files_recursive(base_dir)
    
    for f in all_files:
        ext = f.suffix.lower()
        if ext not in SOURCE_EXTS and ext not in HEADER_EXTS and ext not in OPENCL_EXTS:
            continue
        try:
            content = f.read_text(errors='ignore')
        except Exception:
            continue
        
        content_lower = content.lower()
        
        if re.search(r'\bpass\b.*\bfail\b|\bfail\b.*\bpass\b|"pass"|"fail"|passed|failed', content_lower):
            methods.add("pass/fail output")
        if re.search(r'compare|verify|validation|validate|check.*result|result.*check', content_lower):
            methods.add("result comparison")
        if re.search(r'rmse|rms_error|mean.?sq|l2.?norm|max.?error|abs.?diff|relative.?error', content_lower):
            methods.add("numerical error check")
        if re.search(r'golden|reference|expected|correct', content_lower):
            methods.add("golden reference")
        if re.search(r'printf|fprintf|cout|print.*result|output', content_lower):
            methods.add("prints results")
    
    if not methods:
        return "unknown"
    return "; ".join(sorted(methods))


def analyze_makefile(base_dir):
    """Analyze Makefile for environment variables and dependencies."""
    makefile_path = base_dir / "Makefile"
    if not makefile_path.exists():
        makefile_path = base_dir / "makefile"
    if not makefile_path.exists():
        return {"env_vars": [], "includes_common": False, "notes": "no Makefile found"}
    
    try:
        content = makefile_path.read_text(errors='ignore')
    except Exception:
        return {"env_vars": [], "includes_common": False, "notes": "Makefile unreadable"}
    
    env_vars = set()
    # Look for common env vars
    for var in ["CUDA_DIR", "CUDA_HOME", "CUDA_PATH", "OPENCL_DIR", "OPENCL_INC", 
                "OPENCL_LIB", "SDK_DIR", "CUDA_LIB_DIR", "NV_OPENCL_DIR"]:
        if var in content:
            env_vars.add(var)
    
    includes_common = "common/make.config" in content or "make.config" in content
    
    return {
        "env_vars": sorted(env_vars),
        "includes_common": includes_common,
        "notes": ""
    }


def check_shared_data(rodinia_root):
    """Check the shared data/ directory and any data download scripts."""
    data_dir = rodinia_root / "data"
    data_info = {
        "shared_data_dir_exists": data_dir.exists(),
        "shared_data_contents": [],
        "download_scripts": [],
    }
    
    if data_dir.exists():
        for item in sorted(data_dir.iterdir()):
            if item.is_dir():
                contents = [f.name for f in item.iterdir() if f.is_file()]
                data_info["shared_data_contents"].append({
                    "name": item.name,
                    "files": sorted(contents),
                    "is_dir": True
                })
            else:
                data_info["shared_data_contents"].append({
                    "name": item.name,
                    "is_dir": False
                })
    
    # Check for download scripts
    scripts_dir = rodinia_root / "scripts"
    if scripts_dir.exists():
        for f in scripts_dir.iterdir():
            data_info["download_scripts"].append(str(f.relative_to(rodinia_root)))
    
    # Check top-level for download scripts
    for f in rodinia_root.iterdir():
        if f.is_file() and ('download' in f.name.lower() or 'data' in f.name.lower()):
            if f.suffix in ('.sh', '.py', '.pl', ''):
                data_info["download_scripts"].append(str(f.relative_to(rodinia_root)))
    
    return data_info


def find_data_dependencies(base_dir, kernel_name, rodinia_root):
    """Find data file dependencies by checking source files and Makefiles."""
    deps = set()
    all_files = find_files_recursive(base_dir)
    
    for f in all_files:
        ext = f.suffix.lower()
        if ext not in SOURCE_EXTS and ext not in HEADER_EXTS and f.name not in BUILD_FILES:
            continue
        try:
            content = f.read_text(errors='ignore')
        except Exception:
            continue
        
        # Look for file open patterns
        for match in re.finditer(r'(?:fopen|open|ifstream|read_file)\s*\(\s*["\']([^"\']+)["\']', content):
            filename = match.group(1)
            if not filename.startswith('/dev/') and not filename.startswith('/proc/'):
                deps.add(filename)
        
        # Look for data directory references
        for match in re.finditer(r'(?:../../)?data/\S+', content):
            deps.add(match.group(0))
    
    # Check if there's a local data directory
    local_data = base_dir / "data"
    if local_data.exists() and local_data.is_dir():
        for f in find_files_recursive(local_data):
            deps.add(str(f.relative_to(base_dir)))
    
    # Check shared data dir
    shared_data = rodinia_root / "data" / kernel_name
    if shared_data.exists() and shared_data.is_dir():
        for f in find_files_recursive(shared_data):
            deps.add(f"data/{kernel_name}/{f.relative_to(shared_data)}")
    
    return sorted(deps)


def survey_kernel_api(kernel_name, api_name, api_dir_name, rodinia_root):
    """Survey a single kernel-API variant."""
    kernel_dir = rodinia_root / api_dir_name / kernel_name
    if not kernel_dir.exists():
        return None
    
    files = categorize_files(kernel_dir)
    
    # Check if it has actual source files
    has_sources = (len(files["source_files"]) > 0 or len(files["opencl_files"]) > 0)
    if not has_sources:
        return None
    
    makefile_info = analyze_makefile(kernel_dir)
    verification = detect_verification(kernel_dir)
    build_system = detect_build_system(kernel_dir, files["build_files"])
    
    # Data location
    data_location = ""
    local_data = kernel_dir / "data"
    shared_data = rodinia_root / "data" / kernel_name
    if local_data.exists():
        data_location = f"{api_dir_name}/{kernel_name}/data/"
    if shared_data.exists():
        if data_location:
            data_location += f"; data/{kernel_name}/"
        else:
            data_location = f"data/{kernel_name}/"
    
    # Also check if there are data files in the kernel dir itself
    if files["data_files"] and not data_location:
        data_location = f"{api_dir_name}/{kernel_name}/ (inline)"
    
    notes = []
    if makefile_info["includes_common"]:
        notes.append("includes common/make.config")
    if makefile_info["env_vars"]:
        notes.append(f"env vars: {', '.join(makefile_info['env_vars'])}")
    if makefile_info["notes"]:
        notes.append(makefile_info["notes"])
    
    # Check for subdirectory structure (some kernels like srad have v1/v2)
    subdirs = [d.name for d in kernel_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if subdirs:
        notes.append(f"subdirs: {', '.join(sorted(subdirs))}")
    
    return {
        "source_dir": f"{api_dir_name}/{kernel_name}",
        "source_files": files["source_files"],
        "header_files": files["header_files"],
        "opencl_files": files["opencl_files"],
        "build_files": files["build_files"],
        "data_files": files["data_files"],
        "data_location": data_location,
        "build_system": build_system,
        "verification_method": verification,
        "makefile_env_vars": makefile_info["env_vars"],
        "includes_common_config": makefile_info["includes_common"],
        "notes": "; ".join(notes) if notes else ""
    }


def main():
    print(f"Surveying Rodinia at: {RODINIA_ROOT}")
    
    # Discover all kernel names across all APIs
    all_kernels = set()
    api_kernel_map = defaultdict(set)
    
    for api_name, api_dir_name in API_DIRS.items():
        api_path = RODINIA_ROOT / api_dir_name
        if not api_path.exists():
            print(f"  WARNING: {api_path} does not exist")
            continue
        for item in api_path.iterdir():
            if item.is_dir() and item.name != "util" and not item.name.startswith('.'):
                all_kernels.add(item.name)
                api_kernel_map[api_name].add(item.name)
    
    # Check 'others' directory
    for other_dir in OTHER_DIRS:
        other_path = RODINIA_ROOT / other_dir
        if other_path.exists():
            for item in other_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    all_kernels.add(item.name)
                    api_kernel_map["others"].add(item.name)
    
    all_kernels = sorted(all_kernels)
    print(f"  Found {len(all_kernels)} unique kernels: {', '.join(all_kernels)}")
    
    # Check shared data
    shared_data_info = check_shared_data(RODINIA_ROOT)
    print(f"  Shared data dir exists: {shared_data_info['shared_data_dir_exists']}")
    print(f"  Download scripts: {shared_data_info['download_scripts']}")
    
    # Survey each kernel
    survey_results = []
    red_flags = []
    
    for kernel_name in all_kernels:
        print(f"\n  Surveying kernel: {kernel_name}")
        
        apis = {}
        
        # Check standard API directories
        for api_name, api_dir_name in API_DIRS.items():
            result = survey_kernel_api(kernel_name, api_name, api_dir_name, RODINIA_ROOT)
            if result:
                apis[api_name] = result
                print(f"    {api_name}: {len(result['source_files'])} source files")
                
                # Red flags
                if result["build_system"] == "none detected":
                    red_flags.append(f"{kernel_name}/{api_name}: no build file detected")
                if not result["source_files"] and not result["opencl_files"]:
                    red_flags.append(f"{kernel_name}/{api_name}: no source files found")
        
        # Check 'others'
        other_path = RODINIA_ROOT / "others" / kernel_name
        if other_path.exists():
            files = categorize_files(other_path)
            if files["source_files"]:
                apis["others"] = {
                    "source_dir": f"others/{kernel_name}",
                    "source_files": files["source_files"],
                    "header_files": files["header_files"],
                    "opencl_files": files["opencl_files"],
                    "build_files": files["build_files"],
                    "data_files": files["data_files"],
                    "data_location": "",
                    "build_system": detect_build_system(other_path, files["build_files"]),
                    "verification_method": detect_verification(other_path),
                    "makefile_env_vars": [],
                    "includes_common_config": False,
                    "notes": "in others/ directory"
                }
                print(f"    others: {len(files['source_files'])} source files")
        
        if not apis:
            print(f"    WARNING: No valid API variants found!")
            red_flags.append(f"{kernel_name}: no valid API variants found")
            continue
        
        # Find data dependencies
        data_deps = set()
        for api_name, api_info in apis.items():
            api_dir_name = API_DIRS.get(api_name, api_name)
            kernel_dir = RODINIA_ROOT / api_dir_name / kernel_name
            if kernel_dir.exists():
                deps = find_data_dependencies(kernel_dir, kernel_name, RODINIA_ROOT)
                data_deps.update(deps)
        
        entry = {
            "kernel_name": kernel_name,
            "apis": apis,
            "overlaps_hecbench": kernel_name in HECBENCH_OVERLAPS,
            "data_dependencies": sorted(data_deps),
            "estimated_complexity": KERNEL_COMPLEXITY.get(kernel_name, "unknown"),
        }
        
        survey_results.append(entry)
    
    # === Write JSON ===
    output = {
        "survey_metadata": {
            "rodinia_root": str(RODINIA_ROOT),
            "total_kernels": len(survey_results),
            "total_specs": sum(len(k["apis"]) for k in survey_results),
            "api_counts": {api: len(kernels) for api, kernels in api_kernel_map.items()},
            "hecbench_overlaps": sorted(HECBENCH_OVERLAPS & set(k["kernel_name"] for k in survey_results)),
            "shared_data_info": shared_data_info,
            "red_flags": red_flags,
        },
        "kernels": survey_results,
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote JSON to: {OUTPUT_JSON}")
    
    # === Write Markdown ===
    md_lines = []
    md_lines.append("# Rodinia GPU Benchmark Suite — Survey Report\n")
    md_lines.append(f"**Date**: 2026-03-03  ")
    md_lines.append(f"**Source**: <https://github.com/yuhc/gpu-rodinia>  ")
    md_lines.append(f"**Rodinia root**: `{RODINIA_ROOT}`\n")
    
    md_lines.append("## Summary\n")
    md_lines.append(f"- **Total unique kernels**: {len(survey_results)}")
    md_lines.append(f"- **Total kernel-API specs**: {sum(len(k['apis']) for k in survey_results)}")
    for api_name in sorted(api_kernel_map):
        md_lines.append(f"- **{api_name}** variants: {len(api_kernel_map[api_name])}")
    md_lines.append(f"- **HeCBench overlaps**: {len(HECBENCH_OVERLAPS & set(k['kernel_name'] for k in survey_results))}")
    md_lines.append("")
    
    # API Coverage Matrix
    md_lines.append("## API Coverage Matrix\n")
    all_apis = ["cuda", "opencl", "omp", "others"]
    header = "| Kernel | " + " | ".join(all_apis) + " | HeCBench Overlap |"
    sep = "|--------|" + "|".join(["-------"] * len(all_apis)) + "|------------------|"
    md_lines.append(header)
    md_lines.append(sep)
    
    for entry in survey_results:
        row = f"| {entry['kernel_name']} |"
        for api in all_apis:
            if api in entry['apis']:
                n_src = len(entry['apis'][api]['source_files'])
                row += f" ✅ ({n_src} src) |"
            else:
                row += " ❌ |"
        overlap = "⚠️ YES" if entry['overlaps_hecbench'] else ""
        row += f" {overlap} |"
        md_lines.append(row)
    
    md_lines.append("")
    
    # HeCBench Overlaps
    md_lines.append("## HeCBench Overlaps\n")
    md_lines.append("These kernels exist in both Rodinia and the existing HeCBench selection:")
    md_lines.append("")
    for entry in survey_results:
        if entry["overlaps_hecbench"]:
            apis = ", ".join(sorted(entry["apis"].keys()))
            md_lines.append(f"- **{entry['kernel_name']}**: Rodinia APIs = {apis}")
    md_lines.append("")
    
    # Data Dependencies
    md_lines.append("## Data Dependencies\n")
    md_lines.append("### Shared Data Directory\n")
    if shared_data_info["shared_data_dir_exists"]:
        if shared_data_info["shared_data_contents"]:
            for item in shared_data_info["shared_data_contents"]:
                if item["is_dir"]:
                    md_lines.append(f"- `data/{item['name']}/`: {len(item['files'])} files")
                else:
                    md_lines.append(f"- `data/{item['name']}`")
        else:
            md_lines.append("Shared data directory exists but is **empty**.")
            md_lines.append("Data likely needs to be downloaded separately.")
    else:
        md_lines.append("No shared data directory found.")
    md_lines.append("")
    
    md_lines.append("### Download Scripts\n")
    if shared_data_info["download_scripts"]:
        for s in shared_data_info["download_scripts"]:
            md_lines.append(f"- `{s}`")
    else:
        md_lines.append("No download scripts found.")
    md_lines.append("")
    
    md_lines.append("### Per-Kernel Data Files\n")
    for entry in survey_results:
        if entry["data_dependencies"]:
            md_lines.append(f"- **{entry['kernel_name']}**: {', '.join(entry['data_dependencies'][:5])}")
            if len(entry["data_dependencies"]) > 5:
                md_lines.append(f"  ...and {len(entry['data_dependencies']) - 5} more")
    md_lines.append("")
    
    # Build System Details
    md_lines.append("## Build System Details\n")
    md_lines.append("All kernels use `make` with a shared `common/make.config` that defines:")
    md_lines.append("- `CUDA_DIR` (default `/usr/local/cuda`)")
    md_lines.append("- `CUDA_LIB_DIR`")
    md_lines.append("- `SDK_DIR`")
    md_lines.append("- `OPENCL_DIR`, `OPENCL_INC`, `OPENCL_LIB`")
    md_lines.append("")
    
    # Per-kernel build info
    md_lines.append("### Per-Kernel Build Notes\n")
    for entry in survey_results:
        notes = []
        for api_name, api_info in entry["apis"].items():
            if api_info["notes"]:
                notes.append(f"  - {api_name}: {api_info['notes']}")
        if notes:
            md_lines.append(f"- **{entry['kernel_name']}**")
            for n in notes:
                md_lines.append(n)
    md_lines.append("")
    
    # Red Flags
    md_lines.append("## Red Flags\n")
    if red_flags:
        for flag in red_flags:
            md_lines.append(f"- ⚠️ {flag}")
    else:
        md_lines.append("No red flags detected.")
    md_lines.append("")
    
    # Detailed Kernel Catalog
    md_lines.append("## Detailed Kernel Catalog\n")
    for entry in survey_results:
        md_lines.append(f"### {entry['kernel_name']}\n")
        md_lines.append(f"- **Complexity**: {entry['estimated_complexity']}")
        md_lines.append(f"- **HeCBench overlap**: {'YES' if entry['overlaps_hecbench'] else 'No'}")
        md_lines.append(f"- **APIs**: {', '.join(sorted(entry['apis'].keys()))}")
        md_lines.append("")
        
        for api_name in sorted(entry["apis"].keys()):
            api = entry["apis"][api_name]
            md_lines.append(f"**{api_name}** (`{api['source_dir']}`)")
            md_lines.append(f"- Sources: {', '.join(api['source_files']) if api['source_files'] else 'none'}")
            if api['header_files']:
                md_lines.append(f"- Headers: {', '.join(api['header_files'])}")
            if api['opencl_files']:
                md_lines.append(f"- OpenCL kernels: {', '.join(api['opencl_files'])}")
            md_lines.append(f"- Build: {api['build_system']} ({', '.join(api['build_files']) if api['build_files'] else 'none'})")
            if api['data_files']:
                md_lines.append(f"- Data: {', '.join(api['data_files'])}")
            md_lines.append(f"- Verification: {api['verification_method']}")
            if api['notes']:
                md_lines.append(f"- Notes: {api['notes']}")
            md_lines.append("")
    
    with open(OUTPUT_MD, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Wrote Markdown to: {OUTPUT_MD}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SURVEY COMPLETE")
    print(f"{'='*60}")
    print(f"Unique kernels: {len(survey_results)}")
    print(f"Total specs:    {sum(len(k['apis']) for k in survey_results)}")
    for api_name in sorted(api_kernel_map):
        print(f"  {api_name}: {len(api_kernel_map[api_name])} kernels")
    print(f"HeCBench overlaps: {sorted(HECBENCH_OVERLAPS & set(k['kernel_name'] for k in survey_results))}")
    print(f"Red flags: {len(red_flags)}")
    for flag in red_flags:
        print(f"  - {flag}")


if __name__ == "__main__":
    main()
