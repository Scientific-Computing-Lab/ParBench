#!/usr/bin/env python3
"""Fix Rodinia spec files for this machine's CUDA/OpenCL paths and Makefile targets.

Changes applied:
  A. CUDA specs (15): /usr/local/cuda → /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda
  B. OpenCL specs (18): OPENCL_INC/LIB → HPC SDK paths
  C. CLHelper OpenCL specs (2): add CC_FLAGS='-g -O3 -std=c++14'
  D. OMP specs (4): fix bare `make` to specific make targets

Usage:
    cd /home/samyak/Desktop/parbench_sam
    source env_parbench/bin/activate
    python3 scripts/fix_rodinia_paths.py [--dry-run]
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/samyak/Desktop/parbench_sam")
SPECS_DIR = PROJECT_ROOT / "specs"

OLD_CUDA = "/usr/local/cuda"
NEW_CUDA = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda"
OLD_OCL_INC = "/usr/local/cuda/include"
NEW_OCL_INC = f"{NEW_CUDA}/include"
OLD_OCL_LIB = "/usr/local/cuda/lib64"
NEW_OCL_LIB = f"{NEW_CUDA}/lib64"

DRY_RUN = "--dry-run" in sys.argv

def load(path):
    with open(path) as f:
        return json.load(f)

def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

def fix_cuda_spec(path):
    """Fix CUDA_DIR path in build command and variable default."""
    data = load(path)
    changed = False

    build_cmd = data["build"]["commands"]["build"]
    if OLD_CUDA in build_cmd:
        data["build"]["commands"]["build"] = build_cmd.replace(OLD_CUDA, NEW_CUDA)
        changed = True

    if data["build"].get("variables") and "CUDA_DIR" in data["build"]["variables"]:
        var = data["build"]["variables"]["CUDA_DIR"]
        if var.get("default") == OLD_CUDA:
            var["default"] = NEW_CUDA
            changed = True

    return data, changed

def fix_opencl_spec(path):
    """Fix OPENCL_INC and OPENCL_LIB paths in build command and variable defaults."""
    data = load(path)
    changed = False

    build_cmd = data["build"]["commands"]["build"]
    new_cmd = build_cmd.replace(OLD_OCL_INC, NEW_OCL_INC).replace(OLD_OCL_LIB, NEW_OCL_LIB)
    if new_cmd != build_cmd:
        data["build"]["commands"]["build"] = new_cmd
        changed = True

    if data["build"].get("variables"):
        for var_name, var in data["build"]["variables"].items():
            old_default = var.get("default", "")
            new_default = old_default.replace(OLD_OCL_INC, NEW_OCL_INC).replace(OLD_OCL_LIB, NEW_OCL_LIB)
            if new_default != old_default:
                var["default"] = new_default
                changed = True

    return data, changed

def add_cxx14_flag(data):
    """Append CC_FLAGS='-g -O3 -std=c++14' to build command if not present."""
    build_cmd = data["build"]["commands"]["build"]
    flag = "CC_FLAGS='-g -O3 -std=c++14'"
    if flag not in build_cmd:
        data["build"]["commands"]["build"] = build_cmd + f" {flag}"
        return True
    return False

def fix_omp_target(data, target):
    """Replace bare `make` with `make <target>`."""
    build_cmd = data["build"]["commands"]["build"]
    if build_cmd.strip() == "make":
        data["build"]["commands"]["build"] = f"make {target}"
        return True
    return False


def main():
    if DRY_RUN:
        print("[DRY RUN] No files will be written.\n")

    # ── A. CUDA path fixes ──────────────────────────────────────────────────
    cuda_specs = sorted(SPECS_DIR.glob("rodinia-*-cuda.json"))
    already_fixed_cuda = {"rodinia-bfs-cuda.json"}

    print("=== A. CUDA path fixes ===")
    for path in cuda_specs:
        if path.name in already_fixed_cuda:
            print(f"  SKIP (already fixed): {path.name}")
            continue
        data, changed = fix_cuda_spec(path)
        if changed:
            print(f"  FIX: {path.name}")
            if not DRY_RUN:
                save(path, data)
        else:
            print(f"  OK  (no change needed): {path.name}")

    # ── B. OpenCL path fixes ────────────────────────────────────────────────
    opencl_specs = sorted(SPECS_DIR.glob("rodinia-*-opencl.json"))
    already_fixed_opencl = {"rodinia-bfs-opencl.json"}
    skip_opencl = {"rodinia-lud-opencl.json"}  # uses bare make, no OPENCL_INC/LIB

    print("\n=== B. OpenCL path fixes ===")
    for path in opencl_specs:
        if path.name in already_fixed_opencl or path.name in skip_opencl:
            print(f"  SKIP: {path.name}")
            continue
        data, changed = fix_opencl_spec(path)
        if changed:
            print(f"  FIX: {path.name}")
            if not DRY_RUN:
                save(path, data)
        else:
            print(f"  OK  (no change needed): {path.name}")

    # ── C. OpenCL C++14 fix (CLHelper.h users) ─────────────────────────────
    clhelper_specs = [
        "rodinia-cfd-opencl.json",
        "rodinia-streamcluster-opencl.json",
    ]
    print("\n=== C. OpenCL C++14 CC_FLAGS fix ===")
    for name in clhelper_specs:
        path = SPECS_DIR / name
        data = load(path)
        # First apply OpenCL path fixes if not already done
        data, _ = fix_opencl_spec(path)  # re-load and fix
        changed = add_cxx14_flag(data)
        if changed:
            print(f"  FIX: {name}")
            if not DRY_RUN:
                save(path, data)
        else:
            print(f"  OK  (CC_FLAGS already present): {name}")

    # ── D. OMP Makefile target fixes ───────────────────────────────────────
    omp_targets = {
        "rodinia-bfs-omp.json":   "bfs",
        "rodinia-nw-omp.json":    "needle",
        "rodinia-cfd-omp.json":   "euler3d_cpu",
        "rodinia-lud-omp.json":   "lud_omp",
    }
    print("\n=== D. OMP Makefile target fixes ===")
    for name, target in omp_targets.items():
        path = SPECS_DIR / name
        data = load(path)
        changed = fix_omp_target(data, target)
        if changed:
            print(f"  FIX: {name} → make {target}")
            if not DRY_RUN:
                save(path, data)
        else:
            print(f"  OK  (already has target): {name}")

    print("\nDone." if not DRY_RUN else "\n[DRY RUN] Done. No files written.")


if __name__ == "__main__":
    main()
