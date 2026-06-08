#!/usr/bin/env python3
"""Fix Rodinia spec files with wrong executable paths and run arguments.

Fixes applied after batch run analysis:
  A. Build command fixes
  B. Wrong executable paths (build.outputs.executable + run.executable)
  C. Run argument fixes

Usage:
    cd /home/samyak/Desktop/parbench_sam
    source env_parbench/bin/activate
    python3 scripts/fix_rodinia_run_args.py [--dry-run]
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/samyak/Desktop/parbench_sam")
SPECS_DIR = PROJECT_ROOT / "specs"
DRY_RUN = "--dry-run" in sys.argv


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def fix_exe(data, old, new):
    """Fix both build.outputs.executable and run.executable."""
    changed = False
    if data["build"]["outputs"]["executable"] == old:
        data["build"]["outputs"]["executable"] = new
        changed = True
    if data["run"]["executable"] == old:
        data["run"]["executable"] = new
        changed = True
    return changed


def fix_run_args_all_configs(data, new_args, new_files=None):
    """Replace arguments in ALL run configurations."""
    changed = False
    for cfg_name, cfg in data["run"]["input_configurations"].items():
        if cfg.get("arguments") != new_args:
            cfg["arguments"] = new_args
            changed = True
        if new_files is not None and cfg.get("input_files") != new_files:
            cfg["input_files"] = new_files
            changed = True
    if data["run"].get("default_arguments") != new_args:
        data["run"]["default_arguments"] = new_args
        changed = True
    return changed


def apply(name, mutate_fn):
    path = SPECS_DIR / name
    data = load(path)
    changed = mutate_fn(data)
    if changed:
        print(f"  FIX: {name}")
        if not DRY_RUN:
            save(path, data)
    else:
        print(f"  OK  (no change): {name}")


# ── A. Build command fixes ─────────────────────────────────────────────────

print("=== A. Build command fixes ===")

# backprop-cuda: add CUDA_LIB_DIR to fix -lcudart linking
def fix_backprop_cuda(d):
    cmd = d["build"]["commands"]["build"]
    lib_flag = "CUDA_LIB_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/lib64"
    if lib_flag not in cmd:
        d["build"]["commands"]["build"] = cmd + f" {lib_flag}"
        return True
    return False

apply("rodinia-backprop-cuda.json", fix_backprop_cuda)

# cfd-opencl and streamcluster-opencl: replace CC_FLAGS with FLAGS for C++14
for name in ["rodinia-cfd-opencl.json", "rodinia-streamcluster-opencl.json"]:
    def fix_flags(d, _name=name):
        cmd = d["build"]["commands"]["build"]
        old_flag = "CC_FLAGS='-g -O3 -std=c++14'"
        new_flag = "FLAGS='-O3 -std=c++14'"
        if old_flag in cmd:
            d["build"]["commands"]["build"] = cmd.replace(old_flag, new_flag)
            return True
        if new_flag not in cmd:
            d["build"]["commands"]["build"] = cmd + f" {new_flag}"
            return True
        return False
    apply(name, fix_flags)

# ── B. Executable path fixes ──────────────────────────────────────────────

print("\n=== B. Executable path fixes ===")

exe_fixes = {
    # OpenCL: .out suffix
    "rodinia-backprop-opencl.json":   ("./backprop",      "./backprop.out"),
    "rodinia-gaussian-opencl.json":   ("./gaussian",      "./gaussian.out"),
    "rodinia-hotspot-opencl.json":    ("./hotspot",       "./hotspot.out"),
    "rodinia-kmeans-opencl.json":     ("./kmeans",        "./kmeans.out"),
    "rodinia-nn-opencl.json":         ("./nn",            "./nn.out"),
    "rodinia-srad-opencl.json":       ("./srad",          "./srad.out"),
    "rodinia-dwt2d-opencl.json":      ("./dwt2d",         "./dwt2d.out"),
    "rodinia-nw-opencl.json":         ("./needle",        "./nw.out"),
    "rodinia-lud-opencl.json":        ("./ocl/lud",       "./lud.out"),
    # OMP: wrong executable names
    "rodinia-lud-omp.json":           ("./omp/lud",       "./omp/lud_omp"),
    "rodinia-particlefilter-omp.json":("./particlefilter","./particle_filter"),
    "rodinia-myocyte-omp.json":       ("./myocyte",       "./myocyte.out"),
}

for name, (old, new) in exe_fixes.items():
    apply(name, lambda d, o=old, n=new: fix_exe(d, o, n))

# ── C. Run argument fixes ─────────────────────────────────────────────────

print("\n=== C. Run argument fixes ===")

# hotspot-cuda: CUDA version takes 6 args (single grid size, temp before power)
def fix_hotspot_cuda(d):
    correct_args = ["512", "2", "4",
                    "../../data/hotspot/temp_512",
                    "../../data/hotspot/power_512",
                    "output.out"]
    correct_files = ["../../data/hotspot/temp_512", "../../data/hotspot/power_512"]
    return fix_run_args_all_configs(d, correct_args, correct_files)

apply("rodinia-hotspot-cuda.json", fix_hotspot_cuda)

# srad-cuda: rows/cols must be multiples of 16 (502→512, 458→512)
def fix_srad_cuda(d):
    # current: ['502', '458', '0', '127', '0', '127', '0.5', '2']
    new_args = ["512", "512", "0", "127", "0", "127", "0.5", "2"]
    return fix_run_args_all_configs(d, new_args)

apply("rodinia-srad-cuda.json", fix_srad_cuda)

# srad-omp: same dimension fix (502→512, 458→512)
def fix_srad_omp(d):
    # current: ['502', '458', '0', '127', '0', '127', '2', '0.5', '2']
    new_args = ["512", "512", "0", "127", "0", "127", "2", "0.5", "2"]
    return fix_run_args_all_configs(d, new_args)

apply("rodinia-srad-omp.json", fix_srad_omp)

# nw-omp: needs third arg (num_threads)
def fix_nw_omp(d):
    new_args_correctness = ["2048", "10", "4"]
    changed = False
    cfg = d["run"]["input_configurations"]["correctness"]
    if cfg.get("arguments") != new_args_correctness:
        cfg["arguments"] = new_args_correctness
        changed = True
    if d["run"].get("default_arguments") != new_args_correctness:
        d["run"]["default_arguments"] = new_args_correctness
        changed = True
    # performance config may have bigger size
    if "performance" in d["run"]["input_configurations"]:
        perf = d["run"]["input_configurations"]["performance"]
        perf_args = perf.get("arguments", [])
        if len(perf_args) == 2:
            perf["arguments"] = perf_args + ["4"]
            changed = True
    return changed

apply("rodinia-nw-omp.json", fix_nw_omp)

# heartwall-omp: needs third arg (num_threads)
def fix_heartwall_omp(d):
    new_args = ["../../data/heartwall/test.avi", "20", "4"]
    new_files = ["../../data/heartwall/test.avi"]
    return fix_run_args_all_configs(d, new_args, new_files)

apply("rodinia-heartwall-omp.json", fix_heartwall_omp)

# nn-cuda: filelist.txt → ../../data/nn/filelist.txt
def fix_nn_cuda(d):
    changed = False
    for cfg in d["run"]["input_configurations"].values():
        args = cfg.get("arguments", [])
        new_args = [("../../data/nn/filelist.txt" if a == "filelist.txt" else a)
                    for a in args]
        if new_args != args:
            cfg["arguments"] = new_args
            changed = True
        files = cfg.get("input_files", [])
        new_files = [("../../data/nn/filelist.txt" if f == "filelist.txt" else f)
                     for f in files]
        if new_files != files:
            cfg["input_files"] = new_files
            changed = True
    defaults = d["run"].get("default_arguments", [])
    new_defaults = [("../../data/nn/filelist.txt" if a == "filelist.txt" else a)
                    for a in defaults]
    if new_defaults != defaults:
        d["run"]["default_arguments"] = new_defaults
        changed = True
    return changed

apply("rodinia-nn-cuda.json", fix_nn_cuda)

# nn-omp: same filelist fix
apply("rodinia-nn-omp.json", fix_nn_cuda)

print("\nDone." if not DRY_RUN else "\n[DRY RUN] Done.")
