#!/usr/bin/env python3
"""Second round of Rodinia spec fixes after retry batch analysis.

Fixes:
  1. streamcluster-opencl: wrong exe name (./sc_gpu → ./streamcluster.out)
  2. backprop-cuda: add LD_LIBRARY_PATH for libcudart.so.12 at runtime
  3. hotspot-opencl: fix to 6-arg form (single grid size, temp before power)
  4. nn-opencl: add -f data dir flag, fix filelist path
  5. srad-opencl: 502/458 → 512/512 (must be multiples of 16)
  6. dwt2d-opencl: fix args (use -i flag and -D for dimensions)
  7. nw-opencl: add ./nw.cl as third arg
  8. nn-cuda: use local filelist.txt (already created on disk)
  9. nn-omp: use local filelist.txt (already created on disk)

Usage:
    cd /home/samyak/Desktop/parbench_sam
    source env_parbench/bin/activate
    python3 scripts/fix_rodinia_run_args2.py [--dry-run]
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/samyak/Desktop/parbench_sam")
SPECS_DIR = PROJECT_ROOT / "specs"
DRY_RUN = "--dry-run" in sys.argv

CUDA_LIB = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/lib64"


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


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


# 1. streamcluster-opencl: wrong exe
def fix_streamcluster_opencl(d):
    changed = False
    if d["build"]["outputs"]["executable"] == "./sc_gpu":
        d["build"]["outputs"]["executable"] = "./streamcluster.out"
        changed = True
    if d["run"]["executable"] == "./sc_gpu":
        d["run"]["executable"] = "./streamcluster.out"
        changed = True
    return changed

apply("rodinia-streamcluster-opencl.json", fix_streamcluster_opencl)

# 2. backprop-cuda: add LD_LIBRARY_PATH for runtime
def fix_backprop_cuda_env(d):
    ld_path = f"{CUDA_LIB}:$LD_LIBRARY_PATH"
    env = d["run"].get("environment_variables") or {}
    if env.get("LD_LIBRARY_PATH") != ld_path:
        if not isinstance(env, dict):
            env = {}
        env["LD_LIBRARY_PATH"] = ld_path
        d["run"]["environment_variables"] = env
        return True
    return False

apply("rodinia-backprop-cuda.json", fix_backprop_cuda_env)

# 3. hotspot-opencl: 6-arg form (rows/cols is single value; temp before power)
def fix_hotspot_opencl(d):
    correct_args = ["512", "2", "4",
                    "../../data/hotspot/temp_512",
                    "../../data/hotspot/power_512",
                    "output.out"]
    correct_files = ["../../data/hotspot/temp_512", "../../data/hotspot/power_512"]
    changed = False
    for cfg in d["run"]["input_configurations"].values():
        if cfg.get("arguments") != correct_args:
            cfg["arguments"] = correct_args
            changed = True
        if cfg.get("input_files") != correct_files:
            cfg["input_files"] = correct_files
            changed = True
    if d["run"].get("default_arguments") != correct_args:
        d["run"]["default_arguments"] = correct_args
        changed = True
    return changed

apply("rodinia-hotspot-opencl.json", fix_hotspot_opencl)

# 4. nn-opencl: add -f data dir flag (filelist path already correct from prev fix)
def fix_nn_opencl(d):
    changed = False
    base_args = ["../../data/nn/filelist.txt", "-r", "5", "-lat", "30", "-lng", "90", "-f", "../../data/nn"]
    if d["run"].get("default_arguments") != base_args:
        d["run"]["default_arguments"] = base_args
        changed = True
    for cfg in d["run"]["input_configurations"].values():
        if cfg.get("arguments") != base_args:
            cfg["arguments"] = base_args
            changed = True
        files = ["../../data/nn/filelist.txt",
                 "../../data/nn/cane4_0.db",
                 "../../data/nn/cane4_1.db",
                 "../../data/nn/cane4_2.db",
                 "../../data/nn/cane4_3.db"]
        if cfg.get("input_files") != files:
            cfg["input_files"] = files
            changed = True
    return changed

apply("rodinia-nn-opencl.json", fix_nn_opencl)

# 5. srad-opencl: 502/458 → 512/512
def fix_srad_opencl(d):
    new_args = ["512", "512", "0", "127", "0", "127", "2", "0.5", "2"]
    changed = False
    for cfg in d["run"]["input_configurations"].values():
        if cfg.get("arguments") != new_args:
            cfg["arguments"] = new_args
            changed = True
    if d["run"].get("default_arguments") != new_args:
        d["run"]["default_arguments"] = new_args
        changed = True
    return changed

apply("rodinia-srad-opencl.json", fix_srad_opencl)

# 6. dwt2d-opencl: fix args (use -i flag and -D)
def fix_dwt2d_opencl(d):
    # old: ['../../data/dwt2d/rgb.bmp', '-d', '1024x1024', '-f', '-5', '-l', '3']
    # new: ['rgb.bmp', '-i', '../../data/dwt2d', '-D', '1024x1024', '-f', '-5', '-l', '3']
    new_args = ["rgb.bmp", "-i", "../../data/dwt2d", "-D", "1024x1024", "-f", "-5", "-l", "3"]
    new_files = ["../../data/dwt2d/rgb.bmp"]
    changed = False
    for cfg in d["run"]["input_configurations"].values():
        if cfg.get("arguments") != new_args:
            cfg["arguments"] = new_args
            changed = True
        if cfg.get("input_files") != new_files:
            cfg["input_files"] = new_files
            changed = True
    if d["run"].get("default_arguments") != new_args:
        d["run"]["default_arguments"] = new_args
        changed = True
    return changed

apply("rodinia-dwt2d-opencl.json", fix_dwt2d_opencl)

# 7. nw-opencl: add ./nw.cl as third arg
def fix_nw_opencl(d):
    new_args = ["2048", "10", "./nw.cl"]
    changed = False
    for cfg in d["run"]["input_configurations"].values():
        if cfg.get("arguments") != new_args:
            cfg["arguments"] = new_args
            changed = True
    if d["run"].get("default_arguments") != new_args:
        d["run"]["default_arguments"] = new_args
        changed = True
    return changed

apply("rodinia-nw-opencl.json", fix_nw_opencl)

# 8. nn-cuda: use local filelist.txt
def fix_nn_cuda(d):
    new_args = ["filelist.txt", "-r", "5", "-lat", "30", "-lng", "90"]
    new_files = ["filelist.txt",
                 "../../data/nn/cane4_0.db",
                 "../../data/nn/cane4_1.db",
                 "../../data/nn/cane4_2.db",
                 "../../data/nn/cane4_3.db"]
    changed = False
    if d["run"].get("default_arguments") != new_args:
        d["run"]["default_arguments"] = new_args
        changed = True
    for cfg in d["run"]["input_configurations"].values():
        if cfg.get("arguments") != new_args:
            cfg["arguments"] = new_args
            changed = True
        if cfg.get("input_files") != new_files:
            cfg["input_files"] = new_files
            changed = True
    return changed

apply("rodinia-nn-cuda.json", fix_nn_cuda)

# 9. nn-omp: use local filelist.txt
apply("rodinia-nn-omp.json", fix_nn_cuda)

print("\nDone." if not DRY_RUN else "\n[DRY RUN] Done.")
