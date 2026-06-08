#!/usr/bin/env python3
"""Fix OMP spec files for Batch 3 kernels that need Makefile.aomp build commands.

Updates build.commands.build, build.commands.clean, build.outputs.executable,
and run.executable for kernels that have Makefile.aomp but spec uses plain 'make'.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path("/home/samyak/Desktop/parbench_sam")
SPECS_DIR = PROJECT_ROOT / "specs"

# Mapping: kernel -> correct executable name (from Makefile.aomp dry run)
EXE_MAP = {
    "bezier-surface": "bs",
    "crc64": "main",
    "floydwarshall": "main",
    "gaussian": "main",
    "geglu": "main",
    "heat2d": "main",
    "iso2dfd": "main",
    "jenkins-hash": "main",
    "knn": "main",
    "mandelbrot": "main",
    "mis": "main",
    "murmurhash3": "main",
    "myocyte": "myocyte.out",
    "nw": "main",
    "perplexity": "main",
    "popcount": "main",
    "sobol": "SobolQRNG",
    "stencil1d": "stencil_1d",
    "triad": "triad",
}

# convolution3d: no OMP directory exists, SKIP

FIXED = 0
for kernel, exe in sorted(EXE_MAP.items()):
    spec_path = SPECS_DIR / f"hecbench-{kernel}-omp.json"
    if not spec_path.exists():
        print(f"  SKIP: {spec_path} not found")
        continue

    with open(spec_path) as f:
        spec = json.load(f)

    build = spec.get("build", {})
    commands = build.get("commands", {})
    outputs = build.get("outputs", {})
    run = spec.get("run", {})

    changes = []

    # Fix build command
    old_build = commands.get("build", "")
    new_build = "make -f Makefile.aomp CC=g++ DEVICE=cpu"
    if old_build != new_build:
        commands["build"] = new_build
        changes.append(f"build: {old_build!r} -> {new_build!r}")

    # Fix clean command
    old_clean = commands.get("clean", "")
    new_clean = "make -f Makefile.aomp clean"
    if old_clean != new_clean:
        commands["clean"] = new_clean
        changes.append(f"clean: {old_clean!r} -> {new_clean!r}")

    # Fix executable
    old_exe = outputs.get("executable", "")
    if old_exe != exe:
        outputs["executable"] = exe
        changes.append(f"outputs.exe: {old_exe!r} -> {exe!r}")

    # Fix run executable
    old_run_exe = run.get("executable", "")
    new_run_exe = f"./{exe}"
    if old_run_exe != new_run_exe:
        run["executable"] = new_run_exe
        changes.append(f"run.exe: {old_run_exe!r} -> {new_run_exe!r}")

    if changes:
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")
        FIXED += 1
        print(f"  FIXED: {kernel} — {', '.join(changes)}")
    else:
        print(f"  OK: {kernel} (no changes needed)")

print(f"\nTotal fixed: {FIXED}")
