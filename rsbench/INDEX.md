# RSBench Benchmark Suite — Index

RSBench is a proxy application that models the resonance evaluation portion of Monte Carlo nuclear transport — specifically the Multipole Windowing method used by OpenMC. Unlike XSBench (which models the unionized energy grid lookup), RSBench's workload is more compute-intensive (complex arithmetic over resonance windows) and less memory-bandwidth-bound. ParBench uses 4 RSBench specs (cuda, omp, opencl, omp_target), all PASS.

## Top-level Files

| Path | Description |
|------|-------------|
| `rsbench-src/README.md` | Overview of RSBench: algorithm description (Multipole Windowing), CLI flags, performance tuning, and reference results. |
| `rsbench-src/CHANGES.txt` | Version changelog documenting algorithmic updates, new API ports, and bug fixes. |
| `rsbench-src/LICENSE` | MIT license. |
| `rsbench-src/.travis.yml` | CI configuration for build testing across compilers. |
| `rsbench-src/docs/img/` | Figures used in documentation (algorithm diagrams, scaling plots). |

## API Implementations (`rsbench-src/`)

Each subdirectory is a complete implementation. All share a 6–7 file structure: `main`, `init`, `simulation`, `material`, `io`, `utils`, plus a header.

| Subfolder | Description |
|-----------|-------------|
| `rsbench-src/cuda/` | CUDA implementation. `simulation.cu` is the hotspot: each CUDA thread evaluates resonance windows for one neutron history using complex polynomial arithmetic. `rsbench.cuh` defines data structures. |
| `rsbench-src/openmp-threading/` | CPU OpenMP implementation with `#pragma omp parallel for` over neutron histories. Used as CPU baseline and as a translation source/target. |
| `rsbench-src/openmp-offload/` | OpenMP target offload implementation (`#pragma omp target teams distribute`). Corresponds to `omp_target` spec in ParBench; offloads the simulation loop to GPU. |
| `rsbench-src/opencl/` | OpenCL implementation; compute kernel in `kernel.cl`, host management via `cl_utils.c/h`. Uses command queue for kernel dispatch. |
| `rsbench-src/hip/` | HIP (ROCm) port of the CUDA variant using `hip*` API calls. Not in ParBench's eval corpus. |
| `rsbench-src/sycl/` | SYCL port using DPC++ parallel kernel submission over neutron history range. Not in ParBench's eval corpus. |

## Key Source Files (per implementation)

| File | Role |
|------|------|
| `main.*` | Entry point: parses CLI args (`-s`, `-m`, `-l`), calls init and simulation, prints verified checksum. |
| `simulation.*` | Core kernel: iterates over neutron histories, evaluates Faddeeva function over resonance windows. |
| `init.*` | Allocates and initializes the multipole data structures (windows, poles, residues). |
| `material.*` | Defines material composition: maps materials to nuclide lists used in resonance lookup. |
| `io.*` | Input validation, parameter printing, and checksum output for result verification. |
| `utils.*` | Helper math functions: Faddeeva function evaluation, random number generation. |
| `rsbench.h` / `rsbench.cuh` | Shared header: `Inputs`, `SimulationData`, `Window`, `Pole` struct definitions; function prototypes. |
| `cl_utils.c/h` | (OpenCL only) Platform/device enumeration, buffer allocation, and kernel compilation helpers. |

## ParBench Spec Coverage

- **4 specs**: `rsbench-cuda`, `rsbench-omp`, `rsbench-opencl`, `rsbench-omp_target`
- All 4 PASS; no KNOWN_FAIL
- Spec files: `specs/rsbench-{api}.json`
- Key distinction from XSBench: compute-heavy (Faddeeva function) vs memory-bandwidth-heavy (grid lookup)
