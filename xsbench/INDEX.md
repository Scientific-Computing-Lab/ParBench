# XSBench Benchmark Suite — Index

XSBench is a Monte Carlo nuclear reactor simulation proxy benchmark that models the continuous energy neutron transport cross-section lookup kernel. It is computationally representative of production codes like OpenMC and is highly memory-bandwidth-bound. ParBench uses 4 XSBench specs (cuda, omp, opencl, omp_target), all PASS.

## Top-level Files

| Path | Description |
|------|-------------|
| `xsbench-src/README.md` | Overview of XSBench, algorithm description, command-line flags, and performance tuning guidance. |
| `xsbench-src/CHANGES.txt` | Version changelog documenting algorithmic changes, API additions, and performance regressions. |
| `xsbench-src/LICENSE` | MIT license. |
| `xsbench-src/.travis.yml` | CI configuration for automated build and test across compiler variants. |
| `xsbench-src/docs/XSBench_Theory.pdf` | Technical report describing the cross-section lookup algorithm, nuclide grid structure, and parallelism strategy. |
| `xsbench-src/docs/XSBench_Theory.tex` | LaTeX source for the theory document. |
| `xsbench-src/docs/img/` | Figures used in the theory document (algorithm diagrams, roofline plots). |

## API Implementations (`xsbench-src/`)

Each API subdirectory contains a complete, self-contained implementation. All share the same 7-file structure: `Main`, `Simulation`, `GridInit`, `Materials`, `XSutils`, `io`, and a header.

| Subfolder | Description |
|-----------|-------------|
| `xsbench-src/cuda/` | CUDA implementation using GPU thread blocks for parallel nuclide lookups. `Simulation.cu` is the hotspot: each thread performs one cross-section lookup via binary search on the unionized energy grid. |
| `xsbench-src/openmp-threading/` | CPU OpenMP implementation using `#pragma omp parallel for` over the lookup loop. Used as the baseline for CPU-side performance comparison and as a translation target. |
| `xsbench-src/openmp-offload/` | OpenMP target offload implementation (`#pragma omp target teams distribute`). GPU execution via offload directives; corresponds to the `omp_target` spec in ParBench. |
| `xsbench-src/opencl/` | OpenCL C implementation; host code in `Main.c`, GPU kernel in `kernel.cl`. Uses `CLutils.c/h` for platform/device setup and command queue management. |
| `xsbench-src/hip/` | HIP (ROCm) port; structurally identical to CUDA but using HIP API calls. Not currently in ParBench's eval corpus. |
| `xsbench-src/sycl/` | SYCL implementation using DPC++; C++ parallel STL style with `sycl::queue`. Not in ParBench's eval corpus. |

## Key Source Files (per implementation)

| File | Role |
|------|------|
| `Main.*` | Entry point; parses CLI args, initializes data, launches simulation, prints results. |
| `Simulation.*` | Core kernel loop: iterates over neutron histories, performs nuclide cross-section lookups. |
| `GridInit.*` | Constructs the unionized energy grid and nuclide data structures used in simulation. |
| `Materials.*` | Defines material composition (nuclide fractions) for each material region. |
| `XSutils.*` | Utility functions: random number generation, binary search on energy grid, checksum. |
| `io.*` | Input validation, benchmark parameter printing, and output result reporting. |
| `XSbench_header.*` | Shared header: data structure definitions (`NuclideGridPoint`, `Macro_XS`), function prototypes, constants. |

## ParBench Spec Coverage

- **4 specs**: `xsbench-cuda`, `xsbench-omp`, `xsbench-opencl`, `xsbench-omp_target`
- All 4 PASS; no KNOWN_FAIL
- Spec files: `specs/xsbench-{api}.json`
- Run args: problem size 200000 lookups, 355 nuclides (default medium problem)
