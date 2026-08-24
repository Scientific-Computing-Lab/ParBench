# HeCBench Benchmark Suite — Index

HeCBench (Heterogeneous Computing Benchmark) is a large collection of GPU benchmarks ported across multiple parallel programming APIs (CUDA, HIP, OpenMP offload, SYCL, OpenCL). The full suite contains 1874+ benchmark directories covering a wide range of application domains. ParBench uses a curated subset of 10 kernels (22 specs, 20 PASS, 2 KNOWN_FAIL). The full HeCBench directory is gitignored in ParBench; it is cloned locally at this path.

## Top-level Files

| Path | Description |
|------|-------------|
| `README.md` | Suite overview: purpose, supported APIs, build instructions, and how to add new benchmarks. |
| `CMakeLists.txt` | Top-level CMake build file; scans `src/` for benchmark dirs and configures build targets. |
| `CMakePresets.json` | Predefined CMake build presets for CUDA, HIP, SYCL, and OpenMP configurations. |
| `CMAKE_BUILD.md` | Detailed CMake build guide with per-API flag examples. |
| `benchmarks.yaml` | Machine-readable catalog of all benchmarks: kernel name, supported APIs, categories. |
| `convert_benchmarks.py` | Script to auto-convert between API variants using source-to-source translation templates. |
| `LICENSE` | Open-source license for the HeCBench collection. |
| `tools/` | Utility scripts for batch building, verification, and performance measurement across benchmarks. |
| `results/` | Reference output files and timing baselines used for correctness verification. |
| `cmake/` | CMake helper modules for API-specific compiler detection and flag configuration. |

## Source Structure (`src/`)

The `src/` directory contains 1874 subdirectories named `{kernel}-{api}` (e.g., `md-cuda`, `scan-omp`, `fft-sycl`). Supported API suffixes: `-cuda`, `-hip`, `-omp`, `-sycl`, `-opencl`.

### Application Domain Categories

| Domain | Representative Kernels |
|--------|----------------------|
| **Linear Algebra** | `lud`, `gaussian`, `nw`, `convolution1d`, `convolution3d`, `convolutionseparable`, `fft`, `dct8x8`, `sobel`, `fwt` |
| **Molecular Dynamics / Physics** | `md`, `md5hash`, `laplace3d`, `heat2d`, `iso2dfd`, `jacobi`, `nbody`, `particle-diffusion`, `tissue`, `lulesh` |
| **Machine Learning / Deep Learning** | `backprop`, `nn`, `knn`, `softmax-online`, `rmsnorm`, `geglu`, `perplexity`, `maxpool3d`, `triad`, `merge` |
| **Cryptography / Hashing** | `aes`, `chacha20`, `crc64`, `jenkins-hash`, `keccaktreehash`, `murmurhash3`, `md5hash`, `secp256k1`, `popcount` |
| **Graph Algorithms** | `bfs`, `page-rank`, `mis`, `floydwarshall`, `tsp`, `pso` |
| **Signal / Image Processing** | `bilateral`, `sobel`, `stencil1d`, `scan`, `radixsort`, `babelstream`, `fpc`, `eigenvalue` |
| **Scientific Computing** | `ccsd-trpdrv`, `feynman-kac`, `ising`, `myocyte`, `mandelbrot`, `nqueen`, `binomial`, `ga` |
| **Geometry / Rendering** | `bezier-surface`, `pathfinder`, `particlefilter`, `simplespmv`, `sobol`, `chi2`, `deredundancy` |

### Standard Subdirectory Structure (per `{kernel}-{api}/`)

Each benchmark directory follows a consistent layout:

| File | Description |
|------|-------------|
| `main.cu` / `main.cpp` / `main.c` | Entry point: initializes data, calls compute kernel, verifies output, prints timing. |
| `Makefile` | Build file; links against CUDA/HIP/OpenCL runtime and sets optimization flags. |
| `CMakeLists.txt` | CMake alternative build file for integration with top-level CMake. |
| `*.h` / `*.cuh` | Header files: data structure definitions, constants, function prototypes. |
| `*.cu` / `*.cpp` / `*.cl` | Kernel source: GPU compute logic for the benchmark. |
| `reference.h` | (Many kernels) CPU reference implementation used for output correctness checking. |

## ParBench Curated Subset (10 Kernels, 22 Specs)

These are the HeCBench kernels with verified PASS specs in ParBench. Each has CUDA + OMP (or OMP target) variants.

| Kernel | Description | ParBench Status |
|--------|-------------|-----------------|
| `aes` | AES-128 encryption; byte substitution and shift-row transforms on GPU blocks. | 2 specs, PASS |
| `babelstream` | Memory bandwidth benchmark (copy, scale, add, triad); measures peak DRAM throughput. | 2 specs, PASS |
| `backprop` | Neural network backpropagation; parallel gradient computation across layers. | 2 specs, PASS |
| `bezier-surface` | Bezier surface evaluation; parallel computation of surface points from control lattice. | 2 specs, PASS |
| `bilateral` | Bilateral image filter; edge-preserving smoothing with spatial + range Gaussian weights. | 2 specs, PASS |
| `binomial` | Binomial option pricing (Black-Scholes); parallel DP tree evaluation on GPU. | 2 specs, PASS |
| `ccsd-trpdrv` | CCSD(T) quantum chemistry driver; tensor contraction kernels for coupled-cluster energy. | 2 specs, PASS |
| `md` | Molecular dynamics (Lennard-Jones); short-range pairwise force computation with numeric oracle. | 2 specs, PASS (oracle: numeric_comparison (cuda tol 1e-8, omp_target tol 1e-9)) |
| `scan` | Parallel prefix scan (inclusive/exclusive); Blelloch algorithm with shared-memory optimization. | 3 specs: cuda PASS, omp PASS, omp_target KNOWN_FAIL |
| `stencil1d` | 1D stencil computation; finite-difference update over array elements with halo exchange. | 3 specs: cuda PASS, omp PASS, omp_target KNOWN_FAIL |

Additional curated kernels (chacha20, chi2, convolution1d, convolution3d, convolutionseparable, crc64, dct8x8, deredundancy, eigenvalue, feynman-kac, fft, floydwarshall, fpc, fwt, ga, gaussian, geglu, heat2d, ising, iso2dfd, jacobi, jenkins-hash, keccaktreehash, knn, laplace3d, lud, lulesh, mandelbrot, maxpool3d, md5hash, merge, mis, murmurhash3, myocyte, nbody, nn, nqueen, nw, page-rank, particle-diffusion, pathfinder, perplexity, popcount, pso, radixsort, rmsnorm, secp256k1, simplespmv, sobel, sobol, softmax-online, thomas, tissue, triad, tsp) — partial or future spec coverage.

## Notes for ParBench

- All spec files for HeCBench live in `specs/hecbench-{kernel}-{api}.json`
- Source executables built from `HeCBench-master/src/{kernel}-{api}/`
- HeCBench-master is gitignored; never add it to version control
- The `convert_benchmarks.py` script is NOT used for ParBench eval — source and target are taken as-is from the respective API subdirectories
