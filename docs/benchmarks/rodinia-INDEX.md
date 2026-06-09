# Rodinia Benchmark Suite — Index

Rodinia is a suite of parallel benchmarks covering diverse HPC application domains. ParBench uses it as the primary evaluation corpus (60 specs, 54 PASS, 6 KNOWN_FAIL). The suite provides CUDA, OpenMP, and OpenCL implementations of the same kernels, making it ideal for cross-API translation evaluation.

## Top-level Files and Directories

| Path | Description |
|------|-------------|
| `Makefile` | Top-level build driver; delegates to per-API subdirectories and handles data download. |
| `LICENSE` | Rodinia open-source license (University of Virginia). |
| `README` | Overview of the benchmark suite, kernel descriptions, and build instructions. |
| `common/` | Shared build infrastructure: `make.config` sets CUDA/OpenCL paths; `common.mk` provides reusable compile rules. |
| `data/` | Input datasets for each kernel (graph files, image frames, temperature grids, etc.). Organized per-kernel. |
| `scripts/` | Helper shell scripts for batch building, running, and output diffing across API variants. |
| `others/` | Experimental or deprecated kernel variants not in the main evaluation set. |
| `rodinia-src/` | Git submodule (commit `9c10d3ea`) — canonical upstream Rodinia source. Empty in git worktrees; cloned on the main machine. |

## API Subdirectories

Each API subdirectory contains kernel implementations. The sets differ: CUDA has 23 kernels, OpenMP has 19, OpenCL has 21.

### `cuda/` — CUDA Kernel Implementations

| Subfolder | Description |
|-----------|-------------|
| `cuda/backprop/` | Backpropagation neural network training; CUDA kernels for layered weight updates. |
| `cuda/bfs/` | Breadth-first search on sparse graphs; frontier expansion via CUDA thread blocks. Hardcodes source node 0. |
| `cuda/b+tree/` | B+ tree database index operations; GPU-accelerated node search and range queries. |
| `cuda/cfd/` | Computational fluid dynamics (Euler solver); iterative RK3 time-stepping over unstructured meshes. |
| `cuda/dwt2d/` | 2D discrete wavelet transform; parallel lifting-scheme decomposition on image tiles. |
| `cuda/gaussian/` | Gaussian elimination with partial pivoting; row reduction parallelized across CUDA warps. |
| `cuda/heartwall/` | Cardiac ultrasound image analysis; object tracking across video frames using template matching. |
| `cuda/hotspot/` | 2D thermal simulation (transient heat equation); stencil iterations on a chip power map. |
| `cuda/hotspot3D/` | 3D variant of hotspot; volumetric heat diffusion across stacked chip layers. |
| `cuda/huffman/` | Huffman entropy coding; parallel histogram, prefix-sum, and tree construction on GPU. |
| `cuda/hybridsort/` | Hybrid merge/radix sort; uses CUDA texture references (KNOWN_FAIL on CUDA 12+). |
| `cuda/kmeans/` | K-means clustering; CUDA kernel uses texture memory (KNOWN_FAIL on CUDA 12+). |
| `cuda/lavaMD/` | Molecular dynamics short-range force calculation; neighbor-list particle interactions on GPU. |
| `cuda/leukocyte/` | White blood cell detection in microscopy video; MSER feature extraction + motion estimation. |
| `cuda/lud/` | LU decomposition; blocked tiled algorithm with shared memory for dense linear systems. |
| `cuda/mummergpu/` | DNA sequence alignment (suffix array); uses texture memory (KNOWN_FAIL on CUDA 12+). |
| `cuda/myocyte/` | Cardiac muscle cell ODE simulation; parallel Runge-Kutta integration across cell ensemble. |
| `cuda/nn/` | K-nearest neighbor search in spatial datasets; CUDA parallel distance computation. |
| `cuda/nw/` | Needleman-Wunsch sequence alignment; diagonal wavefront parallelism on alignment matrix. |
| `cuda/particlefilter/` | Particle filter (Sequential Monte Carlo) for object tracking in noisy video. |
| `cuda/pathfinder/` | Dynamic programming shortest path on 2D grid; wavefront sweep with shared-memory tiling. |
| `cuda/srad/` | Speckle reducing anisotropic diffusion (image denoising); iterative PDE on image pixels. |
| `cuda/streamcluster/` | Online clustering (streaming k-median); facility location with CUDA point assignment. |
| `cuda/util/` | CUDA device query and initialization utilities shared across kernels. |

### `openmp/` — OpenMP CPU/Target Implementations

19 kernels (missing vs CUDA: dwt2d, gaussian, huffman, hybridsort). Each uses `#pragma omp parallel` / `omp target` directives. Notable differences from CUDA: BFS has source node commented out; NW only writes `result.txt` in OMP (synthesis asymmetry).

| Subfolder | Description |
|-----------|-------------|
| `openmp/backprop/` | CPU-parallel backpropagation using OpenMP thread teams for layer weight updates. |
| `openmp/bfs/` | BFS with OpenMP parallelism; source node selection differs from CUDA version (commented out). |
| `openmp/b+tree/` | B+ tree operations parallelized with OpenMP; search and range queries across thread pool. |
| `openmp/cfd/` | CFD Euler solver with OpenMP threading; FP reduction order differs from CUDA due to scheduling. |
| `openmp/heartwall/` | Heartwall tracking with OpenMP-parallelized frame processing and template matching. |
| `openmp/hotspot/` | Thermal stencil with OpenMP thread parallelism; FP order diverges from CUDA. |
| `openmp/hotspot3D/` | 3D hotspot with OpenMP parallel loops; verified with `numeric_comparison` oracle (tolerance 1e-7). |
| `openmp/kmeans/` | K-means with OpenMP parallel distance computation and centroid update. |
| `openmp/lavaMD/` | Molecular dynamics with OpenMP-parallelized particle force loops. |
| `openmp/leukocyte/` | Leukocyte cell detection with OpenMP video frame parallelism. |
| `openmp/lud/` | LU decomposition with OpenMP tiled factorization. |
| `openmp/mummergpu/` | Sequence alignment with OpenMP; also uses CUDA kernel (KNOWN_FAIL). |
| `openmp/myocyte/` | Cardiac ODE simulation with OpenMP cell ensemble parallelism. |
| `openmp/nn/` | KNN search with OpenMP parallel distance computation. |
| `openmp/nw/` | Needleman-Wunsch with OpenMP diagonal wavefront; writes result.txt (not in CUDA). |
| `openmp/particlefilter/` | Particle filter with OpenMP parallel weight computation and resampling. |
| `openmp/pathfinder/` | Dynamic programming with OpenMP parallel row sweeps. |
| `openmp/srad/` | SRAD image denoising with OpenMP parallel pixel updates. |
| `openmp/streamcluster/` | Streaming k-median with OpenMP parallel point assignment. |

### `opencl/` — OpenCL Implementations

21 kernels rewritten for OpenCL (missing vs CUDA: huffman, mummergpu). Used for cross-API translation targets in ParBench evaluation.

| Subfolder | Description |
|-----------|-------------|
| `opencl/backprop/` | OpenCL backprop with `cl_kernel` weight-update kernels. |
| `opencl/bfs/` | OpenCL BFS with frontier expansion kernels dispatched via command queue. |
| `opencl/b+tree/` | OpenCL B+ tree index with separate kernels for search and range ops. |
| `opencl/cfd/` | OpenCL CFD Euler solver; RK3 time-stepping via sequential kernel dispatches. |
| `opencl/dwt2d/` | OpenCL 2D discrete wavelet transform with NDRange decomposition. |
| `opencl/gaussian/` | OpenCL Gaussian elimination with 2D NDRange over matrix rows. |
| `opencl/heartwall/` | OpenCL cardiac ultrasound tracking with template matching kernels. |
| `opencl/hotspot/` | OpenCL thermal stencil with 2D work-group tiling. |
| `opencl/hotspot3D/` | OpenCL 3D hotspot with 3D NDRange dispatch. |
| `opencl/hybridsort/` | OpenCL hybrid merge/radix sort. |
| `opencl/kmeans/` | OpenCL K-means with SIGSEGV in runtime (KNOWN_FAIL). |
| `opencl/lavaMD/` | OpenCL molecular dynamics particle force kernels. |
| `opencl/leukocyte/` | OpenCL leukocyte tracking with image convolution kernels. |
| `opencl/lud/` | OpenCL LU decomposition with tiled kernel for blocked factorization. |
| `opencl/myocyte/` | OpenCL myocyte ODE integration; weak oracle (stdout_pattern + exit_code) due to FP divergence from CUDA. |
| `opencl/nn/` | OpenCL KNN with TIMEOUT/SIGSEGV (KNOWN_FAIL). |
| `opencl/nw/` | OpenCL Needleman-Wunsch with wavefront diagonal kernel. |
| `opencl/particlefilter/` | OpenCL particle filter weight and resample kernels. |
| `opencl/pathfinder/` | OpenCL pathfinder; uses `-c/-r/-h` flags (not positional args). |
| `opencl/srad/` | OpenCL SRAD anisotropic diffusion kernels. |
| `opencl/streamcluster/` | OpenCL streaming k-median point assignment. |
| `opencl/util/` | OpenCL device/platform setup utilities shared across kernels. |

## ParBench Spec Coverage

- **60 specs total**: 54 TRUE PASS, 6 KNOWN_FAIL (kmeans-cuda, mummergpu-cuda, mummergpu-omp, hybridsort-cuda, nn-opencl, kmeans-opencl)
- **5 phantom specs deleted**: gaussian-omp, huffman-omp, huffman-opencl, hybridsort-omp, mummergpu-opencl (directories don't exist in source)
- Spec files live in `specs/rodinia-{kernel}-{api}.json`
