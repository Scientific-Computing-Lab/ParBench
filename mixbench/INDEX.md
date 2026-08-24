# Mixbench Benchmark Suite — Index

Mixbench is a GPU micro-benchmark that sweeps the compute-to-memory-access ratio across a configurable range, characterizing the performance roofline of a GPU device. It does not model a specific application — instead it generates synthetic kernels at varying arithmetic intensities. ParBench uses 3 mixbench specs (cuda, omp, opencl), all PASS.

## Top-level Files

| Path | Description |
|------|-------------|
| `mixbench-src/README.md` | Overview: explains the arithmetic intensity sweep methodology, supported APIs, build instructions, and how to interpret output. |
| `mixbench-src/LICENSE` | Open-source license (MIT). |
| `mixbench-src/.clang-format` | Code style configuration used for consistent formatting across all C++ source files. |
| `mixbench-src/.gitattributes` | Git line-ending normalization rules. |
| `mixbench-src/include/` | Shared headers used across multiple API variants. |
| `mixbench-src/include/common.h` | Common macros, data types, and helper functions shared by CUDA, OpenCL, and HIP variants. |
| `mixbench-src/include/timestamp.h` | High-resolution timing utilities for measuring kernel execution time. |

## API Implementations (`mixbench-src/`)

Each subfolder is a self-contained variant targeting one parallel programming API.

| Subfolder | Description |
|-----------|-------------|
| `mixbench-src/mixbench-cuda/` | CUDA implementation. `mix_kernels_cuda.cu` generates kernels at each arithmetic intensity point using template metaprogramming; `main-cuda.cpp` drives the sweep and reports GFLOPS and memory bandwidth per step. |
| `mixbench-src/mixbench-opencl/` | OpenCL implementation. `mix_kernels.cl` is the GPU kernel file; `mix_kernels_ocl.cpp` dispatches it at each intensity ratio via the OpenCL command queue. `loclutil.h` wraps platform/device setup. |
| `mixbench-src/mixbench-cpu/` | CPU-only OpenMP variant for baseline characterization on host hardware. `mix_kernels_cpu.cpp` uses `#pragma omp simd` / `omp parallel for` to drive the intensity sweep. |
| `mixbench-src/mixbench-hip/` | HIP (ROCm) port of the CUDA variant; uses `hipLaunchKernelGGL` with same kernel logic. Not in ParBench's eval corpus. |
| `mixbench-src/mixbench-sycl/` | SYCL port using DPC++ parallel kernel submission. Not in ParBench's eval corpus. |

## Key Source Files (per implementation)

| File | Role |
|------|------|
| `main-*.cpp` | Entry point: parses options, initializes device, calls the kernel sweep, collects and prints results. |
| `mix_kernels_*.cu/.cpp/.cl` | Core benchmark kernel: parameterized by arithmetic intensity ratio; executes FMA + load/store mix. |
| `mix_kernels_*.h` | Header declaring kernel launch signatures and intensity sweep configuration constants. |
| `version_info.h` | Build-time version string embedded in binary output. |
| `lcutil.h` / `loclutil.h` | CUDA/OpenCL device and context setup helpers (query device name, allocate buffers). |
| `CMakeLists.txt` | CMake build file for each variant; sets compiler flags and links against CUDA/OpenCL/HIP runtimes. |

## ParBench Spec Coverage

- **3 specs**: `mixbench-cuda`, `mixbench-omp`, `mixbench-opencl`
- All 3 PASS; no KNOWN_FAIL
- Spec files: `specs/mixbench-{api}.json`
- Benchmark measures: GFLOPS vs memory bandwidth across the arithmetic intensity axis
