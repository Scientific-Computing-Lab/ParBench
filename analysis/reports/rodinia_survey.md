# Rodinia GPU Benchmark Suite — Survey Report

**Date**: 2026-03-03  
**Source**: <https://github.com/yuhc/gpu-rodinia>  
**Rodinia root**: `/home/samyak/Desktop/parbench_sam/rodinia`

## Summary

- **Total unique kernels**: 24
- **Total kernel-API specs**: 64
- **cuda** variants: 23
- **omp** variants: 19
- **opencl** variants: 21
- **others** variants: 1
- **HeCBench overlaps**: 6

## API Coverage Matrix

| Kernel | cuda | opencl | omp | others | HeCBench Overlap |
|--------|-------|-------|-------|-------|------------------|
| b+tree | ✅ (8 src) | ✅ (6 src) | ✅ (5 src) | ❌ |  |
| backprop | ✅ (5 src) | ✅ (4 src) | ✅ (4 src) | ❌ | ⚠️ YES |
| bfs | ✅ (3 src) | ✅ (2 src) | ✅ (1 src) | ❌ |  |
| cfd | ✅ (4 src) | ✅ (1 src) | ✅ (4 src) | ❌ |  |
| dwt2d | ✅ (8 src) | ✅ (3 src) | ❌ | ❌ |  |
| gaussian | ✅ (1 src) | ✅ (5 src) | ❌ | ❌ | ⚠️ YES |
| heartwall | ✅ (6 src) | ✅ (7 src) | ✅ (5 src) | ❌ |  |
| hotspot | ✅ (1 src) | ✅ (2 src) | ✅ (1 src) | ❌ |  |
| hotspot3D | ✅ (2 src) | ✅ (2 src) | ✅ (1 src) | ❌ |  |
| huffman | ✅ (9 src) | ❌ | ❌ | ❌ |  |
| hybridsort | ✅ (6 src) | ✅ (3 src) | ❌ | ❌ |  |
| kmeans | ✅ (7 src) | ✅ (6 src) | ✅ (8 src) | ❌ |  |
| lavaMD | ✅ (6 src) | ✅ (6 src) | ✅ (5 src) | ❌ |  |
| leukocyte | ✅ (82 src) | ✅ (83 src) | ✅ (80 src) | ❌ |  |
| lud | ✅ (6 src) | ✅ (5 src) | ✅ (6 src) | ❌ | ⚠️ YES |
| mummergpu | ✅ (9 src) | ❌ | ✅ (9 src) | ❌ |  |
| myocyte | ✅ (19 src) | ✅ (10 src) | ✅ (10 src) | ❌ |  |
| nn | ✅ (2 src) | ✅ (4 src) | ✅ (2 src) | ❌ | ⚠️ YES |
| nw | ✅ (2 src) | ✅ (1 src) | ✅ (1 src) | ❌ | ⚠️ YES |
| particlefilter | ✅ (2 src) | ✅ (3 src) | ✅ (1 src) | ❌ |  |
| pathfinder | ✅ (1 src) | ✅ (2 src) | ✅ (1 src) | ❌ | ⚠️ YES |
| rng | ❌ | ❌ | ❌ | ✅ (2 src) |  |
| srad | ✅ (14 src) | ✅ (12 src) | ✅ (6 src) | ❌ |  |
| streamcluster | ✅ (4 src) | ✅ (1 src) | ✅ (2 src) | ❌ |  |

## HeCBench Overlaps

These kernels exist in both Rodinia and the existing HeCBench selection:

- **backprop**: Rodinia APIs = cuda, omp, opencl
- **gaussian**: Rodinia APIs = cuda, opencl
- **lud**: Rodinia APIs = cuda, omp, opencl
- **nn**: Rodinia APIs = cuda, omp, opencl
- **nw**: Rodinia APIs = cuda, omp, opencl
- **pathfinder**: Rodinia APIs = cuda, omp, opencl

## Data Dependencies

### Shared Data Directory

The `data/` directory is **empty** (contains only `.gitkeep`).
Data must be downloaded separately from one of these sources:
- **Original**: <http://lava.cs.virginia.edu/Rodinia/download_links.htm>
- **Dropbox mirror**: <https://www.dropbox.com/s/cc6cozpboht3mtu/rodinia-3.1-data.tar.gz>
  - MD5: `0CF70BC7BD538794F626C0E04A6447AE`
  - SHA-256: `B90994D5208EC5A0A133DFB9AB7928A1E8A16741503A91D212884B9E4FCE8CD8`

⚠️ **Do NOT run the download yet** — just noting the location.

### Scripts (not data-download-specific)

- `scripts/run_cpu.sh`
- `scripts/run_gpu.sh`
- `scripts/run_wrap.sh`
- `scripts/parse_result.sh`

### Per-Kernel Data Files

- **b+tree**: ../../data/b+tree/command.txt, ../../data/b+tree/mil.txt
- **bfs**: ../../data/bfs/graph1MW_6.txt, cl.ptx, result.txt
- **cfd**: PD_OCL.txt, cl.ptx, errinfo
- **dwt2d**: ../../data/dwt2d/",srcFilename);, ../../data/dwt2d/";
- **gaussian**: ../../data/gaussian/matrix1024.txt
- **heartwall**: ../../data/heartwall/input.txt, ../../data/heartwall/test.avi
- **hotspot**: ../../data/hotspot/power_1024, ../../data/hotspot/temp_1024
- **hybridsort**:  // "/data/", ../../data/", ../../data/",, data/", data/",
  ...and 1 more
- **lavaMD**: result.txt
- **leukocyte**: iotort.dat, laplace.mat, result.txt
- **lud**: ../../data/lud/256.dat, ./lud_kernel.cl
- **mummergpu**: data/README, data/shortqry.fa, data/shortref.fa, data/timing/gentestdata.py, data/timing/global.cnf
  ...and 2 more
- **myocyte**: ../../data/myocyte/params.txt",, ../../data/myocyte/y.txt",, output.txt
- **nn**: data/cane%d_%d.db",
- **nw**: result.txt
- **particlefilter**: output.txt
- **pathfinder**: kernels.cl
- **srad**: ../../data/srad/image.pgm",
- **streamcluster**: PD.txt, PD_OCL.txt, cl.ptx, data_opencl.txt, errinfo

## Build System Details

All kernels use `make` with a shared `common/make.config` that defines:
- `CUDA_DIR` (default `/usr/local/cuda`)
- `CUDA_LIB_DIR`
- `SDK_DIR`
- `OPENCL_DIR`, `OPENCL_INC`, `OPENCL_LIB`

### Per-Kernel Build Notes

- **b+tree**
  - cuda: includes common/make.config; env vars: CUDA_LIB_DIR; subdirs: kernel, util
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: kernel, util
  - omp: subdirs: kernel, util
- **backprop**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **bfs**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: NV_OPENCL_DIR, OPENCL_DIR, OPENCL_INC, OPENCL_LIB
- **cfd**
  - cuda: includes common/make.config; env vars: SDK_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **dwt2d**
  - cuda: includes common/make.config; env vars: CUDA_DIR; subdirs: dwt_cuda
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: dwt_cl
- **gaussian**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **heartwall**
  - cuda: subdirs: AVI
  - opencl: includes common/make.config; env vars: NV_OPENCL_DIR, OPENCL_DIR, OPENCL_INC, OPENCL_LIB; subdirs: info, kernel, util
  - omp: subdirs: AVI
- **hotspot**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **hotspot3D**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **hybridsort**
  - cuda: includes common/make.config; env vars: CUDA_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **kmeans**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
  - omp: subdirs: kmeans_openmp, kmeans_serial
- **lavaMD**
  - cuda: includes common/make.config; env vars: CUDA_DIR; subdirs: kernel, util
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: kernel, util
  - omp: subdirs: kernel, util
- **leukocyte**
  - cuda: subdirs: CUDA, meschach_lib
  - opencl: subdirs: OpenCL, meschach_lib
  - omp: subdirs: OpenMP, meschach_lib
- **lud**
  - cuda: subdirs: base, common, cuda, tools
  - opencl: subdirs: base, common, ocl, tools
  - omp: subdirs: base, common, omp, tools
- **mummergpu**
  - cuda: subdirs: data, experiments, src
  - omp: subdirs: data, experiments, src
- **myocyte**
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: kernel, util
- **nn**
  - cuda: includes common/make.config; env vars: CUDA_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **nw**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **particlefilter**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **pathfinder**
  - cuda: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
- **rng**
  - others: in others/ directory
- **srad**
  - cuda: subdirs: srad_v1, srad_v2
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: info, kernel, output, util
  - omp: subdirs: srad_v1, srad_v2
- **streamcluster**
  - cuda: includes common/make.config; env vars: CUDA_DIR
  - opencl: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

## Red Flags

No red flags detected.

## Detailed Kernel Catalog

### b+tree

- **Complexity**: B+ tree search and range queries
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/b+tree`)
- Sources: kernel/kernel_gpu_cuda.cu, kernel/kernel_gpu_cuda_2.cu, kernel/kernel_gpu_cuda_wrapper.cu, kernel/kernel_gpu_cuda_wrapper_2.cu, main.c, util/cuda/cuda.cu, util/num/num.c, util/timer/timer.c
- Headers: common.h, kernel/kernel_gpu_cuda_wrapper.h, kernel/kernel_gpu_cuda_wrapper_2.h, main.h, util/cuda/cuda.h, util/num/num.h, util/timer/timer.h
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: CUDA_LIB_DIR; subdirs: kernel, util

**omp** (`openmp/b+tree`)
- Sources: kernel/kernel_cpu.c, kernel/kernel_cpu_2.c, main.c, util/num/num.c, util/timer/timer.c
- Headers: common.h, kernel/kernel_cpu.h, kernel/kernel_cpu_2.h, main.h, util/num/num.h, util/timer/timer.h
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: subdirs: kernel, util

**opencl** (`opencl/b+tree`)
- Sources: kernel/kernel_gpu_opencl_wrapper.c, kernel/kernel_gpu_opencl_wrapper_2.c, main.c, util/num/num.c, util/opencl/opencl.c, util/timer/timer.c
- Headers: common.h, kernel/kernel_gpu_opencl_wrapper.h, kernel/kernel_gpu_opencl_wrapper_2.h, main.h, util/num/num.h, util/opencl/opencl.h, util/timer/timer.h
- OpenCL kernels: kernel/kernel_gpu_opencl.cl, kernel/kernel_gpu_opencl_2.cl
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: kernel, util

### backprop

- **Complexity**: neural network training (backpropagation)
- **HeCBench overlap**: YES
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/backprop`)
- Sources: backprop.c, backprop_cuda.cu, backprop_cuda_kernel.cu, facetrain.c, imagenet.c
- Headers: backprop.h
- Build: make (Makefile)
- Verification: golden reference; prints results
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/backprop`)
- Sources: backprop.c, backprop_kernel.c, facetrain.c, imagenet.c
- Headers: backprop.h
- Build: make (Makefile)
- Verification: prints results

**opencl** (`opencl/backprop`)
- Sources: backprop.c, backprop_ocl.cpp, facetrain.c, imagenet.c
- Headers: backprop.h
- OpenCL kernels: backprop_kernel.cl
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### bfs

- **Complexity**: graph traversal (breadth-first search)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/bfs`)
- Sources: bfs.cu, kernel.cu, kernel2.cu
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/bfs`)
- Sources: bfs.cpp
- Build: make (Makefile)
- Verification: prints results

**opencl** (`opencl/bfs`)
- Sources: bfs.cpp, timer.cc
- Headers: CLHelper.h, timer.h, util.h
- OpenCL kernels: Kernels.cl
- Build: make (Makefile)
- Verification: numerical error check; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: NV_OPENCL_DIR, OPENCL_DIR, OPENCL_INC, OPENCL_LIB

### cfd

- **Complexity**: computational fluid dynamics (Euler solver)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/cfd`)
- Sources: euler3d.cu, euler3d_double.cu, pre_euler3d.cu, pre_euler3d_double.cu
- Build: make (Makefile)
- Verification: pass/fail output; prints results
- Notes: includes common/make.config; env vars: SDK_DIR

**omp** (`openmp/cfd`)
- Sources: euler3d_cpu.cpp, euler3d_cpu_double.cpp, pre_euler3d_cpu.cpp, pre_euler3d_cpu_double.cpp
- Build: make (makefile)
- Verification: prints results

**opencl** (`opencl/cfd`)
- Sources: euler3d.cpp
- Headers: CLHelper.h, util.h
- OpenCL kernels: Kernels.cl
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### dwt2d

- **Complexity**: 2D discrete wavelet transform
- **HeCBench overlap**: No
- **APIs**: cuda, opencl

**cuda** (`cuda/dwt2d`)
- Sources: components.cu, dwt.cu, dwt_cuda/common.cu, dwt_cuda/fdwt53.cu, dwt_cuda/fdwt97.cu, dwt_cuda/rdwt53.cu, dwt_cuda/rdwt97.cu, main.cu
- Headers: common.h, components.h, dwt.h, dwt_cuda/common.h, dwt_cuda/dwt.h, dwt_cuda/io.h, dwt_cuda/transform_buffer.h
- Build: make (Makefile)
- Data: result.txt
- Verification: golden reference; pass/fail output; prints results
- Notes: includes common/make.config; env vars: CUDA_DIR; subdirs: dwt_cuda

**opencl** (`opencl/dwt2d`)
- Sources: components.cpp, dwt.cpp, main.cpp
- Headers: common.h, components.h, dwt.h, dwt_cl/common.h, dwt_cl/dwt_cl.h
- OpenCL kernels: com_dwt.cl
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: dwt_cl

### gaussian

- **Complexity**: Gaussian elimination
- **HeCBench overlap**: YES
- **APIs**: cuda, opencl

**cuda** (`cuda/gaussian`)
- Sources: gaussian.cu
- Build: make (Makefile)
- Data: README.txt
- Verification: golden reference; prints results; result comparison
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**opencl** (`opencl/gaussian`)
- Sources: OriginalParallel.c, clutils.cpp, gaussianElim.cpp, gettimeofday.cpp, utils.cpp
- Headers: clutils.h, gaussianElim.h, gettimeofday.h, utils.h
- OpenCL kernels: gaussianElim_kernels.cl
- Build: make (Makefile)
- Data: README.txt
- Verification: golden reference; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### heartwall

- **Complexity**: heart wall tracking (image processing)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/heartwall`)
- Sources: AVI/avilib.c, AVI/avimod.c, define.c, kernel.cu, main.cu, setdevice.cu
- Headers: AVI/avilib.h, AVI/avimod.h
- Build: make (AVI/makefile, Makefile)
- Data: result.txt
- Verification: golden reference; numerical error check; pass/fail output; prints results
- Notes: subdirs: AVI

**omp** (`openmp/heartwall`)
- Sources: AVI/avilib.c, AVI/avimod.c, define.c, kernel.c, main.c
- Headers: AVI/avilib.h, AVI/avimod.h
- Build: make (AVI/makefile, makefile)
- Data: result.txt
- Verification: golden reference; numerical error check; pass/fail output; prints results
- Notes: subdirs: AVI

**opencl** (`opencl/heartwall`)
- Sources: kernel/kernel_gpu_opencl_wrapper.c, main.c, util/avi/avilib.c, util/avi/avimod.c, util/file/file.c, util/opencl/opencl.c, util/timer/timer.c
- Headers: kernel/kernel_gpu_opencl_wrapper.h, main.h, util/avi/avilib.h, util/avi/avimod.h, util/file/file.h, util/opencl/opencl.h, util/timer/timer.h
- OpenCL kernels: kernel/kernel_gpu_opencl.cl
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results
- Notes: includes common/make.config; env vars: NV_OPENCL_DIR, OPENCL_DIR, OPENCL_INC, OPENCL_LIB; subdirs: info, kernel, util

### hotspot

- **Complexity**: thermal simulation (2D stencil)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/hotspot`)
- Sources: hotspot.cu
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/hotspot`)
- Sources: hotspot_openmp.cpp
- Build: make (Makefile)
- Verification: prints results

**opencl** (`opencl/hotspot`)
- Sources: OpenCL_helper_library.c, hotspot.c
- Headers: OpenCL_helper_library.h, hotspot.h
- OpenCL kernels: hotspot_kernel.cl
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### hotspot3D

- **Complexity**: thermal simulation (3D stencil)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/hotspot3D`)
- Sources: 3D.cu, opt1.cu
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/hotspot3D`)
- Sources: 3D.c
- Build: make (Makefile)
- Verification: prints results

**opencl** (`opencl/hotspot3D`)
- Sources: 3D.c, CL_helper.c
- Headers: CL_helper.h
- OpenCL kernels: hotspotKernel.cl
- Build: make (Makefile)
- Verification: pass/fail output; prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### huffman

- **Complexity**: Huffman encoding/decoding
- **HeCBench overlap**: No
- **APIs**: cuda

**cuda** (`cuda/huffman`)
- Sources: cpuencode.cpp, hist.cu, main_test_cu.cu, pabio_kernels_v2.cu, pack_kernels.cu, scan.cu, scanLargeArray_kernel.cu, stats_logger.cpp, vlc_kernel_sm64huff.cu
- Headers: comparison_helpers.h, cpuencode.h, cuda_helpers.h, cutil.h, huffTree.h, load_data.h, parameters.h, print_helpers.h, stats_logger.h, stdafx.h, testdatagen.h
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison

### hybridsort

- **Complexity**: hybrid sorting (merge + bucket sort)
- **HeCBench overlap**: No
- **APIs**: cuda, opencl

**cuda** (`cuda/hybridsort`)
- Sources: bucketsort.cu, bucketsort_kernel.cu, histogram1024_kernel.cu, main.cu, mergesort.cu, mergesort_kernel.cu
- Headers: bucketsort.cuh, exception.h, helper_cuda.h, helper_string.h, helper_timer.h, mergesort.cuh
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: CUDA_DIR

**opencl** (`opencl/hybridsort`)
- Sources: bucketsort.c, hybridsort.c, mergesort.c
- Headers: bucketsort.h, mergesort.h
- OpenCL kernels: bucketsort_kernels.cl, histogram1024.cl, mergesort.cl
- Build: make (Makefile)
- Data: hybridinput.txt, hybridoutput.txt
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### kmeans

- **Complexity**: k-means clustering
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/kmeans`)
- Sources: cluster.c, getopt.c, kmeans.c, kmeans_clustering.c, kmeans_cuda.cu, kmeans_cuda_kernel.cu, rmse.c
- Headers: getopt.h, kmeans.h, unistd.h
- Build: make (Makefile)
- Verification: golden reference; numerical error check; prints results; result comparison
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/kmeans`)
- Sources: kmeans_openmp/cluster.c, kmeans_openmp/getopt.c, kmeans_openmp/kmeans.c, kmeans_openmp/kmeans_clustering.c, kmeans_serial/cluster.c, kmeans_serial/getopt.c, kmeans_serial/kmeans.c, kmeans_serial/kmeans_clustering.c
- Headers: kmeans_openmp/getopt.h, kmeans_openmp/kmeans.h, kmeans_openmp/unistd.h, kmeans_serial/getopt.h, kmeans_serial/kmeans.h, kmeans_serial/unistd.h
- Build: make (Makefile, kmeans_openmp/Makefile, kmeans_serial/Makefile)
- Verification: golden reference; prints results; result comparison
- Notes: subdirs: kmeans_openmp, kmeans_serial

**opencl** (`opencl/kmeans`)
- Sources: cluster.c, getopt.c, kmeans.cpp, kmeans_clustering.c, read_input.c, rmse.c
- Headers: getopt.h, kmeans.h, unistd.h
- OpenCL kernels: kmeans.cl
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### lavaMD

- **Complexity**: molecular dynamics (Lennard-Jones potential)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/lavaMD`)
- Sources: kernel/kernel_gpu_cuda.cu, kernel/kernel_gpu_cuda_wrapper.cu, main.c, util/device/device.cu, util/num/num.c, util/timer/timer.c
- Headers: kernel/kernel_gpu_cuda_wrapper.h, main.h, util/device/device.h, util/num/num.h, util/timer/timer.h
- Build: make (makefile)
- Data: result.txt
- Verification: prints results
- Notes: includes common/make.config; env vars: CUDA_DIR; subdirs: kernel, util

**omp** (`openmp/lavaMD`)
- Sources: kernel/kernel_cpu.c, main.c, util/device/device.cu, util/num/num.c, util/timer/timer.c
- Headers: kernel/kernel_cpu.h, main.h, util/device/device.h, util/num/num.h, util/timer/timer.h
- Build: make (makefile)
- Verification: prints results
- Notes: subdirs: kernel, util

**opencl** (`opencl/lavaMD`)
- Sources: kernel/kernel_gpu_opencl_wrapper.c, main.c, util/cuda/cuda.cu, util/num/num.c, util/opencl/opencl.c, util/timer/timer.c
- Headers: kernel/kernel_gpu_opencl_wrapper.h, main.h, util/cuda/cuda.h, util/num/num.h, util/opencl/opencl.h, util/timer/timer.h
- OpenCL kernels: kernel/kernel_gpu_opencl.cl
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: kernel, util

### leukocyte

- **Complexity**: leukocyte tracking (image processing)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/leukocyte`)
- Sources: CUDA/avilib.c, CUDA/detect_main.c, CUDA/find_ellipse.c, CUDA/find_ellipse_kernel.cu, CUDA/misc_math.c, CUDA/track_ellipse.c, CUDA/track_ellipse_kernel.cu, meschach_lib/MACHINES/RS6000/machine.c, meschach_lib/arnoldi.c, meschach_lib/bdfactor.c, meschach_lib/bkpfacto.c, meschach_lib/chfactor.c, meschach_lib/conjgrad.c, meschach_lib/copy.c, meschach_lib/dmacheps.c, meschach_lib/err.c, meschach_lib/extras.c, meschach_lib/fft.c, meschach_lib/fmacheps.c, meschach_lib/givens.c, meschach_lib/hessen.c, meschach_lib/hsehldr.c, meschach_lib/init.c, meschach_lib/iotort.c, meschach_lib/iter0.c, meschach_lib/iternsym.c, meschach_lib/itersym.c, meschach_lib/itertort.c, meschach_lib/ivecop.c, meschach_lib/lanczos.c, meschach_lib/lufactor.c, meschach_lib/machine.c, meschach_lib/matlab.c, meschach_lib/matop.c, meschach_lib/matrixio.c, meschach_lib/maxint.c, meschach_lib/meminfo.c, meschach_lib/memory.c, meschach_lib/memstat.c, meschach_lib/memtort.c, meschach_lib/mfunc.c, meschach_lib/mfuntort.c, meschach_lib/norm.c, meschach_lib/otherio.c, meschach_lib/pxop.c, meschach_lib/qrfactor.c, meschach_lib/schur.c, meschach_lib/solve.c, meschach_lib/sparse.c, meschach_lib/sparseio.c, meschach_lib/spbkp.c, meschach_lib/spchfctr.c, meschach_lib/splufctr.c, meschach_lib/sprow.c, meschach_lib/spswap.c, meschach_lib/sptort.c, meschach_lib/submat.c, meschach_lib/svd.c, meschach_lib/symmeig.c, meschach_lib/torture.c, meschach_lib/tutadv.c, meschach_lib/tutorial.c, meschach_lib/update.c, meschach_lib/vecop.c, meschach_lib/version.c, meschach_lib/zcopy.c, meschach_lib/zfunc.c, meschach_lib/zgivens.c, meschach_lib/zhessen.c, meschach_lib/zhsehldr.c, meschach_lib/zlufctr.c, meschach_lib/zmachine.c, meschach_lib/zmatio.c, meschach_lib/zmatlab.c, meschach_lib/zmatop.c, meschach_lib/zmemory.c, meschach_lib/znorm.c, meschach_lib/zqrfctr.c, meschach_lib/zschur.c, meschach_lib/zsolve.c, meschach_lib/ztorture.c, meschach_lib/zvecop.c
- Headers: CUDA/avilib.h, CUDA/find_ellipse.h, CUDA/find_ellipse_kernel.h, CUDA/misc_math.h, CUDA/track_ellipse.h, CUDA/track_ellipse_kernel.h, meschach_lib/MACHINES/Cray/machine.h, meschach_lib/MACHINES/GCC/machine.h, meschach_lib/MACHINES/Linux/machine.h, meschach_lib/MACHINES/MicroSoft/machine.h, meschach_lib/MACHINES/OS2/machine.h, meschach_lib/MACHINES/RS6000/machine.h, meschach_lib/MACHINES/SGI/machine.h, meschach_lib/MACHINES/SPARC/machine.h, meschach_lib/MACHINES/ThinkC/TC-machine-2.h, meschach_lib/MACHINES/ThinkC/TC-machine.h, meschach_lib/MACHINES/ThinkC/machine.h, meschach_lib/MACHINES/TurboC/machine.h, meschach_lib/MACHINES/WatcomPC/machine.h, meschach_lib/confdefs.h, meschach_lib/err.h, meschach_lib/iter.h, meschach_lib/machine.h, meschach_lib/matlab.h, meschach_lib/matrix.h, meschach_lib/matrix2.h, meschach_lib/meminfo.h, meschach_lib/oldnames.h, meschach_lib/sparse.h, meschach_lib/sparse2.h, meschach_lib/zmatrix.h, meschach_lib/zmatrix2.h
- Build: make (CUDA/Makefile, Makefile, meschach_lib/MACHINES/Cray/makefile, meschach_lib/MACHINES/GCC/makefile, meschach_lib/MACHINES/Linux/makefile, meschach_lib/MACHINES/MicroSoft/makefile, meschach_lib/MACHINES/OS2/makefile, meschach_lib/MACHINES/RS6000/makefile, meschach_lib/MACHINES/SGI/makefile, meschach_lib/MACHINES/SPARC/makefile, meschach_lib/makefile)
- Data: meschach_lib/DOC/fnindex.txt, meschach_lib/DOC/tutorial.txt, meschach_lib/ls.dat, meschach_lib/rk4.dat, result.txt
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: subdirs: CUDA, meschach_lib

**omp** (`openmp/leukocyte`)
- Sources: OpenMP/avilib.c, OpenMP/detect_main.c, OpenMP/find_ellipse.c, OpenMP/misc_math.c, OpenMP/track_ellipse.c, meschach_lib/MACHINES/RS6000/machine.c, meschach_lib/arnoldi.c, meschach_lib/bdfactor.c, meschach_lib/bkpfacto.c, meschach_lib/chfactor.c, meschach_lib/conjgrad.c, meschach_lib/copy.c, meschach_lib/dmacheps.c, meschach_lib/err.c, meschach_lib/extras.c, meschach_lib/fft.c, meschach_lib/fmacheps.c, meschach_lib/givens.c, meschach_lib/hessen.c, meschach_lib/hsehldr.c, meschach_lib/init.c, meschach_lib/iotort.c, meschach_lib/iter0.c, meschach_lib/iternsym.c, meschach_lib/itersym.c, meschach_lib/itertort.c, meschach_lib/ivecop.c, meschach_lib/lanczos.c, meschach_lib/lufactor.c, meschach_lib/machine.c, meschach_lib/matlab.c, meschach_lib/matop.c, meschach_lib/matrixio.c, meschach_lib/maxint.c, meschach_lib/meminfo.c, meschach_lib/memory.c, meschach_lib/memstat.c, meschach_lib/memtort.c, meschach_lib/mfunc.c, meschach_lib/mfuntort.c, meschach_lib/norm.c, meschach_lib/otherio.c, meschach_lib/pxop.c, meschach_lib/qrfactor.c, meschach_lib/schur.c, meschach_lib/solve.c, meschach_lib/sparse.c, meschach_lib/sparseio.c, meschach_lib/spbkp.c, meschach_lib/spchfctr.c, meschach_lib/splufctr.c, meschach_lib/sprow.c, meschach_lib/spswap.c, meschach_lib/sptort.c, meschach_lib/submat.c, meschach_lib/svd.c, meschach_lib/symmeig.c, meschach_lib/torture.c, meschach_lib/tutadv.c, meschach_lib/tutorial.c, meschach_lib/update.c, meschach_lib/vecop.c, meschach_lib/version.c, meschach_lib/zcopy.c, meschach_lib/zfunc.c, meschach_lib/zgivens.c, meschach_lib/zhessen.c, meschach_lib/zhsehldr.c, meschach_lib/zlufctr.c, meschach_lib/zmachine.c, meschach_lib/zmatio.c, meschach_lib/zmatlab.c, meschach_lib/zmatop.c, meschach_lib/zmemory.c, meschach_lib/znorm.c, meschach_lib/zqrfctr.c, meschach_lib/zschur.c, meschach_lib/zsolve.c, meschach_lib/ztorture.c, meschach_lib/zvecop.c
- Headers: OpenMP/avilib.h, OpenMP/find_ellipse.h, OpenMP/misc_math.h, OpenMP/track_ellipse.h, meschach_lib/MACHINES/Cray/machine.h, meschach_lib/MACHINES/GCC/machine.h, meschach_lib/MACHINES/Linux/machine.h, meschach_lib/MACHINES/MicroSoft/machine.h, meschach_lib/MACHINES/OS2/machine.h, meschach_lib/MACHINES/RS6000/machine.h, meschach_lib/MACHINES/SGI/machine.h, meschach_lib/MACHINES/SPARC/machine.h, meschach_lib/MACHINES/ThinkC/TC-machine-2.h, meschach_lib/MACHINES/ThinkC/TC-machine.h, meschach_lib/MACHINES/ThinkC/machine.h, meschach_lib/MACHINES/TurboC/machine.h, meschach_lib/MACHINES/WatcomPC/machine.h, meschach_lib/confdefs.h, meschach_lib/err.h, meschach_lib/iter.h, meschach_lib/machine.h, meschach_lib/matlab.h, meschach_lib/matrix.h, meschach_lib/matrix2.h, meschach_lib/meminfo.h, meschach_lib/oldnames.h, meschach_lib/sparse.h, meschach_lib/sparse2.h, meschach_lib/zmatrix.h, meschach_lib/zmatrix2.h
- Build: make (Makefile, OpenMP/Makefile, meschach_lib/MACHINES/Cray/makefile, meschach_lib/MACHINES/GCC/makefile, meschach_lib/MACHINES/Linux/makefile, meschach_lib/MACHINES/MicroSoft/makefile, meschach_lib/MACHINES/OS2/makefile, meschach_lib/MACHINES/RS6000/makefile, meschach_lib/MACHINES/SGI/makefile, meschach_lib/MACHINES/SPARC/makefile, meschach_lib/makefile)
- Data: meschach_lib/DOC/fnindex.txt, meschach_lib/DOC/tutorial.txt, meschach_lib/ls.dat, meschach_lib/rk4.dat
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: subdirs: OpenMP, meschach_lib

**opencl** (`opencl/leukocyte`)
- Sources: OpenCL/OpenCL_helper_library.c, OpenCL/avilib.c, OpenCL/detect_main.c, OpenCL/find_ellipse.c, OpenCL/find_ellipse_opencl.c, OpenCL/misc_math.c, OpenCL/track_ellipse.c, OpenCL/track_ellipse_opencl.c, meschach_lib/MACHINES/RS6000/machine.c, meschach_lib/arnoldi.c, meschach_lib/bdfactor.c, meschach_lib/bkpfacto.c, meschach_lib/chfactor.c, meschach_lib/conjgrad.c, meschach_lib/copy.c, meschach_lib/dmacheps.c, meschach_lib/err.c, meschach_lib/extras.c, meschach_lib/fft.c, meschach_lib/fmacheps.c, meschach_lib/givens.c, meschach_lib/hessen.c, meschach_lib/hsehldr.c, meschach_lib/init.c, meschach_lib/iotort.c, meschach_lib/iter0.c, meschach_lib/iternsym.c, meschach_lib/itersym.c, meschach_lib/itertort.c, meschach_lib/ivecop.c, meschach_lib/lanczos.c, meschach_lib/lufactor.c, meschach_lib/machine.c, meschach_lib/matlab.c, meschach_lib/matop.c, meschach_lib/matrixio.c, meschach_lib/maxint.c, meschach_lib/meminfo.c, meschach_lib/memory.c, meschach_lib/memstat.c, meschach_lib/memtort.c, meschach_lib/mfunc.c, meschach_lib/mfuntort.c, meschach_lib/norm.c, meschach_lib/otherio.c, meschach_lib/pxop.c, meschach_lib/qrfactor.c, meschach_lib/schur.c, meschach_lib/solve.c, meschach_lib/sparse.c, meschach_lib/sparseio.c, meschach_lib/spbkp.c, meschach_lib/spchfctr.c, meschach_lib/splufctr.c, meschach_lib/sprow.c, meschach_lib/spswap.c, meschach_lib/sptort.c, meschach_lib/submat.c, meschach_lib/svd.c, meschach_lib/symmeig.c, meschach_lib/torture.c, meschach_lib/tutadv.c, meschach_lib/tutorial.c, meschach_lib/update.c, meschach_lib/vecop.c, meschach_lib/version.c, meschach_lib/zcopy.c, meschach_lib/zfunc.c, meschach_lib/zgivens.c, meschach_lib/zhessen.c, meschach_lib/zhsehldr.c, meschach_lib/zlufctr.c, meschach_lib/zmachine.c, meschach_lib/zmatio.c, meschach_lib/zmatlab.c, meschach_lib/zmatop.c, meschach_lib/zmemory.c, meschach_lib/znorm.c, meschach_lib/zqrfctr.c, meschach_lib/zschur.c, meschach_lib/zsolve.c, meschach_lib/ztorture.c, meschach_lib/zvecop.c
- Headers: OpenCL/OpenCL_helper_library.h, OpenCL/avilib.h, OpenCL/find_ellipse.h, OpenCL/find_ellipse_opencl.h, OpenCL/misc_math.h, OpenCL/track_ellipse.h, OpenCL/track_ellipse_opencl.h, meschach_lib/MACHINES/Cray/machine.h, meschach_lib/MACHINES/GCC/machine.h, meschach_lib/MACHINES/Linux/machine.h, meschach_lib/MACHINES/MicroSoft/machine.h, meschach_lib/MACHINES/OS2/machine.h, meschach_lib/MACHINES/RS6000/machine.h, meschach_lib/MACHINES/SGI/machine.h, meschach_lib/MACHINES/SPARC/machine.h, meschach_lib/MACHINES/ThinkC/TC-machine-2.h, meschach_lib/MACHINES/ThinkC/TC-machine.h, meschach_lib/MACHINES/ThinkC/machine.h, meschach_lib/MACHINES/TurboC/machine.h, meschach_lib/MACHINES/WatcomPC/machine.h, meschach_lib/confdefs.h, meschach_lib/err.h, meschach_lib/iter.h, meschach_lib/machine.h, meschach_lib/matlab.h, meschach_lib/matrix.h, meschach_lib/matrix2.h, meschach_lib/meminfo.h, meschach_lib/oldnames.h, meschach_lib/sparse.h, meschach_lib/sparse2.h, meschach_lib/zmatrix.h, meschach_lib/zmatrix2.h
- OpenCL kernels: OpenCL/find_ellipse_kernel.cl, OpenCL/track_ellipse_kernel.cl, OpenCL/track_ellipse_kernel_opt.cl, find_ellipse_kernel.cl, track_ellipse_kernel.cl, track_ellipse_kernel_opt.cl
- Build: make (Makefile, OpenCL/Makefile, meschach_lib/MACHINES/Cray/Makefile, meschach_lib/MACHINES/GCC/Makefile, meschach_lib/MACHINES/Linux/Makefile, meschach_lib/MACHINES/MicroSoft/Makefile, meschach_lib/MACHINES/OS2/Makefile, meschach_lib/MACHINES/RS6000/Makefile, meschach_lib/MACHINES/SGI/Makefile, meschach_lib/MACHINES/SPARC/Makefile, meschach_lib/Makefile)
- Data: meschach_lib/DOC/fnindex.txt, meschach_lib/DOC/tutorial.txt, meschach_lib/ls.dat, meschach_lib/rk4.dat
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: subdirs: OpenCL, meschach_lib

### lud

- **Complexity**: LU decomposition
- **HeCBench overlap**: YES
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/lud`)
- Sources: base/lud.c, base/lud_base.c, common/common.c, cuda/lud.cu, cuda/lud_kernel.cu, tools/gen_input.c
- Headers: common/common.h
- Build: make (Makefile, base/Makefile, cuda/Makefile, tools/Makefile)
- Verification: prints results; result comparison
- Notes: subdirs: base, common, cuda, tools

**omp** (`openmp/lud`)
- Sources: base/lud.c, base/lud_base.c, common/common.c, omp/lud.c, omp/lud_omp.c, tools/gen_input.c
- Headers: common/common.h
- Build: make (Makefile, base/Makefile, omp/Makefile, tools/Makefile)
- Verification: prints results; result comparison
- Notes: subdirs: base, common, omp, tools

**opencl** (`opencl/lud`)
- Sources: base/lud.c, base/lud_base.c, common/common.c, ocl/lud.cpp, tools/gen_input.c
- Headers: common/common.h
- OpenCL kernels: lud_kernel.cl
- Build: make (Makefile, base/Makefile, ocl/Makefile, tools/Makefile)
- Verification: pass/fail output; prints results; result comparison
- Notes: subdirs: base, common, ocl, tools

### mummergpu

- **Complexity**: DNA sequence alignment
- **HeCBench overlap**: No
- **APIs**: cuda, omp

**cuda** (`cuda/mummergpu`)
- Sources: src/PoolMalloc.cpp, src/common.cu, src/morton.c, src/mummergpu.cu, src/mummergpu_gold.cpp, src/mummergpu_kernel.cu, src/mummergpu_main.cpp, src/smith-waterman.cpp, src/suffix-tree.cpp
- Headers: src/PoolMalloc.hh, src/mummergpu.h
- Build: make (Makefile, experiments/rules.mk, experiments/test_rule.mk, src/Makefile)
- Data: data/README, data/shortqry.fa, data/shortref.fa, data/timing/gentestdata.py, data/timing/global.cnf, data/timing/instructions.cnf, data/timing/timing-exp.sh, src/michaelinput.txt, src/results.txt
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: subdirs: data, experiments, src

**omp** (`openmp/mummergpu`)
- Sources: src/PoolMalloc.cpp, src/common.cu, src/morton.c, src/mummergpu.cu, src/mummergpu_gold.cpp, src/mummergpu_kernel.cu, src/mummergpu_main.cpp, src/smith-waterman.cpp, src/suffix-tree.cpp
- Headers: src/PoolMalloc.hh, src/mummergpu.h
- Build: make (Makefile, experiments/rules.mk, experiments/test_rule.mk, src/Makefile)
- Data: data/README, data/shortqry.fa, data/shortref.fa, data/timing/gentestdata.py, data/timing/global.cnf, data/timing/instructions.cnf, data/timing/timing-exp.sh, src/michaelinput.txt, src/results.txt
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: subdirs: data, experiments, src

### myocyte

- **Complexity**: cardiac myocyte simulation (ODE solver)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/myocyte`)
- Sources: define.c, embedded_fehlberg_7_8.cu, embedded_fehlberg_7_8_2.cu, file.c, kernel.cu, kernel_2.cu, kernel_cam.cu, kernel_cam_2.cu, kernel_ecc.cu, kernel_ecc_2.cu, kernel_fin.cu, kernel_fin_2.cu, main.cu, master.cu, solver.cu, solver_2.cu, timer.c, work.cu, work_2.cu
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison

**omp** (`openmp/myocyte`)
- Sources: cam.c, define.c, ecc.c, embedded_fehlberg_7_8.c, file.c, fin.c, main.c, master.c, solver.c, timer.c
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison

**opencl** (`opencl/myocyte`)
- Sources: kernel/embedded_fehlberg_7_8.c, kernel/kernel_fin.c, kernel/kernel_gpu_opencl_wrapper.c, kernel/master.c, kernel/solver.c, main.c, util/file/file.c, util/num/num.c, util/opencl/opencl.c, util/timer/timer.c
- Headers: common.h, kernel/kernel_gpu_opencl_wrapper.h, main.h, util/file/file.h, util/num/num.h, util/opencl/opencl.h, util/timer/timer.h
- OpenCL kernels: kernel/kernel_gpu_opencl.cl
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: kernel, util

### nn

- **Complexity**: nearest neighbor search
- **HeCBench overlap**: YES
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/nn`)
- Sources: hurricane_gen.c, nn_cuda.cu
- Build: make (Makefile)
- Data: gen_dataset.sh
- Verification: pass/fail output; prints results
- Notes: includes common/make.config; env vars: CUDA_DIR

**omp** (`openmp/nn`)
- Sources: hurricane_gen.c, nn_openmp.c
- Build: make (Makefile)
- Data: gen_dataset.sh
- Verification: pass/fail output; prints results; result comparison

**opencl** (`opencl/nn`)
- Sources: clutils.cpp, gettimeofday.cpp, nearestNeighbor.cpp, utils.cpp
- Headers: clutils.h, gettimeofday.h, ipoint.h, nearestNeighbor.h, utils.h
- OpenCL kernels: nearestNeighbor_kernel.cl
- Build: make (Makefile)
- Data: README.txt, filelist.txt
- Verification: golden reference; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### nw

- **Complexity**: Needleman-Wunsch sequence alignment
- **HeCBench overlap**: YES
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/nw`)
- Sources: needle.cu, needle_kernel.cu
- Headers: needle.h
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/nw`)
- Sources: needle.cpp
- Build: make (Makefile)
- Verification: golden reference; prints results

**opencl** (`opencl/nw`)
- Sources: nw.c
- OpenCL kernels: nw.cl
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### particlefilter

- **Complexity**: particle filter (Bayesian estimation)
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/particlefilter`)
- Sources: ex_particle_CUDA_float_seq.cu, ex_particle_CUDA_naive_seq.cu
- Build: make (Makefile)
- Data: README.txt
- Verification: golden reference; prints results; result comparison
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/particlefilter`)
- Sources: ex_particle_OPENMP_seq.c
- Build: make (Makefile)
- Verification: golden reference; prints results; result comparison

**opencl** (`opencl/particlefilter`)
- Sources: ex_particle_OCL_double_seq.cpp, ex_particle_OCL_naive_seq.cpp, ex_particle_OCL_single_seq.cpp
- OpenCL kernels: particle_double.cl, particle_naive.cl, particle_single.cl
- Build: make (Makefile)
- Data: README.txt
- Verification: golden reference; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### pathfinder

- **Complexity**: dynamic programming path finding
- **HeCBench overlap**: YES
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/pathfinder`)
- Sources: pathfinder.cu
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: CUDA_DIR, CUDA_LIB_DIR

**omp** (`openmp/pathfinder`)
- Sources: pathfinder.cpp
- Headers: timer.h
- Build: make (Makefile)
- Data: README.txt
- Verification: prints results

**opencl** (`opencl/pathfinder`)
- Sources: OpenCL.cpp, main.cpp
- Headers: OpenCL.h
- OpenCL kernels: kernels.cl
- Build: make (Makefile)
- Verification: pass/fail output; prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB

### rng

- **Complexity**: random number generation
- **HeCBench overlap**: No
- **APIs**: others

**others** (`others/rng`)
- Sources: rng/rng.c, rng/rng.cu
- Build: make (rng/latex/Makefile)
- Data: README.txt
- Verification: unknown
- Notes: in others/ directory

### srad

- **Complexity**: speckle reducing anisotropic diffusion
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/srad`)
- Sources: srad_v1/compress_kernel.cu, srad_v1/define.c, srad_v1/device.c, srad_v1/extract_kernel.cu, srad_v1/graphics.c, srad_v1/main.cu, srad_v1/prepare_kernel.cu, srad_v1/reduce_kernel.cu, srad_v1/resize.c, srad_v1/srad2_kernel.cu, srad_v1/srad_kernel.cu, srad_v1/timer.c, srad_v2/srad.cu, srad_v2/srad_kernel.cu
- Headers: srad_v1/include.h, srad_v2/srad.h
- Build: make (Makefile, srad_v1/makefile, srad_v2/Makefile)
- Data: srad_v1/image_out.pgm
- Verification: prints results
- Notes: subdirs: srad_v1, srad_v2

**omp** (`openmp/srad`)
- Sources: srad_v1/define.c, srad_v1/graphics.c, srad_v1/main.c, srad_v1/resize.c, srad_v1/timer.c, srad_v2/srad.cpp
- Headers: srad_v1/include.h
- Build: make (Makefile, srad_v1/makefile, srad_v2/Makefile)
- Verification: prints results
- Notes: subdirs: srad_v1, srad_v2

**opencl** (`opencl/srad`)
- Sources: kernel/compress_kernel.c, kernel/extract_kernel.c, kernel/kernel_gpu_opencl_wrapper.c, kernel/prepare_kernel.c, kernel/reduce_kernel.c, kernel/srad2_kernel.c, kernel/srad_kernel.c, main.c, util/graphics/graphics.c, util/graphics/resize.c, util/opencl/opencl.c, util/timer/timer.c
- Headers: kernel/kernel_gpu_opencl_wrapper.h, main.h, util/graphics/graphics.h, util/graphics/resize.h, util/opencl/opencl.h, util/timer/timer.h
- OpenCL kernels: kernel/kernel_gpu_opencl.cl
- Build: make (Makefile)
- Verification: prints results
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB; subdirs: info, kernel, output, util

### streamcluster

- **Complexity**: online stream clustering
- **HeCBench overlap**: No
- **APIs**: cuda, omp, opencl

**cuda** (`cuda/streamcluster`)
- Sources: streamcluster_cuda.cu, streamcluster_cuda_cpu.cpp, streamcluster_header.cu, streamcluster_original.cpp
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results
- Notes: includes common/make.config; env vars: CUDA_DIR

**omp** (`openmp/streamcluster`)
- Sources: streamcluster_omp.cpp, streamcluster_original.cpp
- Build: make (Makefile)
- Verification: golden reference; pass/fail output; prints results

**opencl** (`opencl/streamcluster`)
- Sources: streamcluster.cpp
- Headers: CLHelper.h, streamcluster.h, streamcluster_cl.h, util.h
- OpenCL kernels: Kernels.cl
- Build: make (Makefile)
- Verification: golden reference; numerical error check; pass/fail output; prints results; result comparison
- Notes: includes common/make.config; env vars: OPENCL_INC, OPENCL_LIB
