# HeCBench Kernel Source Inspection Report

**Date:** 2025-03-02  
**Source:** `/home/samyak/Downloads/HeCBench-master/src/`

---

## 1. triad

| Property | Details |
|---|---|
| **Files in cuda dir** | `triad.cu` (274L), `main.cpp` (67L), `Option.cpp` (14L), `OptionParser.cpp` (388L), `Timer.cpp` (304L), `config.h` (163L), `Option.h` (35L), `OptionParser.h` (65L), `Timer.h` (75L), `Utility.h` (98L), `LICENSE`, `Makefile`, `CMakeLists.txt` |
| **Main source files** | `triad.cu` (kernel + RunBenchmark), `main.cpp` (entry point) |
| **Header files & roles** | `OptionParser.h/.cpp` — CLI parsing; `Timer.h/.cpp` — timing; `Utility.h` — helpers (HumanReadable, SplitValues, drand48 compat); `config.h` — autoconf defines; `Option.h/.cpp` — option struct |
| **Verification method** | Compares two halves of host array element-by-element. Prints `"PASS"` or `"FAIL"` (also `"Error; hostMem[..."`) |
| **Executable name** | `triad` (from Makefile: `program = triad`) |
| **Run arguments** | No positional args — uses `--passes 100 -v` via OptionParser flags |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | **Many support files** (10 source files total). The OptionParser/Timer library is a heavy SHOC framework dependency. Makefile source: `source = triad.cu main.cpp Option.cpp OptionParser.cpp Timer.cpp`. This is a significantly more complex build than most kernels. Verification per block-size is repeated 9 times (one per size step). |
| **Recommendation** | **Good** — Self-contained, no external data, has PASS/FAIL. But the large number of support files (5 .cpp + 5 .h) makes the prompt payload large (~1483 lines total). |

---

## 2. popcount

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (236L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `main.cu` (single file — everything) |
| **Header files & roles** | None (uses only standard headers) |
| **Verification method** | `checkResults()` compares GPU result to `popcount_ref()` per element. Prints `"Success"` or `"Fail"` — repeated 6 times (one per kernel variant pc1–pc6). |
| **Executable name** | `main` |
| **Run arguments** | `./main <length> <repeat>` → default: `16777216 1000` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | None. Very clean single-file kernel. Verification prints "Success"/"Fail" (not "PASS"/"FAIL"). |
| **Recommendation** | **Excellent** — Single file, self-contained, no data deps, clear verification. |

---

## 3. sobol

| Property | Details |
|---|---|
| **Files in cuda dir** | `sobol.cu` (189L), `sobol_gpu.cu` (196L), `sobol_gold.cu` (170L), `sobol_primitives.cu` (10251L — data table!), `sobol.h` (41L), `sobol_gpu.h` (45L), `sobol_gold.h` (43L), `sobol_primitives.h` (55L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `sobol.cu` (main + driver), `sobol_gpu.cu` (GPU kernel), `sobol_gold.cu` (CPU reference + direction init) |
| **Header files & roles** | `sobol.h` — defines `n_directions=32`; `sobol_gpu.h` — GPU function decl; `sobol_gold.h` — CPU function decls; `sobol_primitives.h` — primitive struct def + extern decl for data table |
| **Verification method** | Computes L1-Error between GPU and CPU outputs. Prints `"PASS"` if `l1error < 1e-6`, else `"FAIL"`. Also prints `"L1-Error: <value>"`. |
| **Executable name** | `main` |
| **Run arguments** | `./main <n_vectors> <n_dimensions> <repeat>` → default: `1000000 1000 100` |
| **External data needed?** | No (primitive data table is compiled in `sobol_primitives.cu` — 10K lines of static data) |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | `sobol_primitives.cu` is 10,251 lines of static data — this makes the total prompt payload ~11K lines if included. The data table is essential but is just a constant array. Multi-file build: `source = sobol.cu sobol_gold.cu sobol_gpu.cu sobol_primitives.cu`. |
| **Recommendation** | **Good** — Clean verification, self-contained. But `sobol_primitives.cu` is a massive data file (997KB) that inflates line count. Would need to truncate or exclude from prompt payload. |

---

## 4. convolution3D

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (299L), `conv3d_s4.cu` (129L — cuDNN variant, `#include`d conditionally), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `main.cu` (kernels + main + verification + reference) |
| **Header files & roles** | None. `conv3d_s4.cu` is only used when `CUDNN_CONV` is defined (not default). |
| **Verification method** | `verify()` template: compares output element-by-element with `fabs(Y[i] - Y_ref[i]) > 1e-3f`. Prints `"PASS"` or `"FAIL"`. Run 3 times (3 kernel variants s1/s2/s3). |
| **Executable name** | `main` |
| **Run arguments** | `./main <N> <C> <M> <Win> <Hin> <K> <repeat>` → default: `32 1 6 32 32 5 100` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | None significant. `conv3d_s4.cu` cuDNN path is opt-in and can be ignored. Clean template-based design. |
| **Recommendation** | **Excellent** — Single main file (ignore cuDNN), self-contained, clear PASS/FAIL, no data deps. |

---

## 5. bfs

| Property | Details |
|---|---|
| **Files in cuda dir** | `bfs.cu` (277L), `Makefile`, `CMakeLists.txt`, `README` |
| **Main source files** | `bfs.cu` (single file — kernels + main + CPU reference) |
| **Header files & roles** | `util.h` — included from `../bfs-sycl/util.h` (via `-I../bfs-sycl` in CFLAGS). Provides `compare_results<T>()` template. |
| **Verification method** | Runs BFS on both CPU and GPU, then `compare_results<int>(h_cost_ref, h_cost, no_of_nodes)` — prints `"Passed"` or `"Failed"`. |
| **Executable name** | `main` |
| **Run arguments** | `./main ../data/bfs/graph1MW_6.txt` |
| **External data needed?** | **YES** — Requires `../data/bfs/graph1MW_6.txt`, distributed as `graph1MW_6.txt.tar.bz` (22MB compressed). Must extract before running. Also README says "download dataset at http://lava.cs.virginia.edu/Rodinia/download.htm". |
| **OMP variant** | Compiler: `icpx`. Cross-directory include: `-I../bfs-sycl` for `util.h`. |
| **Issues/concerns** | **External data file required** — graph file must be extracted from tarball. Cross-directory include (`-I../bfs-sycl`). Input file is read via `fscanf` with custom format (Rodinia graph format). Verification prints "Passed"/"Failed" (not standard PASS/FAIL). |
| **Recommendation** | **Avoid** — External data dependency (must extract .tar.bz), cross-directory header include, non-standard verification string. |

---

## 6. histogram

| Property | Details |
|---|---|
| **Files in cuda dir** | `histogram_compare.cu` (664L), `histogram_gmem_atomics.h` (152L), `histogram_smem_atomics.h` (162L), `mersenne.h` (160L), `test_util.h` (747L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `histogram_compare.cu` (main + TGA parsing + test harness) |
| **Header files & roles** | `histogram_gmem_atomics.h` — global-memory atomics kernel; `histogram_smem_atomics.h` — shared-memory atomics kernel; `mersenne.h` — Mersenne twister RNG; `test_util.h` — CLI parsing, GPU timer, assert/compare utilities |
| **Verification method** | `HistogramGold()` computes CPU reference, `CompareDeviceResults()` (from `test_util.h`) checks. Prints `"PASS"` or `"FAIL"` after each test. Also `AssertEquals(0, compare)`. |
| **Executable name** | `main` |
| **Run arguments** | `./main --i=100` (uses random image generation by default) |
| **External data needed?** | No (generates random image internally; TGA file reading is optional via `--file=` flag) |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. OMP version uses different source names (`.hpp` variants). |
| **Issues/concerns** | `test_util.h` is 747 lines of CUDA-specific utility code (GpuTimer with cudaEvent, CommandLineArgs, etc.). The code uses CUDA-specific types (`float4`, `uchar4`, `uchar1`) heavily — these would be challenging for non-CUDA parsers. Multiple test genres: 1-channel, 3/4-channel, float4 tests. |
| **Recommendation** | **Good** — Self-contained, has PASS/FAIL, no mandatory external data. But the heavy use of CUDA vector types and the complex test_util.h make it moderately complex. Total ~1885 lines. |

---

## 7. d2q9-bgk

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (944L), `Makefile`, `CMakeLists.txt`, `test.tar.gz` (5.4MB) |
| **Main source files** | `main.cu` (single file — everything: kernel, init, file I/O, Reynolds calc) |
| **Header files & roles** | None |
| **Verification method** | **No in-program PASS/FAIL.** Writes `final_state.dat` and `av_vels.dat`, then a separate `make check` target runs `python check/check.py` to validate against reference files. Prints Reynolds number. |
| **Executable name** | `main` |
| **Run arguments** | `./main Inputs/input_256x256.params Obstacles/obstacles_256x256.dat` |
| **External data needed?** | **YES** — Requires `test.tar.gz` to be extracted (contains `Inputs/`, `Obstacles/`, and `check/` directories). The tarball is included in the dir (5.4MB). |
| **OMP variant** | Compiler: `icpx`. Has `-I../include` in CFLAGS. |
| **Issues/concerns** | **External data required** — must extract `test.tar.gz` before building/running. **No in-program PASS/FAIL** — validation is via separate Python script (`make check`). The single source file is large (944 lines). OMP variant references `../include`. |
| **Recommendation** | **Avoid** — External data dependency (tarball must be extracted), no in-program verification (relies on Python script + reference files), large source, complex Lattice Boltzmann physics. |

---

## 8. md5hash

| Property | Details |
|---|---|
| **Files in cuda dir** | `MD5Hash.cu` (620L), `LICENSE`, `Makefile`, `CMakeLists.txt` |
| **Main source files** | `MD5Hash.cu` (single file — MD5 algorithm, CPU/GPU search, main) |
| **Header files & roles** | None |
| **Verification method** | Brute-force searches for a known random key. Checks foundIndex, foundKey, foundDigest match. Prints `"PASS"` or `"FAIL"` per iteration/size. Also prints specific error strings like `"ERROR: mismatch in key index found."`. Run across 4 different problem sizes × passes. |
| **Executable name** | `MD5Hash` (non-standard name) |
| **Run arguments** | `./MD5Hash <offload> <passes>` → default: `1 4` (offload=1 means GPU, passes=4) |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | Executable name is `MD5Hash` (not `main`). The program runs CPU-only when offload=0 and GPU when offload=1. Multiple sizes are tested in a single run (4 sizes). The MD5 algorithm is all inline macros — complex but self-contained. |
| **Recommendation** | **Excellent** — Single file, no data deps, clear PASS/FAIL per iteration. Non-standard exe name is minor. |

---

## 9. mandelbrot

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (72L), `mandel.hpp` (214L), `common.hpp` (33L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `main.cu` (entry point + orchestration), `mandel.hpp` (Mandel classes, GPU kernel, serial reference) |
| **Header files & roles** | `mandel.hpp` — MandelParameters, Mandel base class, MandelSerial, MandelParallel, `mandel()` kernel, `Verify()` method; `common.hpp` — MyTimer duration helper |
| **Verification method** | `m_par.Verify(m_ser)` compares parallel vs serial element-by-element with 5% tolerance. On validation failure throws `std::runtime_error("Verification failure")`. If no exception: prints `"Success"`. Caught exception prints `"Failure"`. |
| **Executable name** | `main` |
| **Run arguments** | `./main <repeat>` → default: `1000` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. OMP uses `mandel.hpp` + `util.hpp` (different header name). |
| **Issues/concerns** | Verification prints "Success"/"Failure" (not PASS/FAIL). Uses `__host__ __device__` annotations heavily. The OMP variant uses a different utility header (`util.hpp` vs `common.hpp`). Total ~319 lines. |
| **Recommendation** | **Excellent** — Small, clean, self-contained. Verification via exception-based approach prints "Success"/"Failure". |

---

## 10. black-scholes

| Property | Details |
|---|---|
| **Files in cuda dir** | `blackScholesAnalyticEngine.cu` (327L), `blackScholesAnalyticEngineKernels.cu` (272L), `blackScholesAnalyticEngineKernelsCpu.cu` (266L), `blackScholesAnalyticEngineKernels.cuh` (74L), `blackScholesAnalyticEngineKernelsCpu.cuh` (70L), `blackScholesAnalyticEngineStructs.cuh` (132L), `errorFunctConsts.cuh` (110L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `blackScholesAnalyticEngine.cu` (main driver — `#include`s the kernel .cu files directly) |
| **Header files & roles** | `blackScholesAnalyticEngineStructs.cuh` — option structs; `blackScholesAnalyticEngineKernels.cuh/.cu` — GPU kernel; `blackScholesAnalyticEngineKernelsCpu.cuh/.cu` — CPU reference; `errorFunctConsts.cuh` — error function constants |
| **Verification method** | **No PASS/FAIL.** Compares GPU vs CPU via printed summation and individual output prices. Reports "Summation of output prices on GPU/CPU" and "Speedup on GPU". User must manually compare outputs. |
| **Executable name** | `main` |
| **Run arguments** | `./main <repeat>` → default: `100` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. OMP uses `.h` and `.cpp` variants of the same files. |
| **Issues/concerns** | **No automated PASS/FAIL verification.** The code only prints output sums and expects manual comparison. Uses `#include` of .cu files (non-standard include pattern). 7 source/header files total. Many long repetitive if-else blocks for 37 option settings. Total ~1251 lines. |
| **Recommendation** | **Good** (with caveats) — Self-contained, no data deps, but **no automated verification** (must parse output sums). Complex multi-file structure with unusual `#include .cu` pattern. |

---

## 11. nqueen

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (267L), `LICENSE`, `Makefile`, `CMakeLists.txt` |
| **Main source files** | `main.cu` (single file — everything) |
| **Header files & roles** | None |
| **Verification method** | For `size==15 && initialDepth==7`: checks `qtd_sols_global == 2279184 && tree_size == 171129071`. Prints `"PASS"` or `"FAIL"`. Also prints solution count and tree size. |
| **Executable name** | `main` |
| **Run arguments** | `./main <size> <initial_depth> <repeat>` → default: `15 7 100` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | Verification only works for the specific case `size=15, initialDepth=7` (hardcoded expected values). Allocates 75M elements × 16 bytes = ~1.2GB for `root_prefixes_h`. This is a large memory allocation. |
| **Recommendation** | **Excellent** — Single file, no data deps, clear PASS/FAIL for default params. Large memory usage is the only concern. |

---

## 12. randomAccess

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (152L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `main.cu` (single file — everything) |
| **Header files & roles** | None |
| **Verification method** | After GPU compute, runs CPU verification loop. Prints `"Found %llu errors in %llu locations (PASS/FAIL)."` — passes if errors ≤ 1% of table size. |
| **Executable name** | `main` |
| **Run arguments** | `./main <repeat>` → default: `10` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. |
| **Issues/concerns** | None. Very clean, minimal kernel. Uses 512MB table (TableSize ~33M entries × 8 bytes = ~256MB). Verification is embedded in output string with `(PASS)` or `(FAIL)` inline. |
| **Recommendation** | **Excellent** — Single file, smallest source (152L), no data deps, clear PASS/FAIL. |

---

## 13. hwt1d

| Property | Details |
|---|---|
| **Files in cuda dir** | `main.cu` (124L), `kernel.cu` (202L), `reference.cu` (49L), `hwt.h` (45L), `Makefile`, `CMakeLists.txt` |
| **Main source files** | `main.cu` (driver + verification), `kernel.cu` (GPU kernel + runKernel()), `reference.cu` (CPU reference) |
| **Header files & roles** | `hwt.h` — common includes, function declarations for `getLevels()`, `calApproxFinalOnHost()`, `runKernel()` |
| **Verification method** | Compares GPU output vs CPU host reference element-by-element with `fabs(dOutData[i] - hOutData[i]) > 0.1f`. Prints `"PASS"` or `"FAIL"`. |
| **Executable name** | `main` |
| **Run arguments** | `./main <signal_length> <repeat>` → default: `8388608 100` |
| **External data needed?** | No |
| **OMP variant** | Compiler: `icpx`. No cross-directory includes. OMP has `main.cpp` + `reference.cpp` (kernel inlined into main). |
| **Issues/concerns** | Multi-file build: `source = main.cu kernel.cu reference.cu`. But total is only 420 lines combined. Clean separation of concerns. Signal length must be power of 2. |
| **Recommendation** | **Excellent** — Clean multi-file but small total (420L), no data deps, clear PASS/FAIL. |

---

## Summary Table

| Kernel | Files | Total Lines | Ext. Data | Verification | OMP Compiler | Recommendation |
|---|---|---|---|---|---|---|
| **triad** | 10 src | ~1483 | No | PASS/FAIL | icpx | Good |
| **popcount** | 1 src | 236 | No | Success/Fail | icpx | **Excellent** |
| **sobol** | 4 src + headers | ~10990 (10K data) | No | PASS/FAIL | icpx | Good (large data file) |
| **convolution3D** | 1 src (+1 optional) | ~299 | No | PASS/FAIL | icpx | **Excellent** |
| **bfs** | 1 src + ext header | 277 | **YES** (graph file) | Passed/Failed | icpx | **Avoid** |
| **histogram** | 1 src + 4 headers | ~1885 | No | PASS/FAIL | icpx | Good |
| **d2q9-bgk** | 1 src | 944 | **YES** (tarball) | External Python | icpx | **Avoid** |
| **md5hash** | 1 src | 620 | No | PASS/FAIL | icpx | **Excellent** |
| **mandelbrot** | 1 src + 2 hpp | ~319 | No | Success/Failure | icpx | **Excellent** |
| **black-scholes** | 3 .cu + 4 .cuh | ~1251 | No | No auto verify | icpx | Good (no PASS/FAIL) |
| **nqueen** | 1 src | 267 | No | PASS/FAIL | icpx | **Excellent** |
| **randomAccess** | 1 src | 152 | No | PASS/FAIL | icpx | **Excellent** |
| **hwt1d** | 3 src + 1 header | ~420 | No | PASS/FAIL | icpx | **Excellent** |

### Key Notes

- **All OMP variants use `icpx`** (Intel DPC++ compiler) with `-fiopenmp -fopenmp-targets=spir64`.
- **Cross-directory includes:** Only `bfs` (`-I../bfs-sycl`) and `d2q9-bgk-omp` (`-I../include`).
- **Avoid:** `bfs` (external graph data + cross-dir include), `d2q9-bgk` (external data + no in-program verify).
- **Best candidates (Excellent):** `popcount`, `convolution3D`, `md5hash`, `mandelbrot`, `nqueen`, `randomAccess`, `hwt1d`.
