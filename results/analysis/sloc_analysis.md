# SLoC Characterization of ParBench Corpus Kernels

**35 kernels** analyzed across 5 suites: hecbench (10), mixbench (1), rodinia (22), rsbench (1), xsbench (1).
Physical SLoC = non-blank, non-comment lines (matches cloc methodology).
Source: actual CUDA source files from the benchmark repositories.

## Per-Kernel SLoC

| Kernel | Category | CUDA SLoC | Src Files | Tgt Files | OMP SLoC | OMP Files | Complexity | Pass Rate |
|--------|----------|----------:|----------:|----------:|---------:|----------:|------------|----------:|
| myocyte | other | 3,304 | 16 | 2 | 1806 | 10 | multi_to_multi | 0.0% |
| mummergpu | other | 2,773 | 3 | 2 | 5325 | 9 | multi_to_single | 0.0% |
| cfd | physics | 1,955 | 4 | 1 | 400 | 1 | single_file | 27.3% |
| xsbench | physics | 1,390 | 6 | 2 | 1238 | 6 | multi_to_multi | 4.5% |
| dwt2d | physics | 1,238 | 8 | 7 | N/A | N/A | unknown | 0.0% |
| heartwall | image | 1,046 | 3 | 2 | 837 | 3 | multi_to_single | 0.0% |
| particlefilter | physics | 1,023 | 2 | 1 | 400 | 1 | single_file | 34.6% |
| rsbench | physics | 1,016 | 6 | 2 | 1092 | 6 | multi_to_single | 0.0% |
| huffman | other | 686 | 7 | 6 | N/A | N/A | unknown | 0.0% |
| hybridsort | sort | 650 | 6 | 6 | N/A | N/A | unknown | 0.0% |
| srad | image | 391 | 2 | 2 | 173 | 1 | multi_to_single | 34.6% |
| streamcluster | other | 372 | 2 | 2 | 981 | 1 | multi_to_single | 4.5% |
| bptree | other | 338 | 5 | 4 | 1721 | 3 | multi_to_multi | 0.0% |
| gaussian | linear_algebra | 329 | 1 | 1 | N/A | N/A | single_file | 0.0% |
| nw | other | 319 | 2 | 2 | 291 | 1 | multi_to_single | 36.7% |
| mixbench | other | 312 | 4 | 1 | 297 | 3 | single_file | 9.1% |
| kmeans | ml | 299 | 2 | 2 | 1048 | 4 | multi_to_single | 0.0% |
| lud | linear_algebra | 271 | 2 | 2 | 400 | 3 | multi_to_single | 46.7% |
| nn | graph | 259 | 1 | 1 | 111 | 1 | single_file | 0.0% |
| hotspot3d | physics | 246 | 2 | 2 | 206 | 1 | multi_to_single | 58.8% |
| hotspot | physics | 243 | 1 | 1 | 262 | 1 | single_file | 46.7% |
| bfs | graph | 242 | 3 | 3 | 144 | 1 | multi_to_single | 44.1% |
| page-rank | graph | 235 | 2 | 1 | N/A | N/A | unknown | 60.0% |
| nqueen | other | 209 | 1 | 1 | N/A | N/A | unknown | 60.0% |
| scan | reduction | 206 | 1 | 1 | 189 | 1 | single_file | 0.0% |
| lavamd | molecular_dynamics | 200 | 3 | 2 | 258 | 2 | multi_to_single | 0.0% |
| convolution1d | stencil | 200 | 1 | 1 | N/A | N/A | unknown | 20.0% |
| backprop | ml | 197 | 2 | 2 | 449 | 4 | multi_to_single | 22.7% |
| iso2dfd | stencil | 196 | 1 | 1 | 170 | 1 | single_file | 84.2% |
| pathfinder | graph | 195 | 1 | 1 | 103 | 1 | single_file | 26.7% |
| floydwarshall | graph | 164 | 1 | 1 | 153 | 1 | single_file | 76.3% |
| jacobi | stencil | 129 | 1 | 1 | N/A | N/A | unknown | 30.0% |
| heat2d | stencil | 125 | 1 | 1 | 122 | 1 | single_file | 76.3% |
| md | molecular_dynamics | 107 | 1 | 1 | N/A | N/A | unknown | 40.0% |
| stencil1d | stencil | 80 | 1 | 1 | 71 | 1 | single_file | 64.3% |

## Summary Statistics

| Metric | Value |
|--------|------:|
| Min SLoC | 80 |
| Max SLoC | 3,304 |
| Mean SLoC | 598.4 |
| Median SLoC | 271 |
| Std Dev | 736.0 |
| Total SLoC (all kernels) | 20,945 |

## SLoC Distribution

| Range | Count |
|-------|------:|
| <100 | 1 |
| 100-500 | 24 |
| 500-1000 | 2 |
| >1000 | 8 |

## ParEval-Repo Comparison

ParEval-Repo uses functions averaging ~133 SLoC (single-function snippets).
ParBench evaluates **full application kernels** with real build systems.

- **31/35** (88.6%) of ParBench kernels exceed ParEval-Repo's 133-SLoC average
- ParBench median: **271 SLoC** (2.0x ParEval-Repo)
- ParBench range: **80** to **3,304** SLoC

## SLoC vs. Pass Rate Correlation

Spearman rank correlation (SLoC vs. overall pass rate): **-0.5924**

Negative correlation suggests larger kernels are harder to translate correctly.
