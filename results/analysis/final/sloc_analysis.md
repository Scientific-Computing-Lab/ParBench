# SLoC Characterization of ParBench Corpus Kernels

**35 kernels** analyzed across 5 suites: hecbench (10), mixbench (1), rodinia (22), rsbench (1), xsbench (1).
Physical SLoC = non-blank, non-comment lines (matches cloc methodology).
Source: actual CUDA source files from the benchmark repositories.

## Per-Kernel SLoC

| Kernel | Category | CUDA SLoC | Src Files | Tgt Files | OMP SLoC | OMP Files | Complexity | Pass Rate |
|--------|----------|----------:|----------:|----------:|---------:|----------:|------------|----------:|
| myocyte | other | 3,304 | 16 | 2 | 1806 | 10 | multi_to_multi | 0.0% |
| mummergpu | other | 2,773 | 3 | 2 | 5325 | 9 | multi_to_single | N/A |
| cfd | physics | 1,955 | 4 | 1 | 400 | 1 | single_file | 67.8% |
| xsbench | physics | 1,390 | 6 | 2 | 1238 | 6 | multi_to_multi | 33.0% |
| dwt2d | physics | 1,238 | 8 | 7 | N/A | N/A | unknown | 0.0% |
| heartwall | image | 1,046 | 3 | 2 | 837 | 3 | multi_to_single | 14.5% |
| particlefilter | physics | 1,023 | 2 | 1 | 400 | 1 | single_file | 50.0% |
| rsbench | physics | 1,016 | 6 | 2 | 1092 | 6 | multi_to_single | 37.2% |
| huffman | other | 686 | 7 | 6 | N/A | N/A | unknown | N/A |
| hybridsort | sort | 650 | 6 | 6 | N/A | N/A | unknown | N/A |
| srad | image | 391 | 2 | 2 | 173 | 1 | multi_to_single | 25.5% |
| streamcluster | other | 372 | 2 | 2 | 981 | 1 | multi_to_single | 0.0% |
| bptree | other | 338 | 5 | 4 | 1721 | 3 | multi_to_multi | 37.8% |
| gaussian | linear_algebra | 329 | 1 | 1 | N/A | N/A | single_file | 0.0% |
| nw | other | 319 | 2 | 2 | 291 | 1 | multi_to_single | 24.5% |
| mixbench | other | 312 | 4 | 1 | 341 | 3 | single_file | N/A |
| kmeans | ml | 299 | 2 | 2 | 1048 | 4 | multi_to_single | N/A |
| lud | linear_algebra | 271 | 2 | 2 | 400 | 3 | multi_to_single | 70.4% |
| nn | graph | 259 | 1 | 1 | 111 | 1 | single_file | 0.0% |
| hotspot3d | physics | 246 | 2 | 2 | 206 | 1 | multi_to_single | 42.7% |
| hotspot | physics | 243 | 1 | 1 | 262 | 1 | single_file | 53.1% |
| bfs | graph | 242 | 3 | 3 | 144 | 1 | multi_to_single | 48.2% |
| page-rank | graph | 235 | 2 | 1 | N/A | N/A | unknown | 81.6% |
| nqueen | other | 209 | 1 | 1 | N/A | N/A | unknown | 86.8% |
| scan | reduction | 206 | 1 | 1 | 189 | 1 | single_file | 82.3% |
| lavamd | molecular_dynamics | 200 | 3 | 2 | 258 | 2 | multi_to_single | 3.0% |
| convolution1d | stencil | 200 | 1 | 1 | N/A | N/A | unknown | 55.3% |
| backprop | ml | 197 | 2 | 2 | 449 | 4 | multi_to_single | 31.8% |
| iso2dfd | stencil | 196 | 1 | 1 | 170 | 1 | single_file | 95.1% |
| pathfinder | graph | 195 | 1 | 1 | 103 | 1 | single_file | 0.0% |
| floydwarshall | graph | 164 | 1 | 1 | 153 | 1 | single_file | 90.2% |
| jacobi | stencil | 129 | 1 | 1 | N/A | N/A | unknown | 81.6% |
| heat2d | stencil | 125 | 1 | 1 | 122 | 1 | single_file | 88.5% |
| md | molecular_dynamics | 107 | 1 | 1 | N/A | N/A | unknown | 81.6% |
| stencil1d | stencil | 80 | 1 | 1 | 71 | 1 | single_file | 88.1% |

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

Spearman rank correlation (SLoC vs. overall pass rate): **-0.5754**

Negative correlation suggests larger kernels are harder to translate correctly.
