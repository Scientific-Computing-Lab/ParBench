# Per-Kernel Augmentation Matrix

Generated: 2026-05-01T07:17:53.677331+00:00
Data source: `results/evaluation/azure-gpt-5.3-codex`
Direction: cuda-to-omp
Kernel count: 24

## Per-Kernel Status Table

| Kernel | Suite | L0 | L1 | L2 | L3 | L4 | Pattern | Known Fail |
|--------|-------|----|----|----|----|----|---------|-----------:|
| backprop | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| bfs | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| bptree | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| cfd | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| floydwarshall | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| heartwall | rodinia | PASS | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | PASS | degradation |  |
| heat2d | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| hotspot | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| hotspot3d | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| iso2dfd | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| lavamd | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| lud | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| mixbench | mixbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| myocyte | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| nn | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| nw | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| particlefilter | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| pathfinder | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| rsbench | rsbench | PASS | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | degradation |  |
| scan | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| srad | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| stencil1d | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| streamcluster | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| xsbench | xsbench | PASS | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | degradation |  |

## Pattern Summary

- **stable_pass** (17): bfs, cfd, floydwarshall, heat2d, hotspot, hotspot3d, iso2dfd, lud, mixbench, nn, nw, particlefilter, pathfinder, scan, srad, stencil1d, streamcluster
- **stable_fail** (0): none
- **degradation** (3): heartwall, rsbench, xsbench
- **improvement** (0): none
- **other** (4): backprop, bptree, lavamd, myocyte

## Aggregate Pass Rates

### All Kernels

| Level | Pass | Total | Rate | 95% CI |
|-------|------|-------|------|--------|
| L0 | 20 | 24 | 83.3% | [64.1%, 93.3%] |
| L1 | 17 | 20 | 85.0% | [63.9%, 94.8%] |
| L2 | 17 | 20 | 85.0% | [63.9%, 94.8%] |
| L3 | 17 | 20 | 85.0% | [63.9%, 94.8%] |
| L4 | 18 | 20 | 90.0% | [69.9%, 97.2%] |

### Excluding KNOWN_FAIL

| Level | Pass | Total | Rate | 95% CI |
|-------|------|-------|------|--------|
| L0 | 20 | 24 | 83.3% | [64.1%, 93.3%] |
| L1 | 17 | 20 | 85.0% | [63.9%, 94.8%] |
| L2 | 17 | 20 | 85.0% | [63.9%, 94.8%] |
| L3 | 17 | 20 | 85.0% | [63.9%, 94.8%] |
| L4 | 18 | 20 | 90.0% | [69.9%, 97.2%] |

## Exceptions

### heartwall (rodinia) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L2=BUILD_FAIL, L3=BUILD_FAIL
- Root cause: Not yet investigated

### rsbench (rsbench) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L2=BUILD_FAIL, L3=BUILD_FAIL, L4=BUILD_FAIL
- Root cause: Not yet investigated

### xsbench (xsbench) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L2=BUILD_FAIL, L3=BUILD_FAIL, L4=BUILD_FAIL
- Root cause: Not yet investigated

## Secondary Directions

| Direction | Kernels | L0 Rate | L1 Rate | L2 Rate | L3 Rate | L4 Rate |
|-----------|---------|---------|---------|---------|---------|---------|
| cuda-to-omp | 20 | 0.0% | 85.0% | 85.0% | 85.0% | 90.0% |
| cuda-to-omp_target | 8 | 0.0% | 87.5% | 100.0% | 100.0% | 75.0% |
| cuda-to-opencl | 12 | 0.0% | 100.0% | 100.0% | 91.7% | 100.0% |
| omp-to-cuda | 15 | 0.0% | 86.7% | 86.7% | 93.3% | 86.7% |
| omp-to-omp_target | 3 | 0.0% | 100.0% | 100.0% | 33.3% | 33.3% |
| omp-to-opencl | 14 | 0.0% | 85.7% | 92.9% | 85.7% | 92.9% |
| omp_target-to-cuda | 8 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| omp_target-to-omp | 3 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| opencl-to-cuda | 5 | 0.0% | 60.0% | 80.0% | 60.0% | 60.0% |
| opencl-to-omp | 9 | 0.0% | 66.7% | 55.6% | 77.8% | 66.7% |
