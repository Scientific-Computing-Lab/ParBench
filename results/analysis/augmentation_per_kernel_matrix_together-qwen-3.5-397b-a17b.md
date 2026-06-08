# Per-Kernel Augmentation Matrix

Generated: 2026-04-27T19:49:46.466538+00:00
Data source: `results/evaluation/together-qwen-3.5-397b-a17b`
Direction: cuda-to-omp
Kernel count: 26

## Per-Kernel Status Table

| Kernel | Suite | L0 | L1 | L2 | L3 | L4 | Pattern | Known Fail |
|--------|-------|----|----|----|----|----|---------|-----------:|
| backprop | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| bfs | rodinia | PASS | PASS | PASS | PASS | BUILD_FAIL | degradation |  |
| bptree | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| cfd | rodinia | PASS | BUILD_FAIL | PASS | PASS | PASS | degradation |  |
| floydwarshall | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| heartwall | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| heat2d | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| hotspot | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| hotspot3d | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| iso2dfd | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| kmeans | rodinia | BUILD_FAIL | ? | ? | ? | ? | other | Yes |
| lavamd | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| lud | rodinia | PASS | PASS | PASS | RUN_FAIL | PASS | degradation |  |
| mixbench | mixbench | BUILD_FAIL | ? | ? | ? | ? | other |  |
| mummergpu | rodinia | BUILD_FAIL | ? | ? | ? | ? | other | Yes |
| myocyte | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| nn | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| nw | rodinia | PASS | PASS | PASS | RUN_FAIL | PASS | degradation |  |
| particlefilter | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| pathfinder | rodinia | PASS | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | PASS | degradation |  |
| rsbench | rsbench | BUILD_FAIL | ? | ? | ? | ? | other |  |
| scan | hecbench | RUN_FAIL | ? | ? | ? | ? | other |  |
| srad | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| stencil1d | hecbench | PASS | BUILD_FAIL | VERIFY_FAIL | PASS | VERIFY_FAIL | degradation |  |
| streamcluster | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| xsbench | xsbench | BUILD_FAIL | ? | ? | ? | ? | other |  |

## Pattern Summary

- **stable_pass** (6): floydwarshall, heat2d, hotspot3d, iso2dfd, particlefilter, srad
- **stable_fail** (0): none
- **degradation** (6): bfs, cfd, lud, nw, pathfinder, stencil1d
- **improvement** (0): none
- **other** (14): backprop, bptree, heartwall, hotspot, kmeans, lavamd, mixbench, mummergpu, myocyte, nn, rsbench, scan, streamcluster, xsbench

## Aggregate Pass Rates

### All Kernels

| Level | Pass | Total | Rate | 95% CI |
|-------|------|-------|------|--------|
| L0 | 12 | 26 | 46.2% | [28.7%, 64.5%] |
| L1 | 9 | 12 | 75.0% | [46.8%, 91.1%] |
| L2 | 10 | 12 | 83.3% | [55.2%, 95.3%] |
| L3 | 9 | 12 | 75.0% | [46.8%, 91.1%] |
| L4 | 10 | 12 | 83.3% | [55.2%, 95.3%] |

### Excluding KNOWN_FAIL

| Level | Pass | Total | Rate | 95% CI |
|-------|------|-------|------|--------|
| L0 | 12 | 24 | 50.0% | [31.4%, 68.6%] |
| L1 | 9 | 12 | 75.0% | [46.8%, 91.1%] |
| L2 | 10 | 12 | 83.3% | [55.2%, 95.3%] |
| L3 | 9 | 12 | 75.0% | [46.8%, 91.1%] |
| L4 | 10 | 12 | 83.3% | [55.2%, 95.3%] |

## Exceptions

### bfs (rodinia) -- degradation
- L0: PASS
- Affected: L4=BUILD_FAIL
- Root cause: Not yet investigated

### cfd (rodinia) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL
- Root cause: Not yet investigated

### lud (rodinia) -- degradation
- L0: PASS
- Affected: L3=RUN_FAIL
- Root cause: Not yet investigated

### nw (rodinia) -- degradation
- L0: PASS
- Affected: L3=RUN_FAIL
- Root cause: Not yet investigated

### pathfinder (rodinia) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L2=BUILD_FAIL, L3=BUILD_FAIL
- Root cause: Not yet investigated

### stencil1d (hecbench) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L2=VERIFY_FAIL, L4=VERIFY_FAIL
- Root cause: Not yet investigated

## Secondary Directions

| Direction | Kernels | L0 Rate | L1 Rate | L2 Rate | L3 Rate | L4 Rate |
|-----------|---------|---------|---------|---------|---------|---------|
| cuda-to-omp | 12 | 0.0% | 75.0% | 83.3% | 75.0% | 83.3% |
| cuda-to-opencl | 3 | 0.0% | 0.0% | 66.7% | 33.3% | 33.3% |
| omp-to-cuda | 8 | 0.0% | 75.0% | 75.0% | 62.5% | 50.0% |
| omp-to-omp_target | 3 | 0.0% | 100.0% | 66.7% | 33.3% | 33.3% |
| omp-to-opencl | 10 | 0.0% | 80.0% | 40.0% | 50.0% | 30.0% |
| omp_target-to-cuda | 8 | 0.0% | 87.5% | 75.0% | 75.0% | 62.5% |
| omp_target-to-omp | 3 | 0.0% | 100.0% | 100.0% | 66.7% | 100.0% |
| opencl-to-omp | 4 | 0.0% | 50.0% | 0.0% | 75.0% | 25.0% |
