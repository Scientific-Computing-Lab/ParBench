# Per-Kernel Augmentation Matrix

Generated: 2026-04-27T19:49:47.808257+00:00
Data source: `results/evaluation/azure-gpt-5.4`
Direction: cuda-to-omp
Kernel count: 24

## Per-Kernel Status Table

| Kernel | Suite | L0 | L1 | L2 | L3 | L4 | Pattern | Known Fail |
|--------|-------|----|----|----|----|----|---------|-----------:|
| backprop | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| bfs | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| bptree | rodinia | BUILD_FAIL | ? | ? | ? | ? | other |  |
| cfd | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| floydwarshall | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| heartwall | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| heat2d | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| hotspot | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| hotspot3d | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| iso2dfd | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| lavamd | rodinia | PASS | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | BUILD_FAIL | degradation |  |
| lud | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| mixbench | mixbench | PASS | PASS | PASS | BUILD_FAIL | PASS | degradation |  |
| myocyte | rodinia | EXTRACTION_FAIL | ? | ? | ? | ? | other |  |
| nn | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| nw | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| particlefilter | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| pathfinder | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| rsbench | rsbench | PASS | PASS | PASS | BUILD_FAIL | PASS | degradation |  |
| scan | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| srad | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| stencil1d | hecbench | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| streamcluster | rodinia | PASS | PASS | PASS | PASS | PASS | stable_pass |  |
| xsbench | xsbench | PASS | BUILD_FAIL | PASS | BUILD_FAIL | BUILD_FAIL | degradation |  |

## Pattern Summary

- **stable_pass** (18): backprop, bfs, cfd, floydwarshall, heartwall, heat2d, hotspot, hotspot3d, iso2dfd, lud, nn, nw, particlefilter, pathfinder, scan, srad, stencil1d, streamcluster
- **stable_fail** (0): none
- **degradation** (4): lavamd, mixbench, rsbench, xsbench
- **improvement** (0): none
- **other** (2): bptree, myocyte

## Aggregate Pass Rates

### All Kernels

| Level | Pass | Total | Rate | 95% CI |
|-------|------|-------|------|--------|
| L0 | 22 | 24 | 91.7% | [74.2%, 97.7%] |
| L1 | 20 | 22 | 90.9% | [72.2%, 97.5%] |
| L2 | 21 | 22 | 95.5% | [78.2%, 99.2%] |
| L3 | 18 | 22 | 81.8% | [61.5%, 92.7%] |
| L4 | 20 | 22 | 90.9% | [72.2%, 97.5%] |

### Excluding KNOWN_FAIL

| Level | Pass | Total | Rate | 95% CI |
|-------|------|-------|------|--------|
| L0 | 22 | 24 | 91.7% | [74.2%, 97.7%] |
| L1 | 20 | 22 | 90.9% | [72.2%, 97.5%] |
| L2 | 21 | 22 | 95.5% | [78.2%, 99.2%] |
| L3 | 18 | 22 | 81.8% | [61.5%, 92.7%] |
| L4 | 20 | 22 | 90.9% | [72.2%, 97.5%] |

## Exceptions

### lavamd (rodinia) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L2=BUILD_FAIL, L3=BUILD_FAIL, L4=BUILD_FAIL
- Root cause: Not yet investigated

### mixbench (mixbench) -- degradation
- L0: PASS
- Affected: L3=BUILD_FAIL
- Root cause: Not yet investigated

### rsbench (rsbench) -- degradation
- L0: PASS
- Affected: L3=BUILD_FAIL
- Root cause: Not yet investigated

### xsbench (xsbench) -- degradation
- L0: PASS
- Affected: L1=BUILD_FAIL, L3=BUILD_FAIL, L4=BUILD_FAIL
- Root cause: Not yet investigated

## Secondary Directions

| Direction | Kernels | L0 Rate | L1 Rate | L2 Rate | L3 Rate | L4 Rate |
|-----------|---------|---------|---------|---------|---------|---------|
| cuda-to-omp | 22 | 0.0% | 90.9% | 95.5% | 81.8% | 90.9% |
| cuda-to-omp_target | 8 | 0.0% | 87.5% | 100.0% | 100.0% | 87.5% |
| cuda-to-opencl | 13 | 0.0% | 92.3% | 100.0% | 84.6% | 92.3% |
| omp-to-cuda | 15 | 0.0% | 86.7% | 86.7% | 93.3% | 93.3% |
| omp-to-omp_target | 3 | 0.0% | 100.0% | 100.0% | 66.7% | 66.7% |
| omp-to-opencl | 14 | 0.0% | 78.6% | 85.7% | 85.7% | 85.7% |
| omp_target-to-cuda | 8 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| omp_target-to-omp | 3 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| opencl-to-cuda | 5 | 0.0% | 80.0% | 60.0% | 60.0% | 100.0% |
| opencl-to-omp | 8 | 0.0% | 87.5% | 75.0% | 87.5% | 87.5% |
