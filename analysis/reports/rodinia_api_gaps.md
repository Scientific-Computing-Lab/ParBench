# Rodinia ParBench — API Gaps Report

**Generated**: 2026-03-03  
**Rodinia commit**: 9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4

## Summary

- **Total specs generated**: 63
- **CUDA** variants: 23
- **OPENCL** variants: 21
- **OMP** variants: 19

## API Coverage Matrix

| Kernel | CUDA | OpenCL | OMP | HeCBench Overlap |
|--------|------|--------|-----|-----------------|
| b+tree | ✅ | ✅ | ✅ |  |
| backprop | ✅ | ✅ | ✅ | ⚠️ Yes |
| bfs | ✅ | ✅ | ✅ |  |
| cfd | ✅ | ✅ | ✅ |  |
| dwt2d | ✅ | ✅ | ❌ |  |
| gaussian | ✅ | ✅ | ❌ | ⚠️ Yes |
| heartwall | ✅ | ✅ | ✅ |  |
| hotspot | ✅ | ✅ | ✅ |  |
| hotspot3D | ✅ | ✅ | ✅ |  |
| huffman | ✅ | ❌ | ❌ |  |
| hybridsort | ✅ | ✅ | ❌ |  |
| kmeans | ✅ | ✅ | ✅ |  |
| lavaMD | ✅ | ✅ | ✅ |  |
| leukocyte | ✅ | ✅ | ✅ |  |
| lud | ✅ | ✅ | ✅ | ⚠️ Yes |
| mummergpu | ✅ | ❌ | ✅ |  |
| myocyte | ✅ | ✅ | ✅ |  |
| nn | ✅ | ✅ | ✅ | ⚠️ Yes |
| nw | ✅ | ✅ | ✅ | ⚠️ Yes |
| particlefilter | ✅ | ✅ | ✅ |  |
| pathfinder | ✅ | ✅ | ✅ | ⚠️ Yes |
| rng | ❌ | ❌ | ❌ |  |
| srad | ✅ | ✅ | ✅ |  |
| streamcluster | ✅ | ✅ | ✅ |  |

## Gap Analysis

### Kernels missing OpenCL

- **huffman**: has cuda, missing OpenCL
- **mummergpu**: has cuda, omp, missing OpenCL

### Kernels missing OpenMP

- **dwt2d**: has cuda, opencl, missing OpenMP
- **gaussian**: has cuda, opencl, missing OpenMP
- **huffman**: has cuda, missing OpenMP
- **hybridsort**: has cuda, opencl, missing OpenMP

### Skipped kernels (unsupported API only)

- **rng**: only has others (not in schema)

## HeCBench Overlaps

These Rodinia kernels overlap with existing HeCBench specs. Rodinia specs are distinct entries with different source paths.

- **backprop**: cuda, omp, opencl
- **gaussian**: cuda, opencl
- **lud**: cuda, omp, opencl
- **nn**: cuda, omp, opencl
- **nw**: cuda, omp, opencl
- **pathfinder**: cuda, omp, opencl
