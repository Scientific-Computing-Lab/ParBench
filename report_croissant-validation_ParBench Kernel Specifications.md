# CROISSANT VALIDATION REPORT
================================================================================
## VALIDATION RESULTS
--------------------------------------------------------------------------------
Starting validation for file: croissant.json
### JSON Format Validation
✓
The file is valid JSON.
### Croissant Schema Validation
✓
The dataset passes Croissant validation.
### Responsible AI Metadata
✓
All required Responsible AI metadata fields are present.
### Records Generation Test
✓
No record sets found to validate.
## JSON-LD REFERENCE
================================================================================
```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {
      "@id": "cr:data",
      "@type": "@json"
    },
    "dataType": {
      "@id": "cr:dataType",
      "@type": "@vocab"
    },
    "dct": "http://purl.org/dc/terms/",
    "examples": {
      "@id": "cr:examples",
      "@type": "@json"
    },
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:FileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:FileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "samplingRate": "cr:samplingRate",
    "citeAs": "cr:citeAs",
    "equivalentProperty": "cr:equivalentProperty",
    "@base": "cr_base_iri/"
  },
  "@type": "sc:Dataset",
  "name": "ParBench Kernel Specifications",
  "description": "206 JSON specification files defining parallel code translation tasks across CUDA, OpenMP, OpenCL, and OpenMP target offload APIs, sourced from 5 HPC benchmark suites (Rodinia, HeCBench, XSBench, RSBench, mixbench).",
  "license": "https://opensource.org/licenses/MIT",
  "url": "https://anonymous.4open.science/r/parbench_artifact_neurips/",
  "version": "1.0.0",
  "keywords": [
    "parallel computing",
    "code translation",
    "CUDA",
    "OpenMP",
    "OpenCL",
    "benchmark",
    "LLM evaluation"
  ],
  "datePublished": "2026-05-06",
  "conformsTo": "http://mlcommons.org/croissant/1.1",
  "citeAs": "Anonymous. ParBench: A Meta-Benchmark for LLM-Based Parallel Code Translation. NeurIPS 2026 Evaluations & Datasets Track (under review).",
  "distribution": [
    {
      "@type": "cr:FileSet",
      "name": "kernel-specs",
      "description": "206 JSON specification files, one per kernel-API variant, defining build, run, and verification configurations for parallel code translation tasks",
      "includes": "specs/*.json",
      "encodingFormat": "application/json",
      "sha256": "7548c3677b45e45158b879c5c3fc132b81219e7129e8874fcf0b56b785bf5a25"
    },
    {
      "@type": "cr:FileObject",
      "name": "spec-schema",
      "description": "JSON Schema (draft-07) defining the spec file structure",
      "contentUrl": "schema/spec_schema.json",
      "encodingFormat": "application/schema+json",
      "sha256": "d69539482601409889b87fae36e44d565e79a7f1950efbe6262d4fa42fa05cfa"
    }
  ],
  "rai:dataCollection": "Specifications curated from existing open-source HPC benchmark suites (Rodinia, HeCBench, XSBench, RSBench, mixbench). Each spec is manually authored by the benchmark developers to define build, run, and verification configurations. No crowdsourcing, scraping, or automated data collection was used.",
  "rai:dataCollectionType": "Curated from existing sources",
  "rai:dataUseCases": "Evaluating the correctness of LLM-generated parallel code translations (e.g., CUDA to OpenMP). The specs define ground-truth build and verification pipelines against which translated code is tested. Not intended for training language models.",
  "rai:dataLimitations": "Covers only four parallel APIs (CUDA, OpenMP, OpenCL, OpenMP target offload). Does not include SYCL, HIP, Kokkos, or other parallel frameworks. Benchmark kernels are drawn from scientific computing workloads and may not represent all parallel programming patterns. Verification oracles vary in strength (2 strong, 5 medium, 46 weak, remainder untagged).",
  "rai:dataBiases": "Kernel selection is biased toward scientific computing and numerical simulation workloads from established HPC benchmark suites. Rodinia contributes the majority of specs (60/206). Underrepresented domains include graph analytics, machine learning training kernels, and I/O-bound workloads.",
  "rai:personalSensitiveInformation": "No personal or sensitive data. All specs describe computational kernels operating on synthetic or public scientific datasets. No human subjects, demographic data, or personally identifiable information is present.",
  "rai:dataSocialImpact": "ParBench enables evaluation of automated code translation tools, which could accelerate software portability across GPU and CPU platforms. Potential positive impact: reducing the cost of maintaining parallel codebases. No known negative social impacts from the benchmark data itself.",
  "rai:dataPreprocessingProtocol": "Kernel source files are used as-is from upstream repositories at pinned commits. No preprocessing, filtering, or transformation is applied to the source code. Spec JSON files are authored manually following a JSON Schema (draft-07) contract.",
  "rai:hasSyntheticData": "No. All kernel specifications reference real source code from existing open-source HPC benchmark suites. No synthetic or generated data is present."
}
```