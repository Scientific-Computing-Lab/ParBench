"""Test that de-anonymization patches generic filenames inside code bodies."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "evaluation"))
from llm_evaluate import _deanonymize_extracted as _deanonymize


def test_include_reference_is_patched() -> None:
    """Generic #include in code body must be replaced with the real filename."""
    anon_map = {"translated_0.cu": "needle.cu", "translated_1.cu": "needle_kernel.cu"}
    anon_extracted = {
        "translated_0.cu": '#include "translated_1.cu"\n__global__ void kernel() {}',
        "translated_1.cu": "// kernel impl",
    }
    result = _deanonymize(anon_map, anon_extracted)
    assert '#include "needle_kernel.cu"' in result["needle.cu"], (
        "Generic #include not patched in code body"
    )
    assert "translated_1.cu" not in result["needle.cu"], (
        "Generic filename still present in code body"
    )


def test_single_quoted_include_is_patched() -> None:
    anon_map = {"translated_0.c": "bfs.c", "translated_1.h": "bfs.h"}
    anon_extracted = {
        "translated_0.c": "#include 'translated_1.h'\nint main() {}",
        "translated_1.h": "// header",
    }
    result = _deanonymize(anon_map, anon_extracted)
    assert "#include 'bfs.h'" in result["bfs.c"]


def test_non_include_content_is_unchanged() -> None:
    """Code that does not reference generic names must be bit-identical after patching."""
    anon_map = {"translated_0.cu": "hotspot.cu"}
    anon_extracted = {"translated_0.cu": "__global__ void kernel(float* a) { a[0] = 1.0f; }"}
    result = _deanonymize(anon_map, anon_extracted)
    assert result["hotspot.cu"] == "__global__ void kernel(float* a) { a[0] = 1.0f; }"


def test_keys_still_de_anonymized() -> None:
    """File keys must still map to real filenames (regression guard)."""
    anon_map = {"translated_0.cpp": "srad.cpp"}
    anon_extracted = {"translated_0.cpp": "// code"}
    result = _deanonymize(anon_map, anon_extracted)
    assert "translated_0.cpp" not in result
    assert "srad.cpp" in result
