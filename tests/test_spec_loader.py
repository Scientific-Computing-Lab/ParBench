"""
Unit tests for harness.spec_loader — synthetic data only.

Validates load_spec, resolve_paths, get_prompt_payload, find_translation_pairs,
and load_manifest using synthetic/mock data that does not depend on disk state.

Run:  python3 -m pytest tests/test_spec_loader.py -v
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from harness.spec_loader import (
    find_translation_pairs,
    get_prompt_payload,
    load_manifest,
    load_spec,
    resolve_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_spec(
    kernel_name: str = "bfs",
    parallel_api: str = "cuda",
    source_suite: str = "rodinia",
    repo_root: str = "rodinia/rodinia-src",
    source_path: str = "cuda/bfs",
    working_directory: str = "cuda/bfs",
    prompt_payload: list[str] | None = None,
    support_files: list[str] | None = None,
    translation_targets: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal spec dict matching the schema's required fields."""
    return {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": kernel_name,
            "parallel_api": parallel_api,
            "unique_id": f"{source_suite}-{kernel_name}-{parallel_api}",
            "source_suite": source_suite,
        },
        "provenance": {
            "repository": {
                "url": "https://example.com/repo.git",
                "commit": "abc123",
                "branch": "main",
            },
            "repo_root": repo_root,
            "source_path": source_path,
            "license": "MIT",
        },
        "files": {
            "prompt_payload": prompt_payload or ["main.cu"],
            "support_files": support_files or ["Makefile"],
            "verification_only": [],
            "translation_targets": translation_targets or ["main.cu"],
        },
        "build": {
            "working_directory": working_directory,
            "commands": {"build": "make", "clean": "make clean"},
        },
    }


# ---------------------------------------------------------------------------
# load_spec
# ---------------------------------------------------------------------------

class TestLoadSpec:
    """Tests for load_spec — JSON file → dict."""

    def test_load_spec_valid(self, tmp_path: Path) -> None:
        """A valid spec JSON is loaded as a dict with expected keys."""
        spec_data = _minimal_spec()
        spec_file = tmp_path / "test-spec.json"
        spec_file.write_text(json.dumps(spec_data), encoding="utf-8")

        result = load_spec(spec_file)
        assert isinstance(result, dict)
        assert result["identity"]["kernel_name"] == "bfs"
        assert result["identity"]["parallel_api"] == "cuda"
        assert result["spec_version"] == "1.0.0"

    def test_load_spec_preserves_all_sections(self, tmp_path: Path) -> None:
        """All top-level sections survive the round-trip."""
        spec_data = _minimal_spec()
        spec_file = tmp_path / "test-spec.json"
        spec_file.write_text(json.dumps(spec_data), encoding="utf-8")

        result = load_spec(spec_file)
        for key in ("spec_version", "identity", "provenance", "files", "build"):
            assert key in result, f"Missing top-level key: {key}"

    def test_load_spec_missing_file(self, tmp_path: Path) -> None:
        """Attempting to load a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_spec(tmp_path / "nonexistent.json")

    def test_load_spec_invalid_json(self, tmp_path: Path) -> None:
        """Malformed JSON raises json.JSONDecodeError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_spec(bad_file)

    def test_load_spec_empty_file(self, tmp_path: Path) -> None:
        """An empty file raises json.JSONDecodeError."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_spec(empty_file)


# ---------------------------------------------------------------------------
# resolve_paths
# ---------------------------------------------------------------------------

class TestResolvePaths:
    """Tests for resolve_paths — relative → absolute path resolution."""

    def _setup_config(self, project_root: Path) -> None:
        """Write a minimal config/paths.json in the temp project root."""
        config_dir = project_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "project_root": str(project_root),
            "downloads_root": str(project_root),
            "hecbench_root": str(project_root),
        }
        (config_dir / "paths.json").write_text(json.dumps(config), encoding="utf-8")

    def test_resolve_paths_adds_resolved_key(self, tmp_path: Path) -> None:
        """resolve_paths returns a dict with _resolved key."""
        self._setup_config(tmp_path)
        spec = _minimal_spec()
        result = resolve_paths(spec, tmp_path)
        assert "_resolved" in result

    def test_resolve_paths_does_not_mutate_original(self, tmp_path: Path) -> None:
        """The original spec dict is not modified."""
        self._setup_config(tmp_path)
        spec = _minimal_spec()
        original_keys = set(spec.keys())
        resolve_paths(spec, tmp_path)
        assert set(spec.keys()) == original_keys
        assert "_resolved" not in spec

    def test_resolve_paths_repo_root(self, tmp_path: Path) -> None:
        """repo_root is resolved to downloads_root / provenance.repo_root."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(repo_root="rodinia/rodinia-src")
        result = resolve_paths(spec, tmp_path)
        expected = (tmp_path / "rodinia" / "rodinia-src").resolve()
        assert result["_resolved"]["repo_root"] == expected

    def test_resolve_paths_source_dir(self, tmp_path: Path) -> None:
        """source_dir is resolved to repo_root / provenance.source_path."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(repo_root="rodinia/rodinia-src", source_path="cuda/bfs")
        result = resolve_paths(spec, tmp_path)
        expected = (tmp_path / "rodinia" / "rodinia-src" / "cuda" / "bfs").resolve()
        assert result["_resolved"]["source_dir"] == expected

    def test_resolve_paths_working_dir(self, tmp_path: Path) -> None:
        """working_dir is resolved to repo_root / build.working_directory."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(
            repo_root="rodinia/rodinia-src",
            working_directory="cuda/bfs",
        )
        result = resolve_paths(spec, tmp_path)
        expected = (tmp_path / "rodinia" / "rodinia-src" / "cuda" / "bfs").resolve()
        assert result["_resolved"]["working_dir"] == expected

    def test_resolve_paths_file_lists(self, tmp_path: Path) -> None:
        """All file list categories are resolved to absolute Path objects."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(
            repo_root="myrepo",
            source_path="src",
            prompt_payload=["a.cu", "b.cu"],
            support_files=["Makefile"],
            translation_targets=["a.cu"],
        )
        result = resolve_paths(spec, tmp_path)
        resolved_files = result["_resolved"]["files"]

        # prompt_payload
        assert len(resolved_files["prompt_payload"]) == 2
        assert all(isinstance(p, Path) for p in resolved_files["prompt_payload"])
        expected_a = (tmp_path / "myrepo" / "src" / "a.cu").resolve()
        assert resolved_files["prompt_payload"][0] == expected_a

        # support_files
        assert len(resolved_files["support_files"]) == 1

        # translation_targets
        assert len(resolved_files["translation_targets"]) == 1

        # verification_only (empty)
        assert resolved_files["verification_only"] == []

    def test_resolve_paths_fallback_without_config(self, tmp_path: Path) -> None:
        """Without config/paths.json, downloads_root falls back to project_root."""
        # Do NOT create config/paths.json
        spec = _minimal_spec(repo_root="myrepo", source_path="src")
        result = resolve_paths(spec, tmp_path)
        expected = (tmp_path / "myrepo" / "src").resolve()
        assert result["_resolved"]["source_dir"] == expected

    def test_resolve_paths_empty_source_path(self, tmp_path: Path) -> None:
        """When source_path is empty, source_dir falls back to repo_root."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(repo_root="myrepo", source_path="")
        result = resolve_paths(spec, tmp_path)
        assert result["_resolved"]["source_dir"] == result["_resolved"]["repo_root"]

    def test_resolve_paths_empty_working_directory(self, tmp_path: Path) -> None:
        """When working_directory is empty, working_dir falls back to source_dir."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(
            repo_root="myrepo",
            source_path="src",
            working_directory="",
        )
        result = resolve_paths(spec, tmp_path)
        assert result["_resolved"]["working_dir"] == result["_resolved"]["source_dir"]

    def test_resolve_paths_hecbench_style(self, tmp_path: Path) -> None:
        """HeCBench specs use repo_root=HeCBench-master/ with src/ prefix in source_path."""
        self._setup_config(tmp_path)
        spec = _minimal_spec(
            kernel_name="bezier-surface",
            source_suite="hecbench",
            repo_root="HeCBench-master/",
            source_path="src/bezier-surface-cuda",
        )
        result = resolve_paths(spec, tmp_path)
        expected = (tmp_path / "HeCBench-master" / "src" / "bezier-surface-cuda").resolve()
        assert result["_resolved"]["source_dir"] == expected


# ---------------------------------------------------------------------------
# get_prompt_payload
# ---------------------------------------------------------------------------

class TestGetPromptPayload:
    """Tests for get_prompt_payload — reads source files into a dict."""

    def _setup_project(
        self,
        tmp_path: Path,
        files: dict[str, str],
        repo_root: str = "myrepo",
        source_path: str = "src",
    ) -> tuple[dict[str, Any], Path]:
        """Create a minimal project with source files on disk."""
        # config
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "project_root": str(tmp_path),
            "downloads_root": str(tmp_path),
        }
        (config_dir / "paths.json").write_text(json.dumps(config), encoding="utf-8")

        # source files
        src_dir = tmp_path / repo_root / source_path
        src_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (src_dir / fname).write_text(content, encoding="utf-8")

        spec = _minimal_spec(
            repo_root=repo_root,
            source_path=source_path,
            prompt_payload=list(files.keys()),
        )
        return spec, tmp_path

    def test_get_prompt_payload_structure(self, tmp_path: Path) -> None:
        """Returned dict maps filenames to their contents."""
        spec, project_root = self._setup_project(
            tmp_path,
            files={"kernel.cu": "__global__ void foo() {}"},
        )
        payload = get_prompt_payload(spec, project_root)
        assert isinstance(payload, dict)
        assert "kernel.cu" in payload
        assert "__global__" in payload["kernel.cu"]

    def test_get_prompt_payload_multiple_files(self, tmp_path: Path) -> None:
        """Multiple files are all present in the payload."""
        spec, project_root = self._setup_project(
            tmp_path,
            files={
                "main.cu": "int main() { return 0; }",
                "kernel.cu": "__global__ void run() {}",
                "utils.h": "#pragma once",
            },
        )
        payload = get_prompt_payload(spec, project_root)
        assert len(payload) == 3
        assert set(payload.keys()) == {"main.cu", "kernel.cu", "utils.h"}

    def test_get_prompt_payload_missing_file(self, tmp_path: Path) -> None:
        """Missing files produce a '<FILE NOT FOUND: ...>' sentinel, not an exception."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "project_root": str(tmp_path),
            "downloads_root": str(tmp_path),
        }
        (config_dir / "paths.json").write_text(json.dumps(config), encoding="utf-8")

        spec = _minimal_spec(
            repo_root="myrepo",
            source_path="src",
            prompt_payload=["nonexistent.cu"],
        )
        # Don't create the file on disk
        payload = get_prompt_payload(spec, tmp_path)
        assert "nonexistent.cu" in payload
        assert "FILE NOT FOUND" in payload["nonexistent.cu"]

    def test_get_prompt_payload_empty_payload_list(self, tmp_path: Path) -> None:
        """An empty prompt_payload list produces an empty dict."""
        spec, project_root = self._setup_project(
            tmp_path,
            files={},
        )
        spec["files"]["prompt_payload"] = []
        payload = get_prompt_payload(spec, project_root)
        assert payload == {}

    def test_get_prompt_payload_preserves_content(self, tmp_path: Path) -> None:
        """File content is preserved verbatim (no extra whitespace or truncation)."""
        content = textwrap.dedent("""\
            #include <stdio.h>

            __global__ void kernel(int *data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) data[idx] *= 2;
            }
        """)
        spec, project_root = self._setup_project(
            tmp_path,
            files={"kernel.cu": content},
        )
        payload = get_prompt_payload(spec, project_root)
        assert payload["kernel.cu"] == content


# ---------------------------------------------------------------------------
# find_translation_pairs
# ---------------------------------------------------------------------------

class TestFindTranslationPairs:
    """Tests for find_translation_pairs — manifest → ordered pair enumeration."""

    def test_single_kernel_two_apis(self) -> None:
        """Two APIs for one kernel produce 2 ordered pairs (A→B and B→A)."""
        manifest = [
            {"kernel_name": "bfs", "parallel_api": "cuda", "source_suite": "rodinia"},
            {"kernel_name": "bfs", "parallel_api": "omp", "source_suite": "rodinia"},
        ]
        pairs = find_translation_pairs(manifest)
        assert len(pairs) == 2
        apis = {(p[2], p[3]) for p in pairs}
        assert ("cuda", "omp") in apis
        assert ("omp", "cuda") in apis

    def test_single_kernel_three_apis(self) -> None:
        """Three APIs produce 6 ordered pairs (3*2)."""
        manifest = [
            {"kernel_name": "bfs", "parallel_api": "cuda", "source_suite": "rodinia"},
            {"kernel_name": "bfs", "parallel_api": "omp", "source_suite": "rodinia"},
            {"kernel_name": "bfs", "parallel_api": "opencl", "source_suite": "rodinia"},
        ]
        pairs = find_translation_pairs(manifest)
        assert len(pairs) == 6

    def test_different_suites_same_kernel_name(self) -> None:
        """Same kernel_name in different suites are kept separate."""
        manifest = [
            {"kernel_name": "nn", "parallel_api": "cuda", "source_suite": "rodinia"},
            {"kernel_name": "nn", "parallel_api": "omp", "source_suite": "rodinia"},
            {"kernel_name": "nn", "parallel_api": "cuda", "source_suite": "hecbench"},
            {"kernel_name": "nn", "parallel_api": "omp", "source_suite": "hecbench"},
        ]
        pairs = find_translation_pairs(manifest)
        # 2 pairs per suite = 4 total
        assert len(pairs) == 4
        rodinia_pairs = [p for p in pairs if p[0] == "rodinia"]
        hecbench_pairs = [p for p in pairs if p[0] == "hecbench"]
        assert len(rodinia_pairs) == 2
        assert len(hecbench_pairs) == 2

    def test_single_api_no_pairs(self) -> None:
        """A kernel with only one API produces no pairs."""
        manifest = [
            {"kernel_name": "solo", "parallel_api": "cuda", "source_suite": "test"},
        ]
        pairs = find_translation_pairs(manifest)
        assert pairs == []

    def test_empty_manifest(self) -> None:
        """An empty manifest produces no pairs."""
        assert find_translation_pairs([]) == []

    def test_pair_tuple_structure(self) -> None:
        """Each pair is a (suite, kernel_name, source_api, target_api) tuple."""
        manifest = [
            {"kernel_name": "bfs", "parallel_api": "cuda", "source_suite": "rodinia"},
            {"kernel_name": "bfs", "parallel_api": "omp", "source_suite": "rodinia"},
        ]
        pairs = find_translation_pairs(manifest)
        for pair in pairs:
            assert len(pair) == 4
            suite, kernel, src_api, tgt_api = pair
            assert suite == "rodinia"
            assert kernel == "bfs"
            assert src_api != tgt_api

    def test_four_apis_produce_twelve_pairs(self) -> None:
        """Four APIs produce 12 ordered pairs (4*3)."""
        manifest = [
            {"kernel_name": "xsbench", "parallel_api": api, "source_suite": "xsbench"}
            for api in ["cuda", "omp", "opencl", "omp_target"]
        ]
        pairs = find_translation_pairs(manifest)
        assert len(pairs) == 12

    def test_missing_source_suite_defaults_to_unknown(self) -> None:
        """Entries without source_suite are grouped under 'unknown'."""
        manifest = [
            {"kernel_name": "test", "parallel_api": "cuda"},
            {"kernel_name": "test", "parallel_api": "omp"},
        ]
        pairs = find_translation_pairs(manifest)
        assert len(pairs) == 2
        assert all(p[0] == "unknown" for p in pairs)

    def test_duplicate_api_entries_degenerate_behavior(self) -> None:
        """Duplicate manifest entries for the same API inflate pair count and produce self-pairs.

        This documents a known degenerate case: find_translation_pairs does NOT deduplicate
        API entries. If the same API appears twice for a kernel, combinations() treats them
        as distinct, yielding (cuda, cuda) self-pairs and duplicate directional pairs.

        The real manifest.jsonl is append-only and guaranteed to have no duplicate entries,
        so this never occurs in practice. This test documents the behavior for future callers.
        """
        manifest = [
            {"kernel_name": "bfs", "parallel_api": "cuda", "source_suite": "rodinia"},
            {"kernel_name": "bfs", "parallel_api": "cuda", "source_suite": "rodinia"},  # dup
            {"kernel_name": "bfs", "parallel_api": "omp", "source_suite": "rodinia"},
        ]
        pairs = find_translation_pairs(manifest)
        # 3 entries → combinations of 3 → 3 combos each doubled → 6 pairs (NOT 2)
        assert len(pairs) == 6
        # Self-pairs (cuda→cuda) are produced — degenerate but expected given no dedup
        self_pairs = [p for p in pairs if p[2] == p[3]]
        assert len(self_pairs) == 2  # (cuda,cuda) produced twice


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    """Tests for load_manifest — JSONL file → list of dicts."""

    def test_load_manifest_basic(self, tmp_path: Path) -> None:
        """Loads a multi-line JSONL file into a list of dicts."""
        manifest_file = tmp_path / "manifest.jsonl"
        lines = [
            json.dumps({"kernel_name": "bfs", "parallel_api": "cuda", "source_suite": "rodinia"}),
            json.dumps({"kernel_name": "bfs", "parallel_api": "omp", "source_suite": "rodinia"}),
        ]
        manifest_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        entries = load_manifest(manifest_file)
        assert len(entries) == 2
        assert entries[0]["kernel_name"] == "bfs"
        assert entries[1]["parallel_api"] == "omp"

    def test_load_manifest_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines between entries are ignored."""
        manifest_file = tmp_path / "manifest.jsonl"
        content = (
            json.dumps({"kernel_name": "a", "parallel_api": "cuda", "source_suite": "test"})
            + "\n\n\n"
            + json.dumps({"kernel_name": "b", "parallel_api": "omp", "source_suite": "test"})
            + "\n"
        )
        manifest_file.write_text(content, encoding="utf-8")
        entries = load_manifest(manifest_file)
        assert len(entries) == 2

    def test_load_manifest_empty_file(self, tmp_path: Path) -> None:
        """An empty manifest file returns an empty list."""
        manifest_file = tmp_path / "manifest.jsonl"
        manifest_file.write_text("", encoding="utf-8")
        entries = load_manifest(manifest_file)
        assert entries == []

    def test_load_manifest_missing_file(self) -> None:
        """Loading a nonexistent manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_manifest(Path("/nonexistent/manifest.jsonl"))
