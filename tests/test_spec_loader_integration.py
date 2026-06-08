"""
Integration smoke tests for harness.spec_loader — real specs, real files.

Loads 1 spec per benchmark suite (Rodinia, XSBench, RSBench, mixbench, HeCBench),
verifies path resolution produces existing directories and non-empty prompt payloads.

Run:  python3 -m pytest tests/test_spec_loader_integration.py -v
       python3 -m pytest tests/test_spec_loader_integration.py -v -m integration

Requires: project files on disk (real benchmark source dirs).
"""

from __future__ import annotations

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


PROJECT_ROOT = Path(".")
SPECS_DIR = PROJECT_ROOT / "specs"
MANIFEST_PATH = PROJECT_ROOT / "manifest.jsonl"

# One known-PASS spec per suite — avoids KNOWN_FAIL specs.
SUITE_SPECS = {
    "rodinia": SPECS_DIR / "rodinia-bfs-cuda.json",
    "xsbench": SPECS_DIR / "xsbench-xsbench-cuda.json",
    "rsbench": SPECS_DIR / "rsbench-rsbench-cuda.json",
    "mixbench": SPECS_DIR / "mixbench-mixbench-cuda.json",
    "hecbench": SPECS_DIR / "hecbench-bezier-surface-cuda.json",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(SUITE_SPECS.keys()), ids=list(SUITE_SPECS.keys()))
def suite_spec(request: pytest.FixtureRequest) -> tuple[str, dict[str, Any]]:
    """Parametrized fixture: returns (suite_name, loaded_spec) for each suite."""
    suite = request.param
    spec_path = SUITE_SPECS[suite]
    spec = load_spec(spec_path)
    return suite, spec


# ---------------------------------------------------------------------------
# load_spec — real files
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLoadRealSpecPerSuite:
    """Verify load_spec works on one real spec from each of the 5 suites."""

    def test_spec_file_exists(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        suite, _ = suite_spec
        assert SUITE_SPECS[suite].exists(), f"Spec file missing: {SUITE_SPECS[suite]}"

    def test_identity_section(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """identity section has kernel_name, parallel_api, unique_id, source_suite."""
        suite, spec = suite_spec
        identity = spec["identity"]
        assert "kernel_name" in identity
        assert "parallel_api" in identity
        assert "unique_id" in identity
        assert identity["source_suite"] == suite

    def test_provenance_section(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """provenance section has repo_root and source_path."""
        _, spec = suite_spec
        provenance = spec["provenance"]
        assert "repo_root" in provenance
        assert "source_path" in provenance

    def test_files_section(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """files section has prompt_payload with at least one entry."""
        _, spec = suite_spec
        files = spec["files"]
        assert "prompt_payload" in files
        assert len(files["prompt_payload"]) >= 1
        assert "translation_targets" in files

    def test_build_section(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """build section has commands.build defined."""
        _, spec = suite_spec
        build = spec["build"]
        assert "commands" in build
        assert "build" in build["commands"]

    def test_unique_id_matches_filename(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """unique_id matches the spec filename convention {suite}-{kernel}-{api}.json."""
        suite, spec = suite_spec
        spec_path = SUITE_SPECS[suite]
        expected_stem = spec["identity"]["unique_id"]
        assert spec_path.stem == expected_stem


# ---------------------------------------------------------------------------
# resolve_paths — real files
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestResolvePathsReal:
    """Verify resolve_paths produces existing directories for real specs."""

    def test_resolved_source_dir_exists(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """source_dir resolves to an existing directory on disk."""
        _, spec = suite_spec
        resolved = resolve_paths(spec, PROJECT_ROOT)
        source_dir = resolved["_resolved"]["source_dir"]
        assert source_dir.is_dir(), f"source_dir does not exist: {source_dir}"

    def test_resolved_repo_root_exists(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """repo_root resolves to an existing directory on disk."""
        _, spec = suite_spec
        resolved = resolve_paths(spec, PROJECT_ROOT)
        repo_root = resolved["_resolved"]["repo_root"]
        assert repo_root.is_dir(), f"repo_root does not exist: {repo_root}"

    def test_resolved_prompt_payload_files_exist(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """All files in the resolved prompt_payload exist on disk."""
        _, spec = suite_spec
        resolved = resolve_paths(spec, PROJECT_ROOT)
        for fpath in resolved["_resolved"]["files"]["prompt_payload"]:
            assert fpath.exists(), f"prompt_payload file missing: {fpath}"

    def test_resolved_paths_are_absolute(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """All resolved paths are absolute."""
        _, spec = suite_spec
        resolved = resolve_paths(spec, PROJECT_ROOT)
        r = resolved["_resolved"]
        assert r["repo_root"].is_absolute()
        assert r["source_dir"].is_absolute()
        assert r["working_dir"].is_absolute()
        for fpath in r["files"]["prompt_payload"]:
            assert fpath.is_absolute()


# ---------------------------------------------------------------------------
# get_prompt_payload — real files
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPromptPayloadReal:
    """Verify get_prompt_payload returns non-empty source code from real specs."""

    def test_payload_is_non_empty(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """Prompt payload for each suite contains at least one non-empty file."""
        _, spec = suite_spec
        payload = get_prompt_payload(spec, PROJECT_ROOT)
        assert len(payload) >= 1
        for fname, content in payload.items():
            assert len(content) > 0, f"Empty content for {fname}"
            assert "FILE NOT FOUND" not in content, f"Missing file: {fname}"

    def test_payload_keys_match_spec(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """Payload filenames match spec's files.prompt_payload list."""
        _, spec = suite_spec
        payload = get_prompt_payload(spec, PROJECT_ROOT)
        expected_names = {Path(f).name for f in spec["files"]["prompt_payload"]}
        assert set(payload.keys()) == expected_names

    def test_payload_contains_code(self, suite_spec: tuple[str, dict[str, Any]]) -> None:
        """Payload files contain recognizable code (not binary junk)."""
        _, spec = suite_spec
        payload = get_prompt_payload(spec, PROJECT_ROOT)
        for fname, content in payload.items():
            # All benchmark source files contain at least one of these
            assert any(
                kw in content
                for kw in ["#include", "void", "int", "float", "double", "return",
                           "__global__", "__kernel", "#pragma", "template"]
            ), f"No recognizable code keywords in {fname}"


# ---------------------------------------------------------------------------
# find_translation_pairs — real manifest
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTranslationPairsReal:
    """Verify find_translation_pairs works on the real manifest."""

    def test_pairs_exist(self) -> None:
        """Real manifest produces a non-trivial number of translation pairs."""
        manifest = load_manifest(MANIFEST_PATH)
        pairs = find_translation_pairs(manifest)
        # With 96+ specs across 5 suites, there should be many pairs
        assert len(pairs) > 50

    def test_all_five_suites_represented(self) -> None:
        """All 5 benchmark suites appear in the generated pairs."""
        manifest = load_manifest(MANIFEST_PATH)
        pairs = find_translation_pairs(manifest)
        suites_in_pairs = {p[0] for p in pairs}
        for suite in ["rodinia", "xsbench", "rsbench", "mixbench", "hecbench"]:
            assert suite in suites_in_pairs, f"Suite {suite} missing from translation pairs"

    def test_pairs_are_directional(self) -> None:
        """For every (A→B) pair, there is a corresponding (B→A) pair."""
        manifest = load_manifest(MANIFEST_PATH)
        pairs = find_translation_pairs(manifest)
        pair_set = {(p[0], p[1], p[2], p[3]) for p in pairs}
        for suite, kernel, src, tgt in list(pair_set)[:20]:  # check first 20
            reverse = (suite, kernel, tgt, src)
            assert reverse in pair_set, f"Missing reverse pair for {suite}-{kernel} {src}→{tgt}"


# ---------------------------------------------------------------------------
# load_manifest — real file
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLoadManifestReal:
    """Verify load_manifest loads the real manifest.jsonl."""

    def test_manifest_loads(self) -> None:
        """Real manifest loads without error and has entries."""
        manifest = load_manifest(MANIFEST_PATH)
        assert len(manifest) > 0

    def test_manifest_entries_have_required_fields(self) -> None:
        """Every manifest entry has kernel_name, parallel_api, source_suite."""
        manifest = load_manifest(MANIFEST_PATH)
        for i, entry in enumerate(manifest):
            assert "kernel_name" in entry, f"Entry {i} missing kernel_name"
            assert "parallel_api" in entry, f"Entry {i} missing parallel_api"
            assert "source_suite" in entry, f"Entry {i} missing source_suite"

    def test_manifest_contains_all_suites(self) -> None:
        """The manifest references all 5 benchmark suites."""
        manifest = load_manifest(MANIFEST_PATH)
        suites = {e["source_suite"] for e in manifest}
        for suite in ["rodinia", "xsbench", "rsbench", "mixbench", "hecbench"]:
            assert suite in suites, f"Suite {suite} not in manifest"
