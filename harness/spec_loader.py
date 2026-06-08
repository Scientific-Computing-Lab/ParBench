"""harness.spec_loader — Load specs, resolve paths, extract prompt payloads."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any


def load_config(project_root: Path) -> dict[str, Any]:
    """Load ``config/paths.json`` from the project root.

    Returns
    -------
    dict:
        Keys include ``project_root``, ``downloads_root``, ``hecbench_root``.
    """
    config_path = project_root / "config" / "paths.json"
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load every entry from a ``manifest.jsonl`` file.

    Parameters
    ----------
    manifest_path:
        Absolute or relative path to the JSONL manifest.

    Returns
    -------
    list[dict]:
        Each dict is one manifest entry.
    """
    entries: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_spec(spec_path: Path) -> dict[str, Any]:
    """Load a single Level-2 spec JSON file.

    Parameters
    ----------
    spec_path:
        Absolute or relative path to the spec JSON.

    Returns
    -------
    dict:
        The parsed spec.
    """
    with open(spec_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_paths(spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Resolve all relative paths in *spec* to absolute paths.

    Uses ``config/paths.json`` to locate ``downloads_root`` so that the
    spec's ``repo_root`` (a relative path like ``HeCBench-master/``) is
    resolved as ``downloads_root / repo_root``.

    The spec is **not** mutated — a shallow copy is returned with a new
    ``_resolved`` key added that contains the computed absolute paths:

    * ``repo_root``  — absolute path to the cloned repo root
    * ``source_dir`` — absolute path to the kernel source directory
    * ``working_dir`` — absolute path for build commands
    * ``files.prompt_payload``  — list of absolute file paths
    * ``files.support_files``   — list of absolute file paths
    * ``files.verification_only`` — list of absolute file paths
    * ``files.translation_targets`` — list of absolute file paths

    Parameters
    ----------
    spec:
        Parsed spec dict (unmodified).
    project_root:
        Absolute path to the *parbench/* project root.

    Returns
    -------
    dict:
        Shallow copy of *spec* with ``_resolved`` dict added.
    """
    project_root = project_root.resolve()

    # Load config to get downloads_root
    try:
        config = load_config(project_root)
        downloads_root = Path(config["downloads_root"]).resolve()
    except (FileNotFoundError, KeyError):
        # Fallback: resolve repo_root relative to project_root
        downloads_root = project_root

    provenance = spec.get("provenance", {})
    repo_root_rel = provenance.get("repo_root", "")
    source_path_rel = provenance.get("source_path", "")

    repo_root = (downloads_root / repo_root_rel).resolve()
    source_dir = (
        (repo_root / source_path_rel).resolve() if source_path_rel else repo_root
    )

    build = spec.get("build", {})
    working_dir_rel = build.get("working_directory", "")
    working_dir = (
        (repo_root / working_dir_rel).resolve() if working_dir_rel else source_dir
    )

    def _resolve_file_list(file_list: list[str]) -> list[Path]:
        """Resolve a list of filenames relative to source_dir."""
        return [(source_dir / fname).resolve() for fname in file_list]

    files = spec.get("files", {})

    resolved: dict[str, Any] = {
        "repo_root": repo_root,
        "source_dir": source_dir,
        "working_dir": working_dir,
        "files": {
            "prompt_payload": _resolve_file_list(files.get("prompt_payload", [])),
            "support_files": _resolve_file_list(files.get("support_files", [])),
            "verification_only": _resolve_file_list(files.get("verification_only", [])),
            "translation_targets": _resolve_file_list(files.get("translation_targets", [])),
        },
    }

    # Return a copy with _resolved attached
    out = dict(spec)
    out["_resolved"] = resolved
    return out


def get_prompt_payload(spec: dict[str, Any], project_root: Path, augment_level: int = 0) -> dict[str, str]:
    """Read every file in *files.prompt_payload* and return their contents.

    This is exactly what would be sent to the LLM for translation.

    Parameters
    ----------
    spec:
        Parsed spec dict.
    project_root:
        Absolute path to the project root.

    Returns
    -------
    dict[str, str]:
        Mapping of ``filename`` → ``file_contents``.
    """
    resolved = resolve_paths(spec, project_root)
    payload: dict[str, str] = {}

    ci_index = None
    aug_config = None
    if augment_level > 0:
        augment_level = max(1, min(4, augment_level))
        try:
            import clang.cindex as ci
            from c_augmentation.augment_dataset import (
                AugmentationConfig,
                ArithmeticTransform,
                SwapCondition,
                PointerArithmeticToArrayIndex,
                TypedefExpansion,
                ChangeNames,
                ChangeFunctionNames,
                augment_code
            )
            ci_index = ci.Index.create()
            aug_config = AugmentationConfig(
                level=augment_level,
                transforms=[
                    ArithmeticTransform(level=augment_level),
                    SwapCondition(level=augment_level),
                    PointerArithmeticToArrayIndex(level=augment_level),
                    TypedefExpansion(level=augment_level),
                    ChangeNames(level=augment_level),
                    ChangeFunctionNames(level=augment_level),
                ]
            )
        except ImportError as e:
            print(f"Warning: Failed to import c_augmentation modules: {e}")
            augment_level = 0

    for abs_path in resolved["_resolved"]["files"]["prompt_payload"]:
        path = Path(abs_path)
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")
            if augment_level > 0 and aug_config and ci_index and path.suffix in [".c", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".cl", ".cc", ".dp.cpp"]:
                content, _ = augment_code(content, aug_config, ci_index, filename=path.name)
            payload[path.name] = content
        else:
            payload[path.name] = f"<FILE NOT FOUND: {path}>"
    return payload


def find_translation_pairs(
    manifest: list[dict[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """Enumerate all valid (suite, kernel, source_api, target_api) translation pairs.

    Each kernel with *n* API variants produces ``n*(n-1)`` ordered pairs
    (A→B **and** B→A).  Pairs are scoped to a single suite — kernels with
    the same name in different suites are kept separate.

    Parameters
    ----------
    manifest:
        List of manifest entries (as returned by :func:`load_manifest`).

    Returns
    -------
    list[tuple[str, str, str, str]]:
        Each tuple is ``(source_suite, kernel_name, source_api, target_api)``.
    """
    from collections import defaultdict

    kernel_apis: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entry in manifest:
        key = (entry.get("source_suite", "unknown"), entry["kernel_name"])
        kernel_apis[key].append(entry["parallel_api"])

    pairs: list[tuple[str, str, str, str]] = []
    for (suite, kname), apis in sorted(kernel_apis.items()):
        for src, tgt in combinations(sorted(apis), 2):
            pairs.append((suite, kname, src, tgt))
            pairs.append((suite, kname, tgt, src))
    return pairs
