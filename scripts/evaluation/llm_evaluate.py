#!/usr/bin/env python3
"""
scripts/evaluation/llm_evaluate.py — LLM-based code translation evaluation.

Given a source spec (e.g. CUDA) and target spec (e.g. OpenMP), sends the
source code to an LLM, asks it to translate to the target API, then runs the
harness build/run/verify pipeline on the translated code.

Metrics captured per ParEval (HPDC 2024), BabelTower (ICML 2022), LASSI-EE:
  - Compilation success rate
  - Functional correctness (Pass@1)
  - Performance preservation (speedup ratio via wall-clock timing)

Usage:
    python3 scripts/evaluation/llm_evaluate.py \\
        --source specs/rodinia-bfs-cuda.json \\
        --target specs/rodinia-bfs-omp.json \\
        --model claude-sonnet-4-20250514 \\
        --project-root . -v

    python3 scripts/evaluation/llm_evaluate.py --list-models

    python3 scripts/evaluation/llm_evaluate.py \\
        --source specs/rodinia-bfs-cuda.json \\
        --target specs/rodinia-bfs-omp.json \\
        --model claude-sonnet-4-20250514 \\
        --project-root . \\
        --dry-run -v
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, NotRequired, TypedDict

# sys.path: scripts/evaluation/ -> scripts/ -> project_root/
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from harness.builder import build_spec  # noqa: E402
from harness.models import MetricResult, Status  # noqa: E402
from harness.runner import run_spec  # noqa: E402
from harness.spec_loader import get_prompt_payload, load_spec, resolve_paths  # noqa: E402
from harness.verifier import extract_metrics, verify_run  # noqa: E402
from scripts.evaluation.pair_contracts import (  # noqa: E402
    PairContract,
    PairContractError,
    PairContractRegistry,
    load_pair_contracts,
    validate_contract_against_specs,
)

# NOTE: 'C1'-'C4' labels in comments below refer to plan 02-10 Step 2 Bucket C items
# (Together/Azure/Groq/Gemini seed & temperature gates). These are UNRELATED to the
# pre-Phase-3 "Campaign 1 / Campaign 2" experimental taxonomy, which was replaced by
# 2026-04-20. Do NOT rename on the assumption they reference campaigns.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ModelRegistryEntry(TypedDict):
    provider: str
    supports_thinking: bool
    notes: str
    api_model: NotRequired[str]
    api_key_env: NotRequired[str]
    endpoint_env: NotRequired[str]
    wire_api: NotRequired[str]


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "supports_thinking": False,
        "notes": "Primary eval model",
    },
    "claude-sonnet-4-6-20260218": {
        "provider": "anthropic",
        "supports_thinking": False,
        "notes": "Latest Sonnet",
    },
    "claude-opus-4-6-20260205": {
        "provider": "anthropic",
        "supports_thinking": False,
        "notes": "Strongest reasoning",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "supports_thinking": False,
        "notes": "Fast/cheap baseline",
    },
    "gpt-4o": {
        "provider": "openai",
        "supports_thinking": False,
        "notes": "Strong general-purpose",
    },
    "o3-2025-04-16": {
        "provider": "openai",
        "supports_thinking": True,
        "notes": "Reasoning model",
    },
    "o4-mini-2025-04-16": {
        "provider": "openai",
        "supports_thinking": True,
        "notes": "Fast reasoning",
    },
    "azure-gpt-5.4": {
        "provider": "azure",
        "supports_thinking": True,
        "notes": "Azure OpenAI GPT-5.4 (Microsoft Foundry GA 2026-03-17; "
                 "requires AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + "
                 "gpt-5.4 deployment on norwayeast; "
                 "reasoning_effort=medium when --thinking=on)",
    },
    "azure-gpt-5.3-codex": {
        "provider": "azure",
        "supports_thinking": True,
        "api_model": "gpt-5.3-codex",
        "api_key_env": "AZURE_OPENAI_API_KEY_CODEX",
        "endpoint_env": "AZURE_OPENAI_ENDPOINT_CODEX",
        "wire_api": "responses",
        "notes": "Azure OpenAI gpt-5.3-codex (Responses API only; "
                 "requires AZURE_OPENAI_API_KEY_CODEX + AZURE_OPENAI_ENDPOINT_CODEX + "
                 "gpt-5.3-codex deployment on swedencentral; "
                 "reasoning_effort=medium when --thinking=on)",
    },
    "groq-llama-3.3-70b-versatile": {
        "provider": "groq",
        "supports_thinking": False,
        "notes": "Llama 3.3 70B via Groq (second eval model, Session 3)",
    },
    "gemini-2.5-flash-lite": {
        "provider": "google",
        "supports_thinking": False,
        "notes": "Gemini 2.5 Flash-Lite via Google AI (OpenAI-compatible endpoint)",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "supports_thinking": False,
        "notes": "Gemini 2.5 Flash via Google AI (thinking disabled via reasoning_effort=none)",
    },
    "together-qwen-3.5-397b-a17b": {
        "provider": "together",
        "supports_thinking": True,
        "api_model": "Qwen/Qwen3.5-397B-A17B",
        "notes": "Qwen 3.5 397B MoE via Together AI (thinking flag-driven, Plan 02-03)",
    },
}

# Human-readable API display names (fallback: .upper())
API_DISPLAY_NAMES: dict[str, str] = {
    "cuda": "CUDA",
    "hip": "HIP",
    "sycl": "SYCL",
    "opencl": "OpenCL",
    "omp": "OpenMP",
    "omp_target": "OpenMP Target Offload",
    "openacc": "OpenACC",
    "kokkos": "Kokkos",
    "raja": "RAJA",
    "mpi": "MPI",
    "omp_mpi": "OpenMP+MPI",
    "tbb": "Intel TBB",
    "stdpar": "C++ stdpar",
    "thrust": "Thrust",
    "serial": "Serial C/C++",
}

# Code fence language hint per API (used in prompts)
LANG_FOR_API: dict[str, str] = {
    "cuda": "cuda",
    "hip": "cpp",
    "sycl": "cpp",
    "opencl": "c",
    "omp": "cpp",
    "omp_target": "cpp",
    "openacc": "cpp",
    "kokkos": "cpp",
    "raja": "cpp",
    "mpi": "cpp",
    "omp_mpi": "cpp",
    "tbb": "cpp",
    "stdpar": "cpp",
    "thrust": "cpp",
    "serial": "c",
}

logger = logging.getLogger(__name__)

# Supported extensions for support file classification
_SUPPORT_HEADER_EXTS = frozenset({".h", ".hpp", ".cuh"})
_SUPPORT_CODE_EXTS = frozenset({".c", ".cpp", ".cu", ".cc", ".cxx"})


def _load_source_files_for_repair(
    source_spec: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    """Load source code files from the source spec for linker symbol search.

    Reads both prompt_payload and support_files (C/C++/CUDA sources and headers)
    from the source spec's source_dir. Used by analyze_build_failure() to locate
    definitions of symbols reported in linker errors.

    Returns:
        dict mapping filename → content. Empty dict if source_dir is missing.
    """
    source_resolved = resolve_paths(source_spec, project_root)
    source_dir: Path = source_resolved["_resolved"]["source_dir"]
    if not source_dir.is_dir():
        return {}

    source_files: dict[str, str] = {}

    # Collect filenames from prompt_payload and support_files
    files_section = source_spec.get("files") or {}
    filenames: list[str] = []
    filenames.extend(files_section.get("prompt_payload", []))
    filenames.extend(files_section.get("support_files", []))

    for fname in filenames:
        fp = source_dir / fname
        ext = Path(fname).suffix.lower()
        if ext in (".c", ".cpp", ".cu", ".cc", ".cxx", ".h", ".hpp", ".cuh"):
            try:
                source_files[fname] = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

    return source_files


def analyze_build_failure(
    build_error_snippet: str,
    source_files: dict[str, str],
) -> str:
    """Parse linker errors and generate targeted repair instructions.

    Analyzes undefined-reference and multiple-definition linker errors,
    searches source files for the missing/duplicate symbols, and returns
    human-readable repair hints for the LLM retry prompt.

    Handles both GCC and Clang linker error formats:
      - GCC:   undefined reference to `symbol'
      - Clang: undefined reference to `symbol'  (same backtick-quote style)
      - ld:    undefined reference to 'symbol'  (single quotes)
      - GCC:   multiple definition of `symbol'
      - Clang: duplicate symbol 'symbol'

    Returns empty string if no linker errors are detected.
    """
    if not build_error_snippet:
        return ""

    # Parse linker error patterns (GCC/Clang/ld formats)
    undefined = re.findall(
        r"undefined reference to [`'](\w+)'", build_error_snippet
    )
    multiple_def = re.findall(
        r"(?:multiple definition of|duplicate symbol) [`'](\w+)'",
        build_error_snippet,
    )

    if not undefined and not multiple_def:
        return ""  # Not a linker error — no hints to add

    repair_hints: list[str] = []

    for symbol in sorted(set(undefined)):
        # Search source files for function definitions matching the symbol
        for fname, content in source_files.items():
            # Match common C/C++/CUDA function definition patterns:
            #   return_type symbol(  or  __global__ void symbol(
            pattern = re.compile(
                rf"(?:^|\n)\s*(?:__global__|__device__|__host__|static|inline|extern|"
                rf"void|int|float|double|unsigned|long|short|char|size_t|auto|bool|"
                rf"template\s*<[^>]*>)\s+.*?\b{re.escape(symbol)}\s*\(",
                re.MULTILINE,
            )
            match = pattern.search(content)
            if match:
                # match.start() may land on the \n anchor before the line;
                # count newlines up through the match start for the correct line
                pos = match.start()
                if pos < len(content) and content[pos] == "\n":
                    pos += 1  # skip the anchor newline to get the actual line
                line_num = content[:pos].count("\n") + 1
                repair_hints.append(
                    f"MISSING SYMBOL: `{symbol}` -- found definition in source file "
                    f"`{fname}` around line {line_num}. You must include this function "
                    f"in your translated output. Inline the function body or ensure it "
                    f"is defined before use."
                )
                break
        else:
            repair_hints.append(
                f"MISSING SYMBOL: `{symbol}` -- not found in source files. "
                f"This may be a CUDA-specific function that needs an OpenMP/target-API "
                f"equivalent, or a utility function that should be included in your "
                f"translation."
            )

    for symbol in sorted(set(multiple_def)):
        repair_hints.append(
            f"DUPLICATE SYMBOL: `{symbol}` is defined in multiple output files. "
            f"Keep it in only one file (the main source) and use a forward "
            f"declaration or #include in others. Alternatively, mark it as "
            f"'static' or 'inline'."
        )

    if repair_hints:
        return (
            "\n\n## Linker Error Analysis\n\n"
            + "\n".join(f"- {h}" for h in repair_hints)
        )
    return ""


def _head_tail(text: str, max_len: int = 1500, head: int = 750, tail: int = 750) -> str:
    """Return head+tail of text if longer than max_len, else full text."""
    if len(text) <= max_len:
        return text
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def _derive_llm_seed(source_spec: str, target_spec: str, sample_id: int) -> int:
    """Deterministic 31-bit LLM sampling seed from (source_spec, target_spec, sample_id).

    Uses SHA-256 (not Python's builtin hash()) because builtin hash() is salted
    per-process via PYTHONHASHSEED. SHA-256 is deterministic across processes,
    so the same (source, target, sample_id) yields the same seed everywhere —
    enabling provider-level reproducibility when the endpoint honors `seed=`.
    """
    digest = hashlib.sha256(
        f"{source_spec}|{target_spec}|{str(sample_id)}".encode()
    ).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _strip_comments(code: str) -> str:
    """Strip C/C++ comments from source code, preserving string and char literals.

    Handles:
      - // line comments
      - /* block comments */
      - String literals ("..." including escaped quotes)
      - Character literals ('...' including escaped quotes)
      - Raw string literals (R"delim(...)delim")

    Uses a state-machine approach to avoid stripping comment-like sequences
    inside string or character literals.
    """
    result: list[str] = []
    i = 0
    n = len(code)

    while i < n:
        c = code[i]

        # Check for raw string literal: R"delim(...)delim"
        if c == 'R' and i + 1 < n and code[i + 1] == '"':
            # Find the delimiter between " and (
            delim_start = i + 2
            paren_pos = code.find('(', delim_start)
            if paren_pos != -1 and paren_pos - delim_start <= 16:
                delim = code[delim_start:paren_pos]
                # Find closing )delim"
                closing = ')' + delim + '"'
                end_pos = code.find(closing, paren_pos + 1)
                if end_pos != -1:
                    end_pos += len(closing)
                    result.append(code[i:end_pos])
                    i = end_pos
                    continue
            # Not a valid raw string — treat R as normal character
            result.append(c)
            i += 1
            continue

        # Check for string literal
        if c == '"':
            result.append(c)
            i += 1
            while i < n and code[i] != '"':
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i])
                    result.append(code[i + 1])
                    i += 2
                else:
                    result.append(code[i])
                    i += 1
            if i < n:
                result.append(code[i])  # closing "
                i += 1
            continue

        # Check for character literal
        if c == "'":
            result.append(c)
            i += 1
            while i < n and code[i] != "'":
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i])
                    result.append(code[i + 1])
                    i += 2
                else:
                    result.append(code[i])
                    i += 1
            if i < n:
                result.append(code[i])  # closing '
                i += 1
            continue

        # Check for line comment
        if c == '/' and i + 1 < n and code[i + 1] == '/':
            # Skip until end of line (but keep the newline)
            i += 2
            while i < n and code[i] != '\n':
                i += 1
            continue

        # Check for block comment
        if c == '/' and i + 1 < n and code[i + 1] == '*':
            i += 2
            while i + 1 < n and not (code[i] == '*' and code[i + 1] == '/'):
                # Preserve newlines to keep line numbers stable
                if code[i] == '\n':
                    result.append('\n')
                i += 1
            if i + 1 < n:
                i += 2  # skip */
            continue

        result.append(c)
        i += 1

    return ''.join(result)


def _read_support_files(
    source_spec: dict[str, Any],
    project_root: Path,
    max_chars: int = 50_000,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read support files from source spec, split into headers and code files.

    Returns:
        (headers_dict, code_dict) — both filename → content.
        headers_dict: all header files (.h, .hpp, .cuh)
        code_dict: code files (.c, .cpp, .cu, .cc, .cxx) up to max_chars cumulative
    """
    source_resolved = resolve_paths(source_spec, project_root)
    source_dir: Path = source_resolved["_resolved"]["source_dir"]
    support_names: list[str] = source_spec.get("files", {}).get("support_files", [])

    headers: dict[str, str] = {}
    code_files: dict[str, str] = {}
    code_chars = 0

    for fname in support_names:
        fp = source_dir / fname
        ext = Path(fname).suffix.lower()
        if ext in _SUPPORT_HEADER_EXTS:
            if fp.exists():
                headers[fname] = fp.read_text(encoding="utf-8", errors="replace")
            else:
                logger.warning("Support header not found on disk: %s", fp)
        elif ext in _SUPPORT_CODE_EXTS:
            if fp.exists():
                content = fp.read_text(encoding="utf-8", errors="replace")
                if code_chars + len(content) <= max_chars:
                    code_files[fname] = content
                    code_chars += len(content)
                else:
                    logger.debug(
                        "Skipping support code file %s (would exceed max_chars=%d)",
                        fname, max_chars,
                    )
        # Makefile, .sh, etc. are intentionally skipped

    return headers, code_files


def _read_target_infrastructure(
    target_spec: dict[str, Any],
    translation_targets: list[str],
    project_root: Path,
    max_chars: int = 50_000,
) -> dict[str, str]:
    """Read non-kernel files from the target spec as infrastructure context.

    Returns filename → content for:
    - prompt_payload files NOT in translation_targets (infrastructure source files)
    - support_files headers (.h, .hpp, .cuh) from the target spec

    The LLM sees these as read-only context — they exist in the target build
    directory and will not be modified.
    """
    target_resolved = resolve_paths(target_spec, project_root)
    source_dir: Path = target_resolved["_resolved"]["source_dir"]

    infrastructure: dict[str, str] = {}
    total_chars = 0
    tt_set = set(translation_targets)

    # Non-kernel prompt_payload files (everything in payload that the LLM won't produce)
    all_payload: list[str] = (target_spec.get("files") or {}).get("prompt_payload", [])
    for fname in all_payload:
        if fname in tt_set:
            continue
        fp = source_dir / fname
        if fp.exists():
            content = fp.read_text(encoding="utf-8", errors="replace")
            if total_chars + len(content) <= max_chars:
                infrastructure[fname] = content
                total_chars += len(content)
            else:
                logger.debug(
                    "Skipping target infra file %s (would exceed max_chars=%d)", fname, max_chars
                )

    # Support file headers from target spec (needed to understand interfaces)
    support_names: list[str] = (target_spec.get("files") or {}).get("support_files", [])
    for fname in support_names:
        if Path(fname).suffix.lower() not in _SUPPORT_HEADER_EXTS:
            continue
        fp = source_dir / fname
        if fp.exists():
            content = fp.read_text(encoding="utf-8", errors="replace")
            if total_chars + len(content) <= max_chars:
                infrastructure[fname] = content
                total_chars += len(content)
            else:
                logger.debug(
                    "Skipping target infra header %s (would exceed max_chars=%d)", fname, max_chars
                )

    return infrastructure


def _lang_hint_from_filename(fname: str) -> str:
    """Return a code fence language hint based on file extension."""
    ext = Path(fname).suffix.lower()
    return {
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
        ".cu": "cuda", ".cuh": "cpp",
        ".c": "c", ".h": "c", ".hpp": "cpp",
        ".cl": "c",
    }.get(ext, "c")


def _stage_support_headers(
    source_spec: dict[str, Any],
    target_spec_resolved: dict[str, Any],
    project_root: Path,
) -> list[Path]:
    """Copy source header files into target build directory (only if missing).

    Returns list of newly-created files for cleanup.
    """
    source_resolved = resolve_paths(source_spec, project_root)
    source_dir: Path = source_resolved["_resolved"]["source_dir"]
    target_dir: Path = target_spec_resolved["_resolved"]["source_dir"]
    support_names: list[str] = source_spec.get("files", {}).get("support_files", [])

    staged: list[Path] = []
    for fname in support_names:
        if Path(fname).suffix.lower() not in _SUPPORT_HEADER_EXTS:
            continue
        src_fp = source_dir / fname
        tgt_fp = target_dir / fname
        if src_fp.exists() and not tgt_fp.exists():
            tgt_fp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fp, tgt_fp)
            staged.append(tgt_fp)
            logger.debug("Staged header %s → %s", src_fp.name, target_dir)

    return staged


def _unstage_support_headers(staged_files: list[Path]) -> None:
    """Remove headers that were staged into the target build directory."""
    for fp in staged_files:
        fp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _anonymize_build_cmd(build_cmd: str, kernel_name: str) -> str:
    """Anonymize a build command by removing kernel-specific names.

    Replaces kernel name references with generic equivalents:
    - 'make bfs' → 'make' (use default target)
    - Preserves compiler flags and options
    """
    if not build_cmd or build_cmd == "(not specified)":
        return build_cmd

    # For simple 'make <target>' commands, strip the target to use default
    # (the Makefile's default target will build the right thing)
    make_with_target = re.match(
        r'^(make)\s+' + re.escape(kernel_name) + r'(\s|$)',
        build_cmd,
    )
    if make_with_target:
        rest = build_cmd[make_with_target.end():].strip()
        return f"make {rest}".strip() if rest else "make"

    # For other commands, replace the kernel name with a generic placeholder
    # (e.g. in 'gcc -o bfs bfs.c', replace 'bfs' but not flags)
    return build_cmd.replace(kernel_name, "program")


def build_translation_prompt(
    source_spec: dict[str, Any],
    target_spec: dict[str, Any],
    source_payload: dict[str, str],
    project_root: Path,
    source_support: tuple[dict[str, str], dict[str, str]] | None = None,
    target_infrastructure: dict[str, str] | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Build system + user messages for the translation request.

    Prompt anonymization is applied by default to prevent LLM data contamination:
    - Kernel name and description are stripped
    - Source code comments are stripped
    - Source/support file headers are genericized (Source File 1, Header File 1, etc.)
    - Target filenames are genericized (translated_0.ext) via anon_map
    - Build commands are anonymized

    Returns:
        (system_msg, user_msg, anon_map) where anon_map maps generic target
        filenames (shown to the LLM) back to real filenames (for disk I/O).
    """
    src_api = source_spec["identity"]["parallel_api"]
    tgt_api = target_spec["identity"]["parallel_api"]
    src_display = API_DISPLAY_NAMES.get(src_api, src_api.upper())
    tgt_display = API_DISPLAY_NAMES.get(tgt_api, tgt_api.upper())
    tgt_lang = LANG_FOR_API.get(tgt_api, "cpp")

    kernel_name = source_spec["identity"]["kernel_name"]

    # Target filenames the LLM must produce (kernel-centric: always use translation_targets)
    target_filenames: list[str] = target_spec["files"]["translation_targets"]

    # Build anonymization map: generic filename → real filename
    # e.g. "translated_0.cpp" → "bfs.cpp", "translated_1.h" → "bfs_kernel.h"
    anon_map: dict[str, str] = {}
    anon_target_filenames: list[str] = []
    for idx, real_fname in enumerate(target_filenames):
        ext = Path(real_fname).suffix  # e.g. ".cpp", ".cu", ".cl"
        generic_fname = f"translated_{idx}{ext}"
        anon_map[generic_fname] = real_fname
        anon_target_filenames.append(generic_fname)

    # Build command and environment from target spec
    build_cmd = (
        target_spec.get("build", {})
        .get("commands", {})
        .get("build", "(not specified)")
    )
    anon_build_cmd = _anonymize_build_cmd(build_cmd, kernel_name)

    sys_deps: list[str] = (
        target_spec.get("build", {})
        .get("environment", {})
        .get("system", {})
        .get("dependencies", [])
    )

    # ---- System message ----
    # No kernel name — only mentions API translation task generically
    system_msg = (
        f"You are a parallel programming expert specializing in {src_display} to "
        f"{tgt_display} translation. Translate the provided source code to {tgt_display}. "
        f"Output ONLY the translated code, no explanations. "
        f"For each file, output a markdown code fence with the filename on the opening line:\n\n"
        f"```{tgt_lang} filename={{filename}}\n<code>\n```\n\n"
        f"Preserve the algorithm, I/O behavior, data formats, and output format exactly. "
        f"The translated code must compile with the provided build command."
    )

    # ---- User message ----
    lines: list[str] = []
    lines.append("## Translation Task")
    # Kernel name and description intentionally omitted (anonymization)
    lines.append(f"Source API: {src_display} → Target API: {tgt_display}")
    lines.append("")

    lines.append("## Target Files to Produce")
    for fname in anon_target_filenames:
        lines.append(f"- {fname}")
    if target_infrastructure:
        lines.append("")
        lines.append(
            "_These files will replace the corresponding files in the target project directory. "
            "All other project files (Makefile, headers, utilities) remain unchanged._"
        )
    lines.append("")

    lines.append("## Build Command (your code must work with this)")
    lines.append("```")
    lines.append(anon_build_cmd)
    lines.append("```")
    lines.append("")

    if sys_deps:
        lines.append("## Build Environment")
        for dep in sys_deps:
            lines.append(f"- {dep}")
        lines.append("")

    lines.append(f"## Source Code ({src_display})")
    for file_idx, (fname, contents) in enumerate(source_payload.items(), start=1):
        src_lang = LANG_FOR_API.get(src_api, "c")
        lines.append(f"### Source File {file_idx}")
        lines.append(f"```{src_lang}")
        lines.append(_strip_comments(contents))
        lines.append("```")
        lines.append("")

    if source_support:
        headers, code = source_support
        if headers or code:
            lines.append("## Support / Header Files")
            lines.append(
                "_These files exist in the **source** build directory but may NOT exist "
                "in the target directory. If your translated code needs definitions from "
                "them, inline the definitions directly rather than using `#include`._"
            )
            lines.append("")
            header_idx = 0
            code_idx = 0
            for fname, contents in {**headers, **code}.items():
                src_lang = LANG_FOR_API.get(src_api, "c")
                ext = Path(fname).suffix.lower()
                if ext in _SUPPORT_HEADER_EXTS:
                    header_idx += 1
                    lines.append(f"### Header File {header_idx}")
                else:
                    code_idx += 1
                    lines.append(f"### Code File {code_idx}")
                lines.append(f"```{src_lang}")
                lines.append(_strip_comments(contents))
                lines.append("```")
                lines.append("")

    if target_infrastructure:
        lines.append("## Target Infrastructure Context (DO NOT MODIFY — for reference only)")
        lines.append(
            "_These files exist in the target build directory and will NOT be modified. "
            "Your translated code must be compatible with them._"
        )
        lines.append("")
        infra_idx = 0
        for fname, contents in target_infrastructure.items():
            infra_idx += 1
            lang = _lang_hint_from_filename(fname)
            lines.append(f"### Infrastructure File {infra_idx}")
            lines.append(f"```{lang}")
            lines.append(_strip_comments(contents))
            lines.append("```")
            lines.append("")

    user_msg = "\n".join(lines)
    return system_msg, user_msg, anon_map


# ---------------------------------------------------------------------------
# LLM call (provider adapter)
# ---------------------------------------------------------------------------


def call_llm(
    model: str,
    system_msg: str,
    messages: list[dict[str, str]],
    verbose: bool = False,
    temperature: float = 0.0,
    thinking_enabled: bool = False,
    seed: int | None = None,
    top_p: float = 1.0,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Call the LLM and return response + usage metadata.

    ``max_completion_tokens`` overrides the default per-call response limit
    (32768; Gemini 65536, Together 81920) on every provider path. The bounded agentic-repair protocol
    (Task 10, gate E-08) passes its dynamic completion cap here; the live
    eval path leaves it ``None`` and keeps the default.

    Args:
        model: Model ID string (routing by prefix).
        system_msg: System-level instruction string.
        messages: Conversation list of {"role": ..., "content": ...} dicts.
            Supports multi-turn for retry loops.
        verbose: Log request info to stderr.
        temperature: Sampling temperature (0.0 = greedy/deterministic,
            0.5+ = stochastic for pass@k multi-sampling).
        seed: Deterministic sampling seed. Passed only to providers that
            accept `seed=` (Together AI, Azure OpenAI). Anthropic does not
            support `seed=` and ignores this argument. Reasoning-model
            paths (o1/o3/o4) omit it along with temperature/top_p.
        top_p: Nucleus-sampling parameter. Default 1.0 = full-distribution
            sampling at the matched temperature (NOT argmax). Explicitly
            set across all providers to decouple from provider-specific
            defaults (Together AI default: 0.7; Azure defaults differ).

    Returns:
        {response_text, prompt_tokens, completion_tokens, duration_seconds, finish_reason}

    Provider routing:
        claude-*          → Anthropic SDK (ANTHROPIC_API_KEY)
        gpt-* / o1-* / o3-* / o4-*  → OpenAI SDK (OPENAI_API_KEY)
        azure-*           → Azure OpenAI SDK (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)
                            Strips "azure-" prefix to get deployment name.
        groq-*            → OpenAI SDK (GROQ_API_KEY, base_url=https://api.groq.com/openai/v1)
                            Strips "groq-" prefix to get model name.
        gemini-*          → OpenAI SDK (GEMINI_API_KEY or GOOGLE_API_KEY,
                            base_url=https://generativelanguage.googleapis.com/v1beta/openai/)
                            Model name passed as-is (no prefix stripping).
        together-*        → OpenAI SDK (TOGETHER_API_KEY, base_url=https://api.together.xyz/v1)
                            Uses api_model from MODEL_REGISTRY, or strips "together-" prefix.

    ParaCodex (future):
        Add `elif model.startswith("paracodex")` branch here.
        Same dict return format — rest of pipeline is unchanged.
    """
    t0 = time.monotonic()

    # Per-call response cap: protocol dynamic cap when provided, else default.
    _response_cap = max_completion_tokens if max_completion_tokens is not None else 32768

    if model.startswith("claude-"):
        # ---- Anthropic path ----
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Set ANTHROPIC_API_KEY environment variable to use Anthropic models."
            )
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: python3 -m pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=api_key)
        if verbose:
            logger.info(
                "Calling Anthropic model=%s messages=%d", model, len(messages)
            )
        # No `thinking` parameter = extended thinking disabled (base capability only).
        # This ensures parity with Gemini (reasoning_effort="none") and Groq (no
        # thinking capability). All models evaluated at equivalent base level.
        # Plan 02-10 Step 2 (C3): top_p=1.0 explicit (full-distribution, not argmax).
        # Anthropic does not accept `seed=`; omit seed here.
        response = client.messages.create(
            model=model,
            max_tokens=_response_cap,
            temperature=temperature,
            top_p=top_p,
            system=system_msg,
            messages=messages,
        )
        response_text: str = response.content[0].text
        prompt_tokens: int = response.usage.input_tokens
        completion_tokens: int = response.usage.output_tokens
        finish_reason: str = response.stop_reason or "unknown"
        # Normalize to common vocabulary: "stop" (completed), "length" (truncated).
        # Anthropic uses "end_turn"/"max_tokens"; OpenAI-compatible use "stop"/"length".
        finish_reason = {"end_turn": "stop", "max_tokens": "length"}.get(finish_reason, finish_reason)

    elif model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        # ---- OpenAI path ----
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY environment variable to use OpenAI models."
            )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: python3 -m pip install openai"
            )

        client = openai.OpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": system_msg}] + messages
        if verbose:
            logger.info(
                "Calling OpenAI model=%s messages=%d", model, len(full_messages)
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": _response_cap,
            "messages": full_messages,
        }
        # o1/o3/o4 reasoning models do not accept temperature or top_p
        # Plan 02-10 Step 2 (C3): top_p=1.0 explicit on non-reasoning paths.
        # `seed` is not added here (out of C1/C2 scope); add if OpenAI direct
        # path is ever used for a Phase 3 model requiring reproducibility.
        if not model.startswith(("o1-", "o3-", "o4-")):
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        response = client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason or "unknown"

    elif model.startswith("azure-"):
        # ---- Azure OpenAI path ----
        _reg = MODEL_REGISTRY.get(model, {})
        _key_env = _reg.get("api_key_env", "AZURE_OPENAI_API_KEY")
        _endpoint_env = _reg.get("endpoint_env", "AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get(_key_env)
        endpoint = os.environ.get(_endpoint_env)
        if not api_key:
            raise ValueError(
                f"Set {_key_env} environment variable to use {model}."
            )
        if not endpoint:
            raise ValueError(
                f"Set {_endpoint_env} environment variable to use {model}."
            )
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: python3 -m pip install openai"
            )

        _api_model = _reg.get("api_model")
        azure_model = _api_model if _api_model is not None else model[len("azure-"):]

        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        base_endpoint = f"{parsed.scheme}://{parsed.netloc}"

        _wire_api = _reg.get("wire_api", "chat")

        if _wire_api == "responses":
            # ---- Responses API path (gpt-5.3-codex and future models) ----
            client_az = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_endpoint,
                api_version="2025-04-01-preview",
            )
            if verbose:
                logger.info(
                    "Calling Azure OpenAI Responses API deployment=%s", azure_model
                )
            _resp_kwargs: dict[str, Any] = {
                "model": azure_model,
                "instructions": system_msg,
                "input": [{"role": m["role"], "content": m["content"]} for m in messages],
                "max_output_tokens": _response_cap,
                "store": False,
            }
            if thinking_enabled and _reg.get("supports_thinking", False):
                _resp_kwargs["reasoning"] = {"effort": "medium"}
            response = client_az.responses.create(**_resp_kwargs)
            response_text = response.output_text or ""
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            finish_reason = response.status or "unknown"
        else:
            # ---- Chat Completions path (gpt-5.4) ----
            client_az = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_endpoint,
                api_version="2025-01-01-preview",
            )
            full_messages = [{"role": "system", "content": system_msg}] + messages
            if verbose:
                logger.info(
                    "Calling Azure OpenAI deployment=%s messages=%d", azure_model, len(full_messages)
                )
            _az_kwargs: dict[str, Any] = {
                "model": azure_model,
                "max_completion_tokens": _response_cap,
                "messages": full_messages,
            }
            if not _reg.get("supports_thinking", False):
                _az_kwargs["temperature"] = temperature
                _az_kwargs["top_p"] = top_p
            if seed is not None:
                _az_kwargs["seed"] = seed
            if thinking_enabled and _reg.get("supports_thinking", False):
                _az_kwargs["reasoning_effort"] = "medium"
            response = client_az.chat.completions.create(**_az_kwargs)
            response_text = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            finish_reason = response.choices[0].finish_reason or "unknown"

    elif model.startswith("groq-"):
        # ---- Groq (OpenAI-compatible) path ----
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GROQ_API_KEY environment variable to use Groq models."
            )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: python3 -m pip install openai"
            )

        groq_model = model[len("groq-"):]  # "groq-llama-3.3-70b-versatile" → "llama-3.3-70b-versatile"
        client_groq = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        full_messages = [{"role": "system", "content": system_msg}] + messages
        if verbose:
            logger.info(
                "Calling Groq model=%s messages=%d", groq_model, len(full_messages)
            )
        # Plan 02-10 Step 2 (C3): top_p=1.0 explicit. Groq is OpenAI-compatible;
        # `seed` is not added here (out of C1/C2 scope, Groq not in Phase 3 lineup).
        response = client_groq.chat.completions.create(
            model=groq_model,
            max_tokens=_response_cap,
            temperature=temperature,
            top_p=top_p,
            messages=full_messages,
        )
        response_text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason or "unknown"

    elif model.startswith("gemini-"):
        # ---- Google AI (OpenAI-compatible) path ----
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable to use Gemini models."
            )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: python3 -m pip install openai"
            )

        client_gemini = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        full_messages = [{"role": "system", "content": system_msg}] + messages
        if verbose:
            logger.info(
                "Calling Gemini model=%s messages=%d", model, len(full_messages)
            )
        # Plan 02-10 Step 2 (C3): top_p=1.0 explicit. `seed` is not added here
        # (out of C1/C2 scope, Gemini historical only).
        response = client_gemini.chat.completions.create(
            model=model,
            max_tokens=(max_completion_tokens if max_completion_tokens is not None else 65536),
            temperature=temperature,
            top_p=top_p,
            messages=full_messages,
            # Explicitly disable thinking/reasoning so all models are evaluated
            # at equivalent base capability — no inference-time compute scaling.
            # Flash Lite has thinking OFF by default, but we force it explicitly.
            # Ref: https://ai.google.dev/gemini-api/docs/openai#thinking
            reasoning_effort="none",
        )
        response_text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason or "unknown"

    elif model.startswith("together-"):
        # ---- Together AI (OpenAI-compatible) path ----
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "Set TOGETHER_API_KEY environment variable to use Together AI models."
            )
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: python3 -m pip install openai"
            )

        # Resolve actual API model name: prefer MODEL_REGISTRY.api_model, else strip prefix
        registry_entry = MODEL_REGISTRY.get(model, {})
        together_model = registry_entry.get("api_model", model[len("together-"):])

        client_together = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
        full_messages = [{"role": "system", "content": system_msg}] + messages
        if verbose:
            logger.info(
                "Calling Together AI model=%s (api_model=%s) messages=%d",
                model, together_model, len(full_messages),
            )
        # Plan 02-10 Step 2 (C1, C3): seed + top_p=1.0 for sampling reproducibility.
        # Together AI accepts `seed=` and `top_p=` (verified 2026-04-18 smoke test).
        # `seed` is best-effort on MoE models — identical seeds do not guarantee
        # bit-identical completions on Qwen3.5-397B (observed empirically). Setting
        # it still enables provider-level reproducibility metadata and is honored
        # by providers that do lock on seed.
        _together_kwargs: dict[str, Any] = {
            "model": together_model,
            "max_tokens": (max_completion_tokens if max_completion_tokens is not None else 81920),
            "temperature": temperature,
            "top_p": top_p,
            "messages": full_messages,
            # enable_thinking driven by --thinking flag via thinking_enabled bool
            # (Plan 02-03). Uses Together AI's chat_template_kwargs passthrough to
            # set enable_thinking in the Jinja2 chat template.
            # Ref: https://docs.together.ai/reference/chat-completions
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": thinking_enabled},
            },
        }
        if seed is not None:
            _together_kwargs["seed"] = seed
        response = client_together.chat.completions.create(**_together_kwargs)
        response_text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason or "unknown"

    else:
        raise ValueError(
            f"Unknown model provider for '{model}'. "
            "Expected prefix: claude-*, gpt-*, o1-*, o3-*, o4-*, azure-*, groq-*, gemini-*, together-*"
        )

    duration = time.monotonic() - t0
    return {
        "response_text": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "duration_seconds": round(duration, 3),
    }


# ---------------------------------------------------------------------------
# Think-tag stripping (belt-and-suspenders for reasoning models)
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DANGLING_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM response text.

    Handles three cases:
      1. Complete <think>...</think> blocks anywhere in the response.
      2. Dangling </think> at the start (opening <think> was in a prior chunk
         or the response started mid-thought).
      3. Unclosed <think>... at the end (response was truncated mid-thought).
    """
    # Case 1: complete blocks
    text = _THINK_BLOCK_RE.sub("", text)
    # Case 2: dangling </think> (no opening tag found — strip everything before it)
    if "</think>" in text and "<think>" not in text:
        text = _DANGLING_THINK_RE.sub("", text)
    # Case 3: unclosed <think> at the end
    if "<think>" in text and "</think>" not in text:
        text = text[: text.index("<think>")]
    return text.strip()


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------


def _deanonymize_extracted(
    anon_map: dict[str, str], anon_extracted: dict[str, str]
) -> dict[str, str]:
    """De-anonymize extracted code: rename dict keys AND patch filenames in code bodies.

    The LLM sees generic filenames (translated_0.cu) and may emit cross-file
    #include "translated_1.cu" references. These must be replaced with real
    filenames so the code compiles correctly after extraction.
    """
    result: dict[str, str] = {}
    for generic_fname, code in anon_extracted.items():
        real_fname = anon_map[generic_fname]
        for gen_fn, real_fn in anon_map.items():
            code = code.replace(f'"{gen_fn}"', f'"{real_fn}"')
            code = code.replace(f"'{gen_fn}'", f"'{real_fn}'")
        result[real_fname] = code
    return result


def extract_code_blocks(
    response_text: str,
    target_filenames: list[str],
) -> dict[str, str]:
    """Extract translated code from LLM response into a filename → code dict.

    Four-tier fallback:
        Tier 1   — Fenced blocks with ``filename=X`` annotation (Claude/GPT style).
        Tier 1.5 — Fenced blocks with space-separated filename (``lang filename.ext``; Llama style).
        Tier 2   — Filename mentioned near a code fence (fuzzy match).
        Tier 3   — Single target file + single code block in response.
    """
    result: dict[str, str] = {}

    # Tier 1: ```lang filename=X\n...\n```  (Claude / GPT-4 style)
    tier1_pattern = re.compile(
        r"```\w*\s+filename=([^\s`]+)\s*\n(.*?)```",
        re.DOTALL,
    )
    for match in tier1_pattern.finditer(response_text):
        raw_fname = match.group(1).strip("\"'")
        code = match.group(2)
        # Match against target filenames (exact or basename)
        for tgt in target_filenames:
            if tgt not in result and (
                raw_fname == tgt or Path(raw_fname).name == Path(tgt).name
            ):
                result[tgt] = code
                break

    unmatched = [f for f in target_filenames if f not in result]
    if not unmatched:
        return result

    # Tier 1.5: ```lang filename.ext\n...\n```  (Llama / space-separated style, no filename=)
    tier1b_pattern = re.compile(
        r"```\w*\s+([^\s`=]+\.[^\s`]+)\s*\n(.*?)```",
        re.DOTALL,
    )
    for match in tier1b_pattern.finditer(response_text):
        raw_fname = match.group(1).strip("\"'")
        code = match.group(2)
        for tgt in list(unmatched):
            if raw_fname == tgt or Path(raw_fname).name == Path(tgt).name:
                result[tgt] = code
                unmatched.remove(tgt)
                break

    if not unmatched:
        return result

    # Extract all generic code blocks for tiers 2 & 3
    generic_pattern = re.compile(r"```\w*\n(.*?)```", re.DOTALL)
    all_blocks = [m.group(1) for m in generic_pattern.finditer(response_text)]

    # Tier 2: filename mentioned within ~500 chars before a code fence
    for fname in list(unmatched):
        fname_pattern = re.compile(
            re.escape(fname) + r".{0,500}?```\w*\n(.*?)```",
            re.DOTALL,
        )
        m = fname_pattern.search(response_text)
        if m:
            result[fname] = m.group(1)
            unmatched.remove(fname)

    # Tier 3: exactly 1 target file remaining + exactly 1 code block total
    if len(unmatched) == 1 and len(all_blocks) == 1:
        result[unmatched[0]] = all_blocks[0]
        unmatched.clear()

    for fname in unmatched:
        logger.warning("Could not extract code for target file: %s", fname)

    return result


# ---------------------------------------------------------------------------
# File backup / restore
# ---------------------------------------------------------------------------


def backup_files(file_paths: list[Path]) -> dict[Path, bool]:
    """Copy each file to <path>.parbench_bak if it exists.

    A pre-existing ``.parbench_bak`` is KEPT, never overwritten: restore
    always deletes the backup, so a surviving one means a prior invocation
    was killed between backup and restore — the current file may be a stale
    translation and the existing backup is the only pristine copy.
    Overwriting it here laundered exactly that contamination into two
    benchmark trees during the 2026-08-12 replay campaign.

    Returns:
        dict mapping Path → bool (True = file existed before backup).
    """
    backup_info: dict[Path, bool] = {}
    for fp in file_paths:
        if fp.exists():
            bak = Path(str(fp) + ".parbench_bak")
            if not bak.exists():
                shutil.copy2(fp, bak)
            backup_info[fp] = True
        else:
            backup_info[fp] = False
    return backup_info


def restore_files(backup_info: dict[Path, bool]) -> None:
    """Restore all backed-up files; delete any that were newly created."""
    for fp, existed in backup_info.items():
        bak = Path(str(fp) + ".parbench_bak")
        if existed and bak.exists():
            shutil.copy2(bak, fp)
            bak.unlink()
        elif not existed and fp.exists():
            fp.unlink()
        # Clean up orphaned .parbench_bak if original was removed
        if bak.exists():
            bak.unlink()


# ---------------------------------------------------------------------------
# Retry feedback
# ---------------------------------------------------------------------------


# Patterns that indicate a runtime error even when exit_code==0 and
# stdout_pattern matches.  These catch false positives like OpenCL kernel
# compilation failures where the host program gracefully continues.
_STDOUT_ERROR_PATTERNS = [
    (re.compile(r"clBuildProgram\s*\(\)\s*=>\s*-\d+", re.IGNORECASE), "OpenCL kernel build failure"),
    (re.compile(r"clCompileProgram\s*\(\)\s*=>\s*-\d+", re.IGNORECASE), "OpenCL kernel compile failure"),
    (re.compile(r"Segmentation fault", re.IGNORECASE), "Segfault in stdout"),
    (re.compile(r"CUDA error:", re.IGNORECASE), "CUDA runtime error"),
    (re.compile(r"cudaError(?!Success)\w*:", re.IGNORECASE), "CUDA error"),
]


def _check_stdout_error_indicators(stdout: str) -> str | None:
    """Return a rejection reason if stdout contains known error indicators, else None."""
    for pattern, reason in _STDOUT_ERROR_PATTERNS:
        m = pattern.search(stdout)
        if m:
            return f"{reason}: {m.group(0)}"
    return None


def _is_kernel_only_translation(target_spec: dict) -> bool:
    """Check if this is a kernel-only translation (e.g., OpenCL .cl files).

    For kernel-only translations, the host code is untouched — only compute
    kernel files are translated. This means the translated binary uses the
    TARGET's argc parsing and stdout format, not the SOURCE's.

    Currently this applies to OpenCL targets where translation_targets are
    all .cl files. CUDA (.cu) and OMP (.c/.cpp) targets contain host code
    and are full-program translations.
    """
    targets = target_spec.get("files", {}).get("translation_targets", [])
    if not targets:
        return False
    return all(t.endswith(".cl") for t in targets)


def _require_pair_contract(
    target_spec: dict, source_spec: dict, pair_contract: PairContract | None
) -> PairContract:
    """Fail-closed gate shared by both cross-API spec builders (Task 2, E-02).

    Raises PairContractError when the contract is missing, unresolved, names
    a different pair than the supplied specs, or declares shared inputs that
    do not match the specs' actual correctness arguments (mismatched inputs).
    The same validation runs pre-model in evaluate_translation(); repeating
    it here keeps the builders safe for every other caller.
    """
    if pair_contract is None:
        source_id = source_spec.get("identity", {}).get("unique_id", "?")
        target_id = target_spec.get("identity", {}).get("unique_id", "?")
        raise PairContractError(
            f"no pair contract supplied for cross-API translation "
            f"{source_id} -> {target_id}; refusing (fail closed)"
        )
    validate_contract_against_specs(pair_contract, source_spec, target_spec)
    return pair_contract


def _build_cross_api_verify_spec(
    target_spec: dict, source_spec: dict, pair_contract: PairContract | None
) -> dict:
    """Build the verification spec for a cross-API translation (Task 2, E-02).

    Verification uses ONLY the declared pair contract: the pair's target
    oracle plus its negative checks, conjunctive under the harness verifier.
    The pre-Task-2 union oracle (source/target success-pattern alternation)
    is gone — source-side success text can no longer satisfy a target-side
    verification, and a missing or unresolved contract fails closed.

    Same-API translations (augmentation-only) should NOT use this function —
    they use the target spec directly.
    """
    contract = _require_pair_contract(target_spec, source_spec, pair_contract)
    verify_spec = copy.deepcopy(target_spec)
    verify_spec["verification"] = verify_spec.get("verification") or {}
    verify_spec["verification"]["strategies"] = (
        copy.deepcopy(contract.target_oracle["strategies"])
        + copy.deepcopy(contract.negative_checks)
    )
    return verify_spec


def _build_cross_api_run_spec(
    target_spec: dict, source_spec: dict, pair_contract: PairContract | None
) -> dict:
    """Build the run spec for a cross-API translation (Task 2, E-02).

    The translated binary runs with the pair contract's declared
    translated_run arguments and environment on top of the TARGET spec's
    executable, timeout, and working directory — never an unconditional copy
    of the source arguments. A missing or unresolved contract fails closed.

    Same-API translations (augmentation-only) should NOT use this function.
    """
    contract = _require_pair_contract(target_spec, source_spec, pair_contract)
    run_spec_dict = copy.deepcopy(target_spec)
    translated_run = contract.translated_run
    configs = run_spec_dict.setdefault("run", {}).setdefault("input_configurations", {})
    correctness = configs.setdefault("correctness", {})
    correctness["arguments"] = list(translated_run["args"])
    environment = translated_run.get("environment") or {}
    if environment:
        run_block = run_spec_dict["run"]
        merged = dict(run_block.get("environment_variables") or {})
        merged.update(environment)
        run_block["environment_variables"] = merged
    return run_spec_dict


def _build_retry_message(
    build_result: Any | None,
    run_result: Any | None,
    verify_result: Any | None,
    *,
    verify_status_only: bool = False,
) -> str:
    """Format error feedback for the LLM retry message.

    With ``verify_status_only=True`` (the bounded agentic-repair protocol,
    ``diagnostics_exposed_to_model: verify_status_only``), a verification
    failure reports only its status: ``verify_result.details`` carries oracle
    contents (stdout patterns, expected hashes, numeric expectations,
    extraction regexes) and must never reach the agent (gate finding 3,
    2026-08-12). The live eval path keeps the detailed message.
    """
    if build_result is not None and build_result.status != Status.PASS:
        # Trim to last 500 chars of stderr to stay within token budget
        stderr_snippet = (build_result.stderr or "")[-500:].strip()
        return (
            f"The translation failed to compile with this error:\n```\n"
            f"{stderr_snippet}\n```\nFix the code so it compiles with the "
            "provided build command."
        )
    if run_result is not None and run_result.status != Status.PASS:
        stderr_snippet = (run_result.stderr or "")[-300:].strip()
        return (
            f"The code compiled but execution failed (exit code {run_result.exit_code}):\n"
            f"```\n{stderr_snippet}\n```\nFix the runtime error."
        )
    if verify_result is not None and verify_result.status != Status.PASS:
        if verify_status_only:
            return (
                "Build and run succeeded but output verification failed "
                "(status only; verifier details are withheld by the bounded "
                "agentic-repair protocol).\n"
                "Ensure the translated code produces the same output format "
                "as the original."
            )
        return (
            f"Build and run succeeded but output verification failed: "
            f"{verify_result.details}\n"
            "Ensure the translated code produces the same output format as the "
            "original."
        )
    return "The translation failed for an unknown reason. Please try again."


# ---------------------------------------------------------------------------
# Bounded agentic-repair protocol (plan Task 10, gate E-08)
# ---------------------------------------------------------------------------
# Enforcement code for the versioned protocol in
# config/agentic_repair_protocol.json (schema:
# schema/agentic_repair_protocol_schema.json). The TRAJECTORY budget below is
# a TOTAL across every attempt of one task and is distinct from the per-call
# response limits hardcoded in call_llm() (e.g. max_tokens=32768).
#
# No measured agentic comparison runs from this code. Two consumers exist:
# scripts/evaluation/dry_run_agentic_protocol.py (synthetic, provider-free:
# it replaces call_llm with a raiser, the replay_stored_translations.py
# pattern) and the opt-in `agentic_protocol=` parameter of
# evaluate_translation() (gate finding 1, 2026-08-12), which no production
# caller passes — a measured agentic run still requires a separately
# approved cost and sampling design.

STOP_REASON_PASS = "PASS"
STOP_REASON_ATTEMPT_LIMIT = "ATTEMPT_LIMIT_REACHED"
STOP_REASON_TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
STOP_REASON_PROVIDER_ERROR = "PROVIDER_ERROR"
STOP_REASON_PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
# The evaluation loop's caller-imposed max_retries ended the trajectory below
# the protocol's own attempt cap (gate 2 finding 2, 2026-08-12): distinct from
# ATTEMPT_LIMIT_REACHED, which means the protocol's 3-attempt bound was hit.
STOP_REASON_CALLER_ATTEMPT_LIMIT = "CALLER_ATTEMPT_LIMIT_REACHED"

AGENTIC_PROTOCOL_CONFIG_PATH = (
    _SCRIPT_DIR.parent.parent / "config" / "agentic_repair_protocol.json"
)
AGENTIC_PROTOCOL_SCHEMA_PATH = (
    _SCRIPT_DIR.parent.parent / "schema" / "agentic_repair_protocol_schema.json"
)

# There is NO length exemption in the visibility check (gate 4 finding,
# 2026-08-12: a 6-char floor let short oracle strings like "PASS" reach the
# provider unchecked). Instead, a snippet is exempt only when it already
# occurs verbatim in the agent's PERMITTED inputs (its own source payload and
# the read-only infrastructure context): content the agent legitimately holds
# carries no hidden information, while any other occurrence - of any length -
# is a leak and fails closed.


class AgenticProtocolError(ValueError):
    """Raised when the protocol config is missing, malformed, or violated."""


class AgenticVisibilityError(AgenticProtocolError):
    """Raised when an agent-visible request contains forbidden content."""


def load_agentic_protocol(
    config_path: Path | None = None,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    """Load and schema-validate the versioned agentic-repair protocol config."""
    config_path = config_path or AGENTIC_PROTOCOL_CONFIG_PATH
    schema_path = schema_path or AGENTIC_PROTOCOL_SCHEMA_PATH
    try:
        protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticProtocolError(f"cannot load protocol config {config_path}: {exc}")
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticProtocolError(f"cannot load protocol schema {schema_path}: {exc}")

    import jsonschema  # deferred: only protocol consumers need it

    try:
        jsonschema.validate(protocol, schema)
    except jsonschema.ValidationError as exc:
        raise AgenticProtocolError(f"protocol config fails its schema: {exc.message}")

    attempts = protocol["attempts"]
    if (
        attempts["max_total_attempts"]
        != attempts["original_attempts"] + attempts["max_repair_attempts"]
    ):
        raise AgenticProtocolError(
            "attempts.max_total_attempts must equal original_attempts + "
            "max_repair_attempts"
        )
    return protocol


def remaining_trajectory_tokens(total_budget: int, normalized_used: int) -> int:
    """Tokens still available in the trajectory budget (never negative)."""
    if total_budget < 0 or normalized_used < 0:
        raise AgenticProtocolError(
            f"token counts must be non-negative "
            f"(total_budget={total_budget}, normalized_used={normalized_used})"
        )
    return max(0, total_budget - normalized_used)


def may_start_attempt(
    remaining: int, next_prompt_tokens: int, maximum_completion_tokens: int
) -> bool:
    """Pre-attempt reservation: prompt + max completion must fit *remaining*."""
    if remaining < 0 or next_prompt_tokens < 0 or maximum_completion_tokens < 0:
        raise AgenticProtocolError("reservation inputs must be non-negative")
    return next_prompt_tokens + maximum_completion_tokens <= remaining


def normalize_provider_usage(raw_usage: dict[str, Any]) -> dict[str, int]:
    """Normalize any provider usage payload to the protocol's common fields.

    Accepts Anthropic (``input_tokens``/``output_tokens``), OpenAI-compatible
    (``prompt_tokens``/``completion_tokens`` — OpenAI, Azure, Groq, Together,
    Gemini-via-OpenAI-endpoint), and Google-native
    (``promptTokenCount``/``candidatesTokenCount``) shapes.

    Returns ``{"prompt_tokens", "completion_tokens", "total_tokens"}``.
    """
    field_pairs = (
        ("prompt_tokens", "completion_tokens"),      # OpenAI-compatible
        ("input_tokens", "output_tokens"),           # Anthropic
        ("promptTokenCount", "candidatesTokenCount"),  # Google native
    )
    for prompt_key, completion_key in field_pairs:
        if prompt_key in raw_usage and completion_key in raw_usage:
            prompt = int(raw_usage[prompt_key])
            completion = int(raw_usage[completion_key])
            if prompt < 0 or completion < 0:
                raise AgenticProtocolError(
                    f"negative token count in provider usage: {raw_usage}"
                )
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
    raise AgenticProtocolError(
        f"unrecognized provider usage shape: {sorted(raw_usage)}"
    )


def estimate_prompt_tokens_conservative(
    system_msg: str,
    messages: list[dict[str, str]],
    *,
    bytes_per_token: float = 1.0,
    per_message_overhead_tokens: int = 8,
) -> int:
    """PROVABLE upper bound on the token count of the complete next prompt.

    Byte-level BPE tokenizers (every provider family the protocol normalizes)
    emit tokens covering at least one byte each, so token count <= UTF-8 byte
    count. Counting BYTES at ``bytes_per_token <= 1`` therefore yields a
    provable upper bound, not a heuristic (gate 2 finding 1, 2026-08-12: the
    earlier chars/3 heuristic under-estimated real usage, letting
    prompt + completion cap exceed the remaining budget in real spend).
    ``bytes_per_token > 1`` would divide the byte count below the bound and
    is refused. The per-message overhead covers chat-format wrapper tokens.
    """
    if bytes_per_token <= 0:
        raise AgenticProtocolError("bytes_per_token must be positive")
    if bytes_per_token > 1:
        raise AgenticProtocolError(
            f"bytes_per_token={bytes_per_token} is not a provable upper bound "
            "for byte-level tokenizers; it must be <= 1"
        )
    if per_message_overhead_tokens < 8:
        # Role encoding and chat-template wrappers ride on every message
        # (gate 4 finding 2, 2026-08-12), so a positive envelope allowance
        # is required. PRECISE GUARANTEE (gate 5 finding 1, 2026-08-12):
        # the byte term provably bounds CONTENT tokens; the 8-token
        # envelope allowance is an assumption, not a proof. What IS proven
        # is the ledger: settlement fails closed when a provider reports
        # prompt usage above this estimate, so recorded usage can never
        # exceed the budget and the worst case is one already-sent request
        # bounded by its reservation (prompt estimate + completion cap),
        # surfaced as a protocol violation. Exact provider-side counting
        # would need each provider's tokenizer at call time, which the
        # no-provider-call protocol artifact deliberately excludes;
        # recorded as a documented limitation in the Task 10 log.
        raise AgenticProtocolError(
            f"per_message_overhead_tokens={per_message_overhead_tokens} does "
            "not cover the provider envelope; it must be >= 8"
        )
    total_bytes = len(system_msg.encode("utf-8")) + sum(
        len(m.get("content", "").encode("utf-8")) for m in messages
    )
    n_messages = 1 + len(messages)  # system counts as one message
    # math.ceil keeps the estimate conservative for ANY positive
    # bytes_per_token; int-truncating first crashed on fractions < 1
    # (gate finding 6, 2026-08-12).
    return math.ceil(total_bytes / bytes_per_token) + n_messages * per_message_overhead_tokens


def dynamic_completion_cap(
    remaining: int, next_prompt_tokens: int, per_call_response_limit: int
) -> int:
    """Provider output cap for the next call: min(per-call limit, remaining - prompt)."""
    return max(0, min(per_call_response_limit, remaining - next_prompt_tokens))


def collect_forbidden_agentic_content(
    source_spec: dict[str, Any],
    target_spec: dict[str, Any],
    project_root: Path,
) -> dict[str, list[str]]:
    """Gather every text snippet the agent path must never receive.

    Categories mirror ``visibility.forbidden_to_agent`` in the protocol config:

    - ``reference_implementations`` / ``target_implementations``: on-disk
      contents of the TARGET spec's ``translation_targets`` files — the
      shipped reference solution of the translation task. Target
      ``prompt_payload`` files that are NOT translation targets are
      infrastructure the prompt design deliberately shows read-only
      (kernel-centric translation, M11) and are NOT forbidden
      (gate finding 4 ruling, 2026-08-12).
    - ``verification_only_files``: contents of both specs'
      ``files.verification_only`` entries.
    - ``hidden_expected_outputs``: contents of declared
      ``verification.reference_files`` and ``file_diff`` reference files.
    - ``oracle_contents``: verifier strategy internals — stdout patterns,
      expected hashes, numeric expected values, extraction regexes.
    """
    forbidden: dict[str, list[str]] = {
        "reference_implementations": [],
        "target_implementations": [],
        "verification_only_files": [],
        "hidden_expected_outputs": [],
        "oracle_contents": [],
    }

    def _read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    target_resolved = resolve_paths(target_spec, project_root)["_resolved"]
    target_files = target_resolved["files"]
    for path in target_files.get("translation_targets", []):
        content = _read(path)
        if content:
            forbidden["reference_implementations"].append(content)
            forbidden["target_implementations"].append(content)

    for spec in (source_spec, target_spec):
        spec_resolved = resolve_paths(spec, project_root)["_resolved"]
        for path in spec_resolved["files"].get("verification_only", []):
            content = _read(path)
            if content:
                forbidden["verification_only_files"].append(content)

        # Reference paths are declared relative to the project root in real
        # specs ("specs/references/..."), but older/synthetic specs use
        # source-dir-relative paths. Read every candidate that exists —
        # extra reads only add forbidden content (gate 3 finding, 2026-08-12).
        def _read_reference(rel: str) -> None:
            for base in (project_root, spec_resolved["source_dir"]):
                content = _read(base / rel)
                if content:
                    forbidden["hidden_expected_outputs"].append(content)

        verification = spec.get("verification") or {}
        for ref in verification.get("reference_files") or []:
            forbidden["hidden_expected_outputs"].append(str(ref.get("sha256", "")))
            _read_reference(str(ref.get("path", "")))
        for strategy in verification.get("strategies") or []:
            for key in ("pattern", "expected_sha256", "extract_regex"):
                if strategy.get(key):
                    forbidden["oracle_contents"].append(str(strategy[key]))
            if strategy.get("type") == "numeric_comparison" and "expected" in strategy:
                forbidden["oracle_contents"].append(str(strategy["expected"]))
            if strategy.get("reference_file"):
                _read_reference(str(strategy["reference_file"]))

    return forbidden


def collect_allowed_agentic_content(
    source_spec: dict[str, Any],
    target_spec: dict[str, Any],
    project_root: Path,
) -> str:
    """Concatenate the agent's PERMITTED inputs for the visibility baseline.

    The baseline is EXACTLY the file-derived text the prompt inserts:
    comment-stripped payload, capped support files, and capped target
    infrastructure, gathered through the same production readers
    (``_read_support_files`` / ``_read_target_infrastructure``) the live
    path passes to ``build_translation_prompt``. Not raw files (falsely
    refused 13 real prompts, gate 5, 2026-08-12), not the whole composed
    prompt (would exempt a prompt-builder bug injecting oracle text,
    gate 6 finding 1). This helper mirrors the live path for corpus tests
    and offline audits.
    """
    payload: dict[str, str] = {}
    source_resolved = resolve_paths(source_spec, project_root)["_resolved"]
    for path in source_resolved["files"].get("prompt_payload", []):
        try:
            payload[Path(path).name] = path.read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            pass
    if not payload:
        return ""
    support_headers, support_code = _read_support_files(source_spec, project_root)
    target_infrastructure = _read_target_infrastructure(
        target_spec,
        (target_spec.get("files") or {}).get("translation_targets", []),
        project_root,
    )
    return "\n".join(
        _strip_comments(content)
        for collection in (
            payload, support_headers, support_code, target_infrastructure)
        for content in collection.values()
    )


def check_agentic_visibility(
    request_messages: list[dict[str, str]] | str,
    forbidden: dict[str, list[str]],
    allowed_text: str | None = None,
) -> list[str]:
    """Return one violation string per forbidden snippet found in the request.

    A snippet of ANY length is a violation when it appears in the request,
    unless it already occurs verbatim in *allowed_text* - the agent's
    permitted inputs (source payload plus read-only infrastructure). A
    snippet the agent legitimately holds carries no hidden information; the
    prior 6-char exemption instead let short oracle strings ("PASS", "0.0")
    through unchecked (gate 4 finding, 2026-08-12). With no *allowed_text*
    the check is strict: every forbidden occurrence flags.
    """
    if isinstance(request_messages, str):
        request_text = request_messages
    else:
        request_text = "\n".join(m.get("content", "") for m in request_messages)
    violations: list[str] = []
    for category, snippets in forbidden.items():
        for snippet in snippets:
            # Prompts insert file contents AFTER comment stripping, so a
            # leaked file appears in its transformed form. Screen both the
            # raw snippet and its comment-stripped variant (gate finding 4,
            # 2026-08-12).
            variants = {snippet.strip(), _strip_comments(snippet).strip()}
            for variant in variants:
                if not variant:
                    continue
                if variant not in request_text:
                    continue
                if allowed_text is not None and variant in allowed_text:
                    continue
                violations.append(
                    f"{category}: forbidden snippet of {len(variant)} chars "
                    f"present in agent-visible request"
                )
                break
    return violations


def enforce_agentic_visibility(
    request_messages: list[dict[str, str]] | str,
    forbidden: dict[str, list[str]],
    allowed_text: str | None = None,
) -> None:
    """Raise :class:`AgenticVisibilityError` when the request leaks forbidden content."""
    violations = check_agentic_visibility(request_messages, forbidden, allowed_text)
    if violations:
        raise AgenticVisibilityError("; ".join(violations))


class AgenticTrajectoryBudget:
    """Stateful enforcement of the bounded agentic-repair protocol.

    One instance covers one trajectory (one task). Before EVERY attempt —
    including the original — call :meth:`begin_attempt` with a conservative
    estimate of the complete next prompt. The reservation refuses any attempt
    whose prompt + maximum completion could exceed the remaining trajectory
    budget, and refuses a fourth attempt unconditionally.

    Check order in :meth:`begin_attempt` is budget first, then attempt cap:
    an attempt that cannot be funded is never considered started. When both
    bounds are hit the refusal record says so.
    """

    def __init__(self, protocol: dict[str, Any]):
        self.protocol = protocol
        budget = protocol["token_budget"]
        self.total_budget: int = budget["trajectory_total_tokens"]
        self.per_call_response_limit: int = budget["per_call_response_limit_tokens"]
        self.protocol_minimum_completion: int = budget[
            "protocol_minimum_completion_tokens"
        ]
        self.max_total_attempts: int = protocol["attempts"]["max_total_attempts"]
        self.normalized_used: int = 0
        self.attempts_started: int = 0
        # Tokens reserved by the last allowed begin_attempt() (prompt estimate
        # + completion cap). record_attempt_usage() fails closed against it.
        self._active_reservation: int | None = None
        # Prompt estimate of the last allowed begin_attempt(); actual prompt
        # usage above it breaks the provable bound (gate 2 finding 1).
        self._active_prompt_estimate: int | None = None

    @property
    def remaining(self) -> int:
        return remaining_trajectory_tokens(self.total_budget, self.normalized_used)

    def begin_attempt(self, next_prompt_tokens: int) -> dict[str, Any]:
        """Decide whether the next attempt may start.

        Returns a decision record::

            {allowed, attempt_number, completion_cap, stop_reason,
             remaining_before, attempt_cap_also_reached}

        ``completion_cap`` is the dynamic provider output cap
        (``min(per-call limit, remaining - prompt)``); when it cannot satisfy
        the protocol minimum, the attempt is refused with
        ``TOKEN_BUDGET_EXHAUSTED``.
        """
        remaining = self.remaining
        cap = dynamic_completion_cap(
            remaining, next_prompt_tokens, self.per_call_response_limit
        )
        attempt_cap_reached = self.attempts_started >= self.max_total_attempts
        decision: dict[str, Any] = {
            "attempt_number": self.attempts_started + 1,
            "next_prompt_tokens": next_prompt_tokens,
            "remaining_before": remaining,
            "completion_cap": cap,
            "attempt_cap_also_reached": attempt_cap_reached,
            "allowed": False,
            "stop_reason": None,
        }
        if (
            cap < self.protocol_minimum_completion
            or not may_start_attempt(remaining, next_prompt_tokens, cap)
        ):
            decision["stop_reason"] = STOP_REASON_TOKEN_BUDGET_EXHAUSTED
            return decision
        if attempt_cap_reached:
            decision["stop_reason"] = STOP_REASON_ATTEMPT_LIMIT
            return decision
        decision["allowed"] = True
        self.attempts_started += 1
        self._active_reservation = next_prompt_tokens + cap
        self._active_prompt_estimate = next_prompt_tokens
        return decision

    def record_attempt_usage(self, raw_usage: dict[str, Any]) -> dict[str, int]:
        """Normalize provider usage and charge it against the trajectory budget.

        Fails closed (gate finding 2, 2026-08-12): usage may only be recorded
        for an attempt begin_attempt() allowed, and the normalized total must
        not exceed that attempt's reservation (prompt estimate + completion
        cap) nor the remaining trajectory budget. A violation raises
        :class:`AgenticProtocolError` and charges nothing.
        """
        usage = normalize_provider_usage(raw_usage)
        if self._active_reservation is None:
            raise AgenticProtocolError(
                "record_attempt_usage called without an allowed begin_attempt "
                "reservation; refusing to charge usage"
            )
        if usage["total_tokens"] > self._active_reservation:
            raise AgenticProtocolError(
                f"provider usage {usage['total_tokens']} tokens exceeds the "
                f"attempt reservation of {self._active_reservation} tokens; "
                "failing closed"
            )
        if usage["prompt_tokens"] > self._active_prompt_estimate:
            # Gate 2 finding 1 (2026-08-12): the reservation is only a bound
            # if the estimate dominates actual prompt usage. A provider
            # counting more prompt tokens than the byte upper bound predicted
            # is a protocol violation, never absorbed.
            raise AgenticProtocolError(
                f"provider prompt usage {usage['prompt_tokens']} tokens "
                f"exceeds the attempt's estimated upper bound of "
                f"{self._active_prompt_estimate} tokens; failing closed"
            )
        if self.normalized_used + usage["total_tokens"] > self.total_budget:
            raise AgenticProtocolError(
                f"provider usage {usage['total_tokens']} tokens would push "
                f"cumulative usage past the trajectory budget "
                f"({self.normalized_used}+{usage['total_tokens']} > "
                f"{self.total_budget}); failing closed"
            )
        self._active_reservation = None
        self._active_prompt_estimate = None
        self.normalized_used += usage["total_tokens"]
        return usage


# ---------------------------------------------------------------------------
# Main orchestrator (importable by batch runner)
# ---------------------------------------------------------------------------


def evaluate_translation(
    source_path: Path,
    target_path: Path,
    model: str,
    project_root: Path,
    augment_level: int = 0,
    max_retries: int = 1,
    verbose: bool = False,
    dry_run: bool = False,
    use_cpu_timing: bool = False,
    temperature: float = 0.0,
    sample_id: int = 0,
    save_to_disk: bool = True,
    thinking: str = "on",
    num_samples: int = 1,
    pair_registry: PairContractRegistry | None = None,
    agentic_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate source → target using the LLM, then build/run/verify.

    Designed for import by run_eval_batch.py (Task 5).

    Args:
        source_path: Path to source spec JSON.
        target_path: Path to target spec JSON.
        model: LLM model ID string.
        project_root: Absolute path to parbench root.
        augment_level: Augmentation level for source code (0 = no transforms).
        max_retries: Max LLM call attempts. 1 = zero-shot. >1 = iterative repair.
        verbose: Log verbose output.
        dry_run: Build and print prompt without calling LLM.
        use_cpu_timing: If True, measure CPU time (user+system) via
            /usr/bin/time -v when running translated code.  The cpu_time_seconds
            field is then preferred over wall_time_seconds for the speedup
            denominator (Linux/GNU time required).
        agentic_protocol: A loaded bounded agentic-repair protocol dict
            (load_agentic_protocol(), Task 10 / gate E-08). When provided,
            every attempt is gated by the trajectory token budget and the
            visibility rule, provider output is capped dynamically, retry
            feedback exposes verification status only, and the result JSON
            gains an "agentic_protocol" block. ``None`` (the default) keeps
            live eval behavior unchanged.

    Returns:
        Result dict matching the schema documented in the plan.
    """
    source_spec = load_spec(source_path)
    target_spec = load_spec(target_path)
    target_spec_resolved = resolve_paths(target_spec, project_root)

    source_id = source_spec["identity"]["unique_id"]
    target_id = target_spec["identity"]["unique_id"]
    kernel_name = source_spec["identity"]["kernel_name"]
    source_api = source_spec.get("identity", {}).get("parallel_api", "")
    target_api = target_spec.get("identity", {}).get("parallel_api", "")

    # Plan 02-03 (D-06..D-10): resolve the --thinking flag against the model's
    # capability once per call. `thinking_enabled` is the authoritative boolean
    # used downstream (call_llm, result JSON). The flag is a no-op for models
    # with supports_thinking=False; log at DEBUG so it's visible without spam.
    _model_caps = MODEL_REGISTRY.get(model, {})
    thinking_enabled: bool = (thinking == "on") and bool(
        _model_caps.get("supports_thinking", False)
    )
    if not _model_caps.get("supports_thinking", False):
        logger.debug(
            "--thinking=%s ignored for %s (supports_thinking=False)", thinking, model
        )

    if verbose:
        logger.info("Evaluating: %s → %s  model=%s", source_id, target_id, model)

    # Plan 02-10 Step 2 (C1/C2/C3/C4): sampling reproducibility metadata.
    # Seed is deterministic across processes via SHA-256 (unlike Python's
    # salted builtin hash()). top_p=1.0 decouples from provider defaults.
    # Both values are recorded in the result JSON (C4) and passed through to
    # providers that accept the corresponding kwargs (C1=Together, C2=Azure;
    # top_p universal — C3).
    llm_sampling_seed: int = _derive_llm_seed(source_id, target_id, sample_id)
    llm_top_p: float = 1.0

    # Get source code (optionally augmented).
    # Seed random before augmentation so L1-L4 transforms are deterministic
    # per (spec, level) pair. Convention: seed = 42 + augment_level (eval-pipeline-specific).
    if augment_level > 0:
        random.seed(42 + augment_level)
    source_payload = get_prompt_payload(source_spec, project_root, augment_level)

    # Read support files (headers + code) to include in prompt
    source_support = _read_support_files(source_spec, project_root)

    # Pre-load source files for linker error analysis during retries
    source_files_for_repair = _load_source_files_for_repair(source_spec, project_root)

    # Kernel-centric target filenames (all specs have translation_targets after SESSION 1.6)
    translation_mode = "kernel_centric"
    prompt_target_filenames: list[str] = target_spec["files"]["translation_targets"]

    # Read target infrastructure context (non-kernel files as read-only reference)
    # Returns {} for Family 3 specs where targets == prompt_payload (no infra to show)
    target_infrastructure = _read_target_infrastructure(
        target_spec, prompt_target_filenames, project_root
    )

    # Build prompt (returns anon_map: generic filename → real filename)
    system_msg, user_msg, anon_map = build_translation_prompt(
        source_spec, target_spec, source_payload, project_root,
        source_support=source_support,
        target_infrastructure=target_infrastructure,
    )
    # Generic filenames shown to LLM (keys of anon_map)
    anon_target_filenames: list[str] = list(anon_map.keys())

    if dry_run:
        print("=" * 70)
        print("SYSTEM MESSAGE")
        print("=" * 70)
        print(system_msg)
        print()
        print("=" * 70)
        print("USER MESSAGE")
        print("=" * 70)
        print(user_msg)
        return {
            "source_spec": source_id,
            "target_spec": target_id,
            "kernel": kernel_name,
            "model": model,
            "augment_level": augment_level,
            "dry_run": True,
            "overall_status": "DRY_RUN",
        }

    # Pair-contract gate (Task 2, E-02): a cross-API translation must have a
    # resolved declared contract BEFORE any model invocation. A missing
    # registry, missing entry, or unresolved entry raises PairContractError
    # here — no provider call is made for an uncontracted pair. (--dry-run
    # returns above without a provider call, so it stays contract-free.)
    pair_contract: PairContract | None = None
    if source_api != target_api:
        if pair_registry is None:
            raise PairContractError(
                f"cross-API translation {source_id} -> {target_id} requires a "
                "pair-contract registry (--pair-contracts); refusing before "
                "model invocation"
            )
        pair_contract = pair_registry.resolve(source_id, target_id)
        # Full pre-model validation, including the declared-vs-actual
        # shared-input comparison — a mismatched contract must never reach
        # the provider (Codex gate pass 1, finding 1).
        validate_contract_against_specs(pair_contract, source_spec, target_spec)

    # Resolve target file paths for backup/restore
    # Use same kernel-centric target_filenames as the prompt (always translation_targets)
    resolved: dict[str, Any] = target_spec_resolved.get("_resolved", {})
    source_dir: Path = resolved.get("source_dir", project_root)
    target_filenames: list[str] = prompt_target_filenames
    target_file_paths: list[Path] = [source_dir / fname for fname in target_filenames]

    # Backup originals (restored in finally regardless of outcome)
    backup_info = backup_files(target_file_paths)

    # Stage source headers into target build dir (safety net for missing includes)
    staged_headers = _stage_support_headers(source_spec, target_spec_resolved, project_root)

    # Baseline timing from target spec (may be null/absent)
    # Use `or {}` guard — .get(key, {}) returns None if key exists with null value
    baseline_wall_time: float | None = (
        (target_spec.get("baseline_results") or {})
        .get("configurations", {})
        .get("correctness", {})
        .get("wall_time_seconds")
    )

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    attempts: list[dict[str, Any]] = []

    # Bounded agentic-repair protocol (Task 10, gate E-08): budget controller
    # + forbidden-content inventory, active only when a protocol is passed.
    agentic_budget: AgenticTrajectoryBudget | None = None
    agentic_forbidden: dict[str, list[str]] | None = None
    agentic_stop_reason: str | None = None
    agentic_refusal: dict[str, Any] | None = None
    if agentic_protocol is not None:
        agentic_budget = AgenticTrajectoryBudget(agentic_protocol)
        agentic_forbidden = collect_forbidden_agentic_content(
            source_spec, target_spec, project_root
        )
        # The permitted baseline is EXACTLY the file-derived text the prompt
        # inserts: the stripped payload, capped support files, and capped
        # target infrastructure (the same objects passed to
        # build_translation_prompt, transformed the same way its lines
        # insert them). Not raw files (falsely refused 13 real prompts,
        # gate 5), not the whole composed prompt (would exempt a
        # prompt-builder bug injecting oracle text, gate 6 finding 1). A
        # forbidden snippet occurring in this baseline carries no hidden
        # information; any other occurrence, at any length, is a leak.
        agentic_allowed_text = "\n".join(
            _strip_comments(content)
            for collection in (
                source_payload, *source_support, target_infrastructure)
            for content in collection.values()
        )

    # Multi-turn conversation (supports iterative repair)
    messages: list[dict[str, str]] = [{"role": "user", "content": user_msg}]

    # Accumulated totals across attempts
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_time = 0.0

    final_build_result = None
    final_run_result = None
    final_verify_result = None
    final_metrics: list[MetricResult] = []
    final_status = "ERROR"
    error_message: str | None = None
    last_llm_response: str = ""  # saved across loop iterations for preview
    _correctness_args: list[str] = []  # populated in run stage for result metadata

    # Accumulated extracted files across all attempts (merge for partial extraction).
    # On partial EXTRACTION_FAIL retries, the LLM produces only missing files.
    # This dict keeps successfully extracted files from prior attempts so they
    # aren't lost when restore_files() wipes the working directory.
    accumulated_extracted: dict[str, str] = {}

    try:
        for attempt_num in range(1, max_retries + 1):
            # Reset per-attempt accumulators so the final result JSON always
            # reflects only the last attempt's pipeline stage results, not
            # stale data from an earlier attempt that reached a later stage.
            final_build_result = None
            final_run_result = None
            final_verify_result = None
            final_metrics = []

            if verbose:
                logger.info("Attempt %d/%d", attempt_num, max_retries)

            attempt_record: dict[str, Any] = {
                "attempt": attempt_num,
                "llm_response_time_seconds": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "finish_reason": None,
                "build_status": None,
                "run_status": None,
                "verify_status": None,
                "error_feedback_sent": None,
                "build_error_snippet": None,
                "run_stderr_snippet": None,
                "run_stdout_snippet": None,
                "extraction_fail": None,
            }

            # -- Bounded agentic-repair protocol gate (Task 10, gate E-08) --
            # Before EVERY attempt: screen the complete agent-visible request
            # for forbidden content, then reserve prompt + capped completion
            # against the trajectory budget. A refusal ends the trajectory
            # with an explicit protocol stop reason, before any provider call.
            llm_max_completion: int | None = None
            _agentic_attempt_t0: float | None = None
            if agentic_budget is not None:
                _estimator = agentic_protocol["token_budget"]["prompt_estimator"]
                _next_prompt_tokens = estimate_prompt_tokens_conservative(
                    system_msg,
                    messages,
                    bytes_per_token=_estimator["bytes_per_token"],
                    per_message_overhead_tokens=_estimator["per_message_overhead_tokens"],
                )
                enforce_agentic_visibility(
                    [{"role": "system", "content": system_msg}, *messages],
                    agentic_forbidden,
                    agentic_allowed_text,
                )
                _decision = agentic_budget.begin_attempt(_next_prompt_tokens)
                if not _decision["allowed"]:
                    agentic_stop_reason = _decision["stop_reason"]
                    agentic_refusal = _decision
                    if not attempts:
                        error_message = (
                            f"agentic protocol refused attempt "
                            f"{_decision['attempt_number']}: "
                            f"{_decision['stop_reason']}"
                        )
                    break
                llm_max_completion = _decision["completion_cap"]
                attempt_record["agentic_prompt_tokens_estimated"] = _next_prompt_tokens
                attempt_record["agentic_completion_cap_tokens"] = llm_max_completion
                # Full-attempt wall clock starts at budget admission and is
                # stamped on EVERY exit path via _stamp_agentic_wall(); the
                # redacted prompt record is captured pre-call so even a
                # provider-error attempt carries it (gate 5 finding 3,
                # 2026-08-12).
                _agentic_attempt_t0 = time.time()
                attempt_record["agentic_prompt_record_redacted"] = [
                    {
                        "role": m["role"],
                        "chars": len(m.get("content", "")),
                        "sha256": hashlib.sha256(
                            m.get("content", "").encode("utf-8")).hexdigest(),
                    }
                    for m in ({"role": "system", "content": system_msg}, *messages)
                ]
                attempt_record["agentic_response_record_redacted"] = None
                # This attempt is now allowed, so the previous attempt's
                # prepared feedback is genuinely part of a provider-bound
                # request: only NOW may it be marked sent (gate 2 finding 3,
                # 2026-08-12). A refused next attempt leaves it unsent.
                if attempts and attempts[-1].get("error_feedback_prepared") is not None:
                    attempts[-1]["error_feedback_sent"] = attempts[-1][
                        "error_feedback_prepared"
                    ]
                # Diagnostics this attempt's request actually carries: the
                # redacted sizes of every prior-failure feedback message in
                # the pre-call message list (empty for the original attempt).
                attempt_record["diagnostics_exposed_to_model"] = [
                    f"[REDACTED: {len(m['content'])} chars of failure diagnostics]"
                    for m in messages[1:]
                    if m["role"] == "user"
                ]

            # -- Call LLM --
            # Plan 02-10 Step 2 (C1/C2/C3): deterministic SHA-256 seed (same for
            # all attempts of this task) + explicit top_p=1.0 (full-distribution
            # at matched temperature, NOT argmax).
            try:
                llm_result = call_llm(
                    model, system_msg, messages,
                    verbose=verbose, temperature=temperature,
                    thinking_enabled=thinking_enabled,
                    seed=llm_sampling_seed,
                    top_p=llm_top_p,
                    max_completion_tokens=llm_max_completion,
                )
            except Exception as exc:
                error_message = f"LLM call failed: {exc}"
                logger.error(error_message)
                attempt_record["error"] = error_message
                # OD-4 (2026-08-20): record a DISTINCT provider-error verdict so
                # the record is clearly rerunnable. Before this, a provider
                # exception on a later attempt left final_status at whatever the
                # PREVIOUS attempt set (e.g. BUILD_FAIL), so a transient outage
                # masqueraded as a deterministic failure that --resume never
                # retried. Narrowed per the 2026-08-21 Codex review: ImportError
                # and ValueError out of call_llm are LOCAL configuration faults
                # (missing SDK, unknown provider), not outages - they keep the
                # generic ERROR verdict (also in --resume's retry set, matching
                # pre-OD-4 behavior); everything else is PROVIDER_ERROR.
                final_status = (
                    "ERROR" if isinstance(exc, (ImportError, ValueError))
                    else STOP_REASON_PROVIDER_ERROR
                )
                if _agentic_attempt_t0 is not None:
                    attempt_record["agentic_attempt_wall_seconds"] = round(
                        time.time() - _agentic_attempt_t0, 3)
                attempts.append(attempt_record)
                if agentic_budget is not None:
                    # The protocol's stop_reasons vocabulary is CLOSED
                    # (config/agentic_repair_protocol.json) and PROVIDER_ERROR
                    # is its only call-failure reason, so a local config fault
                    # keeps it here; the narrowed verdict lives in
                    # overall_status. Extending the vocabulary is a protocol
                    # artifact change - owner's call.
                    agentic_stop_reason = STOP_REASON_PROVIDER_ERROR
                break

            # Charge normalized usage against the trajectory budget. Fails
            # closed (AgenticProtocolError) if the provider exceeded the
            # attempt reservation — a protocol violation, never absorbed.
            if agentic_budget is not None:
                # Redacted response record (gate 4 finding 3, 2026-08-12):
                # length and hash, never content. Recorded BEFORE settlement
                # so a settlement violation still leaves a complete attempt
                # log (gate 6 finding 3, 2026-08-12).
                attempt_record["agentic_response_record_redacted"] = {
                    "chars": len(llm_result["response_text"]),
                    "sha256": hashlib.sha256(
                        llm_result["response_text"].encode("utf-8")).hexdigest(),
                }
                attempt_record["prompt_tokens"] = llm_result["prompt_tokens"]
                attempt_record["completion_tokens"] = llm_result["completion_tokens"]
                try:
                    agentic_budget.record_attempt_usage({
                        "prompt_tokens": llm_result["prompt_tokens"],
                        "completion_tokens": llm_result["completion_tokens"],
                    })
                except AgenticProtocolError as exc:
                    # A provider exceeding the reservation is a protocol
                    # violation: terminate the trajectory with an explicit
                    # stop reason and a complete redacted attempt record -
                    # never an unhandled exception (gate 6 finding 3).
                    error_message = f"agentic protocol violation: {exc}"
                    attempt_record["error"] = error_message
                    if _agentic_attempt_t0 is not None:
                        attempt_record["agentic_attempt_wall_seconds"] = round(
                            time.time() - _agentic_attempt_t0, 3)
                    attempts.append(attempt_record)
                    agentic_stop_reason = STOP_REASON_PROTOCOL_VIOLATION
                    break

            response_text = strip_think_tags(llm_result["response_text"])
            last_llm_response = response_text
            attempt_record["llm_response_time_seconds"] = llm_result["duration_seconds"]
            attempt_record["prompt_tokens"] = llm_result["prompt_tokens"]
            attempt_record["completion_tokens"] = llm_result["completion_tokens"]
            attempt_record["finish_reason"] = llm_result["finish_reason"]

            total_prompt_tokens += llm_result["prompt_tokens"]
            total_completion_tokens += llm_result["completion_tokens"]
            total_llm_time += llm_result["duration_seconds"]

            # -- Extract code blocks using anonymized filenames --
            # The LLM was told to produce generic filenames (translated_0.cpp, etc.)
            anon_extracted = extract_code_blocks(response_text, anon_target_filenames)
            if verbose:
                logger.info(
                    "Extracted %d/%d target files", len(anon_extracted), len(anon_target_filenames)
                )

            # De-anonymize: map generic filenames back to real filenames
            extracted = _deanonymize_extracted(anon_map, anon_extracted)

            # Merge into accumulator — newer extractions win for duplicate keys,
            # so re-emitted files get the latest version while previously-extracted
            # files from earlier attempts are preserved.
            accumulated_extracted.update(extracted)

            # -- Write ALL accumulated files to disk (not just current attempt) --
            # This ensures files from prior partial-extraction attempts survive
            # restore_files() calls that wipe the working directory.
            for fname, code in accumulated_extracted.items():
                fp = source_dir / fname
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(code, encoding="utf-8")

            # -- Guard: EXTRACTION_FAIL if no target files parsed from LLM response --
            # Building without extracted LLM code would compile the reference
            # implementation and produce a false-positive PASS. Fail fast instead.
            if not extracted:
                final_status = "EXTRACTION_FAIL"
                error_message = (
                    f"LLM response did not contain parseable code for any of the "
                    f"expected target files: {anon_target_filenames}"
                )
                attempt_record["extraction_fail"] = True
                if attempt_num < max_retries:
                    # Use anonymized filenames in feedback (LLM knows generic names)
                    extraction_feedback = (
                        "Your response did not include any of the expected output files. "
                        "Please translate each target file using this format:\n\n"
                        + "".join(
                            f"```c {f}\n// your translated code here\n```\n\n"
                            for f in anon_target_filenames
                        )
                    )
                    if agentic_budget is not None:
                        # Prepared, not yet delivered: promoted to
                        # error_feedback_sent only when the next attempt
                        # passes its budget gate (gate 2 finding 3,
                        # 2026-08-12).
                        attempt_record["error_feedback_prepared"] = (
                            extraction_feedback[:200]
                        )
                    else:
                        attempt_record["error_feedback_sent"] = extraction_feedback[:200]
                if _agentic_attempt_t0 is not None:
                    attempt_record["agentic_attempt_wall_seconds"] = round(
                        time.time() - _agentic_attempt_t0, 3)
                attempts.append(attempt_record)
                if attempt_num == max_retries:
                    break
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": extraction_feedback})
                # Do NOT restore_files here — accumulated_extracted files on disk
                # must survive across extraction retries so the LLM can produce
                # only the missing files on the next attempt.
                continue

            # -- Guard: EXTRACTION_FAIL on partial extraction or 0-byte extracted files --
            # Check against accumulated_extracted (not just this attempt's extracted)
            # so that files successfully extracted in prior attempts count toward
            # completeness.  This prevents the pipeline from regressing when a retry
            # produces only the missing files (by design).
            _missing_files = [f for f in target_filenames if f not in accumulated_extracted]
            _empty_files = [f for f, c in accumulated_extracted.items() if f in target_filenames and not c.strip()]
            _partial_fail_files = _missing_files + _empty_files
            # Map real filenames back to anonymized names for feedback to LLM
            _real_to_anon = {v: k for k, v in anon_map.items()}
            _anon_partial_fail = [_real_to_anon.get(f, f) for f in _partial_fail_files]
            if _partial_fail_files:
                final_status = "EXTRACTION_FAIL"
                error_message = (
                    f"Partial extraction on attempt {attempt_num}: "
                    f"missing {len(_missing_files)} file(s), "
                    f"{len(_empty_files)} empty file(s). "
                    f"Missing: {_missing_files}. Empty: {_empty_files}."
                )
                attempt_record["extraction_fail"] = True
                attempt_record["partial_extraction"] = True
                if attempt_num < max_retries:
                    # Use anonymized filenames in feedback (LLM knows generic names)
                    extraction_feedback = (
                        f"Your response was missing {len(_anon_partial_fail)} expected "
                        f"file(s). The other files from your previous response were saved "
                        f"successfully. Please provide ONLY the following missing/empty "
                        f"files:\n\n"
                        + "".join(
                            f"```c {f}\n// your translated code here\n```\n\n"
                            for f in _anon_partial_fail
                        )
                    )
                    if agentic_budget is not None:
                        # Prepared, not yet delivered: promoted to
                        # error_feedback_sent only when the next attempt
                        # passes its budget gate (gate 2 finding 3,
                        # 2026-08-12).
                        attempt_record["error_feedback_prepared"] = (
                            extraction_feedback[:200]
                        )
                    else:
                        attempt_record["error_feedback_sent"] = extraction_feedback[:200]
                if _agentic_attempt_t0 is not None:
                    attempt_record["agentic_attempt_wall_seconds"] = round(
                        time.time() - _agentic_attempt_t0, 3)
                attempts.append(attempt_record)
                if attempt_num == max_retries:
                    break
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": extraction_feedback})
                # Do NOT restore_files here — accumulated_extracted files on disk
                # must survive across extraction retries.  The next attempt will
                # write accumulated_extracted (which includes prior successes)
                # back to disk after merging any newly extracted files.
                continue

            # -- Build --
            build_result = build_spec(
                target_spec_resolved, project_root, verbose=verbose
            )
            final_build_result = build_result
            attempt_record["build_status"] = build_result.status.value

            if build_result.status != Status.PASS:
                attempt_record["build_error_snippet"] = _head_tail(
                    build_result.stderr or ""
                )
                final_status = "BUILD_FAIL"
                error_message = f"Build failed: {(build_result.stderr or '')[-200:].strip()}"
            else:
                # -- Run --
                # Cross-API: LLMs preserve source argc parsing,
                # so use source args for the translated binary.
                if source_api != target_api:
                    cross_run = _build_cross_api_run_spec(
                        target_spec_resolved, source_spec, pair_contract
                    )
                    if verbose:
                        logger.info(
                            "Cross-API (%s→%s): pair-contract translated_run args",
                            source_api, target_api,
                        )
                else:
                    cross_run = target_spec_resolved

                # Capture which args are being used for auditability
                _run_spec_used = cross_run
                _run_configs = (_run_spec_used.get("run") or {}).get("input_configurations", {})
                _correctness_args = (_run_configs.get("correctness") or {}).get("arguments", [])

                run_result = run_spec(
                    cross_run,
                    project_root,
                    verbose=verbose,
                    measure_cpu_time=use_cpu_timing,
                )
                final_run_result = run_result
                attempt_record["run_status"] = run_result.status.value

                if run_result.status != Status.PASS:
                    final_status = "RUN_FAIL"
                    error_message = f"Run failed (exit code {run_result.exit_code})"
                    attempt_record["run_stderr_snippet"] = (run_result.stderr or "")[-500:]
                    attempt_record["run_stdout_snippet"] = (run_result.stdout or "")[-500:]
                else:
                    # -- Direction-aware verification --
                    # For cross-API translations, the LLM's output matches the SOURCE
                    # implementation's stdout patterns, not the TARGET's.
                    if source_api != target_api:
                        verify_spec = _build_cross_api_verify_spec(
                            target_spec, source_spec, pair_contract
                        )
                        if verbose:
                            logger.info(
                                "Cross-API verify (%s→%s): pair-contract target oracle",
                                source_api, target_api,
                            )
                    else:
                        verify_spec = target_spec

                    verify_result = verify_run(
                        verify_spec,
                        run_result,
                        working_dir=target_spec_resolved["_resolved"]["working_dir"],
                    )
                    final_verify_result = verify_result
                    attempt_record["verify_status"] = verify_result.status.value

                    # -- Extract metrics (stdout-parsed timing for HeCBench specs) --
                    final_metrics = extract_metrics(target_spec, run_result)

                    if verify_result.status == Status.PASS:
                        # Post-verification sanity check: reject if stdout
                        # contains error indicators that the pattern matcher
                        # missed (e.g., OpenCL clBuildProgram failures where
                        # the host continues and prints expected output).
                        stdout_text = run_result.stdout or ""
                        reject_reason = _check_stdout_error_indicators(stdout_text)
                        if reject_reason:
                            final_status = "VERIFY_FAIL"
                            error_message = f"False positive rejected: {reject_reason}"
                            if verbose:
                                logger.warning("PASS downgraded to VERIFY_FAIL: %s", reject_reason)
                        else:
                            final_status = "PASS"
                            error_message = None  # clear any error from a previous attempt
                    elif verify_result.status == Status.ERROR:
                        # Attribution fix (Task 2, logged in Task 1's review):
                        # a harness-side verification ERROR (malformed or
                        # unavailable declared check) is an infrastructure
                        # error, not evidence of a wrong translation. It must
                        # not be counted as VERIFY_FAIL in denominators.
                        final_status = "ERROR"
                        error_message = f"Verification error: {verify_result.details}"
                    else:
                        final_status = "VERIFY_FAIL"
                        error_message = f"Verify failed: {verify_result.details}"

            # Build retry feedback and record it BEFORE appending so the
            # attempt_record is fully populated (no post-append mutation).
            if final_status != "PASS" and attempt_num < max_retries:
                feedback = _build_retry_message(
                    final_build_result, final_run_result, final_verify_result,
                    # Bounded protocol: verification failures expose status
                    # only — verifier details carry oracle contents.
                    verify_status_only=agentic_budget is not None,
                )

                # Enrich BUILD_FAIL feedback with linker error analysis
                if final_status == "BUILD_FAIL" and source_files_for_repair:
                    build_stderr = (final_build_result.stderr or "") if final_build_result else ""
                    linker_hints = analyze_build_failure(build_stderr, source_files_for_repair)
                    if linker_hints:
                        feedback += linker_hints
                        logger.info(
                            "Linker error analysis added %d hint(s) to retry feedback",
                            linker_hints.count("- "),
                        )

                if agentic_budget is not None:
                    # Prepared, not yet delivered: promoted to
                    # error_feedback_sent only when the next attempt passes
                    # its budget gate (gate 2 finding 3, 2026-08-12).
                    attempt_record["error_feedback_prepared"] = feedback[:200]
                else:
                    attempt_record["error_feedback_sent"] = feedback[:200]  # abbreviated in record

            if _agentic_attempt_t0 is not None:
                attempt_record["agentic_attempt_wall_seconds"] = round(
                    time.time() - _agentic_attempt_t0, 3)
            attempts.append(attempt_record)

            # If passed or last attempt, stop
            if final_status == "PASS" or attempt_num == max_retries:
                break

            # Extend conversation: assistant response + user error feedback
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": feedback})

            # Restore files to a clean state before the next attempt, so each
            # attempt starts from the original reference files (not stale LLM
            # output from the previous attempt).
            restore_files(backup_info)
            backup_info = backup_files(target_file_paths)

    finally:
        restore_files(backup_info)
        _unstage_support_headers(staged_headers)

    # Protocol stop-reason bookkeeping for trajectories that ended without an
    # explicit refusal or provider error.
    if agentic_budget is not None and agentic_stop_reason is None:
        if final_status == "PASS":
            agentic_stop_reason = STOP_REASON_PASS
        elif agentic_budget.attempts_started >= agentic_budget.max_total_attempts:
            agentic_stop_reason = STOP_REASON_ATTEMPT_LIMIT
        else:
            # The loop ended because the caller's max_retries is below the
            # protocol's attempt cap (gate 2 finding 2, 2026-08-12): the
            # effective attempt limit was exhausted and the stop reason must
            # say so explicitly, never stay null.
            agentic_stop_reason = STOP_REASON_CALLER_ATTEMPT_LIMIT

    # -- Speedup ratio --
    # Numerator: baseline wall_time from the target spec (unchanged).
    # Denominator priority: kernel_time > cpu_time > wall_time.
    # Note: baseline is always wall_time (kernel/cpu baselines not stored in
    # specs yet), so mixing timing methods is imperfect but acceptable for now.
    translated_wall_time: float | None = None
    translated_cpu_time: float | None = None
    translated_kernel_time: float | None = None
    speedup_ratio: float | None = None
    timing_method: str | None = None

    if final_run_result is not None and final_run_result.status == Status.PASS:
        translated_wall_time = round(final_run_result.duration_seconds, 4)
        if final_run_result.cpu_time_seconds is not None:
            translated_cpu_time = round(final_run_result.cpu_time_seconds, 6)
        if final_run_result.kernel_time_seconds is not None:
            translated_kernel_time = round(final_run_result.kernel_time_seconds, 6)

        # Choose denominator with priority: kernel > cpu > wall
        if translated_kernel_time is not None:
            denominator = translated_kernel_time
            timing_method = "kernel_time"
        elif translated_cpu_time is not None:
            denominator = translated_cpu_time
            timing_method = "cpu_time"
        else:
            denominator = translated_wall_time
            timing_method = "wall_time"

        if baseline_wall_time and denominator:
            speedup_ratio = round(baseline_wall_time / denominator, 4)

    # -- Build result JSON --
    # last_llm_response was saved inside the loop before restore_files() ran
    # Extract using anonymized filenames (what the LLM produced), then de-anonymize
    anon_extracted_last = extract_code_blocks(last_llm_response, anon_target_filenames)
    extracted_last = _deanonymize_extracted(anon_map, anon_extracted_last)
    # Store FULL translated files — truncating to 200 chars made retroactive
    # re-verification impossible (discovered in S-VERIFY session, 2026-03-27).
    translated_files_full = {
        f: extracted_last[f] for f in target_filenames if f in extracted_last
    }

    metrics_dict: dict[str, Any] = {}
    if final_metrics:
        metrics_dict = {m.name: {"value": m.value, "unit": m.unit} for m in final_metrics}

    result: dict[str, Any] = {
        "source_spec": source_id,
        "target_spec": target_id,
        "kernel": kernel_name,
        "model": model,
        "augment_level": augment_level,
        "temperature": temperature,
        "sample_id": sample_id,
        # Plan 02-03 (D-09): new top-level fields for thinking-state filtering
        # and single-file pass@k reconstruction.
        "thinking_enabled": thinking_enabled,
        "num_samples": num_samples,
        # Plan 02-10 Step 2 (C4): sampling reproducibility metadata.
        # `seed` is SHA-256-derived from (source, target, sample_id) —
        # deterministic across processes. `top_p=1.0` = full-distribution
        # sampling at matched temperature (NOT argmax). Both fields absent
        # on historical result JSONs written before this schema bump; all
        # downstream consumers tolerate the absence.
        "seed": llm_sampling_seed,
        "top_p": llm_top_p,
        "translation_mode": translation_mode,
        "translation_type": "kernel_only" if _is_kernel_only_translation(target_spec) else "full_program",
        # Task 2 (E-02): all cross-API verification and run arguments come
        # from the declared pair contract; the pre-Task-2 union/kernel-only
        # mode values survive only in historical result JSONs.
        "verification_mode": ("same_api_target_pattern" if source_api == target_api
                              else "pair_contract_target_oracle"),
        "run_args_mode": ("same_api_target_args" if source_api == target_api
                          else "pair_contract_args"),
        # Registry provenance (2026-08-21): which contract registry this run
        # used, so v1 (frozen, sealed-artifact) and v2+ (current) records can
        # never silently mix in analysis. None for same-API tasks (contracts
        # are unused there even when a registry was supplied), --dry-run, and
        # in-memory registries without a file path.
        "pair_contract_registry": (
            {
                "path": str(pair_registry.path),
                "contract_version": pair_registry.contract_version,
                "sha256": pair_registry.sha256,
            }
            if (source_api != target_api
                and pair_registry is not None
                and pair_registry.path is not None)
            else None
        ),
        "run_arguments_used": _correctness_args if _correctness_args else None,
        "timestamp": timestamp,
        # LLM usage (totals across all attempts)
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "llm_response_time_seconds": round(total_llm_time, 3),
        # Build
        "build_status": (
            final_build_result.status.value if final_build_result else None
        ),
        "build_time_seconds": (
            round(final_build_result.duration_seconds, 3) if final_build_result else None
        ),
        "build_error_snippet": (
            _head_tail(final_build_result.stderr or "") if final_build_result and final_build_result.status != Status.PASS else None
        ),
        # Run
        "run_status": (
            final_run_result.status.value if final_run_result else None
        ),
        "run_time_seconds": (
            round(final_run_result.duration_seconds, 3) if final_run_result else None
        ),
        "run_exit_code": (
            final_run_result.exit_code if final_run_result else None
        ),
        "run_stderr_snippet": (
            (final_run_result.stderr or "")[-500:]
            if final_run_result and final_run_result.status != Status.PASS else None
        ),
        # Store stdout for ALL results (not just failures) — needed for
        # retroactive re-verification with stdout_pattern strategies.
        "run_stdout_snippet": (
            (final_run_result.stdout or "")[-500:]
            if final_run_result else None
        ),
        # Verify
        "verify_status": (
            final_verify_result.status.value if final_verify_result else None
        ),
        "verify_strategy": (
            final_verify_result.strategy_used if final_verify_result else None
        ),
        # Overall
        "overall_status": final_status,
        "error_message": error_message,
        # Metrics / timing
        "metrics": metrics_dict,
        "baseline_wall_time_seconds": baseline_wall_time,
        "translated_wall_time_seconds": translated_wall_time,
        "translated_cpu_time_seconds": translated_cpu_time,
        "translated_kernel_time_seconds": translated_kernel_time,
        "timing_method": timing_method,
        "speedup_ratio": speedup_ratio,
        # Code
        "translated_files": translated_files_full,
        "target_files_expected": target_filenames,
        "target_files_extracted": [
            f for f in target_filenames if f in translated_files_full
        ],
        # Retry tracking
        "max_retries": max_retries,
        "total_attempts": len(attempts),
        "attempts": attempts,
    }

    # Bounded agentic-repair protocol block (only when the protocol was
    # active — live eval result JSONs are unchanged).
    if agentic_budget is not None:
        result["agentic_protocol"] = {
            "protocol_name": agentic_protocol.get("protocol_name"),
            "protocol_version": agentic_protocol.get("protocol_version"),
            "stop_reason": agentic_stop_reason,
            "attempts_started": agentic_budget.attempts_started,
            "max_total_attempts": agentic_budget.max_total_attempts,
            "tokens_used_normalized": agentic_budget.normalized_used,
            "trajectory_budget_total": agentic_budget.total_budget,
            "tokens_remaining": agentic_budget.remaining,
            "refused_attempt": agentic_refusal,
        }

    # -- Save result to disk --
    # When called from run_eval_batch.py, save_to_disk=False to avoid dual-write
    # (the batch runner owns file I/O with its own resume/overwrite logic).
    if save_to_disk:
        out_dir = project_root / "results" / "evaluation" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        level_tag = f"-L{augment_level}" if augment_level > 0 else ""
        sample_tag = f"-s{sample_id}" if sample_id > 0 else ""
        out_file = out_dir / f"{source_id}-to-{target_id}{level_tag}{sample_tag}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if verbose:
            logger.info("Result saved: %s", out_file)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_models() -> None:
    """Print the model registry as a table."""
    print(f"{'Model ID':<40} {'Provider':<12} {'Notes'}")
    print("-" * 80)
    for model_id, info in MODEL_REGISTRY.items():
        print(f"{model_id:<40} {info['provider']:<12} {info['notes']}")
    print()
    print("Any model ID is accepted — the registry above is for reference only.")
    print("Routing: claude-* → Anthropic, gpt-*/o1-*/o3-*/o4-* → OpenAI, "
          "azure-* → Azure, groq-* → Groq, gemini-* → Google AI")


def _print_result(result: dict[str, Any], as_json: bool, verbose: bool) -> None:
    """Print final result to stdout."""
    if as_json:
        print(json.dumps(result, indent=2))
        return

    status = result.get("overall_status", "ERROR")
    src = result.get("source_spec", "?")
    tgt = result.get("target_spec", "?")
    model = result.get("model", "?")

    marker = "✓" if status == "PASS" else "✗"
    print(f"\n{marker} {src} → {tgt}  [{model}]  status={status}")
    print(
        f"  build={result.get('build_status')}  "
        f"run={result.get('run_status')}  "
        f"verify={result.get('verify_status')}"
    )
    if result.get("speedup_ratio") is not None:
        print(f"  speedup={result['speedup_ratio']:.3f}x  "
              f"(baseline={result.get('baseline_wall_time_seconds')}s  "
              f"translated={result.get('translated_wall_time_seconds')}s)")
    llm_time = result.get("llm_response_time_seconds")
    llm_time_str = f"{llm_time}s" if llm_time is not None else "N/A"
    print(
        f"  tokens={result.get('prompt_tokens')}+{result.get('completion_tokens')}  "
        f"llm_time={llm_time_str}  "
        f"attempts={result.get('total_attempts')}"
    )
    if result.get("error_message"):
        print(f"  error: {result['error_message']}")
    files_expected = result.get("target_files_expected", [])
    files_extracted = result.get("target_files_extracted", [])
    if verbose or files_expected != files_extracted:
        print(f"  files expected={files_expected}  extracted={files_extracted}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based parallel code translation evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to source spec JSON (e.g. specs/rodinia-bfs-cuda.json)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Path to target spec JSON (e.g. specs/rodinia-bfs-omp.json)",
    )
    parser.add_argument(
        "--model",
        help="Model ID string (e.g. claude-sonnet-4-20250514, gpt-4o)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=False,
        help="Absolute path to parbench root directory (required for harness)",
    )
    parser.add_argument(
        "--augment-level",
        type=int,
        default=0,
        help="Augmentation level for source code (0=none, default: 0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Max LLM attempts with error feedback (1=zero-shot, default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print result as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt without calling LLM",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print known model registry and exit",
    )
    parser.add_argument(
        "--use-cpu-timing",
        action="store_true",
        default=False,
        help=(
            "Measure CPU time (user+system) via /usr/bin/time -v when running "
            "translated code.  Uses cpu_time_seconds in speedup calculation "
            "instead of wall_time_seconds.  Linux only (GNU time required)."
        ),
    )
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        default=False,
        help=(
            "Reserved for future kernel-level profiling (e.g. nsys/nvprof). "
            "Not yet implemented; accepted but has no effect."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0=greedy, 0.5+ for pass@k multi-sampling). Default: 0.0",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=0,
        help="Sample index for pass@k runs (0=default, >0 adds -s{N} to filename). Default: 0",
    )
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="on",
        help=(
            "Enable LLM reasoning/thinking mode (default: on). "
            "'on' → reasoning_effort=medium for Azure reasoning models, "
            "enable_thinking=True for Qwen. 'off' → omit reasoning_effort, "
            "enable_thinking=False. No-op for models where "
            "MODEL_REGISTRY[model]['supports_thinking'] is False."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help=(
            "Number of samples per task (k in pass@k). Single-task CLI default is 1; "
            "run_eval_batch.py overrides. Recorded in the result JSON's num_samples field. "
            "Default: 1"
        ),
    )
    parser.add_argument(
        "--pair-contracts",
        type=Path,
        default=None,
        help=(
            "Path to the pair-contract registry JSON (Task 2, E-02). Required "
            "for any cross-API translation unless --dry-run; a missing or "
            "unresolved pair fails closed before the model is invoked."
        ),
    )

    args = parser.parse_args()

    if args.list_models:
        _print_models()
        return

    # Validate required args for non-list-models mode
    missing = []
    if not args.source:
        missing.append("--source")
    if not args.target:
        missing.append("--target")
    if not args.model:
        missing.append("--model")
    if not args.project_root:
        missing.append("--project-root")
    if missing:
        parser.error(f"Required arguments missing: {', '.join(missing)}")

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    project_root: Path = args.project_root.resolve()

    pair_registry = (
        load_pair_contracts(args.pair_contracts)
        if args.pair_contracts is not None
        else None
    )

    result = evaluate_translation(
        source_path=args.source,
        target_path=args.target,
        model=args.model,
        project_root=project_root,
        augment_level=args.augment_level,
        max_retries=args.max_retries,
        verbose=args.verbose,
        dry_run=args.dry_run,
        use_cpu_timing=args.use_cpu_timing,
        temperature=args.temperature,
        sample_id=args.sample_id,
        thinking=args.thinking,
        num_samples=args.num_samples,
        pair_registry=pair_registry,
    )

    _print_result(result, as_json=args.as_json, verbose=args.verbose)


if __name__ == "__main__":
    main()
