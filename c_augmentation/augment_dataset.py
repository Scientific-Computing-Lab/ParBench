#!/usr/bin/env python3
"""
C/C++ code augmentation framework using libclang-backed AST analysis.

Supports CUDA, OpenCL, OpenMP, and SYCL parallel code.
All transforms discover candidates from libclang cursors and rewrite exact source
spans from the original code instead of scanning with regexes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    import clang.cindex as ci
    from clang.cindex import CursorKind, TokenKind, TypeKind
except ImportError as exc:
    raise ImportError(
        "libclang Python bindings not found. Install with: pip install libclang"
    ) from exc


PARSE_ARGS = [
    "-xc++",
    "-std=c++17",
    "-D__global__=",
    "-D__device__=",
    "-D__host__=",
    "-D__shared__=",
    "-D__constant__=",
    "-D__kernel__=",
    "-D__syncthreads()=",
    "-D__launch_bounds__(x)=",
    "-D__restrict__=",
    "-isystem", "/opt/sycl/lib/clang/23/include",
    "-isystem", "/usr/include/c++/11",
    "-isystem", "/usr/include/x86_64-linux-gnu/c++/11",
    "-isystem", "/usr/include/c++/11/backward",
    "-isystem", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
    "-isystem", "/usr/local/include",
    "-isystem", "/usr/include/x86_64-linux-gnu",
    "-isystem", "/usr/include",
]
PARSE_OPTIONS = (
    ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    | getattr(ci.TranslationUnit, "PARSE_INCOMPLETE", 0)
)
BASELINE_ERROR_THRESHOLD = 100
UNWRAP_KINDS = {
    CursorKind.UNEXPOSED_EXPR,
    getattr(CursorKind, "PAREN_EXPR", CursorKind.UNEXPOSED_EXPR),
}
COMPARISON_OPERATORS = {"<", ">", "<=", ">=", "==", "!="}
COMPOUND_OPERATORS = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}
INTEGER_TYPE_KINDS = {
    TypeKind.BOOL,
    TypeKind.CHAR_U,
    TypeKind.UCHAR,
    TypeKind.CHAR16,
    TypeKind.CHAR32,
    TypeKind.USHORT,
    TypeKind.UINT,
    TypeKind.ULONG,
    TypeKind.ULONGLONG,
    TypeKind.UINT128,
    TypeKind.CHAR_S,
    TypeKind.SCHAR,
    TypeKind.WCHAR,
    TypeKind.SHORT,
    TypeKind.INT,
    TypeKind.LONG,
    TypeKind.LONGLONG,
    TypeKind.INT128,
}
FLOAT_TYPE_KINDS = {
    TypeKind.FLOAT,
    TypeKind.DOUBLE,
    TypeKind.LONGDOUBLE,
    getattr(TypeKind, "HALF", TypeKind.FLOAT),
    getattr(TypeKind, "FLOAT128", TypeKind.DOUBLE),
}
ARRAY_LIKE_TYPE_KINDS = {
    TypeKind.POINTER,
    TypeKind.CONSTANTARRAY,
    TypeKind.INCOMPLETEARRAY,
    TypeKind.VARIABLEARRAY,
    TypeKind.DEPENDENTSIZEDARRAY,
}

# Level-to-fraction mapping for consistent scaling across all transforms
LEVEL_FRACTIONS = {1: 0.0, 2: 0.33, 3: 0.66, 4: 1.0}


@dataclass
class TransformResult:
    """Result of attempting a transform."""

    applied: bool
    code: str
    description: str = ""


class TextEdit(BaseModel):
    start_offset: int
    end_offset: int
    replacement: str


class RewriteCandidate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    description: str
    edits: list[TextEdit] = Field(default_factory=list)


class Transform(ABC):
    """Base class for code transforms."""

    name: str = "BaseTransform"

    def __init__(self, level: int = 4):
        self.level = level

    @abstractmethod
    def is_applicable(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> bool:
        """Check if this transform can be applied to the code."""

    @abstractmethod
    def apply(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> TransformResult:
        """Apply the transform to the code."""

    def _select_fraction(self, candidates: list) -> list:
        """
        Select a fraction of candidates based on self.level.
        Level 1 → 1 candidate (minimum), Level 4 → all candidates.
        The list is shuffled before selection so that level 2/3 pick
        a random subset rather than always the first N items.
        """
        if not candidates:
            return []
        shuffled = list(candidates)
        random.shuffle(shuffled)
        if self.level == 1:
            return shuffled[:1]
        fraction = LEVEL_FRACTIONS[self.level]
        n = max(1, int(len(shuffled) * fraction))
        return shuffled[:n]


def _parse_translation_unit(
    code: str,
    index: ci.Index,
    filename: str = "tmp.cpp",
) -> ci.TranslationUnit:
    return index.parse(
        path=filename,
        args=PARSE_ARGS,
        unsaved_files=[(filename, code)],
        options=PARSE_OPTIONS,
    )


def _is_low_quality_tu(tu: ci.TranslationUnit) -> bool:
    """Return True if the TU has fatal errors or too many diagnostics."""
    error_count = 0
    for d in tu.diagnostics:
        if d.severity >= ci.Diagnostic.Fatal:
            return True
        if d.severity >= ci.Diagnostic.Error:
            error_count += 1
    return error_count > 50


def _iter_cursors(cursor: ci.Cursor) -> Iterable[ci.Cursor]:
    yield cursor
    for child in cursor.get_children():
        yield from _iter_cursors(child)


def _cursor_in_main_file(cursor: ci.Cursor, filename: str) -> bool:
    try:
        location = cursor.location
        return bool(location.file) and location.file.name == filename
    except Exception:
        return False


def _is_macro_expansion(cursor: ci.Cursor) -> bool:
    """Check if the cursor is part of a macro expansion."""
    try:
        loc = cursor.location
        if not loc or not loc.file:
            return False
        # get_instantiation returns (file, line, column, offset) of the expansion site.
        # If it differs from the physical location, it is likely inside a macro.
        # Also check if any parent is a MACRO_EXPANSION/INSTANTIATION.
        s_file, s_line, s_col, s_off = loc._get_instantiation()
        if s_line != loc.line or s_col != loc.column or s_off != loc.offset:
            return True

        # Fallback: check parents for macro instantiation
        curr = cursor
        while curr:
            if curr.kind == ci.CursorKind.MACRO_INSTANTIATION:
                return True
            # Stop if we hit a function or translation unit
            if curr.kind in (ci.CursorKind.FUNCTION_DECL, ci.CursorKind.TRANSLATION_UNIT):
                break
            curr = curr.semantic_parent
        return False
    except Exception:
        return False


def _cursor_offsets(cursor: ci.Cursor) -> tuple[int, int]:
    return cursor.extent.start.offset, cursor.extent.end.offset


def _source_text(code: str, cursor: ci.Cursor) -> str:
    start, end = _cursor_offsets(cursor)
    return code.encode("utf-8")[start:end].decode("utf-8", errors="replace")


def _unwrap_expr(cursor: ci.Cursor) -> ci.Cursor:
    current = cursor
    while current.kind in UNWRAP_KINDS:
        children = list(current.get_children())
        if len(children) != 1:
            break
        current = children[0]
    return current


def _children(cursor: ci.Cursor) -> list[ci.Cursor]:
    return [_unwrap_expr(child) for child in cursor.get_children()]


def _is_preprocessor_line(code: str, offset: int) -> bool:
    """
    Return True if *offset* points to a token that lives on a preprocessor line
    (e.g. #define / #pragma). Token offsets are byte-based; code in this corpus
    is ASCII/C-family source so byte and char positions align.
    """
    if offset < 0:
        return False
    line_start = code.rfind("\n", 0, offset)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = code.find("\n", offset)
    if line_end == -1:
        line_end = len(code)
    line = code[line_start:line_end].lstrip()
    return line.startswith("#")


OMP_PRAGMA_KEYWORDS = {
    "omp",
    "parallel",
    "for",
    "simd",
    "target",
    "data",
    "update",
    "teams",
    "distribute",
    "task",
    "taskloop",
    "single",
    "sections",
    "section",
    "master",
    "critical",
    "atomic",
    "ordered",
    "schedule",
    "collapse",
    "shared",
    "private",
    "firstprivate",
    "lastprivate",
    "reduction",
    "default",
    "map",
    "from",
    "to",
    "in",
    "out",
    "inout",
    "alloc",
    "release",
    "delete",
    "if",
    "nowait",
    "depend",
    "device",
    "num_teams",
    "thread_limit",
    "num_threads",
    "is_device_ptr",
    "use_device_ptr",
    "copyin",
    "copyprivate",
    "linear",
    "aligned",
    "safelen",
    "simdlen",
    "uniform",
    "nontemporal",
    "proc_bind",
    "grainsize",
    "num_tasks",
    "mergeable",
    "untied",
    "final",
    "priority",
    "detach",
    "hint",
}


def _skip_preprocessor_token_rename(code: str, offset: int, spelling: str) -> bool:
    """Skip renaming tokens that are directive keywords on preprocessor lines."""
    if not _is_preprocessor_line(code, offset):
        return False
    line_start = code.rfind("\n", 0, offset)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = code.find("\n", offset)
    if line_end == -1:
        line_end = len(code)
    line = code[line_start:line_end].lstrip()
    if line.startswith("#pragma omp"):
        return spelling in OMP_PRAGMA_KEYWORDS
    return True


def _is_omp_pragma_line(code: str, offset: int) -> bool:
    if not _is_preprocessor_line(code, offset):
        return False
    line_start = code.rfind("\n", 0, offset)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = code.find("\n", offset)
    if line_end == -1:
        line_end = len(code)
    line = code[line_start:line_end].lstrip()
    return line.startswith("#pragma omp")


def _strip_outer_parens(text: str) -> str:
    stripped = text.strip()
    while stripped.startswith("(") and stripped.endswith(")"):
        depth = 0
        balanced = True
        for idx, ch in enumerate(stripped):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and idx != len(stripped) - 1:
                balanced = False
                break
        if not balanced or depth != 0:
            break
        stripped = stripped[1:-1].strip()
    return stripped


def _normalized_expr_text(text: str) -> str:
    return " ".join(_strip_outer_parens(text).split())


def _binary_operator_text(code: str, cursor: ci.Cursor) -> Optional[str]:
    children = _children(cursor)
    if len(children) != 2:
        return None
    left_end = children[0].extent.end.offset
    right_start = children[1].extent.start.offset
    between = code.encode("utf-8")[left_end:right_start].decode("utf-8", errors="replace")
    for operator in sorted(COMPARISON_OPERATORS | set(COMPOUND_OPERATORS) | set(COMPOUND_OPERATORS.values()) | {"="}, key=len, reverse=True):
        if operator in between:
            return operator
    return None


def _unary_prefix_text(code: str, cursor: ci.Cursor) -> str:
    children = _children(cursor)
    if len(children) != 1:
        return ""
    start, _ = _cursor_offsets(cursor)
    child_start = children[0].extent.start.offset
    return code[start:child_start]


def _identifier_range(cursor: ci.Cursor, spelling: Optional[str] = None) -> Optional[tuple[int, int]]:
    target = spelling or cursor.spelling
    for token in cursor.get_tokens():
        if token.kind != TokenKind.IDENTIFIER:
            continue
        if token.spelling != target:
            continue
        return token.extent.start.offset, token.extent.end.offset
    return None


def _apply_candidate(code: str, candidate: RewriteCandidate) -> str:
    # libclang reports byte offsets (UTF-8). Operate on the encoded bytes so that
    # multi-byte Unicode characters (e.g. em-dashes in comments) do not cause a
    # mismatch between Python str character indices and libclang byte offsets.
    rewritten_bytes = code.encode("utf-8")
    seen_ranges: set[tuple[int, int]] = set()
    for edit in sorted(candidate.edits, key=lambda item: item.start_offset, reverse=True):
        key = (edit.start_offset, edit.end_offset)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        replacement_bytes = edit.replacement.encode("utf-8")
        rewritten_bytes = (
            rewritten_bytes[: edit.start_offset]
            + replacement_bytes
            + rewritten_bytes[edit.end_offset :]
        )
    return rewritten_bytes.decode("utf-8")


def _fatal_diagnostics(tu: ci.TranslationUnit) -> int:
    return sum(
        1
        for diag in tu.diagnostics
        if diag.severity >= ci.Diagnostic.Error
    )


def _validate_rewrite(
    candidate: RewriteCandidate,
    code: str,
    index: ci.Index,
    filename: str = "tmp.cpp",
) -> bool:
    """
    Validate that applying candidate does not introduce new errors.
    Only parses twice (baseline + rewritten), not once per candidate.
    """
    rewritten = _apply_candidate(code, candidate)
    try:
        baseline = _parse_translation_unit(code, index, filename)
        updated = _parse_translation_unit(rewritten, index, filename)
    except Exception:
        return False
    return _fatal_diagnostics(updated) <= _fatal_diagnostics(baseline)


def _edits_overlap(c1: RewriteCandidate, c2: RewriteCandidate) -> bool:
    for e1 in c1.edits:
        for e2 in c2.edits:
            if not (e1.end_offset <= e2.start_offset or e2.end_offset <= e1.start_offset):
                return True
    return False


class AstTransform(Transform):
    """Base class for transforms that produce AST-backed rewrite candidates."""

    def is_applicable(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> bool:
        return bool(self._find_candidates(code, index, filename))

    def apply(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> TransformResult:
        tu = _parse_translation_unit(code, index, filename)
        if _fatal_diagnostics(tu) > BASELINE_ERROR_THRESHOLD:
            return TransformResult(False, code, f"{self.name}: too many baseline errors")

        all_candidates = self._find_candidates(code, index, filename)
        if not all_candidates:
            return TransformResult(False, code, f"{self.name}: 0 candidates")

        # FIX: Select fraction first, then validate only the selected set.
        # This avoids O(n) TU parses for large candidate sets at level 4.
        selected = self._select_fraction(all_candidates)

        # Remove overlapping candidates (keep first encountered after shuffle)
        non_overlapping: list[RewriteCandidate] = []
        for c in selected:
            if not any(_edits_overlap(c, existing) for existing in non_overlapping):
                non_overlapping.append(c)

        if not non_overlapping:
            return TransformResult(False, code, f"{self.name}: 0 non-overlapping candidates")

        # Validate the merged candidate once (not per-candidate)
        all_edits = [edit for c in non_overlapping for edit in c.edits]
        merged = RewriteCandidate(
            description=f"{self.name}: {len(non_overlapping)} candidates applied",
            edits=all_edits,
        )
        if not _validate_rewrite(merged, code, index, filename):
            # Merged set failed — fall back to validating candidates one by one
            # and building the largest valid non-overlapping subset.
            non_overlapping = self._greedy_valid_subset(non_overlapping, code, index, filename)
            if not non_overlapping:
                return TransformResult(False, code, f"{self.name}: merged candidate failed validation")
            all_edits = [edit for c in non_overlapping for edit in c.edits]
            merged = RewriteCandidate(
                description=f"{self.name}: {len(non_overlapping)} candidates applied (greedy)",
                edits=all_edits,
            )

        rewritten = _apply_candidate(code, merged)
        return TransformResult(True, rewritten, merged.description)

    def _greedy_valid_subset(
        self,
        candidates: list[RewriteCandidate],
        code: str,
        index: ci.Index,
        filename: str = "tmp.cpp",
    ) -> list[RewriteCandidate]:
        """
        Build the largest non-overlapping subset of candidates that passes
        validation when applied together. Tries adding one candidate at a time.
        """
        accepted: list[RewriteCandidate] = []
        for c in candidates:
            if any(_edits_overlap(c, existing) for existing in accepted):
                continue
            trial_edits = [e for a in accepted for e in a.edits] + list(c.edits)
            trial = RewriteCandidate(description="trial", edits=trial_edits)
            if _validate_rewrite(trial, code, index, filename):
                accepted.append(c)
        return accepted

    @abstractmethod
    def _find_candidates(
        self,
        code: str,
        index: ci.Index,
        filename: str = "tmp.cpp",
    ) -> list[RewriteCandidate]:
        """Collect AST-backed rewrite candidates."""


class ArithmeticTransform(AstTransform):
    """
    Converts augmented assignments to expanded form and vice versa.
    x += 1  <->  x = x + 1
    """

    name = "ArithmeticTransform"

    def _is_simple_scalar_declref(self, cursor: ci.Cursor) -> bool:
        cursor = _unwrap_expr(cursor)
        if cursor.kind != CursorKind.DECL_REF_EXPR:
            return False
        tkind = cursor.type.kind
        return tkind in INTEGER_TYPE_KINDS or tkind in FLOAT_TYPE_KINDS

    def _find_candidates(
        self,
        code: str,
        index: ci.Index,
        filename: str = "tmp.cpp",
    ) -> list[RewriteCandidate]:
        tu = _parse_translation_unit(code, index, filename)
        filename = tu.spelling
        candidates: list[RewriteCandidate] = []
        reverse_ops = {value: key for key, value in COMPOUND_OPERATORS.items()}
        compound_kind = getattr(CursorKind, "COMPOUND_ASSIGNMENT_OPERATOR", None)

        for_incr_ranges: list[tuple[int, int]] = []
        for fc in _iter_cursors(tu.cursor):
            if fc.kind == CursorKind.FOR_STMT:
                kids = list(fc.get_children())
                body_idx = next(
                    (i for i, k in enumerate(kids) if k.kind == CursorKind.COMPOUND_STMT), None
                )
                if body_idx is not None and body_idx > 0:
                    incr = kids[body_idx - 1]
                    for_incr_ranges.append((incr.extent.start.offset, incr.extent.end.offset))

        for cursor in _iter_cursors(tu.cursor):
            if not _cursor_in_main_file(cursor, filename) or _is_macro_expansion(cursor):
                continue

            if compound_kind is not None and cursor.kind == compound_kind:
                c_start = cursor.extent.start.offset
                if any(s <= c_start <= e for s, e in for_incr_ranges):
                    continue
                children = _children(cursor)
                if len(children) != 2:
                    continue
                lhs, rhs = children
                if not self._is_simple_scalar_declref(lhs):
                    continue
                operator = _binary_operator_text(code, cursor)
                if operator not in COMPOUND_OPERATORS:
                    continue
                lhs_text = _source_text(code, lhs).strip()
                rhs_text = _source_text(code, rhs).strip()
                start, end = _cursor_offsets(cursor)
                candidates.append(
                    RewriteCandidate(
                        description=f"{self.name}: expand compound",
                        edits=[
                            TextEdit(
                                start_offset=start,
                                end_offset=end,
                                replacement=f"{lhs_text} = {lhs_text} {COMPOUND_OPERATORS[operator]} ({rhs_text})",
                            )
                        ],
                    )
                )
                continue

            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue

            operator = _binary_operator_text(code, cursor)
            if operator != "=":
                continue

            children = _children(cursor)
            if len(children) != 2:
                continue
            lhs, rhs = children
            if not self._is_simple_scalar_declref(lhs):
                continue
            rhs = _unwrap_expr(rhs)
            if rhs.kind != CursorKind.BINARY_OPERATOR:
                continue
            rhs_operator = _binary_operator_text(code, rhs)
            if rhs_operator not in reverse_ops:
                continue

            rhs_children = _children(rhs)
            if len(rhs_children) != 2:
                continue
            rhs_left, rhs_right = rhs_children
            if not self._is_simple_scalar_declref(rhs_left):
                continue
            if lhs.type.kind not in INTEGER_TYPE_KINDS:
                continue
            lhs_text = _source_text(code, lhs).strip()
            rhs_left_text = _source_text(code, rhs_left).strip()
            rhs_right_text = _source_text(code, rhs_right).strip()
            if _normalized_expr_text(lhs_text) != _normalized_expr_text(rhs_left_text):
                continue

            start, end = _cursor_offsets(cursor)
            candidates.append(
                RewriteCandidate(
                    description=f"{self.name}: fold to compound",
                    edits=[
                        TextEdit(
                            start_offset=start,
                            end_offset=end,
                            replacement=f"{lhs_text} {reverse_ops[rhs_operator]} {rhs_right_text}",
                        )
                    ],
                )
            )

        return candidates


ASSIGNMENT_OPERATORS = {"="} | set(COMPOUND_OPERATORS)


def _contains_assignment(code: str, cursor: ci.Cursor) -> bool:
    """Return True if any node in the subtree performs an assignment."""
    if cursor.kind == CursorKind.BINARY_OPERATOR:
        op = _binary_operator_text(code, cursor)
        if op in ASSIGNMENT_OPERATORS:
            return True
    for child in cursor.get_children():
        if _contains_assignment(code, child):
            return True
    return False


def _contains_side_effects(code: str, cursor: ci.Cursor) -> bool:
    """Conservative side-effect detector for expression-reordering transforms."""
    if cursor.kind == CursorKind.CALL_EXPR:
        return True
    if cursor.kind == CursorKind.UNARY_OPERATOR:
        prefix = _unary_prefix_text(code, cursor)
        if "++" in prefix or "--" in prefix:
            return True
    if cursor.kind == CursorKind.BINARY_OPERATOR:
        op = _binary_operator_text(code, cursor)
        if op in ASSIGNMENT_OPERATORS or op == ",":
            return True
    if cursor.kind == getattr(CursorKind, "CONDITIONAL_OPERATOR", CursorKind.UNEXPOSED_EXPR):
        return True
    for child in cursor.get_children():
        if _contains_side_effects(code, child):
            return True
    return False


class SwapCondition(AstTransform):
    """
    Swaps operands of comparison expressions.
    a < b  ->  b > a
    """

    name = "SwapCondition"
    SWAP_MAP = {
        "<": ">",
        ">": "<",
        "<=": ">=",
        ">=": "<=",
        "==": "==",
        "!=": "!=",
    }

    def _find_candidates(
        self,
        code: str,
        index: ci.Index,
        filename: str = "tmp.cpp",
    ) -> list[RewriteCandidate]:
        tu = _parse_translation_unit(code, index, filename)
        filename = tu.spelling
        candidates: list[RewriteCandidate] = []

        for cursor in _iter_cursors(tu.cursor):
            if cursor.kind != CursorKind.BINARY_OPERATOR:
                continue
            if not _cursor_in_main_file(cursor, filename) or _is_macro_expansion(cursor):
                continue

            operator = _binary_operator_text(code, cursor)
            if operator not in self.SWAP_MAP:
                continue

            raw_children = list(cursor.get_children())
            if len(raw_children) != 2:
                continue
            lhs, rhs = _unwrap_expr(raw_children[0]), _unwrap_expr(raw_children[1])
            # Fix C: skip comparisons where either operand contains an assignment.
            # Swapping would produce a non-lvalue or change program semantics.
            if _contains_assignment(code, lhs) or _contains_assignment(code, rhs):
                continue
            if _contains_side_effects(code, lhs) or _contains_side_effects(code, rhs):
                continue
            if _is_preprocessor_line(code, cursor.extent.start.offset):
                continue

            lhs_start = raw_children[0].extent.start.offset
            rhs_end = raw_children[1].extent.end.offset
            lhs_text = _source_text(code, raw_children[0]).strip()
            rhs_text = _source_text(code, raw_children[1]).strip()
            candidates.append(
                RewriteCandidate(
                    description=f"{self.name}: swap",
                    edits=[
                        TextEdit(
                            start_offset=lhs_start,
                            end_offset=rhs_end,
                            replacement=f"{rhs_text} {self.SWAP_MAP[operator]} {lhs_text}",
                        )
                    ],
                )
            )

        return candidates


def _is_array_like(cursor: ci.Cursor) -> bool:
    return cursor.type.kind in ARRAY_LIKE_TYPE_KINDS


def _is_integer_like(cursor: ci.Cursor) -> bool:
    return cursor.type.kind in INTEGER_TYPE_KINDS


class PointerArithmeticToArrayIndex(AstTransform):
    """
    Converts between pointer arithmetic and array indexing.
    *(arr + i) <-> arr[i]
    *(i + arr) <-> arr[i]
    """

    name = "PointerArithmeticToArrayIndex"

    def _pointer_addition_parts(
        self,
        code: str,
        cursor: ci.Cursor,
    ) -> Optional[tuple[str, str]]:
        if cursor.kind != CursorKind.UNARY_OPERATOR:
            return None
        if "*" not in _unary_prefix_text(code, cursor):
            return None

        children = _children(cursor)
        if len(children) != 1:
            return None
        child = _unwrap_expr(children[0])
        if child.kind != CursorKind.BINARY_OPERATOR:
            return None
        if _binary_operator_text(code, child) != "+":
            return None

        operands = _children(child)
        if len(operands) != 2:
            return None
        left, right = operands
        # Restrict to side-effect free, simple base/index forms. This avoids
        # changing semantics when pointer/index expressions contain calls or
        # increments and reduces fragile rewrites in complex kernels.
        if _contains_side_effects(code, left) or _contains_side_effects(code, right):
            return None
        if left.kind == CursorKind.DECL_REF_EXPR and _is_array_like(left) and right.kind in {CursorKind.DECL_REF_EXPR, CursorKind.INTEGER_LITERAL} and _is_integer_like(right):
            return _source_text(code, left).strip(), _source_text(code, right).strip()
        if right.kind == CursorKind.DECL_REF_EXPR and _is_array_like(right) and left.kind in {CursorKind.DECL_REF_EXPR, CursorKind.INTEGER_LITERAL} and _is_integer_like(left):
            return _source_text(code, right).strip(), _source_text(code, left).strip()
        return None

    def _find_candidates(
        self,
        code: str,
        index: ci.Index,
        filename: str = "tmp.cpp",
    ) -> list[RewriteCandidate]:
        tu = _parse_translation_unit(code, index, filename)
        filename = tu.spelling
        candidates: list[RewriteCandidate] = []

        for cursor in _iter_cursors(tu.cursor):
            if not _cursor_in_main_file(cursor, filename) or _is_macro_expansion(cursor):
                continue

            parts = self._pointer_addition_parts(code, cursor)
            if parts is None:
                continue
            base_text, idx_text = parts
            if not base_text or not idx_text:
                continue
            start, end = _cursor_offsets(cursor)
            candidates.append(
                RewriteCandidate(
                    description=f"{self.name}: ptr->index",
                    edits=[
                        TextEdit(
                            start_offset=start,
                            end_offset=end,
                            replacement=f"({base_text})[{idx_text}]",
                        )
                    ],
                )
            )

        return candidates


class TypedefExpansion(AstTransform):
    """Expands typedef aliases at AST-identified type-reference sites."""

    name = "TypedefExpansion"

    # Underlying types that are unsafe to expand (complex or framework-specific)
    _UNSAFE_UNDERLYING_PREFIXES = (
        "struct ",
        "union ",
        "enum ",
    )
    _SAFE_SCALAR_UNDERLYING = {
        "bool",
        "char",
        "signed char",
        "unsigned char",
        "short",
        "short int",
        "unsigned short",
        "unsigned short int",
        "int",
        "unsigned",
        "unsigned int",
        "long",
        "long int",
        "unsigned long",
        "unsigned long int",
        "long long",
        "long long int",
        "unsigned long long",
        "unsigned long long int",
        "float",
        "double",
        "long double",
    }

    def _is_safe_to_expand(self, alias: str, underlying: str) -> bool:
        """Return True only if the expansion is safe and meaningful."""
        normalized = " ".join(underlying.replace("\t", " ").strip().split())
        normalized = normalized.replace("const ", "").replace("volatile ", "").strip()
        # Never expand to a struct/union/enum tag in C++
        if any(normalized.startswith(p) for p in self._UNSAFE_UNDERLYING_PREFIXES):
            return False
        # Never expand namespace-qualified types (templates, std::, etc.)
        if "::" in normalized:
            return False
        # Never expand template instantiations
        if "<" in normalized or ">" in normalized:
            return False
        # Never expand pointer/reference/function-like aliases.
        if "*" in normalized or "&" in normalized or "(" in normalized or ")" in normalized:
            return False
        # Restrict to primitive scalar aliases only.
        if normalized not in self._SAFE_SCALAR_UNDERLYING:
            return False
        # Heuristic: libclang sometimes resolves complex aliases to plain 'int'
        # Guard against this by requiring the alias name to suggest an integer type.
        if normalized == "int" and alias != "int":
            int_hints = {"int", "count", "size", "offset", "id", "index", "num", "len"}
            if not any(h in alias.lower() for h in int_hints):
                return False
        return True

    def _find_candidates(
        self,
        code: str,
        index: ci.Index,
        filename: str = "tmp.cpp",
    ) -> list[RewriteCandidate]:
        tu = _parse_translation_unit(code, index, filename)
        filename = tu.spelling
        alias_by_usr: dict[str, tuple[str, str]] = {}

        for cursor in _iter_cursors(tu.cursor):
            if cursor.kind != CursorKind.TYPEDEF_DECL:
                continue
            if not _cursor_in_main_file(cursor, filename) or _is_macro_expansion(cursor):
                continue
            usr = cursor.get_usr()
            if not usr:
                continue
            alias_by_usr[usr] = (
                cursor.spelling,
                cursor.underlying_typedef_type.spelling,
            )

        candidates: list[RewriteCandidate] = []
        for cursor in _iter_cursors(tu.cursor):
            if cursor.kind != CursorKind.TYPE_REF:
                continue
            if not _cursor_in_main_file(cursor, filename) or _is_macro_expansion(cursor):
                continue
            referenced = cursor.referenced
            if referenced is None:
                continue
            entry = alias_by_usr.get(referenced.get_usr())
            if entry is None:
                continue
            alias, underlying = entry
            if not self._is_safe_to_expand(alias, underlying):
                continue

            token_range = _identifier_range(cursor, alias)
            if token_range is None:
                continue
            start, end = token_range
            candidates.append(
                RewriteCandidate(
                    description=f"{self.name}: expand {alias}->{underlying}",
                    edits=[
                        TextEdit(
                            start_offset=start,
                            end_offset=end,
                            replacement=underlying,
                        )
                    ],
                )
            )

        return candidates


class ChangeNames(Transform):
    """
    Consistently renames local variables and function parameters using symbol-aware
    reference collection from libclang.
    """

    name = "ChangeNames"
    RESERVED = {
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "int",
        "long",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "bool",
        "true",
        "false",
        "nullptr",
        "class",
        "public",
        "private",
        "protected",
        "virtual",
        "template",
        "typename",
        "namespace",
        "using",
        "new",
        "delete",
        "this",
        "throw",
        "try",
        "catch",
        "printf",
        "scanf",
        "malloc",
        "free",
        "calloc",
        "realloc",
        "memcpy",
        "memset",
        "strlen",
        "strcmp",
        "strcpy",
        "strcat",
        "fopen",
        "fclose",
        "fprintf",
        "fscanf",
        "fread",
        "fwrite",
        "exit",
        "abort",
        "assert",
        "sqrt",
        "pow",
        "exp",
        "log",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "floor",
        "ceil",
        "fabs",
        "abs",
        "min",
        "max",
        "fmin",
        "fmax",
        "omp_get_thread_num",
        "omp_get_num_threads",
        "omp_set_num_threads",
        "size_t",
        "ptrdiff_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "main",
        "argc",
        "argv",
        # C++ standard namespaces and common members
        "std",
        "cuda",
        "thrust",
        "cub",
        "nvcuda",
        "cl",
        "__gnu_cxx",
        "cout",
        "cerr",
        "cin",
        "clog",
        "endl",
        "flush",
        "setw",
        "setprecision",
        "fixed",
        "scientific",
        "hex",
        "dec",
        "oct",
        "ws",
        "left",
        "right",
        "boolalpha",
        "string",
        "vector",
        "map",
        "set",
        "pair",
        "make_pair",
        "sort",
        "find",
        "begin",
        "end",
        "runtime_error",
        "logic_error",
        "exception",
        # CUDA built-ins
        "blockIdx",
        "blockDim",
        "threadIdx",
        "gridDim",
        "warpSize",
        "__syncthreads",
        "atomicAdd",
        "atomicSub",
        "atomicMin",
        "atomicMax",
        "atomicExch",
        "atomicCAS",
        "__shared__",
        "__global__",
        "__device__",
        "__host__",
        "__constant__",
        # OpenCL built-ins
        "get_global_id",
        "get_local_id",
        "get_group_id",
        "get_global_size",
        "get_local_size",
        "get_num_groups",
        "barrier",
        "CLK_LOCAL_MEM_FENCE",
        "CLK_GLOBAL_MEM_FENCE",
        # SYCL
        "sycl",
        "cl::sycl",
        # OpenMP
        "omp_get_wtime",
    } | OMP_PRAGMA_KEYWORDS
    NAME_PREFIXES = ["v", "x", "tmp", "val", "var", "data", "elem", "item"]

    def _file_contains_omp_pragma(self, code: str) -> bool:
        return bool(re.search(r"^\s*#pragma\s+omp\b", code, re.MULTILINE))

    def _owning_function(self, cursor: ci.Cursor) -> Optional[ci.Cursor]:
        current = cursor.semantic_parent
        while current is not None and current.kind != CursorKind.TRANSLATION_UNIT:
            if current.kind in {
                CursorKind.FUNCTION_DECL,
                getattr(CursorKind, "CXX_METHOD", CursorKind.FUNCTION_DECL),
                getattr(CursorKind, "CONSTRUCTOR", CursorKind.FUNCTION_DECL),
                getattr(CursorKind, "DESTRUCTOR", CursorKind.FUNCTION_DECL),
            }:
                return current
            current = current.semantic_parent
        return None

    def _macro_identifiers(self, tu: ci.TranslationUnit) -> set[str]:
        """Collect all identifiers that appear within macro definitions."""
        idents = set()
        for cursor in _iter_cursors(tu.cursor):
            if cursor.kind == CursorKind.MACRO_DEFINITION:
                for token in cursor.get_tokens():
                    if token.kind == TokenKind.IDENTIFIER:
                        idents.add(token.spelling)
        return idents

    def _renamable_decls(self, tu: ci.TranslationUnit) -> list[ci.Cursor]:
        if _is_low_quality_tu(tu):
            return []
        filename = tu.spelling
        # OpenMP pragma clauses are fragile under token-level renaming because
        # libclang does not reliably resolve identifier bindings inside pragmas.
        # Conservatively skip ChangeNames for these files.
        try:
            main_code = Path(filename).read_text(encoding="utf-8", errors="replace")
            if self._file_contains_omp_pragma(main_code):
                return []
        except Exception:
            pass
        macro_idents = self._macro_identifiers(tu)
        decls: list[ci.Cursor] = []
        for cursor in _iter_cursors(tu.cursor):
            if cursor.kind not in {CursorKind.VAR_DECL, CursorKind.PARM_DECL}:
                continue
            if not _cursor_in_main_file(cursor, filename) or _is_macro_expansion(cursor):
                continue
            if len(cursor.spelling) <= 1:
                continue
            if cursor.spelling in self.RESERVED or cursor.spelling in macro_idents:
                continue
            if self._owning_function(cursor) is None:
                continue
            decls.append(cursor)
        return decls

    def _existing_identifiers(self, tu: ci.TranslationUnit) -> set[str]:
        return {
            token.spelling
            for token in tu.get_tokens(extent=tu.cursor.extent)
            if token.kind == TokenKind.IDENTIFIER
        }

    def _fresh_name(self, used_names: set[str], counter: int) -> tuple[str, int]:
        """Return a new unique name and the incremented counter."""
        while True:
            candidate = f"{random.choice(self.NAME_PREFIXES)}_{counter}"
            counter += 1
            if candidate not in used_names and candidate not in self.RESERVED:
                used_names.add(candidate)
                return candidate, counter

    def _symbol_edits(
        self,
        code: str,
        tu: ci.TranslationUnit,
        decl: ci.Cursor,
        new_name: str,
    ) -> list[TextEdit]:
        usr = decl.get_usr()
        if not usr:
            return []

        owner = self._owning_function(decl)
        if owner is None:
            return []

        edits: list[TextEdit] = []
        decl_range = _identifier_range(decl)
        if decl_range is not None:
            edits.append(
                TextEdit(
                    start_offset=decl_range[0],
                    end_offset=decl_range[1],
                    replacement=new_name,
                )
            )

        for cursor in _iter_cursors(owner):
            if cursor.kind != CursorKind.DECL_REF_EXPR:
                continue
            referenced = cursor.referenced
            if referenced is None or referenced.get_usr() != usr:
                continue
            token_range = _identifier_range(cursor)
            if token_range is None:
                continue
            edits.append(
                TextEdit(
                    start_offset=token_range[0],
                    end_offset=token_range[1],
                    replacement=new_name,
                )
            )

        # Fallback token scan for CUDA/OpenCL sites that libclang doesn't emit
        # DECL_REF_EXPR for (e.g. kernel launch arguments).
        # Disable this fallback inside OpenMP pragma-heavy functions because
        # unresolved pragma tokens can be mis-identified and rewritten.
        spelling = decl.spelling
        if spelling and not self._function_has_omp_pragma(code, owner):
            other_decls_with_same_spelling = [
                d for d in self._renamable_decls(tu)
                if d.spelling == spelling and d.get_usr() != usr
                and self._owning_function(d) == owner
            ]
            ast_offsets: set[int] = {e.start_offset for e in edits}
            for token in owner.get_tokens():
                if token.kind != TokenKind.IDENTIFIER:
                    continue
                if token.spelling != spelling:
                    continue
                start = token.extent.start.offset
                end = token.extent.end.offset
                if start in ast_offsets:
                    continue
                if token.spelling in self.RESERVED:
                    continue
                if _skip_preprocessor_token_rename(code, start, token.spelling):
                    continue
                site_cursor = ci.Cursor.from_location(tu, token.extent.start)
                if (
                    site_cursor.kind != CursorKind.INVALID_FILE
                    and site_cursor.referenced
                    and site_cursor.referenced.get_usr() != usr
                ):
                    continue
                if _is_macro_expansion(site_cursor):
                    continue
                if not site_cursor.referenced and other_decls_with_same_spelling:
                    continue
                edits.append(
                    TextEdit(
                        start_offset=start,
                        end_offset=end,
                        replacement=new_name,
                    )
                )

        unique: dict[tuple[int, int], TextEdit] = {}
        for edit in edits:
            unique[(edit.start_offset, edit.end_offset)] = edit
        return sorted(unique.values(), key=lambda item: item.start_offset)

    def _function_has_omp_pragma(self, code: str, owner: ci.Cursor) -> bool:
        start, end = _cursor_offsets(owner)
        owner_text = code[start:end]
        return bool(re.search(r"^\s*#pragma\s+omp\b", owner_text, re.MULTILINE))

    def _decl_used_in_omp_pragma(
        self,
        code: str,
        tu: ci.TranslationUnit,
        decl: ci.Cursor,
    ) -> bool:
        owner = self._owning_function(decl)
        if owner is None:
            return False
        spelling = decl.spelling
        if not spelling:
            return False
        start, end = _cursor_offsets(owner)
        owner_text = code[start:end]
        pattern = re.compile(rf"^\s*#pragma\s+omp\b.*\b{re.escape(spelling)}\b", re.MULTILINE)
        return bool(pattern.search(owner_text))

    def is_applicable(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> bool:
        tu = _parse_translation_unit(code, index, filename)
        return bool(self._renamable_decls(tu))

    def apply(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> TransformResult:
        tu = _parse_translation_unit(code, index, filename)
        all_decls = self._renamable_decls(tu)
        if not all_decls:
            return TransformResult(False, code, f"{self.name}: renamed_vars=0")

        used_names = self._existing_identifiers(tu)

        # _select_fraction shuffles before slicing, so level 2/3 pick a random
        # subset rather than always the first N declarations in AST order.
        # At level 4 all decls are selected (fraction = 1.0).
        selected_decls = self._select_fraction(all_decls)

        # Build one RewriteCandidate per declaration so that we can detect
        # cross-rename edit overlaps (e.g. two vars sharing a spelling at the
        # same site) and fall back gracefully rather than producing corrupt code.
        per_decl_candidates: list[RewriteCandidate] = []
        counter = 0
        for decl in selected_decls:
            if self._decl_used_in_omp_pragma(code, tu, decl):
                continue
            new_name, counter = self._fresh_name(used_names, counter)
            edits = self._symbol_edits(code, tu, decl, new_name)
            if not edits:
                continue
            per_decl_candidates.append(
                RewriteCandidate(
                    description=f"rename {decl.spelling}->{new_name}",
                    edits=edits,
                )
            )

        if not per_decl_candidates:
            return TransformResult(False, code, f"{self.name}: renamed_vars=0")

        # --- Fast path: try merging all renames at once (level 4 common case) ---
        all_edits = [e for c in per_decl_candidates for e in c.edits]
        merged = RewriteCandidate(
            description=f"{self.name}: renamed_vars={len(per_decl_candidates)}",
            edits=all_edits,
        )
        if _validate_rewrite(merged, code, index, filename):
            return TransformResult(
                True,
                _apply_candidate(code, merged),
                merged.description,
            )

        # --- Slow path: greedy accumulation, one rename at a time ---
        # Validate each rename independently first to eliminate broken ones,
        # then re-validate as we accumulate so that interactions are caught.
        accepted: list[RewriteCandidate] = []
        for candidate in per_decl_candidates:
            # Skip if this rename's edit sites collide with an already-accepted one.
            if any(_edits_overlap(candidate, existing) for existing in accepted):
                continue
            # Build trial merge of everything accepted so far plus this rename.
            trial_edits = [e for a in accepted for e in a.edits] + list(candidate.edits)
            trial = RewriteCandidate(description="trial", edits=trial_edits)
            if _validate_rewrite(trial, code, index, filename):
                accepted.append(candidate)

        if not accepted:
            return TransformResult(False, code, f"{self.name}: renamed_vars=0 (all failed validation)")

        final_edits = [e for c in accepted for e in c.edits]
        final = RewriteCandidate(
            description=f"{self.name}: renamed_vars={len(accepted)} (greedy), replaced_occurrences={len(final_edits)}",
            edits=final_edits,
        )
        return TransformResult(True, _apply_candidate(code, final), final.description)


def _string_literals_in_file(tu: ci.TranslationUnit, filename: str) -> set[str]:
    """
    Return the set of string values that appear as string-literal tokens in the
    main file.  Used to detect OpenCL kernel names passed to clCreateKernel.
    """
    values: set[str] = set()
    for token in tu.get_tokens(extent=tu.cursor.extent):
        if token.kind != TokenKind.LITERAL:
            continue
        spelling = token.spelling
        # Only plain string literals: "foo"
        if not (spelling.startswith('"') and spelling.endswith('"')):
            continue
        try:
            loc = token.extent.start
            if loc.file and loc.file.name == filename:
                values.add(spelling[1:-1])  # strip surrounding quotes
        except Exception:
            pass
    return set(values)


class ChangeFunctionNames(Transform):
    """
    Consistently renames user-defined functions (free functions and class methods)
    across their definition and all call sites within the same translation unit.

    Safety rules — a function is skipped if any of the following apply:
      - It is in the RESERVED set (main, standard library calls, framework built-ins)
      - It is a virtual method or overrides a base-class method (would break vtable)
      - It is a kernel function whose name appears in a string literal in the file
        (OpenCL clCreateKernel looks up kernels by name string at runtime)
      - Its definition is not in the main file (header-only / external)
      - It has no renameable call sites in the file (nothing to gain)

    Only internal-linkage functions are renamed (`static` helpers or functions
    in anonymous namespaces). External-linkage renames are intentionally skipped
    because cross-file references are out of scope for single-file rewriting.
    """

    name = "ChangeFunctionNames"

    # Inherits RESERVED from ChangeNames via composition; duplicated here so the
    # class is self-contained.
    RESERVED: set[str] = ChangeNames.RESERVED | {
        # Additional function-level reserved names
        "operator",
        "main",
    }
    NAME_PREFIXES = ["fn", "func", "op", "proc", "compute", "run", "exec", "calc"]
    PARALLEL_ENTRY_BUILTINS = {
        "blockIdx",
        "blockDim",
        "threadIdx",
        "gridDim",
        "warpSize",
        "get_global_id",
        "get_local_id",
        "get_group_id",
        "get_global_size",
        "get_local_size",
        "get_num_groups",
        "item",
        "nd_item",
        "group",
    }

    def _is_kernel_function(self, cursor: ci.Cursor) -> bool:
        """Return True if the function has __global__ or __kernel__ attributes."""
        for token in cursor.get_tokens():
            if token.kind == TokenKind.KEYWORD and token.spelling in ("__global__", "__kernel__"):
                return True
            # Stop scanning once we reach the function body
            if token.spelling == "{":
                break
        return False

    def _uses_parallel_entry_builtins(self, cursor: ci.Cursor) -> bool:
        """Heuristic fallback for parsers that macro-erase kernel attributes."""
        for token in cursor.get_tokens():
            if token.kind == TokenKind.IDENTIFIER and token.spelling in self.PARALLEL_ENTRY_BUILTINS:
                return True
        return False

    def _has_internal_linkage(self, cursor: ci.Cursor) -> bool:
        try:
            return cursor.linkage.name == "INTERNAL"
        except Exception:
            return False

    def _renamable_functions(
        self,
        tu: ci.TranslationUnit,
        string_literals: set[str],
    ) -> list[ci.Cursor]:
        """
        Return renameable free-function declarations defined in the main file
        that have internal linkage.
        """
        if _is_low_quality_tu(tu):
            return []
        filename = tu.spelling
        seen_usrs: set[str] = set()
        result: list[ci.Cursor] = []

        for cursor in _iter_cursors(tu.cursor):
            # Only plain free functions — methods/constructors/destructors are
            # never static helpers and are excluded entirely.
            if cursor.kind != CursorKind.FUNCTION_DECL:
                continue
            if not _cursor_in_main_file(cursor, filename):
                continue
            if _is_macro_expansion(cursor):
                continue

            usr = cursor.get_usr()
            if not usr or usr in seen_usrs:
                continue
            seen_usrs.add(usr)
            if not self._has_internal_linkage(cursor):
                continue

            name = cursor.spelling
            if not name or len(name) <= 1:
                continue
            base_name = name.split("<")[0]  # strip template suffix if any
            if base_name in self.RESERVED:
                continue
            if "operator" in base_name:
                continue

            # Skip all GPU/OpenCL entry points. Their launch/invocation sites
            # commonly live in sibling files, so per-file rewriting is incomplete.
            if self._is_kernel_function(cursor) or self._uses_parallel_entry_builtins(cursor):
                continue

            # Also skip non-kernel functions whose symbol is matched indirectly
            # via string lookup in the current file.
            if base_name in string_literals:
                continue

            result.append(cursor)
        return result

    def _collect_call_sites(
        self,
        tu: ci.TranslationUnit,
        usr: str,
        filename: str,
    ) -> list[tuple[int, int]]:
        """
        Return byte ranges of every CALL_EXPR / DECL_REF_EXPR token that
        references the function with the given USR within the main file.
        """
        ranges: list[tuple[int, int]] = []
        for cursor in _iter_cursors(tu.cursor):
            if not _cursor_in_main_file(cursor, filename):
                continue
            if cursor.kind not in {
                CursorKind.CALL_EXPR,
                CursorKind.DECL_REF_EXPR,
            }:
                continue
            referenced = cursor.referenced
            if referenced is None:
                continue
            if referenced.get_usr() != usr:
                continue
            # Find the identifier token that spells the function name
            token_range = _identifier_range(cursor)
            if token_range is not None:
                ranges.append(token_range)
        return ranges

    def _fresh_name(self, used_names: set[str], counter: int) -> tuple[str, int]:
        while True:
            candidate = f"{random.choice(self.NAME_PREFIXES)}_{counter}"
            counter += 1
            if candidate not in used_names and candidate not in self.RESERVED:
                used_names.add(candidate)
                return candidate, counter

    def _existing_identifiers(self, tu: ci.TranslationUnit) -> set[str]:
        return {
            token.spelling
            for token in tu.get_tokens(extent=tu.cursor.extent)
            if token.kind == TokenKind.IDENTIFIER
        }

    def _build_candidate(
        self,
        code: str,
        tu: ci.TranslationUnit,
        func_cursor: ci.Cursor,
        new_name: str,
    ) -> Optional[RewriteCandidate]:
        """Build a RewriteCandidate that renames func_cursor to new_name everywhere."""
        filename = tu.spelling
        usr = func_cursor.get_usr()
        edits: list[TextEdit] = []

        # Rename the declaration/definition site
        decl_range = _identifier_range(func_cursor, func_cursor.spelling.split("<")[0])
        if decl_range is None:
            return None
        edits.append(TextEdit(
            start_offset=decl_range[0],
            end_offset=decl_range[1],
            replacement=new_name,
        ))

        # Rename all call sites
        for start, end in self._collect_call_sites(tu, usr, filename):
            edits.append(TextEdit(start_offset=start, end_offset=end, replacement=new_name))

        # Deduplicate by offset key (decl site may coincide with a call site in
        # single-file translation units with inline definitions).
        unique: dict[tuple[int, int], TextEdit] = {}
        for edit in edits:
            unique[(edit.start_offset, edit.end_offset)] = edit

        return RewriteCandidate(
            description=f"rename {func_cursor.spelling}->{new_name}",
            edits=sorted(unique.values(), key=lambda e: e.start_offset),
        )

    def is_applicable(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> bool:
        tu = _parse_translation_unit(code, index, filename)
        string_literals = _string_literals_in_file(tu, tu.spelling)
        return bool(self._renamable_functions(tu, string_literals))

    def apply(self, code: str, index: ci.Index, filename: str = "tmp.cpp") -> TransformResult:
        tu = _parse_translation_unit(code, index, filename)
        if _fatal_diagnostics(tu) > BASELINE_ERROR_THRESHOLD:
            return TransformResult(False, code, f"{self.name}: too many baseline errors")

        string_literals = _string_literals_in_file(tu, tu.spelling)
        all_funcs = self._renamable_functions(tu, string_literals)
        if not all_funcs:
            return TransformResult(False, code, f"{self.name}: no renamable functions")

        used_names = self._existing_identifiers(tu)
        selected_funcs = self._select_fraction(all_funcs)

        # Build one candidate per function
        per_func_candidates: list[RewriteCandidate] = []
        counter = 0
        for func in selected_funcs:
            new_name, counter = self._fresh_name(used_names, counter)
            candidate = self._build_candidate(code, tu, func, new_name)
            if candidate is not None:
                per_func_candidates.append(candidate)

        if not per_func_candidates:
            return TransformResult(False, code, f"{self.name}: renamed_funcs=0")

        # Fast path: try all renames together (succeeds for most clean code)
        all_edits = [e for c in per_func_candidates for e in c.edits]
        merged = RewriteCandidate(
            description=f"{self.name}: renamed_funcs={len(per_func_candidates)}",
            edits=all_edits,
        )
        if _validate_rewrite(merged, code, index, filename):
            return TransformResult(True, _apply_candidate(code, merged), merged.description)

        # Slow path: greedy accumulation to salvage as many renames as possible
        accepted: list[RewriteCandidate] = []
        for candidate in per_func_candidates:
            if any(_edits_overlap(candidate, existing) for existing in accepted):
                continue
            trial_edits = [e for a in accepted for e in a.edits] + list(candidate.edits)
            trial = RewriteCandidate(description="trial", edits=trial_edits)
            if _validate_rewrite(trial, code, index, filename):
                accepted.append(candidate)

        if not accepted:
            return TransformResult(False, code, f"{self.name}: renamed_funcs=0 (all failed validation)")

        final_edits = [e for c in accepted for e in c.edits]
        final = RewriteCandidate(
            description=f"{self.name}: renamed_funcs={len(accepted)} (greedy), replaced_occurrences={len(final_edits)}",
            edits=final_edits,
        )
        return TransformResult(True, _apply_candidate(code, final), final.description)


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation process."""
    level: int = 4  # Aggressiveness level (1-4)
    seed: Optional[int] = None
    transforms: list[Transform] = field(default_factory=list)
    probability: float = 0.5  # Per-transform application probability


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping commented lines."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def save_jsonl(samples: list[dict], path: Path) -> None:
    """Save samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def augment_code(
    code: str,
    config: AugmentationConfig,
    index: ci.Index,
    filename: str = "tmp.cpp",
) -> tuple[str, list[str]]:
    """
    Apply augmentation transforms to code.

    Returns:
        Tuple of (augmented_code, list_of_applied_transform_names)
    """
    result = code
    applied_transforms: list[str] = []

    # FIX: select the subset of transforms to run, then iterate *only* that subset.
    # Previously the code iterated config.transforms and used identity checks against
    # transforms_to_try, which was unreliable and effectively ran all transforms.
    transforms_pool = list(config.transforms)
    random.shuffle(transforms_pool)

    if config.level == 1:
        transforms_to_run = transforms_pool[:1]
    elif config.level == 2:
        transforms_to_run = transforms_pool[: max(1, int(len(transforms_pool) * 0.33))]
    elif config.level == 3:
        transforms_to_run = transforms_pool[: max(1, int(len(transforms_pool) * 0.66))]
    else:
        transforms_to_run = transforms_pool  # level 4: all transforms

    for transform in transforms_to_run:
        if transform.is_applicable(result, index, filename):
            transform_result = transform.apply(result, index, filename)
            if transform_result.applied:
                result = transform_result.code
                applied_transforms.append(transform.name)

    return result, applied_transforms


def augment_sample(sample: dict, config: AugmentationConfig, index: ci.Index) -> dict:
    """Augment a single sample."""
    augmented = sample.copy()
    augmented_code = {}
    all_transforms = []

    code_dict = sample.get('code', {})
    for filename, code_content in code_dict.items():
        # FIX: raise the minimum content length guard; 50 chars is too aggressive.
        # An empty string or a pure filename reference should still be skipped, but
        # real short kernels (e.g. a 3-line helper) should be attempted.
        if not isinstance(code_content, str) or code_content.strip() == filename.strip():
            augmented_code[filename] = code_content
            continue

        aug_code, transforms = augment_code(code_content, config, index, filename=filename)
        augmented_code[filename] = aug_code
        all_transforms.extend(transforms)

    augmented['code'] = augmented_code
    augmented['augmentation'] = {
        'transforms_applied': all_transforms,
        'original_kernel': sample.get('kernel_name', 'unknown')
    }

    return augmented


def main():
    parser = argparse.ArgumentParser(
        description='Augment C/C++ code dataset with semantics-preserving transforms'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input JSONL file or directory containing JSONL files'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for augmented dataset'
    )
    parser.add_argument(
        '--augment_level', '-l',
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help='Aggressiveness level (1=light, 4=aggressive) (default: 4)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--include-original',
        action='store_true',
        help='Include original samples alongside augmented ones'
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Initialize libclang
    index = ci.Index.create()

    # Configure transforms
    config = AugmentationConfig(
        level=args.augment_level,
        seed=args.seed,
        transforms=[
            ArithmeticTransform(level=args.augment_level),
            SwapCondition(level=args.augment_level),
            PointerArithmeticToArrayIndex(level=args.augment_level),
            TypedefExpansion(level=args.augment_level),
            ChangeNames(level=args.augment_level),
            ChangeFunctionNames(level=args.augment_level),
        ]
    )

    # Process input
    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob('*.jsonl'))

    total_samples = 0
    total_augmented = 0

    for input_file in input_files:
        print(f"Processing {input_file}...")
        samples = load_jsonl(input_file)

        output_samples = []

        for sample in samples:
            total_samples += 1

            # Optionally include original
            if args.include_original:
                original = sample.copy()
                original['augmentation'] = {'transforms_applied': [], 'is_original': True}
                output_samples.append(original)

            # Create augmented version
            augmented = augment_sample(sample, config, index)

            # Only include if something was actually changed
            if augmented.get('augmentation', {}).get('transforms_applied'):
                output_samples.append(augmented)
                total_augmented += 1
            elif not args.include_original:
                # If nothing changed and we're not including originals, still include the sample
                augmented['augmentation'] = {'transforms_applied': [], 'no_applicable_transforms': True}
                output_samples.append(augmented)

        # Save output
        output_file = output_dir / input_file.name
        save_jsonl(output_samples, output_file)
        print(f"  Saved {len(output_samples)} samples to {output_file}")

    print(f"\nSummary:")
    print(f"  Total input samples: {total_samples}")
    print(f"  Samples with augmentation: {total_augmented}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
