#!/usr/bin/env python3
"""Assertion-backed tests for the AST-driven augmentation transforms."""

from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import clang.cindex as ci

from augment_dataset import (
    ArithmeticTransform,
    AugmentationConfig,
    ChangeFunctionNames,
    ChangeNames,
    PointerArithmeticToArrayIndex,
    SwapCondition,
    TypedefExpansion,
    augment_code,
)


@contextmanager
def deterministic_random():
    orig_random = random.random
    orig_choice = random.choice

    def always_zero() -> float:
        return 0.0

    def first(seq):
        return seq[0]

    random.random = always_zero  # type: ignore[assignment]
    random.choice = first  # type: ignore[assignment]
    try:
        yield
    finally:
        random.random = orig_random  # type: ignore[assignment]
        random.choice = orig_choice  # type: ignore[assignment]


def assert_parseable(code: str) -> None:
    index = ci.Index.create()
    tu = index.parse(
        path="snippet.cpp",
        args=["-xc++", "-std=c++17"],
        unsaved_files=[("snippet.cpp", code)],
        options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    fatals = [diag for diag in tu.diagnostics if diag.severity >= ci.Diagnostic.Fatal]
    assert not fatals, f"fatal parse diagnostics: {[str(diag) for diag in fatals]}"


def apply_deterministically(transform, code: str) -> str:
    index = ci.Index.create()
    with deterministic_random():
        result = transform.apply(code, index)
    assert result.applied, f"{transform.name} did not apply"
    assert_parseable(result.code)
    return result.code


def test_arithmetic_transform() -> None:
    code = """
void foo() {
    int x = 0;
    x += 1;
}
"""
    rewritten = apply_deterministically(ArithmeticTransform(), code)
    assert "x = x + (1)" in rewritten

    code = """
void foo() {
    int x = 0;
    x = x + 1;
}
"""
    rewritten = apply_deterministically(ArithmeticTransform(), code)
    assert "x += 1" in rewritten


def test_swap_condition_positive() -> None:
    code = """
void foo(int a, int b) {
    if (a < b) { }
}
"""
    rewritten = apply_deterministically(SwapCondition(), code)
    assert "if (b > a)" in rewritten


def test_swap_condition_negative_template_and_preprocessor() -> None:
    code = """
#if A < B
#endif

template <typename T>
struct Box {
    T value;
};
"""
    index = ci.Index.create()
    transform = SwapCondition()
    assert not transform.is_applicable(code, index)
    result = transform.apply(code, index)
    assert not result.applied
    assert result.code == code


def test_pointer_arithmetic_positive() -> None:
    code = """
void foo(int* arr) {
    int x = *(arr + 5);
}
"""
    rewritten = apply_deterministically(PointerArithmeticToArrayIndex(), code)
    assert "(arr)[5]" in rewritten


def test_pointer_arithmetic_negative_typedef_decl() -> None:
    code = """
typedef double nRarray[10][20];
"""
    index = ci.Index.create()
    transform = PointerArithmeticToArrayIndex()
    assert not transform.is_applicable(code, index)
    result = transform.apply(code, index)
    assert not result.applied
    assert result.code == code


def test_typedef_expansion() -> None:
    code = """
typedef unsigned int uint;

void foo() {
    uint count = 0;
}
"""
    rewritten = apply_deterministically(TypedefExpansion(), code)
    assert "unsigned int count" in rewritten


def test_change_names() -> None:
    code = """
int global_counter = 0;

void compute(int size, float* data) {
    int counter = 0;
    float result = 0.0f;
    for (int idx = 0; idx < size; idx++) {
        result += data[idx];
        counter++;
    }
}
"""
    rewritten = apply_deterministically(ChangeNames(), code)
    assert "global_counter" in rewritten
    assert "int counter = 0;" not in rewritten
    assert "float result = 0.0f;" not in rewritten


def test_change_function_names_positive() -> None:
    """Static helper functions are renamed; non-static (external) functions are not."""
    code = """
static int helper(int x) {
    return x * 2;
}

int caller(int n) {
    return helper(n) + helper(n + 1);
}
"""
    index = ci.Index.create()
    transform = ChangeFunctionNames()
    assert transform.is_applicable(code, index)
    with deterministic_random():
        result = transform.apply(code, index)
    assert result.applied
    assert_parseable(result.code)
    # helper() must be renamed (static = internal linkage)
    assert "helper" not in result.code
    # fn_0 is the deterministic name: NAME_PREFIXES[0]="fn" + counter=0
    assert "fn_0" in result.code
    # caller() must NOT be renamed (external linkage)
    assert "caller" in result.code


def test_change_function_names_skip_main() -> None:
    """main() is in RESERVED and must never be renamed."""
    code = """
int main(int argc, char** argv) {
    return 0;
}
"""
    index = ci.Index.create()
    transform = ChangeFunctionNames()
    assert not transform.is_applicable(code, index)
    result = transform.apply(code, index)
    assert not result.applied
    assert result.code == code


def test_change_function_names_skip_kernel() -> None:
    """Static functions that use CUDA builtins (threadIdx/blockIdx) must not be renamed."""
    code = """
static void myKernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * 32;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}
"""
    index = ci.Index.create()
    transform = ChangeFunctionNames()
    # _uses_parallel_entry_builtins fires on threadIdx/blockIdx tokens
    assert not transform.is_applicable(code, index)
    result = transform.apply(code, index)
    assert not result.applied


def test_change_function_names_skip_string_literal() -> None:
    """Functions whose name appears in a string literal must not be renamed (OpenCL clCreateKernel pattern)."""
    code = """
static void myFunc(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f;
    }
}

void setup() {
    const char* name = "myFunc";
    (void)name;
}
"""
    index = ci.Index.create()
    transform = ChangeFunctionNames()
    # myFunc is in string_literals so _renamable_functions excludes it
    assert not transform.is_applicable(code, index)
    result = transform.apply(code, index)
    assert not result.applied


def test_real_kernel_pipeline() -> None:
    code = """
void entropy(float* d_entropy, const char* d_val, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            char count[16];
            for (int i = 0; i < 16; i++) count[i] = 0;

            char total = 0;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int xx = x + dx;
                    int yy = y + dy;
                    if (xx >= 0 && yy >= 0 && yy < height && xx < width) {
                        count[d_val[yy * width + xx]] += 1;
                        total += 1;
                    }
                }
            }

            float entropy = 0;
            if (total < 1) {
                total = 1;
            }
            d_entropy[y * width + x] = entropy;
        }
    }
}
"""
    index = ci.Index.create()
    config = AugmentationConfig(
        probability=0.5,
        transforms=[
            ArithmeticTransform(),
            SwapCondition(),
            PointerArithmeticToArrayIndex(),
            TypedefExpansion(),
            ChangeNames(),
        ],
    )
    augmented_code, applied = augment_code(code, config, index)
    assert_parseable(augmented_code)
    assert isinstance(applied, list)


def test_pointer_arithmetic_overlapping_nested() -> None:
    """Bug A: nested subscripts like J[iS[i]*cols+j] must not produce overlapping edits."""
    code = """
void foo(int* J, int* iS, int cols) {
    int i = 0, j = 0;
    int val = J[iS[i] * cols + j];
}
"""
    index = ci.Index.create()
    transform = PointerArithmeticToArrayIndex()
    # Apply at level 4 (all candidates) — must still produce valid C++
    transform.level = 4
    with deterministic_random():
        result = transform.apply(code, index)
    # Whether applied or not, the output must be parseable (no corruption)
    assert_parseable(result.code)


def test_pointer_arithmetic_struct_member() -> None:
    """Fix B: *(ptr+i).member form must produce parseable output with base wrapped in parens.

    The arr[i]->*(ptr+i) reverse direction was removed; this tests the forward direction
    where *(ptr+i) with adjacent struct member access must produce (ptr)[i].member.
    """
    code = """
struct Node { int starting; int no_of_edges; };
void foo(struct Node* g_graph_nodes, int tid) {
    int start = (*(g_graph_nodes + tid)).starting;
}
"""
    rewritten = apply_deterministically(PointerArithmeticToArrayIndex(), code)
    assert_parseable(rewritten)
    # Must use array index form with base wrapped in parens
    assert "(g_graph_nodes)[tid]" in rewritten


def test_swap_condition_skip_assignment_in_operand() -> None:
    """Bug C: comparisons where an operand contains an assignment must be skipped."""
    # Pattern: (x = expr) == constant — swapping would produce (constant == x) = expr
    # which is a non-lvalue assignment error.
    code = """
void foo() {
    int fp;
    if ((fp = 5) == 0) { }
}
"""
    index = ci.Index.create()
    transform = SwapCondition()
    result = transform.apply(code, index)
    # The dangerous swap must not happen; transform should be skipped
    assert not result.applied or "0 == fp" not in result.code
    assert_parseable(result.code)


def main() -> int:
    tests = [
        test_arithmetic_transform,
        test_swap_condition_positive,
        test_swap_condition_negative_template_and_preprocessor,
        test_swap_condition_skip_assignment_in_operand,
        test_pointer_arithmetic_positive,
        test_pointer_arithmetic_negative_typedef_decl,
        test_pointer_arithmetic_overlapping_nested,
        test_pointer_arithmetic_struct_member,
        test_typedef_expansion,
        test_change_names,
        test_change_function_names_positive,
        test_change_function_names_skip_main,
        test_change_function_names_skip_kernel,
        test_change_function_names_skip_string_literal,
        test_real_kernel_pipeline,
    ]

    print("=" * 60)
    print("Testing C/C++ Augmentation Transforms")
    print("=" * 60)
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










