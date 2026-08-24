# C/C++ Code Augmentation Toolkit

Semantics-preserving source-to-source transforms for C/C++ kernel code, powered by libclang.
Designed for data augmentation in GPU kernel translation pipelines.

Candidate discovery is AST-driven: transforms walk libclang cursors, select valid
expression/type/reference nodes, and rewrite exact source spans from the original file.
This avoids the template/typedef/preprocessor false positives that regex-only rewriting
can introduce.

## Transforms

| Transform | What it does |
|---|---|
| **ArithmeticTransform** | `x += 1` <-> `x = x + 1` (augmented/expanded assignment) |
| **SwapCondition** | `a < b` -> `b > a` (swap comparison operands) |
| **PointerArithmeticToArrayIndex** | `*(arr + i)` <-> `arr[i]` |
| **ChangeNames** | Consistently rename local variables (`counter` -> `v_0`) |
| **TypedefExpansion** | Expand typedef aliases to their underlying types |

All transforms are intended to preserve program semantics. The packaged test suite
covers positive cases, negative cases (templates / typedef declarations / preprocessor
constructs), and libclang re-parse checks after rewrite.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

### `augment_dataset.py` -- Core library & bulk augmenter

Contains the transform class definitions and a CLI for bulk-augmenting JSONL datasets
(applies multiple transforms per sample with configurable probability).

```bash
python augment_dataset.py \
    --input dataset.jsonl \
    --output augmented_output/ \
    --probability 0.5 \
    --seed 42 \
    --include-original
```

### `generate_single_aug.py` -- Single-transform dataset generator

Applies exactly **one** randomly selected transform per kernel to isolate individual
transform impact. Reads from a directory of `<kernel>-serial/` source trees and writes
augmented copies alongside an `augmentation_metadata.json` per kernel.

```bash
# 1. Create a text file listing kernel names (one per line):
#    bt
#    cg
#    ep
#    ...

# 2. Run:
python generate_single_aug.py \
    --kernels-file kernels.txt \
    --src-root /path/to/original_serial_dataset/src \
    --out-root /path/to/single_aug_output/src \
    --seed 1337
```

Each output kernel directory will contain the (possibly transformed) source files plus
`augmentation_metadata.json`:

```json
{
  "kernel_name": "bt",
  "parallel_api": "serial",
  "is_original": false,
  "transforms_applied": ["SwapCondition"],
  "sampling_mode": "single_transform",
  "seed": 1337,
  "applicable_transforms": ["SwapCondition", "PointerArithmeticToArrayIndex", "ChangeNames", "ArithmeticTransform"],
  "selected_transform": "SwapCondition"
}
```

### `test_transforms.py` -- Transform sanity and safety checks

Runs assertion-backed checks for each transform on small code snippets, including:

- positive rewrite cases
- negative cases that must not be rewritten
- libclang parse checks on rewritten output

```bash
python test_transforms.py
```

### `validate_augmentation.py` -- Compile-and-run validation

Compiles original and augmented code, runs both, and compares outputs and runtimes
to verify semantic equivalence.

```bash
python validate_augmentation.py \
    --input augmented_dataset.jsonl \
    --output validation_results.json \
    --num-runs 10
```

## Using the transforms programmatically

```python
import clang.cindex as ci
from augment_dataset import (
    ArithmeticTransform,
    SwapCondition,
    PointerArithmeticToArrayIndex,
    ChangeNames,
    AugmentationConfig,
    augment_code,
)

index = ci.Index.create()
code = open("kernel.cpp").read()

# Apply a single transform
t = SwapCondition()
if t.is_applicable(code, index):
    result = t.apply(code, index)
    if result.applied:
        print(result.code)

# Or use the config-driven pipeline
config = AugmentationConfig(
    probability=0.5,
    transforms=[ArithmeticTransform(), SwapCondition(), ChangeNames()],
)
augmented_code, applied = augment_code(code, config, index)
print(f"Applied: {applied}")
```
