#!/usr/bin/env python3
"""
Generate a serial dataset where exactly ONE augmentation transform is sampled per kernel
and applied once (deterministically) to the primary .cpp file.

Isolates individual transform impact compared to baseline and "p=0.5 each" fully-augmented runs.

Usage:
  python generate_single_aug.py \
      --kernels-file kernels.txt \
      --src-root original_serial_dataset/src \
      --out-root single_aug_serial_dataset/src \
      --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import clang.cindex as ci
from pydantic import BaseModel, Field

from augment_dataset import (
    ArithmeticTransform,
    ChangeNames,
    PointerArithmeticToArrayIndex,
    SwapCondition,
    Transform,
)


class SingleAugMetadata(BaseModel):
    kernel_name: str
    parallel_api: str = "serial"
    is_original: bool = False
    transforms_applied: List[str] = Field(default_factory=list)
    sampling_mode: str = "single_transform"
    seed: int
    applicable_transforms: List[str] = Field(default_factory=list)
    selected_transform: Optional[str] = None


@contextmanager
def deterministic_random():
    """
    Force transform.apply() to take its internal random branches so we reliably apply ONCE.
    Temporarily patches random.random and random.choice.
    """
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


def load_kernels(kernels_file: Path) -> List[str]:
    with kernels_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def primary_cpp_file(kernel_dir: Path) -> Path:
    cpp_files = sorted(kernel_dir.glob("*.cpp"))
    if not cpp_files:
        raise FileNotFoundError(f"No .cpp files found in {kernel_dir}")
    for p in cpp_files:
        if p.name == "main.cpp":
            return p
    return cpp_files[0]


def available_transforms() -> List[Transform]:
    return [
        ArithmeticTransform(),
        SwapCondition(),
        PointerArithmeticToArrayIndex(),
        ChangeNames(),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate single-augmentation serial dataset"
    )
    ap.add_argument(
        "--kernels-file", type=Path, required=True,
        help="Text file with one kernel name per line",
    )
    ap.add_argument(
        "--src-root", type=Path, required=True,
        help="Root of the original serial source tree (contains <kernel>-serial/ dirs)",
    )
    ap.add_argument(
        "--out-root", type=Path, required=True,
        help="Output root directory for augmented sources",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    kernels = load_kernels(args.kernels_file)
    if not kernels:
        raise ValueError("kernels list is empty")

    src_root: Path = args.src_root
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    index = ci.Index.create()
    transforms = available_transforms()

    for k in kernels:
        src_dir = src_root / f"{k}-serial"
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source dir: {src_dir}")

        dst_dir = out_root / f"{k}-serial"
        if dst_dir.exists():
            raise FileExistsError(
                f"Output already exists: {dst_dir} (delete it to regenerate)"
            )
        dst_dir.mkdir(parents=True, exist_ok=False)

        for p in src_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, dst_dir / p.name)

        cpp_path = primary_cpp_file(src_dir)
        code = cpp_path.read_text(encoding="utf-8", errors="replace")

        applicable = [t for t in transforms if t.is_applicable(code, index)]
        applicable_names = [t.name for t in applicable]

        rng = random.Random(args.seed + (hash(k) % 10_000))
        selected = rng.choice(applicable) if applicable else None

        applied_name: Optional[str] = None
        if selected is not None:
            with deterministic_random():
                tr = selected.apply(code, index)
            if tr.applied:
                (dst_dir / cpp_path.name).write_text(tr.code, encoding="utf-8")
                applied_name = selected.name

        meta = SingleAugMetadata(
            kernel_name=k,
            seed=args.seed,
            applicable_transforms=applicable_names,
            selected_transform=selected.name if selected is not None else None,
            transforms_applied=[applied_name] if applied_name else [],
        )
        (dst_dir / "augmentation_metadata.json").write_text(
            meta.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    print(f"Wrote single-augmentation dataset to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
