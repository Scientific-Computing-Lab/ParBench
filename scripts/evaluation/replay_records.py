"""Shared normalization for stored-translation replay records (Task 8 format).

Replay records (``replay_kind == "stored_translation_replay"``) carry their
identity - model, source_spec, target_spec, augment_level, sample_id - under
the ``parent`` block, because the replay made no model call of its own.
Analysis loaders expect those fields at the top level; without promotion every
sealed record classifies as unknown model / unknown direction and silently
escapes the spec-eligibility filters. Promotion happens in memory at load
time; the sealed record files themselves are never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

# Identity fields promoted from parent. parent["overall_status"] is the OLD
# (pre-replay) status and must never overwrite the record's own top-level
# overall_status, so it is deliberately absent from this tuple.
PARENT_IDENTITY_FIELDS = (
    "model",
    "source_spec",
    "target_spec",
    "augment_level",
    "sample_id",
)

# Sampling/config fields that are properties of the original model call and are
# therefore INVARIANT under stored-source replay. The Task 8 replay records do
# not store them (no model call happened), so the canonical-vs-legacy split
# (temperature) and pass@k reconstruction (num_samples) would otherwise fail
# over the sealed namespace. They are recovered read-only from the referenced
# submitted record; the sealed files are never rewritten.
CONFIG_FIELDS = (
    "temperature",
    "thinking_enabled",
    "num_samples",
    "seed",
    "top_p",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def promote_parent_metadata(record: dict) -> dict:
    """Fill missing top-level identity fields from ``record["parent"]``, in place.

    Returns the same dict for call-chaining. A field is only promoted when the
    top-level value is absent, None, or "" - an existing 0 (a real
    augment_level / sample_id) is kept. For a stored-translation replay record
    that is missing invariant sampling config, the config is additionally
    backfilled read-only from the referenced submitted record (its
    ``input_record.path``); a non-replay record, or one whose input record is
    unreadable, is left untouched.
    """
    parent = record.get("parent")
    if isinstance(parent, dict):
        for key in PARENT_IDENTITY_FIELDS:
            if record.get(key) in (None, "") and parent.get(key) not in (None, ""):
                record[key] = parent[key]
    _backfill_config_from_input(record)
    return record


def _backfill_config_from_input(record: dict) -> None:
    if record.get("replay_kind") != "stored_translation_replay":
        return
    if all(record.get(k) is not None for k in CONFIG_FIELDS):
        return
    input_record = record.get("input_record")
    if not isinstance(input_record, dict):
        return
    rel = input_record.get("path")
    if not rel:
        return
    for base in (Path.cwd(), _REPO_ROOT):
        candidate = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        try:
            src = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in CONFIG_FIELDS:
            if record.get(key) is None and src.get(key) is not None:
                record[key] = src[key]
        return
