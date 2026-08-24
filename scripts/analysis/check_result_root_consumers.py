#!/usr/bin/env python3
"""Inventory every consumer of the literal ``results/evaluation`` (Task 5).

Scans Python and shell files under ``--scripts-root`` for the literal
``results/evaluation`` (including the ``"results" / "evaluation"`` path-join
spelling) and checks them against the INVENTORY below. The gate fails when:

  1. a scanned file contains the literal but is not inventoried (a new,
     unvetted consumer appeared), or
  2. an inventoried file no longer contains the literal (stale entry), or
  3. with ``--require-explicit-final-root``: a ``final`` consumer does not
     declare its explicit input-root flag (the flag string must appear in
     the file).

Categories:
  final              Final-analysis/release consumer. MUST accept an explicit
                     input root; any retained historical default is annotated
                     in the script and covered by a compatibility test
                     (tests/test_analysis_result_roots.py). The final-evidence
                     orchestrator always passes the explicit root.
  final-task6        (retired) Final consumer whose explicit-root flag was
                     pending plan Task 6; sloc_analysis gained --results-root
                     there and is now categorized final. No entries remain.
  eval-pipeline      Writer/reader of the live eval pipeline; the literal IS
                     its contract (it produces the immutable corpus).
  guard              Provenance/immutability tooling whose job is to point at
                     results/evaluation (inventories, archives, seals, hooks
                     support, replay input default documentation).
  rebuttal           Frozen NeurIPS #1830 evidence chain; reads the submitted
                     corpus by design. Do not retrofit.
  test               Test files; fixtures reference the literal.
  archived           Dead scripts under archive dirs; never run.

Usage:
    python3 scripts/analysis/check_result_root_consumers.py \\
      --scripts-root scripts \\
      --require-explicit-final-root
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LITERAL_RE = re.compile(
    r"results/evaluation"
    r"|\"results\"\s*/\s*\"evaluation\""
    r"|'results'\s*/\s*'evaluation'"
)

# Repo-relative path -> (category, explicit-root flag or None).
INVENTORY: dict[str, tuple[str, str | None]] = {
    # --- final analysis consumers (Task 5 explicit-root contract) ---
    "scripts/analysis/statistical_analysis.py": ("final", "--results-root"),
    "scripts/analysis/quantitative_findings.py": ("final", "--results-root"),
    "scripts/analysis/augmentation_analysis.py": ("final", "--results-root"),
    "scripts/analysis/build_error_taxonomy.py": ("final", "--results-root"),
    "scripts/analysis/token_analysis.py": ("final", "--results-root"),
    "scripts/analysis/generate_paper_data.py": ("final", "--results-dir"),
    "scripts/generate_paper_figures.py": ("final", "--results-root"),
    "scripts/analysis/sloc_analysis.py": ("final", "--results-root"),
    # E-19 generator (Task 6): reads the corpus, emits the suite-composition
    # bound and claim status consumed by Tasks 13/14.
    "scripts/rebuttal/suite_composition.py": ("final", "--results-root"),
    # Task 13 final-evidence orchestrator: reads the submitted corpus for the
    # old column of the old-vs-new comparison (results/evaluation is its
    # --submitted-root default) and always passes an explicit root (gate13
    # finding 3).
    "scripts/analysis/generate_final_evidence.py": ("final", "--submitted-root"),
    # --- live eval pipeline (the literal is its output contract) ---
    "scripts/evaluation/llm_evaluate.py": ("eval-pipeline", None),
    "scripts/evaluation/run_eval_batch.py": ("eval-pipeline", None),
    "scripts/evaluation/analyze_eval.py": ("eval-pipeline", None),
    "scripts/evaluation/derive_l0_passers.py": ("eval-pipeline", None),
    "scripts/evaluation/reverify_pass_results.py": ("eval-pipeline", None),
    "scripts/analysis/classify_translation_pairs.py": ("eval-pipeline", None),
    "scripts/batch/run_phase3.sh": ("eval-pipeline", None),
    # --- provenance / immutability / replay tooling ---
    "scripts/evaluation/replay_stored_translations.py": ("guard", "--input-root"),
    "scripts/provenance/seal_result_namespace.py": ("guard", None),
    "scripts/provenance/inventory_immutable_inputs.py": ("guard", None),
    "scripts/provenance/make_private_archive.py": ("guard", None),
    "scripts/audit_eval_consistency.py": ("guard", None),
    "scripts/audit_eval_schema.py": ("guard", None),
    "scripts/build_artifact.sh": ("guard", None),
    "scripts/build_artifact_zip.sh": ("guard", None),
    "scripts/analysis/check_result_root_consumers.py": ("guard", None),
    # --- frozen rebuttal evidence chain (submitted corpus by design) ---
    "scripts/rebuttal/export_translations.py": ("rebuttal", None),
    "scripts/rebuttal/fullsample_asymmetry.py": ("rebuttal", None),
    "scripts/rebuttal/generate_rebuttal_tables.py": ("rebuttal", None),
    "scripts/rebuttal/gpu_probe.py": ("rebuttal", None),
    "scripts/rebuttal/list_weak_oracle_passes.py": ("rebuttal", None),
    "scripts/rebuttal/lud_before_after.py": ("rebuttal", None),
    "scripts/rebuttal/oracle_restricted_analysis.py": ("rebuttal", None),
    "scripts/rebuttal/oracle_shape_census.py": ("rebuttal", None),
    "scripts/rebuttal/reverify_existing_results.py": ("rebuttal", None),
    "scripts/rebuttal/run_layer_a_gpu.sh": ("rebuttal", None),
    # --- tests ---
    "scripts/analysis/test_augmentation_analysis.py": ("test", None),
    "scripts/analysis/test_generate_paper_data.py": ("test", None),
    "scripts/analysis/test_token_analysis.py": ("test", None),
    # (the three "archived, never run" batch scripts moved out of scripts/ to
    # docs/archive/2026-08-dead-scripts/ on 2026-08-20; the scan no longer sees them)
    # --- added 2026-08-20 (were uninventoried) ---
    "scripts/check_public_snapshot.py": ("guard", None),
    "scripts/analysis/emit_release_inventory.py": ("guard", None),
    # The figure-fragment pair reads the SEALED replay results/evaluation_final/
    # parbench-final-v1 — a frozen namespace fixed by design, documented in their
    # docstrings; the literal matches inside that longer path.
    "scripts/emit_figure_fragments.py": ("guard", None),
    "scripts/verify_figure_fragments.py": ("guard", None),
    "scripts/evaluation/test_generate_paper_figures.py": ("test", None),
}


def scan(scripts_root: Path) -> set[str]:
    """Repo-relative paths of files containing the literal."""
    repo_root = scripts_root.resolve().parent
    hits: set[str] = set()
    for pattern in ("**/*.py", "**/*.sh"):
        for path in sorted(scripts_root.resolve().glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if LITERAL_RE.search(text):
                hits.add(path.relative_to(repo_root).as_posix())
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-root", type=Path, default=Path("scripts"))
    parser.add_argument("--require-explicit-final-root", action="store_true",
                        help="Fail unless every 'final' consumer declares its "
                             "explicit input-root flag")
    args = parser.parse_args(argv)

    scripts_root = args.scripts_root.resolve()
    repo_root = scripts_root.parent
    hits = scan(args.scripts_root)

    problems: list[str] = []
    notes: list[str] = []

    for rel in sorted(hits - set(INVENTORY)):
        problems.append(
            f"uninventoried consumer of literal results/evaluation: {rel} "
            "(add it to INVENTORY in check_result_root_consumers.py with a "
            "category and, for final consumers, an explicit-root flag)"
        )
    for rel in sorted(set(INVENTORY) - hits):
        problems.append(
            f"stale inventory entry (literal no longer present): {rel}"
        )

    for rel in sorted(set(INVENTORY) & hits):
        category, flag = INVENTORY[rel]
        if category == "final":
            text = (repo_root / rel).read_text(encoding="utf-8",
                                               errors="replace")
            if not flag:
                problems.append(f"final consumer with no declared flag: {rel}")
            elif args.require_explicit_final_root and flag not in text:
                problems.append(
                    f"final consumer {rel} does not declare its explicit "
                    f"input-root flag {flag}"
                )
        elif category == "final-task6":
            notes.append(
                f"final-task6 (explicit root arrives with Task 6): {rel}"
            )

    by_category: dict[str, int] = {}
    for rel in sorted(set(INVENTORY) & hits):
        cat = INVENTORY[rel][0]
        by_category[cat] = by_category.get(cat, 0) + 1

    print(f"Scanned {args.scripts_root}: {len(hits)} consumer(s) of the "
          "literal results/evaluation")
    for cat in sorted(by_category):
        print(f"  {cat}: {by_category[cat]}")
    for n in notes:
        print(f"NOTE  {n}")
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        print(f"CHECK FAILED: {len(problems)} problem(s)")
        return 1
    print("CHECK OK: every consumer is inventoried"
          + (" and every final consumer declares an explicit input root"
             if args.require_explicit_final_root else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
