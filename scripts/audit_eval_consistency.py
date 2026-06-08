#!/usr/bin/env python3
"""
Audit 708 Qwen 3.5 eval result JSONs for pipeline consistency.

Checks:
1. attempts[] → top-level sync (last attempt must match top-level statuses/snippets)
2. total_attempts == len(attempts)
3. Self-repair logic (multi-attempt: first attempt failed, error_feedback_sent non-null)
4. max_retries consistency (1 for canonical, check augmentation)
5. sample_id vs filename suffix (s0→0, s1→1, s2→2; L1-L4 → augment_level 1-4)
6. Seed uniqueness per kernel pair (canonical s0/s1/s2 should differ)
7. Temperature consistency (all 0.7)
8. num_samples (3 for canonical, 1 for augmentation)
9. thinking_enabled (true for all)
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("./results/evaluation/together-qwen-3.5-397b-a17b")

def main():
    issues = defaultdict(list)  # category -> list of (filename, detail)
    total_checked = 0
    canonical_count = 0
    augmentation_count = 0
    multi_attempt_count = 0

    # For seed uniqueness check: group canonical files by kernel pair
    # pair_key = (source_spec, target_spec) -> {sample_id: seed}
    pair_seeds = defaultdict(dict)

    # Collect all JSON files
    json_files = sorted(RESULTS_DIR.glob("*.json"))

    for fpath in json_files:
        fname = fpath.name
        total_checked += 1

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            issues["json_parse_error"].append((fname, str(e)))
            continue

        # Determine file type from suffix
        suffix_match = re.search(r'-(s(\d+)|L(\d+))\.json$', fname)
        if not suffix_match:
            issues["filename_format"].append((fname, "Cannot parse suffix (expected -s0/-s1/-s2/-L1..L4)"))
            continue

        is_canonical = suffix_match.group(2) is not None
        if is_canonical:
            canonical_count += 1
            expected_sample_id = int(suffix_match.group(2))
        else:
            augmentation_count += 1
            expected_augment_level = int(suffix_match.group(3))

        attempts = data.get("attempts", [])
        total_attempts = data.get("total_attempts")

        # ── CHECK 2: total_attempts == len(attempts) ──
        if total_attempts != len(attempts):
            issues["total_attempts_mismatch"].append(
                (fname, f"total_attempts={total_attempts} but len(attempts)={len(attempts)}")
            )

        # ── CHECK 1: attempts[] → top-level sync ──
        if len(attempts) > 0:
            last = attempts[-1]

            # Fields that should sync between last attempt and top-level
            sync_fields = [
                ("build_status", "build_status"),
                ("run_status", "run_status"),
                ("verify_status", "verify_status"),
                ("build_error_snippet", "build_error_snippet"),
                ("run_stderr_snippet", "run_stderr_snippet"),
            ]

            for top_field, attempt_field in sync_fields:
                top_val = data.get(top_field)
                attempt_val = last.get(attempt_field)

                # Normalize: the harness uses "fail"/"success" in attempts,
                # and the same in top-level for build_status/run_status/verify_status
                if top_val != attempt_val:
                    issues["attempt_toplevel_sync"].append(
                        (fname, f"{top_field}: top-level={repr(top_val)[:80]} vs last_attempt={repr(attempt_val)[:80]}")
                    )

        # ── CHECK 3: Self-repair logic ──
        if len(attempts) > 1:
            multi_attempt_count += 1
            first = attempts[0]
            first_failed = (
                first.get("build_status") == "fail" or
                first.get("run_status") == "fail" or
                first.get("verify_status") == "fail"
            )
            if not first_failed:
                issues["self_repair_first_not_failed"].append(
                    (fname, f"First attempt didn't fail: build={first.get('build_status')}, "
                            f"run={first.get('run_status')}, verify={first.get('verify_status')}")
                )

            if first.get("error_feedback_sent") is None:
                issues["self_repair_no_feedback"].append(
                    (fname, "First attempt has error_feedback_sent=null despite retry")
                )

        # ── CHECK 4: max_retries consistency ──
        max_retries = data.get("max_retries")
        if is_canonical:
            if max_retries != 1:
                issues["max_retries_canonical"].append(
                    (fname, f"Expected max_retries=1 for canonical, got {max_retries}")
                )
        else:
            if max_retries != 1:
                issues["max_retries_augmentation"].append(
                    (fname, f"max_retries={max_retries} for augmentation (expected 1)")
                )

        # ── CHECK 5: sample_id vs filename / augment_level vs filename ──
        if is_canonical:
            actual_sample_id = data.get("sample_id")
            if actual_sample_id != expected_sample_id:
                issues["sample_id_mismatch"].append(
                    (fname, f"Expected sample_id={expected_sample_id} from suffix, got {actual_sample_id}")
                )
            actual_augment = data.get("augment_level")
            if actual_augment != 0:
                issues["canonical_augment_level"].append(
                    (fname, f"Canonical file has augment_level={actual_augment}, expected 0")
                )
        else:
            actual_augment = data.get("augment_level")
            if actual_augment != expected_augment_level:
                issues["augment_level_mismatch"].append(
                    (fname, f"Expected augment_level={expected_augment_level} from suffix, got {actual_augment}")
                )

        # ── CHECK 6: Seed collection for uniqueness check ──
        if is_canonical:
            pair_key = (data.get("source_spec"), data.get("target_spec"))
            seed = data.get("seed")
            sample_id = data.get("sample_id")
            pair_seeds[pair_key][sample_id] = seed

        # ── CHECK 7: Temperature consistency ──
        temp = data.get("temperature")
        if temp != 0.7:
            issues["temperature_wrong"].append(
                (fname, f"Expected temperature=0.7, got {temp}")
            )

        # ── CHECK 8: num_samples ──
        num_samples = data.get("num_samples")
        if is_canonical:
            if num_samples != 3:
                issues["num_samples_canonical"].append(
                    (fname, f"Expected num_samples=3 for canonical, got {num_samples}")
                )
        else:
            if num_samples != 1:
                issues["num_samples_augmentation"].append(
                    (fname, f"Expected num_samples=1 for augmentation, got {num_samples}")
                )

        # ── CHECK 9: thinking_enabled ──
        thinking = data.get("thinking_enabled")
        if thinking is not True:
            issues["thinking_disabled"].append(
                (fname, f"Expected thinking_enabled=true, got {thinking}")
            )

    # ── CHECK 6 (continued): Seed uniqueness across canonical samples per pair ──
    seed_dup_pairs = 0
    for pair_key, seeds_by_sample in pair_seeds.items():
        seed_values = list(seeds_by_sample.values())
        if len(seed_values) != len(set(seed_values)):
            seed_dup_pairs += 1
            issues["seed_duplicate_in_pair"].append(
                (f"{pair_key[0]}→{pair_key[1]}",
                 f"Seeds: {seeds_by_sample}")
            )

    # ── Additional: check for error_feedback_sent on single-attempt failures ──
    # When max_retries=1 and total_attempts=1 and build_status=fail,
    # the harness should NOT have retried (correct behavior). But let's verify
    # that error_feedback_sent is null on the single attempt (no feedback was sent
    # because no retry happened).
    single_attempt_with_feedback = 0
    for fpath in json_files:
        try:
            data = json.load(open(fpath))
        except:
            continue
        attempts = data.get("attempts", [])
        if len(attempts) == 1 and attempts[0].get("error_feedback_sent") is not None:
            single_attempt_with_feedback += 1
            issues["single_attempt_has_feedback"].append(
                (fpath.name, f"error_feedback_sent={repr(attempts[0].get('error_feedback_sent'))[:80]}")
            )

    # ── REPORT ──
    print("=" * 80)
    print("PIPELINE CONSISTENCY AUDIT — Qwen 3.5 397B Eval Results")
    print("=" * 80)
    print(f"\nDirectory: {RESULTS_DIR}")
    print(f"Total files checked: {total_checked}")
    print(f"  Canonical (s0/s1/s2): {canonical_count}")
    print(f"  Augmentation (L1-L4): {augmentation_count}")
    print(f"  Multi-attempt files: {multi_attempt_count}")
    print(f"  Unique canonical pairs checked for seed uniqueness: {len(pair_seeds)}")

    print(f"\n{'─' * 80}")
    print("RESULTS BY CHECK")
    print(f"{'─' * 80}")

    check_descriptions = {
        "json_parse_error": "JSON parse errors",
        "filename_format": "Filename format issues",
        "total_attempts_mismatch": "CHECK 2: total_attempts != len(attempts)",
        "attempt_toplevel_sync": "CHECK 1: Last attempt != top-level fields",
        "self_repair_first_not_failed": "CHECK 3: Self-repair first attempt didn't fail",
        "self_repair_no_feedback": "CHECK 3: Self-repair no error_feedback_sent",
        "max_retries_canonical": "CHECK 4: max_retries != 1 (canonical)",
        "max_retries_augmentation": "CHECK 4: max_retries != 1 (augmentation)",
        "sample_id_mismatch": "CHECK 5: sample_id doesn't match filename suffix",
        "canonical_augment_level": "CHECK 5: Canonical file has non-zero augment_level",
        "augment_level_mismatch": "CHECK 5: Augmentation augment_level doesn't match suffix",
        "seed_duplicate_in_pair": "CHECK 6: Duplicate seeds within a canonical pair",
        "temperature_wrong": "CHECK 7: Temperature != 0.7",
        "num_samples_canonical": "CHECK 8: num_samples != 3 (canonical)",
        "num_samples_augmentation": "CHECK 8: num_samples != 1 (augmentation)",
        "thinking_disabled": "CHECK 9: thinking_enabled != true",
        "single_attempt_has_feedback": "BONUS: Single attempt with error_feedback_sent",
    }

    all_checks = [
        "json_parse_error", "filename_format",
        "attempt_toplevel_sync", "total_attempts_mismatch",
        "self_repair_first_not_failed", "self_repair_no_feedback",
        "max_retries_canonical", "max_retries_augmentation",
        "sample_id_mismatch", "canonical_augment_level", "augment_level_mismatch",
        "seed_duplicate_in_pair",
        "temperature_wrong",
        "num_samples_canonical", "num_samples_augmentation",
        "thinking_disabled",
        "single_attempt_has_feedback",
    ]

    total_issues = 0
    for check_key in all_checks:
        desc = check_descriptions.get(check_key, check_key)
        issue_list = issues.get(check_key, [])
        count = len(issue_list)
        total_issues += count
        status = "PASS" if count == 0 else "FAIL"
        marker = "  " if count == 0 else "!!"
        print(f"\n{marker} [{status}] {desc}: {count} issues")

        if count > 0:
            # Show up to 5 examples
            for fname, detail in issue_list[:5]:
                print(f"      {fname}: {detail}")
            if count > 5:
                print(f"      ... and {count - 5} more")

    print(f"\n{'─' * 80}")
    print("SUMMARY")
    print(f"{'─' * 80}")

    checks_passed = sum(1 for k in all_checks if len(issues.get(k, [])) == 0)
    checks_total = len(all_checks)

    if total_issues == 0:
        print(f"\n  ALL {checks_total} CHECKS PASSED — 0 issues across {total_checked} files")
        print("  Pipeline operated correctly.")
    else:
        print(f"\n  {checks_passed}/{checks_total} checks passed, {total_issues} total issues found")

        # Determine severity
        critical_keys = [
            "attempt_toplevel_sync", "total_attempts_mismatch",
            "sample_id_mismatch", "augment_level_mismatch",
            "temperature_wrong", "thinking_disabled",
        ]
        critical_issues = sum(len(issues.get(k, [])) for k in critical_keys)
        if critical_issues > 0:
            print(f"  {critical_issues} CRITICAL issues (data integrity at risk)")
        else:
            print("  No critical issues — findings are informational or minor")

    # ── Extra statistics ──
    print(f"\n{'─' * 80}")
    print("SUPPLEMENTAL STATISTICS")
    print(f"{'─' * 80}")

    # Outcome distribution
    outcomes = defaultdict(int)
    for fpath in json_files:
        try:
            data = json.load(open(fpath))
            outcomes[data.get("overall_status", "UNKNOWN")] += 1
        except:
            pass

    print("\n  Overall status distribution:")
    for status in sorted(outcomes):
        print(f"    {status}: {outcomes[status]}")

    # Self-repair summary
    print(f"\n  Self-repair usage: {multi_attempt_count}/{total_checked} files had >1 attempt")
    if multi_attempt_count == 0:
        # Check if any first-attempt failures exist (they should have triggered retry)
        first_attempt_failures = 0
        for fpath in json_files:
            try:
                data = json.load(open(fpath))
                attempts = data.get("attempts", [])
                if len(attempts) == 1:
                    a = attempts[0]
                    if a.get("build_status") == "fail" or a.get("run_status") == "fail" or a.get("verify_status") == "fail":
                        first_attempt_failures += 1
            except:
                pass
        print(f"  Single-attempt failures: {first_attempt_failures}")
        if first_attempt_failures > 0:
            print("  EXPECTED: max_retries=1 means '1 total attempt' (zero-shot, no retry).")
            print("  See llm_evaluate.py:1471 docstring and line 1618: range(1, max_retries+1).")
            print("  All failures were correctly recorded without retry. Pipeline behaved as designed.")

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
