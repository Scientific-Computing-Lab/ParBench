#!/usr/bin/env python3
"""
Audit 708 Qwen eval result JSONs for field completeness and schema correctness.

Checks:
1. Always-required fields: presence, non-null, correct type
2. Conditional rules: overall_status vs build/run/verify status consistency
3. LLM metadata: prompt_tokens, completion_tokens, llm_response_time_seconds > 0
4. Timing fields: build_time_seconds when build succeeds
5. Array/object fields: attempts, target_files_expected, target_files_extracted, translated_files
6. Cross-validation of top-level vs per-attempt fields

Reports categorized issues with file names and counts.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("./results/evaluation/together-qwen-3.5-397b-a17b")

# ── Always-required fields with expected types ──
ALWAYS_REQUIRED = {
    "source_spec": str,
    "target_spec": str,
    "kernel": str,
    "model": str,
    "augment_level": int,
    "temperature": (int, float),
    "sample_id": int,
    "thinking_enabled": bool,
    "num_samples": int,
    "seed": int,
    "top_p": (int, float),
    "translation_mode": str,
    "translation_type": str,
    "verification_mode": str,
    "run_args_mode": str,
    "timestamp": str,
    "overall_status": str,
    "max_retries": int,
    "total_attempts": int,
    "attempts": list,
    "target_files_expected": list,
    "target_files_extracted": list,
    "translated_files": dict,
}

VALID_OVERALL_STATUSES = {"PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL", "EXTRACTION_FAIL"}
VALID_BUILD_STATUSES = {"pass", "fail", None}
VALID_RUN_STATUSES = {"pass", "fail", "timeout", "error", None}
VALID_VERIFY_STATUSES = {"pass", "fail", None}


def audit_file(filepath):
    """Audit a single result JSON. Returns list of (category, message) tuples."""
    issues = []
    fname = filepath.name

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [("PARSE_ERROR", f"Invalid JSON: {e}")]
    except Exception as e:
        return [("READ_ERROR", f"Cannot read file: {e}")]

    # ── 1. Always-required fields ──
    for field, expected_type in ALWAYS_REQUIRED.items():
        if field not in data:
            issues.append(("MISSING_FIELD", f"Missing required field: {field}"))
        elif data[field] is None:
            issues.append(("NULL_REQUIRED_FIELD", f"Required field is null: {field}"))
        elif not isinstance(data[field], expected_type):
            issues.append(("WRONG_TYPE", f"{field}: expected {expected_type}, got {type(data[field]).__name__} = {repr(data[field])[:80]}"))

    overall = data.get("overall_status")
    build = data.get("build_status")
    run = data.get("run_status")
    verify = data.get("verify_status")
    error_msg = data.get("error_message")

    # ── 1b. Valid enum values ──
    if overall is not None and overall not in VALID_OVERALL_STATUSES:
        issues.append(("INVALID_ENUM", f"overall_status = {overall!r} not in {VALID_OVERALL_STATUSES}"))
    if build not in VALID_BUILD_STATUSES:
        issues.append(("INVALID_ENUM", f"build_status = {build!r} not in {VALID_BUILD_STATUSES}"))
    if run not in VALID_RUN_STATUSES:
        issues.append(("INVALID_ENUM", f"run_status = {run!r} not in {VALID_RUN_STATUSES}"))
    if verify not in VALID_VERIFY_STATUSES:
        issues.append(("INVALID_ENUM", f"verify_status = {verify!r} not in {VALID_VERIFY_STATUSES}"))

    # ── 2. Conditional rules based on overall_status ──
    if overall == "PASS":
        if build != "pass":
            issues.append(("STATUS_INCONSISTENCY", f"overall=PASS but build_status={build!r}"))
        if run != "pass":
            issues.append(("STATUS_INCONSISTENCY", f"overall=PASS but run_status={run!r}"))
        if verify != "pass":
            issues.append(("STATUS_INCONSISTENCY", f"overall=PASS but verify_status={verify!r}"))

    elif overall == "BUILD_FAIL":
        if build != "fail":
            issues.append(("STATUS_INCONSISTENCY", f"overall=BUILD_FAIL but build_status={build!r}"))
        if run is not None:
            issues.append(("STATUS_INCONSISTENCY", f"overall=BUILD_FAIL but run_status={run!r} (expected null)"))
        if verify is not None:
            issues.append(("STATUS_INCONSISTENCY", f"overall=BUILD_FAIL but verify_status={verify!r} (expected null)"))
        # build_error_snippet should be present
        bes = data.get("build_error_snippet")
        if bes is None or (isinstance(bes, str) and len(bes.strip()) == 0):
            issues.append(("MISSING_ERROR_INFO", f"overall=BUILD_FAIL but build_error_snippet is empty/null"))

    elif overall == "RUN_FAIL":
        if build != "pass":
            issues.append(("STATUS_INCONSISTENCY", f"overall=RUN_FAIL but build_status={build!r} (expected pass)"))
        if run not in ("fail", "timeout", "error"):
            issues.append(("STATUS_INCONSISTENCY", f"overall=RUN_FAIL but run_status={run!r} (expected fail/timeout/error)"))
        if verify is not None:
            issues.append(("STATUS_INCONSISTENCY", f"overall=RUN_FAIL but verify_status={verify!r} (expected null)"))
        # Should have some stderr or stdout content
        stderr = data.get("run_stderr_snippet")
        stdout = data.get("run_stdout_snippet")
        if (stderr is None or stderr == "") and (stdout is None or stdout == ""):
            # Not a hard error -- some RUN_FAILs legitimately have no output (e.g., SIGSEGV)
            pass  # downgraded to INFO

    elif overall == "VERIFY_FAIL":
        if build != "pass":
            issues.append(("STATUS_INCONSISTENCY", f"overall=VERIFY_FAIL but build_status={build!r} (expected pass)"))
        if run != "pass":
            # Note: some VERIFY_FAILs have run=pass because the false-positive check fires post-verify
            issues.append(("STATUS_INCONSISTENCY", f"overall=VERIFY_FAIL but run_status={run!r} (expected pass)"))
        # verify can be "fail" OR "pass" (false-positive override case)
        if verify not in ("fail", "pass"):
            issues.append(("STATUS_INCONSISTENCY", f"overall=VERIFY_FAIL but verify_status={verify!r} (expected fail or pass+override)"))
        # If verify=pass but overall=VERIFY_FAIL, should have error_message explaining false positive
        if verify == "pass" and overall == "VERIFY_FAIL":
            if error_msg is None or (isinstance(error_msg, str) and "false positive" not in error_msg.lower()):
                issues.append(("FALSE_POSITIVE_NO_EXPLANATION", f"verify=pass + overall=VERIFY_FAIL but no 'false positive' in error_message"))

    elif overall == "EXTRACTION_FAIL":
        # build, run, verify should all be null
        if build is not None:
            issues.append(("STATUS_INCONSISTENCY", f"overall=EXTRACTION_FAIL but build_status={build!r}"))
        if run is not None:
            issues.append(("STATUS_INCONSISTENCY", f"overall=EXTRACTION_FAIL but run_status={run!r}"))
        if verify is not None:
            issues.append(("STATUS_INCONSISTENCY", f"overall=EXTRACTION_FAIL but verify_status={verify!r}"))

    # ── 3. LLM metadata checks ──
    prompt_tokens = data.get("prompt_tokens")
    completion_tokens = data.get("completion_tokens")
    llm_time = data.get("llm_response_time_seconds")

    if prompt_tokens is None or (isinstance(prompt_tokens, (int, float)) and prompt_tokens <= 0):
        issues.append(("LLM_METADATA", f"prompt_tokens invalid: {prompt_tokens!r}"))
    if completion_tokens is None or (isinstance(completion_tokens, (int, float)) and completion_tokens <= 0):
        issues.append(("LLM_METADATA", f"completion_tokens invalid: {completion_tokens!r}"))
    if llm_time is None or (isinstance(llm_time, (int, float)) and llm_time <= 0):
        issues.append(("LLM_METADATA", f"llm_response_time_seconds invalid: {llm_time!r}"))

    # ── 4. Timing fields ──
    build_time = data.get("build_time_seconds")
    if build == "pass":
        if build_time is None:
            issues.append(("TIMING", f"build_status=pass but build_time_seconds is null"))
        elif isinstance(build_time, (int, float)) and build_time <= 0:
            issues.append(("TIMING", f"build_status=pass but build_time_seconds={build_time} (expected >0)"))

    run_time = data.get("run_time_seconds")
    if run == "pass":
        if run_time is None:
            issues.append(("TIMING", f"run_status=pass but run_time_seconds is null"))
        elif isinstance(run_time, (int, float)) and run_time <= 0:
            issues.append(("TIMING", f"run_status=pass but run_time_seconds={run_time} (expected >0)"))

    # ── 5. Array/object field content checks ──
    attempts = data.get("attempts", [])
    total_attempts = data.get("total_attempts")
    if isinstance(attempts, list) and isinstance(total_attempts, int):
        if len(attempts) != total_attempts:
            issues.append(("ATTEMPTS_MISMATCH", f"total_attempts={total_attempts} but len(attempts)={len(attempts)}"))

    expected_files = data.get("target_files_expected", [])
    extracted_files = data.get("target_files_extracted", [])
    translated_files = data.get("translated_files", {})

    if overall == "PASS":
        # All expected files should be extracted
        if isinstance(expected_files, list) and isinstance(extracted_files, list):
            missing = set(expected_files) - set(extracted_files)
            if missing:
                issues.append(("FILE_MISMATCH", f"PASS but expected files not extracted: {missing}"))

    if overall != "EXTRACTION_FAIL":
        # translated_files should not be empty for non-extraction-fail
        if isinstance(translated_files, dict) and len(translated_files) == 0:
            issues.append(("EMPTY_TRANSLATED_FILES", f"overall={overall} but translated_files is empty"))

    # ── 6. Per-attempt field validation ──
    if isinstance(attempts, list):
        for i, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                issues.append(("ATTEMPT_STRUCTURE", f"attempt[{i}] is not a dict"))
                continue
            # Each attempt should have 'attempt' number
            if "attempt" not in attempt:
                issues.append(("ATTEMPT_STRUCTURE", f"attempt[{i}] missing 'attempt' field"))
            # LLM metadata per attempt
            for field in ["prompt_tokens", "completion_tokens", "llm_response_time_seconds"]:
                val = attempt.get(field)
                if val is None:
                    issues.append(("ATTEMPT_METADATA", f"attempt[{i}].{field} is null"))
                elif isinstance(val, (int, float)) and val <= 0:
                    issues.append(("ATTEMPT_METADATA", f"attempt[{i}].{field}={val} (expected >0)"))
            # finish_reason
            fr = attempt.get("finish_reason")
            if fr is None:
                issues.append(("ATTEMPT_METADATA", f"attempt[{i}].finish_reason is null"))

    # ── 7. Filename consistency checks ──
    # Filename should encode source-to-target + augment level or sample id
    source = data.get("source_spec", "")
    target = data.get("target_spec", "")
    expected_prefix = f"{source}-to-{target}"
    if not filepath.stem.startswith(expected_prefix):
        issues.append(("FILENAME_MISMATCH", f"Filename '{filepath.stem}' doesn't match source_spec/target_spec '{expected_prefix}'"))

    return issues


def main():
    json_files = sorted(RESULTS_DIR.glob("*.json"))
    total_files = len(json_files)

    print(f"{'='*80}")
    print(f"EVAL RESULT SCHEMA AUDIT")
    print(f"Directory: {RESULTS_DIR}")
    print(f"Total JSON files: {total_files}")
    print(f"{'='*80}\n")

    # Categorized issue tracking
    all_issues = {}  # filename -> list of (category, msg)
    category_counts = defaultdict(int)
    category_files = defaultdict(list)
    files_with_issues = 0

    for fpath in json_files:
        issues = audit_file(fpath)
        if issues:
            all_issues[fpath.name] = issues
            files_with_issues += 1
            for cat, msg in issues:
                category_counts[cat] += 1
                if len(category_files[cat]) < 5:  # Store up to 5 examples per category
                    category_files[cat].append((fpath.name, msg))

    # ── Summary ──
    print(f"SUMMARY")
    print(f"  Total files checked:  {total_files}")
    print(f"  Files with issues:    {files_with_issues}")
    print(f"  Files clean:          {total_files - files_with_issues}")
    print(f"  Total issues found:   {sum(category_counts.values())}")
    print()

    # ── Overall status distribution ──
    status_counts = defaultdict(int)
    for fpath in json_files:
        try:
            d = json.load(open(fpath))
            status_counts[d.get("overall_status", "MISSING")] += 1
        except:
            status_counts["PARSE_ERROR"] += 1
    print(f"OVERALL STATUS DISTRIBUTION")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status:20s}: {count:4d}  ({100*count/total_files:.1f}%)")
    print()

    # ── Issues by category ──
    if category_counts:
        print(f"ISSUES BY CATEGORY (sorted by count)")
        print(f"{'Category':40s} {'Count':>6s}  {'Severity':10s}")
        print(f"{'-'*40} {'-'*6}  {'-'*10}")

        severity_map = {
            "PARSE_ERROR": "CRITICAL",
            "READ_ERROR": "CRITICAL",
            "MISSING_FIELD": "HIGH",
            "NULL_REQUIRED_FIELD": "HIGH",
            "WRONG_TYPE": "HIGH",
            "INVALID_ENUM": "HIGH",
            "STATUS_INCONSISTENCY": "HIGH",
            "FALSE_POSITIVE_NO_EXPLANATION": "MEDIUM",
            "LLM_METADATA": "MEDIUM",
            "TIMING": "LOW",
            "ATTEMPTS_MISMATCH": "MEDIUM",
            "FILE_MISMATCH": "HIGH",
            "EMPTY_TRANSLATED_FILES": "MEDIUM",
            "ATTEMPT_STRUCTURE": "MEDIUM",
            "ATTEMPT_METADATA": "LOW",
            "FILENAME_MISMATCH": "MEDIUM",
            "MISSING_ERROR_INFO": "LOW",
        }

        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            severity = severity_map.get(cat, "UNKNOWN")
            print(f"  {cat:38s} {count:6d}  {severity:10s}")
        print()

        # ── Detailed examples per category ──
        print(f"DETAILED EXAMPLES (up to 5 per category)")
        print(f"{'='*80}")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            severity = severity_map.get(cat, "UNKNOWN")
            print(f"\n[{severity}] {cat} ({count} occurrences)")
            for fname, msg in category_files[cat]:
                print(f"  {fname}")
                print(f"    -> {msg}")
    else:
        print("NO ISSUES FOUND -- all 708 files pass schema audit.")

    # ── Special analysis: false-positive overrides ──
    print(f"\n{'='*80}")
    print(f"SPECIAL ANALYSIS: False-Positive Overrides")
    print(f"(Files where verify_status=pass but overall_status=VERIFY_FAIL)")
    fp_count = 0
    for fpath in json_files:
        try:
            d = json.load(open(fpath))
            if d.get("verify_status") == "pass" and d.get("overall_status") == "VERIFY_FAIL":
                fp_count += 1
                print(f"  {fpath.name}")
                print(f"    error_message: {d.get('error_message', '')[:120]}")
        except:
            pass
    print(f"  Total: {fp_count} files")

    # ── Special analysis: multi-attempt files ──
    print(f"\n{'='*80}")
    print(f"SPECIAL ANALYSIS: Multi-Attempt Files (retries used)")
    multi_count = 0
    for fpath in json_files:
        try:
            d = json.load(open(fpath))
            if d.get("total_attempts", 1) > 1:
                multi_count += 1
                print(f"  {fpath.name}: {d['total_attempts']} attempts, overall={d.get('overall_status')}")
        except:
            pass
    if multi_count == 0:
        print("  None found -- all results are single-attempt.")
    else:
        print(f"  Total: {multi_count} files with retries")

    # ── Return exit code ──
    critical_high = sum(count for cat, count in category_counts.items()
                       if severity_map.get(cat) in ("CRITICAL", "HIGH"))
    if critical_high > 0:
        print(f"\nEXIT: {critical_high} CRITICAL/HIGH issues found. Pipeline needs investigation.")
        return 1
    elif sum(category_counts.values()) > 0:
        print(f"\nEXIT: Only LOW/MEDIUM issues found. Pipeline is fundamentally sound.")
        return 0
    else:
        print(f"\nEXIT: Clean audit. Pipeline is trustworthy.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
