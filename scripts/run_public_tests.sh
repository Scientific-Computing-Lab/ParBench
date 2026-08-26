#!/bin/bash
# Public-tree test runner: the unit suite minus tests whose subject script or
# fixture data is not part of the public release. CI (.github/workflows/ci.yml),
# CONTRIBUTING.md, and the PR template all call this script, so the exclusion
# list lives in exactly one place. On a full private checkout, plain
# `python3 -m pytest tests/` remains the stricter superset.
set -euo pipefail
cd "$(dirname "$0")/.."

EXCLUDE=(
  # subject scripts/spec_tools/apply_rodinia_patches.sh is not in the public release (10/10 tests)
  --ignore=tests/test_apply_rodinia_patches.py
  # needs scripts/build_artifact.sh and results/analysis/agentic_protocol_dry_run.json (3/3 tests)
  --ignore=tests/test_artifact_protocol_deliverables.py
  # needs the raw results/evaluation/ record tree (1/1 test)
  --ignore=tests/test_emit_release_inventory.py

  # fixtures cite the evidence file results/analysis/pair_witnesses.json
  --deselect tests/test_freeze_final_contract.py::TestHappyPath
  # needs the results/analysis/negative_controls/ records
  --deselect tests/test_mixbench_scope.py::TestNegativeControlEvidence

  # subject scripts/lib/artifact_release_common.sh is not in the public release
  --deselect tests/test_artifact_release_common.py::test_parse_all_flags_and_extras
  --deselect tests/test_artifact_release_common.py::test_parse_defaults_empty
  --deselect tests/test_artifact_release_common.py::test_parse_missing_value_returns_2
  --deselect tests/test_artifact_release_common.py::test_guard_absent_output_ok
  --deselect tests/test_artifact_release_common.py::test_guard_empty_output_ok
  --deselect tests/test_artifact_release_common.py::test_guard_nonempty_no_resume_refuses
  --deselect tests/test_artifact_release_common.py::test_guard_resume_without_sidecar_refuses
  --deselect tests/test_artifact_release_common.py::test_guard_resume_matching_provenance_ok
  --deselect tests/test_artifact_release_common.py::test_guard_resume_mismatch_refuses
  --deselect tests/test_artifact_release_common.py::test_scratch_trap_removes_only_owned_dir

  # subject scripts/orchestration/gate_verdict.sh is not in the public release
  --deselect "tests/test_gate_verdict.py::TestGateVerdict::test_genuine_pass[gate-verdict]"
  --deselect "tests/test_gate_verdict.py::TestGateVerdict::test_genuine_pass[task-result]"
  --deselect "tests/test_gate_verdict.py::TestGateVerdict::test_missing_log_is_exit_2[gate-verdict]"
  --deselect "tests/test_gate_verdict.py::TestGateVerdict::test_missing_log_is_exit_2[task-result]"
  --deselect "tests/test_gate_verdict.py::TestGateVerdict::test_empty_log_is_exit_2[gate-verdict]"
  --deselect "tests/test_gate_verdict.py::TestGateVerdict::test_empty_log_is_exit_2[task-result]"

  # subject scripts/batch/run_phase3.sh is not in the public release
  --deselect tests/test_eval_exclusions.py::test_run_phase3_has_no_copied_exclusion_list
  --deselect tests/test_eval_exclusions.py::test_run_phase3_batch_and_retry_invocations_rely_on_automatic_exclusions

  # needs Rodinia sources on disk; no suite tree is assumed here
  --deselect tests/test_check_spec_argc_contracts.py::test_frozen_v1_registry_still_shows_the_two_streamcluster_findings
)

python3 -m pytest tests/ -q "${EXCLUDE[@]}" "$@"
