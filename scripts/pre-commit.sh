#!/usr/bin/env bash
# Native git pre-commit hook body (wired at .git/hooks/pre-commit).
# Runs the two fast deterministic checks and fails the commit if either fails.
# Replaces the retired PreToolUse .validation_passed sentinel gate (2026-08-14):
# a native git hook is lighter, runs inside git, and does not block every Bash call.
# For the heavier project-health pass (schema, tests, manifest), run /validate on demand.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
python3 .claude/hooks/check_stale_counts.py || { echo "pre-commit: stale-count check failed"; exit 1; }
python3 scripts/check_generated_registry.py || { echo "pre-commit: generated-registry check failed"; exit 1; }
exit 0
