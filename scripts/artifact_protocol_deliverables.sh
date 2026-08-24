#!/usr/bin/env bash
# Stage the bounded agentic-repair protocol deliverables (plan Task 10) into
# an artifact staging root: versioned config, JSON schema, the dry-run command
# (script + its documented invocation), and the redacted dry-run trajectory.
# Shared by scripts/build_artifact.sh and scripts/build_artifact_zip.sh;
# asserted by tests/test_artifact_protocol_deliverables.py.
# Usage: bash scripts/artifact_protocol_deliverables.sh <project_root> <staging_root>
set -euo pipefail

PROJECT_ROOT="$1"
STAGING_ROOT="$2"

mkdir -p "$STAGING_ROOT/config" "$STAGING_ROOT/schema" \
         "$STAGING_ROOT/scripts/evaluation" "$STAGING_ROOT/results/analysis"

cp "$PROJECT_ROOT/config/agentic_repair_protocol.json" "$STAGING_ROOT/config/"
cp "$PROJECT_ROOT/schema/agentic_repair_protocol_schema.json" "$STAGING_ROOT/schema/"
cp "$PROJECT_ROOT/scripts/evaluation/dry_run_agentic_protocol.py" "$STAGING_ROOT/scripts/evaluation/"
cp "$PROJECT_ROOT/results/analysis/agentic_protocol_dry_run.json" "$STAGING_ROOT/results/analysis/"

# Runtime dependencies of the dry-run command (it imports llm_evaluate, which
# imports pair_contracts); harness/*.py is staged by the tarball builder and
# ships whole in the ZIP.
cp "$PROJECT_ROOT/scripts/evaluation/llm_evaluate.py" "$STAGING_ROOT/scripts/evaluation/"
cp "$PROJECT_ROOT/scripts/evaluation/pair_contracts.py" "$STAGING_ROOT/scripts/evaluation/"
cp "$PROJECT_ROOT/scripts/evaluation/__init__.py" "$STAGING_ROOT/scripts/evaluation/"

echo "Staged agentic-repair protocol deliverables into $STAGING_ROOT"
