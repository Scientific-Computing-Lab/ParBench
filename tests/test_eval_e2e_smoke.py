"""Phase 2 / Plan 02-07: End-to-end smoke test for the canonical eval config.

Gated by ``PARBENCH_RUN_LLM_TESTS=1`` (skipped otherwise — real API calls cost money).

For each of 5 suite-representative kernels × 2 models
(``together-qwen-3.5-397b-a17b``, ``azure-gpt-5.4``):

  (a) ``test_smoke_dry_run`` — invoke ``llm_evaluate.py --dry-run`` and assert the
      prompt is built without any API call.
  (b) ``test_smoke_real_invocation`` — invoke ``run_eval_batch.py`` once with the
      canonical config (``num_samples=3``, ``thinking=on``, ``temperature=0.7``,
      ``augment_level=0``, ``direction=cuda-to-omp``) and assert each per-sample
      result JSON carries the seven D-29 schema-guardrail fields and an
      ``overall_status`` ∈ {PASS, BUILD_FAIL, RUN_FAIL, VERIFY_FAIL}.

Reuses the canonical ``SUITE_SPECS`` from ``tests/test_spec_loader_integration.py``
(D-27): rodinia-bfs-cuda, xsbench-xsbench-cuda, rsbench-rsbench-cuda,
mixbench-mixbench-cuda, hecbench-bezier-surface-cuda. Direction: ``cuda-to-omp``.

Budget (D-30, pricing verified 2026-04-17 against provider pages):
  - 5 kernels × 2 models × 3 samples = 30 samples (real-API portion).
  - ``azure-gpt-5.4`` (GPT-5.4 standard tier, ``reasoning_effort=medium``):
    $2.50/1M input + $15/1M output. ~$0.3125/sample.
  - ``together-qwen-3.5-397b-a17b``: $0.60/1M input + $3.60/1M output.
    ~$0.075/sample.
  - Smoke cost = 15 × $0.3125 + 15 × $0.075 = **$5.81** (revised ceiling $6).
  - Worst-case if reasoning tokens double output: ~$10.90.
  - The plan-frontmatter line "Budget ≈ $3.47 for 30 real samples" is the
    pre-2026-04-17 figure superseded by the verified $5.81 above.
  - Note: "gpt-5.4" is the ParBench-internal registry key matching the
    Azure deployment name.

Why ``llm_evaluate.py`` for dry-run vs ``run_eval_batch.py`` for real:
  ``run_eval_batch.py`` does NOT carry a ``--dry-run`` flag (only the single-task
  ``llm_evaluate.py`` does — see ``llm_evaluate.py:1455``). Adding ``--dry-run``
  to the batch launcher is out of scope for this plan (plan ``files_modified``
  is fixed at three files). Per-suite dry-run via ``llm_evaluate.py`` exercises
  the same prompt-construction path that ``run_eval_batch.py`` would use, since
  both call ``evaluate_translation()``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT, stage_tmp_project_root
from tests.test_spec_loader_integration import SUITE_SPECS


# D-29 schema-guardrail fields. Two new (plan 02-03): thinking_enabled, num_samples.
# Five existing (regression-guard): sample_id, temperature, augment_level, model,
# overall_status. Exactly seven — asserted by test_required_fields_count_is_seven.
REQUIRED_RESULT_FIELDS = {
    "thinking_enabled",
    "num_samples",
    "sample_id",
    "temperature",
    "augment_level",
    "model",
    "overall_status",
}

# Budget for 30 real samples (5 suites × 2 models × 3 samples). Documented in
# the module docstring above and verified by test_smoke_budget_documented.
SMOKE_BUDGET_USD = 5.81
SMOKE_NUM_SAMPLES = 30  # 5 × 2 × 3

# Functional-smoke status set: any of these is "the pipeline ran end-to-end and
# produced a deterministic verdict". ERROR / SKIP / EXTRACTION_FAIL would
# indicate the pipeline itself broke before reaching a verdict — which is what
# this smoke test exists to catch.
VALID_SMOKE_STATUSES = {"PASS", "BUILD_FAIL", "RUN_FAIL", "VERIFY_FAIL"}

MODELS_UNDER_TEST = [
    "together-qwen-3.5-397b-a17b",
    "azure-gpt-5.4",
]


def _source_spec_paths() -> list[Path]:
    """Return list of source CUDA spec Paths from SUITE_SPECS (D-27).

    SUITE_SPECS is a dict[suite_name, Path] of one CUDA spec per suite.
    """
    return list(SUITE_SPECS.values())


def _suite_id(spec_path: Path) -> str:
    """Stable parametrize id — uses the spec stem (e.g. rodinia-bfs-cuda)."""
    return spec_path.stem


def _target_spec_path(source_spec: Path) -> Path:
    """Derive the cuda-to-omp target spec Path from a CUDA source spec."""
    stem = source_spec.stem  # e.g. rodinia-bfs-cuda
    if not stem.endswith("-cuda"):
        raise ValueError(f"expected -cuda suffix on source spec stem: {stem!r}")
    target_stem = stem[: -len("-cuda")] + "-omp"
    return source_spec.parent / f"{target_stem}.json"


# Module-level marker — every test in this file is gated. The `llm` mark is
# the cost gate (PARBENCH_RUN_LLM_TESTS=1); `integration` ensures we also
# require real benchmark source dirs (Linux GPU machine).
pytestmark = [pytest.mark.integration, pytest.mark.llm]


# ---------------------------------------------------------------------------
# Schema / budget meta-tests
# ---------------------------------------------------------------------------

def test_required_fields_count_is_seven() -> None:
    """REQUIRED_RESULT_FIELDS has exactly 7 entries (D-29)."""
    assert len(REQUIRED_RESULT_FIELDS) == 7, REQUIRED_RESULT_FIELDS


def test_smoke_budget_documented() -> None:
    """The 30-sample / $5.81 budget is documented in the module docstring."""
    import tests.test_eval_e2e_smoke as mod
    doc = mod.__doc__ or ""
    assert "$5.81" in doc, "$5.81 budget figure missing from module docstring"
    assert "30 samples" in doc, "'30 samples' missing from module docstring"
    # Plan-frontmatter $3.47 figure must be reconciled, not silently dropped.
    assert "$3.47" in doc, "plan-frontmatter $3.47 figure must be acknowledged in docstring"


# ---------------------------------------------------------------------------
# Dry-run smoke (no API call)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
@pytest.mark.parametrize(
    "source_spec",
    _source_spec_paths(),
    ids=[_suite_id(p) for p in _source_spec_paths()],
)
def test_smoke_dry_run(source_spec: Path, model: str) -> None:
    """``llm_evaluate.py --dry-run`` builds the prompt without calling the API.

    Exercises the same ``evaluate_translation()`` code path that the real
    invocation will, but short-circuits before any HTTP call. A failure here
    means the prompt-construction layer broke for this (suite, model) pair.
    """
    target_spec = _target_spec_path(source_spec)
    assert source_spec.exists(), f"source spec missing: {source_spec}"
    assert target_spec.exists(), f"target spec missing: {target_spec}"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluation" / "llm_evaluate.py"),
        "--source", str(source_spec),
        "--target", str(target_spec),
        "--model", model,
        "--project-root", str(PROJECT_ROOT),
        "--temperature", "0.7",
        "--thinking", "on",
        "--num-samples", "3",
        "--augment-level", "0",
        "--dry-run",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"dry-run failed (rc={result.returncode}):\n"
        f"STDOUT: {result.stdout[-2000:]}\n"
        f"STDERR: {result.stderr[-2000:]}"
    )

    # Dry-run with --json prints the SYSTEM/USER MESSAGE banner to stdout
    # first (llm_evaluate.py:1463), then appends the JSON result object via
    # print_result() (:1982). Extract the trailing JSON blob — the final
    # balanced {...} segment in stdout — matching the parse pattern used by
    # tests/test_eval_integration_smoke.py:249.
    stdout = result.stdout
    last_brace = stdout.rfind("\n{")
    payload_text = stdout[last_brace + 1:] if last_brace != -1 else stdout
    payload = json.loads(payload_text)
    assert payload.get("dry_run") is True, payload
    assert payload.get("overall_status") == "DRY_RUN", payload
    assert payload.get("model") == model, payload
    # Functional smoke: prompt construction must have actually occurred.
    assert "SYSTEM MESSAGE" in stdout, "dry-run did not emit SYSTEM MESSAGE banner"
    assert "USER MESSAGE" in stdout, "dry-run did not emit USER MESSAGE banner"


# ---------------------------------------------------------------------------
# Real-API canonical-config smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
@pytest.mark.parametrize(
    "source_spec",
    _source_spec_paths(),
    ids=[_suite_id(p) for p in _source_spec_paths()],
)
def test_smoke_real_invocation(source_spec: Path, model: str, tmp_path: Path) -> None:
    """One canonical-config invocation per (suite, model) pair.

    Asserts:
      - subprocess exits 0.
      - At least one per-sample result JSON is produced.
      - Every result JSON has all 7 D-29 schema fields.
      - ``overall_status`` is in VALID_SMOKE_STATUSES.
      - Canonical-config guardrails: ``augment_level=0``, ``temperature=0.7``,
        ``num_samples=3``, ``thinking_enabled=True``, ``model`` matches.
    """
    target_spec = _target_spec_path(source_spec)
    src_id = source_spec.stem
    tgt_id = target_spec.stem
    suite = src_id.split("-", 1)[0]
    # `--kernels` matches by kernel_name, not unique_id. Strip the leading
    # "{suite}-" and trailing "-cuda" segments to recover the kernel slug.
    kernel_name = src_id.split("-", 1)[1].rsplit("-", 1)[0]

    # Stage tmp_path as a project-root view: symlinks for manifest.jsonl,
    # specs/, and benchmark source dirs; tmp_path/results/ stays fresh so
    # this test never touches real results/evaluation/ (D-15).
    stage_tmp_project_root(tmp_path)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluation" / "run_eval_batch.py"),
        "--suite", suite,
        "--kernels", kernel_name,
        "--direction", "cuda-to-omp",
        "--models", model,
        "--augment-levels", "0",
        "--num-samples", "3",
        "--temperature", "0.7",
        "--thinking", "on",
        "--project-root", str(tmp_path),
        "--no-resume",  # force fresh samples for the smoke
        "-v",
    ]
    # 30-minute ceiling per (suite, model) — three samples, real LLM round-trips,
    # build/run/verify each. The plan estimates ~10 min for the full 30-sample
    # run; per-pair budget is generous to absorb cold-start latency.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"real invocation failed (rc={result.returncode}):\n"
        f"STDOUT[-3000:]: {result.stdout[-3000:]}\n"
        f"STDERR[-3000:]: {result.stderr[-3000:]}"
    )

    # Locate the per-sample result JSONs. run_eval_batch.py writes to:
    #   {project_root}/results/evaluation/{model}/{src}-to-{tgt}{level_tag}{sample_tag}.json
    # With augment_level=0, level_tag="" ; with num_samples=3, sample_tag is
    # "-s0", "-s1", "-s2" (per _result_path() in run_eval_batch.py:228).
    out_dir = tmp_path / "results" / "evaluation" / model
    pattern = f"{src_id}-to-{tgt_id}-s*.json"
    result_files = sorted(out_dir.glob(pattern))
    assert result_files, (
        f"no per-sample result JSONs found at {out_dir}/{pattern}\n"
        f"STDOUT[-2000:]: {result.stdout[-2000:]}"
    )

    for rf in result_files:
        data = json.loads(rf.read_text())
        missing = REQUIRED_RESULT_FIELDS - set(data.keys())
        assert not missing, f"{rf.name} missing required fields: {missing}"
        assert data["overall_status"] in VALID_SMOKE_STATUSES, (
            f"{rf.name} unexpected overall_status={data['overall_status']!r}; "
            f"smoke expects functional pipeline outcome (PASS/BUILD_FAIL/"
            f"RUN_FAIL/VERIFY_FAIL), got error/skip"
        )
        # Canonical config guardrails (D-09 + plan must_haves).
        assert data["augment_level"] == 0, rf.name
        assert data["temperature"] == 0.7, rf.name
        assert data["num_samples"] == 3, rf.name
        assert data["thinking_enabled"] is True, rf.name
        assert data["model"] == model, rf.name
