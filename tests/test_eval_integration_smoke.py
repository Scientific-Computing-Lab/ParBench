"""
Phase 2 / Plan 02-08: Integration smoke + GPT-5.4 handoff runbook.

Three parts:
  Part 1 — Zero-API dry-run matrix: 5 suites × 6 directions × 2 models = 60 cases.
           Marker: @pytest.mark.integration ONLY (no llm). Runs on every default pytest tests/.
  Part 2 — Real-API E2E slice: canonical -> derive_l0_passers -> ablation on one kernel pair.
           Marker: @pytest.mark.integration + @pytest.mark.llm (gated by PARBENCH_RUN_LLM_TESTS=1).
  Part 3 — docs/neurips2026-gpt5-handoff.md: plaintext runbook for Le (documented separately).

Why this test exists (pre-Phase-3 guardrail):
  The highest-risk integration bug surface is --task-list × augmented source ×
  Azure reasoning_effort × result-JSON schema. If that path has a bug, we burn money
  in Phase 3. One dry-run matrix plus one real E2E canonical->derive->ablation trace
  is the cheapest guardrail.

Cost model (updated 2026-04-17 against live provider pricing):
  azure-gpt-5.4 (GPT-5.4, Azure Foundry, <272K context):
    $2.50/1M input tokens, $15.00/1M output tokens
    Source: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-5-4-in-microsoft-foundry/4499785
    Per-sample assumption: 5k prompt + 20k output -> ~$0.3125/sample
  together-qwen-3.5-397b-a17b (Together AI serverless):
    $0.60/1M input tokens, $3.60/1M output tokens
    Source: https://www.together.ai/models/qwen3-5-397b-a17b
    Per-sample assumption: 5k prompt + 20k output -> ~$0.075/sample
  Reasoning tokens at reasoning_effort=medium bill as output tokens on both providers.
  Worst-case envelope ~ $9/run (GPT-5.4 emitting ~80k output tokens/sample at 4× inflation).
  02-08 real-API budget: 8 samples × 2 models worst-case -> ~$3.10 typical, ~$10 worst-case.

Candidate-kernel rule (D-19 / plan file §Part 2):
  CANDIDATE_SOURCE = "rodinia-bfs-cuda", CANDIDATE_TARGET = "rodinia-bfs-omp".
  These are candidates, NOT known passers. The canonical step determines pass/fail.
  If canonical produces 0 passers, pytest.fail is called — that IS the signal this
  test exists to catch before Phase 3 burns money on a broken code path.

pass@1-of-any (D-18): a (source, target) cell is a passer if >=1 of its 3 canonical
  samples has overall_status == "PASS".
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT, stage_tmp_project_root
from tests.test_spec_loader_integration import SUITE_SPECS


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODELS_UNDER_TEST = ["together-qwen-3.5-397b-a17b", "azure-gpt-5.4"]

DIRECTIONS = [
    "cuda-to-omp",
    "omp-to-cuda",
    "cuda-to-opencl",
    "opencl-to-cuda",
    "omp-to-opencl",
    "opencl-to-omp",
]

# Candidate kernel pair for the real-API E2E ablation slice.
# These are CANDIDATES, not known passers — canonical step decides empirically.
CANDIDATE_SOURCE = "rodinia-bfs-cuda"
CANDIDATE_TARGET = "rodinia-bfs-omp"


# ---------------------------------------------------------------------------
# Collection-time loaders (KNOWN_FAIL set + manifest target set)
# ---------------------------------------------------------------------------

def _load_known_fail_ids() -> set[str]:
    """Parse the 8-entry KNOWN_FAIL table from .claude/rules/known-issues.md at collection time.

    Reads the markdown table under §"KNOWN_FAIL Specs (8 — exclude from eval batches)".
    Returns a set of spec_id strings like {'rodinia-kmeans-cuda', ...}.
    Never hard-codes the list — sourced from the authoritative known-issues.md.
    """
    known_issues_path = PROJECT_ROOT / ".claude" / "rules" / "known-issues.md"
    text = known_issues_path.read_text()
    section_match = re.search(r"##\s+KNOWN_FAIL Specs.*?(?=\n##|\Z)", text, re.S)
    if not section_match:
        return set()
    section = section_match.group()
    spec_ids: set[str] = set()
    # Match table rows like: | `rodinia-kmeans-cuda` | ... |
    for line in section.splitlines():
        m = re.match(r"\|\s*`([a-z0-9][a-z0-9_-]+)`\s*\|", line)
        if m:
            spec_ids.add(m.group(1))
    return spec_ids


KNOWN_FAIL_IDS: set[str] = _load_known_fail_ids()


def _load_manifest_targets() -> set[str]:
    """Read manifest.jsonl and return a set of all known spec unique_ids.

    NOTE (Rule 1 deviation from PLAN snippet): manifest.jsonl entries do NOT carry a
    `unique_id` field. The unique_id is derived from `{source_suite}-{kernel_name}-{parallel_api}`,
    which matches the spec_file stem convention. Using `entry.get("unique_id")` (as in the
    plan snippet) would silently return an empty set, so we derive instead.
    """
    manifest_path = PROJECT_ROOT / "manifest.jsonl"
    ids: set[str] = set()
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            suite = entry.get("source_suite")
            kernel = entry.get("kernel_name")
            api = entry.get("parallel_api")
            if suite and kernel and api:
                ids.add(f"{suite}-{kernel}-{api}")
    return ids


MANIFEST_IDS: set[str] = _load_manifest_targets()


def _api_from_direction(direction: str) -> tuple[str, str]:
    """Return (src_api, tgt_api) from a direction string like 'cuda-to-omp'."""
    src, _, tgt = direction.partition("-to-")
    return src, tgt


# ---------------------------------------------------------------------------
# Part 1 — Zero-API dry-run matrix
# ---------------------------------------------------------------------------

def _dry_run_cases() -> list[tuple[str, str, str, str, str, str]]:
    """Build parametrize cases for dry-run matrix (5 suites × 6 directions × 2 models).

    SUITE_SPECS maps suite_name -> Path to JSON spec file (dict shape from
    tests/test_spec_loader_integration.py:34). spec_id is Path.stem
    (e.g. "rodinia-bfs-cuda"). Kernel slug is derived by stripping the leading
    "{suite}-" and the trailing "-{api}" segments.
    """
    cases: list[tuple[str, str, str, str, str, str]] = []
    for suite, spec_path in SUITE_SPECS.items():
        spec_id = spec_path.stem  # e.g. "rodinia-bfs-cuda" or "hecbench-bezier-surface-cuda"
        # spec_id format: {suite}-{kernel_slug}-{api}
        # Strip suite prefix then strip trailing -{api} suffix.
        if not spec_id.startswith(f"{suite}-"):
            continue
        rest = spec_id[len(suite) + 1:]  # e.g. "bfs-cuda" or "bezier-surface-cuda"
        # Drop the trailing -<api> segment.
        if "-" not in rest:
            continue
        kernel_slug, _, _src_api = rest.rpartition("-")
        for direction in DIRECTIONS:
            src_api, tgt_api = _api_from_direction(direction)
            src_spec_id = f"{suite}-{kernel_slug}-{src_api}"
            tgt_spec_id = f"{suite}-{kernel_slug}-{tgt_api}"
            for model in MODELS_UNDER_TEST:
                cases.append(
                    (suite, kernel_slug, direction, src_spec_id, tgt_spec_id, model)
                )
    return cases


_DRY_RUN_CASES = _dry_run_cases()


@pytest.mark.integration
@pytest.mark.parametrize(
    "suite,kernel_slug,direction,src_spec_id,tgt_spec_id,model",
    _DRY_RUN_CASES,
    ids=[
        f"{s}-{k}-{d}-{m.split('-')[0]}"
        for s, k, d, _, _, m in _DRY_RUN_CASES
    ],
)
def test_dryrun_matrix(
    suite: str,
    kernel_slug: str,
    direction: str,
    src_spec_id: str,
    tgt_spec_id: str,
    model: str,
) -> None:
    """Part 1: dry-run matrix — zero API calls, asserts prompt construction + task list.

    Skips when source/target is KNOWN_FAIL or when the target spec does not exist
    in manifest.jsonl (e.g. some suites have no opencl variant).
    """
    if src_spec_id in KNOWN_FAIL_IDS:
        pytest.skip(f"source {src_spec_id!r} is in KNOWN_FAIL — skip")
    if tgt_spec_id in KNOWN_FAIL_IDS:
        pytest.skip(f"target {tgt_spec_id!r} is in KNOWN_FAIL — skip")
    if src_spec_id not in MANIFEST_IDS:
        pytest.skip(f"source {src_spec_id!r} not in manifest")
    if tgt_spec_id not in MANIFEST_IDS:
        pytest.skip(
            f"target {tgt_spec_id!r} not in manifest "
            f"(suite {suite!r} may not have {_api_from_direction(direction)[1]!r} variant)"
        )

    # run_eval_batch.py does NOT carry --dry-run (per 02-07 docstring); use llm_evaluate.py
    # directly with --dry-run, which exercises the same evaluate_translation() prompt path.
    src_spec_file = PROJECT_ROOT / "specs" / f"{src_spec_id}.json"
    tgt_spec_file = PROJECT_ROOT / "specs" / f"{tgt_spec_id}.json"
    if not src_spec_file.exists() or not tgt_spec_file.exists():
        pytest.skip(
            f"spec file missing: {src_spec_file.name} or {tgt_spec_file.name}"
        )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluation" / "llm_evaluate.py"),
        "--source", str(src_spec_file),
        "--target", str(tgt_spec_file),
        "--model", model,
        "--project-root", str(PROJECT_ROOT),
        "--temperature", "0.7",
        "--thinking", "on",
        "--num-samples", "1",
        "--augment-level", "0",
        "--dry-run",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"dry-run exited {result.returncode} for ({suite}, {direction}, {model})\n"
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )
    # Dry-run path prints the SYSTEM/USER prompt to stdout, then (with --json) appends
    # a JSON result object via _print_result. Extract the trailing JSON blob — the
    # final balanced {...} segment in stdout — and verify dry_run/overall_status/model.
    stdout = result.stdout
    last_brace = stdout.rfind("\n{")
    payload_text = stdout[last_brace + 1:] if last_brace != -1 else stdout
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"dry-run trailing JSON not parseable for ({suite}, {direction}, {model}): {e}\n"
            f"stdout (last 500): {stdout[-500:]}"
        )
    assert payload.get("dry_run") is True, payload
    assert payload.get("overall_status") == "DRY_RUN", payload
    assert payload.get("model") == model, payload
    # Functional smoke: prompt construction must have actually occurred.
    assert "SYSTEM MESSAGE" in stdout, "dry-run did not emit the SYSTEM MESSAGE banner"
    assert "USER MESSAGE" in stdout, "dry-run did not emit the USER MESSAGE banner"


# ---------------------------------------------------------------------------
# Part 2 — Real-API E2E slice (canonical -> derive -> ablation)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
def test_real_e2e_canonical_to_ablation(model: str, tmp_path: Path) -> None:
    """Part 2a: full Phase 3 sequence on CANDIDATE_SOURCE -> CANDIDATE_TARGET.

    Canonical (L0, 3 samples) -> derive_l0_passers -> ablation (L1-L4, 1 sample each).
    Uses tmp_path as --project-root; never touches real results/evaluation/ tree.

    CANDIDATE_SOURCE and CANDIDATE_TARGET are candidates, NOT known passers.
    If canonical L0 produces zero passers, pytest.fail is called — that is the signal.
    pass@1-of-any (D-18): cell is a passer if >=1 of 3 canonical samples has
    overall_status==PASS.
    """
    suite = CANDIDATE_SOURCE.split("-")[0]  # "rodinia"
    # Strip "{suite}-" prefix and trailing "-{api}" suffix to get kernel slug.
    rest = CANDIDATE_SOURCE[len(suite) + 1:]
    kernel_slug = rest.rpartition("-")[0]  # "bfs"

    # Stage tmp_path as project-root view (symlinks for manifest/specs/benchmarks;
    # tmp_path/results/ stays fresh so real results/evaluation/ is never touched).
    stage_tmp_project_root(tmp_path)

    required_fields = (
        "thinking_enabled", "num_samples", "augment_level",
        "temperature", "model", "overall_status", "sample_id",
    )

    # ---- Step 1: Canonical run (L0, 3 samples) ---------------------------
    canonical_cmd = [
        sys.executable, "-m", "scripts.evaluation.run_eval_batch",
        "--suite", suite,
        "--kernels", kernel_slug,
        "--direction", "cuda-to-omp",
        "--models", model,
        "--augment-levels", "0",
        "--num-samples", "3",
        "--temperature", "0.7",
        "--thinking", "on",
        "--project-root", str(tmp_path),
        "-v",
    ]
    r = subprocess.run(
        canonical_cmd,
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=1800,
    )
    assert r.returncode == 0, (
        f"Canonical run failed.\nstdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}"
    )

    # ---- Step 2: Assert 3 result JSONs with 02-03 schema fields ----------
    model_results_dir = tmp_path / "results" / "evaluation" / model
    result_jsons = (
        list(model_results_dir.rglob("*.json")) if model_results_dir.exists() else []
    )
    assert result_jsons, f"No result JSONs in {model_results_dir}"
    for rf in result_jsons:
        data = json.loads(rf.read_text())
        for field in required_fields:
            assert field in data, f"{rf.name} missing field {field!r}"
        assert data["augment_level"] == 0
        assert data["num_samples"] == 3
        assert data["thinking_enabled"] is True
        assert data["temperature"] == 0.7
        assert data["model"] == model

    # ---- Step 3: Derive L0 passers ---------------------------------------
    passer_json = tmp_path / "passer.json"
    derive_cmd = [
        sys.executable, "-m", "scripts.evaluation.derive_l0_passers",
        "--canonical-dir", str(model_results_dir),
        "--model", model,
        "--out", str(passer_json),
    ]
    r2 = subprocess.run(
        derive_cmd,
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
    )
    assert r2.returncode == 0, (
        f"derive_l0_passers failed.\nstdout: {r2.stdout}\nstderr: {r2.stderr}"
    )
    assert passer_json.exists(), "passer.json was not created"

    # ---- Step 4: Empty passer list -> pytest.fail (this IS the signal) ---
    passers = json.loads(passer_json.read_text())
    if not passers:
        pytest.fail(
            f"Canonical L0 for {CANDIDATE_SOURCE}->{CANDIDATE_TARGET}/{model} "
            f"produced no passers. Ablation code path unexercised. "
            f"Pick a different candidate kernel or investigate the canonical failure."
        )

    # ---- Step 5: Ablation run (L1-L4, 1 sample each) ---------------------
    ablation_cmd = [
        sys.executable, "-m", "scripts.evaluation.run_eval_batch",
        "--task-list", str(passer_json),
        "--models", model,
        "--augment-levels", "1", "2", "3", "4",
        "--num-samples", "1",
        "--temperature", "0.7",
        "--thinking", "on",
        "--direction", "cuda-to-omp",
        "--project-root", str(tmp_path),
        "-v",
    ]
    r3 = subprocess.run(
        ablation_cmd,
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=2400,
    )
    assert r3.returncode == 0, (
        f"Ablation run failed.\nstdout: {r3.stdout[-2000:]}\nstderr: {r3.stderr[-2000:]}"
    )

    # ---- Step 6: Assert >=4 ablation JSONs (one per L1..L4) --------------
    ablation_jsons = [
        rf for rf in model_results_dir.rglob("*.json")
        if any(tag in rf.stem for tag in ("-L1", "-L2", "-L3", "-L4"))
    ]
    assert len(ablation_jsons) >= 4, (
        f"Expected >=4 ablation JSONs (L1..L4), found {len(ablation_jsons)}: "
        + str([f.name for f in ablation_jsons])
    )
    ablation_levels: set[int] = set()
    for rf in ablation_jsons:
        data = json.loads(rf.read_text())
        assert data["augment_level"] in (1, 2, 3, 4), (
            f"{rf.name}: augment_level={data['augment_level']!r}"
        )
        assert data["num_samples"] == 1, f"{rf.name}: num_samples={data['num_samples']!r}"
        for field in required_fields:
            assert field in data, f"{rf.name} missing field {field!r}"
        ablation_levels.add(data["augment_level"])
    assert ablation_levels == {1, 2, 3, 4}, (
        f"Expected levels {{1,2,3,4}}, got {ablation_levels}"
    )


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
def test_real_direction_independence_omp_to_cuda(
    model: str, tmp_path: Path
) -> None:
    """Part 2b: one real omp-to-cuda cell — proves direction swap is not cuda-to-omp-only.

    Canonical L0, num_samples=1 (pass@1) to cap spend.
    Uses tmp_path as --project-root; never touches real results/evaluation/.
    """
    suite = CANDIDATE_TARGET.split("-")[0]  # "rodinia"
    rest = CANDIDATE_TARGET[len(suite) + 1:]
    kernel_slug = rest.rpartition("-")[0]  # "bfs"

    # Stage tmp_path so run_eval_batch finds manifest/specs/benchmarks; keeps
    # tmp_path/results/ isolated from the real tree.
    stage_tmp_project_root(tmp_path)

    cmd = [
        sys.executable, "-m", "scripts.evaluation.run_eval_batch",
        "--suite", suite,
        "--kernels", kernel_slug,
        "--direction", "omp-to-cuda",
        "--models", model,
        "--augment-levels", "0",
        "--num-samples", "1",
        "--temperature", "0.7",
        "--thinking", "on",
        "--project-root", str(tmp_path),
        "-v",
    ]
    r = subprocess.run(
        cmd,
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=1800,
    )
    assert r.returncode == 0, (
        f"omp-to-cuda run failed.\nstdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}"
    )
    model_results_dir = tmp_path / "results" / "evaluation" / model
    result_jsons = (
        list(model_results_dir.rglob("*.json")) if model_results_dir.exists() else []
    )
    assert result_jsons, f"No result JSONs in {model_results_dir}"
    required_fields = (
        "thinking_enabled", "num_samples", "augment_level",
        "temperature", "model", "overall_status", "sample_id",
    )
    for rf in result_jsons:
        data = json.loads(rf.read_text())
        for field in required_fields:
            assert field in data, (
                f"{rf.name} missing schema field {field!r} (D-29 / 02-03)"
            )
        assert data["model"] == model


# ---------------------------------------------------------------------------
# Token + cost teardown fixture (autouse — only logs when tmp_path has results)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _log_token_costs(request, tmp_path):
    """Teardown: read result JSONs from tmp_path and log actual token usage + cost.

    Only active when the test uses tmp_path AND that tmp_path contains result JSONs.
    Dry-run tests (Part 1) have no result JSONs and are silently skipped.

    Pricing (2026-04-17, verified against provider pages):
      azure-gpt-5.4: $2.50/1M input, $15.00/1M output
      together-qwen-3.5-397b-a17b: $0.60/1M input, $3.60/1M output
    """
    yield  # run the test first
    results_root = tmp_path / "results" / "evaluation"
    if not results_root.exists():
        return
    pricing = {
        "azure-gpt-5.4": (2.50 / 1_000_000, 15.00 / 1_000_000),
        "together-qwen-3.5-397b-a17b": (0.60 / 1_000_000, 3.60 / 1_000_000),
    }
    total_cost = 0.0
    for rf in results_root.rglob("*.json"):
        try:
            data = json.loads(rf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        model = data.get("model", "unknown")
        prompt_tokens = data.get("prompt_tokens", 0) or 0
        completion_tokens = data.get("completion_tokens", 0) or 0
        in_rate, out_rate = pricing.get(model, (0.0, 0.0))
        cost = prompt_tokens * in_rate + completion_tokens * out_rate
        total_cost += cost
        print(
            f"\n[token-log] {rf.name}: "
            f"prompt={prompt_tokens} completion={completion_tokens} "
            f"cost~${cost:.4f} ({model})"
        )
    if total_cost > 0:
        print(f"\n[token-log] TOTAL estimated cost for this test: ${total_cost:.4f}")
