"""Regression pin for rule 28: no stale top-level run fields in result JSONs.

llm_evaluate.py resets final_build_result / final_run_result / final_verify_result
at the top of every attempt (the 2231-2237 block), so when attempt 1 reaches
RUN PASS and attempt 2 regresses to BUILD_FAIL, the top-level run_status /
run_time_seconds / run_exit_code serialize as null, not attempt 1's stale PASS.
That fix is only recorded as "fixed prospectively" - nothing stopped a refactor
from hoisting the resets back out of the loop until this test.

No provider call, no compiler: call_llm, extraction, build, run, and verify are
all monkeypatched at the llm_evaluate module seam. Same-API pair, so no pair
contract is required.
"""

import json
from pathlib import Path

from harness.models import BuildResult, RunResult, Status, VerificationResult
from scripts.evaluation import llm_evaluate

SOURCE = "int main(int argc, char **argv) { return 0; }\n"


def make_spec(root: Path, name: str) -> Path:
    src_dir = root / "tree" / "kern"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "main.cpp").write_text(SOURCE)
    spec = {
        "spec_version": "1.0.0",
        "identity": {
            "kernel_name": "kern",
            "parallel_api": "omp",
            "unique_id": f"fix-kern-{name}",
            "source_suite": "fix",
        },
        "provenance": {"repo_root": "tree", "source_path": "kern"},
        "files": {
            "prompt_payload": ["main.cpp"],
            "support_files": [],
            "verification_only": [],
            "translation_targets": ["main.cpp"],
        },
        "build": {"commands": {"build": "true"}},
        "run": {
            "executable": "./kern.out",
            "input_configurations": {
                "correctness": {"arguments": [], "input_files": []}
            },
        },
        "verification": {
            "strategies": [{"type": "exit_code", "expected": 0}]
        },
    }
    path = root / f"fix-kern-{name}.json"
    path.write_text(json.dumps(spec))
    return path


def test_regressing_second_attempt_leaves_no_stale_run_fields(tmp_path, monkeypatch):
    src = make_spec(tmp_path, "src")
    tgt = make_spec(tmp_path, "tgt")

    monkeypatch.setattr(
        llm_evaluate, "call_llm",
        lambda *a, **k: {
            "response_text": "code follows",
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "duration_seconds": 0.01,
            "finish_reason": "stop",
        },
    )
    monkeypatch.setattr(
        llm_evaluate, "extract_code_blocks",
        lambda text, names: {n: "// translated\n" for n in names},
    )

    builds = []

    def fake_build(spec, project_root, verbose=False):
        builds.append(1)
        if len(builds) == 1:
            return BuildResult(Status.PASS, 0.1, "", "")
        return BuildResult(Status.FAIL, 0.1, "", "fatal error: no such header")

    monkeypatch.setattr(llm_evaluate, "build_spec", fake_build)
    monkeypatch.setattr(
        llm_evaluate, "run_spec",
        lambda spec, project_root, verbose=False, measure_cpu_time=False:
            RunResult(Status.PASS, "correctness", 1.23, 0, "ran fine", ""),
    )
    monkeypatch.setattr(
        llm_evaluate, "verify_run",
        lambda spec, run_result, working_dir=None:
            VerificationResult(Status.FAIL, "exit_code", "wrong output"),
    )
    monkeypatch.setattr(llm_evaluate, "extract_metrics", lambda spec, run_result: [])

    result = llm_evaluate.evaluate_translation(
        source_path=src,
        target_path=tgt,
        model="test-model",
        project_root=tmp_path,
        max_retries=2,
        save_to_disk=False,
    )

    # Attempt 1: RUN PASS then VERIFY_FAIL. Attempt 2: BUILD_FAIL.
    assert len(builds) == 2
    assert result["attempts"][0]["run_status"] == Status.PASS.value
    assert result["overall_status"] == "BUILD_FAIL"

    # The rule under test: the top-level trio reflects ONLY the final attempt,
    # which never ran, so all three must be None - not attempt 1's values.
    assert result["run_status"] is None
    assert result["run_time_seconds"] is None
    assert result["run_exit_code"] is None
