"""Task 10 (gate E-08): agent-path visibility rule.

The agent path must never receive reference implementations, target
implementations, verification_only files, hidden expected outputs, or oracle
contents. Proven here with a captured-request fixture: the complete
agent-visible request (system message, translation prompt, and repair
feedback) is captured from the real prompt-construction code over synthetic
specs whose forbidden files carry unique sentinels, then screened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.models import Status  # noqa: E402
from scripts.evaluation.llm_evaluate import (  # noqa: E402
    AGENTIC_PROTOCOL_CONFIG_PATH,
    AgenticVisibilityError,
    _build_retry_message,
    _read_target_infrastructure,
    _strip_comments,
    build_translation_prompt,
    check_agentic_visibility,
    collect_forbidden_agentic_content,
    enforce_agentic_visibility,
)

REFERENCE_IMPL_SENTINEL = "SENTINEL_TARGET_REFERENCE_IMPLEMENTATION_BODY_74ce01"
VERIFICATION_ONLY_SENTINEL = "SENTINEL_VERIFICATION_ONLY_FILE_BODY_2b9df4"
HIDDEN_EXPECTED_OUTPUT_SENTINEL = "SENTINEL_HIDDEN_EXPECTED_OUTPUT_BODY_c31e88"
ORACLE_PATTERN_SENTINEL = "SENTINEL_ORACLE_STDOUT_PATTERN_5fd207"
ORACLE_SHA256_SENTINEL = "5fd207" * 10 + "beef"  # 64 hex chars, distinctive


@pytest.fixture()
def synthetic_pair(tmp_path):
    """Synthetic source/target specs whose forbidden files carry sentinels."""
    src_dir = tmp_path / "src_cuda"
    tgt_dir = tmp_path / "tgt_omp"
    src_dir.mkdir()
    tgt_dir.mkdir()

    (src_dir / "kernel.cu").write_text(
        "// synthetic CUDA source\n__global__ void k() {}\n", encoding="utf-8"
    )
    # The code body is distinctive on purpose: the checker also screens the
    # comment-stripped variant of every forbidden file, and a generic stripped
    # body ("void k() {}") would falsely match inside the source payload.
    (tgt_dir / "kernel.cpp").write_text(
        f"// {REFERENCE_IMPL_SENTINEL}\nvoid k_ref_impl_74ce01() {{}}\n",
        encoding="utf-8",
    )
    (tgt_dir / "verify_only.dat").write_text(
        f"{VERIFICATION_ONLY_SENTINEL}\n", encoding="utf-8"
    )
    (tgt_dir / "ref_output.txt").write_text(
        f"{HIDDEN_EXPECTED_OUTPUT_SENTINEL}\n", encoding="utf-8"
    )

    source_spec = {
        "identity": {
            "unique_id": "synthetic-kernel-cuda",
            "kernel_name": "kernel",
            "parallel_api": "cuda",
        },
        "provenance": {"repo_root": "", "source_path": "src_cuda"},
        "files": {
            "prompt_payload": ["kernel.cu"],
            "translation_targets": ["kernel.cu"],
            "support_files": [],
            "verification_only": [],
        },
        "build": {"commands": {"build": "make"}},
    }
    target_spec = {
        "identity": {
            "unique_id": "synthetic-kernel-omp",
            "kernel_name": "kernel",
            "parallel_api": "omp",
        },
        "provenance": {"repo_root": "", "source_path": "tgt_omp"},
        "files": {
            "prompt_payload": ["kernel.cpp"],
            "translation_targets": ["kernel.cpp"],
            "support_files": [],
            "verification_only": ["verify_only.dat"],
        },
        "build": {"commands": {"build": "make"}},
        "verification": {
            "strategies": [
                {"type": "stdout_pattern", "pattern": ORACLE_PATTERN_SENTINEL},
                {"type": "exit_code", "expected": 0},
                {"type": "file_hash", "path": "out.bin", "expected_sha256": ORACLE_SHA256_SENTINEL},
            ],
            "reference_files": [
                {"path": "ref_output.txt", "sha256": "0" * 64},
            ],
        },
    }
    return source_spec, target_spec


def _captured_request(source_spec, target_spec, project_root) -> list[dict[str, str]]:
    """Capture the complete agent-visible request via the real prompt code."""
    source_payload = {"kernel.cu": "// synthetic CUDA source\n__global__ void k() {}\n"}
    system_msg, user_msg, _anon_map = build_translation_prompt(
        source_spec, target_spec, source_payload, project_root
    )
    build_failure = SimpleNamespace(
        status=Status.FAIL,
        stderr="kernel.cpp:3: error: expected ';' (synthetic diagnostics)",
    )
    repair_feedback = _build_retry_message(build_failure, None, None)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "// model output placeholder"},
        {"role": "user", "content": repair_feedback},
    ]


def test_forbidden_collection_has_teeth(synthetic_pair, tmp_path):
    source_spec, target_spec = synthetic_pair
    forbidden = collect_forbidden_agentic_content(source_spec, target_spec, tmp_path)
    joined = {cat: "\n".join(snips) for cat, snips in forbidden.items()}
    assert REFERENCE_IMPL_SENTINEL in joined["reference_implementations"]
    assert REFERENCE_IMPL_SENTINEL in joined["target_implementations"]
    assert VERIFICATION_ONLY_SENTINEL in joined["verification_only_files"]
    assert HIDDEN_EXPECTED_OUTPUT_SENTINEL in joined["hidden_expected_outputs"]
    assert ORACLE_PATTERN_SENTINEL in joined["oracle_contents"]
    assert ORACLE_SHA256_SENTINEL in joined["oracle_contents"]


def test_categories_match_versioned_protocol_config(synthetic_pair, tmp_path):
    source_spec, target_spec = synthetic_pair
    forbidden = collect_forbidden_agentic_content(source_spec, target_spec, tmp_path)
    protocol = json.loads(AGENTIC_PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))
    for category in protocol["visibility"]["forbidden_to_agent"]:
        assert category in forbidden, f"config category {category} has no collector"


def test_captured_request_contains_no_forbidden_content(synthetic_pair, tmp_path):
    source_spec, target_spec = synthetic_pair
    request = _captured_request(source_spec, target_spec, tmp_path)
    forbidden = collect_forbidden_agentic_content(source_spec, target_spec, tmp_path)
    violations = check_agentic_visibility(request, forbidden)
    assert violations == []
    enforce_agentic_visibility(request, forbidden)  # must not raise
    # Belt and suspenders: no sentinel appears verbatim anywhere in the request.
    request_text = "\n".join(m["content"] for m in request)
    for sentinel in (
        REFERENCE_IMPL_SENTINEL,
        VERIFICATION_ONLY_SENTINEL,
        HIDDEN_EXPECTED_OUTPUT_SENTINEL,
        ORACLE_PATTERN_SENTINEL,
        ORACLE_SHA256_SENTINEL,
    ):
        assert sentinel not in request_text


# Each leak is the FULL forbidden content (what "the agent received the
# file/oracle" actually means), not just its sentinel substring.
@pytest.mark.parametrize(
    "leak",
    [
        f"// {REFERENCE_IMPL_SENTINEL}\nvoid k_ref_impl_74ce01() {{}}\n",
        f"{VERIFICATION_ONLY_SENTINEL}\n",
        f"{HIDDEN_EXPECTED_OUTPUT_SENTINEL}\n",
        ORACLE_PATTERN_SENTINEL,
        ORACLE_SHA256_SENTINEL,
    ],
)
def test_poisoned_request_is_detected_and_refused(synthetic_pair, tmp_path, leak):
    source_spec, target_spec = synthetic_pair
    request = _captured_request(source_spec, target_spec, tmp_path)
    forbidden = collect_forbidden_agentic_content(source_spec, target_spec, tmp_path)
    poisoned = request + [
        {"role": "user", "content": f"Here is a hint you should use: {leak}"}
    ]
    assert check_agentic_visibility(poisoned, forbidden) != []
    with pytest.raises(AgenticVisibilityError):
        enforce_agentic_visibility(poisoned, forbidden)


def test_repo_relative_reference_files_are_collected_and_poison_detected():
    # Gate 3 finding (2026-08-12): real specs declare
    # verification.reference_files[].path relative to the PROJECT ROOT
    # ("specs/references/..."), not the benchmark source dir. The collector
    # must pick up every such file, and a request carrying one must be
    # refused. The synthetic fixture above uses a source-dir-relative path,
    # so it never caught this.
    repo_specs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((PROJECT_ROOT / "specs").glob("*.json"))
    ]
    repo_specs = [
        s for s in repo_specs
        if (s.get("verification") or {}).get("reference_files")
    ]
    assert repo_specs, "no repo spec declares verification.reference_files"
    checked = 0
    for spec in repo_specs:
        forbidden = collect_forbidden_agentic_content(spec, spec, PROJECT_ROOT)
        for ref in spec["verification"]["reference_files"]:
            content = (PROJECT_ROOT / ref["path"]).read_text(
                encoding="utf-8", errors="replace"
            )
            assert content in forbidden["hidden_expected_outputs"], ref["path"]
            poisoned = [{"role": "user", "content": f"Expected output:\n{content}"}]
            assert check_agentic_visibility(poisoned, forbidden) != [], ref["path"]
            with pytest.raises(AgenticVisibilityError):
                enforce_agentic_visibility(poisoned, forbidden)
            checked += 1
    assert checked == 6  # bptree x2, dwt2d x3, srad x1 (as of 2026-08-12)


def test_comment_stripped_reference_leak_is_detected(synthetic_pair, tmp_path):
    # Gate finding 4 (2026-08-12): prompts insert file contents AFTER comment
    # stripping, so a leaked reference implementation appears in its stripped
    # form. The checker must screen the exact transformed fragment too.
    source_spec, target_spec = synthetic_pair
    request = _captured_request(source_spec, target_spec, tmp_path)
    forbidden = collect_forbidden_agentic_content(source_spec, target_spec, tmp_path)
    raw_reference = (tmp_path / "tgt_omp" / "kernel.cpp").read_text(encoding="utf-8")
    stripped_leak = _strip_comments(raw_reference)
    assert REFERENCE_IMPL_SENTINEL not in stripped_leak  # the sentinel lives in a comment
    poisoned = request + [{"role": "user", "content": stripped_leak}]
    assert check_agentic_visibility(poisoned, forbidden) != []
    with pytest.raises(AgenticVisibilityError):
        enforce_agentic_visibility(poisoned, forbidden)


def test_real_prompt_with_target_infrastructure_is_clean(synthetic_pair, tmp_path):
    # Gate finding 4 (2026-08-12), corrected semantics: target infrastructure
    # (prompt_payload files that are NOT translation targets) is deliberately
    # shown read-only by the prompt design and is NOT forbidden content. The
    # reference solution (translation_targets) stays forbidden in raw AND
    # comment-stripped form.
    source_spec, target_spec = synthetic_pair
    infra_body = "int infra_helper(int v) { return v + 41; } // infra comment\n"
    (tmp_path / "tgt_omp" / "infra.cpp").write_text(infra_body, encoding="utf-8")
    target_spec["files"]["prompt_payload"] = ["kernel.cpp", "infra.cpp"]

    infrastructure = _read_target_infrastructure(
        target_spec, target_spec["files"]["translation_targets"], tmp_path
    )
    assert infrastructure == {"infra.cpp": infra_body}  # nonempty infra, no targets

    source_payload = {"kernel.cu": "// synthetic CUDA source\n__global__ void k() {}\n"}
    system_msg, user_msg, _anon = build_translation_prompt(
        source_spec, target_spec, source_payload, tmp_path,
        target_infrastructure=infrastructure,
    )
    assert "int infra_helper(int v)" in user_msg  # infra really reached the prompt
    forbidden = collect_forbidden_agentic_content(source_spec, target_spec, tmp_path)
    request = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    assert check_agentic_visibility(request, forbidden) == []
    # The reference solution is still fully forbidden.
    joined = "\n".join(forbidden["target_implementations"])
    assert REFERENCE_IMPL_SENTINEL in joined
    assert "infra_helper" not in joined


def test_short_snippets_flag_unless_in_allowed_baseline():
    # Gate 4 re-check (2026-08-12): the old 6-char exemption let short
    # oracle strings ("PASS", "0.0") reach the provider. Now a snippet of
    # ANY length flags, unless it already occurs in the agent's permitted
    # inputs (allowed_text) - content the agent legitimately holds carries
    # no hidden information.
    request = [{"role": "user", "content": "exit code 0, PASS"}]
    forbidden = {"oracle_contents": ["0", "PASS"]}
    # Strict mode (no baseline): both short snippets flag.
    assert len(check_agentic_visibility(request, forbidden)) == 2
    # The source itself prints PASS: exempt. "0" is not in the baseline: flags.
    allowed = 'printf("PASS\\n");'
    violations = check_agentic_visibility(request, forbidden, allowed)
    assert len(violations) == 1 and "1 chars" in violations[0]
    # Both in the baseline: no violation.
    assert check_agentic_visibility(request, forbidden, 'printf("PASS 0");') == []


# ---------------------------------------------------------------------------
# Gate finding 3 (2026-08-12): protocol retries expose verification status
# only. Real verifier-failure fixtures for every oracle strategy prove that
# verify_result.details (patterns, hashes, numeric expectations, extraction
# regexes) never reaches the agent under the bounded protocol.
# ---------------------------------------------------------------------------

ORACLE_PATTERN_SECRET = "ORACLE_SECRET_PATTERN_77aa11"
ORACLE_HASH_SECRET = "77aa11" * 10 + "cafe"  # 64 hex chars
ORACLE_NUMERIC_SECRET = "123.456"
ORACLE_REGEX_SECRET = r"checksum=(ORACLE_CAPTURE_\d+\.\d+)"


def _failing_run(stdout: str = "wrong output\n", exit_code: int = 1):
    from harness.models import RunResult

    return RunResult(
        status=Status.PASS,
        configuration="correctness",
        duration_seconds=0.1,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


def _verify_failure_fixtures(tmp_path):
    """(strategy-name, spec, run_result, oracle secrets) per oracle strategy."""
    (tmp_path / "out.bin").write_bytes(b"wrong bytes")
    (tmp_path / "out.txt").write_text("actual\n", encoding="utf-8")
    ref = tmp_path / "ref.txt"
    ref.write_text("expected reference content\n", encoding="utf-8")
    return [
        (
            "exit_code",
            {"verification": {"strategies": [{"type": "exit_code", "expected": 0}]}},
            _failing_run(exit_code=7),
            [],
        ),
        (
            "stdout_pattern",
            {"verification": {"strategies": [
                {"type": "stdout_pattern", "pattern": ORACLE_PATTERN_SECRET}]}},
            _failing_run(),
            [ORACLE_PATTERN_SECRET],
        ),
        (
            "stdout_exclude_pattern",
            {"verification": {"strategies": [
                {"type": "stdout_exclude_pattern", "pattern": ORACLE_PATTERN_SECRET}]}},
            _failing_run(stdout=f"bad: {ORACLE_PATTERN_SECRET}\n"),
            [ORACLE_PATTERN_SECRET],
        ),
        (
            "numeric_comparison",
            {"verification": {"strategies": [{
                "type": "numeric_comparison",
                "extract_regex": ORACLE_REGEX_SECRET,
                "expected": 123.456,
            }]}},
            _failing_run(stdout="checksum=ORACLE_CAPTURE_9.9\n"),
            [ORACLE_NUMERIC_SECRET, ORACLE_REGEX_SECRET],
        ),
        (
            "file_hash",
            {"verification": {"strategies": [{
                "type": "file_hash", "path": "out.bin",
                "expected_sha256": ORACLE_HASH_SECRET,
            }]}},
            _failing_run(),
            [ORACLE_HASH_SECRET],
        ),
        (
            "file_diff",
            {"verification": {"strategies": [{
                "type": "file_diff", "path": "out.txt", "reference_file": str(ref),
            }]}},
            _failing_run(),
            ["expected reference content"],
        ),
    ]


def test_protocol_retry_exposes_verify_status_only(tmp_path):
    from harness.verifier import verify_run

    fixtures = _verify_failure_fixtures(tmp_path)
    assert len(fixtures) == 6  # every oracle strategy the verifier implements
    for name, spec, run_result, secrets in fixtures:
        vr = verify_run(spec, run_result, working_dir=tmp_path)
        assert vr.status == Status.FAIL, f"{name}: fixture must genuinely fail, got {vr}"
        msg = _build_retry_message(None, None, vr, verify_status_only=True)
        assert "verification failed" in msg
        assert vr.details not in msg, f"{name}: verifier details leaked into protocol retry"
        for secret in secrets:
            assert secret not in msg, f"{name}: oracle content {secret!r} leaked"
        # The live (non-protocol) eval path is unchanged: details still flow.
        live_msg = _build_retry_message(None, None, vr)
        assert vr.details in live_msg


def test_corpus_wide_oracle_content_cannot_reach_provider_unnoticed():
    """Gate 4 re-check (2026-08-12): across ALL frozen pairs, any oracle
    snippet absent from the agent's permitted inputs must flag when injected
    into a request - including short ones like "PASS". Pairs whose benchmark
    trees are absent on this host contribute no baseline and are skipped
    (the Linux host covers all 204)."""
    from scripts.evaluation.llm_evaluate import collect_allowed_agentic_content

    registry = json.loads(
        (PROJECT_ROOT / "config" / "final_pair_contracts.json").read_text())
    checked = injected = 0
    skipped_pairs = []
    for pair in registry["pairs"]:
        src = json.loads(
            (PROJECT_ROOT / "specs" / f"{pair['source_spec']}.json").read_text())
        tgt = json.loads(
            (PROJECT_ROOT / "specs" / f"{pair['target_spec']}.json").read_text())
        # Baseline = the inserted file text via the production readers;
        # request = the ACTUAL system and user messages composed through the
        # full production prompt path (gate 7 finding 3, 2026-08-12).
        allowed = collect_allowed_agentic_content(src, tgt, PROJECT_ROOT)
        if not allowed.strip():
            skipped_pairs.append(pair["source_spec"])
            continue
        forbidden = collect_forbidden_agentic_content(src, tgt, PROJECT_ROOT)
        from harness.spec_loader import resolve_paths as _rp
        from scripts.evaluation.llm_evaluate import (
            _read_support_files,
            _read_target_infrastructure,
        )
        payload = {}
        for p in _rp(src, PROJECT_ROOT)["_resolved"]["files"].get(
                "prompt_payload", []):
            try:
                payload[Path(p).name] = Path(p).read_text(
                    encoding="utf-8", errors="replace")
            except OSError:
                pass
        system_msg, user_msg, _anon = build_translation_prompt(
            src, tgt, payload, PROJECT_ROOT,
            source_support=_read_support_files(src, PROJECT_ROOT),
            target_infrastructure=_read_target_infrastructure(
                tgt,
                (tgt.get("files") or {}).get("translation_targets", []),
                PROJECT_ROOT,
            ),
        )
        base_request = [{"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}]
        assert check_agentic_visibility(base_request, forbidden, allowed) == [], (
            f"false positive on the real prompt for "
            f"{pair['source_spec']} -> {pair['target_spec']}"
        )
        checked += 1
        for snippet in forbidden.get("oracle_contents", []):
            probe = snippet.strip()
            if not probe or probe in allowed:
                continue
            poisoned = base_request + [{"role": "user", "content": probe}]
            assert check_agentic_visibility(poisoned, forbidden, allowed) != [], (
                f"oracle snippet of {len(probe)} chars not caught for "
                f"{pair['source_spec']} -> {pair['target_spec']}"
            )
            injected += 1
            break  # one injected probe per pair keeps the test fast
    if skipped_pairs:
        pytest.skip(
            f"benchmark trees absent for {len(skipped_pairs)} pair(s) on this "
            f"host ({checked} checked); the source-complete Linux host "
            "asserts all 204"
        )
    assert checked == len(registry["pairs"]) == 204
    assert injected > 0
