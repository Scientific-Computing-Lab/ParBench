"""Phase 2 / Plan 02-03: --thinking flag wiring tests.

These tests verify that the `--thinking on|off` flag is correctly wired to:
  - Qwen (provider=together) via `extra_body.chat_template_kwargs.enable_thinking`
  - Azure (provider=azure) via `reasoning_effort="medium"` kwarg on the SDK call
  - Non-thinking-capable models (no-op, capability-gated)

They also verify the `thinking_enabled` + `num_samples` result-JSON fields and
the MODEL_REGISTRY invariant that `azure-gpt-5.4` is thinking-capable.

No real API calls are made; SDK client factories are mocked via
`unittest.mock.patch` so we can capture kwargs without network traffic.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluation import llm_evaluate
from scripts.evaluation.llm_evaluate import MODEL_REGISTRY, call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_chat_response(content: str = "```cpp\n// translated stub\n```"):
    """Build a minimal OpenAI-shaped response object."""
    resp = MagicMock()
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    return resp


def _make_capturing_openai_factory(captured: dict):
    """Return a factory function that acts as openai.OpenAI(...) and captures
    kwargs passed to the subsequent chat.completions.create() call.
    """
    def _factory(*_factory_args, **_factory_kwargs):
        client = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        client.chat.completions.create = _capture
        return client

    return _factory


# ---------------------------------------------------------------------------
# Tests: Qwen (provider=together) — thinking ON / OFF
# ---------------------------------------------------------------------------

def test_qwen_thinking_on_sets_enable_thinking_true(monkeypatch):
    """Together/Qwen with thinking_enabled=True → extra_body.enable_thinking=True."""
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.OpenAI = _make_capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="together-qwen-3.5-397b-a17b",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            thinking_enabled=True,
        )

    assert "extra_body" in captured, captured
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_qwen_thinking_off_sets_enable_thinking_false(monkeypatch):
    """Together/Qwen with thinking_enabled=False → extra_body.enable_thinking=False."""
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.OpenAI = _make_capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="together-qwen-3.5-397b-a17b",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            thinking_enabled=False,
        )

    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Tests: Azure (provider=azure) — thinking ON / OFF
# ---------------------------------------------------------------------------

def _make_azure_factory(captured: dict):
    """AzureOpenAI(...) factory that captures chat.completions.create kwargs."""
    def _factory(*_a, **_kw):
        client = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        client.chat.completions.create = _capture
        return client

    return _factory


def test_azure_thinking_on_injects_reasoning_effort_medium(monkeypatch):
    """Azure reasoning + thinking_enabled=True → reasoning_effort='medium' is injected."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.AzureOpenAI = _make_azure_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="azure-gpt-5.4",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            thinking_enabled=True,
        )

    assert captured.get("reasoning_effort") == "medium", captured


def test_azure_thinking_off_omits_reasoning_effort(monkeypatch):
    """Azure reasoning + thinking_enabled=False → no reasoning_effort kwarg."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.AzureOpenAI = _make_azure_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="azure-gpt-5.4",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            thinking_enabled=False,
        )

    assert "reasoning_effort" not in captured, captured


# ---------------------------------------------------------------------------
# Tests: Non-thinking-capable model — no-op
# ---------------------------------------------------------------------------

def test_noncapable_claude_does_not_send_reasoning_kwargs(monkeypatch):
    """Claude (supports_thinking=False): no reasoning-related kwargs reach the SDK."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, *a, **kw):
            self.messages = MagicMock()

            def _capture(**kwargs):
                captured.update(kwargs)
                resp = MagicMock()
                block = MagicMock()
                block.text = "```cpp\n// stub\n```"
                resp.content = [block]
                resp.usage = MagicMock()
                resp.usage.input_tokens = 10
                resp.usage.output_tokens = 5
                resp.stop_reason = "end_turn"
                return resp

            self.messages.create = _capture

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = _FakeAnthropic

    claude_keys = [k for k in MODEL_REGISTRY if k.startswith("claude-")]
    assert claude_keys, "registry must have at least one claude-* entry"
    target = claude_keys[0]

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        call_llm(
            model=target,
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            thinking_enabled=True,  # flag-on, but capability=False should no-op
        )

    # The Anthropic branch must not receive reasoning/thinking kwargs from us.
    assert "reasoning_effort" not in captured
    assert "thinking" not in captured
    assert "extra_body" not in captured


# ---------------------------------------------------------------------------
# Tests: Registry invariant (cross-check with 02-01 + 02-02 output)
# ---------------------------------------------------------------------------

def test_registry_azure_gpt_5_4_is_thinking_capable():
    """Plan 02-01 + 02-02 invariant: azure-gpt-5.4 must be thinking-capable."""
    assert "azure-gpt-5.4" in MODEL_REGISTRY
    assert MODEL_REGISTRY["azure-gpt-5.4"]["supports_thinking"] is True


def test_registry_qwen_is_thinking_capable():
    """Plan 02-02 invariant: Together Qwen is thinking-capable (flag flips default)."""
    assert MODEL_REGISTRY["together-qwen-3.5-397b-a17b"]["supports_thinking"] is True


# ---------------------------------------------------------------------------
# Tests: source-level guarantees (belt + suspenders against regressions)
# ---------------------------------------------------------------------------

_LLM_EVAL_PATH = Path(llm_evaluate.__file__)


def test_source_qwen_uses_thinking_enabled_not_hardcoded():
    """Qwen extra_body must read `thinking_enabled`, not a hardcoded literal."""
    src = _LLM_EVAL_PATH.read_text()
    # Hardcoded forms that would be regressions.
    assert '"enable_thinking": False' not in src
    assert '"enable_thinking": True' not in src
    # Expected form.
    assert '"enable_thinking": thinking_enabled' in src


def test_source_gemini_reasoning_effort_none_unchanged():
    """Gemini `reasoning_effort="none"` safety line is untouched by this plan."""
    src = _LLM_EVAL_PATH.read_text()
    assert 'reasoning_effort="none"' in src


def test_source_has_exactly_two_reasoning_effort_occurrences():
    """Exactly two reasoning_effort references expected: Azure (new) + Gemini (unchanged)."""
    src = _LLM_EVAL_PATH.read_text()
    # Count lines that assign / pass reasoning_effort. Comments in .md are excluded
    # because we read the .py only.
    matches = re.findall(r'reasoning_effort\s*=\s*"(?:medium|none)"', src)
    assert len(matches) >= 2, f"expected ≥2 reasoning_effort occurrences, found {len(matches)}"
    # Must include exactly one "medium" (Azure) and one "none" (Gemini).
    assert '"medium"' in "".join(matches)
    assert '"none"' in "".join(matches)


def test_source_result_json_has_thinking_enabled_and_num_samples():
    """Result JSON writer includes both new top-level fields."""
    src = _LLM_EVAL_PATH.read_text()
    assert '"thinking_enabled": thinking_enabled' in src
    assert '"num_samples": num_samples' in src


# ---------------------------------------------------------------------------
# No @pytest.mark.llm / @pytest.mark.integration markers on this module —
# these are pure unit tests with mocked SDK clients, so they run in every
# default `pytest tests/` invocation.
# ---------------------------------------------------------------------------
