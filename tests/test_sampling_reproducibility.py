"""Plan 02-10 Step 2 (C1-C4): sampling reproducibility plumbing tests.

Verifies that `_derive_llm_seed()` is deterministic across processes and that
the `seed` + `top_p` kwargs flow into the Together AI and Azure provider
branches. Reasoning-model paths (o1/o3/o4) must omit `top_p` alongside
`temperature`.

No real API calls; SDK client factories are mocked via `unittest.mock.patch`.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.evaluation.llm_evaluate import _derive_llm_seed, call_llm


# ---------------------------------------------------------------------------
# _derive_llm_seed: purity + determinism + differentiation
# ---------------------------------------------------------------------------

def test_derive_llm_seed_deterministic_same_inputs():
    s1 = _derive_llm_seed("rodinia-bfs-cuda", "rodinia-bfs-omp", 0)
    s2 = _derive_llm_seed("rodinia-bfs-cuda", "rodinia-bfs-omp", 0)
    assert s1 == s2


def test_derive_llm_seed_differs_across_sample_ids():
    s0 = _derive_llm_seed("rodinia-bfs-cuda", "rodinia-bfs-omp", 0)
    s1 = _derive_llm_seed("rodinia-bfs-cuda", "rodinia-bfs-omp", 1)
    s2 = _derive_llm_seed("rodinia-bfs-cuda", "rodinia-bfs-omp", 2)
    assert len({s0, s1, s2}) == 3


def test_derive_llm_seed_differs_across_spec_pair():
    s_bfs = _derive_llm_seed("rodinia-bfs-cuda", "rodinia-bfs-omp", 0)
    s_nw = _derive_llm_seed("rodinia-nw-cuda", "rodinia-nw-omp", 0)
    assert s_bfs != s_nw


def test_derive_llm_seed_in_31bit_range():
    s = _derive_llm_seed("a", "b", 0)
    assert 0 <= s < 2**31


# ---------------------------------------------------------------------------
# Helpers (mirror test_thinking_flag conventions)
# ---------------------------------------------------------------------------

def _mock_chat_response(content: str = "stub"):
    resp = MagicMock()
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _capturing_openai_factory(captured: dict):
    def _factory(*_a, **_kw):
        client = MagicMock()

        def _capture(**kwargs):
            captured.update(kwargs)
            return _mock_chat_response()

        client.chat.completions.create = _capture
        return client

    return _factory


# ---------------------------------------------------------------------------
# Together AI: seed + top_p flow into create() kwargs
# ---------------------------------------------------------------------------

def test_together_call_includes_seed_and_top_p(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.OpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="together-qwen-3.5-397b-a17b",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            seed=12345,
            top_p=1.0,
        )

    assert captured.get("seed") == 12345
    assert captured.get("top_p") == 1.0


def test_together_call_omits_seed_when_none(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.OpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="together-qwen-3.5-397b-a17b",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            seed=None,
            top_p=1.0,
        )

    assert "seed" not in captured
    assert captured.get("top_p") == 1.0


# ---------------------------------------------------------------------------
# Azure reasoning models (gpt-5.4, supports_thinking=True):
#   seed flows; temperature + top_p are OMITTED (Azure API 400s with
#   temperature != 1 on reasoning deployments — surfaced 2026-04-19 S7 smoke).
# ---------------------------------------------------------------------------

def test_azure_reasoning_includes_seed(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.AzureOpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="azure-gpt-5.4",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            seed=98765,
            top_p=1.0,
        )

    assert captured.get("seed") == 98765


def test_azure_reasoning_omits_temperature_and_top_p(monkeypatch):
    """Azure reasoning deployments (supports_thinking=True) reject
    temperature != 1 and top_p. Pipeline must omit both kwargs to avoid 400s.
    Surfaced 2026-04-19 in S7 pre-flight live smoke; see .planning/phases/
    03-oracle-framework/04-S7-LIVE.log.
    """
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.AzureOpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="azure-gpt-5.4",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=1.0,
        )

    assert "temperature" not in captured, captured
    assert "top_p" not in captured, captured


def test_azure_reasoning_omits_seed_when_none(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.AzureOpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="azure-gpt-5.4",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            seed=None,
            top_p=1.0,
        )

    assert "seed" not in captured
    assert "top_p" not in captured, captured


# ---------------------------------------------------------------------------
# OpenAI reasoning models: top_p omitted alongside temperature
# ---------------------------------------------------------------------------

def test_openai_reasoning_model_omits_top_p_and_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.OpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="o3-mini",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=1.0,
        )

    assert "top_p" not in captured
    assert "temperature" not in captured


def test_openai_nonreasoning_model_includes_top_p(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict = {}
    fake_openai = MagicMock()
    fake_openai.OpenAI = _capturing_openai_factory(captured)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        call_llm(
            model="gpt-4.1",
            system_msg="sys",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=1.0,
        )

    assert captured.get("top_p") == 1.0
    assert captured.get("temperature") == 0.7
