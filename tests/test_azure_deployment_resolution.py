"""Verify Azure deployment name resolution in call_llm().

The Azure path at llm_evaluate.py:917 resolves the deployment name via
MODEL_REGISTRY api_model (if set) with prefix-strip fallback. These tests
mock openai.AzureOpenAI to capture the deployment name without real API calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.evaluation.llm_evaluate import call_llm, MODEL_REGISTRY


def _make_fake_response(text="OK"):
    """Build a minimal fake OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _call_azure_and_get_deployment(monkeypatch, model_key, registry_patch=None):
    """Call call_llm with an Azure model and return the deployment name sent to the API."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")

    if registry_patch is not None:
        monkeypatch.setitem(MODEL_REGISTRY, model_key, registry_patch)

    with patch("openai.AzureOpenAI") as mock_az_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_fake_response()
        mock_az_cls.return_value = mock_client

        call_llm(
            model=model_key,
            system_msg="test",
            messages=[{"role": "user", "content": "say hi"}],
            temperature=0.7,
            thinking_enabled=True,
            verbose=False,
        )

        return mock_client.chat.completions.create.call_args.kwargs["model"]


def test_azure_gpt54_sends_correct_deployment_name(monkeypatch):
    """azure-gpt-5.4 (no api_model) should resolve via prefix-strip to 'gpt-5.4'."""
    actual = _call_azure_and_get_deployment(monkeypatch, "azure-gpt-5.4")
    assert actual == "gpt-5.4", f"Expected 'gpt-5.4' but got '{actual}'"


def test_azure_api_model_override_takes_precedence(monkeypatch):
    """When api_model is set in MODEL_REGISTRY, it should override prefix-stripping."""
    registry_entry = {
        "provider": "azure",
        "supports_thinking": True,
        "api_model": "my-custom-deployment",
    }
    actual = _call_azure_and_get_deployment(
        monkeypatch, "azure-gpt-5.4", registry_patch=registry_entry,
    )
    assert actual == "my-custom-deployment", (
        f"Expected 'my-custom-deployment' but got '{actual}'"
    )
