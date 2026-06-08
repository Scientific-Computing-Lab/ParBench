#!/usr/bin/env python3
"""Smoke test for Azure OpenAI gpt-5.3-codex via the Responses API.

Usage:
    export AZURE_OPENAI_API_KEY_CODEX="<your-key>"
    export AZURE_OPENAI_ENDPOINT_CODEX="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com"
    python3 scripts/smoke_test_codex.py
"""
from __future__ import annotations

import os
import sys
import time
from urllib.parse import urlparse

CUDA_SNIPPET = """\
__global__ void add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}"""


def main() -> int:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY_CODEX")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_CODEX")
    if not api_key:
        print("FATAL: Set AZURE_OPENAI_API_KEY_CODEX environment variable.")
        return 1
    if not endpoint:
        print("FATAL: Set AZURE_OPENAI_ENDPOINT_CODEX environment variable.")
        return 1

    try:
        from openai import AzureOpenAI
    except ImportError:
        print("FATAL: openai package not installed. Run: python3 -m pip install openai")
        return 1

    parsed = urlparse(endpoint)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}"

    print(f"Endpoint: {base_endpoint}")
    print(f"Model:    gpt-5.3-codex")
    print(f"API:      Responses (api_version=2025-04-01-preview)")
    print()

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=base_endpoint,
        api_version="2025-04-01-preview",
    )

    prompt = f"Translate this CUDA kernel to OpenMP:\n```cuda\n{CUDA_SNIPPET}\n```"
    print(f"Prompt ({len(prompt)} chars): {prompt[:80]}...")
    print()

    t0 = time.time()
    try:
        response = client.responses.create(
            model="gpt-5.3-codex",
            instructions="You are a parallel programming expert. Return only the translated code.",
            input=prompt,
            max_output_tokens=4096,
            reasoning={"effort": "medium"},
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"ERROR after {elapsed:.1f}s: {exc}")
        print()
        print("If you get a 404, the deployment name may differ from 'gpt-5.3-codex'.")
        print("Check your Azure portal > Model deployments for the exact name.")
        return 1

    elapsed = time.time() - t0

    print("=== Response ===")
    print(f"Status:           {response.status}")
    print(f"Input tokens:     {response.usage.input_tokens}")
    print(f"Output tokens:    {response.usage.output_tokens}")
    print(f"Total tokens:     {response.usage.total_tokens}")
    print(f"Duration:         {elapsed:.1f}s")
    print()
    print("=== Output text ===")
    print(response.output_text or "(empty)")
    print()

    if response.status == "completed" and response.output_text:
        print("SMOKE TEST PASSED — Responses API is working.")
        return 0
    else:
        print(f"SMOKE TEST FAILED — status={response.status}, output_text is empty.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
