"""Offline contract for Cerebras's current public shared-endpoint catalog."""

from types import SimpleNamespace
from unittest.mock import patch

from llmx import providers
from llmx.providers import (
    MODEL_RESTRICTIONS,
    PROVIDER_CONFIGS,
    _KNOWN_MODELS,
    get_model_restriction,
)
from llmx.usage_report import CONTEXT_WINDOW, PRICING


def test_cerebras_default_is_current_public_qwen():
    assert PROVIDER_CONFIGS["cerebras"]["model"] == "qwen-3.8-27b"
    assert _KNOWN_MODELS["cerebras"] == ["qwen-3.8-27b", "gpt-oss-120b"]


def test_cerebras_public_models_are_priced_and_windowed():
    assert PRICING["qwen-3.8-27b"] == (0.99, 1.49)
    assert PRICING["gpt-oss-120b"] == (0.35, 0.75)
    assert CONTEXT_WINDOW["qwen-3.8-27b"] == 131_072
    assert CONTEXT_WINDOW["gpt-oss-120b"] == 131_072


def test_cerebras_public_models_declare_their_reasoning_contracts():
    assert get_model_restriction("qwen-3.8-27b") == {
        "reasoning_effort": True,
        "reasoning_effort_levels": ["none", "low", "medium", "high"],
    }
    assert get_model_restriction("gpt-oss-120b") == {
        "reasoning_effort": True,
        "reasoning_effort_levels": ["low", "medium", "high"],
    }
    assert "none" not in MODEL_RESTRICTIONS["gpt-oss-120b"][
        "reasoning_effort_levels"
    ]


def test_cerebras_qwen_reasoning_none_reaches_the_outgoing_request():
    calls = []
    client_kwargs = []

    def create(**kwargs):
        calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="OK", refusal=None),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    def make_client(**kwargs):
        client_kwargs.append(kwargs)
        return client

    with (
        patch("llmx.providers.OpenAI", side_effect=make_client),
        patch("llmx.providers._get_api_key", return_value="test-key"),
        patch("llmx.providers.check_api_key", return_value=None),
        patch("llmx.spend_guard.enforce_daily_cap", return_value=None),
        patch("llmx.usage_log.log_usage", return_value=None),
    ):
        providers.chat(
            prompt="Reply exactly OK.",
            provider="cerebras",
            model="qwen-3.8-27b",
            temperature=0.7,
            reasoning_effort="none",
            stream=False,
            debug=False,
            json_output=False,
            timeout=30,
        )

    assert len(calls) == 1
    assert client_kwargs[0]["max_retries"] == 0
    assert calls[0]["reasoning_effort"] == "none"
