"""Offline contract for Cerebras's current public shared-endpoint catalog."""

from llmx.providers import PROVIDER_CONFIGS, _KNOWN_MODELS
from llmx.usage_report import CONTEXT_WINDOW, PRICING


def test_cerebras_default_is_current_public_qwen():
    assert PROVIDER_CONFIGS["cerebras"]["model"] == "qwen-3.8-27b"
    assert _KNOWN_MODELS["cerebras"] == ["qwen-3.8-27b", "gpt-oss-120b"]


def test_cerebras_public_models_are_priced_and_windowed():
    assert PRICING["qwen-3.8-27b"] == (0.99, 1.49)
    assert PRICING["gpt-oss-120b"] == (0.35, 0.75)
    assert CONTEXT_WINDOW["qwen-3.8-27b"] == 131_072
    assert CONTEXT_WINDOW["gpt-oss-120b"] == 131_072
