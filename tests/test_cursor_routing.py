"""Lock cursor transport routing against accidental paid-API fallback."""

from llmx.providers import infer_provider_from_model as infer


def test_cursor_prefix_overrides_substring_families() -> None:
    for model in (
        "cursor/gemini-3-flash",
        "cursor/kimi-k2.5",
        "cursor/grok-4",
        "cursor/minimax-m3",
        "cursor/qwen-3",
        "cursor/deepseek-v3",
        "cursor/claude-opus-4-8",
        "cursor/gpt-5.6-sol",
    ):
        assert infer(model) == "cursor", f"{model} must route to cursor"


def test_bare_composer_is_cursor() -> None:
    assert infer("composer-2.5") == "cursor"
    assert infer("composer-2.5-fast") == "cursor"


def test_cursor_native_grok45_effort_slugs() -> None:
    for model in (
        "grok-4.5-medium",
        "grok-4.5-high",
        "grok-4.5-xhigh",
        "grok-4.5-fast-medium",
        "grok-4.5-fast-high",
        "grok-4.5-fast-xhigh",
    ):
        assert infer(model) == "cursor", f"{model} must route to cursor"


def test_non_cursor_models_keep_native_routes() -> None:
    expected = {
        "gemini-3-flash": "google",
        "kimi-k2.5": "kimi",
        "grok-4": "xai",
        "grok-4.5": "xai",
        "minimax-m3": "minimax",
        "qwen-3": "cerebras",
        "gpt-5.6-sol": "openai",
        "claude-opus-4-8": "anthropic",
        "deepseek-v3": "deepseek",
        "openrouter/x": "openrouter",
    }
    for model, provider in expected.items():
        assert infer(model) == provider, f"{model}: expected {provider}"
