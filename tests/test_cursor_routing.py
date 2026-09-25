"""Lock cursor transport routing against accidental paid-API fallback."""

from llmx.cli_backends import lite_model_allowed
from llmx.model_ids import CURSOR_GROK46_RETIRED_MODELS, CURSOR_GROK47_MODELS
from llmx.providers import infer_provider_from_model as infer


def test_cursor_prefix_overrides_substring_families() -> None:
    for model in (
        "cursor/gemini-3-flash",
        "cursor/kimi-k2.5",
        "cursor/grok-4",
        "cursor/minimax-m3",
        "cursor/qwen-3",
        "cursor/deepseek-v3",
        "cursor/claude-opus-5-5",
        "cursor/gpt-6-sol",
        "cursor/gpt-6-astra",
    ):
        assert infer(model) == "cursor", f"{model} must route to cursor"


def test_bare_composer_is_cursor() -> None:
    assert infer("composer-2.5") == "cursor"
    assert infer("composer-2.5-fast") == "cursor"


def test_retired_grok46_slugs_are_refused() -> None:
    assert CURSOR_GROK46_RETIRED_MODELS == (
        "cursor-grok-4.6-low",
        "cursor-grok-4.6-low-fast",
        "cursor-grok-4.6-medium",
        "cursor-grok-4.6-medium-fast",
        "cursor-grok-4.6-high",
        "cursor-grok-4.6-high-fast",
        "cursor-grok-4.6-xhigh",
        "cursor-grok-4.6-xhigh-fast",
    )
    for model in CURSOR_GROK46_RETIRED_MODELS:
        assert not lite_model_allowed(model), f"{model} is retired"


def test_cursor_native_grok47_effort_slugs() -> None:
    # `cursor-agent models`, 2026-09-23: 4.7 slugs carry no `cursor-` prefix.
    assert CURSOR_GROK47_MODELS == (
        "grok-4.7-low",
        "grok-4.7-low-fast",
        "grok-4.7-medium",
        "grok-4.7-medium-fast",
        "grok-4.7-high",
        "grok-4.7-high-fast",
        "grok-4.7-xhigh",
        "grok-4.7-xhigh-fast",
    )
    for model in CURSOR_GROK47_MODELS:
        assert infer(model) == "cursor", f"{model} must route to cursor"


def test_unprefixed_grok_slugs_are_gated_exactly() -> None:
    from llmx.cli_backends import lite_model_allowed

    assert lite_model_allowed("grok-4.7-high")
    assert lite_model_allowed("grok-4.7-xhigh-fast")
    # Invented or retired shapes fail closed instead of prefix-matching a real slug.
    assert not lite_model_allowed("grok-4.7-max")
    assert not lite_model_allowed("grok-4.7-high-fast-x")
    assert not lite_model_allowed("cursor-grok-4.7-high")
    # Grok Build ids pass only on the grok-cli transport.
    assert lite_model_allowed("grok-4.7-build-fast", transport="grok-cli")
    assert not lite_model_allowed("grok-4.7-build-fast", transport="cursor-cli")


def test_subscription_allowlist_is_exact_for_cursor_grok() -> None:
    for model in CURSOR_GROK47_MODELS:
        assert lite_model_allowed(model)
    for retired_or_invented in (
        "grok-4.7",
        "grok-4.6",
        "grok-4.6-high",
        "grok-4.6-fast-high",
        "cursor-grok-4.6-high",
        "cursor-grok-4.6-ultra",
        "cursor-grok-4.6-high-preview",
    ):
        assert not lite_model_allowed(retired_or_invented)


def test_non_cursor_models_keep_native_routes() -> None:
    expected = {
        "gemini-3-flash": "google",
        "kimi-k2.5": "kimi",
        "grok-4": "xai",
        "grok-4.5": "xai",
        "grok-4.7": "xai",
        "minimax-m3": "minimax",
        "qwen-3": "cerebras",
        "gpt-6-sol": "openai",
        "gpt-6-astra": "openai",
        "claude-opus-5-5": "anthropic",
        "deepseek-v3": "deepseek",
        "openrouter/x": "openrouter",
    }
    for model, provider in expected.items():
        assert infer(model) == provider, f"{model}: expected {provider}"
