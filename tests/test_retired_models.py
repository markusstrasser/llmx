"""Retired model ids (2026-09-25 Pareto-frontier prune) must fail closed."""

import pytest

from llmx.cli_backends import lite_model_allowed
from llmx.model_ids import CURSOR_GROK46_RETIRED_MODELS
from llmx.providers import infer_provider_from_model

RETIRED = (
    "claude-fable-5",
    "claude-opus-4-8",
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gemini-3-flash-preview",
    "grok-4.6",
    *CURSOR_GROK46_RETIRED_MODELS,
)


@pytest.mark.parametrize("model", RETIRED)
def test_retired_ids_are_not_lite_allowed(model: str) -> None:
    assert lite_model_allowed(model) is False
    for transport in ("claude-cli", "codex-cli", "cursor-cli", "grok-cli"):
        assert lite_model_allowed(model, transport=transport) is False


@pytest.mark.parametrize("model", CURSOR_GROK46_RETIRED_MODELS)
def test_retired_cursor_grok_never_routes_to_paid_xai(model: str) -> None:
    assert infer_provider_from_model(model) == "cursor"


def test_frontier_successors_stay_allowed() -> None:
    for model in ("gpt-6-sol", "gpt-6-luna", "claude-fable-5-1", "claude-opus-5-5", "grok-4.7-high"):
        assert lite_model_allowed(model)


# Pass 2 (2026-09-25): dominated API ids fail closed at dispatch with a successor.
from llmx.model_ids import RETIRED_API_MODELS  # noqa: E402
from llmx.providers import _KNOWN_MODELS, _auto_upgrade_model  # noqa: E402


@pytest.mark.parametrize("model", sorted(RETIRED_API_MODELS))
def test_retired_api_ids_refuse_with_successor(model: str) -> None:
    with pytest.raises(ValueError, match="retired 2026-09-25"):
        _auto_upgrade_model(model)
    assert all(model not in ids for ids in _KNOWN_MODELS.values())


def test_eval_pinned_gpt53_stays_admitted() -> None:
    assert _auto_upgrade_model("gpt-5.3-chat-latest") == "gpt-5.3-chat-latest"
    assert "gpt-5.3-chat-latest" in _KNOWN_MODELS["openai"]


def test_pass2_successors_route() -> None:
    for model in ("gpt-6-luna", "gpt-6-sol", "grok-4.7", "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite"):
        assert _auto_upgrade_model(model) == model
