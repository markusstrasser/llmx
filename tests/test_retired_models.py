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
