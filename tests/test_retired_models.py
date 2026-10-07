"""Retired model ids (2026-09-25 Pareto-frontier prune, 2026-10-07 Composer) fail closed."""

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
    "composer-2.5",
    "composer-2.5-fast",
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
    with pytest.raises(ValueError, match=r"retired 2026-(09-25|10-07)"):
        _auto_upgrade_model(model)
    assert all(model not in ids for ids in _KNOWN_MODELS.values())


def test_gpt53_retired_to_luna() -> None:
    # Operator ruling "6 luna wins" (2026-09-25) revoked the bake-off exception.
    for model in ("gpt-5.3", "gpt-5.3-chat-latest", "gpt-5.3-codex"):
        assert RETIRED_API_MODELS[model] == "gpt-6-luna"
        with pytest.raises(ValueError, match="use -m gpt-6-luna"):
            _auto_upgrade_model(model)


@pytest.mark.parametrize(
    "model", ["composer-2.5", "composer-2.5-fast", "cursor/composer-2.5", "cursor/composer-2.5-fast"]
)
def test_composer_retired_to_astra_low(model: str) -> None:
    # Operator 2026-10-07: "llmx shouldn't use composer 2.5 anymore ... it's outdated".
    with pytest.raises(
        ValueError, match=r"retired 2026-10-07 .*use -m gpt-6-astra --subscription -e low"
    ):
        _auto_upgrade_model(model)


def test_pass2_successors_route() -> None:
    for model in ("gpt-6-luna", "gpt-6-sol", "grok-4.7", "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite"):
        assert _auto_upgrade_model(model) == model


@pytest.mark.parametrize("model", ["composer-2.5", "gpt-5.3"])
def test_cli_refuses_retired_id_with_exit_2(model: str) -> None:
    # A retired id is a usage error (exit 2) naming the successor, never untyped exit 1.
    from click.testing import CliRunner

    from llmx.cli import cli

    result = CliRunner().invoke(cli, ["chat", "-m", model, "hi"])
    assert result.exit_code == 2, result.output
    assert f"use -m {RETIRED_API_MODELS[model]}" in result.output
