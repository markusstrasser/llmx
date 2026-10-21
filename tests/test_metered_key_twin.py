"""Metered twin keys.

The shell exports paid per-token keys only as <NAME>_METERED (policy:
~/dotfiles/scripts/secret-env-policy.sh) so no SDK finds them by the standard
name. llmx is the sanctioned consumer and promotes the twin in-process.
"""

import os

import pytest
from click.testing import CliRunner

from llmx import providers
from llmx.cli import cli

NAMES = (
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture
def env(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name + providers.METERED_SUFFIX, raising=False)
    monkeypatch.setattr(providers, "_keychain_get", lambda name: None)
    return monkeypatch


def test_twin_is_promoted_to_the_standard_name(env):
    env.setenv("OPENAI_API_KEY_METERED", "sk-twin")
    providers.check_api_key("openai")
    assert os.environ["OPENAI_API_KEY"] == "sk-twin"


def test_standard_name_wins_over_twin(env):
    env.setenv("OPENAI_API_KEY", "sk-standard")
    env.setenv("OPENAI_API_KEY_METERED", "sk-twin")
    assert providers._get_api_key("openai") == "sk-standard"


def test_alias_twin_resolves(env):
    env.setenv("GROK_API_KEY_METERED", "xai-twin")
    assert providers._get_api_key("xai") == "xai-twin"
    assert os.environ["GROK_API_KEY"] == "xai-twin"


def test_override_key_twin_resolves(env):
    # anthropic (via OpenRouter) resolves through API_KEY_OVERRIDES.
    env.setenv("OPENROUTER_API_KEY_METERED", "or-twin")
    assert providers._get_api_key("anthropic") == "or-twin"


def test_missing_key_error_names_the_twin(env):
    with pytest.raises(RuntimeError, match="_METERED"):
        providers.check_api_key("openai")


def test_twin_shadowing_a_different_keychain_key_is_named(env):
    env.setenv("OPENAI_API_KEY_METERED", "sk-revoked")
    env.setattr(providers, "_keychain_get", lambda name: "sk-valid")
    assert providers.keychain_shadow("OPENAI_API_KEY") == "OPENAI_API_KEY_METERED"
    providers.check_api_key("openai")  # promotion must not hide the twin's origin
    assert providers.keychain_shadow("OPENAI_API_KEY") == "OPENAI_API_KEY_METERED"
    assert "OPENAI_API_KEY_METERED" in providers.api_key_error_hint("openai")


def test_matching_or_absent_keychain_key_is_not_a_shadow(env):
    env.setenv("OPENAI_API_KEY_METERED", "sk-valid")
    assert providers.keychain_shadow("OPENAI_API_KEY") is None
    env.setattr(providers, "_keychain_get", lambda name: "sk-valid")
    assert providers.keychain_shadow("OPENAI_API_KEY") is None
    assert "llmx keys get OPENAI_API_KEY" in providers.api_key_error_hint("openai")


def test_keys_set_warns_when_the_environment_shadows_the_stored_key(env):
    env.setenv("OPENAI_API_KEY_METERED", "sk-revoked")
    env.setattr(providers, "_keychain_available", lambda: True)
    env.setattr(providers, "_keychain_set", lambda name, value: True)
    env.setattr(providers, "_keychain_get", lambda name: "sk-valid")
    result = CliRunner().invoke(cli, ["keys", "set", "OPENAI_API_KEY", "--value", "sk-valid"])
    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY_METERED is set in the environment" in result.output
    assert "sk-" not in result.output
