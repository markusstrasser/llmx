"""Grok Build CLI subscription transport contracts (offline)."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llmx.auth import resolve_auth
from llmx.cli import cli
from llmx.cli_backends import (
    CLI_PROVIDER_ALIASES,
    CLI_PROVIDER_ALIASES_LITE,
    CLI_PROVIDERS,
    CliBackendFailure,
    _ARG_MAX_BYTES,
    _parse_grok_json,
    cli_chat,
    lite_model_allowed,
    needs_api_fallback,
    resolve_cli_api_fallback,
)
from llmx.dispatch_plan import build_dispatch_plan, resolve_effort
from llmx.providers import LlmxError, infer_provider_from_model


SMOKE_PAYLOAD = {
    "text": "OK",
    "stopReason": "end_turn",
    "sessionId": "01a0732a-test",
    "requestId": "fab4341b-test",
    "thought": "The user wants me to reply with exactly the word OK and nothing else.",
    "usage": {
        "input_tokens": 26055,
        "cache_read_input_tokens": 256,
        "cache_creation_input_tokens": 0,
        "output_tokens": 23,
        "reasoning_tokens": 18,
        "total_tokens": 26334,
    },
    "num_turns": 1,
    "total_cost_usd": 0.00890392,
    "modelUsage": {
        "grok-4.6-build": {
            "inputTokens": 26055,
            "outputTokens": 23,
            "cacheReadInputTokens": 256,
            "cacheCreationInputTokens": 0,
            "modelCalls": 1,
            "costUSD": 0.00890392,
        }
    },
}


def _success_process() -> Mock:
    process = Mock()
    process.pid = 123
    process.returncode = 0
    process.communicate.return_value = (json.dumps(SMOKE_PAYLOAD), "")
    return process


def _plan(**overrides):
    kwargs = {
        "provider": "grok",
        "model": "grok-4.7",
        "reasoning_effort": None,
        "timeout": 300,
        "lite": None,
        "mode": None,
        "auth": None,
        "subscription": False,
        "api_only": None,
        "use_old": False,
    }
    kwargs.update(overrides)
    with patch("llmx.cli_backends.shutil.which", return_value="/usr/local/bin/grok"):
        return build_dispatch_plan(**kwargs)


def test_provider_aliases_and_prefix_are_explicit() -> None:
    assert CLI_PROVIDERS["grok-cli"] == {"binary": "grok", "api_fallback": None}
    assert CLI_PROVIDER_ALIASES["grok"] == "grok-cli"
    assert CLI_PROVIDER_ALIASES_LITE["grok"] == "grok-cli"
    assert infer_provider_from_model("grok/grok-4.7") == "grok"
    assert infer_provider_from_model("grok-4.7") == "xai"


def test_grok_defaults_to_subscription_and_api_auth_is_rejected() -> None:
    auth, source, _, _ = resolve_auth(provider="grok")
    assert (auth, source) == ("subscription", "default_policy")
    with pytest.raises(ValueError, match="subscription-only"):
        resolve_auth(provider="grok", auth="api")


def test_grok_and_grok_cli_default_model_and_transport() -> None:
    for provider in ("grok", "grok-cli"):
        plan = _plan(provider=provider, model=None)
        assert plan.provider == provider
        assert plan.model == "grok-4.7"
        assert plan.transport == "grok-cli"
        assert plan.auth == "subscription"
        assert not any("not on lite allowlist" in warning for warning in plan.warnings)


def test_grok_allowlist_is_transport_scoped() -> None:
    assert lite_model_allowed("grok-4.7", transport="grok-cli")
    assert not lite_model_allowed("grok-4.7")
    assert not lite_model_allowed("grok-4.7", transport="cursor-cli")
    # grok-4.6 retired 2026-09-25: refused on every transport.
    assert not lite_model_allowed("grok-4.6", transport="grok-cli")
    assert not lite_model_allowed("grok-4.6")


def test_bare_grok_subscription_still_rewrites_to_cursor() -> None:
    plan = _plan(provider=None, model="grok-4.7", subscription=True)
    assert plan.provider == "cursor"
    assert plan.transport == "cursor-cli"
    assert plan.model == "grok-4.7-high"
    with pytest.raises(ValueError, match="grok-4.6 is no longer"):
        _plan(provider=None, model="grok-4.6", subscription=True)


def test_effort_mapping_table() -> None:
    expected = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "xhigh",
    }
    for requested, applied in expected.items():
        actual, warnings = resolve_effort(
            requested,
            transport="grok-cli",
            provider="grok",
            model="grok-4.7",
        )
        assert actual == applied
        assert bool(warnings) is (actual != requested or requested == "max")


def test_parse_smoke_json_preserves_requested_and_served_usage() -> None:
    result, usage = _parse_grok_json(json.dumps(SMOKE_PAYLOAD))
    assert result == "OK"
    assert usage == {
        "prompt_tokens": 26055,
        "completion_tokens": 23,
        "reasoning_tokens": 18,
        "cached_tokens": 256,
        "total_cost_usd": 0.00890392,
        "served_model": "grok-4.6-build",
    }


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"text": "partial", "stopReason": "max_tokens"}, "stopReason='max_tokens'"),
        ({"stopReason": "end_turn"}, "no response text"),
        ({"text": "", "stopReason": "end_turn"}, "no response text"),
    ],
)
def test_parse_rejects_non_end_turn_and_missing_text(payload, detail) -> None:
    result, usage = _parse_grok_json(json.dumps(payload))
    assert isinstance(result, CliBackendFailure)
    assert result.kind is LlmxError
    assert detail in result.detail
    assert usage is None


def test_chat_command_is_read_only_neutral_and_subscription_authenticated() -> None:
    process = _success_process()
    with (
        patch("llmx.cli_backends.subprocess.Popen", return_value=process) as popen,
        patch("llmx.cli_backends._grok_cwd", return_value="/tmp/neutral-grok"),
        patch("llmx.usage_log.log_usage") as log_usage,
        patch.dict(
            os.environ,
            {"XAI_API_KEY": "must-not-leak", "GROK_API_KEY": "must-not-leak"},
        ),
    ):
        result = cli_chat("grok-cli", "hi", "grok-4.7", 30, reasoning_effort="high")

    assert result == "OK"
    command = popen.call_args.args[0]
    invocation = popen.call_args.kwargs
    assert command == [
        "grok",
        "-p",
        "hi",
        "--output-format",
        "json",
        "--no-plan",
        "--permission-mode",
        "plan",
        "-m",
        "grok-4.7",
        "--reasoning-effort",
        "high",
    ]
    assert invocation["cwd"] == "/tmp/neutral-grok"
    assert "XAI_API_KEY" not in invocation["env"]
    assert "GROK_API_KEY" not in invocation["env"]
    logged = log_usage.call_args.kwargs
    assert logged["model"] == "grok-4.7"
    assert logged["served_model"] == "grok-4.6-build"
    assert logged["billing"] == "subscription"
    assert logged["reported_cost_usd"] == 0.00890392


def test_agent_command_uses_caller_cwd_and_bypass_permissions() -> None:
    process = _success_process()
    with (
        patch("llmx.cli_backends.subprocess.Popen", return_value=process) as popen,
        patch("llmx.usage_log.log_usage"),
    ):
        result = cli_chat("grok-cli", "inspect", "grok-4.7", 30, mode="agent")

    assert result == "OK"
    command = popen.call_args.args[0]
    assert command[command.index("--permission-mode") + 1] == "bypassPermissions"
    assert popen.call_args.kwargs["cwd"] is None


def test_long_prompt_uses_deleted_prompt_file() -> None:
    process = _success_process()
    captured = {}

    def launch(command, **_kwargs):
        prompt_path = Path(command[command.index("--prompt-file") + 1])
        captured["path"] = prompt_path
        captured["text"] = prompt_path.read_text()
        return process

    prompt = "x" * (_ARG_MAX_BYTES + 1)
    with (
        patch("llmx.cli_backends.subprocess.Popen", side_effect=launch) as popen,
        patch("llmx.cli_backends._grok_cwd", return_value="/tmp/neutral-grok"),
        patch("llmx.usage_log.log_usage"),
    ):
        result = cli_chat("grok-cli", prompt, "grok-4.7", 30)

    assert result == "OK"
    command = popen.call_args.args[0]
    assert "-p" not in command
    assert captured["text"] == prompt
    assert not captured["path"].exists()


def test_missing_binary_is_precise_and_never_falls_back() -> None:
    with patch("llmx.cli_backends.shutil.which", return_value=None):
        reason = needs_api_fallback("grok-cli", None, None, False, False, None)
    assert reason == "grok not found in PATH"
    with pytest.raises(RuntimeError, match="grok-cli failed.*grok not found in PATH"):
        resolve_cli_api_fallback("grok-cli", auth="subscription", reason=reason)


def test_missing_binary_cli_failure_names_grok_transport() -> None:
    with patch("llmx.cli_backends.shutil.which", return_value=None):
        result = CliRunner().invoke(cli, ["chat", "-p", "grok", "hi"])

    assert result.exit_code == 1
    assert "grok-cli failed (grok not found in PATH)" in result.output
    assert "forbids metered API fallback" in result.output
