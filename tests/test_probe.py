"""Tests for the explicit, subscription-only live probe boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from llmx.api import Response
from llmx.cli import cli
from llmx.probe import run_subscription_probe
from llmx.providers import QuotaError, RateLimitError


NOW = datetime(2026, 7, 10, 17, 30, tzinfo=UTC)


def _response(content: str = "OK") -> Response:
    return Response(
        content=content,
        provider="claude-cli",
        model="claude-opus-5-5",
        usage={},
        latency=0.1,
        raw=None,
    )


def _client_factory(*_args, **_kwargs):
    client = Mock()
    client.chat.return_value = _response()
    return client


def test_success_is_typed_and_cached_without_second_call(tmp_path: Path) -> None:
    client = _client_factory()
    client_factory = Mock(return_value=client)
    first = run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[10.0, 10.25]),
        client_factory=client_factory,
    )
    second = run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        now_fn=lambda: NOW + timedelta(minutes=1),
        client_factory=client_factory,
    )

    assert first.verdict == "available"
    assert first.exit_code == 0
    assert first.response_exact_ok is True
    assert first.cached is False
    assert second.cached is True
    assert client_factory.call_count == 1
    client.chat.assert_called_once_with(
        "Reply exactly OK.",
        reasoning_effort="low",
        timeout=120,
    )


def test_quota_failure_preserves_exit_six_and_never_falls_back(tmp_path: Path) -> None:
    client = Mock()
    client.chat.side_effect = QuotaError(
        "You've hit your monthly spend limit",
        provider="claude-cli",
        model="claude-opus-5-5",
        status_code=429,
    )
    result = run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[1.0, 1.1]),
        client_factory=Mock(return_value=client),
    )

    assert result.verdict == "unavailable"
    assert result.exit_code == 6
    assert result.error_type == "quota_exhausted"
    assert result.status_code == 429


def test_transient_rate_limit_is_indeterminate(tmp_path: Path) -> None:
    client = Mock()
    client.chat.side_effect = RateLimitError(
        "try later",
        provider="claude-cli",
        model="claude-opus-5-5",
        status_code=429,
    )
    result = run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[1.0, 1.1]),
        client_factory=Mock(return_value=client),
    )

    assert result.verdict == "indeterminate"
    assert result.exit_code == 3


def test_expired_cache_forces_a_new_call(tmp_path: Path) -> None:
    client_factory = Mock(side_effect=_client_factory)
    run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        cache_ttl_seconds=10,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[1.0, 1.1]),
        client_factory=client_factory,
    )
    run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        cache_ttl_seconds=10,
        now_fn=lambda: NOW + timedelta(seconds=11),
        monotonic_fn=Mock(side_effect=[2.0, 2.1]),
        client_factory=client_factory,
    )

    assert client_factory.call_count == 2


def test_corrupt_cache_is_not_authority(tmp_path: Path) -> None:
    client_factory = Mock(side_effect=_client_factory)
    first = run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[1.0, 1.1]),
        client_factory=client_factory,
    )
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text(
        json.dumps({**first.to_dict(), "verdict": "available", "extra": 1})
    )

    run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        now_fn=lambda: NOW + timedelta(seconds=1),
        monotonic_fn=Mock(side_effect=[2.0, 2.1]),
        client_factory=client_factory,
    )
    assert client_factory.call_count == 2


def test_cache_policy_is_part_of_identity(tmp_path: Path) -> None:
    client_factory = Mock(side_effect=_client_factory)
    run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        cache_ttl_seconds=900,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[1.0, 1.1]),
        client_factory=client_factory,
    )
    run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_dir=tmp_path,
        cache_ttl_seconds=10,
        now_fn=lambda: NOW + timedelta(seconds=1),
        monotonic_fn=Mock(side_effect=[2.0, 2.1]),
        client_factory=client_factory,
    )
    assert client_factory.call_count == 2


def test_probe_refuses_api_transport_before_client_construction(tmp_path: Path) -> None:
    client_factory = Mock()
    with patch("llmx.probe.build_dispatch_plan") as build_plan:
        build_plan.return_value = Mock(
            provider="google",
            model="gemini-3.1-pro-preview",
            auth="subscription",
            subscription=True,
            transport="google-api",
            cli_fallback_reason=None,
        )
        try:
            run_subscription_probe(
                provider="google",
                model="gemini-3.1-pro-preview",
                cache_dir=tmp_path,
                client_factory=client_factory,
            )
        except ValueError as error:
            assert "refuses metered transport" in str(error)
        else:
            raise AssertionError("metered probe route was accepted")
    client_factory.assert_not_called()


def test_cli_emits_cached_typed_quota_exit() -> None:
    quota_result = run_subscription_probe(
        provider="anthropic",
        model="claude-opus-5-5",
        cache_ttl_seconds=0,
        now_fn=lambda: NOW,
        monotonic_fn=Mock(side_effect=[1.0, 1.1]),
        client_factory=Mock(
            return_value=Mock(
                chat=Mock(
                    side_effect=QuotaError(
                        "You've hit your monthly spend limit",
                        provider="claude-cli",
                        model="claude-opus-5-5",
                        status_code=429,
                    )
                )
            )
        ),
    )
    with patch("llmx.probe_cmd.run_subscription_probe", return_value=quota_result):
        result = CliRunner().invoke(cli, ["probe", "--json"])

    assert result.exit_code == 6
    payload = json.loads(result.output)
    assert payload["verdict"] == "unavailable"
    assert payload["error_type"] == "quota_exhausted"
