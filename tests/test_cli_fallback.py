"""Tests for subscription-safe CLI→API fallback."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llmx.api import LLM, Response
from llmx.cli import cli
from llmx.cli_backends import (
    CliBackendFailure,
    _parse_claude_json,
    cli_chat,
    resolve_cli_api_fallback,
    subscription_route,
)
from llmx.providers import (
    ApiKeyError,
    LlmxError,
    ModelError,
    QuotaError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError_,
)


MONTHLY_SPEND_DETAIL = "You've hit your monthly spend limit"
CAPTURED_MONTHLY_SPEND_JSON = json.dumps(
    {
        "type": "result",
        "is_error": True,
        "api_error_status": 429,
        "result": MONTHLY_SPEND_DETAIL,
    }
)
MONTHLY_SPEND_FAILURE = CliBackendFailure(
    kind=QuotaError,
    status=429,
    detail=MONTHLY_SPEND_DETAIL,
)
TRUNCATED_MULTIBLOCK_JSON = json.dumps(
    [
        {
            "type": "assistant",
            "message": {
                "id": "msg-final",
                "content": [
                    {"type": "text", "text": "PREFIX"},
                    {"type": "text", "text": "TAIL"},
                ],
            },
        },
        {"type": "result", "is_error": False, "result": "TAIL"},
    ]
)


class TestSubscriptionRoute(unittest.TestCase):
    def test_auth_subscription(self):
        self.assertTrue(subscription_route(auth="subscription"))

    def test_lite_bare(self):
        self.assertTrue(subscription_route(lite="bare"))

    def test_api_route(self):
        self.assertFalse(subscription_route(auth="api"))


class TestResolveCliApiFallback(unittest.TestCase):
    def test_subscription_blocks(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_cli_api_fallback(
                "claude-cli",
                auth="subscription",
                reason="CLI error",
            )
        self.assertIn("forbids", str(ctx.exception))

    def test_api_allows_anthropic(self):
        self.assertEqual(
            resolve_cli_api_fallback("claude-cli", auth="api", reason="CLI error"),
            "anthropic",
        )

    def test_cursor_no_fallback_even_on_api(self):
        with self.assertRaises(ValueError):
            resolve_cli_api_fallback("cursor-cli", auth="api", reason="schema")


class TestClaudeCliFailureParsing(unittest.TestCase):
    @patch("llmx.cli_backends.subprocess.Popen")
    def test_captured_monthly_spend_json_returns_typed_quota(self, popen):
        process = popen.return_value
        process.pid = 123
        process.returncode = 1
        process.communicate.return_value = (
            CAPTURED_MONTHLY_SPEND_JSON,
            "Claude request failed",
        )

        result = cli_chat(
            "claude-cli",
            "hi",
            "claude-opus-4-8",
            30,
            mode="agent",
        )

        self.assertEqual(result, MONTHLY_SPEND_FAILURE)

    def test_transient_429_is_not_quota(self):
        detail = "Rate limit quota exceeded. Please retry shortly."
        result, usage = _parse_claude_json(
            json.dumps(
                {
                    "type": "result",
                    "is_error": True,
                    "api_error_status": 429,
                    "result": detail,
                }
            )
        )

        self.assertEqual(
            result,
            CliBackendFailure(
                kind=RateLimitError,
                status=429,
                detail=detail,
            ),
        )
        self.assertIsNone(usage)

    def test_other_claude_error_kinds(self):
        cases = (
            (504, "Request timed out", TimeoutError_),
            (401, "Authentication token expired", ApiKeyError),
            (404, "Model not found", ModelError),
            (
                503,
                "Service temporarily unavailable",
                ServiceUnavailableError,
            ),
            (400, "Malformed request", LlmxError),
        )
        for status, detail, expected_kind in cases:
            with self.subTest(status=status, detail=detail):
                result, usage = _parse_claude_json(
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": status,
                            "result": detail,
                        }
                    )
                )
                self.assertIsInstance(result, CliBackendFailure)
                self.assertEqual(result.kind, expected_kind)
                self.assertEqual(result.status, status)
                self.assertEqual(result.detail, detail)
                self.assertIsNone(usage)


class TestLlmSubscriptionFallback(unittest.TestCase):
    def test_subscription_quota_maps_without_api_fallback(self):
        with (
            patch("llmx.api.preferred_cli_provider", return_value="claude-cli"),
            patch("llmx.api.needs_api_fallback", return_value=None),
            patch("llmx.api.cli_chat", return_value=MONTHLY_SPEND_FAILURE),
            patch("llmx.api.resolve_cli_api_fallback") as fallback,
        ):
            llm = LLM(provider="anthropic", auth="subscription", mode="chat")
            with self.assertRaises(QuotaError) as raised:
                llm.chat("hi")

        error = raised.exception
        self.assertEqual(error.exit_code, 6)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(str(error), MONTHLY_SPEND_DETAIL)
        fallback.assert_not_called()

    def test_api_auth_retains_intentional_fallback(self):
        expected = Response(
            content="api response",
            provider="anthropic",
            model="claude-opus-4-8",
            usage={},
            latency=0.1,
            raw=None,
        )
        with (
            patch("llmx.api.preferred_cli_provider", return_value="claude-cli"),
            patch("llmx.api.needs_api_fallback", return_value=None),
            patch("llmx.api.cli_chat", return_value=MONTHLY_SPEND_FAILURE),
        ):
            llm = LLM(
                provider="claude-cli",
                model="claude-opus-4-8",
                auth="subscription",
            )
            with patch("llmx.api.LLM") as fallback_class:
                fallback_client = fallback_class.return_value
                fallback_client.chat.return_value = expected

                result = llm.chat("hi", auth="api", lite=None)

        self.assertIs(result, expected)
        fallback_class.assert_called_once()
        fallback_kwargs = fallback_class.call_args.kwargs
        self.assertEqual(fallback_kwargs["provider"], "anthropic")
        self.assertEqual(fallback_kwargs["auth"], "api")
        self.assertTrue(fallback_kwargs["api_only"])
        fallback_client.chat.assert_called_once_with(
            "hi", system=None, temperature=None, auth="api", lite=None
        )


class TestCliExitCode(unittest.TestCase):
    def test_retired_grok45_subscription_names_supported_routes(self):
        result = CliRunner().invoke(
            cli,
            ["chat", "--dry-run", "--subscription", "--model", "grok-4.5", "hi"],
        )

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("cursor-grok-4.6-high", result.output)
        self.assertIn("-p grok -m grok-4.6", result.output)

    def test_quota_error_exits_6(self):
        quota_error = QuotaError(
            MONTHLY_SPEND_DETAIL,
            provider="claude-cli",
            model="claude-opus-4-8",
            status_code=429,
        )
        with (
            patch("llmx.cli_backends.shutil.which", return_value="/usr/bin/claude"),
            patch("llmx.cli.chat", side_effect=quota_error),
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "chat",
                    "--subscription",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude-opus-4-8",
                    "hi",
                ],
            )

        self.assertEqual(result.exit_code, 6, result.output)
        self.assertIn("type=quota_exhausted", result.output)
        self.assertIn("status=429", result.output)
        self.assertIn(MONTHLY_SPEND_DETAIL, result.output)

    @patch("llmx.cli_backends.subprocess.Popen")
    def test_multiblock_integrity_failure_leaves_output_empty(self, popen):
        process = popen.return_value
        process.pid = 123
        process.returncode = 0
        process.communicate.return_value = (TRUNCATED_MULTIBLOCK_JSON, "")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "llmx.cli_backends.shutil.which",
                return_value="/usr/bin/claude",
            ),
        ):
            output_path = Path(tmp) / "response.md"
            result = CliRunner().invoke(
                cli,
                [
                    "chat",
                    "--subscription",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude-opus-4-8",
                    "--output",
                    str(output_path),
                    "hi",
                ],
            )

            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("omitted assistant text blocks", result.output)
            self.assertNotIn("PREFIX", result.output)
            self.assertNotIn("TAIL", result.output)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(), "")


class TestProviderSubscriptionFallback(unittest.TestCase):
    def test_retired_grok45_subscription_never_reaches_a_transport(self):
        from llmx.providers import chat

        with (
            patch("llmx.cli_backends.cli_chat") as cli_chat_mock,
            patch("llmx.providers.OpenAI") as api_client,
            self.assertRaisesRegex(
                ValueError,
                r"cursor-grok-4\.6-high.*-p grok -m grok-4\.6",
            ),
        ):
            chat(
                "hi",
                provider="xai",
                model="grok-4.5",
                temperature=0.7,
                reasoning_effort=None,
                stream=False,
                debug=False,
                json_output=False,
                auth="subscription",
            )

        cli_chat_mock.assert_not_called()
        api_client.assert_not_called()

    @patch(
        "llmx.cli_backends.needs_api_fallback",
        return_value="structured output not supported by CLI",
    )
    @patch("llmx.cli_backends.preferred_cli_provider", return_value="claude-cli")
    def test_subscription_forced_fallback_raises_with_reason(self, *_mocks):
        from llmx.providers import chat

        with self.assertRaises(RuntimeError) as ctx:
            chat(
                "hi",
                provider="anthropic",
                model="claude-fable-5",
                temperature=0.7,
                reasoning_effort="medium",
                stream=False,
                debug=False,
                json_output=False,
                schema={"type": "object"},
                lite="bare",
                auth="subscription",
            )
        message = str(ctx.exception)
        self.assertIn("structured output not supported by CLI", message)
        self.assertIn("auth=subscription forbids metered API fallback", message)

    def test_subscription_quota_maps_without_api_fallback(self):
        from llmx.providers import chat

        with (
            patch(
                "llmx.cli_backends.preferred_cli_provider",
                return_value="claude-cli",
            ),
            patch("llmx.cli_backends.needs_api_fallback", return_value=None),
            patch(
                "llmx.cli_backends.cli_chat",
                return_value=MONTHLY_SPEND_FAILURE,
            ),
            patch("llmx.cli_backends.resolve_cli_api_fallback") as fallback,
        ):
            with self.assertRaises(QuotaError) as raised:
                chat(
                    "hi",
                    provider="anthropic",
                    model="claude-opus-4-8",
                    temperature=0.7,
                    reasoning_effort="medium",
                    stream=False,
                    debug=False,
                    json_output=False,
                    auth="subscription",
                )

        error = raised.exception
        self.assertEqual(error.exit_code, 6)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(str(error), MONTHLY_SPEND_DETAIL)
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
