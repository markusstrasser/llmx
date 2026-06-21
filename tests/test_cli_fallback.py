"""Tests for subscription-safe CLI→API fallback."""

import unittest
from unittest.mock import patch

from llmx.api import LLM
from llmx.cli_backends import resolve_cli_api_fallback, subscription_route


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


class TestLlmSubscriptionFallback(unittest.TestCase):
    @patch("llmx.api.cli_chat", return_value=None)
    @patch("llmx.api.needs_api_fallback", return_value=None)
    @patch("llmx.api.preferred_cli_provider", return_value="claude-cli")
    def test_subscription_cli_failure_raises(self, *_mocks):
        llm = LLM(provider="anthropic", auth="subscription", mode="chat")
        with self.assertRaises(RuntimeError) as ctx:
            llm.chat("hi")
        self.assertIn("forbids", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
