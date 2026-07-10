"""Tests for dispatch_plan (offline)."""

import unittest

from llmx.dispatch_plan import (
    combine_file_context,
    default_timeout_for,
    map_effort_for_backend,
    normalize_effort_input,
    resolve_effort,
)


class TestEffortNormalize(unittest.TestCase):
    def test_max_allowed(self):
        effort, warns = normalize_effort_input("max")
        self.assertEqual(effort, "max")
        self.assertTrue(warns)

    def test_invalid_rejected(self):
        with self.assertRaises(ValueError):
            normalize_effort_input("turbo")

    def test_claude_max_maps(self):
        applied, _ = map_effort_for_backend("max", transport="claude-cli", provider="anthropic")
        self.assertEqual(applied, "max")

    def test_api_max_maps_xhigh_pre_56(self):
        applied, warns = map_effort_for_backend(
            "max", transport="openai-api", provider="openai", model="gpt-5.4"
        )
        self.assertEqual(applied, "xhigh")
        self.assertTrue(warns)

    def test_api_max_passthrough_gpt56(self):
        applied, warns = map_effort_for_backend(
            "max", transport="openai-api", provider="openai", model="gpt-5.6-sol"
        )
        self.assertEqual(applied, "max")
        self.assertFalse(warns)

    def test_codex_max_passthrough_gpt56(self):
        applied, _ = map_effort_for_backend(
            "max", transport="codex-cli", provider="openai", model="gpt-5.6-terra"
        )
        self.assertEqual(applied, "max")

    def test_resolve_effort_api_max(self):
        applied, _ = resolve_effort(
            "max", transport="openai-api", provider="openai", model="gpt-5.4"
        )
        self.assertEqual(applied, "xhigh")


class TestDefaultTimeout(unittest.TestCase):
    def test_chat_defaults_preserve_short_low_effort_calls(self):
        self.assertEqual(default_timeout_for(mode="chat", effort="low"), 300)

    def test_effort_floors_scale_monotonically(self):
        self.assertEqual(default_timeout_for(mode="chat", effort="high"), 600)
        self.assertEqual(default_timeout_for(mode="chat", effort="xhigh"), 1200)
        self.assertEqual(default_timeout_for(mode="chat", effort="max"), 3600)

    def test_agent_mode_has_tool_use_floor(self):
        self.assertEqual(default_timeout_for(mode="agent", effort="low"), 1800)
        self.assertEqual(default_timeout_for(mode="agent", effort="max"), 3600)


class TestLlmLiteRouting(unittest.TestCase):
    def test_lite_enables_claude_cli(self):
        from llmx.api import LLM

        llm = LLM(provider="anthropic", model="claude-opus-4-8", lite="bare")
        self.assertEqual(llm._cli_provider, "claude-cli")

    def test_anthropic_defaults_subscription_cli(self):
        from llmx.api import LLM

        llm = LLM(provider="anthropic", model="claude-opus-4-8")
        self.assertEqual(llm._cli_provider, "claude-cli")
        self.assertEqual(llm.kwargs.get("lite"), "bare")

    def test_anthropic_dispatch_plan_defaults_subscription(self):
        from llmx.dispatch_plan import build_dispatch_plan

        plan = build_dispatch_plan(
            provider="anthropic",
            model="claude-opus-4-8",
            reasoning_effort=None,
            timeout=300,
            lite=None,
            mode=None,
            auth=None,
            subscription=False,
            api_only=None,
            use_old=False,
        )
        self.assertEqual(plan.auth, "subscription")
        self.assertEqual(plan.mode, "chat")
        self.assertTrue(plan.subscription)
        self.assertEqual(plan.lite, "bare")
        self.assertEqual(plan.transport, "claude-cli")

    def test_anthropic_agent_uses_subscription_cli_without_lite_profile(self):
        from llmx.dispatch_plan import build_dispatch_plan

        plan = build_dispatch_plan(
            provider="anthropic",
            model="claude-opus-4-8",
            reasoning_effort="max",
            timeout=3600,
            lite=None,
            mode="agent",
            auth="subscription",
            subscription=False,
            api_only=None,
            use_old=False,
        )
        self.assertEqual(plan.auth, "subscription")
        self.assertEqual(plan.mode, "agent")
        self.assertIsNone(plan.lite)
        self.assertEqual(plan.transport, "claude-cli")


class TestFileContext(unittest.TestCase):
    def test_boundaries(self):
        out = combine_file_context(("a.md", "b.md"), ["one", "two"])
        self.assertIn("=== File: a.md ===", out)
        self.assertIn("=== File: b.md ===", out)
        self.assertIn("one", out)
        self.assertIn("two", out)


if __name__ == "__main__":
    unittest.main()
