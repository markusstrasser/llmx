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

    def test_claude_xhigh_is_native(self):
        # 2026-09-30: stale mapping silently raised requested xhigh to max.
        applied, warns = resolve_effort(
            "xhigh", transport="claude-cli", provider="anthropic"
        )
        self.assertEqual(applied, "xhigh")
        self.assertEqual(warns, [])

    def test_api_max_maps_xhigh_pre_56(self):
        applied, warns = map_effort_for_backend(
            "max", transport="openai-api", provider="openai", model="gpt-5.4"
        )
        self.assertEqual(applied, "xhigh")
        self.assertTrue(warns)

    def test_api_max_passthrough_gpt6(self):
        applied, warns = map_effort_for_backend(
            "max", transport="openai-api", provider="openai", model="gpt-6-sol"
        )
        self.assertEqual(applied, "max")
        self.assertFalse(warns)

    def test_api_max_passthrough_astra(self):
        applied, warns = map_effort_for_backend(
            "max", transport="openai-api", provider="openai", model="gpt-6-astra"
        )
        self.assertEqual(applied, "max")
        self.assertFalse(warns)

    def test_none_maps_low_for_astra(self):
        applied, warns = map_effort_for_backend(
            "none", transport="openai-api", provider="openai", model="gpt-6-astra"
        )
        self.assertEqual(applied, "low")
        self.assertTrue(warns)

    def test_none_passes_through_for_gpt6_sol_luna(self):
        for model in ("gpt-6-sol", "gpt-6-luna"):
            with self.subTest(model=model):
                applied, warns = map_effort_for_backend(
                    "none", transport="openai-api", provider="openai", model=model
                )
                self.assertEqual(applied, "none")
                self.assertFalse(warns)

    def test_gpt6_sol_luna_registered_priced_and_subscription_allowed(self):
        from llmx.cli_backends import lite_model_allowed
        from llmx.providers import _KNOWN_MODELS
        from llmx.usage_report import PRICING, est_cost

        self.assertEqual(PRICING["gpt-6-sol"], (2.0, 10.0))
        self.assertEqual(PRICING["gpt-6-luna"], (0.10, 0.50))
        for model in ("gpt-6-sol", "gpt-6-luna"):
            self.assertIn(model, _KNOWN_MODELS["openai"])
            self.assertTrue(lite_model_allowed(model))
        # >272K input: 2x input, 1.5x output for the full request (model page).
        cost = est_cost("gpt-6-sol", 300_000, 100_000)
        assert cost is not None
        self.assertAlmostEqual(cost, (300_000 * 4 + 100_000 * 15) / 1e6)

    def test_codex_max_passthrough_gpt6(self):
        applied, _ = map_effort_for_backend(
            "max", transport="codex-cli", provider="openai", model="gpt-6-sol"
        )
        self.assertEqual(applied, "max")

    def test_resolve_effort_api_max(self):
        applied, _ = resolve_effort(
            "max", transport="openai-api", provider="openai", model="gpt-5.4"
        )
        self.assertEqual(applied, "xhigh")


class TestAstraDispatch(unittest.TestCase):
    def test_mirror_exposes_known_models_from_canonical_registry(self):
        from llmx.dispatch_plan import collect_routing_mirror
        from llmx.providers import _KNOWN_MODELS

        known = collect_routing_mirror()["known_models"]
        self.assertEqual(known, _KNOWN_MODELS)
        self.assertIn("gpt-6-astra", known["openai"])
        self.assertIn("gpt-6", known["openai"])
        self.assertIn("gemini-3.8-flash", known["google"])
        self.assertIn("claude-opus-5-5", known["anthropic-direct"])
        self.assertIn("claude-fable-5-1", known["anthropic-direct"])

    def test_subscription_mirror_checks_the_resolved_cli(self):
        from unittest.mock import patch

        from llmx.dispatch_plan import collect_routing_mirror

        for installed in (False, True):
            with self.subTest(installed=installed), patch(
                "shutil.which", side_effect=lambda binary: f"/bin/{binary}" if installed else None
            ):
                routes = collect_routing_mirror()["logical_subscription_routes"]
            for provider in ("openai", "anthropic", "cursor"):
                self.assertEqual(routes[provider]["lite_bare_available"], installed)
            self.assertFalse(routes["google"]["lite_bare_available"])

    def test_openai_default_is_astra(self):
        from llmx.providers import PROVIDER_CONFIGS, get_model_name
        from llmx.usage_report import PRICING

        self.assertEqual(PROVIDER_CONFIGS["openai"]["model"], "gpt-6-astra")
        self.assertEqual(get_model_name("openai"), "gpt-6-astra")
        self.assertEqual(PRICING["gpt-6-astra"], (10.0, 50.0))

    def test_google_default_is_gemini_38_flash(self):
        from llmx.providers import PROVIDER_CONFIGS, get_model_name
        from llmx.usage_report import PRICING

        self.assertEqual(PROVIDER_CONFIGS["google"]["model"], "gemini-3.8-flash")
        self.assertEqual(PROVIDER_CONFIGS["google"]["flash_model"], "gemini-3.8-flash")
        self.assertEqual(get_model_name("google"), "gemini-3.8-flash")
        self.assertEqual(PRICING["gemini-3.8-flash"], (0.75, 3.75))

    def test_explicit_subscription_and_api_report_applied_effort(self):
        from unittest.mock import patch

        from llmx.cli_backends import lite_model_allowed
        from llmx.dispatch_plan import build_dispatch_plan

        self.assertTrue(lite_model_allowed("gpt-6-astra"))
        for auth, transport in (("subscription", "codex-cli"), ("api", "openai-api")):
            for requested in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                expected = "low" if requested in {"none", "minimal"} else requested
                with self.subTest(auth=auth, requested=requested):
                    with patch("llmx.dispatch_plan.binary_available", return_value=True):
                        plan = build_dispatch_plan(
                            provider="openai",
                            model="gpt-6-astra",
                            reasoning_effort=requested,
                            timeout=300,
                            lite=None,
                            mode=None,
                            auth=auth,
                            subscription=False,
                            api_only=None,
                            use_old=False,
                        )
                    self.assertEqual(plan.transport, transport)
                    self.assertEqual(plan.model, "gpt-6-astra")
                    self.assertEqual(plan.requested_effort, requested)
                    self.assertEqual(plan.effort_applied, expected)
                    if requested != expected:
                        self.assertTrue(any("mapped to low" in w for w in plan.effort_warnings))


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
    def test_exact_cursor_grok_slug_is_subscription_cursor(self):
        from unittest.mock import patch

        from llmx.dispatch_plan import build_dispatch_plan

        with (
            patch("llmx.cli_backends.binary_available", return_value=True),
            patch("llmx.cli_backends.shutil.which", return_value="/usr/bin/cursor-agent"),
        ):
            plan = build_dispatch_plan(
                provider=None,
                model="grok-4.7-high",
                reasoning_effort=None,
                timeout=300,
                lite=None,
                mode=None,
                auth=None,
                subscription=False,
                api_only=None,
                use_old=False,
            )

        self.assertEqual(plan.provider, "cursor")
        self.assertEqual(plan.auth, "subscription")
        self.assertEqual(plan.transport, "cursor-cli")

    def test_bare_grok45_subscription_fails_with_migration_paths(self):
        from unittest.mock import patch

        from llmx.dispatch_plan import build_dispatch_plan

        with (
            patch("llmx.cli_backends.binary_available", return_value=True),
            patch("llmx.cli_backends.shutil.which", return_value="/usr/bin/cursor-agent"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"grok-4\.5.*grok-4\.7.*Cursor grok-4\.7-high.*Grok Build",
            ):
                build_dispatch_plan(
                    provider=None,
                    model="grok-4.5",
                    reasoning_effort="high",
                    timeout=300,
                    lite=None,
                    mode=None,
                    auth=None,
                    subscription=True,
                    api_only=None,
                    use_old=False,
                )

    def test_subscription_still_cannot_claim_metered_api_for_cli_less_provider(self):
        # Keep the general invariant independent of the targeted Grok migration
        # error above.
        from llmx.dispatch_plan import build_dispatch_plan

        with self.assertRaisesRegex(ValueError, "subscription.*deepseek-api"):
            build_dispatch_plan(
                provider=None,
                model="deepseek-chat",
                reasoning_effort=None,
                timeout=300,
                lite=None,
                mode=None,
                auth=None,
                subscription=True,
                api_only=None,
                use_old=False,
            )

    def test_bare_grok45_api_route_is_retired(self):
        from llmx.dispatch_plan import build_dispatch_plan

        with self.assertRaisesRegex(ValueError, r"grok-4\.5 was retired.*grok-4\.7"):
            build_dispatch_plan(
                provider=None,
                model="grok-4.5",
                reasoning_effort="high",
                timeout=300,
                lite=None,
                mode=None,
                auth="api",
                subscription=False,
                api_only=None,
                use_old=False,
            )

    def test_lite_enables_claude_cli(self):
        from llmx.api import LLM

        llm = LLM(provider="anthropic", model="claude-opus-5-5", lite="bare")
        self.assertEqual(llm._cli_provider, "claude-cli")

    def test_anthropic_defaults_subscription_cli(self):
        from llmx.api import LLM

        llm = LLM(provider="anthropic", model="claude-opus-5-5")
        self.assertEqual(llm._cli_provider, "claude-cli")
        self.assertEqual(llm.kwargs.get("lite"), "bare")

    def test_anthropic_dispatch_plan_defaults_subscription(self):
        from llmx.dispatch_plan import build_dispatch_plan

        plan = build_dispatch_plan(
            provider="anthropic",
            model="claude-opus-5-5",
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
            model="claude-opus-5-5",
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
