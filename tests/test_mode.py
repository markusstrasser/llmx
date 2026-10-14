"""Tests for mode surface."""

import unittest
from unittest.mock import patch

from click.testing import CliRunner

from llmx.api import LLM
from llmx.cli import cli
from llmx.dispatch_plan import build_dispatch_plan
from llmx.mode import AGENT_IMPLIES_SUBSCRIPTION, resolve_auth_mode, resolve_mode

CODEX = "/usr/local/bin/codex"


class TestResolveMode(unittest.TestCase):
    def test_default_chat(self):
        mode, source, lite, _ = resolve_mode(auth="subscription")
        self.assertEqual(mode, "chat")
        self.assertEqual(source, "default_policy")
        self.assertEqual(lite, "bare")

    def test_explicit_agent(self):
        mode, source, lite, _ = resolve_mode(mode="agent", auth="subscription")
        self.assertEqual(mode, "agent")
        self.assertEqual(source, "explicit")
        self.assertIsNone(lite)

    def test_legacy_research_profile_remains_isolated(self):
        mode, source, lite, warns = resolve_mode(lite="research", auth="subscription")
        self.assertEqual(mode, "agent")
        self.assertEqual(source, "lite_deprecated")
        self.assertEqual(lite, "research")
        self.assertTrue(warns)

    def test_lite_bare_deprecated(self):
        mode, source, lite, warns = resolve_mode(lite="bare", auth="subscription")
        self.assertEqual(mode, "chat")
        self.assertEqual(source, "lite_deprecated")
        self.assertTrue(warns)

    def test_api_defaults_to_chat(self):
        mode, source, lite, warns = resolve_mode(auth="api")
        self.assertEqual(mode, "chat")
        self.assertEqual(source, "default_policy")
        self.assertIsNone(lite)
        self.assertEqual(warns, [])

    def test_api_refuses_agent_instead_of_downgrading(self):
        with self.assertRaisesRegex(ValueError, "agent needs auth=subscription"):
            resolve_mode(mode="agent", auth="api")

    def test_lite_mode_conflict(self):
        with self.assertRaises(ValueError):
            resolve_mode(mode="chat", lite="research", auth="subscription")


class TestResolveAuthMode(unittest.TestCase):
    """2026-10-14: an explicit --mode agent was silently downgraded to an API chat."""

    def test_default_auth_explicit_agent_implies_subscription(self):
        resolved = resolve_auth_mode(mode="agent", provider="openai")
        self.assertEqual((resolved.auth, resolved.auth_source), ("subscription", "mode_implied"))
        self.assertEqual((resolved.mode, resolved.mode_source), ("agent", "explicit"))
        self.assertIsNone(resolved.lite)

    def test_explicit_api_sources_refuse_agent(self):
        for kwargs in (
            {"auth": "api", "provider": "openai"},
            {"provider": "anthropic-direct"},
            {"api_only": True, "provider": "google"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "--mode agent needs subscription auth"):
                    resolve_auth_mode(mode="agent", **kwargs)

    def test_default_auth_default_mode_stays_api_chat(self):
        for provider in ("openai", "google"):
            with self.subTest(provider=provider):
                resolved = resolve_auth_mode(provider=provider)
                self.assertEqual((resolved.auth, resolved.auth_source), ("api", "default_policy"))
                self.assertEqual((resolved.mode, resolved.mode_source), ("chat", "default_policy"))
                self.assertIsNone(resolved.lite)

    def test_explicit_chat_and_subscription_flag_unchanged(self):
        chat = resolve_auth_mode(mode="chat", provider="openai")
        self.assertEqual((chat.auth, chat.mode), ("api", "chat"))
        flagged = resolve_auth_mode(mode="agent", subscription_flag=True, provider="openai")
        self.assertEqual(
            (flagged.auth, flagged.auth_source, flagged.mode),
            ("subscription", "explicit_flag", "agent"),
        )

    def test_legacy_lite_profiles_unchanged(self):
        research = resolve_auth_mode(lite="research", provider="openai")
        self.assertEqual((research.auth, research.mode, research.lite), ("api", "agent", "research"))
        bare = resolve_auth_mode(lite="bare", provider="openai")
        self.assertEqual((bare.auth, bare.mode, bare.lite), ("api", "chat", "bare"))


def _plan(**overrides):
    kwargs = dict(
        provider=None, model="gpt-6.1-sol", reasoning_effort="high", timeout=300,
        lite=None, mode="agent", auth=None, subscription=False, api_only=None, use_old=False,
    )
    kwargs.update(overrides)
    with patch("llmx.cli_backends.shutil.which", return_value=CODEX):
        return build_dispatch_plan(**kwargs)


class TestAgentAuthEntryPoints(unittest.TestCase):
    def _cli(self, *args):
        with patch("llmx.cli_backends.shutil.which", return_value=CODEX):
            return CliRunner().invoke(cli, ["chat", *args, "--dry-run", "review the repo"])

    def test_plan_default_auth_agent_runs_subscription_cli(self):
        plan = _plan()
        self.assertEqual((plan.auth, plan.auth_source), ("subscription", "mode_implied"))
        self.assertEqual((plan.mode, plan.transport), ("agent", "codex-cli"))

    def test_plan_explicit_api_agent_raises(self):
        with self.assertRaisesRegex(ValueError, "--auth api selects the metered API"):
            _plan(auth="api")

    def test_plan_default_openai_stays_api_chat(self):
        plan = _plan(mode=None)
        self.assertEqual((plan.auth, plan.auth_source, plan.mode), ("api", "default_policy", "chat"))

    def test_cli_agent_without_auth_implies_subscription(self):
        result = self._cli("--mode", "agent", "-m", "gpt-6.1-sol", "-e", "high")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stderr.count(f"[llmx] {AGENT_IMPLIES_SUBSCRIPTION}"), 1)
        self.assertIn("auth=subscription mode=agent transport=codex-cli", result.stderr)

    def test_cli_agent_with_explicit_api_is_usage_error(self):
        for args in (
            ("--auth", "api", "-m", "gpt-6.1-sol"),
            ("-p", "anthropic-direct", "-m", "claude-opus-5-5"),
        ):
            with self.subTest(args=args):
                result = self._cli("--mode", "agent", *args)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("--mode agent needs subscription auth", result.stderr)

    def test_cli_default_openai_stays_api_chat(self):
        result = self._cli("-m", "gpt-6.1-sol")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("auth=api mode=chat", result.stderr)
        self.assertNotIn(AGENT_IMPLIES_SUBSCRIPTION, result.stderr)

    def test_python_llm_default_auth_agent_implies_subscription(self):
        with patch("llmx.cli_backends.shutil.which", return_value=CODEX):
            llm = LLM(provider="openai", model="gpt-6.1-sol", mode="agent")
        self.assertEqual(
            (llm.kwargs["auth"], llm.kwargs["mode"], llm._cli_provider),
            ("subscription", "agent", "codex-cli"),
        )

    def test_python_llm_explicit_api_agent_raises(self):
        with self.assertRaisesRegex(ValueError, "agent needs auth=subscription"):
            LLM(provider="openai", model="gpt-6.1-sol", mode="agent", auth="api")

    def test_python_llm_agent_without_cli_backend_raises(self):
        with patch("llmx.cli_backends.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "no CLI backend installed"):
                LLM(provider="openai", model="gpt-6.1-sol", mode="agent")


if __name__ == "__main__":
    unittest.main()
