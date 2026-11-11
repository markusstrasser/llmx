"""A headless CLI launch names its model, effort and tier on the command line.

Left out, each is read from a file an interactive session rewrites: Claude Code
saves `/effort` as the default for new sessions, and the Codex app saves its
model, effort and speed pickers to config.toml. On 2026-11-04 an Ultrafast tier
saved there billed every headless run at 6x.
"""
import subprocess
import unittest
from unittest.mock import patch

from llmx import image
from llmx.cli_backends import CODEX_HEADLESS_TIER_ARGS, cli_chat, headless_default_effort
from llmx.dispatch_plan import build_dispatch_plan
from tests.test_usage_accounting import _claude_verbose_stdout


def _plan(model, *, mode="chat", effort=None):
    return build_dispatch_plan(
        provider=None, model=model, reasoning_effort=effort, timeout=300, lite=None,
        mode=mode, auth="subscription", subscription=True, api_only=None, use_old=False,
    )


def _pairs(command):
    return list(zip(command, command[1:]))


class TestPlanNamesTheEffort(unittest.TestCase):
    def test_default_per_model_and_mode(self):
        for model, mode, expected in (
            ("claude-opus-5-5", "chat", "high"),
            ("claude-fable-5-2", "chat", "high"),
            ("claude-sonnet-5-5", "chat", "medium"),
            ("claude-haiku-5-5", "chat", "medium"),
            ("gpt-6-astra", "chat", "medium"),
            ("gpt-6.1-sol", "chat", "medium"),
            ("gpt-6-astra", "agent", "high"),
        ):
            with self.subTest(model=model, mode=mode):
                plan = _plan(model, mode=mode)
                self.assertEqual(plan.effort_applied, expected)
                self.assertEqual(plan.effort_source, "llmx-default")
                self.assertIsNone(plan.requested_effort)
                self.assertIn(f"effort={expected}", plan.stderr_line())

    def test_caller_effort_wins(self):
        plan = _plan("claude-fable-5-2", effort="max")
        self.assertEqual((plan.effort_applied, plan.effort_source), ("max", "caller"))
        plan = _plan("gpt-6-astra", effort="low")
        self.assertEqual((plan.effort_applied, plan.effort_source), ("low", "caller"))

    def test_api_transport_has_no_llmx_default(self):
        plan = build_dispatch_plan(
            provider="anthropic-direct", model="claude-opus-5-5", reasoning_effort=None,
            timeout=300, lite=None, mode="chat", auth="api", subscription=False,
            api_only=None, use_old=False,
        )
        self.assertIsNone(plan.effort_applied)
        self.assertIsNone(plan.effort_source)
        self.assertIsNone(headless_default_effort("cursor-cli", "grok-4.7-high", "chat"))


class TestCommandNamesTheEffort(unittest.TestCase):
    def _codex(self, *, mode, lite, effort=None):
        with (
            patch("llmx.cli_backends.subprocess.Popen") as popen,
            patch("llmx.cli_backends._codex_rollout_snapshot", return_value={}),
            patch("llmx.cli_backends._latest_codex_rollout_usage", return_value=({}, None)),
            patch("llmx.usage_log.log_usage") as usage,
        ):
            process = popen.return_value
            process.pid = 123
            process.returncode = 0
            process.communicate.return_value = ("OK", "")
            cli_chat("codex-cli", "hi", "gpt-6-astra", 30, mode=mode, lite=lite,
                     reasoning_effort=effort)
        return popen.call_args.args[0], usage.call_args.kwargs

    def test_codex_without_caller_effort(self):
        for mode, lite, expected in (("chat", "bare", "medium"), ("agent", None, "high")):
            with self.subTest(mode=mode):
                command, usage = self._codex(mode=mode, lite=lite)
                pairs = _pairs(command)
                self.assertIn(("-c", f'model_reasoning_effort="{expected}"'), pairs)
                self.assertIn(CODEX_HEADLESS_TIER_ARGS, pairs)
                self.assertIn(("-m", "gpt-6-astra"), pairs)
                self.assertEqual(usage["reasoning_effort"], expected)

    def test_claude_without_caller_effort(self):
        for model, expected in (("claude-fable-5-2", "high"), ("claude-haiku-5-5", "medium")):
            with (
                self.subTest(model=model),
                patch("llmx.cli_backends.subprocess.Popen") as popen,
                patch("llmx.usage_log.log_usage"),
            ):
                process = popen.return_value
                process.pid = 123
                process.returncode = 0
                process.communicate.return_value = (_claude_verbose_stdout("ok"), "")
                cli_chat("claude-cli", "hi", model, 30)
                command = next(
                    c.args[0] for c in popen.call_args_list
                    if c.args and c.args[0][0] == "claude"
                )
                pairs = _pairs(command)
                self.assertIn(("--effort", expected), pairs)
                self.assertIn(("--model", model), pairs)

    def test_no_model_is_refused(self):
        for provider in ("claude-cli", "codex-cli"):
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, "explicit model"):
                cli_chat(provider, "hi", None, 30)


def test_image_relay_names_model_effort_and_tier(tmp_path, monkeypatch):
    out = tmp_path / "out.png"
    seen = {}
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "codex":
            return real_run(cmd, **kw)
        seen["cmd"] = cmd
        out.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(__import__("shutil"), "which", lambda _: "/bin/codex")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("llmx.usage_log.log_usage", lambda **kw: None)
    image.generate_codex_image("a cell", str(out))
    pairs = _pairs(seen["cmd"])
    assert ("-m", image.CODEX_IMAGE_RELAY_MODEL) in pairs
    assert ("-c", f'model_reasoning_effort="{image.CODEX_IMAGE_RELAY_EFFORT}"') in pairs
    assert CODEX_HEADLESS_TIER_ARGS in pairs
