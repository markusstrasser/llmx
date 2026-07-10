"""Workspace semantics for subscription-backed agent mode."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from llmx.cli_backends import cli_chat


def _claude_invocation(popen):
    for call in popen.call_args_list:
        if call.args and call.args[0] and call.args[0][0] == "claude":
            return call.args[0], call.kwargs
    raise AssertionError("claude subprocess was not launched")


class TestClaudeWorkspaceAgent(unittest.TestCase):
    @patch("llmx.cli_backends.subprocess.Popen")
    def test_agent_preserves_caller_cwd_and_native_tools(self, popen) -> None:
        process = popen.return_value
        process.pid = 123
        process.returncode = 0
        process.communicate.return_value = (
            json.dumps({"type": "result", "is_error": False, "result": "OK"}),
            "",
        )

        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "must-not-leak", "CLAUDE_API_KEY": "must-not-leak"},
        ):
            result = cli_chat(
                "claude-cli",
                "inspect the current repository",
                "claude-opus-4-8",
                30,
                mode="agent",
            )

        self.assertEqual(result, "OK")
        command, invocation = _claude_invocation(popen)
        self.assertIn("--permission-mode", command)
        self.assertIn("bypassPermissions", command)
        self.assertNotIn("--allowedTools", command)
        self.assertNotIn("--mcp-config", command)
        self.assertIsNone(invocation["cwd"])
        self.assertNotIn("ANTHROPIC_API_KEY", invocation["env"])
        self.assertNotIn("CLAUDE_API_KEY", invocation["env"])

    @patch("llmx.cli_backends.subprocess.Popen")
    def test_legacy_research_profile_stays_in_isolated_cwd(self, popen) -> None:
        process = popen.return_value
        process.pid = 123
        process.returncode = 0
        process.communicate.return_value = (
            json.dumps({"type": "result", "is_error": False, "result": "OK"}),
            "",
        )

        result = cli_chat(
            "claude-cli",
            "find papers",
            "claude-opus-4-8",
            30,
            lite="research",
            mode="agent",
        )

        self.assertEqual(result, "OK")
        command, invocation = _claude_invocation(popen)
        self.assertIn("--allowedTools", command)
        self.assertIn("mcp__research", command)
        self.assertNotIn("--permission-mode", command)
        self.assertEqual(
            Path(invocation["cwd"]),
            Path.home() / ".cache" / "llmx" / "lite" / "research",
        )


if __name__ == "__main__":
    unittest.main()
