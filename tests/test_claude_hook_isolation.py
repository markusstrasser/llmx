"""One-shot Claude calls must not answer the caller's hooks instead of the prompt.

2026-09-27: a subscription chat call ran all user-level hooks; a Stop hook
blocked, the model replied to the hook, and llmx returned that reply (rc=0) as
the extraction result. Chat and research-profile calls now disable hooks, and
the parser refuses a turn injected after the answer.
"""

import json
import unittest
from unittest.mock import patch

from llmx.cli_backends import CliBackendFailure, _parse_claude_json, cli_chat

HOOKS_OFF = json.dumps({"disableAllHooks": True})


def _events(*, injected: dict | None = None) -> str:
    """Verbose JSON as Claude Code 2.1.283 emits it (shape from a live repro)."""
    first = {"type": "assistant", "message": {"id": "msg-1", "content": [{"type": "text", "text": "OK"}]}}
    if injected is None:
        return json.dumps([first, {"type": "result", "is_error": False, "result": "OK"}])
    reply = "Confirmed: I made no commits."
    return json.dumps(
        [
            first,
            injected,
            {"type": "assistant", "message": {"id": "msg-2", "content": [{"type": "text", "text": reply}]}},
            {"type": "result", "is_error": False, "result": reply, "num_turns": 2},
        ]
    )


STOP_HOOK_TURN = {
    "type": "user",
    "message": {"role": "user", "content": [{"type": "text", "text": "Stop hook feedback:\nconfirm you made no commits."}]},
}
TOOL_RESULT_TURN = {
    "type": "user",
    "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "denied"}]},
}


class TestParserRefusesInjectedTurns(unittest.TestCase):
    def test_single_answer_passes(self) -> None:
        result, _ = _parse_claude_json(_events())
        self.assertEqual(result, "OK")

    def test_stop_hook_reply_is_refused(self) -> None:
        result, usage = _parse_claude_json(_events(injected=STOP_HOOK_TURN))
        assert isinstance(result, CliBackendFailure)
        self.assertIn("Stop hook feedback", result.detail)
        self.assertIsNone(usage)

    def test_agent_mode_may_be_steered_by_hooks(self) -> None:
        result, _ = _parse_claude_json(_events(injected=STOP_HOOK_TURN), allow_continuation=True)
        self.assertEqual(result, "Confirmed: I made no commits.")

    def test_tool_result_turn_is_not_an_injection(self) -> None:
        result, _ = _parse_claude_json(_events(injected=TOOL_RESULT_TURN))
        self.assertEqual(result, "Confirmed: I made no commits.")


class TestChatDisablesHooks(unittest.TestCase):
    def _command(self, **kwargs) -> list[str]:
        with patch("llmx.cli_backends.subprocess.Popen") as popen:
            process = popen.return_value
            process.pid = 123
            process.returncode = 0
            process.communicate.return_value = (_events(), "")
            self.assertEqual(cli_chat("claude-cli", "hi", "claude-opus-5-5", 30, **kwargs), "OK")
        return next(c.args[0] for c in popen.call_args_list if c.args and c.args[0][0] == "claude")

    def _settings(self, command: list[str]) -> str | None:
        return command[command.index("--settings") + 1] if "--settings" in command else None

    def test_chat_disables_hooks(self) -> None:
        self.assertEqual(self._settings(self._command()), HOOKS_OFF)

    def test_research_profile_disables_hooks(self) -> None:
        self.assertEqual(self._settings(self._command(lite="research", mode="agent")), HOOKS_OFF)

    def test_workspace_agent_keeps_hooks(self) -> None:
        self.assertIsNone(self._settings(self._command(mode="agent")))


if __name__ == "__main__":
    unittest.main()
