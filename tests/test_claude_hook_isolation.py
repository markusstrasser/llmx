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


OUTPUT_LIMIT_TURN = {
    "type": "user",
    "message": {"role": "user", "content": [{"type": "text", "text": (
        "Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
        "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")}]},
}


def _cut_reply(*, first_stop: str | None = "max_tokens", then: dict | None = None) -> str:
    """A reply cut at the output cap and resumed (2026-10-14, Claude Code 2.1.299)."""
    first = {"type": "assistant", "message": {"id": "msg-1", "stop_reason": first_stop, "content": [
        {"type": "thinking", "thinking": "..."}, {"type": "text", "text": "## Findings\n1. The rate is sta"}]}}
    if first_stop is None:
        del first["message"]["stop_reason"]
    second = {"type": "assistant", "message": {"id": "msg-2", "stop_reason": "end_turn", "content": [
        {"type": "text", "text": "le.\n2. The cap is stated."}]}}
    events = [first, OUTPUT_LIMIT_TURN, second]
    final = "le.\n2. The cap is stated."
    if then is not None:
        third = {"type": "assistant", "message": {"id": "msg-3", "content": [{"type": "text", "text": "Done."}]}}
        events += [then, third]
        final = "Done."
    return json.dumps(events + [{"type": "result", "is_error": False, "result": final, "num_turns": len(events)}])


class TestOutputLimitContinuation(unittest.TestCase):
    def test_reply_cut_at_the_cap_is_stitched(self) -> None:
        result, usage = _parse_claude_json(_cut_reply())
        self.assertEqual(result, "## Findings\n1. The rate is stale.\n2. The cap is stated.")
        self.assertEqual(usage["output_limit_continuations"], 1)

    def test_event_without_stop_reason_is_stitched(self) -> None:
        result, _ = _parse_claude_json(_cut_reply(first_stop=None))
        self.assertEqual(result, "## Findings\n1. The rate is stale.\n2. The cap is stated.")

    def test_recovery_after_a_finished_message_is_refused(self) -> None:
        result, usage = _parse_claude_json(_cut_reply(first_stop="end_turn"))
        assert isinstance(result, CliBackendFailure)
        self.assertIn("max_tokens", result.detail)
        self.assertIsNone(usage)

    def test_stop_hook_after_a_recovery_is_still_refused(self) -> None:
        result, _ = _parse_claude_json(_cut_reply(then=STOP_HOOK_TURN))
        assert isinstance(result, CliBackendFailure)
        self.assertIn("Stop hook feedback", result.detail)

    def test_agent_mode_keeps_the_final_message(self) -> None:
        result, usage = _parse_claude_json(_cut_reply(), allow_continuation=True)
        self.assertEqual(result, "le.\n2. The cap is stated.")
        self.assertNotIn("output_limit_continuations", usage)


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
