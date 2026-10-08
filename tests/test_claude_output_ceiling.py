"""claude-cli calls raise Claude Code's output cap to the model's ceiling.

2026-10-08: two Opus 5.5 max-effort judge calls stopped at the CLI's output cap
("Claude's response exceeded the 64000 output token maximum") and left 0-byte
answers. The model's own ceiling is 128K; a caller's explicit cap still wins.
"""

import json
import os
import unittest
from unittest.mock import patch

from llmx.cli_backends import cli_chat

CAP = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"


def _events() -> str:
    first = {"type": "assistant", "message": {"id": "msg-1", "content": [{"type": "text", "text": "OK"}]}}
    return json.dumps([first, {"type": "result", "is_error": False, "result": "OK"}])


class TestClaudeOutputCeiling(unittest.TestCase):
    def _env(self, model: str | None, **kwargs) -> dict[str, str]:
        with patch("llmx.cli_backends.subprocess.Popen") as popen:
            process = popen.return_value
            process.pid = 123
            process.returncode = 0
            process.communicate.return_value = (_events(), "")
            self.assertEqual(cli_chat("claude-cli", "hi", model, 30, **kwargs), "OK")
        call = next(c for c in popen.call_args_list if c.args and c.args[0][0] == "claude")
        return call.kwargs["env"]

    def test_subscription_models_get_their_ceiling(self) -> None:
        for model in ("claude-opus-5-5", "claude-fable-5-1", "claude-opus-5", "claude-fable-5-1[1m]"):
            for kwargs in ({}, {"mode": "agent"}, {"reasoning_effort": "max"}):
                with self.subTest(model=model, kwargs=kwargs), patch.dict(os.environ):
                    os.environ.pop(CAP, None)
                    self.assertEqual(self._env(model, **kwargs)[CAP], "128000")

    def test_caller_cap_wins(self) -> None:
        with patch.dict(os.environ, {CAP: "64000"}):
            self.assertEqual(self._env("claude-opus-5-5")[CAP], "64000")

    def test_unnamed_model_keeps_cli_default(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop(CAP, None)
            self.assertNotIn(CAP, self._env(None))


if __name__ == "__main__":
    unittest.main()
