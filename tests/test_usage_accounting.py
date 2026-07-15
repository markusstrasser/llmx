import json
import tempfile
import unittest
from pathlib import Path

from llmx.cli_backends import (
    CliBackendFailure,
    _latest_codex_rollout_usage,
    _parse_claude_json,
)
from llmx.providers import LlmxError, _normalize_usage


def _claude_verbose_stdout(
    result: str,
    *,
    text_blocks: list[str] | None = None,
    usage: dict | None = None,
    model_usage: dict | None = None,
) -> str:
    return json.dumps(
        [
            {
                "type": "assistant",
                "message": {
                    "id": "msg-final",
                    "content": [
                        {"type": "text", "text": text}
                        for text in (text_blocks if text_blocks is not None else [result])
                    ],
                },
            },
            {
                "type": "result",
                "is_error": False,
                "result": result,
                "usage": usage or {},
                "modelUsage": model_usage or {},
            },
        ]
    )


class TestCodexRolloutUsage(unittest.TestCase):
    def test_reads_latest_new_rollout_token_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "07" / "06"
            day.mkdir(parents=True)
            rollout = day / "rollout-test.jsonl"
            rollout.write_text(
                json.dumps({"type": "event_msg", "payload": {"type": "irrelevant"}})
                + "\n"
                + json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "last_token_usage": {
                                    "input_tokens": 11,
                                    "cached_input_tokens": 3,
                                    "output_tokens": 5,
                                    "reasoning_output_tokens": 2,
                                    "total_tokens": 16,
                                }
                            },
                        },
                    }
                )
                + "\n"
            )

            usage, note = _latest_codex_rollout_usage(
                {},
                started_at=0,
                root=root,
            )

        self.assertIsNone(note)
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["cached_tokens"], 3)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["reasoning_tokens"], 2)

    def test_missing_rollout_returns_null_usage_with_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage, note = _latest_codex_rollout_usage(
                {},
                started_at=0,
                root=Path(tmp) / "missing",
            )

        self.assertTrue(note)
        self.assertIsNone(usage["prompt_tokens"])
        self.assertIsNone(usage["completion_tokens"])
        self.assertIsNone(usage["reasoning_tokens"])


class TestClaudeCliUsage(unittest.TestCase):
    def test_parse_claude_json_keeps_absent_reasoning_null(self):
        stdout = _claude_verbose_stdout(
            "ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 4,
            },
            model_usage={"claude-fable-5[1m]": {}},
        )

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "ok")
        self.assertEqual(usage["model"], "claude-fable-5")
        self.assertIsNone(usage["reasoning_tokens"])

    def test_parse_claude_json_reads_reasoning_when_exposed(self):
        stdout = _claude_verbose_stdout(
            "ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        )

        _, usage = _parse_claude_json(stdout)

        self.assertEqual(usage["reasoning_tokens"], 5)

    def test_parse_claude_json_preserves_reported_zero_reasoning(self):
        stdout = _claude_verbose_stdout(
            "ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "reasoning_tokens": 0,
            },
        )

        _, usage = _parse_claude_json(stdout)

        self.assertEqual(usage["reasoning_tokens"], 0)

    def test_parse_claude_json_rejects_last_block_only_projection(self):
        stdout = json.dumps(
            [
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg-final",
                        "content": [{"type": "text", "text": "PREFIX"}],
                    },
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg-final",
                        "content": [{"type": "text", "text": "TAIL"}],
                    },
                },
                {
                    "type": "result",
                    "is_error": False,
                    "result": "TAIL",
                },
            ]
        )

        result, usage = _parse_claude_json(stdout)

        self.assertIsInstance(result, CliBackendFailure)
        self.assertEqual(result.kind, LlmxError)
        self.assertIn("omitted assistant text blocks", result.detail)
        self.assertIn("reconstructed_chars=10", result.detail)
        self.assertIn("result_chars=4", result.detail)
        self.assertIn("omitted_chars=6", result.detail)
        self.assertIsNone(usage)

    def test_parse_claude_json_accepts_complete_multiblock_projection(self):
        stdout = _claude_verbose_stdout(
            "PREFIXTAIL",
            text_blocks=["PREFIX", "TAIL"],
        )

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "PREFIXTAIL")
        self.assertIsInstance(usage, dict)

    def test_parse_claude_json_accepts_complete_single_block_projection(self):
        stdout = _claude_verbose_stdout("complete response")

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "complete response")
        self.assertIsInstance(usage, dict)


class TestOpenRouterUsage(unittest.TestCase):
    def test_normalize_usage_accepts_dict_reasoning_details(self):
        usage = _normalize_usage(
            "openrouter",
            {
                "prompt_tokens": 13,
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 6},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        )

        self.assertEqual(usage["prompt_tokens"], 13)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual(usage["total_tokens"], 21)
        self.assertEqual(usage["reasoning_tokens"], 6)
        self.assertEqual(usage["cached_tokens"], 4)

    def test_normalize_usage_accepts_reasoning_variant(self):
        usage = _normalize_usage(
            "openrouter",
            {
                "input_tokens": 13,
                "output_tokens": 8,
                "output_tokens_details": {"reasoning": 6},
                "input_tokens_details": {"cached_input_tokens": 4},
            },
        )

        self.assertEqual(usage["prompt_tokens"], 13)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual(usage["reasoning_tokens"], 6)
        self.assertEqual(usage["cached_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
