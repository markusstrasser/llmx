import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmx.cli_backends import (
    CliBackendFailure,
    _codex_session_id,
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


def _write_rollout(path: Path, input_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "event_msg", "payload": {"type": "irrelevant"}})
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": input_tokens,
                            "cached_input_tokens": 3,
                            "output_tokens": 5,
                            "reasoning_output_tokens": 2,
                            "total_tokens": input_tokens + 5,
                        }
                    },
                },
            }
        )
        + "\n"
    )


_OURS = "01a0e8c2-1c3c-7cf3-814f-a9ba9ee4657e"
_OTHER = "01a0e8c2-1c3c-78d3-826a-e70715a36db1"


class TestCodexRolloutUsage(unittest.TestCase):
    def test_reads_rollout_named_by_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(root / "2026" / "07" / "06" / f"rollout-2026-07-06T01-03-50-{_OURS}.jsonl", 11)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=root)

        self.assertIsNone(note)
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["cached_tokens"], 3)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["reasoning_tokens"], 2)

    def test_parallel_calls_each_read_their_own_rollout(self):
        # 2026-09-29: parallel codex calls logged the newest rollout, which belonged to
        # another call (still running, so sometimes null tokens).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "09" / "29"
            ours = day / f"rollout-2026-09-29T01-03-50-{_OURS}.jsonl"
            other = day / f"rollout-2026-09-29T01-03-50-{_OTHER}.jsonl"
            _write_rollout(ours, 11)
            _write_rollout(other, 99)
            later = ours.stat().st_mtime_ns + 1_000_000_000
            os.utime(other, ns=(later, later))

            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=root)

        self.assertIsNone(note)
        self.assertEqual(usage["prompt_tokens"], 11)

    def test_without_session_id_several_new_rollouts_leave_usage_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "09" / "29"
            _write_rollout(day / f"rollout-2026-09-29T01-03-50-{_OURS}.jsonl", 11)
            _write_rollout(day / f"rollout-2026-09-29T01-03-50-{_OTHER}.jsonl", 99)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, root=root)

        self.assertIsNone(usage["prompt_tokens"])
        self.assertIn("2 new codex rollouts", note)

    def test_without_session_id_single_new_rollout_is_attributed_by_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(root / "2026" / "09" / "29" / f"rollout-2026-09-29T01-03-50-{_OURS}.jsonl", 11)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, root=root)

        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertIn("attributed by time", note)

    def test_changed_parent_rollout_is_never_attributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "2026" / "09" / "29" / f"rollout-2026-09-29T00-00-00-{_OTHER}.jsonl"
            _write_rollout(parent, 99)
            before = {parent: parent.stat().st_mtime_ns - 1}

            usage, note = _latest_codex_rollout_usage(before, started_at=0, root=root)

        self.assertIsNone(usage["prompt_tokens"])
        self.assertIn("0 new codex rollouts", note)

    def test_session_id_without_rollout_leaves_usage_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=Path(tmp))

        self.assertIsNone(usage["prompt_tokens"])
        self.assertIn(f"0 codex rollouts match session {_OURS}", note)

    def test_session_id_parsed_from_stderr_header(self):
        stderr = (
            "Reading additional input from stdin...\n"
            "OpenAI Codex v0.156.1\n--------\nworkdir: /tmp/x\nmodel: gpt-6-astra\n"
            "session id: 01A0E8D3-D4D8-7530-8ACC-D342521484F1\n--------\nuser\nhi\n"
        )
        self.assertEqual(_codex_session_id(stderr), "01a0e8d3-d4d8-7530-8acc-d342521484f1")
        self.assertIsNone(_codex_session_id("user\nsession id: not-an-id\n"))
        self.assertIsNone(_codex_session_id(None))

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
    def test_cache_write_usage_is_preserved_and_missing_stays_unknown(self):
        raw = {
            "input_tokens": 15_000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 12_000, "cache_write_tokens": 3_000},
        }
        usage = _normalize_usage("openai", raw)
        self.assertEqual(usage["cache_write_tokens"], 3_000)
        self.assertTrue(usage["completion_includes_reasoning"])
        del raw["input_tokens_details"]["cache_write_tokens"]
        self.assertIsNone(_normalize_usage("openai", raw)["cache_write_tokens"])

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


class TestAstraCost(unittest.TestCase):
    def test_report_and_guard_agree_when_cache_writes_are_unreported(self):
        from datetime import datetime, timezone

        from llmx.spend_guard import metered_spend_today
        from llmx.usage_report import est_cost, summarize

        now = datetime.now(timezone.utc)
        row = {
            "ts": now.isoformat(), "model": "gpt-6-astra", "transport": "api",
            "prompt_tokens": 300_000, "completion_tokens": 10_000,
            "cached_tokens": 0, "cache_write_tokens": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(metered_spend_today(path), (8.25, True))
            report = summarize(by="model", since=now.date().isoformat(), log=path)
        self.assertIn("$8.25", report)
        self.assertIn("conservative estimate", report)
        self.assertIn("cache-write rate", report)
        self.assertEqual(est_cost("gpt-6-astra", 300_000, 10_000), 6.75)

    def test_unreported_totals_are_unknown_but_explicit_zero_is_known(self):
        from llmx.usage_report import cost_for_usage

        complete = {"model": "gpt-6-astra", "prompt_tokens": 0, "completion_tokens": 0}
        self.assertEqual(cost_for_usage(complete), 0.0)
        for field in ("prompt_tokens", "completion_tokens"):
            with self.subTest(field=field):
                row = {**complete, field: None}
                self.assertIsNone(cost_for_usage(row))
                del row[field]
                self.assertIsNone(cost_for_usage(row))

    def test_rollup_marks_missing_usage_unknown_in_group_and_total(self):
        from llmx.usage_report import summarize

        row = {
            "ts": "2026-09-05T00:00:00Z", "model": "gpt-6-astra", "provider": "openai",
            "prompt_tokens": None, "completion_tokens": None, "error": "GeneratorExit: ",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text(json.dumps(row) + "\n")
            report = summarize(by="model", since="2026-09-05", log=path)
        self.assertEqual(report.count("$0.00+?"), 2)
        self.assertIn("unreported usage", report)

    def test_usage_log_preserves_cache_and_output_semantics_for_pricing(self):
        from llmx.usage_log import log_usage
        from llmx.usage_report import cost_for_usage

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            with patch("llmx.usage_log._LOG_PATH", path), patch(
                "llmx.usage_log._resolve_caller", return_value="offline-test"
            ):
                log_usage(
                    provider="openai", model="gpt-6-astra", transport="api",
                    reasoning_effort="max", prompt_tokens=300_000,
                    completion_tokens=10_000, reasoning_tokens=8_000,
                    cached_tokens=200_000, cache_write_tokens=50_000,
                    completion_includes_reasoning=True, latency_s=0.1,
                )
            row = json.loads(path.read_text())
        self.assertEqual(row["cache_write_tokens"], 50_000)
        self.assertTrue(row["completion_includes_reasoning"])
        self.assertAlmostEqual(cost_for_usage(row), 3.40)

    def test_pricing_boundaries_and_disjoint_cache_categories(self):
        from llmx.usage_report import est_cost

        cases = [
            (272_000, 10_000, 0, 0, 3.22),
            (272_001, 10_000, 0, 0, 6.19002),
            (300_000, 10_000, 0, 0, 6.75),
            (100_000, 10_000, 80_000, 0, 0.78),
            (100_000, 0, 0, 100_000, 1.25),
            (300_000, 10_000, 200_000, 50_000, 3.40),
        ]
        for model in ("gpt-6-astra", "gpt-6"):
            for prompt, output, cached, written, expected in cases:
                with self.subTest(model=model, prompt=prompt, cached=cached, written=written):
                    self.assertAlmostEqual(est_cost(
                        model, prompt, output, cached_tokens=cached, cache_write_tokens=written,
                    ), expected)
        self.assertEqual(est_cost("gpt-5.6-luna", 1_000_000, 1_000_000), 1.40)
        self.assertIsNone(est_cost("unknown", 1, 1))

    def test_reasoning_subset_is_not_added_twice(self):
        from llmx.usage_report import cost_for_usage, output_tokens

        row = {
            "model": "gpt-6-astra", "provider": "openai",
            "prompt_tokens": 10_000, "completion_tokens": 10_000, "reasoning_tokens": 8_000,
        }
        self.assertEqual(output_tokens(row), 10_000)
        self.assertAlmostEqual(cost_for_usage(row), 0.60)
        row.update(provider="google", model="gemini-3.8-flash", completion_includes_reasoning=False)
        self.assertEqual(output_tokens(row), 18_000)
        row.update(provider="openrouter", model="qwen/qwen3.8-27b", completion_includes_reasoning=True)
        self.assertEqual(output_tokens(row), 10_000)

    def test_rollup_prices_request_rows_before_aggregation(self):
        from llmx.usage_report import summarize

        row = {
            "ts": "2026-09-05T00:00:00Z", "model": "gpt-6-astra", "provider": "openai",
            "prompt_tokens": 100_000, "completion_tokens": 10_000, "reasoning_tokens": 8_000,
            "cached_tokens": 80_000, "cache_write_tokens": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text((json.dumps(row) + "\n") * 3)
            report = summarize(by="model", since="2026-09-05", log=path)
        self.assertIn("2.34", report)


if __name__ == "__main__":
    unittest.main()
