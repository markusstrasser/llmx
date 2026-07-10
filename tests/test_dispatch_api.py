"""Tests for llmx.api.dispatch / DispatchResult (ADR P1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmx.dispatch_api import (
    DispatchResult,
    classify_dispatch_error,
    compose_prompt,
    dispatch,
    load_context_paths,
)
from llmx.providers import RateLimitError, QuotaError, TimeoutError_


class ComposeContextTest(unittest.TestCase):
    def test_multi_file_concat_with_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.md"
            b = root / "b.md"
            a.write_text("alpha")
            b.write_text("beta")
            text = load_context_paths([a, b])
            self.assertIn("=== File:", text)
            self.assertIn("alpha", text)
            self.assertIn("beta", text)
            full, warns = compose_prompt("Q?", context_paths=[a, b])
            self.assertIn("Q?", full)
            self.assertIn("alpha", full)
            self.assertTrue(any("concatenated 2" in w for w in warns))


class ClassifyTest(unittest.TestCase):
    def test_typed_errors(self) -> None:
        self.assertEqual(classify_dispatch_error(RateLimitError("429"))[0], "rate_limit")
        self.assertEqual(classify_dispatch_error(QuotaError("billing"))[0], "quota")
        self.assertEqual(classify_dispatch_error(TimeoutError_("t"))[0], "timeout")

    def test_heuristic(self) -> None:
        self.assertEqual(
            classify_dispatch_error(RuntimeError("429 resource_exhausted"))[0],
            "rate_limit",
        )


class DispatchApiTest(unittest.TestCase):
    def test_dry_run_returns_plan(self) -> None:
        result = dispatch(
            "hello",
            provider="google",
            model="gemini-3.5-flash",
            dry_run=True,
            auth="api",
        )
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.dry_run_plan)
        self.assertEqual(result.dry_run_plan["provider"], "google")
        self.assertIn("transport", result.dry_run_plan)

    def test_subscription_dry_run_claude(self) -> None:
        result = dispatch(
            "hello",
            model="claude-opus-4-8",
            subscription=True,
            dry_run=True,
        )
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.auth, "subscription")
        self.assertIn("claude", result.transport)

    def test_live_ok_via_mocked_chat(self) -> None:
        mock_resp = MagicMock()
        mock_resp.content = "pong"
        mock_resp.latency = 0.1
        mock_resp.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        mock_resp.provider = "google"
        mock_resp.model = "gemini-3.5-flash"

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.md"
            with patch("llmx.api.chat", return_value=mock_resp):
                result = dispatch(
                    "ping",
                    provider="google",
                    model="gemini-3.5-flash",
                    auth="api",
                    output_path=out,
                )
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.text, "pong")
            self.assertEqual(out.read_text(), "pong")
            self.assertTrue(result.ok())

    def test_live_rate_limit_returns_status(self) -> None:
        with patch("llmx.api.chat", side_effect=RateLimitError("429")):
            result = dispatch("x", provider="google", model="gemini-3.5-flash", auth="api")
        self.assertEqual(result.status, "rate_limit")
        self.assertTrue(result.retryable)
        self.assertEqual(result.exit_code, 3)

    def test_empty_output(self) -> None:
        mock_resp = MagicMock()
        mock_resp.content = "   "
        mock_resp.latency = 0.0
        mock_resp.usage = {}
        mock_resp.provider = "google"
        mock_resp.model = "m"
        with patch("llmx.api.chat", return_value=mock_resp):
            result = dispatch("x", provider="google", model="m", auth="api")
        self.assertEqual(result.status, "empty_output")

    def test_dispatch_result_content_alias(self) -> None:
        r = DispatchResult(status="ok", retryable=False, text="hi")
        self.assertEqual(r.content, "hi")


if __name__ == "__main__":
    unittest.main()
