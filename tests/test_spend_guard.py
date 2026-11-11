"""Decision-logic tests for the metered-spend hard cap (llmx/spend_guard.py).

No real dispatch — the guard is exercised against fixture ledgers via log_path.
Covers: over-cap refuse, under-cap allow, unpriced-model refuse, override bypass,
missing/unreadable ledger fail-open, subscription rows excluded, agent-api counted,
and the research (check_model_priced=False) path.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

from llmx.providers import SpendCapError
from llmx import spend_guard as sg


def _row(model, transport, ptok=0, ctok=0, rtok=0, *, ts=None):
    ts = ts or (datetime.now(timezone.utc).date().isoformat() + "T10:00:00Z")
    return json.dumps({
        "ts": ts, "model": model, "transport": transport,
        "prompt_tokens": ptok, "completion_tokens": ctok, "reasoning_tokens": rtok,
    })


def _ledger(*rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    f.write("\n".join(rows) + ("\n" if rows else ""))
    f.close()
    return f.name


class TestSpendGuard(unittest.TestCase):
    def setUp(self):
        os.environ.pop(sg._OVERRIDE_ENV, None)
        self._tmp = []

    def tearDown(self):
        os.environ.pop(sg._OVERRIDE_ENV, None)
        for p in self._tmp:
            Path(p).unlink(missing_ok=True)

    def _ledger(self, *rows):
        p = _ledger(*rows)
        self._tmp.append(p)
        return p

    # --- transport predicate ---
    def test_is_metered_transport(self):
        for t in ("api", "agent-api", "openai-api", "anthropic-direct-api"):
            self.assertTrue(sg.is_metered_transport(t), t)
        for t in ("claude-cli", "codex-cli", "", None):
            self.assertFalse(sg.is_metered_transport(t), t)

    # --- spend summation ---
    def test_metered_spend_sums_only_metered_today(self):
        # claude-opus-5 = (5,25)/M → 1M in + 1M out = $30 metered.
        # A subscription row (claude-cli) and a yesterday row must NOT count.
        yday = "2020-01-01T10:00:00Z"
        p = self._ledger(
            _row("claude-opus-5", "api", 1_000_000, 1_000_000),
            _row("claude-opus-5", "claude-cli", 9_000_000, 9_000_000),
            _row("claude-opus-5", "api", 1_000_000, 1_000_000, ts=yday),
        )
        spend, ok = sg.metered_spend_today(p)
        self.assertTrue(ok)
        self.assertAlmostEqual(spend, 30.0, places=4)

    def test_agent_api_counts(self):
        # An unpriced metered row makes the known subtotal incomplete.
        p = self._ledger(_row("agent:deep-research", "agent-api", 100, 100))
        spend, ok = sg.metered_spend_today(p)
        self.assertFalse(ok)
        self.assertEqual(spend, 0.0)

    def test_interrupted_null_usage_is_incomplete_and_fails_open_loudly(self):
        row = json.loads(_row("gpt-6-astra", "api", None, None, None))
        row.update(error="GeneratorExit: ", note="provider usage unreported")
        path = self._ledger(json.dumps(row))
        self.assertEqual(sg.metered_spend_today(path), (0.0, False))
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            sg.enforce_daily_cap("gpt-6-astra", log_path=path)
        self.assertIn("[DEGRADED]", stderr.getvalue())
        self.assertIn("incomplete", stderr.getvalue())

    def test_null_usage_never_bypasses_known_cap_breach(self):
        unknown = _row("gpt-6-astra", "api", None, None, None)
        known = json.loads(_row("gpt-6-astra", "api", 300_000, 10_000))
        known.update(cached_tokens=0, cache_write_tokens=0)
        for unknown_first in (True, False):
            with self.subTest(unknown_first=unknown_first):
                rows = [json.dumps(known)] * 4
                rows.insert(0 if unknown_first else len(rows), unknown)
                path = self._ledger(*rows)
                self.assertEqual(sg.metered_spend_today(path), (27.0, False))
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaisesRegex(SpendCapError, "cap reached"):
                    sg.enforce_daily_cap("gpt-6-astra", log_path=path)
                self.assertIn("[DEGRADED]", stderr.getvalue())

    # --- enforce: refuse / allow ---
    def test_over_cap_refuses(self):
        p = self._ledger(_row("claude-opus-5", "api", 1_000_000, 1_000_000))  # $30
        with self.assertRaises(SpendCapError) as cm:
            sg.enforce_daily_cap("claude-opus-5", log_path=p)
        self.assertEqual(cm.exception.exit_code, 7)

    def test_under_cap_allows(self):
        p = self._ledger(_row("claude-opus-5", "api", 100_000, 100_000))  # ~$3
        sg.enforce_daily_cap("claude-opus-5", log_path=p)  # no raise

    def test_exactly_at_cap_refuses(self):
        # 1M in @ $5 + 200k out @ $25 = $5 + $5 = $10.00 == cap → refuse (>=)
        self.assertEqual(sg.DAILY_CAP_USD, 10.0)
        p = self._ledger(_row("claude-opus-5", "api", 1_000_000, 200_000))
        with self.assertRaises(SpendCapError):
            sg.enforce_daily_cap("claude-opus-5", log_path=p)

    def test_unpriced_model_refuses(self):
        p = self._ledger()  # empty → $0 spend, well under cap
        with self.assertRaises(SpendCapError) as cm:
            sg.enforce_daily_cap("totally-unknown-model", log_path=p)
        self.assertEqual(cm.exception.exit_code, 7)
        self.assertIn("unpriced", str(cm.exception).lower())

    def test_unknown_model_chat_and_stream_refuse_before_sdk_construction(self):
        from unittest.mock import patch

        from llmx.api import LLM

        with (
            patch("llmx.api.check_api_key"),
            patch("llmx.providers.OpenAI") as chat_client,
            patch("openai.OpenAI") as stream_client,
        ):
            llm = LLM(provider="openai", model="unknown-test-model", auth="api")
            with self.assertRaisesRegex(SpendCapError, "unpriced"):
                llm.chat("hi")
            with self.assertRaisesRegex(SpendCapError, "unpriced"):
                list(llm.stream("hi"))
        chat_client.assert_not_called()
        stream_client.assert_not_called()

    def test_astra_long_context_spend_blocks_at_actual_tier(self):
        row = json.loads(_row("gpt-6-astra", "api", 300_000, 10_000))
        row.update(provider="openai", cached_tokens=0, cache_write_tokens=0)
        path = self._ledger(*(json.dumps(row) for _ in range(4)))
        self.assertAlmostEqual(sg.metered_spend_today(path)[0], 27.0)
        with self.assertRaisesRegex(SpendCapError, "cap reached"):
            sg.enforce_daily_cap("gpt-6-astra", log_path=path)

    def test_astra_unknown_cache_writes_use_conservative_guard_rate(self):
        path = self._ledger(_row("gpt-6-astra", "api", 300_000, 10_000))
        self.assertAlmostEqual(sg.metered_spend_today(path)[0], 8.25)

    def test_research_path_skips_unpriced_but_enforces_cap(self):
        empty = self._ledger()
        # unpriced agent model allowed under cap when check_model_priced=False
        sg.enforce_daily_cap("agent:deep-research", log_path=empty, check_model_priced=False)
        # but still refused over cap
        over = self._ledger(_row("claude-opus-5", "api", 1_000_000, 1_000_000))
        with self.assertRaises(SpendCapError):
            sg.enforce_daily_cap("agent:deep-research", log_path=over, check_model_priced=False)

    # --- override ---
    def test_override_bypasses_over_cap(self):
        os.environ[sg._OVERRIDE_ENV] = "1"
        p = self._ledger(_row("claude-opus-5", "api", 1_000_000, 1_000_000))  # $30
        sg.enforce_daily_cap("claude-opus-5", log_path=p)  # no raise

    def test_override_bypasses_unpriced(self):
        os.environ[sg._OVERRIDE_ENV] = "1"
        p = self._ledger()
        sg.enforce_daily_cap("totally-unknown-model", log_path=p)  # no raise

    # --- image rows (transport images-api) ---
    def test_metered_image_row_is_priced_and_keeps_ledger_ok(self):
        p = self._ledger(_row("gpt-image-2.6-flare", "images-api", 35, 555))
        spend, ok = sg.metered_spend_today(p)
        self.assertTrue(ok)
        # No modality split in the row: image rates, $8 in and $30 out per MTok.
        self.assertAlmostEqual(spend, (35 * 8 + 555 * 30) / 1_000_000, places=9)
        self.assertGreater(spend, 0)

    def test_truly_unpriced_row_makes_ledger_not_ok(self):
        # Positive control: the guard still notices a row it cannot price.
        p = self._ledger(_row("gpt-image-99-unpriced", "images-api", 35, 555))
        spend, ok = sg.metered_spend_today(p)
        self.assertFalse(ok)
        self.assertEqual(spend, 0.0)

    def test_image_row_missing_usage_makes_ledger_not_ok(self):
        row = json.dumps({
            "ts": datetime.now(timezone.utc).date().isoformat() + "T10:00:00Z",
            "model": "gpt-image-2.6-flare", "transport": "images-api",
            "prompt_tokens": None, "completion_tokens": None, "billing": "metered",
        })
        self.assertFalse(sg.metered_spend_today(self._ledger(row))[1])

    # --- fail-open on unreadable ledger ---
    def test_missing_ledger_fails_open(self):
        # No raise; guard fails open (loud [DEGRADED] warning to stderr).
        sg.enforce_daily_cap("claude-opus-5", log_path="/nonexistent/does/not/exist.jsonl")

    def test_missing_ledger_spend_reports_not_ok(self):
        spend, ok = sg.metered_spend_today("/nonexistent/does/not/exist.jsonl")
        self.assertFalse(ok)
        self.assertEqual(spend, 0.0)


if __name__ == "__main__":
    unittest.main()


def test_cap_notification_fires_once_per_day(tmp_path):
    from unittest.mock import patch

    with patch.object(sg.shutil, "which", return_value="/usr/bin/true"), patch.object(
        sg.subprocess, "Popen"
    ) as popen:
        assert sg.notify_cap_reached(12.5, 10.0, marker_dir=tmp_path) is True
        assert sg.notify_cap_reached(13.0, 10.0, marker_dir=tmp_path) is False
    assert popen.call_count == 1
    assert "$12.50 metered today, cap $10.00" in " ".join(popen.call_args.args[0])


def test_cap_notification_never_raises_without_a_notifier(tmp_path):
    from unittest.mock import patch

    with patch.object(sg.shutil, "which", return_value=None):
        assert sg.notify_cap_reached(12.5, 10.0, marker_dir=tmp_path) is False


def test_override_above_cap_still_notifies_the_operator(tmp_path):
    from unittest.mock import patch

    with patch.dict(sg.os.environ, {"LLMX_SPEND_OVERRIDE": "1"}), patch.object(
        sg, "metered_spend_today", return_value=(14.0, True)
    ), patch.object(sg, "notify_cap_reached") as notify:
        sg.enforce_daily_cap("gpt-6-astra")
    assert notify.call_args.kwargs["kind"] == "override"
