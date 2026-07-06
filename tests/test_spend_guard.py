"""Decision-logic tests for the metered-spend hard cap (llmx/spend_guard.py).

No real dispatch — the guard is exercised against fixture ledgers via log_path.
Covers: over-cap refuse, under-cap allow, unpriced-model refuse, override bypass,
missing/unreadable ledger fail-open, subscription rows excluded, agent-api counted,
and the research (check_model_priced=False) path.
"""

import json
import os
import tempfile
import unittest
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
        # claude-opus-4-8 = (5,25)/M → 1M in + 1M out = $30 metered.
        # A subscription row (claude-cli) and a yesterday row must NOT count.
        yday = "2020-01-01T10:00:00Z"
        p = self._ledger(
            _row("claude-opus-4-8", "api", 1_000_000, 1_000_000),
            _row("claude-opus-4-8", "claude-cli", 9_000_000, 9_000_000),
            _row("claude-opus-4-8", "api", 1_000_000, 1_000_000, ts=yday),
        )
        spend, ok = sg.metered_spend_today(p)
        self.assertTrue(ok)
        self.assertAlmostEqual(spend, 30.0, places=4)

    def test_agent_api_counts(self):
        # agent-api rows are metered; an unpriced model contributes $0 to the sum
        # (can't be priced) but the transport is still counted as metered.
        p = self._ledger(_row("agent:deep-research", "agent-api", 100, 100))
        spend, ok = sg.metered_spend_today(p)
        self.assertTrue(ok)
        self.assertEqual(spend, 0.0)  # unpriced → $0 historical, but see over-cap test

    # --- enforce: refuse / allow ---
    def test_over_cap_refuses(self):
        p = self._ledger(_row("claude-opus-4-8", "api", 1_000_000, 1_000_000))  # $30
        with self.assertRaises(SpendCapError) as cm:
            sg.enforce_daily_cap("claude-opus-4-8", log_path=p)
        self.assertEqual(cm.exception.exit_code, 7)

    def test_under_cap_allows(self):
        p = self._ledger(_row("claude-opus-4-8", "api", 100_000, 100_000))  # ~$3
        sg.enforce_daily_cap("claude-opus-4-8", log_path=p)  # no raise

    def test_exactly_at_cap_refuses(self):
        # 1M in @ $5 + 800k out @ $25 = $5 + $20 = $25.00 == cap → refuse (>=)
        p = self._ledger(_row("claude-opus-4-8", "api", 1_000_000, 800_000))
        with self.assertRaises(SpendCapError):
            sg.enforce_daily_cap("claude-opus-4-8", log_path=p)

    def test_unpriced_model_refuses(self):
        p = self._ledger()  # empty → $0 spend, well under cap
        with self.assertRaises(SpendCapError) as cm:
            sg.enforce_daily_cap("totally-unknown-model", log_path=p)
        self.assertEqual(cm.exception.exit_code, 7)
        self.assertIn("unpriced", str(cm.exception).lower())

    def test_research_path_skips_unpriced_but_enforces_cap(self):
        empty = self._ledger()
        # unpriced agent model allowed under cap when check_model_priced=False
        sg.enforce_daily_cap("agent:deep-research", log_path=empty, check_model_priced=False)
        # but still refused over cap
        over = self._ledger(_row("claude-opus-4-8", "api", 1_000_000, 1_000_000))
        with self.assertRaises(SpendCapError):
            sg.enforce_daily_cap("agent:deep-research", log_path=over, check_model_priced=False)

    # --- override ---
    def test_override_bypasses_over_cap(self):
        os.environ[sg._OVERRIDE_ENV] = "1"
        p = self._ledger(_row("claude-opus-4-8", "api", 1_000_000, 1_000_000))  # $30
        sg.enforce_daily_cap("claude-opus-4-8", log_path=p)  # no raise

    def test_override_bypasses_unpriced(self):
        os.environ[sg._OVERRIDE_ENV] = "1"
        p = self._ledger()
        sg.enforce_daily_cap("totally-unknown-model", log_path=p)  # no raise

    # --- fail-open on unreadable ledger ---
    def test_missing_ledger_fails_open(self):
        # No raise; guard fails open (loud [DEGRADED] warning to stderr).
        sg.enforce_daily_cap("claude-opus-4-8", log_path="/nonexistent/does/not/exist.jsonl")

    def test_missing_ledger_spend_reports_not_ok(self):
        spend, ok = sg.metered_spend_today("/nonexistent/does/not/exist.jsonl")
        self.assertFalse(ok)
        self.assertEqual(spend, 0.0)


if __name__ == "__main__":
    unittest.main()
