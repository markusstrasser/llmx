"""DeepInfra transport + DeepSeek V4.1 Flash pricing (offline).

Pins the new `deepinfra` lane end to end at the request/usage boundary:
routing by `-p deepinfra` and by the prefixed model id, the outgoing
`reasoning_effort` (default "high", CLI effort overrides it), reasoning-token
accounting, and spend-guard pricing for both DeepSeek V4.1 Flash routes.

Every test patches the SDK client, so no request leaves the machine.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llmx.dispatch_plan import build_dispatch_plan, collect_routing_mirror
from llmx.providers import (
    _KNOWN_MODELS,
    OPENAI_COMPAT_URLS,
    PROVIDER_CONFIGS,
    _normalize_usage,
    get_model_name,
    infer_provider_from_model,
    list_providers,
)
from llmx.spend_guard import enforce_daily_cap
from llmx.usage_report import PRICING, cost_for_usage, est_cost

DEEPINFRA_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def _usage(*, prompt=12, completion=50, reasoning=37, cached=0):
    """The usage shape DeepInfra's OpenAI-compatible endpoint reports."""
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
    )


def _response(content="ok", usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None),
                finish_reason="stop",
            )
        ],
        usage=usage or _usage(),
        model=DEEPINFRA_MODEL,
    )


class _CapturingClient:
    """Records the kwargs handed to chat.completions.create."""

    def __init__(self, calls: list, client_kwargs: list, response):
        self._calls = calls
        self._client_kwargs = client_kwargs
        self._response = response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self._calls.append(kwargs)
        return self._response


class _Capture:
    """Patch the SDK client + usage log; collect request bodies and usage rows."""

    def __init__(self, response=None):
        self.calls: list = []
        self.client_kwargs: list = []
        self.usage_rows: list = []
        self._response = response or _response()
        self._patches: list = []

    def __enter__(self):
        def make_client(**kwargs):
            self.client_kwargs.append(kwargs)
            return _CapturingClient(self.calls, self.client_kwargs, self._response)

        self._patches = [
            patch("llmx.providers.OpenAI", make_client),
            patch("llmx.providers._get_api_key", return_value="test-key"),
            patch("llmx.providers.check_api_key", return_value=None),
            patch("llmx.spend_guard.enforce_daily_cap", return_value=None),
            patch(
                "llmx.usage_log.log_usage",
                side_effect=lambda **row: self.usage_rows.append(row),
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False

    @property
    def body(self) -> dict:
        if len(self.calls) != 1:
            raise AssertionError(f"expected exactly 1 API call, got {len(self.calls)}")
        return self.calls[0]


def _chat(provider="deepinfra", model=None, effort=None, temperature=0.7):
    """Drive the full dispatch path (llmx.providers.chat)."""
    from llmx import providers

    return providers.chat(
        prompt="hi",
        provider=provider,
        model=model,
        temperature=temperature,
        reasoning_effort=effort,
        stream=False,
        debug=False,
        json_output=False,
        timeout=30,
    )


class TestDeepInfraRouting(unittest.TestCase):
    def test_provider_config_and_base_url(self):
        cfg = PROVIDER_CONFIGS["deepinfra"]
        self.assertEqual(cfg["model"], DEEPINFRA_MODEL)
        self.assertEqual(cfg["env_var"], "DEEPINFRA_API_KEY")
        self.assertEqual(OPENAI_COMPAT_URLS["deepinfra"], "https://api.deepinfra.com/v1/openai")
        self.assertIn("deepinfra", list_providers())

    def test_prefixed_model_routes_to_deepinfra(self):
        self.assertEqual(infer_provider_from_model(f"deepinfra/{DEEPINFRA_MODEL}"), "deepinfra")
        self.assertEqual(infer_provider_from_model(DEEPINFRA_MODEL), "deepinfra")

    def test_bare_platform_alias_still_routes_to_deepseek(self):
        self.assertEqual(infer_provider_from_model("deepseek-flash"), "deepseek")

    def test_prefix_is_stripped_before_the_wire(self):
        self.assertEqual(get_model_name("deepinfra", f"deepinfra/{DEEPINFRA_MODEL}"), DEEPINFRA_MODEL)
        self.assertEqual(get_model_name("deepinfra"), DEEPINFRA_MODEL)

    def test_dispatch_plan_by_provider_and_by_model(self):
        by_provider = build_dispatch_plan(
            provider="deepinfra",
            model=None,
            reasoning_effort=None,
            timeout=300,
            lite=None,
            mode="chat",
            auth="api",
            subscription=False,
            api_only=None,
            use_old=False,
        )
        self.assertEqual(by_provider.provider, "deepinfra")
        self.assertEqual(by_provider.model, DEEPINFRA_MODEL)
        self.assertEqual(by_provider.transport, "deepinfra-api")

        by_model = build_dispatch_plan(
            provider=None,
            model=f"deepinfra/{DEEPINFRA_MODEL}",
            reasoning_effort="low",
            timeout=300,
            lite=None,
            mode="chat",
            auth="api",
            subscription=False,
            api_only=None,
            use_old=False,
        )
        self.assertEqual(by_model.provider, "deepinfra")
        self.assertEqual(by_model.model, DEEPINFRA_MODEL)
        self.assertEqual(by_model.effort_applied, "low")

    def test_deepseek_default_is_deepseek_flash(self):
        self.assertEqual(PROVIDER_CONFIGS["deepseek"]["model"], "deepseek-flash")
        self.assertEqual(get_model_name("deepseek"), "deepseek-flash")
        self.assertNotIn("deepseek-chat", _KNOWN_MODELS["deepseek"])

    def test_info_lists_new_provider_and_models(self):
        mirror = collect_routing_mirror()
        self.assertIn(DEEPINFRA_MODEL, mirror["known_models"]["deepinfra"])
        self.assertIn("deepseek-flash", mirror["known_models"]["deepseek"])


class TestDeepInfraReasoning(unittest.TestCase):
    def test_default_request_asks_for_high_effort(self):
        with _Capture() as cap:
            _chat()
        self.assertEqual(cap.body["model"], DEEPINFRA_MODEL)
        self.assertEqual(cap.body["reasoning_effort"], "high")
        self.assertEqual(cap.client_kwargs[0]["base_url"], "https://api.deepinfra.com/v1/openai")

    def test_cli_effort_overrides_the_default(self):
        with _Capture() as cap:
            _chat(effort="low")
        self.assertEqual(cap.body["reasoning_effort"], "low")

    def test_max_survives_to_the_wire(self):
        with _Capture() as cap:
            _chat(effort="max")
        self.assertEqual(cap.body["reasoning_effort"], "max")

    def test_reasoning_tokens_are_logged_from_usage(self):
        with _Capture() as cap:
            _chat(effort="high")
        self.assertEqual(len(cap.usage_rows), 1)
        row = cap.usage_rows[0]
        self.assertEqual(row["provider"], "deepinfra")
        self.assertEqual(row["model"], DEEPINFRA_MODEL)
        self.assertEqual(row["reasoning_effort"], "high")
        self.assertEqual(row["prompt_tokens"], 12)
        self.assertEqual(row["completion_tokens"], 50)
        self.assertEqual(row["reasoning_tokens"], 37)
        self.assertTrue(row["completion_includes_reasoning"])

    def test_normalize_usage_reads_completion_tokens_details(self):
        normalized = _normalize_usage("deepinfra", _usage(reasoning=99, cached=4))
        self.assertEqual(normalized["reasoning_tokens"], 99)
        self.assertEqual(normalized["cached_tokens"], 4)
        self.assertTrue(normalized["completion_includes_reasoning"])


class TestDeepSeekV41Pricing(unittest.TestCase):
    def test_pricing_rows_match_the_fetched_rates(self):
        # DeepInfra: $0.20/$0.60 list, 51% discount. DeepSeek: PEAK platform rate.
        self.assertEqual(PRICING[DEEPINFRA_MODEL], (0.098, 0.294))
        self.assertEqual(PRICING["deepseek-flash"], (0.30, 1.20))

    def test_uncached_estimate(self):
        self.assertAlmostEqual(
            est_cost(DEEPINFRA_MODEL, 1_000_000, 1_000_000), 0.392, places=9
        )
        self.assertAlmostEqual(
            est_cost("deepseek-flash", 1_000_000, 1_000_000), 1.50, places=9
        )

    def test_cached_input_uses_the_published_cache_rate(self):
        # DeepInfra: 3% of $0.098 = $0.00294/M. DeepSeek peak: 2% of $0.30 = $0.006/M.
        self.assertAlmostEqual(
            est_cost(DEEPINFRA_MODEL, 1_000_000, 0, cached_tokens=1_000_000), 0.00294, places=9
        )
        self.assertAlmostEqual(
            est_cost("deepseek-flash", 1_000_000, 0, cached_tokens=1_000_000), 0.006, places=9
        )

    def test_reasoning_tokens_bill_at_the_output_rate(self):
        row = {
            "provider": "deepinfra",
            "model": DEEPINFRA_MODEL,
            "prompt_tokens": 0,
            "completion_tokens": 1_000_000,
            "reasoning_tokens": 1_000_000,
            "completion_includes_reasoning": True,
        }
        self.assertAlmostEqual(cost_for_usage(row), 0.294, places=9)

    def test_spend_guard_accepts_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            ledger.write_text("")
            for model in (DEEPINFRA_MODEL, "deepseek-flash"):
                with self.subTest(model=model):
                    enforce_daily_cap(model, log_path=ledger)


if __name__ == "__main__":
    unittest.main()
