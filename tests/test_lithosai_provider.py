"""LithosAI transport + DeepSeek V4.1 Flash pricing (offline).

Pins the `lithosai` lane end to end at the request/usage boundary: routing by
`-p lithosai`, by the prefixed and the bare `deepseek-ai/…` model id, the tier
aliases, the outgoing `reasoning_effort` (default "high", CLI effort overrides
it), reasoning-token accounting, and spend-guard pricing for every DeepSeek
V4.1 Flash id. DeepInfra was retired 2026-10-26: its route fails loud, and its
pricing row stays so historical usage rows still price.

Every test patches the SDK client, so no request leaves the machine.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llmx.dispatch_plan import build_dispatch_plan, collect_routing_mirror
from llmx.model_ids import RetiredModelError
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

LITHOS_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
LITHOS_TIERS = {
    "deepseek-flash": LITHOS_MODEL,
    "deepseek-flash-fast": f"{LITHOS_MODEL}-fast",
    "deepseek-flash-ultra": f"{LITHOS_MODEL}-ultra",
    "deepseek-flash-ultrafast": f"{LITHOS_MODEL}-ultra",
}
DEEPINFRA_HISTORICAL_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def _usage(*, prompt=12, completion=50, reasoning=37, cached=0):
    """The usage shape LithosAI's OpenAI-compatible endpoint reports."""
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
        model=LITHOS_MODEL,
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


def _chat(provider="lithosai", model=None, effort=None, temperature=0.7):
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


class TestLithosRouting(unittest.TestCase):
    def test_provider_config_and_base_url(self):
        cfg = PROVIDER_CONFIGS["lithosai"]
        self.assertEqual(cfg["model"], LITHOS_MODEL)
        self.assertEqual(cfg["env_var"], "LITHOSAI_API_KEY")
        self.assertEqual(OPENAI_COMPAT_URLS["lithosai"], "https://api.lithosai.cloud/v1")
        self.assertIn("lithosai", list_providers())

    def test_prefixed_and_bare_vendor_ids_route_to_lithos(self):
        self.assertEqual(infer_provider_from_model(f"lithosai/{LITHOS_MODEL}"), "lithosai")
        self.assertEqual(infer_provider_from_model(LITHOS_MODEL), "lithosai")
        self.assertEqual(infer_provider_from_model(f"{LITHOS_MODEL}-ultra"), "lithosai")

    def test_bare_platform_alias_still_routes_to_deepseek(self):
        self.assertEqual(infer_provider_from_model("deepseek-flash"), "deepseek")

    def test_tier_aliases_and_prefix_resolve_before_the_wire(self):
        for alias, wire in LITHOS_TIERS.items():
            with self.subTest(alias=alias):
                self.assertEqual(get_model_name("lithosai", alias), wire)
        self.assertEqual(get_model_name("lithosai", f"lithosai/{LITHOS_MODEL}-fast"), f"{LITHOS_MODEL}-fast")
        self.assertEqual(get_model_name("lithosai"), LITHOS_MODEL)

    def test_deepinfra_is_retired_and_fails_loud(self):
        self.assertNotIn("deepinfra", PROVIDER_CONFIGS)
        self.assertNotIn("deepinfra", OPENAI_COMPAT_URLS)
        self.assertNotIn("deepinfra", _KNOWN_MODELS)
        with self.assertRaises(RetiredModelError):
            infer_provider_from_model(f"deepinfra/{DEEPINFRA_HISTORICAL_MODEL}")
        with self.assertRaisesRegex(ValueError, "lithosai"):
            get_model_name("deepinfra")

    def test_dispatch_plan_by_provider_and_by_model(self):
        by_provider = build_dispatch_plan(
            provider="lithosai",
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
        self.assertEqual(by_provider.provider, "lithosai")
        self.assertEqual(by_provider.model, LITHOS_MODEL)
        self.assertEqual(by_provider.transport, "lithosai-api")

        by_model = build_dispatch_plan(
            provider=None,
            model=f"lithosai/{LITHOS_MODEL}",
            reasoning_effort="low",
            timeout=300,
            lite=None,
            mode="chat",
            auth="api",
            subscription=False,
            api_only=None,
            use_old=False,
        )
        self.assertEqual(by_model.provider, "lithosai")
        self.assertEqual(by_model.model, LITHOS_MODEL)
        self.assertEqual(by_model.effort_applied, "low")

    def test_deepseek_default_is_deepseek_flash(self):
        self.assertEqual(PROVIDER_CONFIGS["deepseek"]["model"], "deepseek-flash")
        self.assertEqual(get_model_name("deepseek"), "deepseek-flash")
        self.assertNotIn("deepseek-chat", _KNOWN_MODELS["deepseek"])

    def test_info_lists_lithos_tiers(self):
        mirror = collect_routing_mirror()
        for wire in set(LITHOS_TIERS.values()):
            self.assertIn(wire, mirror["known_models"]["lithosai"])
        self.assertIn("deepseek-flash", mirror["known_models"]["deepseek"])
        self.assertNotIn("deepinfra", mirror["known_models"])


class TestLithosReasoning(unittest.TestCase):
    def test_default_request_asks_for_high_effort(self):
        with _Capture() as cap:
            _chat()
        self.assertEqual(cap.body["model"], LITHOS_MODEL)
        self.assertEqual(cap.body["reasoning_effort"], "high")
        self.assertEqual(cap.client_kwargs[0]["base_url"], "https://api.lithosai.cloud/v1")

    def test_every_tier_gets_the_high_default(self):
        for alias, wire in LITHOS_TIERS.items():
            with self.subTest(alias=alias), _Capture() as cap:
                _chat(model=alias)
            self.assertEqual(cap.body["model"], wire)
            self.assertEqual(cap.body["reasoning_effort"], "high")

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
        self.assertEqual(row["provider"], "lithosai")
        self.assertEqual(row["model"], LITHOS_MODEL)
        self.assertEqual(row["reasoning_effort"], "high")
        self.assertEqual(row["prompt_tokens"], 12)
        self.assertEqual(row["completion_tokens"], 50)
        self.assertEqual(row["reasoning_tokens"], 37)
        self.assertTrue(row["completion_includes_reasoning"])

    def test_normalize_usage_reads_completion_tokens_details(self):
        normalized = _normalize_usage("lithosai", _usage(reasoning=99, cached=4))
        self.assertEqual(normalized["reasoning_tokens"], 99)
        self.assertEqual(normalized["cached_tokens"], 4)
        self.assertTrue(normalized["completion_includes_reasoning"])


class TestDeepSeekV41Pricing(unittest.TestCase):
    def test_pricing_rows_match_the_fetched_rates(self):
        # LithosAI early-access rates; DeepSeek: PEAK platform rate; DeepInfra: history.
        self.assertEqual(PRICING[LITHOS_MODEL], (0.15, 0.60))
        self.assertEqual(PRICING[f"{LITHOS_MODEL}-fast"], (0.25, 1.00))
        self.assertEqual(PRICING[f"{LITHOS_MODEL}-ultra"], (0.35, 1.40))
        self.assertEqual(PRICING["deepseek-flash"], (0.30, 1.20))
        self.assertEqual(PRICING[DEEPINFRA_HISTORICAL_MODEL], (0.098, 0.294))

    def test_uncached_estimate(self):
        self.assertAlmostEqual(est_cost(LITHOS_MODEL, 1_000_000, 1_000_000), 0.75, places=9)
        self.assertAlmostEqual(
            est_cost("deepseek-flash", 1_000_000, 1_000_000), 1.50, places=9
        )

    def test_cached_input_uses_the_published_cache_rate(self):
        # LithosAI: 2% of $0.15 = $0.003/M. DeepSeek peak: 2% of $0.30 = $0.006/M.
        self.assertAlmostEqual(
            est_cost(LITHOS_MODEL, 1_000_000, 0, cached_tokens=1_000_000), 0.003, places=9
        )
        self.assertAlmostEqual(
            est_cost("deepseek-flash", 1_000_000, 0, cached_tokens=1_000_000), 0.006, places=9
        )

    def test_reasoning_tokens_bill_at_the_output_rate(self):
        row = {
            "provider": "lithosai",
            "model": f"{LITHOS_MODEL}-fast",
            "prompt_tokens": 0,
            "completion_tokens": 1_000_000,
            "reasoning_tokens": 1_000_000,
            "completion_includes_reasoning": True,
        }
        self.assertAlmostEqual(cost_for_usage(row), 1.00, places=9)

    def test_historical_deepinfra_rows_still_price(self):
        row = {
            "provider": "deepinfra",
            "model": DEEPINFRA_HISTORICAL_MODEL,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "cached_tokens": 0,
            "completion_includes_reasoning": True,
        }
        self.assertAlmostEqual(cost_for_usage(row), 0.392, places=9)

    def test_spend_guard_accepts_every_dispatchable_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            ledger.write_text("")
            for model in (*set(LITHOS_TIERS.values()), "deepseek-flash"):
                with self.subTest(model=model):
                    enforce_daily_cap(model, log_path=ledger)


if __name__ == "__main__":
    unittest.main()
