"""Cost/usage rollups over the llmx usage log (~/.claude/llmx-usage.jsonl).

The log (usage_log.py) records exact tokens per call and stays pricing-free on
purpose — pricing changes, tokens are durable. Pricing + the rollup live HERE so
there is ONE place to edit rates, surfaced as `llmx usage` (and the thin
scripts/usage_summary.py wrapper). Cost is an ESTIMATE; tokens are exact.
"""

from __future__ import annotations

import collections
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .model_ids import CURSOR_GROK46_RETIRED_MODELS, CURSOR_GROK_MODELS

DEFAULT_LOG = Path(
    os.environ.get("LLMX_USAGE_LOG", str(Path.home() / ".claude" / "llmx-usage.jsonl"))
)

# Per-MTok (input, output). Output rate also applies to reasoning tokens. Approximate —
# verify before quoting. Edit as pricing changes (this is the single source).
PRICING: dict[str, tuple[float, float]] = {
    # Gemini rates re-verified against ai.google.dev/gemini-api/docs/pricing 2026-07-22
    # (paid tier, standard, text). The 3-flash and 3.1-flash-lite entries below were
    # UNDERSTATED 6.7-10x and 5-7.5x respectively — a cost estimator that lowballs the
    # provider behind June's ~EUR700 Gemini bill is the wrong direction to be wrong in.
    # Audio input is priced higher for the 3.x flash tiers; text rate registered.
    "gemini-3-flash-preview": (0.50, 3.00),  # retired 2026-09-25: accounting only
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.0),  # retired 2026-09-25: accounting only
    # 2026-07-21 launch: 3.6 Flash supersedes 3.5 Flash at a LOWER output rate
    # ($7.50 vs $9.00) with ~17% fewer output tokens claimed.
    # Gemini 3.8 Flash (ai.google.dev/gemini-api/docs/pricing, 2026-09-05):
    # intro $0.75/$3.75 through 2026-12-31; standard $1.50/$7.50 from 2027-01-01.
    # 3.7 Flash is the same intro rate. Register both so the spend guard prices them.
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),  # retired 2026-09-25: accounting only
    "gemini-3.6-flash": (1.50, 7.50),  # retired 2026-09-25: accounting only
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (1.25, 10.0),
    # Gemini 4 Argon (ai.google.dev/gemini-api/docs/pricing, 2026-10-14): intro $2/$10
    # through 2026-11-30; standard $4/$20 from 2026-12-01 (PRICE_CHANGES below).
    # No prompt-length tiers; not on the free tier.
    "gemini-4-argon": (2.0, 10.0),
    # Gemini 4 Carbon (aistudio.google.com/docs/pricing, 2026-11-11): same schedule as
    # Argon, intro $2/$10 through 2026-11-30, standard $4/$20 from 2026-12-01
    # (PRICE_CHANGES below).
    "gemini-4-carbon": (2.0, 10.0),
    # GPT-6 Astra (developers.openai.com/api/docs/models/gpt-6-astra, 2026-09-05):
    # $10/$50 short context; 2x/1.5x above 272K input. Fast mode is 2x Standard.
    # Alias gpt-6 → astra. Subscription (codex-cli) remains $0 against the ChatGPT plan.
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6": (10.0, 50.0),
    # GPT-6 Sol / Luna (developers.openai.com/api/docs/models/gpt-6-sol, gpt-6-luna,
    # 2026-09-22): same cache and >272K structure as Astra.
    "gpt-6-sol": (2.0, 10.0),  # alias → gpt-6.1-sol since 2026-10-09: accounting only
    # GPT-6.1 Sol (developers.openai.com/api/docs/models/gpt-6.1-sol, read 2026-10-09):
    # $2/$10, cached $0.10, same cache-write and >272K structure.
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6-luna": (0.10, 0.50),
    # GPT-5.6 suite (developers.openai.com/api/docs/pricing, GA 2026-07-09;
    # price cut 2026-07-30: Luna -80% to $0.20/$1.20, Terra -20% to $2/$12, Sol unchanged —
    # openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6;
    # Sol cut 2026-08-21 to $4/$20 (from $5/$30), promotional "at least through 2026-11-21"
    # per developers.openai.com/api/docs/pricing.md — re-verify after that date)
    # Alias gpt-5.6 → sol. Pro mode bills at same model rates (more tokens).
    # retired 2026-09-25: accounting only (gpt-5.6 suite)
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6": (4.0, 20.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.3-chat-latest": (1.75, 14.0),  # retired 2026-09-25: accounting only
    "gpt-5.3-codex": (1.25, 10.0),  # retired 2026-09-25: accounting only
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),  # retired 2026-09-25: accounting only
    "claude-fable-5": (10.0, 50.0),  # retired 2026-09-25: accounting only
    # Fable 5.1 (2026-09-01): same $10/$50; cache read $0.25 vs $1.00 on Fable 5
    # (platform.claude.com/docs/en/models/fable-5-1/overview).
    "claude-fable-5-1": (10.0, 50.0),  # retired 2026-11-10: accounting only
    # Opus 5.5 (2026-09-22): $4/$20; cache read $0.20, 5m write $5, 1h write $8
    # (platform.claude.com/docs/en/models/opus-5-5/overview).
    "claude-opus-5-5": (4.0, 20.0),
    # Fable 5.2 (2026-11-09): $8/$40; cache read $0.20 (2.5%), 5m write $10, 1h write $16
    # (platform.claude.com/docs/en/models/fable-5-2/overview, read 2026-11-10).
    "claude-fable-5-2": (8.0, 40.0),
    # Sonnet 5.5: $2/$10, cache read $0.10 (5%). Haiku 5.5: $0.10/$0.50 for prompts up to
    # 100K tokens, $0.50/$2.50 above (est_cost applies the tier); pricing page 2026-11-10.
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5": (2.0, 10.0),  # retired 2026-11-10: accounting only
    "claude-sonnet-4-6": (3.0, 15.0),
    # Haiku 4.5: $1/$5. claude-cli logs the dated id.
    # claude-cli still logs Haiku 4.5 as a helper model in modelUsage; keep both rows.
    "claude-haiku-4-5": (1.0, 5.0),  # retired 2026-11-10 from routing
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # SpaceXAI Grok 4.5 API (docs.x.ai 2026-07-08): base $2/$6.
    "grok-4.5": (2.0, 6.0),  # retired 2026-09-25: accounting only
    # Grok 4.7 launch page (x.ai/news/grok-4-7, 2026-09-21): from $2/$6.
    # Fast tier is 2×. Long-context cliff not restated on that page.
    "grok-4.7": (2.0, 6.0),
    "grok-4.6": (2.0, 6.0),  # retired 2026-09-25: accounting only
    # Kimi K3 (kimi.com research announcement 2026-07-16): $3.00/MTok cache-miss
    # input, $15.00/MTok output; cache-hit input $0.30/MTok (>90% hit rate claimed
    # in coding workloads — priced here at the conservative cache-miss rate).
    "kimi-k3": (3.0, 15.0),
    # openrouter (verified live 2026-07-07: /api/v1/models pricing.prompt/completion)
    "qwen/qwen3.6-27b": (0.285, 2.40),
    # Qwen3.8 family (released 2026-08-14; verified live 2026-08-17 via
    # /api/v1/models pricing.prompt/completion; 27b endpoints: Chutes fp8 /
    # Io Net fp8 / AkashML bf16)
    "qwen/qwen3.8-27b": (0.45, 3.20),
    "qwen/qwen3.8-2.4t-a95b": (2.0, 6.0),
    "qwen/qwen3.8-max": (2.0, 6.0),
    # Cerebras public shared endpoints (inference-docs.cerebras.ai model cards,
    # verified 2026-09-03). Distinct IDs from the OpenRouter Qwen entries above.
    "qwen-3.8-27b": (0.99, 1.49),
    "gpt-oss-120b": (0.35, 0.75),
    # dense-student screen candidates (arc-agi research/2026-07-10-dense-student-candidates.md,
    # web-verified 2026-07-10; per-M USD prompt/completion)
    "google/gemma-4-31b-it": (0.12, 0.35),
    "qwen/qwen3-32b": (0.08, 0.28),
    "mistralai/mistral-small-3.2-24b-instruct": (0.075, 0.20),
    # DeepSeek V4 Flash (released 2026-07-31; verified live 2026-08-01 via
    # openrouter /api/v1/models pricing.prompt/completion — both ids same price)
    "deepseek/deepseek-v4-flash-0731": (0.14, 0.28),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    # DeepInfra DeepSeek V4.1 Flash (api.deepinfra.com/models/deepseek-ai/
    # DeepSeek-V4.1-Flash-1017, fetched 2026-10-24): list $0.20 in / $0.60 out per M
    # with a 51% discount applied → $0.098 / $0.294. Cached input is 3% of the
    # input rate ($0.00294/M); reasoning bills at the output rate. The DeepInfra
    # route was retired 2026-10-26: this row prices historical usage only.
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (0.098, 0.294),  # retired: accounting only
    # LithosAI tiers (lithosai.com/pricing, read 2026-10-24). Every tier serves the
    # same V4.1 Flash 1017 weights; only speed and price change. The listed prices
    # are an "early access discount" — each is half its struck-through list rate
    # (base was $1.20 out, fast $2.00, ultra $2.80), so re-read this table when the
    # promo ends. Cached input is 2% of the input rate on every tier; reasoning
    # bills at the output rate. ultra-chat prices like ultra despite measuring a
    # worse TTFT (10.6 s vs 1.3 s).
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.15, 0.60),
    "deepseek-ai/DeepSeek-V4.1-Flash-fast": (0.25, 1.00),
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra": (0.35, 1.40),
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra-chat": (0.35, 1.40),
    # DeepSeek platform (api-docs.deepseek.com/quick_start/pricing, read 2026-10-24):
    # deepseek-flash is served by DeepSeek-V4.1-Flash-1017. Peak hours bill 2x the
    # off-peak rate, and PRICING feeds the spend-guard CAP, so register the PEAK
    # rate: $0.30 cache-miss input / $1.20 output per M (off-peak $0.15 / $0.60);
    # cache-hit input $0.006 peak ($0.003 off-peak).
    "deepseek-flash": (0.30, 1.20),
}
# Shadow prices for the Cursor subscription lanes ($0 billed).
# 4.7 fast is 2× list ($4/$12) per the 2026-09-21 launch page.
# 4.6 fast stays at the prior $4/$18 shadow until that lane is re-priced.
# retired 2026-09-25: accounting only — CURSOR_GROK46_RETIRED_MODELS priced for history.
for cursor_model in (*CURSOR_GROK_MODELS, *CURSOR_GROK46_RETIRED_MODELS):
    if "grok-4.7" in cursor_model and cursor_model.endswith("-fast"):
        PRICING[cursor_model] = (4.0, 12.0)
    elif cursor_model.endswith("-fast"):
        PRICING[cursor_model] = (4.0, 18.0)
    else:
        PRICING[cursor_model] = (2.0, 6.0)

# GPT Image rates per MTok: (text input, image input, text output, image output).
# developers.openai.com/api/docs/pricing#image-generation (Standard table, read
# 2026-11-05). None = the page lists no text output rate for that model. The dated
# gpt-image-2 id is that model's default snapshot per
# developers.openai.com/api/docs/models/gpt-image-2 (same read date); the pricing page
# has no row of its own for it.
# Cached rates are left out on purpose: llmx calls the Images API, where 2 and 2.5 get
# no cached rate (developers.openai.com/api/docs/guides/image-generation#cached-input-pricing)
# and whose usage reports no cached count, so these rows price as uncached.
_GPT_IMAGE_2_RATES = (5.0, 8.0, None, 30.0)
_GPT_IMAGE_1_5_RATES = (5.0, 8.0, 10.0, 32.0)
_OPENAI_IMAGE_RATES: dict[str, tuple[float, float, float | None, float]] = {
    "gpt-image-2.6-sunburst": _GPT_IMAGE_2_RATES,
    "gpt-image-2.6-flare": _GPT_IMAGE_2_RATES,
    "gpt-image-2.5-sunburst": _GPT_IMAGE_2_RATES,
    "gpt-image-2.5-flare": _GPT_IMAGE_2_RATES,
    "gpt-image-2": _GPT_IMAGE_2_RATES,
    "gpt-image-2-2026-04-21": _GPT_IMAGE_2_RATES,
    "gpt-image-1.5": _GPT_IMAGE_1_5_RATES,
    "chatgpt-image-latest": _GPT_IMAGE_1_5_RATES,
    "gpt-image-1-mini": (2.0, 2.5, None, 8.0),
}
# PRICING holds the image input and output rates, the upper bound for a row that does
# not split its tokens by modality. est_cost uses the split when the row has it.
for image_model, (_, image_input_rate, _, image_output_rate) in _OPENAI_IMAGE_RATES.items():
    PRICING[image_model] = (image_input_rate, image_output_rate)

# Context-window limit (max input tokens) per model. Static capability, not from the
# log — surfaced so `llmx usage --by model` shows headroom vs the biggest call sent.
CONTEXT_WINDOW: dict[str, int] = {
    "gemini-3-flash-preview": 1_000_000,  # retired 2026-09-25: accounting only
    "gemini-3-flash": 1_000_000,
    "gemini-3.1-flash-lite-preview": 1_000_000,
    "gemini-3.1-flash-lite": 1_000_000,
    "gemini-3.5-flash": 1_000_000,  # retired 2026-09-25: accounting only
    "gemini-3.8-flash": 1_000_000,
    "gemini-3.7-flash": 1_000_000,  # retired 2026-09-25: accounting only
    "gemini-3.6-flash": 1_000_000,  # retired 2026-09-25: accounting only
    "gemini-3.5-flash-lite": 1_000_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-4-argon": 1_048_576,
    "gemini-4-carbon": 1_048_576,
    "gpt-6-astra": 1_050_000,
    "gpt-6": 1_050_000,
    "gpt-6-sol": 1_050_000,
    "gpt-6.1-sol": 1_050_000,
    "gpt-6-luna": 1_050_000,
    # retired 2026-09-25: accounting only (gpt-5.6 suite)
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-5.3-chat-latest": 400_000,  # retired 2026-09-25: accounting only
    "gpt-5.3-codex": 400_000,  # retired 2026-09-25: accounting only
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,  # retired 2026-09-25: accounting only
    "claude-fable-5": 1_000_000,  # retired 2026-09-25: accounting only
    "claude-fable-5-1": 1_000_000,  # retired 2026-11-10: accounting only
    "claude-opus-5-5": 1_000_000,
    "claude-fable-5-2": 1_000_000,
    "claude-sonnet-5-5": 1_000_000,
    "claude-haiku-5-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,  # retired 2026-11-10: accounting only
    "claude-sonnet-4-6": 1_000_000,
    # dense-student screen candidates (2026-07-10): gemma4 256K, qwen3-32b 32K native
    # (131K YaRN — native registered), mistral-small-3.2 128K
    "google/gemma-4-31b-it": 256_000,
    "qwen/qwen3-32b": 32_768,
    "mistralai/mistral-small-3.2-24b-instruct": 128_000,
    # docs.x.ai Chat API Pricing table (2026-07-09): grok-4.5 context 500k.
    # 4.6/4.7 carry the same window; the 4.7 news post did not restate it.
    "grok-4.5": 500_000,  # retired 2026-09-25: accounting only
    "grok-4.6": 500_000,  # retired 2026-09-25: accounting only
    "grok-4.7": 500_000,
    # Kimi K3 (2026-07-16): 1M-token context window
    "kimi-k3": 1_048_576,
    # Paid-tier public endpoint windows; free trial is 64/65k.
    "qwen-3.8-27b": 131_072,
    "gpt-oss-120b": 131_072,
    # DeepSeek V4.1 Flash: 1M context (retired DeepInfra route's model JSON
    # max_tokens=1,048,576; api-docs.deepseek.com lists CONTEXT LENGTH 1M).
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": 1_048_576,
    # LithosAI serves the same weights with the same 1M window (docs limits page).
    "deepseek-ai/DeepSeek-V4.1-Flash": 1_048_576,
    "deepseek-ai/DeepSeek-V4.1-Flash-fast": 1_048_576,
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra": 1_048_576,
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra-chat": 1_048_576,
    "deepseek-flash": 1_000_000,
}
for cursor_model in (*CURSOR_GROK_MODELS, *CURSOR_GROK46_RETIRED_MODELS):
    CONTEXT_WINDOW[cursor_model] = 500_000

# Claude cache-read rate as a fraction of the input rate where it is not the usual 10%
# (sources in the PRICING comments).
_CLAUDE_CACHE_READ = {
    "claude-opus-5-5": 0.05,
    "claude-sonnet-5-5": 0.05,
    "claude-fable-5-2": 0.025,
    "claude-fable-5-1": 0.025,
}
# Haiku 5.5 bills every token category at 5x when the prompt exceeds 100K tokens.
_CLAUDE_LONG_PROMPT_TIER = {"claude-haiku-5-5": (100_000, 5.0)}

# GPT-6 cache-read rate as a fraction of the input rate where it is not the usual 10%.
# developers.openai.com/api/docs/models/gpt-6.1-sol (read 2026-11-01): $2 input,
# $0.10 cached (5%). Astra ($10/$1) and Luna ($0.10/$0.01) stay at 10%. gpt-6-sol
# aliases to 6.1 Sol, so it prices the same.
_OPENAI_CACHE_READ = {"gpt-6.1-sol": 0.05, "gpt-6-sol": 0.05}

# DeepSeek cache-hit rate as a fraction of the input rate. The retired DeepInfra
# route stated rate_per_input_token_cached=0.03 (3% of input); the DeepSeek platform's
# $0.006 peak cache-hit vs $0.30 peak input is the same 2% at either time of day.
# Cache writes are not billed separately by either vendor, so they price as
# ordinary input.
_DEEPSEEK_CACHE_READ = {
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": 0.03,
    # LithosAI bills cached input at $0.003/$0.005/$0.007 against $0.15/$0.25/$0.35
    # input — 2% on every tier.
    "deepseek-ai/DeepSeek-V4.1-Flash": 0.02,
    "deepseek-ai/DeepSeek-V4.1-Flash-fast": 0.02,
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra": 0.02,
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra-chat": 0.02,
    "deepseek-flash": 0.02,
}


# Announced price changes: model -> [(first UTC date the new rate applies, (input, output))],
# oldest first. PRICING holds the rate in force before the first change. rate_for() picks
# by date, so the spend cap prices a ledger row at the rate that applied when it ran and
# nobody has to remember to edit PRICING on the day.
PRICE_CHANGES: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "gemini-4-argon": [("2026-12-01", (4.0, 20.0))],
    "gemini-4-carbon": [("2026-12-01", (4.0, 20.0))],
    "gemini-3.8-flash": [("2027-01-01", (1.50, 7.50))],
}


def rate_for(model: str, on: str | None = None) -> tuple[float, float] | None:
    """Per-MTok (input, output) for ``model`` on UTC date ``on``; today when None.

    ``on`` is an ISO date or timestamp (a ledger row's ``ts``); only its date is read.
    """
    rate = PRICING.get(model)
    day = (on or datetime.now(timezone.utc).date().isoformat())[:10]
    for start, new_rate in PRICE_CHANGES.get(model, ()):
        if day >= start:
            rate = new_rate
    return rate


def est_cost(
    model: str,
    prompt: int,
    out: int,
    *,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    conservative: bool = False,
    image_input_tokens: int | None = None,
    text_output_tokens: int | None = None,
    on: str | None = None,
) -> float | None:
    """Price total input and billable output; cache counts partition total input.

    ``on`` is the date the call ran (a ledger ``ts``); it selects the rate when a
    model has a dated price change. None prices at today's rate.

    Three-argument callers retain an uncached-input estimate. For Astra, the
    conservative guard prices unknown input categories at the cache-write rate.
    Explicit zero cache counts mean known absence; None means unreported.
    GPT Image tokens with no modality split price at the image rates, the higher ones.
    """
    rate = rate_for(model, on)
    if rate is None:
        return None
    input_rate, output_rate = rate
    if model in _OPENAI_IMAGE_RATES:
        text_in, image_in, text_out, image_out = _OPENAI_IMAGE_RATES[model]
        image_input = prompt if image_input_tokens is None else max(0, min(image_input_tokens, prompt))
        text_output = 0 if text_out is None else max(0, min(text_output_tokens or 0, out))
        return (
            (prompt - image_input) * text_in
            + image_input * image_in
            + text_output * (text_out or 0.0)
            + (out - text_output) * image_out
        ) / 1_000_000
    if model in {"gpt-6-astra", "gpt-6", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-luna"}:
        # developers.openai.com/api/docs/models/gpt-6-{astra,luna}: cached reads 10%
        # (6.1 Sol 5%, _OPENAI_CACHE_READ), cache writes 1.25x of the uncached input rate.
        read_rate = input_rate * _OPENAI_CACHE_READ.get(model, 0.10)
        write_rate = input_rate * 1.25
        if prompt > 272_000:
            input_rate *= 2
            read_rate *= 2
            write_rate *= 2
            output_rate *= 1.5
        ordinary_rate = write_rate if conservative and cache_write_tokens is None else input_rate
    elif model.startswith("claude-"):
        # Anthropic prompt caching: reads 10% of input (Opus/Sonnet 5.5 5%, Fable 2.5%; see
        # PRICING), writes 1.25x for 5-minute entries and 2x for 1-hour ones. claude-cli
        # writes 1-hour entries and llmx's API path sets no cache_control, so writes are 2x.
        threshold, multiplier = _CLAUDE_LONG_PROMPT_TIER.get(model, (None, 1.0))
        if threshold is not None and prompt > threshold:
            input_rate *= multiplier
            output_rate *= multiplier
        read_rate = input_rate * _CLAUDE_CACHE_READ.get(model, 0.10)
        write_rate, ordinary_rate = input_rate * 2, input_rate
    elif model in _DEEPSEEK_CACHE_READ:
        # DeepSeek (LithosAI, the platform, historical DeepInfra rows): cache hits are a small fraction of
        # input; no separate cache-write rate is published.
        read_rate = input_rate * _DEEPSEEK_CACHE_READ[model]
        write_rate, ordinary_rate = input_rate, input_rate
    else:
        return (prompt * input_rate + out * output_rate) / 1_000_000
    read = max(0, min(cached_tokens or 0, prompt))
    written = max(0, min(cache_write_tokens or 0, prompt - read))
    ordinary = prompt - read - written
    return (
        ordinary * ordinary_rate + read * read_rate + written * write_rate + out * output_rate
    ) / 1_000_000


def output_tokens(row: dict) -> int:
    """Return billable output, without adding OpenAI's reasoning subset twice.

    New SDK rows declare the relationship explicitly. Historical native OpenAI
    rows use the SDK contract; other historical rows retain their prior semantics.
    """
    completion = row.get("completion_tokens") or 0
    included = row.get("completion_includes_reasoning")
    if included is None:
        included = row.get("provider") == "openai" or (row.get("model") or "").startswith("gpt-")
    return completion if included else completion + (row.get("reasoning_tokens") or 0)


# Errors carry the HTTP status: the OpenAI SDK as "RateLimitError: Error code: 429 - ...", native
# Claude as "Anthropic HTTP 400", google-genai as "ClientError: 429 RESOURCE_EXHAUSTED. ...". A 4xx
# means the provider refused the request before generating, so nothing was billed.
_CLIENT_REJECTION = re.compile(r"\b(?:Error code:|Anthropic HTTP|ClientError:) 4\d\d\b")


def cost_for_usage(row: dict, *, conservative: bool = False) -> float | None:
    """Price reported totals; absent input/output totals leave cost unknown.

    A cost the CLI reports itself (claude-cli, grok-cli) wins over the estimate.
    A request refused with a 4xx and no reported usage costs $0, not unknown: an
    exhausted-credit 429 otherwise left the doctor's daily total permanently
    [DEGRADED] (2026-10-07).
    """
    if row.get("reported_cost_usd") is not None:
        return float(row["reported_cost_usd"])
    if row.get("prompt_tokens") is None or row.get("completion_tokens") is None:
        if _CLIENT_REJECTION.search(row.get("error") or ""):
            return 0.0
        return None
    return est_cost(
        row.get("model") or "",
        row["prompt_tokens"],
        output_tokens(row),
        cached_tokens=row.get("cached_tokens"),
        cache_write_tokens=row.get("cache_write_tokens"),
        conservative=conservative,
        image_input_tokens=row.get("image_input_tokens"),
        text_output_tokens=row.get("text_output_tokens"),
        on=row.get("ts") or None,
    )


def summarize(
    by: str = "caller",
    days: int = 30,
    since: str | None = None,
    model: str | None = None,
    log: str | Path | None = None,
) -> str:
    """Return a formatted usage rollup. by ∈ {caller,cwd,model,provider}."""
    floor = (
        since[:10]
        if since
        else (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    )
    log_path = Path(log) if log else DEFAULT_LOG
    if not log_path.exists():
        return f"✗ no usage log at {log_path}"

    groups: dict[str, dict] = collections.defaultdict(
        lambda: {
            "calls": 0,
            "prompt": 0,
            "out": 0,
            "max_in": 0,
            "cost": 0.0,
            "cost_known": True,
            "errors": 0,
        }
    )
    total = {"calls": 0, "prompt": 0, "out": 0, "cost": 0.0}
    n = 0
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("ts") or "")[:10] < floor:
            continue
        if model and r.get("model") != model:
            continue
        n += 1
        key = r.get(by)
        if key is None:
            key = "(unattributed)" if by in ("caller", "cwd") else "?"
        if by == "cwd" and isinstance(key, str) and key != "(unattributed)":
            key = os.path.basename(key.rstrip("/")) or key
        prompt = r.get("prompt_tokens") or 0
        out = output_tokens(r)
        g = groups[key]
        g["calls"] += 1
        g["prompt"] += prompt
        g["out"] += out
        g["max_in"] = max(g["max_in"], prompt)
        if r.get("error"):
            g["errors"] += 1
        c = cost_for_usage(r, conservative=True)
        if c is None:
            g["cost_known"] = False
        else:
            g["cost"] += c
            total["cost"] += c
        total["calls"] += 1
        total["prompt"] += prompt
        total["out"] += out

    if not groups:
        return f"No records since {floor}" + (f" for model {model}" if model else "")

    rows = sorted(groups.items(), key=lambda kv: (kv[1]["cost"], kv[1]["calls"]), reverse=True)
    w = min(40, max(len(k) for k, _ in rows))
    show_ctx = by == "model"
    out_lines = [
        f"\nllmx usage by {by} — since {floor}"
        + (f" — model={model}" if model else "")
        + f"  ({n} calls)\n",
        f"  {'(' + by + ')':<{w}}  {'calls':>6}  {'in_tok':>11}  {'out_tok':>11}  {'max_in':>9}  {'est_cost':>9}"
        + (f"  {'ctx_win':>9}  {'%used':>6}" if show_ctx else ""),
        f"  {'-' * w}  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 9}  {'-' * 9}"
        + (f"  {'-' * 9}  {'-' * 6}" if show_ctx else ""),
    ]
    for key, g in rows:
        cost = f"${g['cost']:.2f}" + ("" if g["cost_known"] else "+?")
        err = f"  ({g['errors']} err)" if g["errors"] else ""
        line = (
            f"  {key[:w]:<{w}}  {g['calls']:>6}  {g['prompt']:>11,}  {g['out']:>11,}  "
            f"{g['max_in']:>9,}  {cost:>9}"
        )
        if show_ctx:
            cw = CONTEXT_WINDOW.get(key)
            pct = f"{100 * g['max_in'] / cw:.0f}%" if cw else "?"
            line += f"  {(format(cw, ',') if cw else '?'):>9}  {pct:>6}"
        out_lines.append(line + err)
    out_lines.append(
        f"  {'-' * w}  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 9}  {'-' * 9}"
        + (f"  {'-' * 9}  {'-' * 6}" if show_ctx else "")
    )
    total_cost = f"${total['cost']:.2f}" + (
        "" if all(g["cost_known"] for g in groups.values()) else "+?"
    )
    out_lines.append(
        f"  {'TOTAL':<{w}}  {total['calls']:>6}  {total['prompt']:>11,}  {total['out']:>11,}  "
        f"{'':>9}  {total_cost:>9}"
    )
    out_lines.append(
        "\n  (cost = the CLI's own figure when it reports one, else a conservative "
        "estimate from PRICING; unknown Astra input is charged "
        "at the cache-write rate. '+?' = unpriced model or unreported usage. "
        "Token sums include reported counts. "
        "max_in = biggest single call's input; %used vs ctx_win.)"
    )
    return "\n".join(out_lines)
