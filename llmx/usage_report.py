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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .model_ids import CURSOR_GROK_MODELS

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
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.0),
    # 2026-07-21 launch: 3.6 Flash supersedes 3.5 Flash at a LOWER output rate
    # ($7.50 vs $9.00) with ~17% fewer output tokens claimed.
    # Gemini 3.8 Flash (ai.google.dev/gemini-api/docs/pricing, 2026-09-05):
    # intro $0.75/$3.75 through 2026-12-31; standard $1.50/$7.50 from 2027-01-01.
    # 3.7 Flash is the same intro rate. Register both so the spend guard prices them.
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (1.25, 10.0),
    # GPT-6 Astra (developers.openai.com/api/docs/models/gpt-6-astra, 2026-09-05):
    # $10/$50 short context; 2x/1.5x above 272K input. Fast mode is 2x Standard.
    # Alias gpt-6 → astra. Subscription (codex-cli) remains $0 against the ChatGPT plan.
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6": (10.0, 50.0),
    # GPT-5.6 suite (developers.openai.com/api/docs/pricing, GA 2026-07-09;
    # price cut 2026-07-30: Luna -80% to $0.20/$1.20, Terra -20% to $2/$12, Sol unchanged —
    # openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6;
    # Sol cut 2026-08-21 to $4/$20 (from $5/$30), promotional "at least through 2026-11-21"
    # per developers.openai.com/api/docs/pricing.md — re-verify after that date)
    # Alias gpt-5.6 → sol. Pro mode bills at same model rates (more tokens).
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6": (4.0, 20.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.3-chat-latest": (1.75, 14.0),
    "gpt-5.3-codex": (1.25, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),  # cyber fallback + legacy pin
    "claude-fable-5": (10.0, 50.0),
    # Fable 5.1 (2026-09-01): same $10/$50; cache read $0.25 vs $1.00 on Fable 5
    # (platform.claude.com/docs/en/models/fable-5-1/overview).
    "claude-fable-5-1": (10.0, 50.0),
    # Opus 5.5 (2026-09-22): $4/$20; cache read $0.20, 5m write $5, 1h write $8
    # (platform.claude.com/docs/en/models/opus-5-5/overview).
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # SpaceXAI Grok 4.5 API (docs.x.ai 2026-07-08): base $2/$6.
    "grok-4.5": (2.0, 6.0),
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
}
# Shadow prices for the Cursor subscription lanes ($0 billed); 4.6 mirrors 4.5
# until xAI publishes its own list price.
for cursor_model in CURSOR_GROK_MODELS:
    PRICING[cursor_model] = (4.0, 18.0) if cursor_model.endswith("-fast") else (2.0, 6.0)

# Context-window limit (max input tokens) per model. Static capability, not from the
# log — surfaced so `llmx usage --by model` shows headroom vs the biggest call sent.
CONTEXT_WINDOW: dict[str, int] = {
    "gemini-3-flash-preview": 1_000_000,
    "gemini-3-flash": 1_000_000,
    "gemini-3.1-flash-lite-preview": 1_000_000,
    "gemini-3.1-flash-lite": 1_000_000,
    "gemini-3.5-flash": 1_000_000,
    "gemini-3.8-flash": 1_000_000,
    "gemini-3.7-flash": 1_000_000,
    "gemini-3.6-flash": 1_000_000,
    "gemini-3.5-flash-lite": 1_000_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gpt-6-astra": 1_050_000,
    "gpt-6": 1_050_000,
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-5.3-chat-latest": 400_000,
    "gpt-5.3-codex": 400_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-fable-5-1": 1_000_000,
    "claude-opus-5-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    # dense-student screen candidates (2026-07-10): gemma4 256K, qwen3-32b 32K native
    # (131K YaRN — native registered), mistral-small-3.2 128K
    "google/gemma-4-31b-it": 256_000,
    "qwen/qwen3-32b": 32_768,
    "mistralai/mistral-small-3.2-24b-instruct": 128_000,
    # docs.x.ai Chat API Pricing table (2026-07-09): grok-4.5 context 500k
    "grok-4.5": 500_000,
    # Kimi K3 (2026-07-16): 1M-token context window
    "kimi-k3": 1_048_576,
    # Paid-tier public endpoint windows; free trial is 64/65k.
    "qwen-3.8-27b": 131_072,
    "gpt-oss-120b": 131_072,
}
for cursor_model in CURSOR_GROK_MODELS:
    CONTEXT_WINDOW[cursor_model] = 500_000


def est_cost(
    model: str,
    prompt: int,
    out: int,
    *,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    conservative: bool = False,
) -> float | None:
    """Price total input and billable output; cache counts partition total input.

    Three-argument callers retain an uncached-input estimate. For Astra, the
    conservative guard prices unknown input categories at the cache-write rate.
    Explicit zero cache counts mean known absence; None means unreported.
    """
    rate = PRICING.get(model)
    if rate is None:
        return None
    input_rate, output_rate = rate
    if model not in {"gpt-6-astra", "gpt-6"}:
        return (prompt * input_rate + out * output_rate) / 1_000_000

    # https://developers.openai.com/api/docs/models/gpt-6-astra
    read_rate, write_rate = 1.0, 12.50
    if prompt > 272_000:
        input_rate *= 2
        read_rate *= 2
        write_rate *= 2
        output_rate *= 1.5
    read = max(0, min(cached_tokens or 0, prompt))
    written = max(0, min(cache_write_tokens or 0, prompt - read))
    ordinary = prompt - read - written
    ordinary_rate = write_rate if conservative and cache_write_tokens is None else input_rate
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


def cost_for_usage(row: dict, *, conservative: bool = False) -> float | None:
    """Price reported totals; absent input/output totals leave cost unknown."""
    if row.get("prompt_tokens") is None or row.get("completion_tokens") is None:
        return None
    return est_cost(
        row.get("model") or "",
        row["prompt_tokens"],
        output_tokens(row),
        cached_tokens=row.get("cached_tokens"),
        cache_write_tokens=row.get("cache_write_tokens"),
        conservative=conservative,
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
        "\n  (cost = conservative estimate from PRICING; unknown Astra input is charged "
        "at the cache-write rate. '+?' = unpriced model or unreported usage. "
        "Token sums include reported counts. "
        "max_in = biggest single call's input; %used vs ctx_win.)"
    )
    return "\n".join(out_lines)
