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

DEFAULT_LOG = Path(os.environ.get("LLMX_USAGE_LOG", str(Path.home() / ".claude" / "llmx-usage.jsonl")))

# Per-MTok (input, output). Output rate also applies to reasoning tokens. Approximate —
# verify before quoting. Edit as pricing changes (this is the single source).
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3-flash-preview": (0.075, 0.30),
    "gemini-3-flash": (0.075, 0.30),
    "gemini-3.1-flash-lite-preview": (0.05, 0.20),
    "gemini-3.5-flash": (1.50, 9.0),
    "gemini-3.1-pro-preview": (1.25, 10.0),
    "gpt-5.5": (1.25, 10.0),
    "gpt-5.5-pro": (15.0, 120.0),
    "gpt-5.3-chat-latest": (1.75, 14.0),
    "gpt-5.3-codex": (1.25, 10.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}

# Context-window limit (max input tokens) per model. Static capability, not from the
# log — surfaced so `llmx usage --by model` shows headroom vs the biggest call sent.
CONTEXT_WINDOW: dict[str, int] = {
    "gemini-3-flash-preview": 1_000_000, "gemini-3-flash": 1_000_000,
    "gemini-3.1-flash-lite-preview": 1_000_000, "gemini-3.5-flash": 1_000_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gpt-5.5": 1_050_000, "gpt-5.5-pro": 1_050_000,
    "gpt-5.3-chat-latest": 400_000, "gpt-5.3-codex": 400_000,
    "claude-opus-4-8": 1_000_000, "claude-fable-5": 1_000_000, "claude-sonnet-4-6": 1_000_000,
}


def est_cost(model: str, prompt: int, out: int) -> float | None:
    rate = PRICING.get(model)
    if rate is None:
        return None
    return (prompt / 1_000_000) * rate[0] + (out / 1_000_000) * rate[1]


def summarize(by: str = "caller", days: int = 30, since: str | None = None,
              model: str | None = None, log: str | Path | None = None) -> str:
    """Return a formatted usage rollup. by ∈ {caller,cwd,model,provider}."""
    floor = since[:10] if since else (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    log_path = Path(log) if log else DEFAULT_LOG
    if not log_path.exists():
        return f"✗ no usage log at {log_path}"

    groups: dict[str, dict] = collections.defaultdict(
        lambda: {"calls": 0, "prompt": 0, "out": 0, "max_in": 0, "cost": 0.0, "cost_known": True, "errors": 0})
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
        m = r.get("model") or "?"
        key = r.get(by)
        if key is None:
            key = "(unattributed)" if by in ("caller", "cwd") else "?"
        if by == "cwd" and isinstance(key, str) and key != "(unattributed)":
            key = os.path.basename(key.rstrip("/")) or key
        prompt = r.get("prompt_tokens") or 0
        out = (r.get("completion_tokens") or 0) + (r.get("reasoning_tokens") or 0)
        g = groups[key]
        g["calls"] += 1
        g["prompt"] += prompt
        g["out"] += out
        g["max_in"] = max(g["max_in"], prompt)
        if r.get("error"):
            g["errors"] += 1
        c = est_cost(m, prompt, out)
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
        f"\nllmx usage by {by} — since {floor}" + (f" — model={model}" if model else "") + f"  ({n} calls)\n",
        f"  {'(' + by + ')':<{w}}  {'calls':>6}  {'in_tok':>11}  {'out_tok':>11}  {'max_in':>9}  {'est_cost':>9}"
        + (f"  {'ctx_win':>9}  {'%used':>6}" if show_ctx else ""),
        f"  {'-' * w}  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 9}  {'-' * 9}"
        + (f"  {'-' * 9}  {'-' * 6}" if show_ctx else ""),
    ]
    for key, g in rows:
        cost = f"${g['cost']:.2f}" + ("" if g["cost_known"] else "+?")
        err = f"  ({g['errors']} err)" if g["errors"] else ""
        line = (f"  {key[:w]:<{w}}  {g['calls']:>6}  {g['prompt']:>11,}  {g['out']:>11,}  "
                f"{g['max_in']:>9,}  {cost:>9}")
        if show_ctx:
            cw = CONTEXT_WINDOW.get(key)
            pct = f"{100 * g['max_in'] / cw:.0f}%" if cw else "?"
            line += f"  {(format(cw, ',') if cw else '?'):>9}  {pct:>6}"
        out_lines.append(line + err)
    out_lines.append(f"  {'-' * w}  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 9}  {'-' * 9}"
                     + (f"  {'-' * 9}  {'-' * 6}" if show_ctx else ""))
    out_lines.append(f"  {'TOTAL':<{w}}  {total['calls']:>6}  {total['prompt']:>11,}  {total['out']:>11,}  "
                     f"{'':>9}  {'$' + format(total['cost'], '.2f'):>9}")
    out_lines.append("\n  (cost = estimate from PRICING; '+?' = unpriced model. Tokens exact. "
                     "max_in = biggest single call's input; %used vs ctx_win.)")
    return "\n".join(out_lines)
