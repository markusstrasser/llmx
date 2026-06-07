#!/usr/bin/env python3
"""Ad-hoc cost rollups over the llmx usage log (~/.claude/llmx-usage.jsonl).

The log records exact tokens per call (durable); pricing is applied HERE because
it changes. Group by caller (default), cwd, model, or provider to find what is
actually spending.

Examples:
  uv run python3 scripts/usage_summary.py                      # caller, last 30d
  uv run python3 scripts/usage_summary.py --by model --days 7
  uv run python3 scripts/usage_summary.py --by caller --model gemini-3-flash-preview
  uv run python3 scripts/usage_summary.py --by cwd --since 2026-06-01

Token totals are exact. Cost is an ESTIMATE from the editable PRICING table
below (per-MTok input/output; reasoning bills as output). Unknown models show
tokens but cost '?'. Cached tokens are not discounted in the estimate.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Per-MTok (input, output). Edit as pricing changes. Output rate also applies to
# reasoning tokens (they bill as output). Sources: llmx transport-routing notes +
# substack extract MODEL_PRICING (2026-06). Approximate — verify before quoting.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3-flash-preview": (0.075, 0.30),
    "gemini-3-flash": (0.075, 0.30),
    "gemini-3.1-flash-lite-preview": (0.05, 0.20),
    "gemini-3.5-flash": (1.50, 9.0),
    "gemini-3.1-pro-preview": (1.25, 10.0),
    "gpt-5.5": (1.25, 10.0),
    "gpt-5.3-chat-latest": (1.75, 14.0),
    "gpt-5.3-codex": (1.25, 10.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}

DEFAULT_LOG = Path(os.environ.get("LLMX_USAGE_LOG", str(Path.home() / ".claude" / "llmx-usage.jsonl")))


def _est_cost(model: str, prompt: int, out: int) -> float | None:
    rate = PRICING.get(model)
    if rate is None:
        return None
    return (prompt / 1_000_000) * rate[0] + (out / 1_000_000) * rate[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by", choices=["caller", "cwd", "model", "provider"], default="caller")
    ap.add_argument("--days", type=int, default=30, help="look back N days (default 30)")
    ap.add_argument("--since", help="ISO date floor (overrides --days)")
    ap.add_argument("--model", help="filter to one model")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    if args.since:
        floor = args.since[:10]
    else:
        floor = (datetime.now(timezone.utc) - timedelta(days=args.days)).date().isoformat()

    log = Path(args.log)
    if not log.exists():
        print(f"✗ no usage log at {log}")
        return 1

    groups: dict[str, dict] = collections.defaultdict(lambda: {"calls": 0, "prompt": 0, "out": 0, "cost": 0.0, "cost_known": True, "errors": 0})
    total = {"calls": 0, "prompt": 0, "out": 0, "cost": 0.0}
    n_scanned = 0
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = (r.get("ts") or "")[:10]
        if ts and ts < floor:
            continue
        if args.model and r.get("model") != args.model:
            continue
        n_scanned += 1
        model = r.get("model") or "?"
        key = r.get(args.by)
        if key is None:
            key = "(unattributed)" if args.by in ("caller", "cwd") else "?"
        if args.by == "cwd" and isinstance(key, str) and key != "(unattributed)":
            key = os.path.basename(key.rstrip("/")) or key
        prompt = r.get("prompt_tokens") or 0
        out = (r.get("completion_tokens") or 0) + (r.get("reasoning_tokens") or 0)
        g = groups[key]
        g["calls"] += 1
        g["prompt"] += prompt
        g["out"] += out
        if r.get("error"):
            g["errors"] += 1
        c = _est_cost(model, prompt, out)
        if c is None:
            g["cost_known"] = False
        else:
            g["cost"] += c
            total["cost"] += c
        total["calls"] += 1
        total["prompt"] += prompt
        total["out"] += out

    if not groups:
        print(f"No records since {floor}" + (f" for model {args.model}" if args.model else ""))
        return 0

    rows = sorted(groups.items(), key=lambda kv: (kv[1]["cost"], kv[1]["calls"]), reverse=True)
    width = min(40, max(len(k) for k, _ in rows))
    print(f"\nllmx usage by {args.by} — since {floor}" + (f" — model={args.model}" if args.model else "") + f"  ({n_scanned} calls)\n")
    print(f"  {'(' + args.by + ')':<{width}}  {'calls':>6}  {'in_tok':>11}  {'out_tok':>11}  {'est_cost':>9}")
    print(f"  {'-' * width}  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 9}")
    for key, g in rows:
        cost = f"${g['cost']:.2f}" + ("" if g["cost_known"] else "+?")
        err = f"  ({g['errors']} err)" if g["errors"] else ""
        print(f"  {key[:width]:<{width}}  {g['calls']:>6}  {g['prompt']:>11,}  {g['out']:>11,}  {cost:>9}{err}")
    print(f"  {'-' * width}  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 9}")
    print(f"  {'TOTAL':<{width}}  {total['calls']:>6}  {total['prompt']:>11,}  {total['out']:>11,}  {'$' + format(total['cost'], '.2f'):>9}")
    print("\n  (cost = estimate from PRICING table; '+?' = some models unpriced. Tokens are exact.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
