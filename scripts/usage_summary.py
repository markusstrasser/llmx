#!/usr/bin/env python3
"""Thin wrapper over llmx.usage_report.summarize — kept for back-compat.

Prefer `llmx usage` (same logic, single source). PRICING + the rollup live in
llmx/usage_report.py so there is ONE place to edit rates.

  uv run python3 scripts/usage_summary.py --by model --days 7
"""
from __future__ import annotations

import argparse

from llmx.usage_report import summarize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by", choices=["caller", "cwd", "model", "provider"], default="caller")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--since", help="ISO date floor (overrides --days)")
    ap.add_argument("--model", help="filter to one model")
    ap.add_argument("--log")
    args = ap.parse_args()
    print(summarize(by=args.by, days=args.days, since=args.since, model=args.model, log=args.log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
