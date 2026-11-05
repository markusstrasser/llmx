"""Metered-spend hard cap, enforced at the dispatch funnel.

Every genuinely-billed llmx call (transport ends in ``api``) passes through the
native-SDK branch of ``api.LLM.chat``. ``enforce_daily_cap`` runs there, BEFORE
the provider call fires: it sums today's already-billed metered spend from the
usage ledger and refuses once the daily cap is reached. Subscription / CLI
transports ($0 — ``claude-cli``, ``codex-cli``) never reach this guard.

Design (agent-infra ``decisions/2026-06-25-metered-spend-funnel-enforcement.md``,
approved 2026-07-06):

- **The block lives at the funnel, not at a proxy surface.** A backgrounded
  worker's metered escalation bypasses the foreground-Bash ``pretool-cost-guard``
  but still appends to this ledger like every other call — so the funnel is the
  only surface-agnostic choke point (epistemic-discipline #8: don't guard a proxy).
- **Fail-open on missing accounting, but LOUD.** An unreadable/incomplete ledger
  prints ``[DEGRADED]`` to stderr. The known subtotal still blocks at the cap;
  unknown spend alone does not wedge dispatch because a log rotated or a provider
  omitted usage.
- **Fail-LOUD refuse on an unpriced model.** A model with no ``PRICING`` entry can't
  be metered, so it's refused (never priced at $0 — that would let an unpriced model
  spend unbounded). Add it to ``usage_report.PRICING`` or set the override.
- **Explicit per-run override.** ``LLMX_SPEND_OVERRIDE=1`` bypasses the block for a
  deliberately-intended large job; the block defaults on.

The pricing map is single-sourced in ``usage_report.PRICING`` (this module and
agent-infra's ``scripts/usage-check.py`` import it).

Known gap: ``image.py``'s Images API route counts toward the cap through its ledger
rows but never calls ``enforce_daily_cap``, so an image call is not refused at the cap.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .providers import SpendCapError
from .usage_report import DEFAULT_LOG, PRICING, cost_for_usage

# The constitution / invariants.md daily cap. ONE number; the launchd alarm
# (usage-check.py --alarm) and pretool-cost-guard reconcile to it.
DAILY_CAP_USD = 25.0

_OVERRIDE_ENV = "LLMX_SPEND_OVERRIDE"
def is_metered_transport(transport: str | None) -> bool:
    """True iff the transport is genuinely billed (per-token API spend).

    Metered rows in the ledger are ``api`` (native SDK) and ``*-api``
    (``agent-api`` = perplexity research, ``anthropic-direct-api``, …).
    Subscription/CLI transports (``claude-cli``, ``codex-cli``) are $0.
    """
    if not transport:
        return False
    return transport == "api" or transport.endswith("-api")


def metered_spend_today(log_path: str | Path | None = None) -> tuple[float, bool]:
    """Sum today's genuinely-billed (metered) spend from the usage ledger.

    Returns ``(known_subtotal_usd, ledger_ok)``. ``ledger_ok`` is False when the
    ledger is missing/unreadable or today's metered rows have unknown cost.
    The caller warns and checks the known subtotal even when it is incomplete.
    A NEW dispatch of an unpriced model is still refused up front.
    """
    path = Path(log_path) if log_path else DEFAULT_LOG
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    ledger_ok = True
    try:
        raw = path.read_text()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return 0.0, False
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not (r.get("ts", "") or "").startswith(today):
            continue
        if not is_metered_transport(r.get("transport")):
            continue
        c = cost_for_usage(r, conservative=True)
        if c is None:
            ledger_ok = False
        else:
            total += c
    return total, ledger_ok


def enforce_daily_cap(
    model: str,
    *,
    log_path: str | Path | None = None,
    cap_usd: float = DAILY_CAP_USD,
    check_model_priced: bool = True,
) -> None:
    """Refuse a metered dispatch that would breach policy. No-op when allowed.

    Raises ``SpendCapError`` (exit 7) when, absent the override:
      1. ``model`` has no ``PRICING`` entry (can't be metered → fail-loud refuse), or
      2. today's already-billed metered spend has reached ``cap_usd``.

    ``LLMX_SPEND_OVERRIDE=1`` bypasses both. Missing/unreadable/incomplete accounting
    warns loudly with ``[DEGRADED]`` and fails open below the known subtotal's cap.

    ``check_model_priced=False`` skips (1) for providers that self-report cost and
    are legitimately absent from ``PRICING`` (e.g. the Perplexity Agent research
    path, ``transport=='agent-api'``) — the cumulative-cap check (2) still applies.

    Call this ONLY on the metered path — the caller has already established the
    transport is billed. Subscription/CLI dispatch must not reach here.
    """
    if os.environ.get(_OVERRIDE_ENV) == "1":
        print(
            f"[llmx:SPEND] override active ({_OVERRIDE_ENV}=1) — metered-spend cap "
            f"bypassed for this run (model={model})",
            file=sys.stderr,
        )
        return

    if check_model_priced and model not in PRICING:
        raise SpendCapError(
            f"unpriced model {model!r} — add it to usage_report.PRICING or set "
            f"{_OVERRIDE_ENV}=1 to dispatch anyway. Refusing to meter an unpriced "
            f"model (an unknown price cannot be capped).",
            model=model,
        )

    spend, ok = metered_spend_today(log_path)
    if not ok:
        print(
            "[DEGRADED] spend guard: ledger unreadable or incomplete — checking "
            "the known metered subtotal only (failing open below cap). "
            "Unreported spend remains unknown; check the usage ledger.",
            file=sys.stderr,
        )

    if spend >= cap_usd:
        raise SpendCapError(
            f"daily metered-spend cap reached: ${spend:.2f} billed today "
            f">= ${cap_usd:.2f} cap. Refusing further metered dispatch. Set "
            f"{_OVERRIDE_ENV}=1 for a deliberately-intended large job.",
            model=model,
        )
