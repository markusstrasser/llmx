"""Metered-spend hard cap, enforced at the dispatch funnel.

Every genuinely-billed llmx call (transport ends in ``api``) passes through the
native-SDK branch of ``api.LLM.chat``. ``enforce_daily_cap`` runs there, BEFORE
the provider call fires: it sums today's already-billed metered spend from the
usage ledger and refuses once the daily cap is reached. Subscription / CLI
transports ($0 — ``claude-cli``, ``codex-cli``) never reach this guard.

Design (agent-infra ``decisions/2026-06-25-metered-spend-funnel-enforcement.md``,
approved 2026-07-06):

- **The block lives at the funnel, not at a proxy surface.** A backgrounded
  worker's metered escalation is invisible to a foreground-Bash hook
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
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .providers import SpendCapError
from .usage_report import DEFAULT_LOG, PRICING, cost_for_usage

# The constitution / invariants.md daily cap on metered API spend (subscriptions are $0
# and never reach this guard). ONE number: usage-check.py reports it to doctor. $10 since 2026-11-11 (operator: "10$ should
# be the daily cap (API ... not subscriptions) ... or at least tell me when it's hit");
# $25 before. In the 30 days before the change, 4 of 27 spending days passed $10.
DAILY_CAP_USD = 10.0
_NOTIFY_DIR = Path.home() / ".claude" / "cache"

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


def cap_day(ts: str | None = None) -> str:
    """The local calendar date a ledger timestamp falls on (now when None); "" if unparseable.

    The cap's day is the operator's day. A UTC day resets at 08:00 in UTC+8, which let one
    working day spend the cap twice. Ledger timestamps are UTC; a naive one is read as UTC.
    """
    if ts is None:
        return datetime.now().astimezone().date().isoformat()
    try:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone().date().isoformat()


def metered_spend_today(log_path: str | Path | None = None) -> tuple[float, bool]:
    """Sum today's genuinely-billed (metered) spend from the usage ledger.

    Returns ``(known_subtotal_usd, ledger_ok)``. ``ledger_ok`` is False when the
    ledger is missing/unreadable or today's metered rows have unknown cost.
    The caller warns and checks the known subtotal even when it is incomplete.
    A NEW dispatch of an unpriced model is still refused up front.
    """
    path = Path(log_path) if log_path else DEFAULT_LOG
    today = cap_day()
    # A local day spans at most three UTC dates; skip parsing every other row.
    utc_now = datetime.now(timezone.utc).date()
    near = {(utc_now + timedelta(days=d)).isoformat() for d in (-1, 0, 1)}
    total = 0.0
    ledger_ok = True
    try:
        raw = path.read_text()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return 0.0, False
    for line in raw.splitlines():
        # A row from a near date contains that date as text; skip the JSON parse otherwise
        # (the ledger is tens of thousands of rows and this runs before every paid call).
        if not any(day in line for day in near):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = r.get("ts", "") or ""
        if ts[:10] not in near or cap_day(ts) != today:
            continue
        if not is_metered_transport(r.get("transport")):
            continue
        c = cost_for_usage(r, conservative=True)
        if c is None:
            ledger_ok = False
        else:
            total += c
    return total, ledger_ok


def notify_cap_reached(
    spend: float, cap_usd: float, *, marker_dir: Path = _NOTIFY_DIR, kind: str = "refused"
) -> bool:
    """Tell the operator, once per local day, that the cap stopped a metered call.

    The refusal itself only reaches the agent that made the call. A desktop notification
    reaches the person who can decide to override. Best-effort: a missing notifier or an
    unwritable marker never blocks or changes the refusal. Returns True when it notified.
    """
    today = cap_day()
    marker = marker_dir / f"llmx-spend-cap-{kind}-{today}"
    notifier = shutil.which("terminal-notifier")
    try:
        if marker.exists() or notifier is None:
            return False
        marker_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [
                notifier,
                "-title", "llmx: daily API cap reached",
                "-message",
                f"${spend:.2f} metered today, cap ${cap_usd:.2f}. "
                + (
                    f"A run is spending past the cap with {_OVERRIDE_ENV}=1."
                    if kind == "override"
                    else f"Paid calls are refused until local midnight unless {_OVERRIDE_ENV}=1 is set."
                ),
                "-group", "llmx-spend-cap",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        marker.write_text(f"{spend:.2f}\n")  # after the launch: a failed launch retries
        return True
    except OSError:
        return False


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
        # The override skips the refusal, so it would also skip the operator's notice.
        spend, _ = metered_spend_today(log_path)
        if spend >= cap_usd and log_path is None:
            notify_cap_reached(spend, cap_usd, kind="override")
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
        if log_path is None:  # the real ledger, not a caller-supplied one
            notify_cap_reached(spend, cap_usd)
        raise SpendCapError(
            f"daily metered-spend cap reached: ${spend:.2f} billed today "
            f">= ${cap_usd:.2f} cap. Refusing further metered dispatch until local midnight. "
            f"{_OVERRIDE_ENV}=1 lifts it, and granting that is the operator's decision: "
            f"report this refusal and use a subscription route meanwhile; do not set it yourself.",
            model=model,
        )
