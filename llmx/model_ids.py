"""Canonical model identifiers shared by routing, auth, and accounting."""

from typing import Optional


# Requested model IDs accepted by the Grok Build subscription transport.
# The CLI may report a distinct served model (4.6 smoke: grok-4.6-build).
# grok 1.0.41 `models` (2026-09-23): grok-4.7 (default), grok-4.7-build-fast,
# grok-4.6, grok-4.5. 1.0.13 did not list 4.7. 4.6 retired from routing 2026-09-25.
GROK_BUILD_MODELS = ("grok-4.7", "grok-4.7-build-fast")

# Grok 4.7 lanes, verified live via `cursor-agent models` on 2026-09-23. Unlike
# 4.6, Cursor lists them WITHOUT the `cursor-` prefix; `cursor-grok-4.7-*`
# does not exist (the 2026-09-22 guess, made while CLI auth was unavailable).
CURSOR_GROK47_MODELS: tuple[str, ...] = (
    "grok-4.7-low",
    "grok-4.7-low-fast",
    "grok-4.7-medium",
    "grok-4.7-medium-fast",
    "grok-4.7-high",
    "grok-4.7-high-fast",
    "grok-4.7-xhigh",
    "grok-4.7-xhigh-fast",
)

# Grok 4.6 lanes (verified 2026-09-05). RETIRED from routing 2026-09-25 (Pareto
# frontier prune; successor grok-4.7-X). Kept only so usage_report can price
# historical llmx-usage.jsonl rows — never admit these for dispatch.
CURSOR_GROK46_RETIRED_MODELS: tuple[str, ...] = (
    "cursor-grok-4.6-low",
    "cursor-grok-4.6-low-fast",
    "cursor-grok-4.6-medium",
    "cursor-grok-4.6-medium-fast",
    "cursor-grok-4.6-high",
    "cursor-grok-4.6-high-fast",
    "cursor-grok-4.6-xhigh",
    "cursor-grok-4.6-xhigh-fast",
)

# Every exact Cursor Grok subscription slug. Consumers that gate by prefix must
# check membership here; retired or invented versions must fail closed.
CURSOR_GROK_MODELS: tuple[str, ...] = CURSOR_GROK47_MODELS

GROK47_SUBSCRIPTION_DEFAULT = "grok-4.7-high"
GROK_SUBSCRIPTION_DEFAULT = GROK47_SUBSCRIPTION_DEFAULT

_GROK_BARE_TO_SUBSCRIPTION = {
    "grok-4.7": GROK47_SUBSCRIPTION_DEFAULT,
}


class RetiredModelError(ValueError):
    """A retired id was requested. The CLI exits 2 (usage error) and names the successor."""


def resolve_grok_subscription_slug(model: Optional[str]) -> Optional[str]:
    """Resolve a supported bare Grok subscription id, rejecting retired 4.5.

    Single source for the bare-grok-under-subscription alias — loaded by both
    dispatch_plan.build_dispatch_plan (plan/dry-run reporting) and providers.chat
    (the real CLI dispatch) so the two resolution paths can't diverge. Exact
    CURSOR_GROK_MODELS slugs are untouched.
    """
    if not model:
        return None
    normalized = model.strip().lower().removeprefix("xai/")
    if normalized in ("grok-4.5", "grok-4.6"):
        raise RetiredModelError(
            f"{normalized} is no longer the current subscription model; use -m grok-4.7 "
            "(Cursor grok-4.7-high) or -p grok -m grok-4.7 (Grok Build)"
        )
    return _GROK_BARE_TO_SUBSCRIPTION.get(normalized)


# API ids retired from routing 2026-09-25 (Pareto-frontier prune, pass 2) → successor.
# Dispatch fails closed on these (providers._auto_upgrade_model raises). Pricing rows
# stay in usage_report for accounting only. gpt-5.3* retired by operator ruling
# ("6 luna wins", 2026-09-25), superseding evals DECISIONS.md intel-extract-model.
_GROK_SUCCESSOR = "grok-4.7 (Cursor grok-4.7-high) or -p grok -m grok-4.7 (Grok Build)"

RETIRED_API_MODELS: dict[str, str] = {
    "gpt-5.4": "gpt-6-sol",
    "gpt-5.3": "gpt-6-luna",
    "gpt-5.3-chat-latest": "gpt-6-luna",
    "gpt-5.3-codex": "gpt-6-luna",
    "gpt-5.2": "gpt-6-sol",
    "gpt-5.1": "gpt-6-sol",
    "gpt-5.1-mini": "gpt-6-luna",
    "gpt-5": "gpt-6-sol",
    "gpt-5-pro": "gpt-6-sol",
    "gpt-5-codex": "gpt-6-sol",
    "grok-4.5": _GROK_SUCCESSOR,
    "grok-4": _GROK_SUCCESSOR,
    "grok-4-1-fast-reasoning": _GROK_SUCCESSOR,
    "grok-4-1-fast-non-reasoning": _GROK_SUCCESSOR,
    "grok-4.20-0309-reasoning": _GROK_SUCCESSOR,
    "grok-4.20-0309-non-reasoning": _GROK_SUCCESSOR,
    "grok-beta": _GROK_SUCCESSOR,
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3.7-flash": "gemini-3.8-flash",
    "gemini-3.6-flash": "gemini-3.8-flash",
    "gemini-3.5-flash": "gemini-3.8-flash",
    # Pass-1 ids: pass 1 only dropped them from the subscription allowlist, so the API
    # lane still dispatched them (model-guide known-issues 2026-09-25).
    "gpt-5.6": "gpt-6-sol",
    "gpt-5.6-sol": "gpt-6-sol",
    "gpt-5.6-terra": "gpt-6-sol",
    "gpt-5.6-luna": "gpt-6-luna",
    "claude-fable-5": "claude-fable-5-1",
    "claude-opus-4-8": "claude-opus-5-5",
    "gemini-3-flash-preview": "gemini-3.8-flash",
}

# Cursor Composer retired 2026-10-07 (operator: "outdated"; `cursor-agent models` lists no
# newer Composer). Successor by operator choice: GPT-6 Astra at low effort on the $0 codex
# subscription. Never priced in usage_report (Cursor pool, no list price), so no history row.
COMPOSER_RETIRED_MODELS: tuple[str, ...] = ("composer-2.5", "composer-2.5-fast")
COMPOSER_SUCCESSOR = "gpt-6-astra --subscription -e low"
RETIRED_API_MODELS.update({model: COMPOSER_SUCCESSOR for model in COMPOSER_RETIRED_MODELS})

# Retirement date and reason per id; ids absent here retired in the 2026-09-25 prune.
_PRUNE_2026_09_25 = "2026-09-25 (Pareto-frontier prune)"
RETIRED_ON: dict[str, str] = {
    model: "2026-10-07 (Composer outdated)" for model in COMPOSER_RETIRED_MODELS
}


def retired_on(model: str) -> str:
    return RETIRED_ON.get(model, _PRUNE_2026_09_25)
