"""Canonical model identifiers shared by routing, auth, and accounting."""

from typing import Optional


# Requested model IDs accepted by the Grok Build subscription transport.
# The CLI may report a distinct served model (4.6 smoke: grok-4.6-build).
# grok 1.0.41 `models` (2026-09-23): grok-4.7 (default), grok-4.7-build-fast,
# grok-4.6, grok-4.5. 1.0.13 did not list 4.7.
GROK_BUILD_MODELS = ("grok-4.7", "grok-4.7-build-fast", "grok-4.6")

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

# Grok 4.6 lanes, verified live via `cursor-agent models` on 2026-09-05.
# Still admitted: this Cursor session's catalog still lists a 4.6 slug.
CURSOR_GROK46_MODELS: tuple[str, ...] = (
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
CURSOR_GROK_MODELS: tuple[str, ...] = CURSOR_GROK47_MODELS + CURSOR_GROK46_MODELS

GROK47_SUBSCRIPTION_DEFAULT = "grok-4.7-high"
GROK46_SUBSCRIPTION_DEFAULT = "cursor-grok-4.6-high"
GROK_SUBSCRIPTION_DEFAULT = GROK47_SUBSCRIPTION_DEFAULT

_GROK_BARE_TO_SUBSCRIPTION = {
    "grok-4.7": GROK47_SUBSCRIPTION_DEFAULT,
    "grok-4.6": GROK46_SUBSCRIPTION_DEFAULT,
}


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
    if normalized == "grok-4.5":
        raise ValueError(
            "grok-4.5 is no longer the current subscription model; use -m grok-4.7 "
            "(Cursor grok-4.7-high) or -p grok -m grok-4.7 (Grok Build)"
        )
    return _GROK_BARE_TO_SUBSCRIPTION.get(normalized)
