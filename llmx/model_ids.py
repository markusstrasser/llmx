"""Canonical model identifiers shared by routing, auth, and accounting."""

from typing import Optional


# Requested model IDs accepted by the Grok Build subscription transport.
# The CLI may report a distinct served model (currently grok-4.6-build).
GROK_BUILD_MODELS = ("grok-4.6",)

# Grok 4.6 lanes, verified live via `cursor-agent models` on 2026-09-05.
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
CURSOR_GROK_MODELS: tuple[str, ...] = CURSOR_GROK46_MODELS

GROK46_SUBSCRIPTION_DEFAULT = "cursor-grok-4.6-high"

_GROK_BARE_TO_SUBSCRIPTION = {
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
            "grok-4.5 is no longer available under subscription auth because "
            "Cursor removed its 4.5 models; use -m grok-4.6 "
            "(Cursor cursor-grok-4.6-high) or -p grok -m grok-4.6 (Grok Build)"
        )
    return _GROK_BARE_TO_SUBSCRIPTION.get(normalized)
