"""Canonical model identifiers shared by routing, auth, and accounting."""

from typing import Optional

CURSOR_GROK45_MODELS: tuple[str, ...] = (
    "cursor-grok-4.5-low",
    "cursor-grok-4.5-low-fast",
    "cursor-grok-4.5-medium",
    "cursor-grok-4.5-medium-fast",
    "cursor-grok-4.5-high",
    "cursor-grok-4.5-high-fast",
)

# Grok 4.6 lanes, verified live via `cursor-agent --list-models` on 2026-08-27
# (Cursor added an xhigh tier for 4.6; 4.5 still has none).
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

# Every exact Cursor Grok subscription slug. Consumers that gate by prefix
# ("cursor-grok-") must check membership here, never a hard-coded version.
CURSOR_GROK_MODELS: tuple[str, ...] = CURSOR_GROK45_MODELS + CURSOR_GROK46_MODELS

# Bare "grok-4.5" (xAI's flagship model id — see MODEL_RESTRICTIONS["grok-4.5"] in
# providers.py: "reasoning low/medium/high (default high)") has no 1:1 Cursor slug.
# Cursor only exposes the exact effort-suffixed ids above (verified live via
# `cursor-agent --list-models`; no "xhigh" tier exists there). f156e1f deliberately
# stopped guessing/aliasing effort-suffixed slugs to block silent metered-API
# fallback masquerading as subscription. This constant + helper fill the one
# remaining gap: under subscription auth, bare "grok-4.5" resolves to the "high"
# slug (xAI's own documented default effort) instead of dead-ending on the
# metered xai-api transport.
GROK45_SUBSCRIPTION_DEFAULT = "cursor-grok-4.5-high"
GROK46_SUBSCRIPTION_DEFAULT = "cursor-grok-4.6-high"

_GROK_BARE_TO_SUBSCRIPTION = {
    "grok-4.5": GROK45_SUBSCRIPTION_DEFAULT,
    "grok-4.6": GROK46_SUBSCRIPTION_DEFAULT,
}


def resolve_grok_subscription_slug(model: Optional[str]) -> Optional[str]:
    """Return the Cursor subscription slug for a bare 'grok-4.x' id, else None.

    Single source for the bare-grok-under-subscription alias — loaded by both
    dispatch_plan.build_dispatch_plan (plan/dry-run reporting) and providers.chat
    (the real CLI dispatch) so the two resolution paths can't diverge. Exact
    CURSOR_GROK_MODELS slugs (e.g. 'cursor-grok-4.6-low') are untouched — this
    only recognizes the bare xAI ids ('grok-4.5' → high, 'grok-4.6' → high).
    """
    if not model:
        return None
    normalized = model.strip().lower().removeprefix("xai/")
    return _GROK_BARE_TO_SUBSCRIPTION.get(normalized)
