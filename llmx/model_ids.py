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


def resolve_grok_subscription_slug(model: Optional[str]) -> Optional[str]:
    """Return the Cursor subscription slug for bare 'grok-4.5', else None.

    Single source for the bare-grok-4.5-under-subscription alias — loaded by both
    dispatch_plan.build_dispatch_plan (plan/dry-run reporting) and providers.chat
    (the real CLI dispatch) so the two resolution paths can't diverge. Exact
    CURSOR_GROK45_MODELS slugs (e.g. 'cursor-grok-4.5-low') are untouched — this
    only recognizes the bare xAI id.
    """
    if not model:
        return None
    normalized = model.strip().lower().removeprefix("xai/")
    if normalized == "grok-4.5":
        return GROK45_SUBSCRIPTION_DEFAULT
    return None
