"""Explicit interaction mode — chat (req/res) vs agent (tools/MCP loop)."""

from __future__ import annotations

from typing import Literal, Optional

from .auth import AuthKind

ModeKind = Literal["chat", "agent"]
MODE_CHOICES: tuple[ModeKind, ...] = ("chat", "agent")

# Internal CLI profile names (legacy --lite); derived from mode, not caller-facing.
LITE_FOR_MODE: dict[ModeKind, str] = {
    "chat": "bare",
    "agent": "research",
}


def normalize_mode(value: Optional[str]) -> ModeKind:
    if value is None:
        raise ValueError("mode is required")
    token = value.strip().lower()
    if token not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {', '.join(MODE_CHOICES)}; got {value!r}")
    return token  # type: ignore[return-value]


def resolve_mode(
    *,
    mode: Optional[str] = None,
    lite: Optional[str] = None,
    auth: AuthKind,
) -> tuple[ModeKind, str, Optional[str], list[str]]:
    """Resolve caller interaction intent.

    Returns (mode, mode_source, effective_lite, warnings).
    mode_source: explicit | lite_deprecated | default_policy | auth_forced
    """
    warnings: list[str] = []

    if lite is not None and mode is not None:
        lite_mode = "chat" if lite == "bare" else "agent" if lite == "research" else None
        if lite_mode and lite_mode != normalize_mode(mode):
            raise ValueError(f"--lite {lite} conflicts with --mode {mode}")

    if lite == "bare":
        warnings.append("--lite bare is deprecated; use --mode chat")
        return "chat", "lite_deprecated", "bare", warnings
    if lite == "research":
        warnings.append("--lite research is deprecated; use --mode agent")
        return "agent", "lite_deprecated", "research", warnings

    if mode is not None:
        resolved = normalize_mode(mode)
        source = "explicit"
    else:
        resolved = "chat"
        source = "default_policy"

    if auth == "api":
        if resolved == "agent":
            warnings.append(
                "auth=api supports mode=chat only (API completion); agent mode needs auth=subscription"
            )
            resolved = "chat"
            source = "auth_forced"
        return resolved, source, None, warnings

    # subscription → CLI transport; lite profile selects tool surface
    effective_lite = LITE_FOR_MODE[resolved]
    return resolved, source, effective_lite, warnings


def mode_to_llmx_kwargs(
    mode: ModeKind,
    *,
    auth: AuthKind,
    lite: Optional[str] = None,
) -> dict:
    """Map mode + auth to llmx kwargs."""
    _, _, effective_lite, _ = resolve_mode(mode=mode, lite=lite, auth=auth)
    out: dict = {"mode": mode}
    if effective_lite:
        out["lite"] = effective_lite
    return out
