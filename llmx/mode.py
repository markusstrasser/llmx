"""Explicit interaction mode — chat (req/res) vs agent (tools/MCP loop)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .auth import AuthKind, resolve_auth

ModeKind = Literal["chat", "agent"]
MODE_CHOICES: tuple[ModeKind, ...] = ("chat", "agent")

# Rendered once by each surface (CLI: stderr) when auth_source == "mode_implied".
AGENT_IMPLIES_SUBSCRIPTION = "--mode agent implies --auth subscription"

# How each explicit api auth source is spelled back to the caller.
_API_SOURCE_SPELLING = {
    "explicit": "--auth api",
    "explicit_provider": "-p anthropic-direct",
    "api_only_deprecated": "api_only=True",
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
    """Resolve caller interaction intent for an already-resolved auth.

    Returns (mode, mode_source, effective_lite, warnings).
    mode_source: explicit | lite_deprecated | default_policy

    Raises ValueError for mode=agent with auth=api: an API completion cannot run
    the workspace agent, and downgrading to chat would silently drop the
    repository access the caller asked for. Entry points that still have to
    choose auth go through ``resolve_auth_mode``.
    """
    warnings: list[str] = []

    if lite is not None and mode is not None:
        lite_mode = "chat" if lite == "bare" else "agent" if lite == "research" else None
        if lite_mode and lite_mode != normalize_mode(mode):
            raise ValueError(f"--lite {lite} conflicts with --mode {mode}")

    if lite == "bare":
        warnings.append("--lite bare selects the legacy isolated no-tools profile")
        return "chat", "lite_deprecated", "bare", warnings
    if lite == "research":
        warnings.append(
            "--lite research selects the legacy isolated research-MCP profile; "
            "--mode agent is caller-workspace execution"
        )
        return "agent", "lite_deprecated", "research", warnings

    if mode is not None:
        resolved = normalize_mode(mode)
        source = "explicit"
    else:
        resolved = "chat"
        source = "default_policy"

    if auth == "api":
        if resolved == "agent":
            raise ValueError(
                "mode=agent needs auth=subscription (a CLI agent in the caller's "
                "workspace); auth=api supports mode=chat only"
            )
        return resolved, source, None, warnings

    # ``chat`` is the neutral request/response profile. ``agent`` is not a lite
    # profile: it keeps the caller's cwd, project instructions, and native CLI
    # tools. Legacy ``--lite research`` remains an explicitly isolated research
    # profile and is resolved above.
    effective_lite = "bare" if resolved == "chat" else None
    return resolved, source, effective_lite, warnings


@dataclass(frozen=True)
class AuthMode:
    """Billing route and interaction mode resolved together (``resolve_auth_mode``)."""

    auth: AuthKind
    auth_source: str
    mode: ModeKind
    mode_source: str
    lite: Optional[str]
    warnings: tuple[str, ...]


def resolve_auth_mode(
    *,
    mode: Optional[str] = None,
    lite: Optional[str] = None,
    auth: Optional[str] = None,
    provider: Optional[str] = None,
    subscription_flag: bool = False,
    api_only: Optional[bool] = None,
) -> AuthMode:
    """Resolve auth and mode as one decision: the rule the CLI and Python API share.

    A workspace agent (explicit ``mode="agent"`` without a legacy ``lite``
    profile) runs a subscription CLI, so an explicit agent request settles the
    billing route instead of being downgraded by it:

    * auth left to provider default policy -> ``subscription`` with auth_source
      ``mode_implied``; surfaces render ``AGENT_IMPLIES_SUBSCRIPTION`` once;
    * auth=api chosen explicitly (``--auth api``, ``-p anthropic-direct``,
      ``api_only=True``) -> ``ValueError``.

    Observed 2026-10-14: three ``--mode agent -m gpt-6.1-sol`` reviews took
    OpenAI's default auth=api, were silently downgraded to chat, and failed on
    "Invalid API key"; with a valid key they would have run metered one-shot
    chats with no repository access.
    """
    resolved_auth, auth_source, effective_lite, warnings = resolve_auth(
        auth=auth,
        provider=provider,
        subscription_flag=subscription_flag,
        lite=lite,
        api_only=api_only,
    )
    explicit_agent = lite is None and mode is not None and normalize_mode(mode) == "agent"
    if explicit_agent and resolved_auth == "api":
        if auth_source != "default_policy":
            spelled = _API_SOURCE_SPELLING.get(auth_source, f"auth source {auth_source!r}")
            raise ValueError(
                f"--mode agent needs subscription auth, but {spelled} selects the "
                "metered API; drop --mode agent for an API chat, or use --auth subscription"
            )
        resolved_auth, auth_source = "subscription", "mode_implied"
    resolved_mode, mode_source, effective_lite, mode_warnings = resolve_mode(
        mode=mode, lite=effective_lite, auth=resolved_auth
    )
    return AuthMode(
        auth=resolved_auth,
        auth_source=auth_source,
        mode=resolved_mode,
        mode_source=mode_source,
        lite=effective_lite,
        warnings=tuple(warnings + mode_warnings),
    )


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
