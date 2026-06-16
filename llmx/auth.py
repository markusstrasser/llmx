"""Explicit billing/auth surface — api vs subscription."""

from __future__ import annotations

from typing import Literal, Optional

AuthKind = Literal["api", "subscription"]
AUTH_CHOICES: tuple[AuthKind, ...] = ("api", "subscription")


def normalize_auth(value: Optional[str]) -> AuthKind:
    if value is None:
        raise ValueError("auth is required")
    token = value.strip().lower()
    if token not in AUTH_CHOICES:
        raise ValueError(f"auth must be one of {', '.join(AUTH_CHOICES)}; got {value!r}")
    return token  # type: ignore[return-value]


def resolve_auth(
    *,
    auth: Optional[str] = None,
    provider: Optional[str] = None,
    subscription_flag: bool = False,
    lite: Optional[str] = None,
    api_only: Optional[bool] = None,
) -> tuple[AuthKind, str, Optional[str], list[str]]:
    """Resolve caller auth intent.

    Returns (auth, auth_source, effective_lite, warnings).
    auth_source: explicit | explicit_flag | explicit_provider | api_only_deprecated | default_policy
    """
    warnings: list[str] = []
    prov = (provider or "").strip().lower()

    if api_only is not None and auth is not None:
        raise ValueError("pass auth= or api_only=, not both")

    if api_only is not None:
        warnings.append("api_only= is deprecated; use auth='api' or auth='subscription'")
        resolved: AuthKind = "api" if api_only else "subscription"
        return _finalize(resolved, "api_only_deprecated", lite, subscription_flag, warnings)

    if subscription_flag:
        if auth is not None and normalize_auth(auth) != "subscription":
            raise ValueError("--subscription conflicts with --auth api")
        return _finalize("subscription", "explicit_flag", lite, True, warnings)

    if auth is not None:
        return _finalize(normalize_auth(auth), "explicit", lite, False, warnings)

    if prov in ("anthropic-direct",):
        return _finalize("api", "explicit_provider", lite, False, warnings)

    if prov in ("cursor", "cursor-cli", "composer"):
        return _finalize("subscription", "default_policy", lite, False, warnings)

    if prov in ("anthropic", "claude-cli"):
        warnings.append(
            "anthropic auth defaults to subscription; pass auth=api or -p anthropic-direct for metered API"
        )
        return _finalize("subscription", "default_policy", lite, False, warnings)

    if prov in ("openai", "codex-cli", "google", "gemini", ""):
        return _finalize("api", "default_policy", lite, False, warnings)

    return _finalize("api", "default_policy", lite, False, warnings)


def _finalize(
    auth: AuthKind,
    source: str,
    lite: Optional[str],
    _sub_flag: bool,
    warnings: list[str],
) -> tuple[AuthKind, str, Optional[str], list[str]]:
    effective_lite = lite
    if auth == "subscription" and not effective_lite:
        # Lite profile is chosen by mode resolver; auth only signals billing route.
        pass
    return auth, source, effective_lite, warnings


def auth_to_llmx_kwargs(auth: AuthKind, *, lite: Optional[str] = None, mode: Optional[str] = None) -> dict:
    """Map explicit auth (+ optional mode) to llmx.api / LLM.chat kwargs."""
    from .mode import resolve_mode

    resolved_mode, _, effective_lite, _ = resolve_mode(
        mode=mode, lite=lite, auth=auth
    )
    if auth == "subscription":
        return {
            "api_only": False,
            "lite": effective_lite or "bare",
            "auth": auth,
            "mode": resolved_mode,
        }
    return {"api_only": True, "lite": None, "auth": auth, "mode": resolved_mode}
