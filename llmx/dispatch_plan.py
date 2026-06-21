"""Dispatch resolution helpers — shared by chat, dry-run, and info."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .auth import AuthKind, resolve_auth
from .mode import ModeKind, resolve_mode
from .cli_backends import (
    CLI_PROVIDERS,
    LITE_ALLOWED_MODELS,
    binary_available,
    configured_cli_provider,
    lite_model_allowed,
    needs_api_fallback,
    preferred_cli_provider,
    subscription_route,
)
from .providers import get_model_name, get_model_restriction, infer_provider_from_model

SCHEMA_VERSION = "llmx-routing.v1"

# User-facing effort tokens (before backend mapping).
EFFORT_ALIASES = {
    "max": "max",  # resolved per-backend below
}

CANONICAL_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def normalize_effort_input(value: Optional[str]) -> tuple[Optional[str], list[str]]:
    """Normalize user effort; return (token, warnings). Raises ValueError if invalid."""
    if value is None:
        return None, []
    token = value.strip().lower()
    if token not in CANONICAL_EFFORTS:
        raise ValueError(
            f"Invalid effort {value!r}. "
            f"Use one of: {', '.join(sorted(CANONICAL_EFFORTS))}"
        )
    warnings: list[str] = []
    if token == "max":
        warnings.append("effort=max is backend-specific; see effort_applied in dispatch plan")
    return token, warnings


def map_effort_for_backend(
    effort: Optional[str],
    *,
    transport: str,
    provider: str,
) -> tuple[Optional[str], list[str]]:
    """Map canonical/user effort to what the chosen backend will receive."""
    if not effort:
        return None, []
    warnings: list[str] = []
    e = effort.lower()
    if transport.endswith("-api") or provider in {"openai", "google", "anthropic-direct"}:
        if e == "max":
            return "xhigh", ["effort max mapped to xhigh for API transport"]
        return e if e != "max" else "xhigh", warnings
    if transport == "claude-cli":
        # Claude Code headless --effort: low|medium|high|max
        mapping = {
            "none": "low",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "max",
            "max": "max",
        }
        applied = mapping.get(e, "high")
        if e in {"none", "minimal", "xhigh"}:
            warnings.append(f"effort {e} mapped to {applied} for claude-cli")
        return applied, warnings
    if transport == "codex-cli":
        mapping = {
            "max": "xhigh",
            "xhigh": "xhigh",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "minimal": "minimal",
            "none": "minimal",
        }
        applied = mapping.get(e, "high")
        if e == "max":
            warnings.append("effort max mapped to xhigh for codex-cli")
        return applied, warnings
    # cursor-cli and unknown: effort not forwarded
    warnings.append(f"effort ignored for transport {transport}")
    return None, warnings


def resolve_effort(
    effort: Optional[str],
    *,
    transport: str,
    provider: str,
) -> tuple[Optional[str], list[str]]:
    """Normalize user effort then map to backend-specific value."""
    if not effort:
        return None, []
    token, warns = normalize_effort_input(effort)
    applied, backend_warns = map_effort_for_backend(
        token, transport=transport, provider=provider
    )
    return applied, warns + backend_warns


def combine_file_context(file_paths: tuple[str, ...], parts: list[str]) -> str:
    """Join -f contents with explicit path boundaries."""
    blocks: list[str] = []
    for fp, part in zip(file_paths, parts, strict=True):
        blocks.append(f"=== File: {fp} ===\n{part}")
    return "\n\n".join(blocks)


@dataclass
class DispatchPlan:
    schema_version: str = SCHEMA_VERSION
    provider: str = ""
    model: str = ""
    transport: str = ""
    auth: AuthKind = "api"
    auth_source: str = "default_policy"
    mode: ModeKind = "chat"
    mode_source: str = "default_policy"
    subscription: bool = False
    lite: Optional[str] = None
    requested_effort: Optional[str] = None
    effort_applied: Optional[str] = None
    effort_warnings: list[str] = field(default_factory=list)
    timeout: int = 300
    cli_fallback_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stderr_line(self) -> str:
        effort = self.effort_applied or self.requested_effort or "-"
        return (
            f"[llmx] auth={self.auth} mode={self.mode} transport={self.transport} "
            f"model={self.model} effort={effort} timeout={self.timeout}s"
        )


def build_dispatch_plan(
    *,
    provider: Optional[str],
    model: Optional[str],
    reasoning_effort: Optional[str],
    timeout: int,
    lite: Optional[str],
    mode: Optional[str],
    auth: Optional[str],
    subscription: bool,
    api_only: Optional[bool],
    use_old: bool,
    schema: Optional[dict] = None,
    system: Optional[str] = None,
    search: bool = False,
    stream: bool = False,
    max_tokens: Optional[int] = None,
) -> DispatchPlan:
    warnings: list[str] = []
    effort_token, effort_warn = normalize_effort_input(reasoning_effort)
    warnings.extend(effort_warn)

    final_provider = provider or "google"
    if model and not provider:
        inferred = infer_provider_from_model(model)
        if inferred:
            final_provider = inferred

    effective_lite = lite
    resolved_auth, auth_source, effective_lite, auth_warns = resolve_auth(
        auth=auth,
        provider=final_provider,
        subscription_flag=subscription,
        lite=effective_lite,
        api_only=api_only,
    )
    warnings.extend(auth_warns)

    resolved_mode, mode_source, effective_lite, mode_warns = resolve_mode(
        mode=mode,
        lite=effective_lite,
        auth=resolved_auth,
    )
    warnings.extend(mode_warns)

    cli_provider = preferred_cli_provider(final_provider, lite=effective_lite)
    if cli_provider:
        logical_provider = (
            CLI_PROVIDERS[cli_provider]["api_fallback"]
            if final_provider in CLI_PROVIDERS
            else final_provider
        )
        planned_model = model or get_model_name(logical_provider, None, use_old)
        cli_fallback_reason = needs_api_fallback(
            cli_provider,
            schema,
            system,
            search,
            stream,
            effort_token,
            max_tokens,
        )
        if cli_fallback_reason and not subscription_route(
            auth=resolved_auth, lite=effective_lite
        ):
            api_fb = CLI_PROVIDERS[cli_provider]["api_fallback"]
            planned_transport = f"{api_fb}-api" if api_fb else cli_provider
        else:
            planned_transport = cli_provider
    else:
        planned_model = get_model_name(final_provider, model, use_old)
        planned_transport = f"{final_provider}-api"
        cli_fallback_reason = None

    effort_applied, backend_warn = map_effort_for_backend(
        effort_token,
        transport=planned_transport,
        provider=final_provider,
    )
    warnings.extend(backend_warn)

    if effective_lite == "bare" and planned_model and not lite_model_allowed(planned_model):
        warnings.append(f"model {planned_model!r} not on lite allowlist")

    return DispatchPlan(
        provider=final_provider,
        model=planned_model,
        transport=planned_transport,
        auth=resolved_auth,
        auth_source=auth_source,
        mode=resolved_mode,
        mode_source=mode_source,
        subscription=resolved_auth == "subscription",
        lite=effective_lite,
        requested_effort=effort_token,
        effort_applied=effort_applied,
        effort_warnings=backend_warn,
        timeout=timeout,
        cli_fallback_reason=cli_fallback_reason,
        warnings=warnings,
    )


def collect_routing_mirror() -> dict[str, Any]:
    """Filesystem-friendly routing facts for agents (not judgment/policy)."""
    cli_status = {}
    for name, cfg in CLI_PROVIDERS.items():
        binary = cfg["binary"]
        cli_status[name] = {
            "binary": binary,
            "installed": bool(shutil.which(binary)),
            "api_fallback": cfg.get("api_fallback"),
        }
    logical_aliases = {}
    for logical in ("openai", "anthropic", "google", "cursor"):
        cli = configured_cli_provider(logical, lite="bare")
        logical_aliases[logical] = {
            "lite_bare_cli": cli,
            "lite_bare_available": binary_available(logical) if cli else False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "cli_providers": cli_status,
        "anthropic_default_auth": "subscription",
        "anthropic_api_opt_in": "-p anthropic-direct or --auth api",
        "auth_surface": "Use --auth api|subscription (or auth= in Python). --subscription is an alias for --auth subscription.",
        "mode_surface": "Use --mode chat|agent. chat=one-shot req/res; agent=CLI tools/MCP loop (subscription only). --lite bare|research are deprecated aliases.",
        "logical_subscription_routes": logical_aliases,
        "lite_allowed_models": sorted(LITE_ALLOWED_MODELS),
        "effort_aliases": sorted(CANONICAL_EFFORTS),
        "note": (
            "Transport facts only. Task-class economics and cosigner policy live in "
            "model-guide/SKILL.md — not duplicated here."
        ),
    }
