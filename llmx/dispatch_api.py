"""Structured dispatch API — transport truth for library callers.

ADR: agent-infra `decisions/2026-06-15-llmx-refactor-dispatch-layer.md` P1.
Skills keep named profiles; this module owns status taxonomy + context concat +
auth resolution for one-shot calls. Prefer ``dispatch()`` over raw ``chat()``
when the caller needs structured failure classes (not exceptions alone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from .dispatch_plan import build_dispatch_plan, combine_file_context
from .providers import (
    EXIT_API_KEY,
    EXIT_GENERAL,
    EXIT_MODEL_ERROR,
    EXIT_QUOTA,
    EXIT_RATE_LIMIT,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    ApiKeyError,
    LlmxError,
    ModelError,
    QuotaError,
    RateLimitError,
    SpendCapError,
    TimeoutError_,
)

PathLike = Union[str, Path]

# Status strings align with skills/llm_dispatch taxonomy where they overlap.
STATUS_EXIT_CODES: dict[str, int] = {
    "ok": EXIT_SUCCESS,
    "dry_run": EXIT_SUCCESS,
    "timeout": EXIT_TIMEOUT,
    "rate_limit": EXIT_RATE_LIMIT,
    "quota": EXIT_QUOTA,
    "api_key": EXIT_API_KEY,
    "model_error": EXIT_MODEL_ERROR,
    "schema_error": EXIT_MODEL_ERROR,
    "empty_output": EXIT_MODEL_ERROR,
    "config_error": EXIT_GENERAL,
    "dependency_error": EXIT_GENERAL,
    "dispatch_error": EXIT_GENERAL,
    "spend_cap": EXIT_QUOTA,
}

RETRYABLE_STATUSES: dict[str, bool] = {
    "ok": False,
    "dry_run": False,
    "timeout": True,
    "rate_limit": True,
    "quota": False,
    "api_key": False,
    "model_error": False,
    "schema_error": False,
    "empty_output": True,
    "config_error": False,
    "dependency_error": False,
    "dispatch_error": False,
    "spend_cap": False,
}


@dataclass
class DispatchResult:
    """Structured one-shot dispatch outcome (library API).

    ``status`` is the machine class; ``exit_code`` maps to CLI exit codes.
    On success, ``text`` holds the model output (also ``response.content``).
    """

    status: str
    retryable: bool
    text: str = ""
    provider: str = ""
    model: str = ""
    transport: str = ""
    auth: str = ""
    mode: str = ""
    effort_applied: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    dry_run_plan: Optional[dict[str, Any]] = None
    response: Any = None

    @property
    def exit_code(self) -> int:
        return STATUS_EXIT_CODES.get(self.status, EXIT_GENERAL)

    @property
    def exit_class(self) -> str:
        return self.status

    @property
    def content(self) -> str:
        """Alias so callers that expect chat-like ``.content`` keep working."""
        return self.text

    def ok(self) -> bool:
        return self.status == "ok"


def classify_dispatch_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception to (status, message). Prefer typed LlmxError subclasses."""
    message = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, SpendCapError):
        return "spend_cap", message
    if isinstance(exc, TimeoutError_):
        return "timeout", message
    if isinstance(exc, RateLimitError):
        return "rate_limit", message
    if isinstance(exc, QuotaError):
        return "quota", message
    if isinstance(exc, ApiKeyError):
        return "api_key", message
    if isinstance(exc, ModelError):
        return "model_error", message
    if isinstance(exc, TimeoutError):
        return "timeout", message
    if isinstance(exc, ImportError):
        return "dependency_error", message
    if isinstance(exc, LlmxError):
        code = getattr(exc, "exit_code", EXIT_GENERAL)
        by_code = {
            EXIT_TIMEOUT: "timeout",
            EXIT_RATE_LIMIT: "rate_limit",
            EXIT_QUOTA: "quota",
            EXIT_API_KEY: "api_key",
            EXIT_MODEL_ERROR: "model_error",
        }
        return by_code.get(code, "dispatch_error"), message

    lowered = message.lower()
    if any(m in lowered for m in ("timed out", "timeout", "deadline exceeded")):
        return "timeout", message
    if any(
        m in lowered
        for m in (
            "rate limit",
            "rate-limit",
            "resource_exhausted",
            "429",
            "too many requests",
            "overloaded",
            "503",
            "unavailable",
        )
    ):
        return "rate_limit", message
    if any(
        m in lowered
        for m in (
            "insufficient_quota",
            "quota",
            "billing",
            "credit",
            "payment required",
            "exhausted balance",
        )
    ):
        return "quota", message
    if any(m in lowered for m in ("schema", "response_format", "additionalproperties")):
        return "schema_error", message
    if "api key" in lowered or "api_key" in lowered or "authentication" in lowered:
        return "api_key", message
    return "model_error", message


def load_context_paths(paths: Sequence[PathLike]) -> str:
    """Read and join context files with ``=== File: path ===`` boundaries."""
    if not paths:
        return ""
    str_paths = tuple(str(p) for p in paths)
    parts: list[str] = []
    for fp in str_paths:
        parts.append(Path(fp).read_text().strip())
    return combine_file_context(str_paths, parts)


def compose_prompt(
    prompt: str,
    *,
    context_paths: Optional[Sequence[PathLike]] = None,
    context_text: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Build the full prompt. Returns (full_prompt, warnings)."""
    warnings: list[str] = []
    chunks: list[str] = []
    if context_paths:
        if len(context_paths) > 1:
            warnings.append(
                f"concatenated {len(context_paths)} context files with path boundaries"
            )
        chunks.append(load_context_paths(context_paths))
    if context_text and context_text.strip():
        chunks.append(context_text.strip())
    body = "\n\n".join(c for c in chunks if c)
    if body and prompt:
        return body + "\n\n---\n\n" + prompt, warnings
    if body:
        return body, warnings
    return prompt, warnings


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    try:
        return dict(usage)
    except Exception:
        return {}


def dispatch(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    context_paths: Optional[Sequence[PathLike]] = None,
    context_text: Optional[str] = None,
    output_path: Optional[PathLike] = None,
    auth: Optional[str] = None,
    subscription: bool = False,
    api_only: Optional[bool] = None,
    mode: Optional[str] = None,
    effort: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    schema: Optional[dict[str, Any]] = None,
    timeout: Optional[int] = None,
    dry_run: bool = False,
    temperature: float = 0.7,
    search: bool = False,
    caller: Optional[str] = None,
    **kwargs: Any,
) -> DispatchResult:
    """One-shot structured dispatch.

    Prefer ``auth='api'|'subscription'`` (or ``subscription=True``). ``api_only``
    remains accepted with a deprecation warning inside auth resolution.

    ``dry_run=True`` resolves the plan and returns ``status='dry_run'`` without
    calling a model. Live failures return a non-ok ``DispatchResult`` (do not
    raise) except for programmer errors (invalid auth/mode) which raise
    ``ValueError``.
    """
    warnings: list[str] = []
    try:
        full_prompt, compose_warns = compose_prompt(
            prompt, context_paths=context_paths, context_text=context_text
        )
        warnings.extend(compose_warns)
    except OSError as exc:
        return DispatchResult(
            status="config_error",
            retryable=False,
            error_type="config_error",
            error_message=str(exc),
            warnings=warnings,
            provider=provider or "",
            model=model or "",
        )

    if schema is None and "response_format" in kwargs:
        schema = kwargs.pop("response_format")

    effort_token = effort or reasoning_effort or kwargs.pop("reasoning_effort", None)
    if effort and reasoning_effort and effort != reasoning_effort:
        raise ValueError("pass effort= or reasoning_effort=, not conflicting values")

    timeout_val = int(timeout) if timeout is not None else int(kwargs.pop("timeout", 300))

    try:
        plan = build_dispatch_plan(
            provider=provider,
            model=model,
            reasoning_effort=effort_token,
            timeout=timeout_val,
            lite=kwargs.get("lite"),
            mode=mode,
            auth=auth,
            subscription=subscription,
            api_only=api_only,
            use_old=bool(kwargs.pop("use_old", False)),
            schema=schema,
            system=system,
            search=search,
            stream=False,
            max_tokens=kwargs.get("max_tokens"),
        )
    except ValueError as exc:
        return DispatchResult(
            status="config_error",
            retryable=False,
            error_type="config_error",
            error_message=str(exc),
            warnings=warnings,
            provider=provider or "",
            model=model or "",
        )

    warnings.extend(plan.warnings)
    if dry_run:
        return DispatchResult(
            status="dry_run",
            retryable=False,
            text="",
            provider=plan.provider,
            model=plan.model,
            transport=plan.transport,
            auth=plan.auth,
            mode=plan.mode,
            effort_applied=plan.effort_applied,
            warnings=warnings,
            dry_run_plan=plan.to_dict(),
        )

    # Lazy import avoids circular import (api.chat → providers → …).
    from .api import chat as _chat

    call_kwargs: dict[str, Any] = {
        "provider": plan.provider,
        "model": plan.model,
        "system": system,
        "temperature": temperature,
        "search": search,
        "auth": plan.auth,
        "mode": plan.mode,
        "timeout": plan.timeout,
        **kwargs,
    }
    if plan.effort_applied:
        call_kwargs["reasoning_effort"] = plan.effort_applied
    if schema is not None:
        call_kwargs["response_format"] = schema
    if caller:
        call_kwargs["caller"] = caller
    # Drop None values so chat()/LLM don't store junk
    call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

    try:
        response = _chat(full_prompt, **call_kwargs)
    except Exception as exc:
        status, message = classify_dispatch_error(exc)
        return DispatchResult(
            status=status,
            retryable=RETRYABLE_STATUSES.get(status, False),
            provider=plan.provider,
            model=plan.model,
            transport=plan.transport,
            auth=plan.auth,
            mode=plan.mode,
            effort_applied=plan.effort_applied,
            warnings=warnings,
            error_type=status,
            error_message=message,
        )

    text = str(getattr(response, "content", "") or "")
    latency = float(getattr(response, "latency", 0.0) or 0.0)
    usage = _usage_dict(response)
    if not text.strip():
        return DispatchResult(
            status="empty_output",
            retryable=True,
            text=text,
            provider=getattr(response, "provider", plan.provider) or plan.provider,
            model=getattr(response, "model", plan.model) or plan.model,
            transport=plan.transport,
            auth=plan.auth,
            mode=plan.mode,
            effort_applied=plan.effort_applied,
            warnings=warnings,
            usage=usage,
            latency=latency,
            error_type="empty_output",
            error_message="empty model output",
            response=response,
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    return DispatchResult(
        status="ok",
        retryable=False,
        text=text,
        provider=getattr(response, "provider", plan.provider) or plan.provider,
        model=getattr(response, "model", plan.model) or plan.model,
        transport=plan.transport,
        auth=plan.auth,
        mode=plan.mode,
        effort_applied=plan.effort_applied,
        warnings=warnings,
        usage=usage,
        latency=latency,
        response=response,
    )
