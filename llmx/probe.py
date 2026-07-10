"""Typed, cached live checks for subscription-backed model routes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

from .api import LLM
from .dispatch_plan import DispatchPlan, build_dispatch_plan
from .providers import ApiKeyError, LlmxError, ModelError, QuotaError


PROBE_SCHEMA_VERSION = "llmx-live-probe.v1"
PROBE_PROMPT_VERSION = "entitlement-ok-v1"
DEFAULT_CACHE_TTL_SECONDS = 15 * 60
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llmx" / "probes"

ProbeVerdict = Literal["available", "unavailable", "indeterminate"]


@dataclass(frozen=True)
class ProbeResult:
    schema_version: str
    provider: str
    model: str
    auth: Literal["subscription"]
    transport: str
    checked_at: str
    expires_at: str
    verdict: ProbeVerdict
    cached: bool
    latency_seconds: float
    exit_code: int
    error_type: str | None
    status_code: int
    detail: str | None
    response_nonempty: bool
    response_exact_ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> ProbeResult:
        if not isinstance(payload, dict):
            raise ValueError("probe cache is not a JSON object")
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("probe cache fields do not match the current schema")
        string_fields = {
            "schema_version",
            "provider",
            "model",
            "auth",
            "transport",
            "checked_at",
            "expires_at",
            "verdict",
        }
        if any(not isinstance(payload[field], str) for field in string_fields):
            raise ValueError("probe cache string field has the wrong type")
        if any(
            type(payload[field]) is not bool
            for field in {"cached", "response_nonempty", "response_exact_ok"}
        ):
            raise ValueError("probe cache boolean field has the wrong type")
        if any(
            type(payload[field]) is not int for field in {"exit_code", "status_code"}
        ):
            raise ValueError("probe cache integer field has the wrong type")
        if isinstance(payload["latency_seconds"], bool) or not isinstance(
            payload["latency_seconds"], (int, float)
        ):
            raise ValueError("probe cache latency has the wrong type")
        if any(
            payload[field] is not None and not isinstance(payload[field], str)
            for field in {"error_type", "detail"}
        ):
            raise ValueError("probe cache optional string field has the wrong type")
        result = cls(**payload)
        if result.schema_version != PROBE_SCHEMA_VERSION:
            raise ValueError("probe cache schema version mismatch")
        if result.auth != "subscription":
            raise ValueError("probe cache is not subscription-scoped")
        if result.verdict not in {"available", "unavailable", "indeterminate"}:
            raise ValueError("probe cache verdict is invalid")
        if result.cached:
            raise ValueError("probe cache persisted a derived cached=true projection")
        _parse_timestamp(result.checked_at)
        _parse_timestamp(result.expires_at)
        return result


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("probe timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _cache_key(plan: DispatchPlan, *, cache_ttl_seconds: int) -> str:
    identity = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "prompt_version": PROBE_PROMPT_VERSION,
        "provider": plan.provider,
        "model": plan.model,
        "auth": plan.auth,
        "transport": plan.transport,
        "mode": plan.mode,
        "lite": plan.lite,
        "timeout": plan.timeout,
        "cache_ttl_seconds": cache_ttl_seconds,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(plan: DispatchPlan, cache_dir: Path, *, cache_ttl_seconds: int) -> Path:
    return cache_dir / f"{_cache_key(plan, cache_ttl_seconds=cache_ttl_seconds)}.json"


def _load_cached_result(
    plan: DispatchPlan,
    *,
    cache_dir: Path,
    cache_ttl_seconds: int,
    now: datetime,
) -> ProbeResult | None:
    path = _cache_path(plan, cache_dir, cache_ttl_seconds=cache_ttl_seconds)
    try:
        result = ProbeResult.from_dict(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if (
        result.provider != plan.provider
        or result.model != plan.model
        or result.transport != plan.transport
        or _parse_timestamp(result.expires_at) <= now
    ):
        return None
    return replace(result, cached=True)


def _write_cached_result(
    result: ProbeResult,
    *,
    plan: DispatchPlan,
    cache_dir: Path,
    cache_ttl_seconds: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(plan, cache_dir, cache_ttl_seconds=cache_ttl_seconds)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=cache_dir,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _probe_plan(*, provider: str, model: str, timeout: int) -> DispatchPlan:
    plan = build_dispatch_plan(
        provider=provider,
        model=model,
        reasoning_effort="low",
        timeout=timeout,
        lite=None,
        mode="chat",
        auth="subscription",
        subscription=False,
        api_only=None,
        use_old=False,
    )
    if not plan.subscription or plan.auth != "subscription":
        raise ValueError("probe route did not resolve to subscription auth")
    if not plan.transport.endswith("-cli"):
        raise ValueError(
            f"probe refuses metered transport {plan.transport!r}; choose a subscription model"
        )
    if plan.cli_fallback_reason:
        raise ValueError(
            "probe route requires an API fallback and is not subscription-safe: "
            f"{plan.cli_fallback_reason}"
        )
    return plan


def _verdict_for_error(error: LlmxError) -> ProbeVerdict:
    if isinstance(error, (QuotaError, ApiKeyError, ModelError)):
        return "unavailable"
    return "indeterminate"


def run_subscription_probe(
    *,
    provider: str,
    model: str,
    timeout: int = 120,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    refresh: bool = False,
    cache_dir: Path | None = None,
    now_fn: Callable[[], datetime] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    client_factory: Callable[..., LLM] = LLM,
) -> ProbeResult:
    """Run one fixed, bounded subscription call or return its unexpired typed result."""
    if timeout < 1:
        raise ValueError("probe timeout must be positive")
    if cache_ttl_seconds < 0:
        raise ValueError("probe cache TTL cannot be negative")

    plan = _probe_plan(provider=provider, model=model, timeout=timeout)
    raw_now = (now_fn or (lambda: datetime.now(UTC)))()
    if raw_now.tzinfo is None:
        raise ValueError("probe clock must be timezone-aware")
    now = raw_now.astimezone(UTC)
    resolved_cache_dir = cache_dir or Path(
        os.environ.get("LLMX_PROBE_CACHE_DIR", DEFAULT_CACHE_DIR)
    )
    if not refresh and cache_ttl_seconds > 0:
        cached = _load_cached_result(
            plan,
            cache_dir=resolved_cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            now=now,
        )
        if cached is not None:
            return cached

    started = monotonic_fn()
    try:
        client = client_factory(
            provider=plan.provider,
            model=plan.model,
            auth="subscription",
            mode="chat",
        )
        response = client.chat(
            "Reply exactly OK.",
            reasoning_effort="low",
            timeout=timeout,
        )
        response_text = response.content.strip()
        result = ProbeResult(
            schema_version=PROBE_SCHEMA_VERSION,
            provider=plan.provider,
            model=plan.model,
            auth="subscription",
            transport=plan.transport,
            checked_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=cache_ttl_seconds)).isoformat(),
            verdict="available" if response_text else "indeterminate",
            cached=False,
            latency_seconds=round(max(0.0, monotonic_fn() - started), 3),
            exit_code=0 if response_text else 1,
            error_type=None if response_text else "empty_response",
            status_code=0,
            detail=None
            if response_text
            else "subscription probe returned empty output",
            response_nonempty=bool(response_text),
            response_exact_ok=response_text == "OK",
        )
    except LlmxError as error:
        result = ProbeResult(
            schema_version=PROBE_SCHEMA_VERSION,
            provider=plan.provider,
            model=plan.model,
            auth="subscription",
            transport=plan.transport,
            checked_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=cache_ttl_seconds)).isoformat(),
            verdict=_verdict_for_error(error),
            cached=False,
            latency_seconds=round(max(0.0, monotonic_fn() - started), 3),
            exit_code=error.exit_code,
            error_type=error.error_type,
            status_code=error.status_code,
            detail=str(error),
            response_nonempty=False,
            response_exact_ok=False,
        )
    except Exception as error:
        result = ProbeResult(
            schema_version=PROBE_SCHEMA_VERSION,
            provider=plan.provider,
            model=plan.model,
            auth="subscription",
            transport=plan.transport,
            checked_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=cache_ttl_seconds)).isoformat(),
            verdict="indeterminate",
            cached=False,
            latency_seconds=round(max(0.0, monotonic_fn() - started), 3),
            exit_code=1,
            error_type=type(error).__name__,
            status_code=0,
            detail=str(error),
            response_nonempty=False,
            response_exact_ok=False,
        )

    if cache_ttl_seconds > 0:
        _write_cached_result(
            result,
            plan=plan,
            cache_dir=resolved_cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    return result
