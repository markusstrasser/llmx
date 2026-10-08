"""CLI-backed providers: codex-cli, claude-cli, cursor-cli, grok-cli.

Shell out to Codex CLI / Claude Code instead of API for subscription pricing.
Fall back to metered API only on auth=api routes — subscription forbids silent billing.

Gemini routing was removed 2026-05-31: Google retired the free Gemini CLI
consumer tier (Antigravity migration, hard cutoff 2026-06-18), and the
replacement `agy` CLI can't pin a model headlessly (print mode is locked to
the account's default). Google now routes straight to the paid Gemini
Developer API. See ~/.claude/rules/llmx-routing.md.

CLI flag reference (verified 2026-03):
  codex exec [PROMPT] [-m <model>] [--output-schema schema.json]
         reads stdin when PROMPT is "-" or omitted
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias

from .logger import logger
from .model_ids import CURSOR_GROK_MODELS, GROK_BUILD_MODELS
from .providers import (
    CLAUDE_CLI_MAX_OUTPUT_TOKENS,
    ApiKeyError,
    LlmxError,
    ModelError,
    QuotaError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError_,
)


@dataclass(frozen=True)
class CliBackendFailure:
    """Typed CLI transport failure before conversion to the public error API."""

    kind: type[LlmxError]
    status: int
    detail: str

    def fallback_reason(self) -> str:
        status = f" status={self.status}" if self.status else ""
        return f"{self.kind.__name__}{status}: {self.detail}"


CliBackendResult: TypeAlias = str | CliBackendFailure

# CLI provider configs — kept separate from PROVIDER_CONFIGS (different lifecycle)
CLI_PROVIDERS = {
    "codex-cli": {
        "binary": "codex",
        "api_fallback": "openai",
    },
    "claude-cli": {
        "binary": "claude",
        "api_fallback": "anthropic",
    },
    # Cursor CLI (cursor-agent) in headless `-p` mode with the user's Cursor
    # app subscription auth. NO api_fallback: Cursor-native slugs (grok-4.7-*)
    # have no $0 API path, and proxied models (claude/gpt/gemini via the sub) have
    # no $0 API path either — a feature the CLI can't do raises, never silently
    # routes to a paid API. Always-on (not lite-gated): cursor-agent --mode ask
    # is already lightweight (~1s startup, no MCP).
    "cursor-cli": {
        "binary": "cursor-agent",
        "api_fallback": None,
    },
    "grok-cli": {
        "binary": "grok",
        "api_fallback": None,
    },
}

# Prefer subscription CLIs for logical providers when available.
# Google has NO CLI alias — it always routes to the paid Gemini API (the free
# gemini-cli consumer tier was retired 2026-05-31). OpenAI routes to codex-cli
# only in --lite/--subscription mode. Anthropic defaults to claude-cli
# subscription (OAuth, API key stripped) unless -p anthropic-direct / api_only.
# `cursor` always resolves to cursor-cli (subscription-only, no API path), so
# it lives in the non-lite alias map — reachable in every mode, not just --lite.
CLI_PROVIDER_ALIASES = {"cursor": "cursor-cli", "grok": "grok-cli"}
CLI_PROVIDER_ALIASES_LITE = {
    "openai": "codex-cli",
    "anthropic": "claude-cli",
    "cursor": "cursor-cli",
    "grok": "grok-cli",
}

# Logical identity is separate from API fallback policy. Subscription-only CLIs
# need a default model without acquiring a paid fallback.
CLI_LOGICAL_PROVIDERS = {
    "codex-cli": "openai",
    "claude-cli": "anthropic",
    "cursor-cli": "cursor",
    "grok-cli": "grok",
}

# Lite cwd is split between two locations:
#
#   Skeleton (package, read-only):   llmx/lite/{bare,research}/
#       Encapsulated description of what each lite mode looks like.
#       Ships with llmx, ignored at runtime — the package stays clean
#       even when CLIs scribble session state into their cwd.
#
#   Runtime (cache, read-write):     ~/.cache/llmx/lite/{bare,research}/
#       Actual cwd handed to the CLI subprocess. Auto-created from the
#       skeleton on first use, idempotent across runs. claude-cli writes
#       its <cwd>/.claude/current-session-id marker here even with
#       --no-session-persistence, so we keep that pollution out of the
#       package by routing it to a cache location the user can wipe.
#
# Per-CLI isolation (no project AGENTS.md / GEMINI.md / CLAUDE.md autoload,
# no user config.toml / settings.json) comes from CLI flags — not from
# HOME / CODEX_HOME redirects. Auth still flows through the user's normal
# HOME, where each CLI keeps it.
_LITE_PACKAGE_SKEL = Path(__file__).resolve().parent / "lite"
_LITE_RUNTIME_ROOT = Path(os.path.expanduser("~/.cache/llmx/lite"))
_LITE_MODES = ("bare", "research")

LITE_PROMPT_PREFIX = {
    "bare": (
        "[Environment: no tools, no web search, no file access. "
        "Answer from training knowledge only.]\n\n"
    ),
    "research": (
        "[Environment: no general web search. A 'research' MCP is available "
        "for academic paper / preprint search and web archive lookups "
        "(search_papers, search_preprints, verify_claim, deep_research, etc.). "
        "No other tools.]\n\n"
    ),
}


def _research_mcp_dir() -> Optional[str]:
    """Resolve the research-mcp project dir.

    Order: $LLMX_RESEARCH_MCP_DIR → developer default at ~/Projects/research-mcp
    if it exists → None. Returning None makes --lite research raise
    LiteEnvironmentError so the user gets a setup hint instead of a
    cryptic 'uv run --directory' failure inside the subprocess.
    """
    env_dir = os.environ.get("LLMX_RESEARCH_MCP_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    if env_dir:
        # User set the env var but the path is wrong — surface that.
        return env_dir
    default = os.path.expanduser("~/Projects/research-mcp")
    if os.path.isdir(default):
        return default
    return None


def _research_mcp_args() -> list[str]:
    """uv invocation args for the research MCP. Raises if not configured."""
    target = _research_mcp_dir()
    if not target:
        raise LiteEnvironmentError(
            "--lite research needs the research-mcp project. Set "
            "LLMX_RESEARCH_MCP_DIR=/path/to/research-mcp or clone it to "
            "~/Projects/research-mcp."
        )
    return ["run", "--directory", target, "research-mcp"]


# Lite mode is restricted to frontier models. Anthropic routes via
# claude-cli (Claude Code) in headless `-p` mode with OAuth subscription auth
# (ANTHROPIC_API_KEY unset, --disable-slash-commands, empty mcp-config or
# research-mcp only); gpt-6-* route via codex-cli. Allowlist = Pareto frontier
# only (2026-09-25 prune retired gpt-5.6-*, claude-fable-5, claude-opus-4-8,
# gemini-3-flash-preview, grok-4.6 lanes; 2026-10-07 retired composer-2.5*).
LITE_ALLOWED_MODELS = {
    "gpt-6-astra",
    "gpt-6",  # alias → astra
    # 2026-09-25: codex-cli subscription serves both (live `codex exec -m` probe).
    "gpt-6-sol",
    "gpt-6-luna",
    # Opus 5: cyber / dual-use-bio fallback niche.
    "claude-opus-5",
    # 2026-09-01: Fable 5.1 (same transport; interactive default on the Max
    # account since launch day). Plan-vs-usage-credit billing is per plan/seat —
    # Claude Code's /model picker says "Requires usage credits" when it applies.
    "claude-fable-5-1",
    # 2026-09-22: Opus 5.5 (same transport; Claude Code 2.1.280's default Opus).
    # Thinking can't be disabled — effort is the only control (API default medium).
    "claude-opus-5-5",
    # Cursor subscription pool: exact Grok 4.7 slugs (composer-2.5 retired 2026-10-07).
    *CURSOR_GROK_MODELS,
}


def lite_model_allowed(model: Optional[str], *, transport: Optional[str] = None) -> bool:
    """Return True if the resolved model is on the lite allowlist.

    Lenient match — `gemini-3.1-pro` and `gemini-3.1-pro-preview` both pass.
    """
    if not model:
        return False
    # The bare xAI id is admitted only on Grok Build's subscription transport.
    # Adding it to the shared set would also widen Cursor's exact-slug gate.
    if model in GROK_BUILD_MODELS:
        return transport == "grok-cli"
    # Any other grok id must be an exact Cursor slug: 4.7 slugs carry no
    # `cursor-` prefix, so the prefix loop below would admit invented ones.
    if model.startswith(("cursor-grok-", "grok-")):
        return model in CURSOR_GROK_MODELS
    for allowed in LITE_ALLOWED_MODELS:
        base = allowed.removesuffix("-preview")
        if model == allowed or model == base or model.startswith(base + "-"):
            return True
    return False


# Max bytes for command-line argument before switching to stdin.
# macOS ARG_MAX is ~260KB but shells/tools choke earlier.
_ARG_MAX_BYTES = 100_000
_CODEX_SESSIONS_DIR = Path(os.path.expanduser("~/.codex/sessions"))
# `codex exec` prints this line in its stderr header (observed v0.156.1); the id also
# ends the rollout's filename, rollout-<timestamp>-<id>.jsonl.
_CODEX_SESSION_ID_RE = re.compile(r"^session id:\s*([0-9a-fA-F-]{36})\s*$", re.MULTILINE)


def configured_cli_provider(provider: str, lite: Optional[str] = None) -> Optional[str]:
    """Return the CLI backend associated with a provider, if any."""
    if provider in CLI_PROVIDERS:
        return provider
    aliases = CLI_PROVIDER_ALIASES_LITE if lite else CLI_PROVIDER_ALIASES
    return aliases.get(provider)


def binary_available(provider: str) -> bool:
    """Return whether the CLI binary for this provider is available."""
    cli_provider = configured_cli_provider(provider) or provider
    config = CLI_PROVIDERS.get(cli_provider)
    if not config:
        return False
    return shutil.which(config["binary"]) is not None


def preferred_cli_provider(
    provider: str,
    lite: Optional[str] = None,
    *,
    subscription: bool = False,
) -> Optional[str]:
    """Return the CLI backend to prefer for a provider.

    Explicit CLI providers always resolve, even if the binary is missing, so callers can
    surface a precise fallback reason. Logical providers (openai/google) only resolve when
    the corresponding CLI is installed.

    `lite` ('bare' or 'research') routes logical providers through an isolated
    CLI profile. ``subscription=True`` routes the same logical providers through
    their native CLI without implying isolation; this is what workspace agent
    mode uses.
    """
    cli_provider = configured_cli_provider(
        provider,
        lite=lite or ("bare" if subscription else None),
    )
    if not cli_provider:
        return None
    if provider in CLI_PROVIDERS:
        return cli_provider
    # Keep subscription-only logical providers selected even when their binary
    # is absent so the caller reports the actual CLI failure. Returning None
    # here fabricates a nonexistent metered API route.
    if CLI_PROVIDERS[cli_provider]["api_fallback"] is None:
        return cli_provider
    return cli_provider if binary_available(cli_provider) else None


def needs_api_fallback(
    provider: str,
    schema,
    system: Optional[str],
    search: bool,
    stream: bool,
    reasoning_effort: Optional[str],
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Check if request requires features the CLI can't handle.

    Returns reason string if fallback needed, None if CLI can handle it.
    """
    config = CLI_PROVIDERS[provider]
    binary = config["binary"]

    if not shutil.which(binary):
        return f"{binary} not found in PATH"
    # codex-cli WAS exempted here (it has `codex exec --output-schema`), but that path is broken on
    # codex v0.140.0 — it exits 1 with a generic CLI error instead of honoring the schema (reproduced
    # 2026-06-18: GPT `--subscription --schema`). Treat schema as CLI-unsupported for ALL CLIs:
    # this falls back to the API (which does structured output) on the metered lane, or raises a clear
    # "use --auth api" error on subscription — instead of an opaque codex failure. Re-add the
    # `and provider != "codex-cli"` guard if/when codex --output-schema is fixed upstream.
    if schema:
        return "structured output not supported by CLI"
    # system messages: folded into prompt as <system> XML tag (no CLI flag needed)
    if search:
        return "web search not supported by CLI"
    if stream:
        return "streaming not supported by CLI"
    if max_tokens:
        return "max_tokens not supported by subscription CLI transports (drop --max-tokens or use --auth api)"
    # Reasoning effort is never an API-fallback trigger. Grok forwards its
    # mapped value; other CLIs either map it later or use their native default.

    return None


def subscription_route(*, auth: Optional[str] = None, lite: Optional[str] = None) -> bool:
    """True when the caller chose subscription billing (CLI OAuth / app sub)."""
    return auth == "subscription" or lite in _LITE_MODES


def resolve_cli_api_fallback(
    cli_provider: str,
    *,
    auth: Optional[str] = None,
    lite: Optional[str] = None,
    reason: str,
) -> str:
    """Return API provider for CLI→API fallback, or raise if blocked."""
    if subscription_route(auth=auth, lite=lite):
        raise RuntimeError(
            f"{cli_provider} failed ({reason}) but auth=subscription forbids "
            f"metered API fallback. Fix the CLI issue or pass auth=api."
        )
    api_provider = CLI_PROVIDERS[cli_provider]["api_fallback"]
    if api_provider is None:
        raise ValueError(
            f"{cli_provider} cannot handle this request ({reason}) "
            f"and has no API fallback. Drop the unsupported option "
            f"(e.g. --schema/--search/--stream) or pick a different model."
        )
    return api_provider


class LiteEnvironmentError(RuntimeError):
    """Raised when --lite mode {!r} doesn't have a packaged cwd.

    Should never fire for shipped modes (bare/research) since their dirs
    travel with the package. Catches typos and forward-compat slips.
    """


def _lite_cwd(lite: str) -> str:
    """Return (auto-creating) the runtime cwd for the requested lite mode.

    Bootstraps ~/.cache/llmx/lite/{mode}/ from the package skeleton at
    llmx/lite/{mode}/ on first call. Both shipped modes (bare, research)
    have empty skeletons today, so the bootstrap is just mkdir -p — but
    the indirection lets us add preloaded files (extra MCP configs,
    pinned context fragments) to the package skeleton later without
    pushing those into a user-owned dir.
    """
    if lite not in _LITE_MODES:
        raise LiteEnvironmentError(f"--lite {lite!r} unknown. Supported: {list(_LITE_MODES)}")
    skel = _LITE_PACKAGE_SKEL / lite
    if not skel.is_dir():  # only fires on broken installs
        raise LiteEnvironmentError(
            f"--lite mode {lite!r} skeleton {skel} missing — llmx install corrupt?"
        )
    runtime = _LITE_RUNTIME_ROOT / lite
    caller = Path.cwd()
    if _is_llmx_cache_path(caller):
        runtime.mkdir(parents=True, exist_ok=True)
        return str(runtime)
    return str(_caller_cache_subdir(runtime, caller))


_CURSOR_RUNTIME_DIR = Path(os.path.expanduser("~/.cache/llmx/cursor"))
_GROK_RUNTIME_DIR = Path(os.path.expanduser("~/.cache/llmx/grok"))
_LLMX_CACHE_ROOT = Path(os.path.expanduser("~/.cache/llmx"))
_DISPATCH_ATTRIBUTION = _LLMX_CACHE_ROOT / "dispatch-attribution.jsonl"
_CALLER_MARKER = ".llmx-caller-cwd"


def _is_llmx_cache_path(path: str | Path) -> bool:
    """True when path is under ~/.cache/llmx (resolved), not a substring false-positive."""
    try:
        resolved = Path(path).expanduser().resolve()
        root = _LLMX_CACHE_ROOT.resolve()
        return resolved == root or root in resolved.parents
    except (OSError, RuntimeError, ValueError):
        return False


def _caller_cache_subdir(base: Path, caller: Path) -> Path:
    """Per-caller empty cache dir so concurrent dispatches don't share one cwd.

    Shared ~/.cache/llmx/cursor (or lite/bare) cannot attribute two in-flight
    projects — Opus 2026-07-12 FIX-THEN-LAND. Hash keeps paths short + stable.
    """
    import hashlib

    digest = hashlib.sha1(str(caller.resolve()).encode()).hexdigest()[:12]
    d = base / digest
    d.mkdir(parents=True, exist_ok=True)
    try:
        (d / _CALLER_MARKER).write_text(str(caller.resolve()) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug(f"[cli] caller-marker write failed: {exc}")
    return d


def _record_dispatch_attribution(*, cli_cwd: str, caller_cwd: str) -> None:
    """Append durable caller→cache-cwd row (bounded; agentlogs prefers cwd marker)."""
    import json
    from datetime import datetime, timezone

    try:
        _LLMX_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cli_cwd": str(Path(cli_cwd).expanduser().resolve()),
            "caller_cwd": str(Path(caller_cwd).expanduser().resolve()),
            "pid": os.getpid(),
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _DISPATCH_ATTRIBUTION.open("a", encoding="utf-8") as fh:
            fh.write(line)
        # Soft rotation: keep last ~5000 lines so pid reuse can't resurrect weeks-old rows.
        try:
            raw = _DISPATCH_ATTRIBUTION.read_text(encoding="utf-8").splitlines()
            if len(raw) > 5000:
                _DISPATCH_ATTRIBUTION.write_text("\n".join(raw[-4000:]) + "\n", encoding="utf-8")
        except OSError:
            pass
    except OSError as exc:
        logger.debug(f"[cli] dispatch-attribution write failed: {exc}")


def _cursor_cwd() -> str:
    """Neutral empty cwd for cursor-agent, scoped per caller workspace.

    cursor-agent reads the workspace it runs in (rules, AGENTS.md, file tree).
    We still avoid folding the caller's tree into context by using an empty
    cache dir — but the dir is per-caller so agentlogs can attribute the session.
    """
    caller = Path.cwd()
    if _is_llmx_cache_path(caller):
        _CURSOR_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        return str(_CURSOR_RUNTIME_DIR)
    return str(_caller_cache_subdir(_CURSOR_RUNTIME_DIR, caller))


def _grok_cwd() -> str:
    """Neutral empty cwd for Grok Build chat, scoped per caller workspace."""
    caller = Path.cwd()
    if _is_llmx_cache_path(caller):
        _GROK_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        return str(_GROK_RUNTIME_DIR)
    return str(_caller_cache_subdir(_GROK_RUNTIME_DIR, caller))


def _codex_rollout_snapshot(root: Path = _CODEX_SESSIONS_DIR) -> dict[Path, int]:
    """Capture known Codex rollout mtimes before launching `codex exec`."""
    if not root.exists():
        return {}
    out: dict[Path, int] = {}
    try:
        for path in root.glob("*/*/*/rollout-*.jsonl"):
            try:
                out[path] = path.stat().st_mtime_ns
            except OSError:
                continue
    except OSError:
        return {}
    return out


def _read_codex_rollout_usage(
    path: Path,
) -> tuple[dict[str, Optional[int]], Optional[str]]:
    """Read the last token_count event from a Codex rollout JSONL file."""
    last_usage = None
    try:
        with path.open() as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = event.get("payload") if isinstance(event, dict) else None
                if not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue
                info = payload.get("info") or {}
                usage = info.get("last_token_usage") or info.get("total_token_usage")
                if isinstance(usage, dict):
                    last_usage = usage
    except OSError as exc:
        return _null_codex_usage(), f"could not read codex rollout {path}: {exc}"

    if not last_usage:
        return _null_codex_usage(), f"codex rollout had no token_count event: {path}"

    return {
        "prompt_tokens": _int_or_none(last_usage.get("input_tokens")),
        "completion_tokens": _int_or_none(last_usage.get("output_tokens")),
        "reasoning_tokens": _int_or_none(last_usage.get("reasoning_output_tokens")),
        "cached_tokens": _int_or_none(last_usage.get("cached_input_tokens")),
    }, None


def _int_or_none(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(mapping: dict, *keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _null_codex_usage() -> dict[str, Optional[int]]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
    }


def _codex_session_id(stderr: Optional[str]) -> Optional[str]:
    """Return the session id from the `codex exec` stderr header, if printed."""
    if not stderr:
        return None
    match = _CODEX_SESSION_ID_RE.search(stderr)
    return match.group(1).lower() if match else None


def _latest_codex_rollout_usage(
    before: dict[Path, int],
    *,
    started_at: float,
    session_id: Optional[str] = None,
    root: Path = _CODEX_SESSIONS_DIR,
) -> tuple[dict[str, Optional[int]], Optional[str]]:
    """Find this codex-cli call's rollout and parse its tokens.

    The session id from the stderr header names the rollout exactly. Without it,
    time is the only clue, and it is safe only when a single rollout appeared
    since launch: parallel calls each create one (the newest file can be another
    call's, still running), and a parent Codex session keeps updating its own.
    Otherwise usage stays unknown, with a note, rather than taking another
    call's tokens.
    """
    if not root.exists():
        return _null_codex_usage(), f"codex sessions dir not found: {root}"

    if session_id:
        try:
            matches = sorted(root.glob(f"*/*/*/rollout-*-{session_id}.jsonl"))
        except OSError as exc:
            return _null_codex_usage(), f"could not list codex rollouts: {exc}"
        if len(matches) != 1:
            return _null_codex_usage(), f"{len(matches)} codex rollouts match session {session_id}"
        return _read_codex_rollout_usage(matches[0])

    new_files: list[Path] = []
    cutoff_ns = int((started_at - 2.0) * 1_000_000_000)
    try:
        paths = list(root.glob("*/*/*/rollout-*.jsonl"))
    except OSError as exc:
        return _null_codex_usage(), f"could not list codex rollouts: {exc}"

    for path in paths:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            continue
        if mtime_ns >= cutoff_ns and path not in before:
            new_files.append(path)

    if len(new_files) != 1:
        return _null_codex_usage(), (
            f"no session id in codex stderr and {len(new_files)} new codex rollouts "
            "since launch; usage unknown rather than guessed"
        )
    usage, note = _read_codex_rollout_usage(new_files[0])
    return usage, note or "attributed by time: no session id in codex stderr"


_QUOTA_MARKERS = (
    "billing",
    # Subscription plan limits that reset on a clock. codex-cli prints "You've hit your usage
    # limit ... try again at 4:21 PM." (observed 2026-09-16); Claude surfaces "You've hit your
    # session limit · resets 3pm". Retrying before the reset only burns time, so treat as quota;
    # the detail keeps the reset time for the caller.
    "hit your session limit",
    "hit your usage limit",
    "monthly spend limit",
    "monthly usage limit",
    "credit balance",
    "insufficient quota",
    "insufficient_quota",
    "spend limit",
    "usage cap",
)
_TIMEOUT_MARKERS = ("deadline exceeded", "deadline_exceeded", "timed out", "timeout")
_MODEL_MARKERS = (
    "invalid model",
    "model does not exist",
    "model is not available",
    "model isn't available",
    "model not found",
    "unknown model",
    "unsupported model",
)
_AUTH_MARKERS = (
    "authentication",
    "invalid api key",
    "invalid oauth",
    "invalid_api_key",
    "login required",
    "not logged in",
    "oauth token",
    "please log in",
    "token expired",
    "unauthorized",
)
_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "requests per minute",
    "tokens per minute",
    "too many requests",
)
_SERVICE_UNAVAILABLE_MARKERS = (
    "at capacity",
    "overloaded",
    "temporarily unavailable",
)


def _cli_failure_detail(stderr: str, stdout: str) -> str:
    """Failure text for a non-zero CLI exit: explicit ERROR lines first, else the output tail.

    codex-cli prints its startup banner (version, workdir, model, sandbox) on stderr before any
    work and its failure last. Keeping the head of stderr kept only the banner, so a subscription
    usage limit classified as a generic failure and callers recorded empty output (2026-09-16).
    """
    for stream in (stderr, stdout):
        error_lines = [
            line.strip() for line in (stream or "").splitlines() if line.lstrip().startswith("ERROR")
        ]
        if error_lines:
            return " | ".join(dict.fromkeys(error_lines))[:500]
    stderr_tail = (stderr or "").strip()[-300:]
    stdout_tail = (stdout or "").strip()[-200:]
    return stderr_tail or stdout_tail or "unknown error"


def _classify_cli_failure(detail: str, status: int = 0) -> CliBackendFailure:
    """Classify transport detail, using text to disambiguate overloaded statuses."""
    normalized = detail.casefold()
    has_rate_limit_marker = any(marker in normalized for marker in _RATE_LIMIT_MARKERS)
    if (
        status == 402
        or any(marker in normalized for marker in _QUOTA_MARKERS)
        or ("quota" in normalized and not has_rate_limit_marker)
    ):
        kind = QuotaError
    elif status in {408, 504} or any(marker in normalized for marker in _TIMEOUT_MARKERS):
        kind = TimeoutError_
    elif status == 404 or any(marker in normalized for marker in _MODEL_MARKERS):
        kind = ModelError
    elif status in {401, 403} or any(marker in normalized for marker in _AUTH_MARKERS):
        kind = ApiKeyError
    elif status == 429 or has_rate_limit_marker:
        kind = RateLimitError
    elif status in {500, 502, 503, 529} or any(
        marker in normalized for marker in _SERVICE_UNAVAILABLE_MARKERS
    ):
        kind = ServiceUnavailableError
    else:
        kind = LlmxError
    return CliBackendFailure(kind=kind, status=status, detail=detail)


def _claude_error_detail(value) -> str:
    if isinstance(value, str) and value:
        return value
    if value is not None:
        try:
            return json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            return repr(value)
    return "Claude CLI reported an error without detail"


def _claude_payload_reports_error(stdout: str) -> bool:
    """Return whether stdout is structured Claude JSON with an error result."""
    try:
        events = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return False
    if isinstance(events, dict):
        events = [events]
    return isinstance(events, list) and any(
        isinstance(event, dict) and event.get("type") == "result" and bool(event.get("is_error"))
        for event in events
    )


def _claude_final_assistant_text(
    events: list,
) -> tuple[Optional[str], Optional[str]]:
    """Reconstruct the final Claude message from verbose assistant events.

    Claude Code 2.1.210's non-verbose ``result`` projection keeps only the
    final content block.  Verbose JSON retains the assistant messages, so the
    transport can independently reconstruct the response and refuse a lossy
    projection.  A message can span multiple events; preserve both event and
    content-block order for the final message id.
    """
    assistant_messages: list[tuple[int, str, dict]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            return None, (
                f"Claude CLI verbose assistant event {event_index} contained no message object"
            )
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id:
            return None, (
                f"Claude CLI verbose assistant event {event_index} contained no message id"
            )
        assistant_messages.append((event_index, message_id, message))

    if not assistant_messages:
        return None, "Claude CLI verbose JSON contained no assistant event"

    final_message_id = assistant_messages[-1][1]
    text_parts: list[str] = []
    for event_index, message_id, message in assistant_messages:
        if message_id != final_message_id:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return None, (
                f"Claude CLI verbose assistant event {event_index} contained no content block list"
            )
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                return None, (
                    "Claude CLI verbose assistant event "
                    f"{event_index} contained malformed content block {block_index}"
                )
            if block.get("type") != "text":
                continue
            block_text = block.get("text")
            if not isinstance(block_text, str):
                return None, (
                    "Claude CLI verbose assistant event "
                    f"{event_index} text block {block_index} contained no text"
                )
            text_parts.append(block_text)

    if not text_parts:
        return None, (
            "Claude CLI final verbose assistant message "
            f"{final_message_id!r} contained no text blocks"
        )
    return "".join(text_parts), None


def _claude_injected_turn(events: list) -> Optional[str]:
    """Text of the first user turn injected after the model had answered, if any.

    After the first assistant event, a one-shot call only sees tool results. A
    text turn means something forced a continuation, such as a blocking Stop
    hook ("Stop hook feedback: ..."), and the final message then answers that
    turn instead of the caller's prompt.
    """
    answered = False
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "assistant":
            answered = True
            continue
        if not answered or event.get("type") != "user":
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return content
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    return None


def _parse_claude_json(
    stdout: str,
    *,
    allow_continuation: bool = False,
) -> tuple[CliBackendResult, Optional[dict]]:
    """Unwrap verbose Claude JSON into text or a typed integrity failure.

    text = the result event's `result`, accepted only when it exactly matches
    the final assistant message reconstructed from verbose content blocks.
    usage = real tokens + API-equivalent total_cost_usd (present even on subscription).
    Structured errors preserve their kind, API status, and exact result detail.
    Unless `allow_continuation` (workspace agent mode, where hooks legitimately
    steer the run), a turn injected after the answer refuses the response.
    """
    try:
        events = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        return (
            CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail=f"Claude CLI returned invalid JSON: {exc}",
            ),
            None,
        )
    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        return (
            CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail="Claude CLI JSON did not contain a result event list",
            ),
            None,
        )
    assistant_text, assistant_error = _claude_final_assistant_text(events)
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "result":
            continue
        if event.get("is_error"):
            status = _int_or_none(event.get("api_error_status")) or 0
            detail = _claude_error_detail(event.get("result"))
            return _classify_cli_failure(detail, status), None
        result = event.get("result")
        if not isinstance(result, str):
            return (
                CliBackendFailure(
                    kind=LlmxError,
                    status=0,
                    detail="Claude CLI result event contained no response text",
                ),
                None,
            )
        if assistant_error is not None:
            return (
                CliBackendFailure(
                    kind=LlmxError,
                    status=0,
                    detail=assistant_error,
                ),
                None,
            )
        injected = None if allow_continuation else _claude_injected_turn(events)
        if injected is not None:
            return (
                CliBackendFailure(
                    kind=LlmxError,
                    status=0,
                    detail=(
                        "Claude CLI was forced to continue after answering "
                        f"(injected turn: {injected[:160]!r}); the final text answers "
                        "that turn, not the prompt; refusing response"
                    ),
                ),
                None,
            )
        assert assistant_text is not None
        if result != assistant_text:
            omitted_chars = max(len(assistant_text) - len(result), 0)
            if omitted_chars and assistant_text.endswith(result):
                mismatch = "omitted assistant text blocks"
            else:
                mismatch = "disagreed with reconstructed assistant text"
            return (
                CliBackendFailure(
                    kind=LlmxError,
                    status=0,
                    detail=(
                        f"Claude CLI result {mismatch}: "
                        f"reconstructed_chars={len(assistant_text)} "
                        f"result_chars={len(result)} "
                        f"omitted_chars={omitted_chars}; refusing response"
                    ),
                ),
                None,
            )
        raw_usage = event.get("usage") or {}
        model_usage = event.get("modelUsage") or {}
        model_key = next(iter(model_usage), None)
        # Preserve the existing pricing model, but a served-model receipt must
        # be unambiguous provider evidence, never the requested-model fallback.
        served_model = (
            model_key.split("[")[0]
            if isinstance(model_usage, dict)
            and len(model_usage) == 1
            and isinstance(model_key, str)
            and model_key
            else None
        )
        details = (
            raw_usage.get("output_tokens_details")
            or raw_usage.get("completion_tokens_details")
            or {}
        )
        reasoning_tokens = _first_present(
            raw_usage,
            "reasoning_tokens",
            "thinking_tokens",
        )
        if reasoning_tokens is None:
            reasoning_tokens = _first_present(
                details,
                "reasoning_tokens",
                "thinking_tokens",
            )
        # Claude Code's current result JSON exposes input/output/cache tokens.
        # Some versions/models may add thinking/reasoning tokens; if absent,
        # keep null rather than writing 0, which would falsely claim measurement.
        usage = {
            "served_model": served_model,
            "model": model_key.split("[")[0]
            if isinstance(model_key, str)
            else None,  # strip [1m] etc → PRICING key
            "input_tokens": raw_usage.get("input_tokens"),
            "output_tokens": raw_usage.get("output_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "cache_read_input_tokens": raw_usage.get("cache_read_input_tokens"),
            "total_cost_usd": event.get("total_cost_usd"),
        }
        return result, usage
    return (
        CliBackendFailure(
            kind=LlmxError,
            status=0,
            detail="Claude CLI JSON contained no result event",
        ),
        None,
    )


def _parse_grok_json(stdout: str) -> tuple[CliBackendResult, Optional[dict]]:
    """Unwrap Grok Build JSON and preserve its measured subscription usage."""
    try:
        event = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        return (
            CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail=f"Grok CLI returned invalid JSON: {exc}",
            ),
            None,
        )
    if not isinstance(event, dict):
        return (
            CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail="Grok CLI JSON was not an object",
            ),
            None,
        )
    stop_reason = event.get("stopReason")
    if stop_reason != "end_turn":
        return (
            CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail=f"Grok CLI stopped with stopReason={stop_reason!r}",
            ),
            None,
        )
    result = event.get("text")
    if not isinstance(result, str) or not result:
        return (
            CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail="Grok CLI JSON contained no response text",
            ),
            None,
        )
    raw_usage = event.get("usage")
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    model_usage = event.get("modelUsage")
    served_model = next(iter(model_usage), None) if isinstance(model_usage, dict) else None
    usage = {
        "prompt_tokens": raw_usage.get("input_tokens"),
        "completion_tokens": raw_usage.get("output_tokens"),
        "reasoning_tokens": raw_usage.get("reasoning_tokens"),
        "cached_tokens": raw_usage.get("cache_read_input_tokens"),
        "total_cost_usd": event.get("total_cost_usd"),
        "served_model": served_model,
    }
    return result, usage


def cli_chat(
    provider: str,
    prompt: str,
    model: Optional[str],
    timeout: int,
    *,
    schema=None,
    system: Optional[str] = None,
    lite: Optional[str] = None,
    mode: str = "chat",
    reasoning_effort: Optional[str] = None,
) -> CliBackendResult:
    """Execute one-shot chat via CLI binary.

    Returns response text on success or a typed failure for caller policy handling.
    For long prompts (>100KB), pipes via stdin to avoid ARG_MAX limits.

    `lite` ('bare' or 'research') runs the CLI in a stripped-down profile —
    no MCPs (bare) or research-MCP only (research), empty cwd, prompt prefix
    advising the model what's available.

    ``mode='agent'`` is deliberately different: it preserves the caller's cwd,
    project instructions, and native CLI tool surface. Agent mode is explicit
    autonomous execution, so headless CLIs run without interactive approvals.
    """
    # Fold system message into prompt — CLIs don't have a system flag
    if system:
        prompt = f"<system>\n{system}\n</system>\n\n{prompt}"

    # Note: lite-mode prompt prefix is injected in cli.py before chat() so
    # the Anthropic API path gets it too. Don't double-prefix here.
    _ = lite  # used for cwd routing below

    config = CLI_PROVIDERS[provider]
    binary = config["binary"]
    stdin_input = None
    use_stdin = len(prompt.encode()) > _ARG_MAX_BYTES
    temp_schema_path = None
    temp_prompt_path = None
    codex_rollouts_before: dict[Path, int] = {}

    try:
        if binary == "codex":
            codex_rollouts_before = _codex_rollout_snapshot()
            # codex exec [PROMPT] [-m <model>] [--output-schema schema.json]
            cmd = ["codex", "exec", "--skip-git-repo-check"]
            if mode == "agent" and not lite:
                cmd.extend(["-s", "workspace-write"])
            else:
                cmd.extend(["-s", "read-only"])
            if lite:
                # Lite mode: skip config.toml entirely so codex doesn't re-enable
                # bundled plugins on each launch. Inject MCPs via -c overrides.
                # --ignore-rules skips execpolicy .rules files. The isolated
                # cwd, not that flag, avoids caller-project AGENTS.md autoload.
                cmd.append("--ignore-user-config")
                cmd.append("--ignore-rules")
                if lite == "research":
                    args_json = json.dumps(_research_mcp_args())
                    cmd.extend(
                        [
                            "-c",
                            'mcp_servers.research.command="uv"',
                            "-c",
                            f"mcp_servers.research.args={args_json}",
                        ]
                    )
            if model:
                cmd.extend(["-m", model])
            if reasoning_effort and reasoning_effort in {
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
                "none",
            }:
                from .dispatch_plan import resolve_effort

                codex_effort, effort_warnings = resolve_effort(
                    reasoning_effort,
                    transport="codex-cli",
                    provider="openai",
                    model=model,
                )
                for warning in effort_warnings:
                    logger.warn(warning)
                if codex_effort:
                    cmd.extend(["-c", f'model_reasoning_effort="{codex_effort}"'])
                    reasoning_effort = codex_effort
            if schema:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as temp_file:
                    json.dump(schema, temp_file)
                    temp_schema_path = temp_file.name
                cmd.extend(["--output-schema", temp_schema_path])
            if use_stdin:
                cmd.append("-")
                stdin_input = prompt
            else:
                cmd.append(prompt)
        elif binary == "claude":
            # claude -p (headless). Lite profiles skip project context and
            # restrict MCP/tools; agent mode intentionally keeps the caller's
            # project context and native tool surface. Auth: drop ANTHROPIC_API_KEY from env so
            # the OAuth subscription path is used (api-key path can fail
            # with low credit balance even when subscription works fine).
            cmd = [
                "claude",
                "-p",
                "--no-session-persistence",
                # json (not text): the result event carries REAL usage + the
                # API-equivalent total_cost_usd even on the OAuth subscription path
                # (verified 2026-06-16). We unwrap result.result for the caller, so
                # this is transparent to the text-return contract. Closes the
                # subscription-usage blind spot (was 100% api-transport in the log).
                "--output-format",
                "json",
                # Claude Code's non-verbose result projection can retain only
                # the last text block. Verbose JSON includes assistant events,
                # which _parse_claude_json reconstructs and checks exactly.
                "--verbose",
                "--disable-slash-commands",
                # Unattended host: anything that would prompt is auto-denied while the
                # active permission mode keeps deciding (CC 2.1.259). Explicit, so a
                # future permission mode cannot hang a headless call on a prompt.
                "--permission-prompts",
                "none",
            ]
            if mode == "agent" and not lite:
                cmd.extend(["--permission-mode", "bypassPermissions"])
            elif lite == "research":
                mcp_cfg = json.dumps(
                    {
                        "mcpServers": {
                            "research": {
                                "command": "uv",
                                "args": _research_mcp_args(),
                            }
                        }
                    }
                )
                cmd.extend(
                    [
                        "--mcp-config",
                        mcp_cfg,
                        "--allowedTools",
                        "mcp__research",
                    ]
                )
            else:
                cmd.extend(
                    [
                        "--mcp-config",
                        '{"mcpServers":{}}',
                        "--allowedTools",
                        "",
                    ]
                )
            if not (mode == "agent" and not lite):
                # The empty cwd skips project settings, but user-level hooks still
                # ran (40 events per call). A blocking Stop hook makes the model
                # answer the hook, and that reply became the result: a 2026-09-27
                # extraction returned "the stop hook flagged a false positive".
                # Other user settings (env, effort) keep loading.
                cmd.extend(["--settings", json.dumps({"disableAllHooks": True})])
            if model:
                cmd.extend(["--model", model])
            if reasoning_effort:
                from .dispatch_plan import resolve_effort

                claude_effort, _ = resolve_effort(
                    reasoning_effort,
                    transport="claude-cli",
                    provider="anthropic",
                )
                if claude_effort:
                    cmd.extend(["--effort", claude_effort])
            # Always pipe prompt via stdin — keeps long prompts off argv.
            stdin_input = prompt
        elif binary == "cursor-agent":
            # cursor-agent -p (headless print). --mode ask = read-only Q&A (no
            # edits/shell), --trust required for non-interactive runs. Auth comes
            # from the user's Cursor app login. Prompt via stdin (-p with no
            # positional reads stdin); errors exit non-zero with stderr, so the
            # shared returncode/empty-output handling below catches them.
            cmd = [
                "cursor-agent",
                "-p",
                "--output-format",
                "text",
                "--mode",
                "ask",
                "--trust",
            ]
            if model:
                cmd.extend(["--model", model])
            stdin_input = prompt
        elif binary == "grok":
            cmd = ["grok"]
            if use_stdin:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as temp_file:
                    temp_file.write(prompt)
                    temp_prompt_path = temp_file.name
                cmd.extend(["--prompt-file", temp_prompt_path])
            else:
                cmd.extend(["-p", prompt])
            cmd.extend(
                [
                    "--output-format",
                    "json",
                    "--no-plan",
                    "--permission-mode",
                    "bypassPermissions" if mode == "agent" and not lite else "plan",
                ]
            )
            if model:
                cmd.extend(["-m", model])
            if reasoning_effort:
                from .dispatch_plan import resolve_effort

                grok_effort, effort_warnings = resolve_effort(
                    reasoning_effort,
                    transport="grok-cli",
                    provider="grok",
                    model=model,
                )
                for warning in effort_warnings:
                    logger.warn(warning)
                if grok_effort:
                    cmd.extend(["--reasoning-effort", grok_effort])
                    reasoning_effort = grok_effort
        else:
            return CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail=f"Unsupported CLI binary: {binary}",
            )

        start = time.time()
        # Use Popen with process group + threading timer for reliable timeout.
        # subprocess.run(timeout=) and SIGALRM both fail to interrupt blocking
        # waitpid() on macOS when the child spawns its own subprocesses.
        import os as _os
        import signal as _signal
        import threading as _threading

        # Lite mode: run from the runtime cwd (auto-bootstrapped from the
        # package skeleton). No HOME redirect — auth lives in the user's
        # normal HOME. Per-CLI flags (--ignore-user-config / --skip-trust /
        # --mcp-config '{}') strip user-config + project context; the
        # empty cwd handles AGENTS.md / GEMINI.md / CLAUDE.md autoload.
        #
        # Env scrubs:
        #   CLAUDE_SESSION_ID — Claude Code injects this into every subprocess.
        #     claude-cli writes <cwd>/.claude/current-session-id when it sees
        #     it (even with --no-session-persistence, the marker propagates
        #     for prepare-commit-msg). Drop it so the runtime cwd stays
        #     ephemeral instead of accumulating session history.
        #   ANTHROPIC_API_KEY (claude-cli only) — force OAuth subscription.
        env = None
        cwd = None
        if binary == "cursor-agent":
            # Always run cursor from a neutral empty cwd so the answer depends
            # only on the prompt, never the caller's workspace context.
            cwd = _cursor_cwd()
            logger.debug(f"[cli] cursor cwd={cwd}")
        elif binary == "grok":
            if mode != "agent":
                cwd = _grok_cwd()
                logger.debug(f"[cli] grok cwd={cwd}")
            env = dict(os.environ)
            env.pop("XAI_API_KEY", None)
            env.pop("GROK_API_KEY", None)
            logger.debug("[cli] grok-cli subscription auth (API keys stripped)")
        elif lite or binary == "claude":
            if lite:
                cwd = _lite_cwd(lite)
                logger.debug(f"[cli] lite={lite} cwd={cwd}")
            elif binary == "claude" and mode != "agent":
                cwd = _lite_cwd("bare")
                logger.debug("[cli] claude subscription cwd (bare cache)")
            elif binary == "claude":
                logger.debug(f"[cli] claude workspace agent cwd={Path.cwd()}")
            env = {k: v for k, v in os.environ.items() if k != "CLAUDE_SESSION_ID"}
            if binary == "claude":
                env.pop("ANTHROPIC_API_KEY", None)
                env.pop("CLAUDE_API_KEY", None)
                logger.debug("[cli] claude-cli OAuth (API keys stripped)")
                # A caller's explicit cap wins. 2026-10-08: two Opus 5.5 max-effort judge
                # calls stopped at the CLI cap and left 0-byte answers.
                model_id = (model or "").removesuffix("[1m]")
                output_ceiling = CLAUDE_CLI_MAX_OUTPUT_TOKENS.get(model_id)
                if output_ceiling and "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in env:
                    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(output_ceiling)

        # When the CLI runs from llmx's cache cwd, record durable attribution
        # (cwd marker + sidecar). Child env does NOT survive into transcripts.
        if cwd and _is_llmx_cache_path(cwd):
            caller = Path.cwd()
            if not _is_llmx_cache_path(caller):
                _record_dispatch_attribution(cli_cwd=str(cwd), caller_cwd=str(caller))
                # Marker may already exist from _caller_cache_subdir; refresh.
                try:
                    Path(cwd).mkdir(parents=True, exist_ok=True)
                    (Path(cwd) / _CALLER_MARKER).write_text(
                        str(caller.resolve()) + "\n", encoding="utf-8"
                    )
                except OSError as exc:
                    logger.debug(f"[cli] caller-marker refresh failed: {exc}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,  # new process group for clean kill
        )

        timed_out = False

        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            try:
                _os.killpg(proc.pid, _signal.SIGKILL)
            except OSError:
                proc.kill()

        timer = _threading.Timer(timeout, _kill_on_timeout)
        timer.start()

        # proc.communicate() blocks on read() until it sees EOF on stdout/stderr.
        # _kill_on_timeout's killpg() only reaches processes sharing proc's own
        # process group. A codex-cli descendant that re-sessions (setsid) before
        # inheriting the pipe fds escapes that kill: proc itself dies, but the
        # escaped grandchild still holds the pipe's write end open, so
        # communicate() never sees EOF and hangs indefinitely past `timeout` with
        # no further signal (the grandchild-pipe wedge — row
        # llmx-codex-timeout-not-enforced). Run communicate() on a helper thread,
        # bound the JOIN explicitly, and if it's still blocked once the kill
        # should have landed, force-close our end of the pipes: a closed fd makes
        # a blocked read() error out immediately regardless of any surviving
        # writer elsewhere.
        _COMMUNICATE_GRACE_S = 20
        _comm_result: list = []
        _comm_error: list = []

        def _do_communicate():
            try:
                _comm_result.append(proc.communicate(input=stdin_input))
            except Exception as exc:  # pragma: no cover - defensive
                _comm_error.append(exc)

        comm_thread = _threading.Thread(target=_do_communicate, daemon=True)
        comm_thread.start()
        comm_thread.join(timeout + _COMMUNICATE_GRACE_S)

        if comm_thread.is_alive():
            timed_out = True
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
            try:
                _os.killpg(proc.pid, _signal.SIGKILL)
            except OSError:
                proc.kill()
            comm_thread.join(_COMMUNICATE_GRACE_S)

        timer.cancel()

        if comm_thread.is_alive() or (not _comm_result and not _comm_error):
            # Force-closing the pipes should always unblock a read(); this is the
            # last-resort backstop so we NEVER return unbounded, even if some
            # platform's pipe semantics surprise us.
            timed_out = True
            stdout, stderr = (
                "",
                (
                    "llmx: codex-cli subprocess wedged past timeout+grace even after "
                    "killpg and force-closing pipes (grandchild-pipe wedge, unrecovered)"
                ),
            )
        elif _comm_error:
            stdout, stderr = (
                "",
                (f"llmx: communicate() raised after force-close: {_comm_error[0]!r}"),
            )
        else:
            stdout, stderr = _comm_result[0]

        elapsed = time.time() - start

        def _log_codex_usage(
            note_prefix: Optional[str] = None, error: Optional[str] = None
        ) -> None:
            if binary != "codex":
                return
            try:
                from .usage_log import log_usage

                usage, note = _latest_codex_rollout_usage(
                    codex_rollouts_before,
                    started_at=start,
                    session_id=_codex_session_id(stderr),
                )
                if note_prefix and note:
                    note = f"{note_prefix}; {note}"
                elif note_prefix:
                    note = note_prefix
                log_usage(
                    provider=provider,
                    model=model or "?",
                    transport="codex-cli",
                    reasoning_effort=reasoning_effort,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=usage.get("reasoning_tokens"),
                    cached_tokens=usage.get("cached_tokens"),
                    latency_s=elapsed,
                    error=error,
                    source="codex-rollout",
                    note=note,
                )
            except Exception as exc:
                logger.debug(f"[cli] codex usage log skipped: {exc}")

        if timed_out:
            _log_codex_usage(
                note_prefix=f"codex-cli timed out after {timeout}s",
                error="timeout",
            )
            logger.info(f"[cli→api] {binary} timed out after {timeout}s (killed process group)")
            return CliBackendFailure(
                kind=TimeoutError_,
                status=0,
                detail=f"{binary} timed out after {timeout}s",
            )

        if proc.returncode != 0:
            if binary == "claude" and _claude_payload_reports_error(stdout):
                parsed_result, _ = _parse_claude_json(
                    stdout, allow_continuation=(mode == "agent" and not lite)
                )
                if isinstance(parsed_result, CliBackendFailure):
                    logger.info(f"[cli] claude failed: {parsed_result.fallback_reason()}")
                    return parsed_result
            detail = _cli_failure_detail(stderr, stdout)
            _log_codex_usage(
                note_prefix=f"codex-cli exited {proc.returncode}: {detail}",
                error=f"exit_{proc.returncode}",
            )
            logger.info(f"[cli→api] {binary} exited {proc.returncode}: {detail}")
            return _classify_cli_failure(detail)

        text = stdout.strip()
        if not text:
            logger.info(f"[cli→api] {binary} returned empty output")
            return CliBackendFailure(
                kind=LlmxError,
                status=0,
                detail=f"{binary} returned empty output",
            )

        # claude --output-format json: unwrap the result text + log REAL usage at this
        # chokepoint (both CLI/sub paths funnel here). Closes the blind spot where
        # subscription calls never reached log_usage. Best-effort: a log failure never
        # breaks the call; typed parse failures are returned to the policy boundary.
        if binary == "claude":
            parsed_result, usage = _parse_claude_json(
                text, allow_continuation=(mode == "agent" and not lite)
            )
            if isinstance(parsed_result, CliBackendFailure):
                logger.info(f"[cli] claude failed: {parsed_result.fallback_reason()}")
                return parsed_result
            text = parsed_result
            if usage:
                try:
                    from .usage_log import log_usage

                    log_usage(
                        provider=provider,
                        model=usage.get("model") or model or "?",
                        served_model=usage.get("served_model"),
                        source="claude-cli-json",
                        note=(
                            "Claude CLI modelUsage missing or ambiguous; served_model unknown"
                            if usage.get("served_model") is None else None
                        ),
                        transport="claude-cli",  # subscription/CLI — cost is API-EQUIVALENT, not spend
                        reasoning_effort=reasoning_effort,
                        prompt_tokens=usage.get("input_tokens"),
                        completion_tokens=usage.get("output_tokens"),
                        reasoning_tokens=usage.get("reasoning_tokens"),
                        cached_tokens=usage.get("cache_read_input_tokens"),
                        latency_s=elapsed,
                    )
                except Exception as exc:
                    logger.debug(f"[cli] usage log skipped: {exc}")
        elif binary == "grok":
            parsed_result, usage = _parse_grok_json(text)
            if isinstance(parsed_result, CliBackendFailure):
                logger.info(f"[cli] grok failed: {parsed_result.fallback_reason()}")
                return parsed_result
            text = parsed_result
            if usage:
                try:
                    from .usage_log import log_usage

                    note_parts = ["subscription"]
                    if usage.get("served_model"):
                        note_parts.append(f"served_model={usage['served_model']}")
                    if usage.get("total_cost_usd") is not None:
                        note_parts.append(
                            f"reported_total_cost_usd={usage['total_cost_usd']}"
                        )
                    log_usage(
                        provider=provider,
                        model=model or GROK_BUILD_MODELS[0],
                        served_model=usage.get("served_model"),
                        transport="grok-cli",
                        billing="subscription",
                        reasoning_effort=reasoning_effort,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        reasoning_tokens=usage.get("reasoning_tokens"),
                        cached_tokens=usage.get("cached_tokens"),
                        latency_s=elapsed,
                        reported_cost_usd=usage.get("total_cost_usd"),
                        source="grok-json",
                        note="; ".join(note_parts),
                    )
                except Exception as exc:
                    logger.debug(f"[cli] usage log skipped: {exc}")
        elif binary == "codex":
            _log_codex_usage()

        logger.debug(f"[cli] {binary} responded in {elapsed:.1f}s ({len(text)} chars)")
        return text

    except subprocess.TimeoutExpired:
        logger.info(f"[cli→api] {binary} timed out after {timeout}s")
        return CliBackendFailure(
            kind=TimeoutError_,
            status=0,
            detail=f"{binary} timed out after {timeout}s",
        )
    except FileNotFoundError:
        logger.info(f"[cli→api] {binary} not found")
        return CliBackendFailure(
            kind=LlmxError,
            status=0,
            detail=f"{binary} not found",
        )
    finally:
        if temp_schema_path:
            try:
                os.unlink(temp_schema_path)
            except OSError:
                pass
        if temp_prompt_path:
            try:
                os.unlink(temp_prompt_path)
            except OSError:
                pass
