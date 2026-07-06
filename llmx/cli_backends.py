"""CLI-backed providers: codex-cli, claude-cli.

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
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .logger import logger

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
    # app subscription auth. NO api_fallback: composer-2.5 is Cursor-exclusive
    # (no public API), and proxied models (claude/gpt/gemini via the sub) have
    # no $0 API path either — a feature the CLI can't do raises, never silently
    # routes to a paid API. Always-on (not lite-gated): cursor-agent --mode ask
    # is already lightweight (~1s startup, no MCP).
    "cursor-cli": {
        "binary": "cursor-agent",
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
CLI_PROVIDER_ALIASES = {"cursor": "cursor-cli"}
CLI_PROVIDER_ALIASES_LITE = {
    "openai": "codex-cli",
    "anthropic": "claude-cli",
    "cursor": "cursor-cli",
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

# Lite mode is restricted to three frontier models. Anthropic routes via
# claude-cli (Claude Code) in headless `-p` mode with OAuth subscription auth
# (ANTHROPIC_API_KEY unset, --disable-slash-commands, empty mcp-config or
# research-mcp only); gpt-5.5 routes via codex-cli. gemini-3-flash-preview
# stays allowed for back-compat but no longer has a CLI backend — with the
# free gemini-cli retired (2026-06-18) it routes to the paid Gemini API and
# --lite only contributes the no-tools prompt prefix (no cwd/MCP stripping,
# no cost saving) for Google.
LITE_ALLOWED_MODELS = {
    "gpt-5.5",
    "gemini-3-flash-preview",
    "claude-opus-4-8",
    # 2026-07-05: Fable 5 ships on claude-cli subscription (same OAuth headless
    # -p transport as opus) — operator-directed for arc-agi H1/H2 dispatch lanes.
    "claude-fable-5",
}


def lite_model_allowed(model: Optional[str]) -> bool:
    """Return True if the resolved model is on the lite allowlist.

    Lenient match — `gemini-3.1-pro` and `gemini-3.1-pro-preview` both pass.
    """
    if not model:
        return False
    for allowed in LITE_ALLOWED_MODELS:
        base = allowed.removesuffix("-preview")
        if model == allowed or model == base or model.startswith(base + "-"):
            return True
    return False

# Max bytes for command-line argument before switching to stdin.
# macOS ARG_MAX is ~260KB but shells/tools choke earlier.
_ARG_MAX_BYTES = 100_000
_CODEX_SESSIONS_DIR = Path(os.path.expanduser("~/.codex/sessions"))


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


def preferred_cli_provider(provider: str, lite: Optional[str] = None) -> Optional[str]:
    """Return the CLI backend to prefer for a provider.

    Explicit CLI providers always resolve, even if the binary is missing, so callers can
    surface a precise fallback reason. Logical providers (openai/google) only resolve when
    the corresponding CLI is installed.

    `lite` ('bare' or 'research') routes openai → codex-cli (cost-saving mode).
    """
    cli_provider = configured_cli_provider(provider, lite=lite)
    if not cli_provider:
        return None
    if provider in CLI_PROVIDERS:
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
    # 2026-06-18: `gpt-5.5 --subscription --schema`). Treat schema as CLI-unsupported for ALL CLIs:
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
        return "max_tokens not supported by CLI (Gemini defaults to 8K)"
    # CLIs use their own default reasoning (high/thinking). Don't fall back just
    # because the caller asked for a specific effort — the CLI will ignore it,
    # which is fine for the "same model, free tier" use case.

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
        raise LiteEnvironmentError(
            f"--lite {lite!r} unknown. Supported: {list(_LITE_MODES)}"
        )
    skel = _LITE_PACKAGE_SKEL / lite
    if not skel.is_dir():  # only fires on broken installs
        raise LiteEnvironmentError(
            f"--lite mode {lite!r} skeleton {skel} missing — llmx install corrupt?"
        )
    runtime = _LITE_RUNTIME_ROOT / lite
    runtime.mkdir(parents=True, exist_ok=True)
    return str(runtime)


_CURSOR_RUNTIME_DIR = Path(os.path.expanduser("~/.cache/llmx/cursor"))


def _cursor_cwd() -> str:
    """Neutral empty cwd for cursor-agent.

    cursor-agent reads the workspace it runs in (rules, AGENTS.md, file tree)
    and folds it into context. For a clean model-query transport we run from an
    empty cache dir so the answer depends only on the prompt — not on wherever
    llmx happened to be invoked. Auto-created, idempotent.
    """
    _CURSOR_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return str(_CURSOR_RUNTIME_DIR)


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


def _read_codex_rollout_usage(path: Path) -> tuple[dict[str, Optional[int]], Optional[str]]:
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


def _latest_codex_rollout_usage(
    before: dict[Path, int],
    *,
    started_at: float,
    root: Path = _CODEX_SESSIONS_DIR,
) -> tuple[dict[str, Optional[int]], Optional[str]]:
    """Find the rollout created/updated by this codex-cli call and parse tokens.

    `codex exec` creates a fresh rollout in ~/.codex/sessions. In agent-driven
    runs the parent Codex session is also being updated, so prefer newly created
    files and use changed files only as a fallback.
    """
    if not root.exists():
        return _null_codex_usage(), f"codex sessions dir not found: {root}"

    new_files: list[tuple[int, Path]] = []
    changed_files: list[tuple[int, Path]] = []
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
        if mtime_ns < cutoff_ns:
            continue
        previous = before.get(path)
        if previous is None:
            new_files.append((mtime_ns, path))
        elif previous != mtime_ns:
            changed_files.append((mtime_ns, path))

    candidates = sorted(new_files or changed_files, reverse=True)
    if not candidates:
        return _null_codex_usage(), "no codex rollout changed after cli invocation"

    usage, note = _read_codex_rollout_usage(candidates[0][1])
    return usage, note


def _parse_claude_json(stdout: str):
    """Unwrap `claude -p --output-format json` (an events list) → (text, usage|None).

    text = the result event's `result` (the model's answer, schema-string or prose).
    usage = real tokens + API-equivalent total_cost_usd (present even on subscription).
    On is_error or parse failure → (None, None): the caller treats None as a CLI miss
    and falls back to API (the pre-existing failure semantics — never returns broken text).
    """
    try:
        events = json.loads(stdout)
    except Exception:
        return None, None
    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        return None, None
    for e in events:
        if not isinstance(e, dict) or e.get("type") != "result":
            continue
        if e.get("is_error"):
            return None, None
        r = e.get("result")
        if not isinstance(r, str):
            return None, None
        u = e.get("usage") or {}
        mu = e.get("modelUsage") or {}
        model_key = next(iter(mu), None)
        details = u.get("output_tokens_details") or u.get("completion_tokens_details") or {}
        reasoning_tokens = _first_present(
            u,
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
            "model": model_key.split("[")[0] if isinstance(model_key, str) else None,  # strip [1m] etc → PRICING key
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "total_cost_usd": e.get("total_cost_usd"),
        }
        return r, usage
    return None, None


def cli_chat(
    provider: str,
    prompt: str,
    model: Optional[str],
    timeout: int,
    *,
    schema=None,
    system: Optional[str] = None,
    lite: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Execute one-shot chat via CLI binary.

    Returns response text on success, None on failure (caller should fall back to API).
    For long prompts (>100KB), pipes via stdin to avoid ARG_MAX limits.

    `lite` ('bare' or 'research') runs the CLI in a stripped-down profile —
    no MCPs (bare) or research-MCP only (research), empty cwd, prompt prefix
    advising the model what's available.
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
    codex_rollouts_before: dict[Path, int] = {}

    try:
        if binary == "codex":
            codex_rollouts_before = _codex_rollout_snapshot()
            # codex exec [PROMPT] [-m <model>] [--output-schema schema.json]
            cmd = ["codex", "exec", "--skip-git-repo-check"]
            if lite:
                # Lite mode: skip config.toml entirely so codex doesn't re-enable
                # bundled plugins on each launch. Inject MCPs via -c overrides.
                # --ignore-rules (codex 0.122, PR #18646) strips project AGENTS.md
                # in addition to user config — pairs with empty cwd.
                cmd.append("--ignore-user-config")
                cmd.append("--ignore-rules")
                if lite == "research":
                    args_json = json.dumps(_research_mcp_args())
                    cmd.extend([
                        "-c", 'mcp_servers.research.command="uv"',
                        "-c", f"mcp_servers.research.args={args_json}",
                    ])
            if model:
                cmd.extend(["-m", model])
            if reasoning_effort and reasoning_effort in {
                "minimal", "low", "medium", "high", "xhigh", "max", "none"
            }:
                from .dispatch_plan import resolve_effort

                codex_effort, _ = resolve_effort(
                    reasoning_effort,
                    transport="codex-cli",
                    provider="openai",
                )
                if codex_effort:
                    cmd.extend(["-c", f'model_reasoning_effort="{codex_effort}"'])
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
            # claude -p (headless). Lite-only path — Claude Code is heavy by
            # default; we pass flags to skip skills/CLAUDE.md/sessions and
            # restrict MCP/tools. Auth: drop ANTHROPIC_API_KEY from env so
            # the OAuth subscription path is used (api-key path can fail
            # with low credit balance even when subscription works fine).
            cmd = [
                "claude", "-p",
                "--no-session-persistence",
                # json (not text): the result event carries REAL usage + the
                # API-equivalent total_cost_usd even on the OAuth subscription path
                # (verified 2026-06-16). We unwrap result.result for the caller, so
                # this is transparent to the text-return contract. Closes the
                # subscription-usage blind spot (was 100% api-transport in the log).
                "--output-format", "json",
                "--disable-slash-commands",
            ]
            if lite == "research":
                mcp_cfg = json.dumps({
                    "mcpServers": {
                        "research": {
                            "command": "uv",
                            "args": _research_mcp_args(),
                        }
                    }
                })
                cmd.extend([
                    "--mcp-config", mcp_cfg,
                    "--allowedTools", "mcp__research",
                ])
            else:
                cmd.extend([
                    "--mcp-config", '{"mcpServers":{}}',
                    "--allowedTools", "",
                ])
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
                "cursor-agent", "-p",
                "--output-format", "text",
                "--mode", "ask",
                "--trust",
            ]
            if model:
                cmd.extend(["--model", model])
            stdin_input = prompt
        else:
            return None

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
        elif lite or binary == "claude":
            if lite:
                cwd = _lite_cwd(lite)
                logger.debug(f"[cli] lite={lite} cwd={cwd}")
            elif binary == "claude":
                cwd = _lite_cwd("bare")
                logger.debug("[cli] claude subscription cwd (bare cache)")
            env = {k: v for k, v in os.environ.items() if k != "CLAUDE_SESSION_ID"}
            if binary == "claude":
                env.pop("ANTHROPIC_API_KEY", None)
                env.pop("CLAUDE_API_KEY", None)
                logger.debug("[cli] claude-cli OAuth (API keys stripped)")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, stdin=subprocess.PIPE, env=env, cwd=cwd,
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
        try:
            stdout, stderr = proc.communicate(input=stdin_input)
        finally:
            timer.cancel()

        elapsed = time.time() - start

        def _log_codex_usage(note_prefix: Optional[str] = None, error: Optional[str] = None) -> None:
            if binary != "codex":
                return
            try:
                from .usage_log import log_usage

                usage, note = _latest_codex_rollout_usage(
                    codex_rollouts_before,
                    started_at=start,
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
            return None

        if proc.returncode != 0:
            stderr_hint = stderr.strip()[:300] if stderr else ""
            stdout_hint = stdout.strip()[:200] if stdout else ""
            detail = stderr_hint or stdout_hint or "unknown error"
            _log_codex_usage(
                note_prefix=f"codex-cli exited {proc.returncode}: {detail}",
                error=f"exit_{proc.returncode}",
            )
            logger.info(f"[cli→api] {binary} exited {proc.returncode}: {detail}")
            return None

        text = stdout.strip()
        if not text:
            logger.info(f"[cli→api] {binary} returned empty output")
            return None

        # claude --output-format json: unwrap the result text + log REAL usage at this
        # chokepoint (both CLI/sub paths funnel here). Closes the blind spot where
        # subscription calls never reached log_usage. Best-effort: a log failure never
        # breaks the call; a json-parse failure falls back to API (clean miss).
        if binary == "claude":
            parsed_text, usage = _parse_claude_json(text)
            if parsed_text is None:
                logger.info(f"[cli→api] claude json parse/error; treating as CLI miss")
                return None
            text = parsed_text
            if usage:
                try:
                    from .usage_log import log_usage
                    log_usage(
                        provider=provider,
                        model=usage.get("model") or model or "?",
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
        elif binary == "codex":
            _log_codex_usage()

        logger.debug(f"[cli] {binary} responded in {elapsed:.1f}s ({len(text)} chars)")
        return text

    except subprocess.TimeoutExpired:
        logger.info(f"[cli→api] {binary} timed out after {timeout}s")
        return None
    except FileNotFoundError:
        logger.info(f"[cli→api] {binary} not found")
        return None
    finally:
        if temp_schema_path:
            try:
                os.unlink(temp_schema_path)
            except OSError:
                pass
