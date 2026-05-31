"""CLI-backed providers: codex-cli, claude-cli.

Shell out to Codex CLI / Claude Code instead of API for subscription pricing.
Fall back to API transparently when CLI can't handle requested features.

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
}

# Prefer subscription CLIs for logical providers when available.
# Google has NO CLI alias — it always routes to the paid Gemini API (the free
# gemini-cli consumer tier was retired 2026-06-18). OpenAI/Anthropic route to
# codex-cli/claude-cli only in --lite mode (full MCP load = ~37K context
# overhead + ~10s startup; lite profile disables MCPs and loads in ~1s).
CLI_PROVIDER_ALIASES = {}
CLI_PROVIDER_ALIASES_LITE = {
    "openai": "codex-cli",
    "anthropic": "claude-cli",
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
    "claude-opus-4-7",
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
    if schema and provider != "codex-cli":
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

    try:
        if binary == "codex":
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
                "minimal", "low", "medium", "high", "xhigh"
            }:
                cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
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
                "--output-format", "text",
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
            # Always pipe prompt via stdin — keeps long prompts off argv.
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
        if lite:
            cwd = _lite_cwd(lite)
            logger.debug(f"[cli] lite={lite} cwd={cwd}")
            env = {k: v for k, v in os.environ.items() if k != "CLAUDE_SESSION_ID"}
            if binary == "claude":
                env.pop("ANTHROPIC_API_KEY", None)
                logger.debug("[cli] lite claude (OAuth, no API key)")

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

        if timed_out:
            logger.info(f"[cli→api] {binary} timed out after {timeout}s (killed process group)")
            return None

        elapsed = time.time() - start

        if proc.returncode != 0:
            stderr_hint = stderr.strip()[:300] if stderr else ""
            stdout_hint = stdout.strip()[:200] if stdout else ""
            detail = stderr_hint or stdout_hint or "unknown error"
            logger.info(f"[cli→api] {binary} exited {proc.returncode}: {detail}")
            return None

        text = stdout.strip()
        if not text:
            logger.info(f"[cli→api] {binary} returned empty output")
            return None

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
