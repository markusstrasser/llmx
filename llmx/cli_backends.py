"""CLI-backed providers: gemini-cli, codex-cli.

Shell out to Gemini CLI / Codex CLI instead of API for subscription/free-tier pricing.
Fall back to API transparently when CLI can't handle requested features.

CLI flag reference (verified 2026-03):
  gemini -p/--prompt <text> [-m <model>] [-o/--output-format text|json|stream-json]
         stdin is prepended to prompt when both provided
  codex exec [PROMPT] [-m <model>] [--output-schema schema.json]
         reads stdin when PROMPT is "-" or omitted
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from .logger import logger

# CLI provider configs — kept separate from PROVIDER_CONFIGS (different lifecycle)
CLI_PROVIDERS = {
    "gemini-cli": {
        "binary": "gemini",
        "api_fallback": "google",
    },
    "codex-cli": {
        "binary": "codex",
        "api_fallback": "openai",
    },
    "claude-cli": {
        "binary": "claude",
        "api_fallback": "anthropic",
    },
}

# Prefer subscription/free CLIs for logical providers when available.
# Default routing: gemini-cli for google. OpenAI → codex-cli only in --lite mode
# (codex with full MCP load = 37K context overhead + 10s startup; lite profile
# disables MCPs and loads in ~1s).
CLI_PROVIDER_ALIASES = {
    "google": "gemini-cli",
}
CLI_PROVIDER_ALIASES_LITE = {
    "google": "gemini-cli",
    "openai": "codex-cli",
    "anthropic": "claude-cli",
}

# Empty cwd reused across calls — prevents AGENTS.md / GEMINI.md autoload from
# the user's project tree. Initialized as a git repo so codex --skip-git-repo-check
# isn't strictly needed, but we pass it anyway for safety.
_LITE_CWD = os.path.expanduser("~/.cache/llmx/empty-cwd")

# Lite-mode HOME / CODEX_HOME selectors.
_LITE_GEMINI_HOMES = {
    "bare": os.path.expanduser("~/.gemini-bare"),
    "research": os.path.expanduser("~/.gemini-research"),
}
_LITE_CODEX_HOMES = {
    "bare": os.path.expanduser("~/.codex-bare"),
    "research": os.path.expanduser("~/.codex-research"),
}

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

# Lite mode is restricted to three frontier models. Anthropic routes via
# claude-cli (Claude Code) in headless `-p` mode with OAuth subscription auth
# (ANTHROPIC_API_KEY unset, --disable-slash-commands, empty mcp-config or
# research-mcp only). Gemini-3.1-pro is excluded — capacity-limited / not
# accessible without Pro sub at the time of writing; gemini-3-flash-preview
# is the fast tier.
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
    _ = lite  # used for HOME / cwd routing below

    config = CLI_PROVIDERS[provider]
    binary = config["binary"]
    stdin_input = None
    use_stdin = len(prompt.encode()) > _ARG_MAX_BYTES
    temp_schema_path = None

    try:
        if binary == "gemini":
            # gemini -p <prompt> -o text [-m <model>]
            # When prompt is large, pipe via stdin and use -p with short instruction
            if use_stdin:
                cmd = ["gemini", "-p", "Respond to the input provided on stdin.", "-o", "text"]
                stdin_input = prompt
            else:
                cmd = ["gemini", "-p", prompt, "-o", "text"]
            if lite:
                # Empty-cwd lite profile isn't in gemini's trusted-folders list;
                # --skip-trust bypasses the headless rejection.
                cmd.append("--skip-trust")
            if model:
                # Strip gemini/ prefix — CLI doesn't want it
                clean_model = model.removeprefix("gemini/")
                cmd.extend(["-m", clean_model])
        elif binary == "codex":
            # codex exec [PROMPT] [-m <model>] [--output-schema schema.json]
            cmd = ["codex", "exec", "--skip-git-repo-check"]
            if lite:
                # Lite mode: skip config.toml entirely so codex doesn't re-enable
                # bundled plugins on each launch. Inject MCPs via -c overrides.
                cmd.append("--ignore-user-config")
                if lite == "research":
                    cmd.extend([
                        "-c", 'mcp_servers.research.command="uv"',
                        "-c",
                        'mcp_servers.research.args=["run","--directory",'
                        '"/Users/alien/Projects/research-mcp","research-mcp"]',
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
                cmd.extend([
                    "--mcp-config",
                    '{"mcpServers":{"research":{"command":"uv","args":'
                    '["run","--directory","/Users/alien/Projects/research-mcp",'
                    '"research-mcp"]}}}',
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

        # Lite mode: route HOME / CODEX_HOME to a stripped-down profile and
        # run from an empty cwd so no project AGENTS.md / GEMINI.md is autoloaded.
        # Default (non-lite) gemini path keeps the original ~/.gemini-bare HOME
        # override for backward compatibility.
        env = None
        cwd = None
        if lite:
            cwd = _LITE_CWD if os.path.isdir(_LITE_CWD) else None
            if binary == "gemini":
                home = _LITE_GEMINI_HOMES.get(lite)
                if home and os.path.isdir(home):
                    env = {**os.environ, "HOME": home}
                    logger.debug(f"[cli] lite={lite} HOME={home}")
            elif binary == "codex":
                home = _LITE_CODEX_HOMES.get(lite)
                if home and os.path.isdir(home):
                    env = {**os.environ, "CODEX_HOME": home}
                    logger.debug(f"[cli] lite={lite} CODEX_HOME={home}")
            elif binary == "claude":
                # Use OAuth subscription, not the API key path (which can hit
                # low-credit failures while the subscription is fine).
                env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
                logger.debug(f"[cli] lite={lite} claude (OAuth, no API key)")
        elif binary == "gemini":
            bare_home = os.path.join(os.path.expanduser("~"), ".gemini-bare")
            if os.path.isdir(bare_home):
                env = {**os.environ, "HOME": bare_home}
                logger.debug(f"[cli] using bare HOME={bare_home}")

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
