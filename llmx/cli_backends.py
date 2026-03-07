"""CLI-backed providers: gemini-cli, codex-cli.

Shell out to Gemini CLI / Codex CLI instead of LiteLLM for subscription/free-tier pricing.
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
}

# Prefer subscription/free CLIs for logical providers when available.
CLI_PROVIDER_ALIASES = {
    "google": "gemini-cli",
    "openai": "codex-cli",
}

# Max bytes for command-line argument before switching to stdin.
# macOS ARG_MAX is ~260KB but shells/tools choke earlier.
_ARG_MAX_BYTES = 100_000


def configured_cli_provider(provider: str) -> Optional[str]:
    """Return the CLI backend associated with a provider, if any."""
    if provider in CLI_PROVIDERS:
        return provider
    return CLI_PROVIDER_ALIASES.get(provider)


def binary_available(provider: str) -> bool:
    """Return whether the CLI binary for this provider is available."""
    cli_provider = configured_cli_provider(provider) or provider
    config = CLI_PROVIDERS.get(cli_provider)
    if not config:
        return False
    return shutil.which(config["binary"]) is not None


def preferred_cli_provider(provider: str) -> Optional[str]:
    """Return the CLI backend to prefer for a provider.

    Explicit CLI providers always resolve, even if the binary is missing, so callers can
    surface a precise fallback reason. Logical providers (openai/google) only resolve when
    the corresponding CLI is installed.
    """
    cli_provider = configured_cli_provider(provider)
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
    if system:
        return "system messages not supported by CLI"
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
) -> Optional[str]:
    """Execute one-shot chat via CLI binary.

    Returns response text on success, None on failure (caller should fall back to API).
    For long prompts (>100KB), pipes via stdin to avoid ARG_MAX limits.
    """
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
            if model:
                # Strip gemini/ prefix — CLI doesn't want it
                clean_model = model.removeprefix("gemini/")
                cmd.extend(["-m", clean_model])
        elif binary == "codex":
            # codex exec [PROMPT] [-m <model>] [--output-schema schema.json]
            cmd = ["codex", "exec"]
            if model:
                cmd.extend(["-m", model])
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
        else:
            return None

        start = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            input=stdin_input,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            stderr = result.stderr.strip()[:300] if result.stderr else ""
            stdout_hint = result.stdout.strip()[:200] if result.stdout else ""
            detail = stderr or stdout_hint or "unknown error"
            logger.info(f"[cli→api] {binary} exited {result.returncode}: {detail}")
            return None

        text = result.stdout.strip()
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
