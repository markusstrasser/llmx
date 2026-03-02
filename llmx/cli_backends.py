"""CLI-backed providers: gemini-cli, codex-cli.

Shell out to Gemini CLI / Codex CLI instead of LiteLLM for subscription/free-tier pricing.
Fall back to API transparently when CLI can't handle requested features.
"""

import shutil
import subprocess
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


def needs_api_fallback(
    provider: str,
    schema,
    system: Optional[str],
    search: bool,
    stream: bool,
    reasoning_effort: Optional[str],
) -> Optional[str]:
    """Check if request requires features the CLI can't handle.

    Returns reason string if fallback needed, None if CLI can handle it.
    """
    config = CLI_PROVIDERS[provider]
    binary = config["binary"]

    if not shutil.which(binary):
        return f"{binary} not found in PATH"
    if schema:
        return "structured output not supported by CLI"
    if system:
        return "system messages not supported by CLI"
    if search:
        return "web search not supported by CLI"
    if stream:
        return "streaming not supported by CLI"
    if reasoning_effort and reasoning_effort != "high":
        return f"reasoning_effort={reasoning_effort} not adjustable via CLI"

    return None


def cli_chat(
    provider: str, prompt: str, model: Optional[str], timeout: int
) -> Optional[str]:
    """Execute one-shot chat via CLI binary.

    Returns response text on success, None on failure (caller should fall back to API).
    """
    config = CLI_PROVIDERS[provider]
    binary = config["binary"]

    if binary == "gemini":
        cmd = ["gemini", "--prompt", prompt, "--output-format", "text"]
        if model:
            # Strip gemini/ prefix — CLI doesn't want it
            clean_model = model.removeprefix("gemini/")
            cmd.extend(["--model", clean_model])
    elif binary == "codex":
        cmd = ["codex", "exec"]
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)
    else:
        return None

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            stderr = result.stderr.strip()[:200] if result.stderr else "unknown error"
            logger.info(f"[cli→api] {binary} exited {result.returncode}: {stderr}")
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
