"""Provider management using native SDKs (google-genai, openai)"""

import difflib
import os
import signal
import sys
import threading
import time
from typing import Optional, Dict, Any

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import openai as openai_module
from openai import OpenAI
from rich.console import Console

from .logger import logger

console = Console()


class _WallClockTimeout(Exception):
    """Raised by SIGALRM when wall-clock deadline is exceeded."""
    pass


# ============================================================================
# Structured Error Types — distinct exit codes for agent consumption
# ============================================================================

EXIT_SUCCESS = 0
EXIT_GENERAL = 1
EXIT_API_KEY = 2
EXIT_RATE_LIMIT = 3    # 429, 503, quota exhausted
EXIT_TIMEOUT = 4
EXIT_MODEL_ERROR = 5   # context too large, model not found, invalid request


class LlmxError(RuntimeError):
    """Base error with structured diagnostics."""
    exit_code = EXIT_GENERAL

    def __init__(self, message: str, *, provider: str = "", model: str = "",
                 status_code: int = 0, error_type: str = "general"):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.error_type = error_type

    def diagnostic_line(self) -> str:
        """Parseable one-liner for stderr. Agents can grep for [llmx:ERROR]."""
        parts = [f"[llmx:ERROR] type={self.error_type}"]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        parts.append(f"exit={self.exit_code}")
        return " ".join(parts)


class RateLimitError(LlmxError):
    exit_code = EXIT_RATE_LIMIT

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_type="rate_limit", **kwargs)


class TimeoutError_(LlmxError):
    exit_code = EXIT_TIMEOUT

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_type="timeout", **kwargs)


class ApiKeyError(LlmxError):
    exit_code = EXIT_API_KEY

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_type="api_key", **kwargs)


class ModelError(LlmxError):
    exit_code = EXIT_MODEL_ERROR

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_type="model_error", **kwargs)

# Model-specific parameter restrictions
MODEL_RESTRICTIONS = {
    # OpenAI GPT-5.x thinking models: temperature=1 only, support reasoning_effort
    "gpt-5.4": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["none", "minimal", "low", "medium", "high", "xhigh"], "default_effort": "high"},
    "gpt-5.2": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"], "default_effort": "high"},
    "gpt-5.1": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"], "default_effort": "high"},
    "gpt-5.1-mini": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"], "default_effort": "high"},
    # Legacy GPT-5 models
    "gpt-5": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"], "default_effort": "high"},
    "gpt-5-pro": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"], "default_effort": "high"},
    "gpt-5-codex": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["low", "medium", "high"], "default_effort": "high"},  # No minimal
    # Gemini 3.x thinking models: temperature=1 required (lower causes looping/degraded reasoning)
    "gemini-3.1-pro": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["low", "medium", "high"]},
    "gemini-3-pro": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["low", "medium", "high"]},
    "gemini-3-flash": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"]},
    "gemini-3.1-flash-lite": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["low", "medium", "high"]},
    # OpenAI GPT-5.3 (Mar 2026): "Instant" variant, reduced hallucination, max reasoning_effort=medium
    "gpt-5.3": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["medium"], "default_effort": "medium"},
    # Kimi K2.5 thinking model (Jan 2026)
    "kimi-k2.5": {"temperature": 1.0, "fixed": True, "reasoning_effort": False},  # No reasoning_effort support
    # Legacy K2 variants
    "kimi-k2-thinking": {"temperature": 1.0, "fixed": True, "reasoning_effort": False},
}

# ============================================================================
# Provider configurations — native SDKs, no LiteLLM prefixes
# ============================================================================

PROVIDER_CONFIGS = {
    "google": {
        "model": "gemini-3.1-pro-preview",
        "legacy_model": "gemini-3-pro-preview",
        "env_var": "GEMINI_API_KEY or GOOGLE_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
        "flash_model": "gemini-3-flash-preview",
        "flash_lite_model": "gemini-3.1-flash-lite-preview",
    },
    "openai": {
        "model": "gpt-5.4",
        "legacy_model": "gpt-5.2",
        "env_var": "OPENAI_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "anthropic": {
        "model": "claude-opus-4-6",
        "legacy_model": "claude-sonnet-4-6",
        "env_var": "OPENROUTER_API_KEY",  # routed via OpenRouter (no Anthropic API credits)
        "temperature_range": (0.0, 1.0),
        "supports_streaming": True,
    },
    "xai": {
        "model": "grok-4",
        "fast_model": "grok-4-1-fast-reasoning",
        "non_thinking_model": "grok-4-1-fast-non-reasoning",
        "legacy_model": "grok-beta",
        "env_var": "XAI_API_KEY or GROK_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "deepseek": {
        "model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "openrouter": {
        "model": "openai/gpt-4o",
        "env_var": "OPENROUTER_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "kimi": {
        "model": "kimi-k2.5",
        "old_model": "kimi-k2-thinking",
        "legacy_model": "kimi-k2-0711-preview",
        "env_var": "MOONSHOT_API_KEY or KIMI_API_KEY",
        "temperature_range": (0.0, 1.0),
        "supports_streaming": True,
    },
    "cerebras": {
        "model": "qwen-3-coder-480b",
        "env_var": "CEREBRAS_API_KEY",
        "temperature_range": (0.0, 1.5),
        "supports_streaming": True,
    },
    # CLI-backed providers: shell out to subscription CLIs instead of per-token API
    "gemini-cli": {
        "model": None,  # CLI uses its own default
        "env_var": None,  # Google account auth
        "temperature_range": (0.0, 2.0),
        "supports_streaming": False,
        "api_fallback": "google",
    },
    "codex-cli": {
        "model": None,
        "env_var": None,  # ChatGPT subscription auth
        "temperature_range": (0.0, 2.0),
        "supports_streaming": False,
        "api_fallback": "openai",
    },
}


# ============================================================================
# OpenAI-compatible provider routing
# ============================================================================

OPENAI_COMPAT_URLS = {
    "openai": None,  # default
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "anthropic": "https://openrouter.ai/api/v1",  # Anthropic via OpenRouter
}

# API key overrides for providers routed through another service
API_KEY_OVERRIDES = {
    "anthropic": "OPENROUTER_API_KEY",
}


# ============================================================================
# Model normalization — accept old LiteLLM-prefixed names gracefully
# ============================================================================

_STRIP_PREFIXES = {
    "google": "gemini/",
    "kimi": "moonshot/",
    "xai": "xai/",
    "anthropic": "anthropic/",
    "deepseek": "deepseek/",
    "cerebras": "cerebras/",
    "openrouter": "openrouter/",
}


def _normalize_model(provider: str, model: str) -> str:
    """Strip synthetic LiteLLM prefixes, preserve real provider model IDs."""
    prefix = _STRIP_PREFIXES.get(provider)
    if prefix and model.startswith(prefix):
        logger.debug(f"Stripped prefix '{prefix}' from model '{model}'")
        return model[len(prefix):]
    # OpenRouter models use real slashes (anthropic/claude-sonnet-4-6) — preserve
    return model


# Known models for typo detection
_KNOWN_MODELS = {
    "google": ["gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"],
    "openai": ["gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1", "gpt-5.1-mini", "gpt-5", "gpt-5-pro", "gpt-5-codex"],
    "xai": ["grok-4", "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok-beta"],
    "kimi": ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-0711-preview"],
    "deepseek": ["deepseek-chat"],
    "cerebras": ["qwen-3-coder-480b"],
}


def _warn_unknown_model(model: str, provider: str):
    """Warn about potentially misspelled model names. Never hard-fail."""
    known = _KNOWN_MODELS.get(provider)
    if known and model not in known:
        close = difflib.get_close_matches(model, known, n=1, cutoff=0.6)
        if close:
            logger.warn(f"Unknown model '{model}' for {provider}. Did you mean '{close[0]}'?")


def infer_provider_from_model(model: str) -> Optional[str]:
    """Infer provider from model name"""
    model_lower = model.lower()

    # Check for explicit prefixes first
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("moonshot/") or "kimi" in model_lower:
        return "kimi"
    if model.startswith("cerebras/") or "qwen" in model_lower:
        return "cerebras"
    if model.startswith("gemini/") or "gemini" in model_lower:
        return "google"
    if model.startswith("xai/") or "grok" in model_lower:
        return "xai"
    if model.startswith("anthropic/") or "claude" in model_lower:
        return "anthropic"
    if model.startswith("deepseek/") or "deepseek" in model_lower:
        return "deepseek"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("chatgpt"):
        return "openai"

    return None


def check_gemini_flash_usage(model_name: str, prompt: str) -> None:
    """Info log when using Flash Lite for awareness"""
    if not model_name or "flash" not in model_name.lower():
        return
    if "gemini-3-flash" in model_name.lower() or "3-flash" in model_name.lower():
        return
    if "lite" not in model_name.lower():
        return

    logger.debug(
        "Using Flash Lite",
        {"model": model_name, "note": "$0.25/M in, $1.50/M out, 1M context"}
    )


def get_model_name(provider: str, model: Optional[str] = None, use_old: bool = False) -> str:
    """Get model name for provider — no prefixes needed (native SDKs)"""
    if model:
        # Strip any leftover LiteLLM prefixes
        return _normalize_model(provider, model)

    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    if use_old and "old_model" in config:
        return config["old_model"]

    return config["model"]


def get_model_restriction(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model-specific parameter restrictions"""
    if not model_name:
        return None

    model_lower = model_name.lower()

    # Sort keys by length descending to match longest (most specific) first
    sorted_keys = sorted(MODEL_RESTRICTIONS.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if key.lower() in model_lower:
            return MODEL_RESTRICTIONS[key]

    return None


def validate_and_adjust_temperature(
    temperature: float,
    model_name: str,
    provider: str,
    user_specified: bool = False
) -> tuple[float, bool]:
    """Validate and adjust temperature for model/provider.
    Returns (adjusted_temperature, was_adjusted)
    """
    restriction = get_model_restriction(model_name)

    if restriction and restriction.get("fixed"):
        required_temp = restriction["temperature"]
        if temperature != required_temp:
            if user_specified:
                logger.warn(
                    f"Temperature override ignored: {model_name} only supports temperature={required_temp}",
                    {"requested": temperature, "using": required_temp}
                )
            else:
                logger.debug(
                    f"Auto-adjusted temperature for {model_name}",
                    {"from": temperature, "to": required_temp}
                )
            return required_temp, True
        return temperature, False

    config = PROVIDER_CONFIGS.get(provider)
    if config and "temperature_range" in config:
        min_temp, max_temp = config["temperature_range"]
        if temperature < min_temp or temperature > max_temp:
            clamped = max(min_temp, min(max_temp, temperature))
            logger.warn(
                f"Temperature clamped to {provider} valid range [{min_temp}, {max_temp}]",
                {"requested": temperature, "using": clamped}
            )
            return clamped, True

    return temperature, False


def check_api_key(provider: str) -> None:
    """Check if API key is available for provider"""
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}. Use --list-providers to see available providers.")

    # CLI providers use subscription auth, no API key needed
    if config.get("env_var") is None:
        return

    # Check common environment variables
    key_vars = config["env_var"].replace(" or ", ",").split(",")
    for var in key_vars:
        var = var.strip()
        if os.getenv(var):
            logger.debug(f"Found API key: {var}")
            return

    # No key found - provide helpful error
    logger.error(f"API key missing for {provider}", {"env_var": config["env_var"]})

    error_msg = f"API key not found for provider '{provider}'.\n"
    error_msg += f"Set one of these environment variables: {config['env_var']}\n"
    error_msg += f"Example: export {key_vars[0].strip()}=your-key-here"

    raise RuntimeError(error_msg)


def _get_api_key(provider: str) -> Optional[str]:
    """Resolve the API key for a provider."""
    # Check overrides first (e.g., anthropic -> OPENROUTER_API_KEY)
    key_env = API_KEY_OVERRIDES.get(provider)
    if key_env:
        return os.environ.get(key_env)

    config = PROVIDER_CONFIGS.get(provider)
    if not config or not config.get("env_var"):
        return None

    key_vars = config["env_var"].replace(" or ", ",").split(",")
    for var in key_vars:
        var = var.strip()
        val = os.getenv(var)
        if val:
            return val
    return None


# ============================================================================
# Native SDK backends
# ============================================================================

def _google_chat(prompt, model, system, temperature, timeout, stream,
                 max_tokens, search, schema, reasoning_effort):
    """Google Gemini via google-genai SDK. Returns response text."""
    client = genai.Client(
        http_options=types.HttpOptions(
            timeout=max(timeout * 1000, 10_000) if timeout else 300_000
        )
    )
    config = types.GenerateContentConfig(temperature=temperature)

    if system:
        config.system_instruction = system
    if max_tokens:
        config.max_output_tokens = max_tokens
    if reasoning_effort:
        level_map = {"low": "low", "medium": "medium", "high": "high",
                     "minimal": "low", "none": "low", "xhigh": "high"}
        config.thinking_config = types.ThinkingConfig(
            thinking_level=level_map.get(reasoning_effort, "high")
        )
    if schema:
        config.response_mime_type = "application/json"
        config.response_schema = schema
    if search:
        config.tools = [types.Tool(google_search=types.GoogleSearch())]

    result_text = ""
    finish_reason = None

    if stream:
        for chunk in client.models.generate_content_stream(
            model=model, contents=prompt, config=config
        ):
            if chunk.text:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
                result_text += chunk.text
        sys.stdout.write("\n")
    else:
        response = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
        # Guard empty candidates from safety filter
        if not response.candidates:
            feedback = getattr(response, 'prompt_feedback', None)
            raise ModelError(
                f"Response blocked by safety filter: {feedback}",
                provider="google", model=model,
            )
        result_text = response.text
        finish_reason = str(response.candidates[0].finish_reason)
        print(result_text)

    # Truncation detection
    if finish_reason and "MAX_TOKENS" in str(finish_reason):
        logger.warn(f"[llmx:WARN] output may be truncated (hit {max_tokens or 8192} token limit)")

    return result_text


def _openai_chat(prompt, model, provider, system, temperature, timeout,
                 stream, max_tokens, schema, reasoning_effort):
    """OpenAI-compatible API via openai SDK. Returns response text."""
    base_url = OPENAI_COMPAT_URLS.get(provider)
    api_key = _get_api_key(provider)

    # Anthropic via OpenRouter needs model prefix
    if provider == "anthropic" and not model.startswith("anthropic/"):
        model = f"anthropic/{model}"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(timeout) if timeout else 300.0,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema},
        }

    result_text = ""
    finish_reason = None

    if stream:
        for chunk in client.chat.completions.create(**kwargs, stream=True):
            # Guard empty choices from OpenRouter keepalive chunks
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                result_text += delta.content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        sys.stdout.write("\n")
    else:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            refusal = getattr(response.choices[0].message, 'refusal', None)
            raise ModelError(
                f"Model returned no content. Refusal: {refusal}",
                provider=provider, model=model,
            )
        result_text = content
        finish_reason = response.choices[0].finish_reason
        print(result_text)

    # Truncation detection
    if finish_reason == "length":
        logger.warn(f"[llmx:WARN] output may be truncated (hit max_tokens limit)")

    return result_text


def _build_search_kwargs(provider: str, model_name: str) -> dict:
    """Build provider-specific kwargs for web search grounding.
    Only meaningful for legacy callers — search is handled natively in _google_chat.
    """
    if provider == "google":
        return {}  # handled in _google_chat via config.tools
    elif provider == "xai":
        logger.warn("xAI web search not yet supported via OpenAI SDK — ignoring --search")
        return {}
    elif provider == "openai":
        logger.warn(
            "OpenAI web search requires search-specific models (gpt-4o-search-preview). "
            "Use 'llmx research' for OpenAI deep research instead."
        )
        return {}
    else:
        logger.warn(f"Web search not supported for provider '{provider}' — ignoring --search")
        return {}


# ============================================================================
# Main chat dispatcher
# ============================================================================

def chat(
    prompt: str,
    provider: str,
    model: Optional[str],
    temperature: float,
    reasoning_effort: Optional[str],
    stream: bool,
    debug: bool,
    json_output: bool,
    use_old: bool = False,
    user_specified_temp: bool = False,
    timeout: int = 300,
    search: bool = False,
    system: Optional[str] = None,
    schema: Optional[dict] = None,
    max_tokens: Optional[int] = None,
) -> None:
    """Execute chat with single provider"""
    start_time = time.time()
    model_name = model or "default"  # Initialize early for error handlers

    # Wall-clock timeout via SIGALRM (main thread only)
    use_alarm = threading.current_thread() is threading.main_thread()

    def _alarm_handler(signum, frame):
        raise _WallClockTimeout(f"Wall-clock timeout after {timeout}s")

    _alarm_handler_prev = None
    if use_alarm:
        _alarm_handler_prev = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        # CLI backend handling — intercept before API logic
        from .cli_backends import (
            CLI_PROVIDERS,
            needs_api_fallback,
            cli_chat,
            preferred_cli_provider,
        )

        cli_provider = preferred_cli_provider(provider)
        if cli_provider:
            logical_provider = (
                provider if provider not in CLI_PROVIDERS else CLI_PROVIDERS[cli_provider]["api_fallback"]
            )
            cli_model = model or get_model_name(logical_provider, None, use_old)
            fallback_reason = needs_api_fallback(
                cli_provider, schema, system, search, stream, reasoning_effort, max_tokens
            )
            if fallback_reason:
                api_provider = CLI_PROVIDERS[cli_provider]["api_fallback"]
                logger.info(f"[cli→api] {cli_provider} → {api_provider} ({fallback_reason})")
                provider = api_provider
                model = cli_model
            else:
                text = cli_chat(cli_provider, prompt, cli_model, timeout, schema=schema)
                if text is not None:
                    print(text)
                    return
                # CLI failed — fall back to API
                elapsed_cli = int(time.time() - start_time)
                remaining = max(timeout - elapsed_cli, 5)
                if use_alarm:
                    signal.alarm(remaining)
                api_provider = CLI_PROVIDERS[cli_provider]["api_fallback"]
                logger.info(f"[cli→api] {cli_provider} → {api_provider} (CLI returned error, {remaining}s remaining)")
                provider = api_provider
                model = cli_model

        check_api_key(provider)
        model_name = get_model_name(provider, model, use_old)
        requested_reasoning_effort = reasoning_effort

        # Warn on potentially unknown model
        _warn_unknown_model(model_name, provider)

        # Validate and adjust temperature
        adjusted_temp, was_adjusted = validate_and_adjust_temperature(
            temperature, model_name, provider, user_specified_temp
        )

        # Check for Gemini Flash misuse
        check_gemini_flash_usage(model_name, prompt)

        # Validate reasoning_effort parameter
        restriction = get_model_restriction(model_name)
        if reasoning_effort:
            if not restriction or not restriction.get("reasoning_effort"):
                logger.warn(
                    f"Model {model_name} does not support --reasoning-effort parameter (ignoring)",
                    {"model": model_name, "provider": provider}
                )
                reasoning_effort = None
            elif restriction.get("reasoning_effort_levels"):
                valid_levels = restriction["reasoning_effort_levels"]
                if reasoning_effort not in valid_levels:
                    logger.error(
                        f"Invalid reasoning_effort for {model_name}",
                        {"valid": valid_levels, "requested": reasoning_effort}
                    )
                    raise ValueError(
                        f"Model {model_name} only supports reasoning_effort: {', '.join(valid_levels)}\n"
                        f"You requested: {reasoning_effort}\n"
                        f"Use one of the supported values."
                    )

        # Default reasoning_effort for thinking models that support it
        if restriction and restriction.get("reasoning_effort") and not reasoning_effort:
            default_effort = restriction.get("default_effort")
            if default_effort:
                reasoning_effort = default_effort
                logger.info(f"Defaulting to --reasoning-effort {default_effort} for {model_name}")

        logger.debug(
            "Starting chat",
            {
                "provider": provider,
                "model": model_name,
                "temperature": adjusted_temp,
                "stream": stream,
                "prompt_length": len(prompt),
                "requested_reasoning_effort": requested_reasoning_effort,
                "effective_reasoning_effort": reasoning_effort,
            },
        )

        # Dispatch to native SDK backend
        if provider == "google":
            _google_chat(
                prompt=prompt, model=model_name, system=system,
                temperature=adjusted_temp, timeout=timeout, stream=stream,
                max_tokens=max_tokens, search=search, schema=schema,
                reasoning_effort=reasoning_effort,
            )
        else:
            # All non-google providers go through OpenAI-compatible SDK
            if search:
                _build_search_kwargs(provider, model_name)  # logs warning for unsupported
            _openai_chat(
                prompt=prompt, model=model_name, provider=provider,
                system=system, temperature=adjusted_temp, timeout=timeout,
                stream=stream, max_tokens=max_tokens, schema=schema,
                reasoning_effort=reasoning_effort,
            )

        elapsed = time.time() - start_time
        logger.debug("Generation complete", {"elapsed_sec": round(elapsed, 2)})

    except _WallClockTimeout:
        elapsed = time.time() - start_time
        logger.error(
            f"Wall-clock timeout after {elapsed:.1f}s",
            {"provider": provider, "model": model_name, "timeout": timeout}
        )
        raise TimeoutError_(
            f"Request timed out after {timeout}s (wall-clock). Model may be overloaded or prompt too large.",
            provider=provider, model=model_name, status_code=0,
        )
    except (LlmxError, ValueError):
        # Already typed — re-raise as-is
        raise
    except genai_errors.ClientError as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise RateLimitError(
                f"Rate limit exceeded for google/{model_name}. Wait or use --fallback.",
                provider="google", model=model_name, status_code=429,
            ) from e
        elif "404" in msg:
            raise ModelError(
                f"Model not found: {model_name}",
                provider="google", model=model_name, status_code=404,
            ) from e
        raise LlmxError(msg, provider="google", model=model_name) from e
    except genai_errors.ServerError as e:
        msg = str(e)
        if "DEADLINE_EXCEEDED" in msg:
            raise TimeoutError_(
                f"Google API deadline exceeded for {model_name}.",
                provider="google", model=model_name,
            ) from e
        raise LlmxError(str(e), provider="google", model=model_name) from e
    except openai_module.RateLimitError as e:
        raise RateLimitError(
            f"Rate limit exceeded for {provider}/{model_name}. Wait or use --fallback.",
            provider=provider, model=model_name, status_code=429,
        ) from e
    except openai_module.APITimeoutError as e:
        raise TimeoutError_(
            f"Request timed out for {provider}/{model_name}.",
            provider=provider, model=model_name,
        ) from e
    except openai_module.AuthenticationError as e:
        raise ApiKeyError(
            f"Invalid API key for {provider}. Check your API key configuration.",
            provider=provider, model=model_name,
        ) from e
    except openai_module.NotFoundError as e:
        raise ModelError(
            f"Model not found: {model_name}",
            provider=provider, model=model_name, status_code=404,
        ) from e
    except openai_module.APIStatusError as e:
        if e.status_code == 402:
            raise RateLimitError(
                f"Insufficient credits for {provider}: {e}",
                provider=provider, model=model_name, status_code=402,
            ) from e
        raise LlmxError(str(e), provider=provider, model=model_name) from e
    except TimeoutError as error:
        elapsed = time.time() - start_time
        raise TimeoutError_(
            f"Request timed out after {timeout}s. Model may be overloaded or prompt too large.",
            provider=provider, model=model_name, status_code=0,
        ) from error
    except Exception as error:
        elapsed = time.time() - start_time
        error_type = error.__class__.__name__
        error_msg = str(error)

        logger.error(
            "Chat failed",
            {"provider": provider, "model": model_name, "error_type": error_type, "elapsed": f"{elapsed:.1f}s"}
        )
        raise LlmxError(
            f"{error_type}: {error_msg}",
            provider=provider, model=model_name,
        ) from error
    finally:
        if use_alarm:
            signal.alarm(0)
            if _alarm_handler_prev is not None:
                signal.signal(signal.SIGALRM, _alarm_handler_prev)


# ============================================================================
# Compare: parallel provider calls
# ============================================================================

def compare(
    prompt: str,
    providers: list[str],
    temperature: float,
    reasoning_effort: Optional[str],
    debug: bool,
    json_output: bool,
    use_old: bool = False,
    user_specified_temp: bool = False,
    timeout: int = 300,
    search: bool = False,
) -> None:
    """Compare responses from multiple providers"""
    import concurrent.futures
    import io

    start_time = time.time()
    logger.info(f"Comparing {len(providers)} providers")

    def call_provider(provider: str) -> tuple[str, Optional[str], Optional[str]]:
        """Call a single provider and return (provider, text, error)"""
        provider_start = time.time()
        try:
            logger.debug(f"Calling {provider}")

            check_api_key(provider)
            model_name = get_model_name(provider, use_old=use_old)
            model_name = _normalize_model(provider, model_name)

            adjusted_temp, _ = validate_and_adjust_temperature(
                temperature, model_name, provider, user_specified_temp
            )

            effective_effort = reasoning_effort
            # No SIGALRM in worker threads — rely on SDK/server timeouts

            if provider == "google":
                # Capture stdout to avoid interleaved output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    text = _google_chat(
                        prompt=prompt, model=model_name, system=None,
                        temperature=adjusted_temp, timeout=timeout, stream=False,
                        max_tokens=None, search=search, schema=None,
                        reasoning_effort=effective_effort,
                    )
                finally:
                    sys.stdout = old_stdout
            else:
                if search:
                    _build_search_kwargs(provider, model_name)

                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    text = _openai_chat(
                        prompt=prompt, model=model_name, provider=provider,
                        system=None, temperature=adjusted_temp, timeout=timeout,
                        stream=False, max_tokens=None, schema=None,
                        reasoning_effort=effective_effort,
                    )
                finally:
                    sys.stdout = old_stdout

            elapsed = time.time() - provider_start
            logger.debug(
                f"{provider} complete",
                {"elapsed_sec": round(elapsed, 2), "response_length": len(text) if text else 0},
            )

            return (provider, text, None)

        except Exception as error:
            elapsed = time.time() - provider_start
            logger.warn(
                f"{provider} failed",
                {"elapsed_sec": round(elapsed, 2), "error": str(error)},
            )
            return (provider, None, str(error))

    # Call providers in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = [executor.submit(call_provider, p) for p in providers]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Sort by provider name for consistent output
    results.sort(key=lambda x: x[0])

    # Display results
    print()
    for provider, text, error in results:
        print("=" * 60)
        print(provider.upper())
        print("=" * 60)

        if error:
            print(f"Error: {error}")
        else:
            print(text)
        print()

    elapsed = time.time() - start_time
    successful = sum(1 for _, text, error in results if text and not error)
    failed = len(results) - successful

    logger.info(
        "Comparison complete",
        {
            "total_elapsed_sec": round(elapsed, 2),
            "successful": successful,
            "failed": failed,
        },
    )


def list_providers() -> list[str]:
    """List available providers"""
    return list(PROVIDER_CONFIGS.keys())
