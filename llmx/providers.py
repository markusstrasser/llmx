"""Provider management using LiteLLM"""

import os
import sys
import time
from typing import Optional, Dict, Any
from litellm import completion, stream_chunk_builder
from rich.console import Console

from .logger import logger

console = Console()


# ============================================================================
# Structured Error Types — distinct exit codes for agent consumption
# ============================================================================

# Exit codes: callers (agents, scripts) can branch on these
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
    # LiteLLM maps reasoning_effort -> thinkingConfig.thinkingBudget natively
    # Gemini 3.x defaults to thinkingLevel=high server-side, so no default_effort needed
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

# Provider configurations
PROVIDER_CONFIGS = {
    "google": {
        "model": "gemini/gemini-3.1-pro-preview",  # Default: Gemini 3.1 Pro (Feb 2026) - THINKING MODEL
        "legacy_model": "gemini/gemini-3-pro-preview",  # Legacy: Gemini 3 Pro (Nov 2025)
        "env_var": "GEMINI_API_KEY or GOOGLE_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
        "flash_model": "gemini/gemini-3-flash-preview",  # Gemini 3 Flash: Pro-grade reasoning with Flash speed
        "flash_lite_model": "gemini/gemini-3.1-flash-lite-preview",  # Budget: file/semantic search only
    },
    "openai": {
        "model": "gpt-5.4",  # Default: GPT-5.4 (Mar 2026)
        "legacy_model": "gpt-5.2",  # Legacy: GPT-5.2
        "env_var": "OPENAI_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "anthropic": {
        "model": "anthropic/claude-opus-4-6",  # Default: Claude Opus 4.6 (Feb 2026)
        "legacy_model": "anthropic/claude-sonnet-4-6",  # Legacy: Claude Sonnet 4.6
        "env_var": "ANTHROPIC_API_KEY",
        "temperature_range": (0.0, 1.0),
        "supports_streaming": True,
    },
    "xai": {
        "model": "xai/grok-4",  # Default: Grok 4 Flagship (Jul 2025) - Best quality ($3/15 per M tokens)
        "fast_model": "xai/grok-4-1-fast-reasoning",  # Fast: Grok 4.1 reasoning (Nov 2025, $0.20/0.50)
        "non_thinking_model": "xai/grok-4-1-fast-non-reasoning",  # Fast non-thinking variant
        "legacy_model": "xai/grok-beta",  # Legacy
        "env_var": "XAI_API_KEY or GROK_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "deepseek": {
        "model": "deepseek/deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "openrouter": {
        "model": "openrouter/openai/gpt-4o",
        "env_var": "OPENROUTER_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "kimi": {
        "model": "moonshot/kimi-k2.5",  # Default: Kimi K2.5 (Jan 2026), thinking model
        "old_model": "moonshot/kimi-k2-thinking",  # Legacy: K2 thinking (Nov 2025)
        "legacy_model": "moonshot/kimi-k2-0711-preview",  # Legacy: July 2025, 128K context
        "env_var": "MOONSHOT_API_KEY or KIMI_API_KEY",
        "temperature_range": (0.0, 1.0),
        "supports_streaming": True,
    },
    "cerebras": {
        "model": "cerebras/qwen-3-coder-480b",  # Latest: Aug 2025, #2 coding model, 480B params
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
    """Info log when using Flash Lite for awareness (3.1 Flash-Lite is capable: 1432 Elo, 86.9% GPQA)"""
    if not model_name or "flash" not in model_name.lower():
        return

    # Gemini 3 Flash (preview) is a full-capability model, no note needed
    if "gemini-3-flash" in model_name.lower() or "3-flash" in model_name.lower():
        return

    # Only note for lite models
    if "lite" not in model_name.lower():
        return

    logger.debug(
        "Using Flash Lite",
        {"model": model_name, "note": "$0.25/M in, $1.50/M out, 1M context"}
    )


def get_model_name(provider: str, model: Optional[str] = None, use_old: bool = False) -> str:
    """Get model name for provider"""
    if model:
        # If model already has a prefix, use as-is
        if "/" in model:
            return model

        # Add prefix for providers that need it
        # OpenAI doesn't need prefix, most others do
        # Special cases: kimi uses moonshot/, google uses gemini/
        if provider == "openai":
            return model
        elif provider == "kimi":
            return f"moonshot/{model}"
        elif provider == "google":
            return f"gemini/{model}"
        else:
            return f"{provider}/{model}"

    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    # Check if use_old flag is set and old_model exists
    if use_old and "old_model" in config:
        return config["old_model"]

    return config["model"]


def get_model_restriction(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model-specific parameter restrictions"""
    if not model_name:
        return None

    model_lower = model_name.lower()

    # Sort keys by length descending to match longest (most specific) first
    # This ensures "gpt-5-codex" matches before "gpt-5"
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
    """
    Validate and adjust temperature for model/provider.
    Returns (adjusted_temperature, was_adjusted)
    """
    # Check model-specific restrictions first
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

    # Check provider-specific ranges
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


def _build_search_kwargs(provider: str, model_name: str) -> dict:
    """Build provider-specific kwargs for web search grounding.

    Returns dict to merge into completion_kwargs. Warns if provider unsupported.
    """
    if provider == "google":
        return {"tools": [{"googleSearch": {}}]}
    elif provider == "anthropic":
        return {"web_search_options": {"search_context_size": "medium"}}
    elif provider == "xai":
        # xAI web search requires Responses API (/v1/responses), but LiteLLM completion()
        # still routes to /v1/chat/completions. Not supported until LiteLLM fixes routing.
        logger.warn(
            "xAI web search requires the Responses API which LiteLLM completion() doesn't support yet — ignoring --search"
        )
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
    timeout: int = 120,
    search: bool = False,
    system: Optional[str] = None,
    schema: Optional[dict] = None,
    max_tokens: Optional[int] = None,
) -> None:
    """Execute chat with single provider"""
    # CLI backend handling — intercept before any LiteLLM-specific logic
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
            cli_provider, schema, system, search, stream, reasoning_effort
        )
        if fallback_reason:
            api_provider = CLI_PROVIDERS[cli_provider]["api_fallback"]
            logger.info(f"[cli→api] {cli_provider} → {api_provider} ({fallback_reason})")
            provider = api_provider
            model = cli_model
            # Fall through to normal LiteLLM flow
        else:
            text = cli_chat(cli_provider, prompt, cli_model, timeout, schema=schema)
            if text is not None:
                print(text)
                return
            # CLI failed — fall back to API
            api_provider = CLI_PROVIDERS[cli_provider]["api_fallback"]
            logger.info(f"[cli→api] {cli_provider} → {api_provider} (CLI returned error)")
            provider = api_provider
            model = cli_model
            # Fall through to normal LiteLLM flow

    start_time = time.time()
    model_name = model or "default"  # Initialize for error handling

    try:
        check_api_key(provider)
        model_name = get_model_name(provider, model, use_old)
        requested_reasoning_effort = reasoning_effort

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
                # Model doesn't support reasoning_effort
                logger.warn(
                    f"Model {model_name} does not support --reasoning-effort parameter (ignoring)",
                    {"model": model_name, "provider": provider}
                )
                reasoning_effort = None  # Don't pass it to API
            elif restriction.get("reasoning_effort_levels"):
                # Check if the level is valid for this model
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

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build completion kwargs
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": adjusted_temp,
            "timeout": timeout,
        }

        # Max output tokens (critical for Gemini which defaults to 8K)
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
            logger.debug(f"Using max_tokens: {max_tokens}")

        # Add reasoning_effort for models that support it (OpenAI, Gemini)
        # LiteLLM maps reasoning_effort to thinkingConfig for Gemini natively
        if reasoning_effort and provider in ("openai", "google"):
            completion_kwargs["reasoning_effort"] = reasoning_effort
            logger.debug(f"Using reasoning_effort: {reasoning_effort}")

        # Add JSON schema for structured output
        # LiteLLM handles translation to provider-native format (OpenAI, Gemini, etc.)
        if schema:
            completion_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
            logger.debug("Structured output enabled", {"provider": provider})

        # Add web search grounding
        if search:
            search_kwargs = _build_search_kwargs(provider, model_name)
            completion_kwargs.update(search_kwargs)
            logger.debug(f"Web search enabled", {"provider": provider, "search_kwargs": search_kwargs})

        if stream:
            # Streaming mode
            logger.debug("Starting streaming generation")

            response = completion(
                **completion_kwargs,
                stream=True,
            )

            total_chars = 0
            for chunk in response:
                if hasattr(chunk.choices[0].delta, "content"):
                    content = chunk.choices[0].delta.content
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        total_chars += len(content)

            sys.stdout.write("\n")
            sys.stdout.flush()

            elapsed = time.time() - start_time
            logger.debug(
                "Streaming complete",
                {
                    "total_chars": total_chars,
                    "elapsed_sec": round(elapsed, 2),
                    "chars_per_sec": int(total_chars / elapsed) if elapsed > 0 else 0,
                },
            )
        else:
            # Non-streaming mode
            logger.debug("Starting non-streaming generation")

            response = completion(
                **completion_kwargs,
                stream=False,
            )

            text = response.choices[0].message.content
            print(text)

            elapsed = time.time() - start_time
            logger.debug(
                "Generation complete",
                {"response_length": len(text), "elapsed_sec": round(elapsed, 2)},
            )

    except TimeoutError as error:
        elapsed = time.time() - start_time
        logger.error(
            f"Request timeout after {elapsed:.1f}s",
            {"provider": provider, "model": model_name, "timeout": timeout}
        )
        raise TimeoutError_(
            f"Request timed out after {timeout}s. Model may be overloaded or prompt too large.",
            provider=provider, model=model_name, status_code=0,
        ) from error
    except Exception as error:
        elapsed = time.time() - start_time
        error_type = error.__class__.__name__
        error_msg = str(error)

        # Classify error and raise typed exception
        if "rate_limit" in error_msg.lower() or "429" in error_msg or "503" in error_msg:
            status = 429 if "429" in error_msg else 503 if "503" in error_msg else 429
            logger.error("Rate limit exceeded", {"provider": provider, "model": model_name})
            raise RateLimitError(
                f"Rate limit exceeded for {provider}/{model_name}. Wait or use --fallback.",
                provider=provider, model=model_name, status_code=status,
            ) from error
        elif "invalid" in error_msg.lower() and "api" in error_msg.lower():
            logger.error("Invalid API key", {"provider": provider})
            raise ApiKeyError(
                f"Invalid API key for {provider}. Check your API key configuration.",
                provider=provider, model=model_name,
            ) from error
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            logger.error("Timeout", {"provider": provider, "model": model_name})
            raise TimeoutError_(
                f"Request timed out for {provider}/{model_name}.",
                provider=provider, model=model_name,
            ) from error
        elif "context" in error_msg.lower() and ("too" in error_msg.lower() or "length" in error_msg.lower()):
            logger.error("Context too large", {"provider": provider, "model": model_name})
            raise ModelError(
                f"Context too large for {model_name}: {error_msg}",
                provider=provider, model=model_name,
            ) from error
        elif "temperature" in error_msg.lower():
            logger.error("Temperature parameter error", {"provider": provider, "model": model_name})
            raise ModelError(
                f"Temperature error with {model_name}: {error_msg}",
                provider=provider, model=model_name,
            ) from error
        else:
            logger.error(
                "Chat failed",
                {"provider": provider, "model": model_name, "error_type": error_type, "elapsed": f"{elapsed:.1f}s"}
            )
            raise LlmxError(
                f"{error_type}: {error_msg}",
                provider=provider, model=model_name,
            ) from error


def compare(
    prompt: str,
    providers: list[str],
    temperature: float,
    reasoning_effort: Optional[str],
    debug: bool,
    json_output: bool,
    use_old: bool = False,
    user_specified_temp: bool = False,
    timeout: int = 120,
    search: bool = False,
) -> None:
    """Compare responses from multiple providers"""
    import concurrent.futures

    start_time = time.time()
    logger.info(f"Comparing {len(providers)} providers")

    def call_provider(provider: str) -> tuple[str, Optional[str], Optional[str]]:
        """Call a single provider and return (provider, text, error)"""
        provider_start = time.time()
        try:
            logger.debug(f"Calling {provider}")

            check_api_key(provider)
            model_name = get_model_name(provider, use_old=use_old)

            # Validate and adjust temperature for this provider/model
            adjusted_temp, was_adjusted = validate_and_adjust_temperature(
                temperature, model_name, provider, user_specified_temp
            )

            messages = [{"role": "user", "content": prompt}]

            # Build completion kwargs
            completion_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": adjusted_temp,
                "stream": False,
                "timeout": timeout,
            }

            # Add reasoning_effort for models that support it
            if reasoning_effort and provider in ("openai", "google"):
                completion_kwargs["reasoning_effort"] = reasoning_effort
                logger.debug(f"Using reasoning_effort: {reasoning_effort}")

            # Add web search grounding
            if search:
                search_kwargs = _build_search_kwargs(provider, model_name)
                completion_kwargs.update(search_kwargs)

            response = completion(**completion_kwargs)

            text = response.choices[0].message.content
            elapsed = time.time() - provider_start

            logger.debug(
                f"{provider} complete",
                {"elapsed_sec": round(elapsed, 2), "response_length": len(text)},
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
    print()  # Blank line
    for provider, text, error in results:
        print("=" * 60)
        print(provider.upper())
        print("=" * 60)

        if error:
            print(f"❌ Error: {error}")
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
