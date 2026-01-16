"""Provider management using LiteLLM"""

import os
import sys
import time
from typing import Optional, Dict, Any
from litellm import completion, stream_chunk_builder
from rich.console import Console

from .logger import logger

console = Console()

# Model-specific parameter restrictions
MODEL_RESTRICTIONS = {
    # OpenAI GPT-5.1 thinking models: temperature=1 only, support reasoning_effort
    "gpt-5.1": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"]},
    "gpt-5.1-mini": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"]},
    # Legacy GPT-5 models
    "gpt-5": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"]},
    "gpt-5-pro": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["minimal", "low", "medium", "high"]},
    "gpt-5-codex": {"temperature": 1.0, "fixed": True, "reasoning_effort": True, "reasoning_effort_levels": ["low", "medium", "high"]},  # No minimal
    # Kimi thinking models (only k2-thinking, not instruct variants)
    "kimi-k2-thinking": {"temperature": 1.0, "fixed": True, "reasoning_effort": False},  # No reasoning_effort support
    # Note: kimi-k2-0905-preview (instruct) and kimi-k2-0711-preview do NOT have these restrictions
}

# Provider configurations
PROVIDER_CONFIGS = {
    "google": {
        "model": "gemini/gemini-3-pro-preview",  # Default: Gemini 3 Pro (Nov 2025) - THINKING MODEL
        "legacy_model": "gemini/gemini-2.5-pro",  # Legacy: Gemini 2.5 Pro
        "env_var": "GEMINI_API_KEY or GOOGLE_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
        "flash_model": "gemini/gemini-3-flash-preview",  # Gemini 3 Flash: Pro-grade reasoning with Flash speed
        "flash_lite_model": "gemini/gemini-2.5-flash-lite",  # Budget: file/semantic search only
    },
    "openai": {
        "model": "gpt-5.1",  # Default: GPT-5.1 THINKING model (Nov 2025)
        "legacy_model": "gpt-4o",  # Legacy: GPT-4o
        "env_var": "OPENAI_API_KEY",
        "temperature_range": (0.0, 2.0),
        "supports_streaming": True,
    },
    "anthropic": {
        "model": "anthropic/claude-opus-4-5",  # Default: Claude 4.5 Opus THINKING model (Nov 2025)
        "legacy_model": "anthropic/claude-sonnet-4-5",  # Legacy: Claude Sonnet 4.5
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
        "model": "moonshot/kimi-k2-thinking",  # Default: Nov 2025, reasoning/thinking model
        "old_model": "moonshot/kimi-k2-0905-preview",  # Instruct (non-thinking): Sept 2025, 256K context, faster
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
    """Warn if using Gemini Flash Lite for non-search tasks (Gemini 3 Flash is OK for general use)"""
    if not model_name or "flash" not in model_name.lower():
        return

    # Gemini 3 Flash (preview) is a full-capability model, no warning needed
    if "gemini-3-flash" in model_name.lower() or "3-flash" in model_name.lower():
        return

    # Only warn for lite/budget flash models
    if "lite" not in model_name.lower() and "2.5-flash" not in model_name.lower():
        return

    # Check if prompt looks like a search task
    search_keywords = ["search", "find", "lookup", "retrieve", "locate", "query", "semantic", "similar"]
    prompt_lower = prompt.lower()

    is_search_task = any(keyword in prompt_lower for keyword in search_keywords)

    if not is_search_task:
        logger.warn(
            "Flash Lite models should ONLY be used for file/semantic search tasks",
            {"model": model_name, "note": "Use gemini-3-flash or gemini-3-pro-preview for reasoning/analysis"}
        )
        logger.warn(
            "⚠️  Flash Lite will underperform on this task. Consider using gemini-3-flash instead."
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
) -> None:
    """Execute chat with single provider"""
    start_time = time.time()
    model_name = model or "default"  # Initialize for error handling

    try:
        check_api_key(provider)
        model_name = get_model_name(provider, model, use_old)

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

        logger.debug(
            "Starting chat",
            {
                "provider": provider,
                "model": model_name,
                "temperature": adjusted_temp,
                "stream": stream,
                "prompt_length": len(prompt),
            },
        )

        # Tip for models that support reasoning_effort
        if restriction and restriction.get("reasoning_effort") and not reasoning_effort and provider == "openai":
            logger.info(
                f"Tip: {model_name} supports --reasoning-effort (low/medium/high) for better results"
            )

        messages = [{"role": "user", "content": prompt}]

        # Build completion kwargs
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": adjusted_temp,
            "timeout": timeout,
        }

        # Add reasoning_effort for OpenAI models if specified
        if reasoning_effort and provider == "openai":
            completion_kwargs["reasoning_effort"] = reasoning_effort
            logger.debug(f"Using reasoning_effort: {reasoning_effort}")

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
        raise RuntimeError(
            f"Request timed out after {timeout}s. The model may be overloaded or the prompt too complex. "
            f"Try again or use a different model."
        ) from error
    except Exception as error:
        elapsed = time.time() - start_time
        error_type = error.__class__.__name__
        error_msg = str(error)

        # Provide helpful error messages
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            logger.error("Rate limit exceeded", {"provider": provider, "model": model_name})
            raise RuntimeError(
                f"Rate limit exceeded for {provider}. Wait a moment and try again."
            ) from error
        elif "invalid" in error_msg.lower() and "api" in error_msg.lower():
            logger.error("Invalid API key", {"provider": provider})
            raise RuntimeError(
                f"Invalid API key for {provider}. Check your API key configuration."
            ) from error
        elif "temperature" in error_msg.lower():
            logger.error("Temperature parameter error", {"provider": provider, "model": model_name})
            raise RuntimeError(
                f"Temperature error with {model_name}: {error_msg}\n"
                f"Some models have restrictions. Try without --temperature or use --debug for details."
            ) from error
        else:
            logger.error(
                "Chat failed",
                {"provider": provider, "model": model_name, "error_type": error_type, "elapsed": f"{elapsed:.1f}s"}
            )
            raise RuntimeError(f"{error_type}: {error_msg}") from error


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

            # Add reasoning_effort for OpenAI models if specified
            if reasoning_effort and provider == "openai":
                completion_kwargs["reasoning_effort"] = reasoning_effort
                logger.debug(f"Using reasoning_effort: {reasoning_effort}")

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
