"""Provider management using LiteLLM"""

import os
import sys
import time
from typing import Optional
from litellm import completion, stream_chunk_builder
from rich.console import Console

from .logger import logger

console = Console()

# Provider configurations
PROVIDER_CONFIGS = {
    "google": {
        "model": "gemini/gemini-2.5-pro",
        "env_var": "GEMINI_API_KEY or GOOGLE_API_KEY",
    },
    "openai": {
        "model": "gpt-4o",
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "env_var": "ANTHROPIC_API_KEY",
    },
    "xai": {
        "model": "xai/grok-beta",
        "env_var": "XAI_API_KEY or GROK_API_KEY",
    },
    "deepseek": {
        "model": "deepseek/deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
    },
    "openrouter": {
        "model": "openrouter/openai/gpt-4o",
        "env_var": "OPENROUTER_API_KEY",
    },
    "kimi": {
        "model": "moonshot/kimi-k2-0905-preview",  # Latest: Sept 2025, 256K context
        "old_model": "moonshot/kimi-k2-0711-preview",  # Old: July 2025, 128K context
        "env_var": "MOONSHOT_API_KEY or KIMI_API_KEY",
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


def get_model_name(provider: str, model: Optional[str] = None, use_old: bool = False) -> str:
    """Get model name for provider"""
    if model:
        # If model already has a prefix, use as-is
        if "/" in model:
            return model

        # Add prefix for providers that need it
        # OpenAI doesn't need prefix, everyone else does
        # Kimi uses moonshot/ prefix instead of kimi/
        if provider == "openai":
            return model
        elif provider == "kimi":
            return f"moonshot/{model}"
        else:
            return f"{provider}/{model}"

    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    # Check if use_old flag is set and old_model exists
    if use_old and "old_model" in config:
        return config["old_model"]

    return config["model"]


def check_api_key(provider: str) -> None:
    """Check if API key is available for provider"""
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    # Check common environment variables
    key_vars = config["env_var"].replace(" or ", ",").split(",")
    for var in key_vars:
        var = var.strip()
        if os.getenv(var):
            logger.debug(f"Found API key: {var}")
            return

    # No key found
    logger.error(f"API key missing for {provider}", {"env_var": config["env_var"]})
    raise RuntimeError(
        f"API key not found for provider '{provider}'. "
        f"Set one of: {config['env_var']}"
    )


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
) -> None:
    """Execute chat with single provider"""
    start_time = time.time()

    try:
        check_api_key(provider)
        model_name = get_model_name(provider, model, use_old)

        logger.debug(
            "Starting chat",
            {
                "provider": provider,
                "model": model_name,
                "temperature": temperature,
                "stream": stream,
                "prompt_length": len(prompt),
            },
        )

        messages = [{"role": "user", "content": prompt}]

        # Build completion kwargs
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
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

    except Exception as error:
        logger.error("Chat failed", {"provider": provider, "error": str(error)})
        raise


def compare(
    prompt: str,
    providers: list[str],
    temperature: float,
    reasoning_effort: Optional[str],
    debug: bool,
    json_output: bool,
    use_old: bool = False,
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

            messages = [{"role": "user", "content": prompt}]

            # Build completion kwargs
            completion_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
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
