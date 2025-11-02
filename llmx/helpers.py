"""Utility helpers for common LLM patterns

Provides decorators and functions for:
- Retry with exponential backoff
- Response caching
- Input validation
- Output formatting
"""

import time
import hashlib
import json
from functools import wraps
from typing import Callable, Any, Optional, Tuple


# Simple in-memory cache
# Format: {key: (value, timestamp)}
_cache: dict[str, Tuple[Any, float]] = {}


def retry(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Initial backoff in seconds, doubles each retry (default: 1.0)
        exceptions: Tuple of exceptions to catch (default: all exceptions)

    Returns:
        Decorated function that retries on failure

    Example:
        >>> from llmx import chat
        >>> from llmx.helpers import retry
        >>>
        >>> @retry(max_attempts=3, backoff=2.0)
        >>> def flaky_call():
        ...     return chat("Your prompt", provider="openai")
        >>>
        >>> response = flaky_call()  # Retries up to 3 times with 2s, 4s, 8s backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed, re-raise
                        raise

                    # Calculate backoff (exponential)
                    wait = backoff * (2 ** attempt)
                    print(f"⚠ Attempt {attempt + 1} failed: {e}")
                    print(f"  Retrying in {wait}s... ({max_attempts - attempt - 1} attempts left)")
                    time.sleep(wait)

            # Should never reach here
            return func(*args, **kwargs)

        return wrapper
    return decorator


def cache(ttl: Optional[int] = None):
    """Cache decorator for LLM calls

    Args:
        ttl: Time-to-live in seconds (None = cache forever)

    Returns:
        Decorated function with caching

    Example:
        >>> from llmx import chat
        >>> from llmx.helpers import cache
        >>>
        >>> @cache(ttl=3600)  # Cache for 1 hour
        >>> def expensive_analysis(code):
        ...     return chat(f"Analyze this code: {code}", provider="gpt-5-pro")
        >>>
        >>> result1 = expensive_analysis("def foo(): pass")  # Calls LLM
        >>> result2 = expensive_analysis("def foo(): pass")  # Returns cached result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name, args, and kwargs
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            key = hashlib.md5(key_str.encode()).hexdigest()

            # Check cache
            if key in _cache:
                cached_value, cached_time = _cache[key]
                # Check if still valid
                if ttl is None or (time.time() - cached_time) < ttl:
                    return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            _cache[key] = (result, time.time())

            return result

        # Add cache management methods
        wrapper.clear_cache = lambda: _cache.clear()
        wrapper.cache_info = lambda: {
            "size": len(_cache),
            "ttl": ttl,
        }

        return wrapper

    return decorator


def clear_cache():
    """Clear all cached results

    Example:
        >>> from llmx.helpers import clear_cache
        >>> clear_cache()  # Clears all cached LLM responses
    """
    _cache.clear()


def validate_prompt(
    prompt: str,
    min_length: int = 1,
    max_length: int = 100000,
    strip: bool = True
) -> str:
    """Validate and clean prompt

    Args:
        prompt: Input prompt to validate
        min_length: Minimum allowed length (default: 1)
        max_length: Maximum allowed length (default: 100000)
        strip: Whether to strip whitespace (default: True)

    Returns:
        Cleaned prompt

    Raises:
        ValueError: If prompt is invalid

    Example:
        >>> from llmx.helpers import validate_prompt
        >>> prompt = validate_prompt("  Hello  ", min_length=3)
        >>> print(prompt)
        Hello
    """
    if not isinstance(prompt, str):
        raise ValueError(f"Prompt must be a string, got {type(prompt)}")

    if strip:
        prompt = prompt.strip()

    if not prompt or len(prompt) < min_length:
        raise ValueError(f"Prompt too short: {len(prompt)} < {min_length}")

    if len(prompt) > max_length:
        raise ValueError(f"Prompt too long: {len(prompt)} > {max_length}")

    return prompt


def format_response(response, format: str = "text") -> Any:
    """Format LLM response

    Args:
        response: Response object or string
        format: Output format (text, json, markdown, dict)

    Returns:
        Formatted output

    Raises:
        ValueError: If format is unknown
        json.JSONDecodeError: If format=json and content is not valid JSON

    Example:
        >>> from llmx import chat
        >>> from llmx.helpers import format_response
        >>>
        >>> response = chat("Generate JSON: {name, age}", provider="openai")
        >>> data = format_response(response, format="json")
        >>> print(data["name"])
    """
    # Extract content from Response object or use string directly
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    if format == "text":
        return content

    elif format == "json":
        # Parse as JSON
        return json.loads(content)

    elif format == "markdown":
        # Pretty print as markdown (requires rich)
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            console = Console()
            console.print(Markdown(content))
            return content
        except ImportError:
            print("⚠ rich not installed, falling back to plain text")
            return content

    elif format == "dict":
        # Return response as dict if it has attributes
        if hasattr(response, "__dict__"):
            return response.__dict__
        elif hasattr(response, "_asdict"):
            return response._asdict()
        else:
            return {"content": content}

    else:
        raise ValueError(f"Unknown format: {format}. Use text, json, markdown, or dict")


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length

    Args:
        text: Text to truncate
        max_length: Maximum length (default: 100)
        suffix: Suffix to add when truncated (default: "...")

    Returns:
        Truncated text

    Example:
        >>> from llmx.helpers import truncate
        >>> long_text = "This is a very long text" * 10
        >>> print(truncate(long_text, max_length=50))
        This is a very long textThis is a very long te...
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count for text

    Args:
        text: Input text
        chars_per_token: Average characters per token (default: 4)

    Returns:
        Estimated token count

    Note:
        This is a rough estimate. Actual token count may vary by model.

    Example:
        >>> from llmx.helpers import estimate_tokens
        >>> text = "Hello, how are you?"
        >>> tokens = estimate_tokens(text)
        >>> print(f"Estimated tokens: {tokens}")
        Estimated tokens: 5
    """
    return len(text) // chars_per_token


# Export public API
__all__ = [
    "retry",
    "cache",
    "clear_cache",
    "validate_prompt",
    "format_response",
    "truncate",
    "estimate_tokens",
]
