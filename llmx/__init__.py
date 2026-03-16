"""llmx - Unified CLI and API for LLM providers via native SDKs

Simple tool for calling LLM APIs from scripts, skills, and Python code.

CLI:
    llmx "prompt"
    llmx -p openai "prompt"
    llmx --search "latest news on X"

Python:
    from llmx import chat
    response = chat("What is 2+2?", provider="openai")
    print(response.content)
"""

__version__ = "0.6.0"

# Import public API
from .api import LLM, Response, chat, batch, batch_submit, batch_status, batch_get
from .providers import (
    LlmxError, RateLimitError, QuotaError, TimeoutError_, ApiKeyError, ModelError,
    EXIT_SUCCESS, EXIT_GENERAL, EXIT_API_KEY, EXIT_RATE_LIMIT, EXIT_TIMEOUT, EXIT_MODEL_ERROR, EXIT_QUOTA,
)

# Import for convenience (optional, can be imported from submodules)
from . import inspect
from . import helpers

__all__ = [
    # Core API
    "LLM",
    "Response",
    "chat",
    "batch",
    "batch_submit",
    "batch_status",
    "batch_get",
    # Error types
    "LlmxError",
    "RateLimitError",
    "QuotaError",
    "TimeoutError_",
    "ApiKeyError",
    "ModelError",
    # Exit codes
    "EXIT_SUCCESS",
    "EXIT_GENERAL",
    "EXIT_API_KEY",
    "EXIT_RATE_LIMIT",
    "EXIT_TIMEOUT",
    "EXIT_MODEL_ERROR",
    "EXIT_QUOTA",
    # Version
    "__version__",
    # Submodules (for from llmx.inspect import ...)
    "inspect",
    "helpers",
]
