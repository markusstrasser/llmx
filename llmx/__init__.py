"""llmx - Unified API wrapper for 100+ LLM providers

Simple tool for calling LLM APIs from scripts, skills, and Python code.

CLI Usage:
    llmx "What is 2+2?"
    llmx --provider openai "Explain Python"
    llmx --compare "tabs or spaces?"

Python API Usage:
    from llmx import LLM, chat, batch

    # Simple one-shot
    response = chat("What is 2+2?", provider="openai")
    print(response.content)

    # Stateful client
    llm = LLM(provider="google", model="gemini-2.5-pro")
    r1 = llm.chat("First question")
    r2 = llm.chat("Second question")

    # Batch processing
    responses = batch(["Q1", "Q2", "Q3"], provider="openai", parallel=3)

Inspection:
    from llmx.inspect import last_request, last_response, stats

    chat("Hello")
    print(last_request())   # See what was sent
    print(last_response())  # See what was received
    print(stats())          # Get usage statistics

Helpers:
    from llmx.helpers import retry, cache

    @retry(max_attempts=3)
    def flaky_call():
        return chat("prompt")

    @cache(ttl=3600)  # Cache for 1 hour
    def expensive_call(code):
        return chat(f"Analyze: {code}")
"""

__version__ = "0.3.0"

# Import public API
from .api import LLM, Response, chat, batch

# Import for convenience (optional, can be imported from submodules)
from . import inspect
from . import helpers

__all__ = [
    # Core API
    "LLM",
    "Response",
    "chat",
    "batch",
    # Version
    "__version__",
    # Submodules (for from llmx.inspect import ...)
    "inspect",
    "helpers",
]
