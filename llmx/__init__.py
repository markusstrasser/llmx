"""llmx - Unified API wrapper for 100+ LLM providers

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

__version__ = "0.4.0"

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
