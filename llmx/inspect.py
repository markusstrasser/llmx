"""Inspection and debugging tools for llmx

Captures and stores traces of all LLM calls for debugging and analysis.

Usage:
    from llmx import chat
    from llmx.inspect import last_request, last_response, stats, history

    # Make some calls
    chat("Hello", provider="openai")
    chat("World", provider="google")

    # Inspect
    print(last_request())    # See last request details
    print(last_response())   # See last response details
    print(stats())           # Get aggregate statistics
    print(history(limit=5))  # Get last 5 calls
"""

import time
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime


@dataclass
class Trace:
    """Single LLM call trace

    Captures full request/response details for debugging.
    """
    provider: str
    model: str
    messages: List[Dict[str, str]]
    request_time: datetime = field(default_factory=datetime.now)
    response_time: Optional[datetime] = None
    response: Optional[Any] = None
    error: Optional[Exception] = None
    latency: Optional[float] = None
    usage: Dict[str, int] = field(default_factory=dict)

    def set_response(self, response: Any):
        """Mark response received

        Args:
            response: Response data from provider
        """
        self.response_time = datetime.now()
        self.response = response
        self.latency = (self.response_time - self.request_time).total_seconds()

        # Extract usage if available
        if isinstance(response, dict):
            self.usage = response.get("usage", {})

    def set_error(self, error: Exception):
        """Mark error occurred

        Args:
            error: Exception that occurred
        """
        self.response_time = datetime.now()
        self.error = error
        self.latency = (self.response_time - self.request_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization

        Returns:
            Dictionary representation of trace
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "request_time": self.request_time.isoformat(),
            "response_time": self.response_time.isoformat() if self.response_time else None,
            "latency": self.latency,
            "usage": self.usage,
            "error": str(self.error) if self.error else None,
            "success": self.error is None,
        }


class Inspector:
    """Global inspector for all LLM calls

    Thread-safe storage of call traces with statistics.
    """

    def __init__(self, max_history: int = 100):
        """Initialize inspector

        Args:
            max_history: Maximum number of traces to keep (default: 100)
        """
        self.max_history = max_history
        self._history: List[Trace] = []
        self._lock = threading.Lock()

    def add_trace(self, trace: Trace):
        """Add trace to history

        Args:
            trace: Trace object to add
        """
        with self._lock:
            self._history.append(trace)
            # Keep only last N traces
            if len(self._history) > self.max_history:
                self._history.pop(0)

    def last_request(self) -> Optional[Dict[str, Any]]:
        """Get last request details

        Returns:
            Dict with provider, model, messages, timestamp
            None if no requests yet
        """
        with self._lock:
            if not self._history:
                return None

            trace = self._history[-1]
            return {
                "provider": trace.provider,
                "model": trace.model,
                "messages": trace.messages,
                "time": trace.request_time.isoformat(),
            }

    def last_response(self) -> Optional[Dict[str, Any]]:
        """Get last response details

        Returns:
            Dict with content, usage, latency, timestamp
            None if no responses yet
        """
        with self._lock:
            if not self._history:
                return None

            trace = self._history[-1]
            if not trace.response:
                return None

            content = trace.response.get("content") if isinstance(trace.response, dict) else str(trace.response)

            return {
                "content": content,
                "usage": trace.usage,
                "latency": trace.latency,
                "time": trace.response_time.isoformat() if trace.response_time else None,
                "success": trace.error is None,
            }

    def stats(self) -> Dict[str, Any]:
        """Get aggregate statistics

        Returns:
            Dict with:
                - total_calls: Total number of calls
                - errors: Number of failed calls
                - success_rate: Percentage of successful calls
                - total_tokens: Total tokens used
                - avg_latency: Average latency in seconds
                - by_provider: Stats broken down by provider
        """
        with self._lock:
            if not self._history:
                return {
                    "total_calls": 0,
                    "errors": 0,
                    "success_rate": 0,
                    "total_tokens": 0,
                    "avg_latency": 0,
                    "by_provider": {},
                }

            total_calls = len(self._history)
            errors = sum(1 for t in self._history if t.error)
            total_tokens = sum(
                t.usage.get("total_tokens", 0) for t in self._history if t.usage
            )
            total_latency = sum(
                t.latency for t in self._history if t.latency
            )
            avg_latency = total_latency / total_calls if total_calls else 0

            # Break down by provider
            by_provider = {}
            for trace in self._history:
                if trace.provider not in by_provider:
                    by_provider[trace.provider] = {
                        "calls": 0,
                        "tokens": 0,
                        "errors": 0,
                        "avg_latency": 0,
                    }

                by_provider[trace.provider]["calls"] += 1
                by_provider[trace.provider]["tokens"] += trace.usage.get("total_tokens", 0)
                if trace.error:
                    by_provider[trace.provider]["errors"] += 1

            # Calculate avg latency per provider
            for provider, data in by_provider.items():
                provider_traces = [t for t in self._history if t.provider == provider]
                provider_latency = sum(t.latency for t in provider_traces if t.latency)
                data["avg_latency"] = provider_latency / data["calls"] if data["calls"] else 0

            return {
                "total_calls": total_calls,
                "errors": errors,
                "success_rate": (total_calls - errors) / total_calls if total_calls else 0,
                "total_tokens": total_tokens,
                "avg_latency": avg_latency,
                "by_provider": by_provider,
            }

    def history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent history

        Args:
            limit: Number of recent calls to return (default: 10)

        Returns:
            List of trace dicts (most recent last)
        """
        with self._lock:
            return [t.to_dict() for t in self._history[-limit:]]

    def clear(self):
        """Clear all history"""
        with self._lock:
            self._history.clear()


# Global inspector instance
_inspector = Inspector()


@contextmanager
def capture_call(provider: str, model: str, messages: List[Dict]):
    """Context manager to capture LLM call

    Args:
        provider: Provider name
        model: Model name
        messages: List of message dicts

    Yields:
        Trace object being captured

    Example:
        with capture_call("openai", "gpt-4", messages) as trace:
            result = make_api_call()
            trace.set_response(result)
    """
    trace = Trace(provider=provider, model=model, messages=messages)
    try:
        yield trace
        _inspector.add_trace(trace)
    except Exception as e:
        trace.set_error(e)
        _inspector.add_trace(trace)
        raise


# Public API functions

def last_request() -> Optional[Dict[str, Any]]:
    """Get last request details

    Returns:
        Dict with provider, model, messages, timestamp
        None if no requests yet

    Example:
        >>> from llmx import chat
        >>> from llmx.inspect import last_request
        >>> chat("Hello")
        >>> print(last_request())
        {'provider': 'google', 'model': 'gemini-2.5-pro', 'messages': [...], 'time': '2025-10-30T...'}
    """
    return _inspector.last_request()


def last_response() -> Optional[Dict[str, Any]]:
    """Get last response details

    Returns:
        Dict with content, usage, latency, timestamp
        None if no responses yet

    Example:
        >>> from llmx import chat
        >>> from llmx.inspect import last_response
        >>> chat("Hello")
        >>> print(last_response())
        {'content': 'Hello!', 'usage': {...}, 'latency': 1.2, 'time': '2025-10-30T...'}
    """
    return _inspector.last_response()


def stats() -> Dict[str, Any]:
    """Get aggregate statistics

    Returns:
        Dict with total calls, errors, tokens, latency, per-provider stats

    Example:
        >>> from llmx import chat
        >>> from llmx.inspect import stats
        >>> chat("Hello", provider="openai")
        >>> chat("World", provider="google")
        >>> print(stats())
        {
            'total_calls': 2,
            'errors': 0,
            'success_rate': 1.0,
            'total_tokens': 42,
            'avg_latency': 1.5,
            'by_provider': {
                'openai': {'calls': 1, 'tokens': 20, ...},
                'google': {'calls': 1, 'tokens': 22, ...}
            }
        }
    """
    return _inspector.stats()


def history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent call history

    Args:
        limit: Number of recent calls to return (default: 10)

    Returns:
        List of trace dicts (most recent last)

    Example:
        >>> from llmx import chat
        >>> from llmx.inspect import history
        >>> chat("Q1")
        >>> chat("Q2")
        >>> for call in history(limit=2):
        ...     print(call['messages'], call['latency'])
    """
    return _inspector.history(limit)


def clear():
    """Clear all inspection history

    Example:
        >>> from llmx.inspect import clear, stats
        >>> stats()  # {'total_calls': 10, ...}
        >>> clear()
        >>> stats()  # {'total_calls': 0, ...}
    """
    _inspector.clear()


# Export public API
__all__ = ["last_request", "last_response", "stats", "history", "clear", "capture_call"]
