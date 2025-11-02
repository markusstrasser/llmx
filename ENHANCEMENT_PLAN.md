# llmx Enhancement Plan

> Transform llmx from CLI-only to full-featured library with API, helpers, and inspection tools

## Current State (CLI-Only)

```python
# Can only use via shell
$ llmx "prompt"
```

**Limitations:**
- No programmatic API
- No inspection/debugging tools
- No helper utilities (retry, cache, batch)
- Hard to use from Python scripts
- Can't inspect requests/responses

## Target State (Library + CLI)

```python
# Programmatic API
from llmx import LLM, chat, batch, inspect

# Simple usage
response = chat("What is 2+2?", provider="openai")

# Full control
llm = LLM(provider="openai", model="gpt-5-pro")
response = llm.chat("Your prompt", temperature=0.7)

# Batch processing
responses = batch(prompts, provider="google")

# Inspection
inspect.last_request()   # See what was sent
inspect.last_response()  # See what was received
inspect.stats()          # Token usage, timing, costs
```

## Architecture

### Phase 1: Core API (300 LOC)

```
llmx/
├── api.py           (NEW - Programmatic API)
│   ├── LLM class    (Stateful client)
│   ├── chat()       (Simple function API)
│   ├── batch()      (Batch processing)
│   └── stream()     (Streaming responses)
│
├── inspect.py       (NEW - Inspection tools)
│   ├── Trace        (Request/response capture)
│   ├── Stats        (Token usage, timing)
│   └── History      (Keep last N calls)
│
├── helpers.py       (NEW - Utility functions)
│   ├── retry()      (Auto-retry with backoff)
│   ├── cache()      (Response caching)
│   ├── validate()   (Input validation)
│   └── format()     (Output formatting)
│
├── cli.py           (EXISTING - Enhanced to use api.py)
├── providers.py     (EXISTING - Used by api.py)
└── logger.py        (EXISTING)
```

### Phase 2: Advanced Features (200 LOC)

```
llmx/
├── conversation.py  (NEW - Multi-turn conversations)
│   ├── Conversation class
│   └── Auto-context management
│
├── tools.py         (NEW - Function calling)
│   ├── Tool registry
│   └── Auto-execute tools
│
└── templates.py     (NEW - Prompt templates)
    ├── Template system
    └── Built-in templates
```

## Detailed Design

### 1. Core API (`api.py`)

```python
"""Programmatic API for llmx"""

from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass
import threading

from .providers import chat as _chat, get_model_name, check_api_key
from .inspect import Trace, capture_call


# Thread-local storage for inspection
_local = threading.local()


@dataclass
class Response:
    """LLM response with metadata"""
    content: str
    provider: str
    model: str
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    latency: float  # seconds
    raw: Any  # Raw LiteLLM response


class LLM:
    """Stateful LLM client for multiple calls"""

    def __init__(
        self,
        provider: str = "google",
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize LLM client

        Args:
            provider: Provider name (google, openai, anthropic, xai, deepseek)
            model: Model name (overrides provider default)
            temperature: Temperature 0-1
            **kwargs: Additional provider-specific args
        """
        self.provider = provider
        self.model = model or get_model_name(provider)
        self.temperature = temperature
        self.kwargs = kwargs

        # Validate API key on init
        check_api_key(provider)

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Response:
        """
        Send chat message

        Args:
            prompt: User prompt
            system: System message (optional)
            temperature: Override default temperature
            **kwargs: Override any init kwargs

        Returns:
            Response object with content and metadata
        """
        # Merge kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        temp = temperature if temperature is not None else self.temperature

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Call with inspection
        with capture_call(self.provider, self.model, messages) as trace:
            result = _chat(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                system=system,
                temperature=temp,
                **call_kwargs
            )

            trace.set_response(result)

        return Response(
            content=result["content"],
            provider=self.provider,
            model=self.model,
            usage=result.get("usage", {}),
            latency=result.get("latency", 0),
            raw=result
        )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream response chunks"""
        # TODO: Implement streaming
        raise NotImplementedError("Streaming not yet implemented")


# Simple function API
def chat(
    prompt: str,
    provider: str = "google",
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> Response:
    """
    Simple chat function - one-shot calls

    Args:
        prompt: User prompt
        provider: Provider name
        model: Model name
        system: System message
        temperature: Temperature 0-1
        **kwargs: Additional args

    Returns:
        Response object

    Example:
        >>> from llmx import chat
        >>> response = chat("What is 2+2?", provider="openai")
        >>> print(response.content)
        4
    """
    llm = LLM(provider=provider, model=model, temperature=temperature, **kwargs)
    return llm.chat(prompt, system=system)


def batch(
    prompts: List[str],
    provider: str = "google",
    model: Optional[str] = None,
    parallel: int = 3,
    **kwargs
) -> List[Response]:
    """
    Process multiple prompts in parallel

    Args:
        prompts: List of prompts
        provider: Provider name
        model: Model name
        parallel: Number of parallel requests
        **kwargs: Additional args

    Returns:
        List of Response objects

    Example:
        >>> from llmx import batch
        >>> prompts = ["What is 2+2?", "What is 3+3?"]
        >>> responses = batch(prompts, provider="google", parallel=2)
    """
    from concurrent.futures import ThreadPoolExecutor

    llm = LLM(provider=provider, model=model, **kwargs)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        return list(executor.map(llm.chat, prompts))
```

### 2. Inspection API (`inspect.py`)

```python
"""Inspection and debugging tools"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime


# Thread-local storage for per-thread traces
_local = threading.local()


@dataclass
class Trace:
    """Single LLM call trace"""
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
        """Mark response received"""
        self.response_time = datetime.now()
        self.response = response
        self.latency = (self.response_time - self.request_time).total_seconds()
        self.usage = response.get("usage", {})

    def set_error(self, error: Exception):
        """Mark error"""
        self.response_time = datetime.now()
        self.error = error
        self.latency = (self.response_time - self.request_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "request_time": self.request_time.isoformat(),
            "response_time": self.response_time.isoformat() if self.response_time else None,
            "latency": self.latency,
            "usage": self.usage,
            "error": str(self.error) if self.error else None,
        }


class Inspector:
    """Global inspector for all LLM calls"""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._history: List[Trace] = []
        self._lock = threading.Lock()

    def add_trace(self, trace: Trace):
        """Add trace to history"""
        with self._lock:
            self._history.append(trace)
            if len(self._history) > self.max_history:
                self._history.pop(0)

    def last_request(self) -> Optional[Dict[str, Any]]:
        """Get last request details"""
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
        """Get last response details"""
        with self._lock:
            if not self._history:
                return None
            trace = self._history[-1]
            if not trace.response:
                return None
            return {
                "content": trace.response.get("content"),
                "usage": trace.usage,
                "latency": trace.latency,
                "time": trace.response_time.isoformat() if trace.response_time else None,
            }

    def stats(self) -> Dict[str, Any]:
        """Get aggregate statistics"""
        with self._lock:
            if not self._history:
                return {}

            total_calls = len(self._history)
            errors = sum(1 for t in self._history if t.error)
            total_tokens = sum(
                t.usage.get("total_tokens", 0) for t in self._history if t.usage
            )
            total_latency = sum(
                t.latency for t in self._history if t.latency
            )
            avg_latency = total_latency / total_calls if total_calls else 0

            by_provider = {}
            for trace in self._history:
                if trace.provider not in by_provider:
                    by_provider[trace.provider] = {"calls": 0, "tokens": 0}
                by_provider[trace.provider]["calls"] += 1
                by_provider[trace.provider]["tokens"] += trace.usage.get("total_tokens", 0)

            return {
                "total_calls": total_calls,
                "errors": errors,
                "success_rate": (total_calls - errors) / total_calls if total_calls else 0,
                "total_tokens": total_tokens,
                "avg_latency": avg_latency,
                "by_provider": by_provider,
            }

    def history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent history"""
        with self._lock:
            return [t.to_dict() for t in self._history[-limit:]]

    def clear(self):
        """Clear history"""
        with self._lock:
            self._history.clear()


# Global inspector instance
_inspector = Inspector()


@contextmanager
def capture_call(provider: str, model: str, messages: List[Dict]):
    """Context manager to capture LLM call"""
    trace = Trace(provider=provider, model=model, messages=messages)
    try:
        yield trace
        _inspector.add_trace(trace)
    except Exception as e:
        trace.set_error(e)
        _inspector.add_trace(trace)
        raise


# Public API
def last_request():
    """Get last request"""
    return _inspector.last_request()


def last_response():
    """Get last response"""
    return _inspector.last_response()


def stats():
    """Get statistics"""
    return _inspector.stats()


def history(limit: int = 10):
    """Get call history"""
    return _inspector.history(limit)


def clear():
    """Clear history"""
    _inspector.clear()
```

### 3. Helpers (`helpers.py`)

```python
"""Utility helpers for common patterns"""

import time
import hashlib
import json
from functools import wraps
from typing import Callable, Any, Optional


# Simple in-memory cache
_cache = {}


def retry(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of attempts
        backoff: Initial backoff in seconds (doubles each retry)
        exceptions: Tuple of exceptions to catch

    Example:
        @retry(max_attempts=3, backoff=1.0)
        def flaky_call():
            return llm.chat("prompt")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = backoff * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator


def cache(ttl: Optional[int] = None):
    """
    Cache decorator for LLM calls

    Args:
        ttl: Time-to-live in seconds (None = cache forever)

    Example:
        @cache(ttl=3600)  # Cache for 1 hour
        def expensive_call(prompt):
            return llm.chat(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args/kwargs
            key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
            key = hashlib.md5(key_data.encode()).hexdigest()

            # Check cache
            if key in _cache:
                cached_value, cached_time = _cache[key]
                if ttl is None or (time.time() - cached_time) < ttl:
                    return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            _cache[key] = (result, time.time())
            return result

        return wrapper
    return decorator


def validate_prompt(prompt: str, max_length: int = 100000) -> str:
    """
    Validate and clean prompt

    Args:
        prompt: Input prompt
        max_length: Maximum allowed length

    Returns:
        Cleaned prompt

    Raises:
        ValueError: If prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if len(prompt) > max_length:
        raise ValueError(f"Prompt too long: {len(prompt)} > {max_length}")

    return prompt.strip()


def format_response(response, format: str = "text") -> Any:
    """
    Format LLM response

    Args:
        response: Response object or string
        format: Output format (text, json, markdown)

    Returns:
        Formatted output
    """
    content = response.content if hasattr(response, "content") else str(response)

    if format == "text":
        return content
    elif format == "json":
        return json.loads(content)
    elif format == "markdown":
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        console.print(Markdown(content))
        return content
    else:
        raise ValueError(f"Unknown format: {format}")
```

### 4. Enhanced CLI (`cli.py` updates)

```python
# Add new commands to cli.py

@click.group()
def cli():
    """llmx - Unified LLM API"""
    pass

@cli.command()
@click.argument("prompt")
def chat(prompt):
    """Send a chat message"""
    from .api import chat as api_chat
    response = api_chat(prompt)
    click.echo(response.content)

@cli.command()
def stats():
    """Show usage statistics"""
    from .inspect import stats as get_stats
    import json
    click.echo(json.dumps(get_stats(), indent=2))

@cli.command()
@click.option("--limit", default=10, help="Number of calls to show")
def history(limit):
    """Show call history"""
    from .inspect import history as get_history
    import json
    click.echo(json.dumps(get_history(limit), indent=2))
```

## Usage Examples

### 1. Simple API

```python
from llmx import chat

# One-liner
response = chat("What is 2+2?", provider="openai")
print(response.content)  # "4"
print(response.usage)    # {prompt_tokens: 10, completion_tokens: 2, total_tokens: 12}
print(response.latency)  # 1.23
```

### 2. Stateful Client

```python
from llmx import LLM

llm = LLM(provider="openai", model="gpt-5-pro", temperature=0.3)

# Multiple calls with same settings
r1 = llm.chat("Explain Python")
r2 = llm.chat("Explain Rust")
r3 = llm.chat("Compare them", temperature=0.7)  # Override temp
```

### 3. Batch Processing

```python
from llmx import batch

prompts = [
    "What is 2+2?",
    "What is 3+3?",
    "What is 4+4?",
]

responses = batch(prompts, provider="google", parallel=3)
for r in responses:
    print(r.content)
```

### 4. Inspection

```python
from llmx import chat
from llmx.inspect import last_request, last_response, stats

# Make a call
response = chat("Hello", provider="openai")

# Inspect what was sent
print(last_request())
# {'provider': 'openai', 'model': 'gpt-4o', 'messages': [...]}

# Inspect what was received
print(last_response())
# {'content': 'Hello!', 'usage': {...}, 'latency': 1.2}

# Get stats
print(stats())
# {'total_calls': 10, 'total_tokens': 1234, 'avg_latency': 1.5, ...}
```

### 5. Helpers

```python
from llmx import chat
from llmx.helpers import retry, cache

# Auto-retry on failure
@retry(max_attempts=3, backoff=1.0)
def flaky_call():
    return chat("Your prompt", provider="openai")

# Cache expensive calls
@cache(ttl=3600)  # 1 hour
def expensive_analysis(code):
    return chat(f"Analyze this code: {code}", provider="gpt-5-pro")
```

## Implementation Plan

### Phase 1 (1 day)
- [x] Design complete
- [ ] Implement `api.py` (LLM class, chat, batch)
- [ ] Implement `inspect.py` (Trace, Inspector)
- [ ] Update `cli.py` to use new API
- [ ] Tests

### Phase 2 (half day)
- [ ] Implement `helpers.py` (retry, cache, validate)
- [ ] Add inspection commands to CLI
- [ ] Documentation
- [ ] Examples

### Phase 3 (half day - optional)
- [ ] Conversation API (multi-turn)
- [ ] Tool calling support
- [ ] Prompt templates

## Benefits

1. **Programmatic API** - Use from Python scripts, notebooks, etc
2. **Inspection** - Debug requests/responses, track usage
3. **Helpers** - Retry, caching, validation out-of-box
4. **REPL-friendly** - Interactive exploration
5. **Backwards compatible** - CLI still works as before
6. **Testable** - Mock-friendly API

## Migration Path

**Old (CLI only):**
```bash
llmx "prompt"
```

**New (API + CLI):**
```python
# Python
from llmx import chat
response = chat("prompt")

# CLI still works
$ llmx "prompt"

# Plus new CLI commands
$ llmx stats
$ llmx history
```

Zero breaking changes!
