"""Programmatic API for llmx"""

import io
import os
import random
import sys
import time
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass

from .providers import (
    get_model_name,
    check_api_key,
    _build_search_kwargs,
    infer_provider_from_model,
    _normalize_model,
    _get_api_key,
    _google_chat,
    _openai_chat,
    OPENAI_COMPAT_URLS,
    RateLimitError,
)

# Dispatch auto-retry on TRANSIENT failures (RateLimitError + its ServiceUnavailableError
# subclass = 429/503/overload/connection). Exponential backoff + JITTER (the jitter matters:
# under concurrent multi-session load, deterministic backoff thundering-herds). Non-transient
# errors (QuotaError/ModelError/ApiKeyError) are NOT RateLimitError → propagate immediately.
# Override attempts via LLMX_MAX_RETRIES env or max_retries kwarg. Default 4 (3 retries).
_RETRY_BASE = 2.0  # seconds; waits ≈ 2,4,8 (+jitter)
_RETRY_CAP = 30.0  # max single backoff


def _resolve_max_attempts(call_kwargs: dict) -> int:
    raw = os.environ.get("LLMX_MAX_RETRIES")
    if raw is None:
        raw = call_kwargs.get("max_retries", 4)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 4


from .auth import auth_to_llmx_kwargs, resolve_auth
from .cli_backends import (
    CLI_PROVIDERS,
    needs_api_fallback,
    cli_chat,
    preferred_cli_provider,
)
from .logger import logger
from .inspect import capture_call


@dataclass
class Response:
    """LLM response with metadata

    Attributes:
        content: Response text
        provider: Provider name (google, openai, etc)
        model: Model name
        usage: Token usage {prompt_tokens, completion_tokens, total_tokens}
        latency: Response time in seconds
        raw: Raw response from provider
    """

    content: str
    provider: str
    model: str
    usage: Dict[str, int]
    latency: float
    raw: Any

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        tokens = self.usage.get("total_tokens", 0)
        return f"Response(content='{self.content[:50]}...', tokens={tokens}, latency={self.latency:.2f}s)"


class LLM:
    """Stateful LLM client for multiple calls

    Example:
        >>> llm = LLM(provider="openai", model="gpt-5.5", temperature=0.7)
        >>> response = llm.chat("What is 2+2?")
        >>> print(response.content)
        4
    """

    def __init__(
        self,
        provider: str = "google",
        model: Optional[str] = None,
        temperature: float = 0.7,
        search: bool = False,
        **kwargs,
    ):
        self.provider = provider
        self._is_cli = provider in CLI_PROVIDERS
        auth = kwargs.pop("auth", None)
        api_only = kwargs.pop("api_only", None)
        mode = kwargs.pop("mode", None)
        if auth is not None:
            # Caller already resolved billing (e.g. llm_dispatch auth=).
            api_only = None
            resolved_auth = auth  # type: ignore[assignment]
        else:
            resolved_auth, _, _, _ = resolve_auth(
                auth=auth,
                provider=provider,
                lite=kwargs.get("lite"),
                api_only=api_only,
            )
        kwargs.update(
            auth_to_llmx_kwargs(
                resolved_auth, lite=kwargs.get("lite"), mode=mode
            )
        )
        kwargs["auth"] = resolved_auth
        lite = kwargs.get("lite")
        self._cli_provider = preferred_cli_provider(provider, lite=lite)
        self.temperature = temperature
        self.search = search
        self.kwargs = kwargs

        if model is not None:
            if self._is_cli:
                self.model = model
            else:
                self.model = get_model_name(provider, model)
        else:
            if self._cli_provider:
                logical_provider = (
                    provider
                    if not self._is_cli
                    else CLI_PROVIDERS[self._cli_provider]["api_fallback"]
                )
                self.model = get_model_name(logical_provider)
            elif self._is_cli:
                self.model = None
            else:
                self.model = get_model_name(provider)

        if not self._cli_provider and not self._is_cli:
            check_api_key(provider)

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Response:
        """Send chat message and return Response."""
        # CLI backend — try CLI first, fall back to API
        if self._cli_provider and not kwargs.pop("api_only", False):
            # Per-call kwargs override stored self.kwargs (e.g. someone built
            # an LLM with default timeout but passes max_tokens=65536 on a
            # specific .chat() call).
            merged = {**self.kwargs, **kwargs}
            schema = merged.get("response_format")
            reasoning_effort = merged.get("reasoning_effort")
            max_tokens = merged.get("max_tokens")
            fallback_reason = needs_api_fallback(
                self._cli_provider,
                schema,
                system,
                self.search,
                False,
                reasoning_effort,
                max_tokens,
            )
            if not fallback_reason:
                start_time = time.time()
                text = cli_chat(
                    self._cli_provider,
                    prompt,
                    self.model,
                    merged.get("timeout", 300),
                    schema=schema,
                    system=system,
                    lite=merged.get("lite"),
                    reasoning_effort=reasoning_effort,
                )
                if text is not None:
                    latency = time.time() - start_time
                    return Response(
                        content=text,
                        provider=self._cli_provider,
                        model=self.model or self.provider,
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        latency=latency,
                        raw=None,
                    )
            # Fall back to API provider — disable CLI on fallback to prevent recursion
            api_provider = CLI_PROVIDERS[self._cli_provider]["api_fallback"]
            logger.info(
                f"[cli→api] {self._cli_provider} → {api_provider} ({fallback_reason or 'CLI error'})"
            )
            fallback = LLM(
                provider=api_provider,
                model=self.model,
                temperature=self.temperature,
                search=self.search,
                **self.kwargs,
            )
            fallback._cli_provider = None  # prevent infinite fallback loop
            return fallback.chat(
                prompt, system=system, temperature=temperature, **kwargs
            )

        # Native SDK call
        call_kwargs = {**self.kwargs, **kwargs}
        temp = temperature if temperature is not None else self.temperature

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        with capture_call(self.provider, self.model, messages) as trace:
            try:
                # Auto-retry TRANSIENT failures (429/503/overload/connection) with backoff+jitter,
                # so a load-shed (esp. --flex) doesn't surface as a swallowable hard failure.
                # Non-transient errors (QuotaError/ModelError/ApiKeyError) aren't RateLimitError →
                # propagate immediately. stdout is captured/restored PER attempt.
                max_attempts = _resolve_max_attempts(call_kwargs)
                usage_out: dict = {}
                content = None
                for attempt in range(max_attempts):
                    usage_out = {}  # discard a failed attempt's partial usage
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    try:
                        if self.provider == "google":
                            content = _google_chat(
                                prompt=prompt,
                                model=self.model,
                                system=system,
                                temperature=temp,
                                timeout=call_kwargs.get("timeout", 300),
                                stream=False,
                                max_tokens=call_kwargs.get("max_tokens"),
                                search=self.search,
                                schema=call_kwargs.get("response_format"),
                                reasoning_effort=call_kwargs.get("reasoning_effort"),
                                usage_out=usage_out,
                                service_tier=call_kwargs.get("service_tier"),
                            )
                        else:
                            if self.search:
                                _build_search_kwargs(self.provider, self.model)
                            content = _openai_chat(
                                prompt=prompt,
                                model=self.model,
                                provider=self.provider,
                                system=system,
                                temperature=temp,
                                timeout=call_kwargs.get("timeout", 300),
                                stream=False,
                                max_tokens=call_kwargs.get("max_tokens"),
                                schema=call_kwargs.get("response_format"),
                                reasoning_effort=call_kwargs.get("reasoning_effort"),
                                usage_out=usage_out,
                            )
                    except RateLimitError as e:
                        if attempt == max_attempts - 1:
                            raise  # retries exhausted — still exit 3, caller decides
                        wait = min(
                            _RETRY_BASE * (2**attempt) + random.uniform(0, _RETRY_BASE),
                            _RETRY_CAP,
                        )
                        print(
                            f"[llmx:RETRY] {self.provider}/{self.model} {e.error_type} "
                            f"(attempt {attempt + 1}/{max_attempts}) — sleeping {wait:.1f}s",
                            file=sys.stderr,
                        )
                        time.sleep(wait)
                        continue
                    finally:
                        sys.stdout = old_stdout
                    break  # success

                latency = time.time() - start_time
                usage = usage_out or {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "reasoning_tokens": None,
                    "cached_tokens": None,
                }

                trace.set_response(
                    {"content": content, "usage": usage, "latency": latency}
                )

                return Response(
                    content=content,
                    provider=self.provider,
                    model=self.model,
                    usage=usage,
                    latency=latency,
                    raw=None,
                )

            except Exception as e:
                trace.set_error(e)
                raise

    def stream(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> Iterator[str]:
        """Stream response chunks."""
        temp = self.temperature

        if self.provider == "google":
            from google import genai
            from google.genai import types

            timeout = kwargs.get("timeout", 300)
            client = genai.Client(
                http_options=types.HttpOptions(
                    timeout=max(timeout * 1000, 10_000) if timeout else 300_000
                )
            )
            config = types.GenerateContentConfig(temperature=temp)
            if system:
                config.system_instruction = system

            for chunk in client.models.generate_content_stream(
                model=self.model, contents=prompt, config=config
            ):
                if chunk.text:
                    yield chunk.text
        else:
            from openai import OpenAI

            base_url = OPENAI_COMPAT_URLS.get(self.provider)
            api_key = _get_api_key(self.provider)
            timeout = kwargs.get("timeout", 300)

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout) if timeout else 300.0,
            )

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            model = self.model
            if self.provider == "anthropic" and not model.startswith("anthropic/"):
                model = f"anthropic/{model}"

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                stream=True,
            )

            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content


# Simple function API


def chat(
    prompt: str,
    provider: str = "google",
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    search: bool = False,
    **kwargs,
) -> Response:
    """Simple chat function - one-shot calls"""
    auth = kwargs.pop("auth", None)
    api_only = kwargs.pop("api_only", None)
    mode = kwargs.pop("mode", None)
    resolved_auth, _, _, _ = resolve_auth(
        auth=auth,
        provider=provider,
        lite=kwargs.get("lite"),
        api_only=api_only,
    )
    kwargs.update(
        auth_to_llmx_kwargs(resolved_auth, lite=kwargs.get("lite"), mode=mode)
    )
    kwargs["auth"] = resolved_auth
    # Pull init-time fields out so LLM.__init__ doesn't see them as **kwargs
    # carried into self.kwargs (timeout/max_tokens/reasoning_effort/response_format
    # are per-call options, not constructor state).
    init_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "timeout",
            "max_tokens",
            "reasoning_effort",
            "response_format",
            "service_tier",
        }
    }
    call_kwargs = {k: v for k, v in kwargs.items() if k not in init_kwargs}
    llm = LLM(
        provider=provider,
        model=model,
        temperature=temperature,
        search=search,
        **init_kwargs,
    )
    return llm.chat(prompt, system=system, **call_kwargs)


def batch(
    prompts: List[str],
    provider: str = "google",
    model: Optional[str] = None,
    system: Optional[str] = None,
    parallel: int = 3,
    **kwargs,
) -> List[Response]:
    """Process multiple prompts in parallel"""
    from concurrent.futures import ThreadPoolExecutor

    llm = LLM(provider=provider, model=model, **kwargs)

    def _chat_with_system(prompt: str) -> Response:
        return llm.chat(prompt, system=system)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        return list(executor.map(_chat_with_system, prompts))


# ============================================================================
# Gemini Batch API (async, 50% discount)
# ============================================================================


def batch_submit(
    input_file: str,
    model: str = "gemini-3-flash-preview",
    display_name: Optional[str] = None,
) -> str:
    """Submit a Gemini batch job from a JSONL file. Returns job name."""
    from .gemini_batch import parse_input_jsonl, submit

    requests = parse_input_jsonl(input_file)
    return submit(requests, model=model, display_name=display_name)


def batch_status(job_name: str) -> Dict[str, Any]:
    """Get batch job status."""
    from .gemini_batch import status

    return status(job_name)


def batch_get(job_name: str, keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Fetch results from a completed batch job."""
    from .gemini_batch import fetch

    results = fetch(job_name, original_keys=keys)
    return [
        {
            "key": r.key,
            **({"content": r.content} if r.content else {}),
            **({"error": r.error} if r.error else {}),
        }
        for r in results
    ]


# Export public API
__all__ = [
    "LLM",
    "Response",
    "chat",
    "batch",
    "batch_submit",
    "batch_status",
    "batch_get",
]
