"""Programmatic API for llmx"""

import time
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass

from .providers import get_model_name, check_api_key, _build_search_kwargs, infer_provider_from_model

# Import litellm directly for API calls
try:
    from litellm import completion
except ImportError:
    raise ImportError(
        "litellm not installed. Install llmx with: uv tool install llmx"
    )

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
        tokens = self.usage.get('total_tokens', 0)
        return f"Response(content='{self.content[:50]}...', tokens={tokens}, latency={self.latency:.2f}s)"


class LLM:
    """Stateful LLM client for multiple calls

    Example:
        >>> llm = LLM(provider="openai", model="gpt-4o", temperature=0.7)
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
        **kwargs
    ):
        """Initialize LLM client

        Args:
            provider: Provider name (google, openai, anthropic, xai, deepseek)
            model: Model name (overrides provider default)
            temperature: Temperature 0-1 (default: 0.7)
            search: Enable web search grounding (google, anthropic, xai)
            **kwargs: Additional provider-specific arguments

        Raises:
            ValueError: If provider is unknown
            RuntimeError: If API key not found
        """
        self.provider = provider
        self.model = model or get_model_name(provider)
        self.temperature = temperature
        self.search = search
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
        """Send chat message

        Args:
            prompt: User prompt
            system: System message (optional)
            temperature: Override default temperature
            **kwargs: Override any init kwargs

        Returns:
            Response object with content and metadata

        Example:
            >>> llm = LLM(provider="openai")
            >>> response = llm.chat("Explain Python", system="You are a teacher")
            >>> print(response.content)
        """
        # Merge kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        temp = temperature if temperature is not None else self.temperature

        # Add web search grounding
        if self.search:
            search_kwargs = _build_search_kwargs(self.provider, self.model)
            call_kwargs.update(search_kwargs)

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Call LiteLLM with inspection
        start_time = time.time()

        with capture_call(self.provider, self.model, messages) as trace:
            try:
                # Call LiteLLM completion
                response = completion(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    stream=False,
                    **call_kwargs
                )

                # Extract content and metadata
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                }
                latency = time.time() - start_time

                # Update trace
                trace.set_response({
                    "content": content,
                    "usage": usage,
                    "latency": latency
                })

                return Response(
                    content=content,
                    provider=self.provider,
                    model=self.model,
                    usage=usage,
                    latency=latency,
                    raw=response
                )

            except Exception as e:
                trace.set_error(e)
                raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream response chunks

        Args:
            prompt: User prompt
            **kwargs: Override any init kwargs

        Yields:
            Response chunks as they arrive

        Example:
            >>> llm = LLM(provider="openai")
            >>> for chunk in llm.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Call LiteLLM with streaming
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            **kwargs
        )

        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content"):
                content = chunk.choices[0].delta.content
                if content:
                    yield content


# Simple function API

def chat(
    prompt: str,
    provider: str = "google",
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    search: bool = False,
    **kwargs
) -> Response:
    """Simple chat function - one-shot calls

    Args:
        prompt: User prompt
        provider: Provider name (default: google)
        model: Model name (overrides provider default)
        system: System message (optional)
        temperature: Temperature 0-1 (default: 0.7)
        search: Enable web search grounding (google, anthropic, xai)
        **kwargs: Additional provider arguments

    Returns:
        Response object

    Example:
        >>> from llmx import chat
        >>> response = chat("What is 2+2?", provider="openai")
        >>> print(response.content)
        4
        >>> print(response.usage)
        {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
    """
    llm = LLM(provider=provider, model=model, temperature=temperature, search=search, **kwargs)
    return llm.chat(prompt, system=system)


def batch(
    prompts: List[str],
    provider: str = "google",
    model: Optional[str] = None,
    parallel: int = 3,
    **kwargs
) -> List[Response]:
    """Process multiple prompts in parallel

    Args:
        prompts: List of prompts to process
        provider: Provider name (default: google)
        model: Model name (overrides provider default)
        parallel: Number of parallel requests (default: 3)
        **kwargs: Additional provider arguments

    Returns:
        List of Response objects (same order as input prompts)

    Example:
        >>> from llmx import batch
        >>> prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
        >>> responses = batch(prompts, provider="google", parallel=2)
        >>> for r in responses:
        ...     print(r.content)
        4
        6
        8
    """
    from concurrent.futures import ThreadPoolExecutor

    llm = LLM(provider=provider, model=model, **kwargs)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        return list(executor.map(llm.chat, prompts))


# Export public API
__all__ = ["LLM", "Response", "chat", "batch"]
