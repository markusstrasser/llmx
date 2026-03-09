#!/usr/bin/env python3
"""Test direct SDK calls without LiteLLM.

Validates that google-genai and openai SDKs can replace LiteLLM for all
features used by llmx: chat, streaming, system messages, max_output_tokens,
structured output, search grounding, thinking/reasoning config, timeout.

Also tests openai SDK with base_url for OpenAI-compatible providers (xAI, etc.).
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Google GenAI SDK — direct
# ---------------------------------------------------------------------------

def test_google_basic():
    """Basic generate_content — replaces litellm.completion() for Google."""
    from google import genai
    from google.genai import types

    client = genai.Client()  # uses GEMINI_API_KEY or GOOGLE_API_KEY
    start = time.time()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="What is 2+2? Reply with just the number.",
    )
    elapsed = time.time() - start
    print(f"[google/basic] {elapsed:.1f}s — {response.text.strip()}")
    assert "4" in response.text, f"Unexpected: {response.text}"
    return True


def test_google_system_instruction():
    """System instruction — the feature that forces LiteLLM fallback."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="What is your name?",
        config=types.GenerateContentConfig(
            system_instruction="You are a cat named Neko. Always meow.",
        ),
    )
    print(f"[google/system] {response.text.strip()[:100]}")
    return True


def test_google_max_output_tokens():
    """max_output_tokens — the feature that forces LiteLLM fallback."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Write a haiku about programming.",
        config=types.GenerateContentConfig(
            max_output_tokens=100,
        ),
    )
    print(f"[google/max_tokens] {response.text.strip()[:100]}")
    return True


def test_google_streaming():
    """Streaming — replaces litellm streaming."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    start = time.time()
    chunks = []
    for chunk in client.models.generate_content_stream(
        model="gemini-3-flash-preview",
        contents="Count from 1 to 5, one number per line.",
    ):
        if chunk.text:
            chunks.append(chunk.text)
    elapsed = time.time() - start
    full = "".join(chunks)
    print(f"[google/stream] {elapsed:.1f}s — {len(chunks)} chunks — {full.strip()[:80]}")
    return True


def test_google_thinking():
    """Thinking/reasoning config — replaces litellm reasoning_effort mapping."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    start = time.time()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="What is 17 * 23?",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    elapsed = time.time() - start
    print(f"[google/thinking] {elapsed:.1f}s — {response.text.strip()[:80]}")
    return True


def test_google_structured_output():
    """Structured output via response_schema."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Name 2 planets.",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "planets": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["planets"],
            },
        ),
    )
    parsed = json.loads(response.text)
    print(f"[google/schema] {parsed}")
    assert "planets" in parsed
    return True


def test_google_search():
    """Google Search grounding — replaces litellm tools injection."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="What was the latest SpaceX launch?",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    print(f"[google/search] {response.text.strip()[:120]}")
    return True


def test_google_timeout():
    """Timeout config via http_options on Client."""
    from google import genai
    from google.genai import types

    # Client-level timeout
    client = genai.Client(http_options=types.HttpOptions(timeout=5000))  # 5s in ms
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Say hi.",
    )
    print(f"[google/timeout] {response.text.strip()[:80]}")
    return True


# ---------------------------------------------------------------------------
# OpenAI SDK — direct (and OpenAI-compatible providers via base_url)
# ---------------------------------------------------------------------------

def test_openai_basic():
    """Basic chat completion."""
    from openai import OpenAI

    client = OpenAI(timeout=30.0)
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap model for testing
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    print(f"[openai/basic] {elapsed:.1f}s — {text.strip()}")
    return True


def test_openai_streaming():
    """Streaming chat completion."""
    from openai import OpenAI

    client = OpenAI(timeout=30.0)
    start = time.time()
    chunks = []
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count 1 to 5."}],
        max_tokens=50,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    elapsed = time.time() - start
    print(f"[openai/stream] {elapsed:.1f}s — {len(chunks)} chunks — {''.join(chunks).strip()[:80]}")
    return True


def test_openai_system():
    """System message."""
    from openai import OpenAI

    client = OpenAI(timeout=30.0)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
            {"role": "user", "content": "Hello."},
        ],
        max_tokens=50,
    )
    text = response.choices[0].message.content
    print(f"[openai/system] {text.strip()[:100]}")
    return True


def test_openai_structured():
    """Structured output via response_format."""
    from openai import OpenAI

    client = OpenAI(timeout=30.0)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Name 2 colors."}],
        max_tokens=100,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "colors",
                "schema": {
                    "type": "object",
                    "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
                    "required": ["colors"],
                },
            },
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    print(f"[openai/schema] {parsed}")
    assert "colors" in parsed
    return True


def test_xai_via_openai():
    """xAI/Grok via openai SDK with base_url."""
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        print("[xai/basic] SKIP — no XAI_API_KEY")
        return True

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=30.0,
    )
    start = time.time()
    response = client.chat.completions.create(
        model="grok-3-mini-fast",  # cheapest for testing
        messages=[{"role": "user", "content": "What is 2+2? Just the number."}],
        max_tokens=10,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    print(f"[xai/basic] {elapsed:.1f}s — {text.strip()}")
    return True


def test_deepseek_via_openai():
    """DeepSeek via openai SDK with base_url."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[deepseek/basic] SKIP — no DEEPSEEK_API_KEY")
        return True

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30.0,
    )
    start = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "What is 2+2? Just the number."}],
        max_tokens=10,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    print(f"[deepseek/basic] {elapsed:.1f}s — {text.strip()}")
    return True


def test_openrouter_via_openai():
    """OpenRouter via openai SDK with base_url."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[openrouter/basic] SKIP — no OPENROUTER_API_KEY")
        return True

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
    )
    start = time.time()
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Just the number."}],
        max_tokens=10,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    print(f"[openrouter/basic] {elapsed:.1f}s — {text.strip()}")
    return True


def test_cerebras_via_openai():
    """Cerebras via openai SDK with base_url."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("[cerebras/basic] SKIP — no CEREBRAS_API_KEY")
        return True

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
        timeout=30.0,
    )
    start = time.time()
    response = client.chat.completions.create(
        model="llama-4-scout-17b-16e-instruct",  # cheapest available
        messages=[{"role": "user", "content": "What is 2+2? Just the number."}],
        max_tokens=10,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    print(f"[cerebras/basic] {elapsed:.1f}s — {text.strip()}")
    return True


def test_kimi_via_openai():
    """Kimi/Moonshot via openai SDK with base_url."""
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if not api_key:
        print("[kimi/basic] SKIP — no MOONSHOT_API_KEY")
        return True

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
        timeout=30.0,
    )
    start = time.time()
    response = client.chat.completions.create(
        model="moonshot-v1-8k",  # cheapest for testing
        messages=[{"role": "user", "content": "What is 2+2? Just the number."}],
        max_tokens=10,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    print(f"[kimi/basic] {elapsed:.1f}s — {text.strip()}")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        # Google native
        ("google/basic", test_google_basic),
        ("google/system", test_google_system_instruction),
        ("google/max_tokens", test_google_max_output_tokens),
        ("google/stream", test_google_streaming),
        ("google/thinking", test_google_thinking),
        ("google/schema", test_google_structured_output),
        ("google/search", test_google_search),
        ("google/timeout", test_google_timeout),
        # OpenAI native
        ("openai/basic", test_openai_basic),
        ("openai/stream", test_openai_streaming),
        ("openai/system", test_openai_system),
        ("openai/schema", test_openai_structured),
        # OpenAI-compat providers
        ("xai", test_xai_via_openai),
        ("deepseek", test_deepseek_via_openai),
        ("openrouter", test_openrouter_via_openai),
        ("cerebras", test_cerebras_via_openai),
        ("kimi", test_kimi_via_openai),
    ]

    passed, failed, skipped = 0, 0, 0
    for name, fn in tests:
        try:
            result = fn()
            if result:
                passed += 1
        except Exception as e:
            print(f"[{name}] FAIL — {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
