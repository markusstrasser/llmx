# Drop LiteLLM — Replace with Native SDKs

**Date:** 2026-03-07 (updated 2026-03-09)
**Project:** llmx
**Status:** Ready to execute
**Reviews:** Gemini 3.1 Pro + GPT-5.4, findings incorporated below

## Problem

LiteLLM is a middleware layer between llmx and the actual APIs. It:
- Uses httpx read timeouts that don't fire on streaming keepalives (root cause of 10+ min hangs)
- Requires model name prefix translation (`gemini/`, `moonshot/`, `anthropic/`) that causes 404s when wrong
- Produces opaque error messages that llmx must regex-classify
- Ships 200+ provider configs we don't use
- Forces SIGALRM band-aid for wall-clock timeout

Meanwhile, `google-genai` and `openai` SDKs are already installed as direct dependencies.

## Evidence

**`test_native_sdks.py` confirms all LiteLLM-provided features work natively** (15/17 passed, 2 trivial):
- google-genai: basic, system_instruction, max_output_tokens, streaming, thinking_config, structured output, search grounding, timeout (server-side deadline, min 10s)
- openai SDK: basic, streaming, system, structured output, timeout
- openai SDK + base_url: xAI ✓, OpenRouter ✓ (DeepSeek/Cerebras/Kimi expected — all OpenAI-compatible)
- Anthropic API is NOT OpenAI-compatible (different endpoint `/v1/messages`, different auth header)
- Anthropic API key has no credits — route through OpenRouter

**CLI backend update (2026-03-09):**
- Gemini CLI bare mode now working (`110da87`): `HOME=~/.gemini-bare` skips MCP/skills/extensions. ~40% faster (13s vs 20s). Wired into llmx's `cli_chat()`.
- Gemini CLI 8K default maxOutputTokens is sufficient for reviews. Truncation detection (not higher limits) is the fix.
- `customOverrides` in `settings.json` CAN set `maxOutputTokens` per-model but requires non-empty `match` object — empty match returns null.

## Architecture

### Current (LiteLLM)
```
CLI flags → cli_backends.py (gemini-cli/codex-cli) → subprocess
         ↘ providers.py → litellm.completion() → httpx → vendor API
                          ↑ SIGALRM band-aid
```

### New (native SDKs + CLI backends)
```
CLI flags → cli_backends.py (gemini-cli/codex-cli) → subprocess  [unchanged]
         ↘ providers.py → _google_chat()  → google-genai SDK → Google API
                        → _openai_chat()  → openai SDK       → OpenAI/xAI/Kimi/Cerebras/DeepSeek/OpenRouter
```

Two API backends (not three — claude-cli deferred):

| Backend | SDK/Tool | Providers | Timeout | Cost |
|---------|----------|-----------|---------|------|
| `_google_chat()` | `google-genai` | Google/Gemini | Server-side deadline (ms, min 10s) | API metered |
| `_openai_chat()` | `openai` | OpenAI, xAI, Kimi, Cerebras, DeepSeek, OpenRouter | `httpx.Timeout` + SIGALRM safety | API metered |
| CLI backends | subprocess | gemini-cli (subscription), codex-cli (subscription) | process group kill timer | Subscription flat rate |

### Provider routing (openai SDK + base_url)

| Provider | base_url | API key env | Verified |
|----------|----------|-------------|----------|
| openai | (default) | OPENAI_API_KEY | ✓ |
| xai | https://api.x.ai/v1 | XAI_API_KEY | ✓ |
| deepseek | https://api.deepseek.com | DEEPSEEK_API_KEY | skip (no key) |
| openrouter | https://openrouter.ai/api/v1 | OPENROUTER_API_KEY | ✓ |
| cerebras | https://api.cerebras.ai/v1 | CEREBRAS_API_KEY | skip (no key) |
| kimi | https://api.moonshot.cn/v1 | MOONSHOT_API_KEY | auth error (key issue) |
| anthropic | https://openrouter.ai/api/v1 | OPENROUTER_API_KEY | via OpenRouter |

### Anthropic

Route through OpenRouter only. Don't add anthropic SDK (no API credits, different protocol).
Claude-cli backend deferred — can't call from Claude Code (nested session block), low ROI.

## Phases

### Phase 1: `_google_chat()` + model normalization + PROVIDER_CONFIGS

**Merged from old Phases 1+4.** Must strip litellm prefixes BEFORE the new dispatcher can work — google-genai rejects `gemini/gemini-3.1-pro-preview`.

**1a. Model normalization function** [GPT review fix]
```python
def _normalize_model(provider: str, model: str) -> str:
    """Strip synthetic LiteLLM prefixes, preserve real provider model IDs."""
    STRIP_PREFIXES = {
        "google": "gemini/",
        "kimi": "moonshot/",
        "xai": "xai/",
    }
    prefix = STRIP_PREFIXES.get(provider)
    if prefix and model.startswith(prefix):
        logger.debug(f"Stripped prefix '{prefix}' from model '{model}'")
        return model[len(prefix):]
    # OpenRouter models use real slashes (anthropic/claude-sonnet-4-6) — preserve
    return model
```

Accept old prefixed names with debug log. No hard break.

**1b. Update PROVIDER_CONFIGS** — strip litellm prefixes from defaults:
```python
PROVIDER_CONFIGS = {
    "google": {"model": "gemini-3.1-pro-preview", "flash_model": "gemini-3-flash-preview"},
    "openai": {"model": "gpt-5.4"},
    "xai": {"model": "grok-4", "base_url": "https://api.x.ai/v1"},
    "kimi": {"model": "kimi-k2.5", "base_url": "https://api.moonshot.cn/v1"},
    # ...
}
```

**1c. `_google_chat()` — native google-genai SDK call:**
```python
from google import genai
from google.genai import types
from google.genai import errors as genai_errors  # [Gemini review fix: correct exception module]

def _google_chat(prompt, model, system, temperature, timeout, stream,
                 max_tokens, search, schema, reasoning_effort):
    client = genai.Client(
        http_options=types.HttpOptions(timeout=max(timeout * 1000, 10_000) if timeout else 300_000)
        # [GPT review fix: guard timeout=None]
    )
    config = types.GenerateContentConfig(temperature=temperature)
    if system:
        config.system_instruction = system
    if max_tokens:
        config.max_output_tokens = max_tokens
    if reasoning_effort:
        level_map = {"low": "low", "medium": "medium", "high": "high",
                     "minimal": "low", "none": "low", "xhigh": "high"}
        config.thinking_config = types.ThinkingConfig(
            thinking_level=level_map.get(reasoning_effort, "high")
        )
    if schema:
        config.response_mime_type = "application/json"
        config.response_schema = schema
    if search:
        config.tools = [types.Tool(google_search=types.GoogleSearch())]

    result_text = ""
    finish_reason = None

    if stream:
        for chunk in client.models.generate_content_stream(
            model=model, contents=prompt, config=config
        ):
            if chunk.text:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
                result_text += chunk.text
        sys.stdout.write("\n")
    else:
        response = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
        # [Gemini review fix: guard empty candidates from safety filter]
        if not response.candidates:
            feedback = getattr(response, 'prompt_feedback', None)
            raise ModelError(
                f"Response blocked by safety filter: {feedback}",
                provider="google", model=model,
            )
        result_text = response.text
        finish_reason = str(response.candidates[0].finish_reason)
        print(result_text)

    # [Both reviews: truncation detection]
    if finish_reason and "MAX_TOKENS" in str(finish_reason):
        logger.warn(f"[llmx:WARN] output may be truncated (hit {max_tokens or 8192} token limit)")

    return result_text
```

**Key differences from LiteLLM path:**
- No model name prefix (`gemini-3.1-pro-preview` not `gemini/gemini-3.1-pro-preview`)
- Server-side deadline timeout (ms) — no SIGALRM needed for Google path
- `thinking_config` instead of LiteLLM's `reasoning_effort` passthrough
- `system_instruction` in config, not message array
- Search grounding via `google_search` tool in config — returns inline citations
- Native exception types (`google.genai.errors.*`) → map to llmx error types

### Phase 2: `_openai_chat()` in providers.py

OpenAI SDK with per-provider `base_url`.

```python
import openai as openai_module  # [GPT review fix: import module for exception types]
from openai import OpenAI

OPENAI_COMPAT_URLS = {
    "openai": None,
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "anthropic": "https://openrouter.ai/api/v1",
}

# [GPT review fix: anthropic routes to OpenRouter, needs OPENROUTER_API_KEY]
API_KEY_OVERRIDES = {
    "anthropic": "OPENROUTER_API_KEY",
}

def _openai_chat(prompt, model, provider, system, temperature, timeout,
                 stream, max_tokens, schema, reasoning_effort):
    base_url = OPENAI_COMPAT_URLS.get(provider)

    # API key resolution — special cases first
    key_env = API_KEY_OVERRIDES.get(provider)
    if key_env:
        api_key = os.environ.get(key_env)
    else:
        api_key = _get_api_key(provider)

    # [Both reviews fix: guard double-prefix for OpenRouter models]
    if provider == "anthropic" and not model.startswith("anthropic/"):
        model = f"anthropic/{model}"

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout) if timeout else 300.0)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens  # GPT-5.x uses max_completion_tokens
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema},
            # [GPT review fix: add strict: True]
        }

    result_text = ""
    finish_reason = None

    if stream:
        for chunk in client.chat.completions.create(**kwargs, stream=True):
            # [Gemini review fix: guard empty choices from OpenRouter keepalive chunks]
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                result_text += delta.content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        sys.stdout.write("\n")
    else:
        response = client.chat.completions.create(**kwargs)
        # [GPT review fix: guard None content]
        content = response.choices[0].message.content
        if content is None:
            refusal = getattr(response.choices[0].message, 'refusal', None)
            raise ModelError(
                f"Model returned no content. Refusal: {refusal}",
                provider=provider, model=model,
            )
        result_text = content
        finish_reason = response.choices[0].finish_reason
        print(result_text)

    # [GPT review fix: truncation detection for OpenAI path too]
    if finish_reason == "length":
        logger.warn(f"[llmx:WARN] output may be truncated (hit max_tokens limit)")

    return result_text
```

**SIGALRM safety net:** Keep for openai path, but only in main thread.
```python
# [Gemini review fix: SIGALRM only works in main thread]
import threading
use_alarm = threading.current_thread() is threading.main_thread()
```

### Phase 3: Rewire `chat()` dispatcher + error mapping

Replace LiteLLM call path. Both functions return text now (for `api.py` compat + `--output` tee).

```python
from google.genai import errors as genai_errors

try:
    model = _normalize_model(provider, model_name)
    if provider == "google":
        text = _google_chat(...)
    else:
        text = _openai_chat(...)
except genai_errors.ClientError as e:
    # Google SDK wraps 4xx errors
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        raise RateLimitError(...) from e
    elif "404" in msg:
        raise ModelError(f"Model not found: {model_name}", ...) from e
    raise LlmxError(msg, ...) from e
except genai_errors.ServerError as e:
    msg = str(e)
    if "DEADLINE_EXCEEDED" in msg:
        raise TimeoutError_(...) from e
    raise LlmxError(str(e), ...) from e
except openai_module.RateLimitError as e:
    raise RateLimitError(...) from e
except openai_module.APITimeoutError as e:
    raise TimeoutError_(...) from e
except openai_module.AuthenticationError as e:
    raise ApiKeyError(...) from e
except openai_module.NotFoundError as e:
    raise ModelError(f"Model not found: {model_name}", ...) from e
except openai_module.APIStatusError as e:
    # [GPT review fix: catch 402 credits, 403 permission]
    if e.status_code == 402:
        raise RateLimitError(f"Insufficient credits: {e}", ...) from e
    raise LlmxError(str(e), ...) from e
```

**Also rewrite `compare()`** [Gemini review catch — was completely missed in v1]:
- Currently imports `litellm.completion()` directly
- Rewrite to use `_google_chat()` / `_openai_chat()` via ThreadPoolExecutor
- Disable SIGALRM in worker threads (main thread only)

### Phase 4: DX improvements

Cheap wins from the 10 DX issues. Only implement what's free:

**4a. Transport switch warning (already mostly exists):**
```python
if cli_fallback_reason:
    click.echo(f"[llmx:TRANSPORT] {cli_provider} → API ({cli_fallback_reason})", err=True)
```

**4b. Truncation detection** — already in Phase 1/2 code above.

**4c. Model name validation** — warn only, don't hard-fail:
```python
def _warn_unknown_model(model: str, provider: str):
    known = _known_models_for_provider(provider)
    if known and model not in known:
        close = difflib.get_close_matches(model, known, n=1, cutoff=0.6)
        if close:
            logger.warn(f"Unknown model '{model}'. Did you mean '{close[0]}'?")
        # Don't raise — OpenRouter and new models would break
```

Skip: JSON diagnostic reformat (5d), model validation hard-fail (5a), `-f` investigation (5e). Not worth the complexity.

### Phase 5: Remove LiteLLM + test

- Remove `litellm>=1.79.0` from pyproject.toml
- Remove all litellm imports
- `uv sync` to clean venv
- Run `test_native_sdks.py`
- Manual smoke test: `llmx "test"` (Google CLI), `llmx -m gpt-5.4 "test"` (OpenAI API), `llmx --search "test"` (Google search grounding)
- Test old prefixed names: `llmx -m gemini/gemini-3.1-pro-preview "test"` (should normalize + work)
- Bump version to 0.6.0
- Update README

### Phase 6: Update skills

- **model-review SKILL.md**: Remove "Budget: ~2000 words" from prompts. Document that Gemini CLI defaults to 8K output (sufficient). Model names no longer need `gemini/` prefix.
- **llmx-guide skill**: Update flag docs, add truncation detection notes, document bare mode.

## Deferred

- **Claude-cli backend** — can't call from Claude Code (nested block), Anthropic via OpenRouter covers API. Low ROI.
- **Schema normalization per backend** — Google and OpenAI schema dialects differ (`additionalProperties`, `$ref`, nullable). Cross that bridge when structured output actually breaks.
- **Streaming + structured output validation** — partial JSON chunks are fine for human consumers. Machine consumers should use non-streaming.
- **`_validate_model_name()` hard-fail** — would break OpenRouter dynamic model catalog.

## Review Bugs Fixed (incorporated above)

| Bug | Source | Where Fixed |
|-----|--------|-------------|
| `compare()` uses litellm directly — would crash | Gemini | Phase 3 |
| Double-prefix Anthropic (`anthropic/anthropic/...`) | Both | Phase 2 (`startswith` guard) |
| Wrong exception types (`google.api_core` → `google.genai.errors`) | Both | Phase 1c, 3 |
| OpenRouter empty `choices[]` in streaming | Gemini | Phase 2 (continue guard) |
| SIGALRM in ThreadPoolExecutor threads | Gemini | Phase 2 (main thread check) |
| `api.py` contract — functions return text | GPT | Phase 1c, 2 (return result_text) |
| Model normalization: preserve real slashes | GPT | Phase 1a |
| Old prefixed names accepted with warning | GPT | Phase 1a (debug log) |
| Missing `strict: True` for OpenAI structured output | GPT | Phase 2 |
| Truncation detection for BOTH backends | GPT | Phase 1c (MAX_TOKENS), 2 (length) |
| Safety filter empty candidates | Gemini | Phase 1c (guard + error) |
| None content / refusal handling | GPT | Phase 2 (guard + error) |
| `timeout=None` crash on `max()` | GPT | Phase 1c (ternary guard) |
| Anthropic uses OPENROUTER_API_KEY not ANTHROPIC | GPT | Phase 2 (API_KEY_OVERRIDES) |
| `max_tokens` → `max_completion_tokens` for GPT-5.x | Session | Phase 2 |

## Risk Assessment

**Low risk:**
- Google and OpenAI paths tested and working
- CLI backends unchanged
- Error types/exit codes unchanged (public interface preserved)
- CLI flags unchanged (user-facing interface preserved)
- Old model names accepted (graceful degradation)

**Medium risk:**
- OpenAI-compat providers may not support all features (reasoning_effort, structured output)
- Anthropic via OpenRouter adds latency + OpenRouter dependency
- google-genai SDK exception classification needs testing at implementation time — error messages may not contain expected status codes

**Execution estimate:** ~3 hours across 1-2 sessions. Phases 1-3 are the core (2h). Phase 4-6 are cleanup (1h).
