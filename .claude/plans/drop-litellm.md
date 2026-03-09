# Drop LiteLLM — Replace with Native SDKs

**Date:** 2026-03-07
**Project:** llmx
**Status:** Ready to execute

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
- Anthropic API key has no credits — route through OpenRouter or `claude -p` from non-nested contexts

**CLI backend findings:**
- Gemini CLI v0.32.1 (latest stable): no `--max-tokens` flag, no system prompt, no streaming, no structured output. Preview v0.33/0.34 nightly available but no new output control flags.
- Codex CLI v0.111.0: has `-o FILE` (output), `--output-schema FILE` (structured), `--json` (JSONL events). Model-restricted to ChatGPT subscription models (o4-mini rejected; gpt-5.4 works).
- Claude Code v2.1.71: rich `-p` mode (`--system-prompt`, `--json-schema`, `--output-format`, `--fallback-model`, `--max-budget-usd`, `--tools ""` for raw LLM). **Cannot be called from within Claude Code** (nested session block). Usable from orchestrator/scripts/cron only.

**Session failure analysis (10 distinct DX issues from today's transcripts):**
1. `-f` flag hangs with Gemini CLI — no timeout, no stderr, agent polls 47 cycles
2. Hallucinated CLI flags — no "did you mean?" suggestions, no model name validation
3. Empty output with no diagnostic signal — agents retry blindly
4. Shell redirect buffering — fixed with `--output` but agents still use `>`
5. Socket timeout ≠ wall-clock — fixed with SIGALRM but Python API still vulnerable
6. Model downgrade as "fix" — no capability-loss warning
7. Reasoning effort compatibility — silently ignored on unsupported models
8. Gemini 8K silent truncation — no truncation indicator in output
9. `-s` system prompt silently switches transport — no notification
10. Silent claim loss in structured extraction — valid JSON with missing entries

## Architecture

### Current (LiteLLM)
```
CLI flags → cli_backends.py (gemini-cli/codex-cli) → subprocess
         ↘ providers.py → litellm.completion() → httpx → vendor API
                          ↑ SIGALRM band-aid
```

### New (native SDKs + CLI backends)
```
CLI flags → cli_backends.py (gemini-cli/codex-cli/claude-cli) → subprocess  [extended]
         ↘ providers.py → _google_chat()  → google-genai SDK → Google API
                        → _openai_chat()  → openai SDK       → OpenAI/xAI/Kimi/Cerebras/DeepSeek/OpenRouter
```

Three backends:

| Backend | SDK/Tool | Providers | Timeout | Cost |
|---------|----------|-----------|---------|------|
| `_google_chat()` | `google-genai` | Google/Gemini | Server-side deadline (ms, min 10s) | API metered |
| `_openai_chat()` | `openai` | OpenAI, xAI, Kimi, Cerebras, DeepSeek, OpenRouter | `httpx.Timeout` + SIGALRM safety | API metered |
| CLI backends | subprocess | gemini-cli (subscription), codex-cli (subscription), claude-cli (subscription) | `subprocess.run(timeout=)` | Subscription flat rate |

### Provider routing (openai SDK + base_url)

| Provider | base_url | API key env | Verified |
|----------|----------|-------------|----------|
| openai | (default) | OPENAI_API_KEY | ✓ |
| xai | https://api.x.ai/v1 | XAI_API_KEY | ✓ |
| deepseek | https://api.deepseek.com | DEEPSEEK_API_KEY | skip (no key) |
| openrouter | https://openrouter.ai/api/v1 | OPENROUTER_API_KEY | ✓ |
| cerebras | https://api.cerebras.ai/v1 | CEREBRAS_API_KEY | skip (no key) |
| kimi | https://api.moonshot.cn/v1 | MOONSHOT_API_KEY | auth error (key issue) |

### Anthropic

Anthropic API is NOT OpenAI-compatible (different auth, different endpoint). Options:
1. ~~Add `anthropic` SDK~~ — third SDK, API key has no credits
2. Route through OpenRouter (`openrouter/anthropic/claude-sonnet-4-6`)
3. `claude -p` CLI backend (subscription, but blocked in nested sessions)

**Decision:** Option 2 (OpenRouter) as API fallback + Option 3 (claude-cli) as CLI backend for non-nested contexts. Don't add anthropic SDK.

### Claude CLI backend (new)

Add `claude-cli` to `cli_backends.py`:
```python
"claude-cli": {
    "binary": "claude",
    "api_fallback": "anthropic",  # routes to OpenRouter
}
```

`needs_api_fallback()` for claude-cli:
- ✓ system prompt (`--system-prompt`)
- ✓ structured output (`--json-schema`)
- ✓ streaming (`--output-format stream-json`)
- ✗ max_tokens (not exposed — fallback to API)
- ✗ nested session (CLAUDECODE env var set — fallback to API)

Detection: `if os.environ.get("CLAUDECODE"): return "nested Claude Code session"`

## Phases

### Phase 1: `_google_chat()` in providers.py

Native google-genai SDK call replacing litellm for Google provider.

```python
from google import genai
from google.genai import types

def _google_chat(prompt, model, system, temperature, timeout, stream,
                 max_tokens, search, schema, reasoning_effort):
    client = genai.Client(
        http_options=types.HttpOptions(timeout=max(timeout * 1000, 10_000))
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

    if stream:
        for chunk in client.models.generate_content_stream(
            model=model, contents=prompt, config=config
        ):
            if chunk.text:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        response = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
        print(response.text)
```

**Key differences from LiteLLM path:**
- No model name prefix (`gemini-3.1-pro-preview` not `gemini/gemini-3.1-pro-preview`)
- Server-side deadline timeout (ms) — no SIGALRM needed
- `thinking_config` instead of LiteLLM's `reasoning_effort` passthrough
- `system_instruction` in config, not message array
- Structured output via `response_mime_type` + `response_schema`
- Search grounding via `google_search` tool in config
- Native exception types (google.genai.errors.*) → map to llmx error types

### Phase 2: `_openai_chat()` in providers.py

OpenAI SDK with per-provider `base_url`.

```python
from openai import OpenAI

OPENAI_COMPAT_URLS = {
    "openai": None,  # default
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "anthropic": "https://openrouter.ai/api/v1",  # via OpenRouter
}

def _openai_chat(prompt, model, provider, system, temperature, timeout,
                 stream, max_tokens, schema, reasoning_effort):
    base_url = OPENAI_COMPAT_URLS.get(provider)
    api_key = _get_api_key(provider)

    # Anthropic via OpenRouter needs model prefix
    if provider == "anthropic":
        model = f"anthropic/{model}"

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }

    if stream:
        for chunk in client.chat.completions.create(**kwargs, stream=True):
            delta = chunk.choices[0].delta.content
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        response = client.chat.completions.create(**kwargs)
        print(response.choices[0].message.content)
```

**SIGALRM safety net:** Keep for openai path. The openai SDK still uses httpx internally, same streaming keepalive risk. Cheap insurance.

### Phase 3: Rewire `chat()` dispatcher + error mapping

Replace LiteLLM call with:
```python
try:
    if provider == "google":
        _google_chat(...)
    else:
        _openai_chat(...)
except google.api_core.exceptions.ResourceExhausted as e:
    raise RateLimitError(...) from e
except google.api_core.exceptions.DeadlineExceeded as e:
    raise TimeoutError_(...) from e
except openai.RateLimitError as e:
    raise RateLimitError(...) from e
except openai.APITimeoutError as e:
    raise TimeoutError_(...) from e
# etc.
```

**Native SDK exceptions → typed llmx errors.** No more regex on error messages.

### Phase 4: Update PROVIDER_CONFIGS + model names

Strip LiteLLM prefixes:
```python
PROVIDER_CONFIGS = {
    "google": {
        "model": "gemini-3.1-pro-preview",        # was "gemini/gemini-3.1-pro-preview"
        "flash_model": "gemini-3-flash-preview",    # was "gemini/gemini-3-flash-preview"
        "base_url": None,                           # uses google-genai SDK
    },
    "openai": {
        "model": "gpt-5.4",                        # unchanged
        "base_url": None,                           # openai SDK default
    },
    "xai": {
        "model": "grok-4",                         # was "xai/grok-4"
        "base_url": "https://api.x.ai/v1",
    },
    "kimi": {
        "model": "kimi-k2.5",                      # was "moonshot/kimi-k2.5"
        "base_url": "https://api.moonshot.cn/v1",
    },
    # ...
}
```

Simplify `get_model_name()`: no more prefix logic. Return bare model name.

### Phase 5: DX improvements (from session failure analysis)

These are cheap wins that fix the 10 DX issues found in today's transcripts:

**5a. Model name validation with suggestions:**
```python
def _validate_model_name(model: str, provider: str) -> str:
    """Validate model name, suggest corrections for common errors."""
    known = _known_models_for_provider(provider)
    if model not in known:
        close = difflib.get_close_matches(model, known, n=1, cutoff=0.6)
        if close:
            logger.error(f"Model '{model}' not found. Did you mean '{close[0]}'?")
        else:
            logger.error(f"Model '{model}' not found for {provider}. Known: {', '.join(known)}")
        raise ModelError(f"Unknown model: {model}", provider=provider, model=model)
    return model
```

**5b. Transport switch notification (already logged, make more visible):**
```python
# In chat() dispatcher, after CLI fallback decision:
if cli_fallback_reason:
    logger.warn(f"[TRANSPORT] {cli_provider} → API ({cli_fallback_reason})")
    # Stderr: [llmx:TRANSPORT] gemini-cli → google-api (max_tokens not supported by CLI)
```

**5c. Reasoning effort validation at parse time (already exists, make error actionable):**
```python
# Already validates — but make the error message agent-friendly:
raise ModelError(
    f"Model {model} supports reasoning_effort: {', '.join(valid_levels)}. "
    f"You passed: {reasoning_effort}. "
    f"Fix: use --reasoning-effort {valid_levels[-1]}",
    provider=provider, model=model,
)
```

**5d. JSON diagnostic on stderr for all errors:**
```python
# Replace free-text stderr with structured JSON:
def diagnostic_line(self) -> str:
    """Machine-parseable JSON for agent consumption."""
    return json.dumps({
        "error": self.error_type,
        "provider": self.provider,
        "model": self.model,
        "exit_code": self.exit_code,
        "action": self._suggested_action(),
    })

def _suggested_action(self) -> str:
    if self.exit_code == EXIT_RATE_LIMIT:
        return "use --fallback MODEL or wait 60s"
    elif self.exit_code == EXIT_TIMEOUT:
        return "increase --timeout or reduce context size"
    elif self.exit_code == EXIT_MODEL_ERROR:
        return "check model name and parameters"
    return "check stderr with --debug"
```

**5e. `-f` flag: fail fast with Gemini CLI transport:**
Currently `-f` is handled by llmx before provider dispatch (reads file, prepends to prompt). The hang is in the CLI backend receiving a huge prompt via command arg. Fix: when prompt exceeds `_ARG_MAX_BYTES`, the CLI backend already switches to stdin pipe. Verify this works for gemini-cli. If not, force API fallback with clear message.

**5f. Truncation detection for Google responses:**
```python
# After google-genai response:
if response.candidates[0].finish_reason == "MAX_TOKENS":
    logger.warn(f"[TRUNCATED] Output hit max_tokens limit ({max_tokens or '8192 server default'})")
```

### Phase 6: Update CLI backends

**6a. Add claude-cli backend:**
```python
CLI_PROVIDERS["claude-cli"] = {
    "binary": "claude",
    "api_fallback": "anthropic",  # → OpenRouter
}
CLI_PROVIDER_ALIASES["anthropic"] = "claude-cli"
```

`needs_api_fallback()` additions:
- Check `os.environ.get("CLAUDECODE")` → "nested Claude Code session"
- claude-cli supports: system prompt (`--system-prompt`), structured output (`--json-schema`), streaming (`--output-format stream-json`)
- claude-cli does NOT support: max_tokens, search grounding, reasoning_effort

`cli_chat()` additions for claude-cli:
```python
elif binary == "claude":
    cmd = ["claude", "-p", "--output-format", "text", "--tools", ""]
    if model:
        cmd.extend(["--model", model])
    # --tools "" disables agent tools for raw LLM mode
    if use_stdin:
        cmd.append("-")  # read from stdin
        stdin_input = prompt
    else:
        cmd.append(prompt)
```

**6b. Update codex-cli to use `-o` and `--output-schema`:**
The codex CLI now supports `-o FILE` (output last message to file) and `--output-schema FILE` (structured output). Wire these into `cli_chat()` when schema is provided.

### Phase 7: Remove LiteLLM + test

- Remove `litellm>=1.79.0` from pyproject.toml
- Remove `from litellm import completion, stream_chunk_builder`
- `uv sync` to clean venv
- Run `test_native_sdks.py`
- Manual model-review test with real context
- Bump version to 0.6.0
- Update README

### Phase 8: Update llmx-guide skill + model-review skill

- Update flag documentation (model names no longer need prefixes)
- Add new DX features to troubleshooting section
- Update model-review templates to use `--fallback` consistently
- Document JSON stderr diagnostics for agent consumption

## Risk Assessment

**Low risk:**
- Google and OpenAI paths tested and working
- CLI backends unchanged (extended, not modified)
- Error types/exit codes unchanged (public interface preserved)
- CLI flags unchanged (user-facing interface preserved)

**Medium risk:**
- OpenAI-compat providers may not support all features (reasoning_effort, structured output)
- Anthropic via OpenRouter adds latency + OpenRouter dependency
- google-genai SDK exception types need mapping (verify at implementation time)

**Not a risk:**
- "New provider?" → Add base_url. One line.
- "LiteLLM had feature X" → We tested every feature llmx actually uses. None are LiteLLM-only.

## What NOT to change

- `cli.py` — CLI interface unchanged
- `api.py` — Python API (calls providers.chat internally, gets the fix for free)
- `helpers.py` — retry/cache decorators, orthogonal
- Error types and exit codes — public interface
- `--output`, `--fallback` CLI flags — orthogonal
- `image.py`, `vision.py`, `research.py` — use google-genai / openai directly already
