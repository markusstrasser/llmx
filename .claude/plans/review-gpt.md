## What Will Break

1. **`api.py` will break if `providers.chat()` currently returns data instead of printing**
   - In both new helpers you do output directly:
     ```python
     # _google_chat()
     print(response.text)
     # _openai_chat()
     print(response.choices[0].message.content)
     ```
     and for streaming:
     ```python
     sys.stdout.write(delta)
     ```
   - That is CLI behavior, not provider behavior.
   - You explicitly say:
     > `api.py` — Python API unchanged
   - That is only true if `providers.chat()` already prints to stdout today. If `api.py` expects a string/result object/generator, this migration is a contract break.
   - Related: removing `stream_chunk_builder` is not free. If current code uses it to assemble streaming text into a final return value, you need an equivalent.

2. **Model naming will break in two different ways**
   - You are correctly removing **LiteLLM synthetic prefixes** like:
     - `gemini/...`
     - `moonshot/...`
     - maybe `xai/...`
   - But Phase 4 overgeneralizes:
     > “Simplify `get_model_name()`: no more prefix logic. Return bare model name.”
   - That will break **real provider model IDs that legitimately contain slashes**, especially OpenRouter:
     - `anthropic/claude-sonnet-4-6`
     - `google/gemini-2.5-pro`
     - `deepseek/deepseek-r1`
   - Those are not LiteLLM artifacts; they are actual model IDs.
   - You also have a double-prefix bug here:
     ```python
     if provider == "anthropic":
         model = f"anthropic/{model}"
     ```
     If config or user input already contains `anthropic/...`, you generate `anthropic/anthropic/...`.

3. **Existing scripts using old prefixed model names will start failing**
   - Today users may pass:
     - `gemini/gemini-3.1-pro-preview`
     - `moonshot/kimi-k2.5`
     - `xai/grok-4`
   - The migration says model names “no longer need prefixes”, but if you remove support entirely, that is a breaking change.
   - Since you claim “CLI flags unchanged” and “public interface preserved”, you should accept old names and normalize them with a deprecation warning.

4. **Error mapping as written is wrong / incomplete**
   - Phase 3 shows:
     ```python
     except google.api_core.exceptions.ResourceExhausted as e:
     except google.api_core.exceptions.DeadlineExceeded as e:
     except openai.RateLimitError as e:
     except openai.APITimeoutError as e:
     ```
   - But your code only imports:
     ```python
     from openai import OpenAI
     ```
     not `import openai`, so those exception references do not exist in that snippet.
   - More importantly, `google-genai` errors are typically from `google.genai.errors`, not necessarily `google.api_core.exceptions`.
   - If you ship this as written, typed llmx errors will regress into generic failures.
   - Also missing important cases:
     - 401 auth
     - 402 insufficient credits (very relevant for OpenRouter)
     - 403 permission/model access
     - 404 unknown model
     - connection failures / DNS / TLS
     - generic 5xx retryable errors

5. **Anthropic routing will likely use the wrong API key**
   - You define:
     ```python
     "anthropic": "https://openrouter.ai/api/v1"
     ```
     and:
     ```python
     "api_fallback": "anthropic"  # routes to OpenRouter
     ```
   - If `_get_api_key(provider)` maps provider name to env name mechanically, `provider="anthropic"` will look for `ANTHROPIC_API_KEY`.
   - But your plan says Anthropic API credits are unavailable and the real route is OpenRouter, so this needs `OPENROUTER_API_KEY`.
   - If you do not special-case that, fallback from `claude-cli` to API will fail with the wrong credential.

6. **`CLI_PROVIDER_ALIASES["anthropic"] = "claude-cli"` changes behavior more than the plan admits**
   - This means “anthropic” stops meaning “Anthropic API-like provider” and starts meaning “local Claude Code subscription transport if available”.
   - That changes:
     - auth path
     - pricing model
     - feature support
     - nested-session behavior
   - In particular, inside a nested Claude session, `anthropic` would now silently fall back to OpenRouter API if available, or fail if not.
   - That is not “public interface preserved”.

7. **Claude CLI support is declared, but not actually wired in the snippet**
   - You say Claude CLI supports:
     - system prompt
     - structured output
     - streaming
   - But the actual command builder shown does not pass any of those:
     ```python
     cmd = ["claude", "-p", "--output-format", "text", "--tools", ""]
     if model:
         cmd.extend(["--model", model])
     ```
   - Missing:
     - `--system-prompt`
     - `--json-schema`
     - `--output-format stream-json` when `stream=True`
   - So unless `needs_api_fallback()` forces API for those cases, you’ll silently ignore requested features.

8. **Timeout semantics will change in user-visible ways**
   - Google path:
     ```python
     timeout=max(timeout * 1000, 10_000)
     ```
   - That means `--timeout 5` becomes **10 seconds minimum**.
   - If users rely on short timeouts today, behavior changes.
   - OpenAI path:
     ```python
     client = OpenAI(..., timeout=float(timeout))
     ```
     does **not** solve your original wall-clock hang problem on its own.
   - And SIGALRM is not a general fix:
     - main thread only
     - Unix only
     - problematic in threaded/API environments
   - So the claim:
     > `api.py` gets the fix for free
     is too optimistic.

9. **Structured output is still not reliable with the proposed code**
   - OpenAI:
     ```python
     kwargs["response_format"] = {
         "type": "json_schema",
         "json_schema": {"name": "response", "schema": schema},
     }
     ```
   - Missing `"strict": True` for OpenAI strict structured outputs.
   - More importantly, you do not parse + validate the returned JSON against the schema after generation.
   - That means DX issue #10:
     > Silent claim loss in structured extraction
     is **not fixed**. You are still trusting provider-side conformance.
   - Same problem on Google:
     ```python
     config.response_schema = schema
     ```
     without validating the returned object/text.

10. **Truncation detection is only addressed for one path, and even there likely wrong**
   - You add Google-only logic:
     ```python
     if response.candidates[0].finish_reason == "MAX_TOKENS":
     ```
   - Problems:
     - likely enum vs string mismatch
     - only non-stream Google path
     - no OpenAI/OpenRouter equivalent (`finish_reason == "length"`)
     - no streamed truncation handling
   - So truncated JSON/text will still often look like a success.

---

## Missing Edge Cases

1. **`timeout` can be `None`**
   - This crashes:
     ```python
     max(timeout * 1000, 10_000)
     ```
   - If `timeout` is optional anywhere in the call chain, `_google_chat()` needs guarding.

2. **`_validate_model_name()` will reject valid OpenRouter/custom models**
   - This is risky:
     ```python
     known = _known_models_for_provider(provider)
     if model not in known:
         raise ModelError(...)
     ```
   - For OpenRouter especially, models are dynamic and vendor-scoped.
   - Hard-failing unknown models will block legitimate use cases.

3. **Google schema dialect != OpenAI schema dialect**
   - Reusing one `schema` object across both backends is not safe.
   - Likely failure points:
     - `oneOf` / `anyOf`
     - `$defs` / `$ref`
     - `additionalProperties`
     - nullable semantics
     - enum typing
   - You need schema normalization per backend, not direct passthrough.

4. **Structured output + streaming is not addressed**
   - If `schema` + `stream=True`, your current behavior is “print partial JSON chunks to stdout”.
   - That is fine for humans, bad for machine consumers.
   - On provider/CLI failure mid-stream, you leave invalid JSON in output with no post-validation.

5. **`response.choices[0].message.content` can be `None`**
   - On OpenAI-compatible APIs, structured/refusal/tool-call responses can put useful data in:
     - `message.refusal`
     - tool call fields
     - provider-specific content structures
   - Current code assumes plain text always exists.

6. **Streaming chunks can be empty or non-text**
   - This is unsafe:
     ```python
     delta = chunk.choices[0].delta.content
     ```
   - You need to handle:
     - empty `choices`
     - `delta.content is None`
     - reasoning-only/tool-call chunks
     - final chunks carrying only `finish_reason`

7. **`temperature` / `reasoning_effort` / `max_tokens` support is not uniform**
   - The plan acknowledges this, but code still sends params unconditionally.
   - Concrete examples:
     - some OpenAI-compatible models reject `reasoning_effort`
     - some reject `response_format`
     - some reject `temperature`
   - You need per-model capability gating before request creation.

8. **Old/new auth env names are not covered**
   - For Google, are you relying on SDK-default env detection, or preserving current llmx env names?
   - For Anthropic-via-OpenRouter, provider name and env name now differ.
   - This migration needs an explicit env compatibility policy.

9. **Nested Claude detection via only `CLAUDECODE` is fragile**
   - Good first pass, but incomplete.
   - You should also detect the actual CLI failure mode from process stderr/exit code, because env-based detection can false-negative.

10. **CLI prompt transport should probably default to stdin, not only on size threshold**
   - Your note says:
     > verify `_ARG_MAX_BYTES` → stdin pipe works for gemini-cli
   - I’d go further: for all CLI backends, prefer stdin by default.
   - That avoids:
     - shell quoting issues
     - command-line length limits
     - newline mangling
     - accidental prompt leakage in process listings

11. **Codex/Claude file-output paths need cleanup/error handling**
   - If you switch Codex to `-o FILE` / `--output-schema FILE`, make sure you handle:
     - temp file cleanup on timeout/error
     - reading file before exit
     - interactions with llmx `--output`
     - empty file vs failed run

12. **OpenRouter credit exhaustion is not the same as rate limiting**
   - You already observed Anthropic credits issues.
   - OpenRouter often returns useful billing errors that should become actionable llmx errors, not generic `APIStatusError`.

---

## Top 5 Recommendations

1. **Preserve the existing provider contract; do not print from `_google_chat()` / `_openai_chat()`**
   - Make providers return a normalized result object, e.g.:
     ```python
     @dataclass
     class ChatResult:
         text: str
         finish_reason: str | None
         usage: dict | None
         raw: Any
     ```
   - Then CLI decides whether to print, stream, save to file, etc.
   - This avoids breaking `api.py`, retries/cache decorators, and `--output`.

2. **Do not globally remove “prefix logic”; replace it with model normalization**
   - You need to distinguish:
     - **synthetic LiteLLM prefixes** to strip
     - **real provider model IDs** to preserve
   - Example:
     ```python
     def normalize_model(provider: str, model: str) -> str:
         if provider == "google":
             return model.removeprefix("gemini/")
         if provider == "kimi":
             return model.removeprefix("moonshot/")
         if provider == "xai":
             return model.removeprefix("xai/")
         if provider == "anthropic":
             m = model.removeprefix("anthropic/")
             return f"anthropic/{m}"   # OpenRouter route
         if provider == "openrouter":
             return model             # preserve vendor/model
         return model
     ```
   - Also accept old user-supplied prefixed names with a warning for at least one release.

3. **Add a capability matrix and filter params before request construction**
   - Right now the plan knows features differ, but the code still blindly passes params.
   - Add a table like:
     ```python
     CAPS = {
         ("google", "gemini-3.1-pro-preview"): {"system", "stream", "schema", "search", "reasoning"},
         ("openai", "gpt-5.4"): {"system", "stream", "schema", "reasoning"},
         ("openrouter", "anthropic/claude-sonnet-4-6"): {"system", "stream", "schema"},
     }
     ```
   - Then:
     - omit unsupported kwargs
     - force API/CLI fallback before making the call
     - raise actionable errors instead of provider 400s

4. **Treat timeout and error mapping as first-class migration work**
   - Concretely:
     - import the real exception modules
     - map by status code where needed
     - do not promise wall-clock guarantees you cannot enforce
   - At minimum:
     ```python
     import openai
     from google.genai import errors as genai_errors
     ```
   - And map:
     - 401/403 auth/permission
     - 402 billing/credits
     - 404 model
     - 429 rate limit
     - timeout / connection
     - 5xx retryable
   - Also: make JSON diagnostics **additive**, not a replacement for human stderr, unless gated by a flag. Replacing stderr format is itself a breaking change.

5. **Before removing LiteLLM, add integration tests for the real compatibility boundaries**
   - `test_native_sdks.py` is helpful, but it is not enough.
   - Add end-to-end tests for:
     - old prefixed model names still accepted
     - OpenRouter slashed model IDs preserved
     - anthropic→OpenRouter key resolution
     - structured output + truncation detection
     - refusal / empty-output handling
     - timeout in a worker thread
     - CLAUDECODE nested fallback
     - CLI stdin transport with large `-f` prompts
   - I would not remove LiteLLM until these pass in the llmx interface layer, not just in isolated SDK probes.

Overall: **direction is good**, but the plan is not “ready to execute” yet. The biggest gaps are **interface compatibility**, **model normalization**, **error mapping**, and **structured-output correctness**.