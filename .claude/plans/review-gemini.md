Here is a concrete review of your migration plan. You have several critical blind spots where the plan explicitly contradicts your own codebase, which will cause outright crashes in production.

### 1. What Will Break That the Plan Doesn't Account For

*   **`compare()` is completely ignored:** The migration plan focuses entirely on `chat()`. However, your current `providers.py` imports LiteLLM directly in the `compare()` function (`response = completion(**completion_kwargs)`). If you remove LiteLLM in Phase 7 without rewriting `compare()` to use your new `_google_chat` / `_openai_chat` dispatchers, `llmx --compare` will crash with an `ImportError`.
*   **Double-prefixing Anthropic models:** In Phase 2, you wrote: `if provider == "anthropic": model = f"anthropic/{model}"`. But in Phase 4's `PROVIDER_CONFIGS`, you left the Anthropic model as `"model": "anthropic/claude-opus-4-6"`. This will result in `anthropic/anthropic/claude-opus-4-6` being sent to OpenRouter, causing an immediate 404.
*   **Wrong Google SDK exceptions:** In Phase 3, you map `google.api_core.exceptions.ResourceExhausted`. The new `google-genai` SDK *does not use* `google.api_core` for exceptions. It uses `google.genai.errors.APIError`. Your `except` block will throw an `AttributeError` on the module, or fail to catch the exception entirely. 
*   **OpenAI streaming crashes on heartbeats:** In Phase 2, your streaming loop relies on `delta = chunk.choices[0].delta.content`. OpenRouter (which you are using for fallbacks) frequently sends keep-alive chunks where the `choices` array is empty (`[]`). `chunk.choices[0]` will throw an `IndexError`. You must check `if not chunk.choices:` before accessing index 0.

### 2. Missing Edge Cases in the SDK Migration

*   **`SIGALRM` in multithreaded contexts:** You noted keeping the `SIGALRM` band-aid for the OpenAI path. However, `compare()` uses a `ThreadPoolExecutor`. `signal.alarm()` *only works in the main thread* in Python. If `compare()` calls your updated `_openai_chat()` dispatcher in a worker thread, `signal.alarm(timeout)` will raise a `ValueError`. You must disable the alarm fallback if `threading.current_thread() != threading.main_thread()`.
*   **Google Safety Filter truncations:** In Phase 5f, you check `if response.candidates[0].finish_reason == "MAX_TOKENS"`. If Google's safety filters block the prompt, `response.candidates` is often an empty list, and the block reason is stored in `response.prompt_feedback`. Accessing `candidates[0]` in this state throws an `IndexError`.
*   **JSON Schema compatibility:** LiteLLM invisibly mapped your raw schema dictionaries into provider-compliant subsets. Passing `schema` directly to `openai` (Phase 2) will fail if your schema doesn't strictly adhere to OpenAI's structured outputs requirements (e.g., you must set `"additionalProperties": False` on all objects, and you cannot have optional fields without defaults). You need a normalization step for `schema` before passing it to native SDKs.

### 3. DX Gaps (Improvements missing or wrong)

*   **One-sided Truncation Detection:** You added truncation detection for Google (Phase 5f), but entirely missed it for OpenAI/xAI/Anthropic. OpenAI compatible APIs return `finish_reason == "length"`. Agents using OpenAI models will still experience the "8K silent truncation" DX issue (Issue #8) unless you add this to `_openai_chat`.
*   **JSON diagnostic on stderr (Phase 5d):** You plan to replace free-text stderr with structured JSON via `diagnostic_line()`. However, you are raising exceptions (`raise RateLimitError(...)`). Unless you catch these at the top-level CLI entrypoint and intentionally suppress the Python stack trace to *only* print the `.diagnostic_line()`, the agent will still get a messy Python traceback wrapping your JSON, defeating the purpose of machine-parseable output.
*   **Claude CLI raw mode:** In Phase 6, you use `claude -p ... --tools ""`. The Claude Code CLI (v2.1.71) uses `claude -p` exclusively for agent loops. To get true raw LLM output without agentic overhead, Claude Code expects you to rely on standard stdin pipes. Forcing `--tools ""` is a hack that occasionally causes Claude to hallucinate tool execution attempts in the text block.

### 4. Phase Ordering Issues

**Phase 4 must happen *before* or *simultaneously with* Phase 3.** 
If you rewire the dispatcher (Phase 3) before stripping the prefixes from `PROVIDER_CONFIGS` (Phase 4), your code will pass `gemini/gemini-3.1-pro-preview` to the `google-genai` SDK. The SDK doesn't strip prefixes natively; it will pass the literal string to the Google API, returning a `404 Model Not Found`. You cannot test Phase 3 successfully without Phase 4 already being complete.

### 5. Overconfidence Risks & Blind Spots

*   **"Error types/exit codes unchanged (public interface preserved)"**
    You are overconfident here. By bypassing LiteLLM, you are losing its status-code normalization. You assume `google.genai.errors.APIError` will clearly differentiate between RateLimit and ContextWindow exceeded, but `google-genai` often wraps 429s and 400s in generic string messages. Your Phase 3 mapping needs careful inspection of `e.code` and `e.message` to maintain the precise `EXIT_RATE_LIMIT` and `EXIT_MODEL_ERROR` codes, or agents will fall back to `EXIT_GENERAL`, breaking their retry logic.
*   **"CLI backends unchanged (extended, not modified)"**
    You assume the `codex` CLI supports `--output-schema` smoothly (Phase 6b). Codex CLI historically struggles with raw path references in piped execution contexts. Writing the schema to a `NamedTemporaryFile` and passing the path is fine, but you aren't deleting it if `subprocess.run` hangs and is killed by a signal. You have a resource leak in `cli_backends.py` if the timeout expires.
*   **API Fallbacks and Context Size**
    Your CLI fallback logic (`needs_api_fallback`) does not check prompt length. If a user passes a 500k token context to `gemini-cli`, it will attempt to process it locally/via the CLI. Does the CLI handle 500k context without OOMing the local Node/Python process? LiteLLM handled chunking/transport for massive contexts; you should enforce an explicit `api_fallback` if `len(prompt) > 200_000` to prevent CLI backend crashes.