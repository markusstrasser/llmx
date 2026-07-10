# llmx

Unified CLI and Python API for LLM providers — one transport layer for Google,
OpenAI, Anthropic, xAI, Cursor, Codex, OpenRouter, and friends.

**Who owns what**

| Layer | Owns | Does not own |
|-------|------|--------------|
| **CLI** (`llmx chat`, …) | Flags, files, stderr dispatch line, exit codes | Model judgment / cosigner policy |
| **`dispatch()`** | Structured one-shot: status taxonomy, context concat, dry-run plan, `auth=` | Profiles, critique axes (skills) |
| **`chat()` / `LLM`** | Transport engine (CLI backends + paid APIs); raises on failure | Status taxonomy for callers |
| **`llmx info` / mirror** | Installed CLIs, subscription routes, effort aliases | Which model to pick for a task |

Task-class model choice lives in the **model-guide** skill. Footguns during
migration: **llmx-guide** skill. Routing facts: `llmx info --write-mirror` →
`~/.claude/cache/llmx-routing.json`.

## Install

```bash
# From GitHub
uv tool install git+https://github.com/markusstrasser/llmx

# Local editable
uv tool install --editable /path/to/llmx
```

## CLI

`llmx` with no subcommand is `llmx chat`.

```bash
# Model auto-infers provider
llmx -m gpt-5.6-sol "Explain Python"
llmx -m claude-opus-4-8 "Write code"          # → claude-cli subscription
llmx -m kimi-k2.5 "Complex task"
llmx -m cerebras/qwen-3-coder-480b "Fast coding"

# Pipe / files as context (repeatable -f concatenates with === File: path ===)
cat code.py | llmx -m claude-opus-4-8 "Review this"
llmx chat -f a.md -f b.md -m gpt-5.6-sol "Synthesize"

# Subscription routes (OAuth / pool — preferred for Claude + GPT batch work)
llmx chat --subscription -m gpt-5.6-sol "Quick task"
llmx chat --subscription -m claude-opus-4-8 "Review this"

# Workspace agent: caller cwd + project rules + native CLI tools
llmx chat --subscription --mode agent -m claude-opus-4-8 -e max \
  --timeout 3600 -o out.md \
  "Inspect this repository read-only; cite file:line evidence."

# Probe resolved transport before spend
llmx chat --dry-run --subscription -m claude-opus-4-8 -e max "ping"
llmx info --write-mirror          # → ~/.claude/cache/llmx-routing.json
llmx probe --provider anthropic   # one bounded live subscription call

# Auth surface: --auth api|subscription  (--subscription aliases --auth subscription)
llmx chat --auth api -p openai -m gpt-5.6-luna "cheap extract"
llmx -p anthropic-direct -m claude-opus-4-8 "metered Claude API (opt-in)"

# Effort / timeout / output
llmx -m gpt-5.6-sol -e max --timeout 3600 -o out.md "Hard task"
llmx -m gemini-3.5-flash -e high "Hard task"
llmx --fast "Quick question"          # Gemini Flash + low effort
llmx --search "Latest on fusion"      # Google grounding
llmx --stream "Tell me a story"
llmx --json "Generate {name, age}"
llmx --compare "Tabs or spaces?"
llmx -s "You are terse" "Reply with OK"
```

### Timeouts

Default wall-clock is **300s**. llmx auto-raises defaults for agent mode and
high/max effort (up to **1800s** agent / **3600s** max). Ceiling is 3600s.
Pass `--timeout` explicitly when the caller owns a tighter or longer budget.
Use `-o FILE` for long runs (never bare `> file` in background).

### Subcommands

```bash
llmx chat …          # text generation (default)
llmx info            # routing facts; --write-mirror for agents
llmx probe           # bounded live subscription entitlement check
llmx usage           # cost/token rollup from ~/.claude/llmx-usage.jsonl
llmx usage --by model --days 7
llmx keys …          # macOS Keychain helpers
llmx batch …         # Gemini Batch API (async, ~50% off)
llmx image …         # image gen/edit (GPT Image 2 default)
llmx svg …           # SVG via Gemini
llmx vision …        # image/video analysis
llmx research …      # deep research (OpenAI o3/o4-mini or Perplexity)
```

## Python API

Two call styles share the same transport engine:

| API | Returns | Failures | Use when |
|-----|---------|----------|----------|
| **`dispatch()`** | `DispatchResult` | Status + `exit_code` (does not raise on rate-limit/quota/timeout) | Scripts, skills, anything that needs structured outcomes |
| **`chat()` / `LLM`** | `Response` | Raises `LlmxError` subclasses | Simple scripts, streaming, when you want exceptions |

`dispatch()` calls `chat()` internally after resolving auth/mode/effort.

### `dispatch()` — structured one-shot

```python
from llmx import dispatch, DispatchResult

result = dispatch(
    "What is 2+2?",
    provider="openai",          # optional if model implies provider
    auth="api",                 # or subscription=True / auth="subscription"
    context_paths=["a.md", "b.md"],  # concatenated with path boundaries
    effort="high",
    timeout=300,
    output_path="out.md",       # optional
    caller="my_script.py",      # usage-log attribution
)

if result.ok():
    print(result.text, result.usage, result.transport, result.effort_applied)
else:
    print(result.status, result.error_message, result.exit_code)
    # status ∈ ok|dry_run|timeout|rate_limit|quota|api_key|model_error|
    #          schema_error|empty_output|config_error|dependency_error|
    #          dispatch_error|spend_cap

# Resolve transport without calling a model
plan = dispatch("ping", model="claude-opus-4-8", subscription=True, dry_run=True)
assert plan.status == "dry_run"
print(plan.dry_run_plan)   # provider, transport, auth, effort_applied, warnings, …
print(plan.warnings)
```

`api_only=` is still accepted but deprecated — use `auth="api"|"subscription"`.

### `chat()` — transport engine

```python
from llmx import chat, LLM, batch

response = chat("What is 2+2?", provider="openai", auth="api")
print(response.content, response.usage, response.latency)

response = chat(
    "Latest on CRISPR?",
    provider="google",
    search=True,
    system="Be concise",
)

# Stateful client (multi-turn / stream)
llm = LLM(provider="openai", model="gpt-5.6-sol", temperature=0.3)
r1 = llm.chat("Explain Python")
r2 = llm.chat("Now compare to Rust", temperature=0.7)
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)

responses = batch(["Q1", "Q2", "Q3"], provider="google", parallel=3)
```

### Result types

```python
@dataclass
class DispatchResult:
    status: str
    retryable: bool
    text: str
    provider: str
    model: str
    transport: str          # e.g. claude-cli, google-api, codex-cli
    auth: str               # api | subscription
    mode: str               # chat | agent
    effort_applied: str | None
    warnings: list[str]
    usage: dict
    latency: float
    error_type: str | None
    error_message: str | None
    dry_run_plan: dict | None
    # .ok() → status == "ok"
    # .exit_code → CLI-aligned exit code
    # .content → alias for .text

@dataclass
class Response:
    content: str
    provider: str
    model: str
    usage: dict             # prompt_tokens, completion_tokens, total_tokens, …
    latency: float
    raw: Any
```

### Exit codes (CLI + `DispatchResult.exit_code`)

| Code | Meaning |
|------|---------|
| 0 | Success / dry-run |
| 1 | General / config |
| 2 | API key |
| 3 | Rate limit / transient 503 |
| 4 | Timeout |
| 5 | Model / request error |
| 6 | Quota / billing / spend cap |

### Inspection & helpers

```python
from llmx.inspect import stats, last_request, last_response, history, clear
from llmx.helpers import retry, cache, validate_prompt

chat("Hello", provider="openai")
stats()            # totals + by_provider
last_request()
history(limit=5)
clear()

@retry(max_attempts=3, backoff=2.0)
def flaky():
    return chat("prompt", provider="openai")

@cache(ttl=3600)
def expensive(code):
    return chat(f"Analyze: {code}", provider="openai")
```

## Providers & transport

| Provider | Default model | Notes |
|----------|---------------|-------|
| `google` | Gemini 3.x | Paid API (free Gemini CLI retired 2026-05-31) |
| `openai` | GPT-5.6 Sol | API by default; `--subscription` → `codex-cli` |
| `anthropic` | Claude Opus 4.8 | **claude-cli subscription by default**; keys stripped |
| `anthropic-direct` | Claude Opus 4.8 | Metered Anthropic API (explicit opt-in) |
| `xai` | Grok 4.5 | xAI API; Cursor effort slugs also routed |
| `cursor` / `cursor-cli` | Composer / pool models | Subscription / Cursor pool |
| `kimi` | Kimi K2.5 | Moonshot |
| `cerebras` | Qwen 3 Coder 480B | Fast coding |
| `deepseek` | DeepSeek Chat | |
| `openrouter` | 400+ models | |
| `zai` | GLM-5.2 | Via OpenRouter id today |

**Claude policy:** never route Claude through paid API unless explicitly requested
(`-p anthropic-direct` or `auth="api"`). Default:

```bash
llmx chat --subscription -m claude-opus-4-8 …
```

**Modes**

- `--mode chat` — one-shot request/response (isolated cache, tools off on CLI routes).
- `--mode agent` — workspace agent: caller cwd, project instructions, native tools.
  Subscription only. Not the same as deprecated `--lite research` (research-MCP-only profile).

**Auth**

- Prefer `--auth api|subscription` / Python `auth=`.
- `--subscription` is an alias for `--auth subscription`.
- `--lite bare|research` still work as deprecated aliases for mode/auth shaping.

**Other**

- GPT-5.6 suite: `gpt-5.6-sol` (alias `gpt-5.6`), `gpt-5.6-terra`, `gpt-5.6-luna`; effort includes `max`.
- Thinking models (GPT-5.x, Gemini 3.x, Kimi K2.5, …) fix temperature at 1.0.
- `-s` / `system=` works on CLI transports (folded into the prompt).
- Multi-`-f` / `context_paths=` concatenate with `=== File: path ===` boundaries.
- Usage log: every call → `~/.claude/llmx-usage.jsonl`; roll up with `llmx usage`.
- Metered spend cap enforced at the dispatch funnel (see spend-guard); subscription CLI usage is logged separately from API $.

## API keys

Resolved in order: process env → `.env` in cwd → macOS Keychain (`llmx keys`).

```bash
export OPENAI_API_KEY=sk-...
# or
llmx keys set OPENAI_API_KEY
llmx keys list
llmx keys get OPENAI_API_KEY
llmx keys delete OPENAI_API_KEY
```

Optional `.zshrc` export so other tools see Keychain values:

```bash
for _k in OPENAI_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY XAI_API_KEY MOONSHOT_API_KEY; do
    [ -z "${!_k:-}" ] && val=$(security find-generic-password -a "llmx" -s "$_k" -w 2>/dev/null) && export "$_k=$val"
done
```

| Provider | Env var | Get key |
|----------|---------|---------|
| Google | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) — needed only for `anthropic-direct` |
| xAI | `XAI_API_KEY` | [console.x.ai](https://console.x.ai/) |
| Kimi | `MOONSHOT_API_KEY` | [platform.moonshot.cn/console/api-keys](https://platform.moonshot.cn/console/api-keys) |
| Cerebras | `CEREBRAS_API_KEY` | [cloud.cerebras.ai](https://cloud.cerebras.ai/) |
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

Claude/Codex **subscription** routes use local CLI OAuth — do not put API keys on those paths (llmx strips them).

## Design notes

- **Architecture over argv folklore.** Transport resolution, effort mapping, and
  subscription-vs-API policy live in llmx; skills keep named profiles
  (`llm_dispatch`) and critique orchestration.
- Anchor: agent-infra `decisions/2026-06-15-llmx-refactor-dispatch-layer.md`.
- Prefer `dispatch()` in new Python callers. Prefer `llmx chat --dry-run` /
  `llmx info` before batch spend. Prefer `--subscription` for Claude.
