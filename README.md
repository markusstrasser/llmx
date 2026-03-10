# llmx

Unified CLI and Python API for LLM providers.

## Install

```bash
# From GitHub
uv tool install git+https://github.com/markusstrasser/llmx

# Local editable
uv tool install --editable /path/to/llmx
```

## CLI

```bash
# Default provider (Gemini 3.1 Pro via Gemini CLI when installed)
llmx "What is 2+2?"

# Model auto-infers provider
llmx -m gpt-5.4 "Explain Python"
llmx -m claude-opus-4-6 "Write code"
llmx -m kimi-k2.5 "Complex task"
llmx -m cerebras/qwen-3-coder-480b "Fast coding"

# Pipe input
cat code.py | llmx -m claude-sonnet-4-6 "Review this"

# Web search grounding (Google, Anthropic)
llmx --search "Latest news on fusion energy"

# Fast mode (Gemini Flash + low reasoning effort)
llmx --fast "Quick question"

# Control thinking budget (OpenAI, Gemini)
llmx -m gpt-5.4 --reasoning-effort xhigh "Hard task"
llmx -m gemini-3-flash --reasoning-effort high "Hard task"

# Force direct APIs instead of subscription CLIs
llmx -p openai "Reply with OK"
llmx -p google --search "Latest news on fusion energy"

# System prompt (works with both CLI and API transport)
llmx -s "You are terse" "Reply with OK"

# Compare providers side-by-side
llmx --compare "Tabs or spaces?"

# Stream output
llmx --stream "Tell me a story"

# JSON output
llmx --json "Generate {name, age}"
```

### Subcommands

```bash
# Image generation (Gemini 3 Pro Image)
llmx image "a cute robot mascot" -o robot.png
llmx image "pixel art knight" -r 2K -a 16:9

# SVG generation
llmx svg "momentum arrow icon" -o arrow.svg

# Vision analysis (images + video)
llmx vision screenshot.png -p "What UI issues do you see?"
llmx vision gameplay.mp4 -p "List all UI elements"

# Deep research (OpenAI o3/o4-mini, takes 2-10 min)
llmx research "economic impact of semaglutide on healthcare"
llmx research --mini "compare React vs Svelte" -o report.md
```

## Python API

```python
from llmx import chat, LLM, batch

# One-shot call
response = chat("What is 2+2?", provider="openai")
print(response.content)   # "4"
print(response.usage)     # {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
print(response.latency)   # 1.23

# With system message and search
response = chat("Latest on CRISPR?", provider="google", search=True, system="Be concise")

# Stateful client
llm = LLM(provider="anthropic", model="claude-sonnet-4-6", temperature=0.3)
r1 = llm.chat("Explain Python")
r2 = llm.chat("Now compare to Rust", temperature=0.7)  # override temp

# Streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Batch (parallel)
responses = batch(["Q1", "Q2", "Q3"], provider="google", parallel=3)
```

### Response object

```python
@dataclass
class Response:
    content: str           # LLM text output
    provider: str          # "google", "openai", etc.
    model: str             # Full model name
    usage: dict            # {prompt_tokens, completion_tokens, total_tokens}
    latency: float         # Seconds
    raw: Any               # Raw provider response
```

### Inspection

```python
from llmx.inspect import stats, last_request, last_response, history, clear

chat("Hello", provider="openai")
chat("World", provider="google")

stats()
# {'total_calls': 2, 'total_tokens': 42, 'avg_latency': 1.5,
#  'by_provider': {'openai': {'calls': 1, ...}, 'google': {'calls': 1, ...}}}

last_request()   # {provider, model, messages, time}
last_response()  # {content, usage, latency, time}
history(limit=5) # List of trace dicts
clear()          # Reset
```

### Helpers

```python
from llmx.helpers import retry, cache, validate_prompt

@retry(max_attempts=3, backoff=2.0)
def flaky_call():
    return chat("prompt", provider="openai")

@cache(ttl=3600)  # 1 hour
def expensive_analysis(code):
    return chat(f"Analyze: {code}", provider="anthropic")

prompt = validate_prompt(user_input, min_length=5, max_length=10000)
```

## Providers

| Provider | Default model | Flag |
|----------|--------------|------|
| `google` | Gemini 3.1 Pro | (default) |
| `openai` | GPT-5.4 | `-p openai` |
| `anthropic` | Claude Opus 4.6 | `-p anthropic` |
| `xai` | Grok 4 | `-p xai` |
| `kimi` | Kimi K2.5 | `-p kimi` |
| `cerebras` | Qwen 3 Coder 480B | `-p cerebras` |
| `deepseek` | DeepSeek Chat | `-p deepseek` |
| `openrouter` | 400+ models | `-p openrouter` |

Transport defaults:

- `google` prefers `gemini` CLI when installed, then falls back to the Gemini API.
- `openai` prefers `codex exec` when installed, then falls back to the OpenAI API.
- `codex-cli` supports JSON schema output via `codex exec --output-schema`.
- `-s` (system messages) works with CLI transports — they're folded into the prompt as `<system>` XML.

All thinking models (GPT-5.x, Gemini 3.x, Kimi K2.5) have temperature fixed at 1.0.

## API Keys

Set in `.env` or environment:

```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MOONSHOT_API_KEY=...
CEREBRAS_API_KEY=...
XAI_API_KEY=...
```
