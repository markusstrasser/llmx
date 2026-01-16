# llmx

Unified CLI for LLM providers via LiteLLM. Simple tool for calling LLM APIs from scripts.

## Installation

```bash
uv tool install --editable /path/to/llmx
```

## Quick Start

```bash
# Default (Gemini)
llmx "What is 2+2?"

# Specify model (provider auto-inferred)
llmx --model gpt-5-pro "Explain Python"
llmx --model claude-sonnet-4-5 "Write code"
llmx --model kimi-k2-thinking "Complex task"
llmx --model cerebras/qwen-3-coder-480b "Fast coding"

# Pipe input
cat code.txt | llmx --model claude-sonnet-4-5 "Review this"

# Compare models
llmx --compare "Which is better: tabs or spaces?"
```

## Models

### GPT-5 (OpenAI)
```bash
llmx --model gpt-5-pro --reasoning-effort high "complex task"
llmx --model gpt-5-codex --reasoning-effort medium "code task"
```
- **gpt-5-pro**: minimal/low/medium/high effort
- **gpt-5-codex**: low/medium/high (coding specialist)
- Temperature fixed at 1.0

### Gemini (Google)
```bash
llmx --model gemini-2.5-pro "general task"
llmx --model gemini-2.5-flash "search query"  # Search ONLY
```
- **Pro**: General tasks, reasoning, code
- **Flash**: Search/retrieval ONLY (warns if misused)

### Claude (Anthropic)
```bash
llmx --model claude-sonnet-4-5 "your task"
```
- Best coding model, complex agents
- Temperature 0.0-1.0

### Kimi K2 (Moonshot)
```bash
llmx --model kimi-k2-thinking "complex reasoning"
llmx --model moonshot/kimi-k2-0905-preview "fast task"
```
- **thinking**: Agentic reasoning, temp=1.0 fixed
- **0905-preview**: Fast instruct, 256K context, variable temp

### Cerebras
```bash
llmx --model cerebras/qwen-3-coder-480b "coding task"
llmx --model cerebras/qwen-3-235b "general task"
llmx --model cerebras/qwen-3-32b "fast reasoning"
```
- Ultra-fast inference (1400+ tokens/sec)
- Free tier: 1M tokens/day

## Options

```bash
llmx --model MODEL --temperature 0.7 "prompt"
llmx --model MODEL --stream "prompt"
llmx --model MODEL --timeout 300 "prompt"
llmx --model MODEL --json "prompt"
llmx --model MODEL --debug "prompt"
llmx --compare "prompt"  # Compare multiple providers
```

## Temperature

Auto-adjusted per model:
- **GPT-5, Kimi K2 Thinking**: Fixed at 1.0
- **Claude**: 0.0-1.0
- **Cerebras**: 0.0-1.5
- **Gemini**: 0.0-2.0

Warnings shown if override attempted.

## Providers

```bash
llmx --list-providers
```

- `google` - Gemini 2.5
- `openai` - GPT-5
- `anthropic` - Claude Sonnet 4.5
- `kimi` - Kimi K2
- `cerebras` - Qwen 3
- `xai` - Grok
- `deepseek` - DeepSeek
- `openrouter` - 400+ models

## API Keys

Set in `.env`:
```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
MOONSHOT_API_KEY=...
CEREBRAS_API_KEY=...
```

## Image Generation

Generate images using Gemini 3 Pro Image (Nano Banana Pro):

```bash
# Basic image generation
llmx image "a cute robot mascot" -o robot.png

# With options
llmx image "pixel art knight" -o knight.png -r 2K          # 2K resolution
llmx image "game background" -a 16:9                        # Aspect ratio
llmx image "physics diagram" -m pro --debug                 # Debug mode

# Generate SVGs
llmx svg "momentum arrow icon" -o arrow.svg
```

### Image Options
- `-o, --output` - Output file path (default: auto-generated)
- `-m, --model` - `flash` or `pro` (both use gemini-3-pro-image-preview)
- `-r, --resolution` - `1K`, `2K`, or `4K`
- `-a, --aspect-ratio` - `1:1`, `16:9`, `9:16`, `4:3`, etc.

**Note:** Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable.

## Features

- Auto-provider detection from model name
- Temperature validation per model
- Reasoning effort (GPT-5 only)
- Compare mode (parallel requests)
- Streaming & JSON output
- Smart warnings (Flash misuse, etc)
- **Image generation** (Gemini 3 Pro Image)
