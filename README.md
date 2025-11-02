# llmx

Unified CLI for 100+ LLM providers via LiteLLM. Simple tool for calling LLM APIs from scripts and skills.

## Installation

```bash
# Install with uv (editable mode for development)
uv tool install --editable /path/to/llmx

# Or install from directory
uv tool install /path/to/llmx
```

## Usage

### Basic Examples

```bash
# Default provider (Google Gemini)
llmx "What is 2+2?"

# Specific provider
llmx --provider openai "Explain Python"

# Pipe input
cat code.txt | llmx "Review this code"

# Compare providers
llmx --compare "Which is better: tabs or spaces?"
```

### OpenAI GPT-5 Models

GPT-5 models (gpt-5-pro, gpt-5-codex) have special requirements:

```bash
# GPT-5 Pro with reasoning
llmx --provider openai --model gpt-5-pro --reasoning-effort high "your prompt"

# Temperature is automatically adjusted to 1.0 (GPT-5 only supports temperature=1)
llmx --provider openai --model gpt-5-pro "your prompt"  # temp auto-set to 1.0
```

**Important:**
- GPT-5 models **only support temperature=1**
- llmx automatically adjusts temperature to 1.0 for GPT-5 models
- No manual override needed - just use GPT-5 and it works
- Use `--reasoning-effort high` for deeper reasoning

### OpenRouter

Access 400+ models through OpenRouter:

```bash
# Use default model (GPT-4o via OpenRouter)
llmx --provider openrouter "your prompt"

# Use specific OpenRouter model
llmx --provider openrouter --model "openrouter/anthropic/claude-3.5-sonnet" "your prompt"

# Or just specify the model directly (auto-detected)
llmx --model "openrouter/google/gemini-2.5-pro" "your prompt"
```

### Kimi K2

Moonshot AI's Kimi K2 model (default: 0905 version with 256K context):

```bash
# Use latest Kimi K2 model (0905 - Sept 2025, 256K context)
llmx --provider kimi "your prompt"

# Use older Kimi K2 model (0711 - July 2025, 128K context)
llmx --provider kimi --use-old "your prompt"

# Use specific Kimi model version
llmx --provider kimi --model "kimi-k2-0905-preview" "your prompt"

# Or use moonshot prefix directly
llmx --model "moonshot/kimi-k2-0905-preview" "your prompt"
```

### Advanced Options

```bash
# Custom temperature (non-GPT-5 models)
llmx --provider google --temperature 0.3 "your prompt"

# Streaming output
llmx --provider openai --stream "your prompt"

# JSON output
llmx --provider openai --json "your prompt"

# Debug logging
llmx --provider openai --debug "your prompt"

# Use older model version (currently only for Kimi K2)
llmx --provider kimi --use-old "your prompt"

# Compare specific providers
llmx --compare --providers "google,openai,xai" "your prompt"
```

## Supported Providers

- `google` - Gemini 2.5 Pro (default)
- `openai` - GPT-4o, GPT-5 Pro, GPT-5 Codex, o1
- `anthropic` - Claude 3.5 Sonnet
- `xai` - Grok Beta
- `deepseek` - DeepSeek Chat
- `openrouter` - Access 400+ models through OpenRouter
- `kimi` - Kimi K2 by Moonshot AI

List all providers:
```bash
llmx --list-providers
```

## Environment Variables

Set API keys in `.env` or environment:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
GROK_API_KEY=...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
MOONSHOT_API_KEY=...
KIMI_API_KEY=...
```

## Development

The project uses LiteLLM under the hood for unified API access.

### Key Features

- **Auto-provider detection**: Infers provider from model name
- **GPT-5 auto-adjustment**: Automatically sets temperature=1 for GPT-5 models
- **Reasoning effort**: Supports OpenAI reasoning effort parameter
- **Comparison mode**: Compare responses from multiple providers in parallel
- **Streaming**: Optional streaming output
- **JSON mode**: Structured JSON output for scripting

### Code Structure

```
llmx/
├── llmx/
│   ├── __init__.py      # Version info
│   ├── cli.py           # Main CLI interface
│   ├── providers.py     # Provider configs and chat logic
│   └── logger.py        # Structured logging
├── pyproject.toml       # Package config
└── README.md           # This file
```

## Usage in Scripts

```python
# Not recommended - use the CLI instead
# This is a CLI tool, not a library
```

## License

See project root for license information.
