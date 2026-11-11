# llmx — Claude Code Instructions

llmx is the transport layer every project calls for LLM providers. `README.md` says which layer owns what. It is editable-installed, so a source change reaches every caller at once.

This codebase is yours: how it's built is your call, and you'll keep building on it. What it's for stays with Markus.

- `just test`: quick check that GPT and Gemini work. `just install`: reinstall globally.
- Routing rules for callers live in `~/.claude/rules/llmx-routing.md`; update them in the same change when a flag, exit code or model id moves.
