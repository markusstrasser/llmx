# llmx Skill Review — GEMINI

Here is a concrete review of the three files, highlighting inconsistencies, regex bugs, and contradictory guidance.

### 1. Regex Bugs & Missing Edge Cases in the Guard Hook (`pretool-llmx-guard.sh`)
*   **Misses bare `llmx` calls:** The shell redirect regex on line 17 (`llmx\s+chat.*>`) explicitly requires the `chat` subcommand. However, `llmx-guide/SKILL.md` lines 148, 151, and 152 demonstrate using llmx without `chat` (e.g., `llmx -m gpt-5.4 "query" > output.md`). The guard will fail to warn on these broken redirects.
*   **Misses common file characters:** The redirect target regex on line 17 (`">\s*["\$/"a-zA-Z]"`) explicitly excludes dots, numbers, and underscores. Redirecting to `> .output.md`, `> 1.txt`, or `> _out` will bypass the guard entirely.
*   **End-of-line bug in `max_tokens` check:** Line 32 uses `[^0-9]` to ensure the token count isn't longer than 4 digits. If `--max-tokens 4096` is the very last string in the command (no trailing space), the regex fails to match because `[^0-9]` requires a character to be present. It should use `\b` or `($|\s)`.
*   **Misses alternative flag syntax:** The LiteLLM prefix check on line 38 (`-m\s+(gemini/...)`) expects a space. It will fail to catch `-m"gemini/..."` or `--model=gemini/...`.

### 2. Critical Cross-File Contradictions
*   **The CLI-Transport API Fallback Trap:** 
    *   `model-review/SKILL.md` lines 38-47 aggressively pitches a "CLI-First Prompting Rule" (inlining `<system>` tags instead of `-s`) to avoid API fallback and preserve CLI transport.
    *   *However*, the actual dispatch templates completely defeat this. The Gemini dispatch (line 205) passes `$GEMINI_MAX_TOKENS` (`--max-tokens 65536`), which `llmx-guide/SKILL.md` line 238 explicitly states **forces API fallback**. The GPT dispatch (line 239) passes `$GPT_EFFORT` (`--stream`), which `llmx-guide` line 235 states **forces API fallback**. The inline `<system>` advice is rendered useless by the template's own flags.
*   **The Bash Timeout vs. llmx Timeout Collision:**
    *   `model-review/SKILL.md` line 189 and `llmx-guide/SKILL.md` line 161 instruct setting the Claude Bash tool execution timeout to `360000` (6 minutes).
    *   *However*, `model-review` line 184 configures `$GPT_TIMEOUT` as `--timeout 600` (10 minutes), aligning with `llmx-guide` line 113 ("For review dispatches use `--timeout 600`"). A 6-minute Bash boundary will aggressively kill the process 4 minutes before `llmx`'s own internal timeout has a chance to trigger.
*   **Gemini Timeout Discrepancy:** `llmx-guide/SKILL.md` line 13 and 113 insist that reasoning/review dispatches need `--timeout 600`. Yet, the Gemini dispatch template in `model-review/SKILL.md` line 205 hardcodes `--timeout 300` despite processing up to 800K context (line 165).

### 3. Incomplete Guidance & Unhandled Edge Cases
*   **Flash Fallback Max Tokens Crash:** `model-review/SKILL.md` lines 178-179 setup Gemini 3.1 Pro with `--max-tokens 65536` and `--fallback gemini-3-flash-preview`. However, `llmx-guide/SKILL.md` line 62 explicitly states Gemini 3 Flash has a strict Max Output of `65,535`. If the Pro model fails, the fallback dispatch will likely instantly crash with Exit Code 5 (Model error) because it attempts to pass `65536` to Flash.
*   **Codex CLI Reasoning No-Op:** `llmx-guide/SKILL.md` line 236 mentions "CLIs ignore explicit `--reasoning-effort`". The guide fails to connect the dots for the user: if you successfully maintain Codex CLI transport, passing `--reasoning-effort high` is a silent no-op. You simply get whatever is in `~/.codex/config.toml` (line 268), which might be `low` or `none`.

### 4. Stale Information, Typos & Discrepancies
*   **Duplicate Table Row:** `llmx-guide/SKILL.md` lines 36 and 37 contain duplicate entries for "Gemini 3 Flash".
*   **Missing Model:** `llmx-guide/SKILL.md` line 60 lists `o4-mini` in the Token Limits table, but the model is missing from the foundational "Model Names & Defaults" table (lines 33-46).
*   **Mismatched Review Directory Example:** `model-review/SKILL.md` line 108 gives `.model-review/2026-03-01-hook-architecture/` as an example directory name, but this is missing the random hex `${REVIEW_ID}` suffix that the creation script on line 102 guarantees will be present.

