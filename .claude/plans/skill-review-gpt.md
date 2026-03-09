# llmx Skill Review — GPT

Here’s a concrete review, grouped by your 7 requested categories.

## 1. Internal inconsistencies between `llmx-guide` and the hook

### 1.1 Redirect guidance is broad; hook only catches one narrow form
- **Guide:** “Never use `> file` shell redirects” is global guidance (`llmx-guide:15`, `llmx-guide:144-156`).
- **Hook:** only warns on `llmx chat ... > file` (`pretool-llmx-guard.sh:16-19`).

**Problem:** the hook misses the forms the guide itself shows as broken:
- `llmx -m gpt-5.4 "query" > output.md` (`llmx-guide:151-152`)
- any non-`chat` subcommand (`image`, `vision`, `research`)
- plain `llmx ... > file`

### 1.2 Deprecated model-prefix guidance is broad; hook only checks one syntax
- **Guide:** no provider prefixes needed; old prefixed names are deprecated (`llmx-guide:16`, `llmx-guide:47`).
- **Hook:** only checks `llmx ... -m gemini/...` with a bare `-m` form (`pretool-llmx-guard.sh:36-39`).

**Misses:**
- `--model gemini/...`
- `-m "gemini/..."`
- `--model="gemini/..."`
- prefixed fallback models like `--fallback gemini/...`

### 1.3 The guide’s main timeout/stream footgun is not guarded at all
- **Guide:** reasoning models need `--timeout 600` or `--stream` (`llmx-guide:13`, `llmx-guide:109-127`).
- **Hook:** no check for GPT-5.x without `--stream`/`--timeout`.

This is a bigger real-world failure mode than the hook’s current `--max-tokens` heuristic.

### 1.4 The guide highlights transport-switch triggers; the hook ignores them
- **Guide:** `-s`, `--search`, `--stream`, `--max-tokens`, `--schema` can force API fallback depending on provider (`llmx-guide:17`, `llmx-guide:232-245`).
- **Hook:** no warning for any of those.

If the hook is meant to catch “common llmx dispatch mistakes” (`pretool-llmx-guard.sh:2`), this is a major omission.

---

## 2. Missing edge cases in the hook

### 2.1 Redirect patterns it should catch but doesn’t
Current redirect regex is `llmx\s+chat.*>\s*["\$/"a-zA-Z]` (`pretool-llmx-guard.sh:17`).

It misses:
- `llmx -m gpt-5.4 ... > out.md`
- `llmx research ... > out.md`
- `llmx vision ... > out.txt`
- `llmx chat ... >> out.md`
- `llmx chat ... 1> out.md`
- `llmx chat ... > ./out.md`
- `llmx chat ... > ../out.md`
- `llmx chat ... > ~/out.md`
- filenames beginning with a digit

### 2.2 Known bad model names from the guide are not checked
The guide explicitly lists common traps:
- `claude-sonnet-4.6` vs `claude-sonnet-4-6` (`llmx-guide:12`, `llmx-guide:45`)
- `gemini-3-flash` missing `-preview`
- `gemini-flash-3` wrong order
- `gpt-5.3` missing `-chat-latest` (`llmx-guide:49-51`)

The hook doesn’t warn on any of these.

### 2.3 Missing check for `-s` on CLI-first provider usage
The guide and model-review both say `-s` triggers API fallback for `-p google` / `-p openai` (`llmx-guide:234-245`, `model-review:38-46`).

The hook should warn on combinations like:
- `llmx -p google -s "..."`
- `llmx -p openai -s "..."`

### 2.4 Missing check for `--stream` / `--max-tokens` forcing API transport
Per guide:
- Gemini CLI falls back to API for `--max-tokens` and `--stream` (`llmx-guide:234`, `llmx-guide:238`)
- Codex CLI falls back to API for `--stream` (`llmx-guide:235`)

The hook should warn when the user appears to want CLI transport but has added flags that force API.

### 2.5 Missing check for ignored `--reasoning-effort` on CLI transports
Guide says both CLIs ignore explicit `--reasoning-effort` (`llmx-guide:236`, plus Codex detail at `llmx-guide:264-285`).

The hook should warn on:
- `llmx -p openai --reasoning-effort xhigh ...`
- `llmx -p google --reasoning-effort low ...`

unless some other flag is already forcing API.

### 2.6 Missing check for invalid reasoning-effort per model
Guide defines per-model valid values (`llmx-guide:176-185`).

The hook should catch at least:
- `gpt-5.3-chat-latest --reasoning-effort high` (`llmx-guide:178`)
- `gpt-5-codex --reasoning-effort minimal` (`llmx-guide:181`)
- `kimi-k2.5 --reasoning-effort ...` (`llmx-guide:184`)

### 2.7 Missing check for `shell=True` in inline Python run via Bash tool
Guide calls this out explicitly (`llmx-guide:14`, `llmx-guide:164-172`).

Since this is a **Bash** pretool hook, it can still catch common inline Python cases like:
- `python -c '... shell=True ... llmx ...'`
- heredoc Python with `shell=True`

It currently doesn’t.

---

## 3. Wrong or outdated parameter names, model names, flag names

### 3.1 `GPT-5.2` context size contradicts itself
- `llmx-guide:41` says: **“GPT-5.2 (legacy) ... 400K context.”**
- `llmx-guide:58` says: **Max Input = 272,000**

One of these is stale/wrong.

### 3.2 Duplicate/confusing Gemini rows in the model table
- `llmx-guide:36-37` repeats **Gemini 3 Flash / `gemini-3-flash-preview`** twice with different notes.

This looks like an accidental duplicate, not two distinct models.

### 3.3 Duplicate/confusing reasoning-effort rows
- `llmx-guide:182` has **Gemini 3 Flash**
- `llmx-guide:183` has **Gemini 3.x (Pro/Flash)**

These overlap. As written, Flash is listed twice.

### 3.4 `model-review` still references a temperature knob that the guide says is ineffective
- `model-review:73` says: **“Lower temperature for Gemini (`-t 0.3`)”**
- `llmx-guide:186` says: **Temperature locked to 1.0 for GPT-5 and Gemini 3.x thinking models**

So `-t 0.3` is stale/no-op guidance for Gemini 3.x thinking models.

---

## 4. Stale information or contradictions between files

### 4.1 `model-review` says “CLI-first,” but its own templates force API transport
- CLI-first guidance: `model-review:38-46`
- Gemini template uses `--max-tokens 65536`: `model-review:178`, `model-review:203-206`
- Guide says `--max-tokens` forces API for Gemini CLI: `llmx-guide:234`, `llmx-guide:238`
- GPT template uses `--stream`: `model-review:183`, `model-review:237-240`
- Guide says `--stream` forces API for Codex CLI: `llmx-guide:235`

So the inline `<system>...</system>` trick in `model-review` is not actually preserving CLI transport for the recommended review commands.

### 4.2 “Never downgrade models on failure” contradicts the configured Gemini fallback
- Guide: “Never swap to a weaker model as a fix” (`llmx-guide:19-29`)
- `model-review`: same message (`model-review:193`)
- But `model-review` also sets `--fallback gemini-3-flash-preview` (`model-review:179`)
- Guide says fallback triggers on **rate limit or timeout** (`llmx-guide:86-92`)
- Gemini command uses `--timeout 300` (`model-review:205`)

So the deep-review skill can silently turn a Pro review into a Flash review on timeout. That directly conflicts with the “never downgrade” message.

### 4.3 `model-review` understates fallback behavior
- `model-review:174` says: **“Gemini — Pro with Flash fallback for rate limits”**
- But guide says `--fallback` also triggers on **timeout** (`llmx-guide:86`)

That comment is inaccurate.

### 4.4 Timeout guidance is inconsistent
- Guide checklist says reasoning models need `--timeout 600` or `--stream` (`llmx-guide:13`)
- Guide fallback example uses `--timeout 300` for Gemini Pro (`llmx-guide:89`)
- `model-review` Gemini uses `--timeout 300` (`model-review:205`)
- optional Flash audit uses `--timeout 120` (`model-review:274-277`)

Either `llmx-guide:13` is too broad, or the templates are under-timeout’d relative to the guide.

### 4.5 `model-review` has contradictory context-size advice
- Token budget table says:
  - Gemini sweet spot **80K-150K** (`model-review:165`)
  - GPT sweet spot **40K-100K** (`model-review:166`)
- But `model-review:195` says **50K context** makes API calls take 5-10 min and get killed, recommending 2K summaries.

That is internally contradictory unless those numbers are in different units, which the text does not say.

---

## 5. Incomplete guidance / important gotchas not covered

### 5.1 `model-review` does not preserve stderr, so transport/fallback diagnostics are lost
- Guide says diagnostics, transport switches, truncation warnings, and fallback notices are on stderr (`llmx-guide:25-27`, `llmx-guide:81-92`)
- `model-review` only saves stdout via `-o` (`model-review:203-206`, `model-review:237-240`)

That means review artifacts won’t show:
- whether Gemini fell back to Flash
- whether transport switched to API
- truncation warnings

For this skill, stderr should be saved per call, e.g. separate `*.stderr.log`.

### 5.2 Constitution extraction from `CLAUDE.md` is wrong
- `model-review:56-60` detects an inline `## Constitution` section in `CLAUDE.md`
- But `model-review:128-132` then does `cat "$CONSTITUTION"` into the context

If `CONSTITUTION="$CLAUDE_MD"`, this injects the **entire** `CLAUDE.md`, not just the Constitution section.

### 5.3 GOALS injection logic is broken in two cases
The prose says:
- inject GOALS if it exists (`model-review:66`)
- warn only if neither constitution nor goals exists (`model-review:67`)

But the code only appends GOALS inside the `elif [ -n "$CONSTITUTION" ]` branch (`model-review:127-132`).

So GOALS is **not injected** when:
1. `.context/constitution.xml` exists (`model-review:124-126`)
2. `GOALS.md` exists but no constitution exists at all

That contradicts the prose.

### 5.4 `$TOPIC` is used before the skill ever shows how to set it
- Slug code uses `$TOPIC` (`model-review:100`)
- Explanation of what `$TOPIC` means comes later (`model-review:106`)

There’s no explicit assignment snippet. As written, this is incomplete.

### 5.5 Timeout validation guidance is ambiguous for Google
- Guide says CLI validates `--timeout` in range **1-900** (`llmx-guide:113`)
- But Google timeout minimum is **10s** (`llmx-guide:119`)

It never says whether `--timeout 1..9` on Google errors, clamps, or behaves differently.

---

## 6. Regex bugs or logic errors in the hook script

### 6.1 Redirect regex misses valid bad cases
`pretool-llmx-guard.sh:17`
```bash
grep -qE 'llmx\s+chat.*>\s*["\$/"a-zA-Z]'
```

Problems:
- only matches `llmx chat`
- misses `>>`
- misses `1>`
- misses paths starting with `.`, `~`, or digits
- misses plain `llmx -m ... > file`

### 6.2 Redirect regex can false-positive on `>` inside a quoted prompt
Because it just looks for `.*>` after `llmx chat`, this can warn on something like:
```bash
llmx chat "compare a > b in this expression"
```
It is not distinguishing shell syntax from quoted prompt content.

### 6.3 `--max-tokens` regex misses 5-digit values that are still too small
`pretool-llmx-guard.sh:32`
```bash
--max-tokens\s+[0-9]{1,4}[^0-9]
```

But the warning says use **16384+** (`pretool-llmx-guard.sh:33`).

So values like:
- `10000`
- `12000`
- `16383`

should warn, but won’t, because they are 5 digits.

### 6.4 `--max-tokens` regex misses common syntax forms
Same line (`pretool-llmx-guard.sh:32`) misses:
- `--max-tokens=4096`
- `--max-tokens "4096"`
- end-of-line `--max-tokens 4096` with no trailing char
- env-var values

### 6.5 The GPT-model match is not tied to the actual model flag
`pretool-llmx-guard.sh:32` checks for `gpt-5\.[234]` anywhere in the command string.

That can false-trigger if `gpt-5.4` appears in prompt text, filenames, or comments.

### 6.6 Deprecated-prefix regex is too narrow
`pretool-llmx-guard.sh:37`
```bash
llmx.*-m\s+(gemini/|openai/|xai/|moonshot/)
```

Misses:
- `--model`
- quoted values
- `-m=...`
- `--model=...`
- prefixed fallback models

### 6.7 The comment on redirect detection is misleading
- Comment says: “not context file building with cat/echo” (`pretool-llmx-guard.sh:16`)
- Regex does not actually implement that distinction.

### 6.8 Shell robustness: using `echo` for arbitrary JSON/commands is brittle
- `pretool-llmx-guard.sh:8`, and repeated throughout

`echo` is less safe than `printf '%s\n'` for arbitrary content. Not catastrophic here, but this hook is parsing structured JSON and shell commands; `printf` would be safer.

---

## 7. Inconsistencies between `llmx-guide` and `model-review` dispatch templates

### 7.1 Gemini temperature guidance is both stale and not implemented
- `model-review:73` says use `-t 0.3`
- `llmx-guide:186` says temp is locked to 1.0
- actual Gemini command doesn’t include `-t 0.3` anyway (`model-review:203-206`)

This is a direct doc/template mismatch.

### 7.2 Gemini template forces API transport, so CLI-first prompt advice is moot
- CLI-first guidance: `model-review:38-46`
- Gemini template uses `$GEMINI_MAX_TOKENS` (`model-review:178`, used at `model-review:205`)
- Guide says `--max-tokens` forces API for Gemini CLI (`llmx-guide:234`, `llmx-guide:238`)

If the intended behavior is API transport, say that explicitly.

### 7.3 GPT template also forces API transport
- GPT template uses `--stream` (`model-review:183`, `model-review:239`)
- Guide says Codex CLI falls back to API for `--stream` (`llmx-guide:235`)

So the same issue applies to OpenAI.

### 7.4 Gemini template is configured in a way that can silently replace the primary reviewer
- fallback to Flash: `model-review:179`
- timeout 300: `model-review:205`
- guide says fallback retries on timeout too (`llmx-guide:86-92`)

For a “deep review” skill, that means long-running Pro reviews can become Flash reviews without the output file showing it, since stderr isn’t captured.

### 7.5 Optional Flash audit timeout also conflicts with the guide’s generic timeout warning
- Guide: reasoning models need `--timeout 600` or `--stream` (`llmx-guide:13`)
- Optional Flash audit uses `--timeout 120` on a “large codebase” context (`model-review:271-289`)

At minimum, the guide’s line 13 is too absolute; at worst, the Flash template is under-timeout’d.

---

## Highest-impact fixes to make first

1. **Fix `model-review` transport/fallback semantics**
   - Update `model-review:38-46`, `174-179`, `193`, `203-206`, `237-240`
   - Be explicit that the recommended templates are **API transport**
   - Remove or rethink Flash fallback for the primary Gemini reviewer

2. **Fix `model-review` context injection logic**
   - `model-review:56-60`, `124-132`
   - extract only the Constitution section from `CLAUDE.md`
   - inject `GOALS.md` independently of constitution source

3. **Rewrite the hook checks around actual argv patterns, not narrow regexes**
   - `pretool-llmx-guard.sh:17`, `32`, `37`
   - especially add checks for:
     - generic `> / >> / 1>` redirect use
     - missing `--timeout`/`--stream` on GPT-5.x
     - `-s` / `--stream` / `--max-tokens` transport switches
     - known bad model names
     - invalid reasoning-effort values

If you want, I can turn this into a patch-style diff for all three files.
