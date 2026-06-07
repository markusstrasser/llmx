"""Per-call usage telemetry. Appends one JSONL line to ~/.claude/llmx-usage.jsonl.

Token counts come from provider responses (OpenAI .usage, Gemini .usage_metadata).
CLI transports don't expose usage — those lines have null token fields but still
record model, effort, latency, and transport for call-volume accounting.

Cost estimation is intentionally NOT computed here. Pricing changes; the raw
tokens are durable. See scripts/usage_summary.py for ad-hoc cost rollups.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_LOG_PATH = Path(os.environ.get("LLMX_USAGE_LOG", str(Path.home() / ".claude" / "llmx-usage.jsonl")))

# Tokens that are wrappers/interpreters, not the actual caller, when walking the
# process ancestry for the invoking script. "snapshot-" is Claude Code's bash-tool
# shell snapshot (snapshot-zsh-*.sh / snapshot-bash-*.sh) — present in the ancestry
# of any agent-initiated call, never the real caller.
_WRAPPER_HINTS = ("/llmx", "site-packages", "/uv/", "/bin/uv", "pytest", "snapshot-", "/.claude/")
_SCRIPT_RE = re.compile(r"(\S+\.(?:py|sh))")


def _resolve_caller() -> Optional[str]:
    """Best-effort attribution of WHO invoked llmx, so the usage log can be
    grouped by spender. Precedence:
      1. LLMX_CALLER env var — explicit, set by callers that want precision.
      2. The nearest real script in the process ancestry (skips shells/uv/llmx).
      3. The entry script (sys.argv[0]) when llmx is imported as a library.
    Returns a compact basename, or None. Never raises.
    """
    explicit = os.environ.get("LLMX_CALLER")
    if explicit:
        return explicit[:200]
    try:
        pid = os.getppid()
        for _ in range(6):  # walk up to 6 ancestors
            if pid <= 1:
                break
            out = subprocess.run(
                ["ps", "-o", "ppid=,command=", "-p", str(pid)],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if not out:
                break
            ppid_str, _, cmd = out.partition(" ")
            for tok in _SCRIPT_RE.findall(cmd):
                if not any(h in tok for h in _WRAPPER_HINTS):
                    return os.path.basename(tok)
            try:
                pid = int(ppid_str.strip())
            except ValueError:
                break
    except Exception:
        pass
    try:
        base = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        if base and base not in ("llmx", "__main__.py", "-c", "pytest"):
            return base
    except Exception:
        pass
    return None


def log_usage(
    *,
    provider: str,
    model: str,
    transport: str,
    reasoning_effort: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    reasoning_tokens: Optional[int],
    cached_tokens: Optional[int],
    latency_s: float,
    error: Optional[str] = None,
) -> None:
    """Append one usage record. Best-effort — never raises."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "provider": provider,
            "model": model,
            "transport": transport,
            "reasoning_effort": reasoning_effort,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "latency_s": round(latency_s, 3),
            "error": error,
            "caller": _resolve_caller(),  # WHO invoked llmx (attribution for cost rollups)
            "cwd": os.getcwd(),           # WHICH project (group by basename)
        }
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"[llmx:usage_log] failed: {exc}", file=sys.stderr)
