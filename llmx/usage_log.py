"""Per-call usage telemetry. Appends one JSONL line to ~/.claude/llmx-usage.jsonl.

Token counts come from provider responses (OpenAI .usage, Gemini .usage_metadata).
CLI transports don't expose usage — those lines have null token fields but still
record model, effort, latency, and transport for call-volume accounting.

Cost estimation is intentionally NOT computed here. Pricing changes; the raw
tokens are durable. See scripts/usage_summary.py for ad-hoc cost rollups.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_LOG_PATH = Path(os.environ.get("LLMX_USAGE_LOG", str(Path.home() / ".claude" / "llmx-usage.jsonl")))


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
        }
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"[llmx:usage_log] failed: {exc}", file=sys.stderr)
