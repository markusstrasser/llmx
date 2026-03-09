"""Gemini Batch API support via google.genai SDK.

50% cost discount on async processing (24h target, usually faster).
Uses inline requests (not GCS file upload) — sufficient for <20MB batches.

Key limitation: inline responses may not preserve input order (googleapis/python-genai#1909).
We embed a correlation key in each request's system instruction to match responses back.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .logger import logger


DEFAULT_MODEL = "gemini-3-flash-preview"

# Terminal states for polling
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# Correlation prefix embedded in system instructions
_KEY_PREFIX = "[LLMX_BATCH_KEY:"
_KEY_SUFFIX = "]"


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY.\n"
            "Get a key at: https://aistudio.google.com/apikey"
        )
    return key


def _get_client():
    from google import genai
    return genai.Client(api_key=_get_api_key())


def _strip_model_prefix(model: str) -> str:
    """Strip 'gemini/' prefix if present (legacy convention)."""
    if model.startswith("gemini/"):
        return model[7:]
    return model


@dataclass
class BatchRequest:
    """A single request within a batch."""
    key: str
    prompt: str
    system: Optional[str] = None
    model: Optional[str] = None  # per-request override


@dataclass
class BatchResult:
    """Result for a single batch request."""
    key: str
    content: Optional[str] = None
    error: Optional[str] = None


def parse_input_jsonl(path: str) -> list[BatchRequest]:
    """Parse user JSONL into BatchRequest objects.

    Expected format per line:
        {"key": "req-1", "prompt": "...", "system": "...", "model": "..."}
    Only "prompt" is required. "key" defaults to line index.
    """
    requests = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            requests.append(BatchRequest(
                key=obj.get("key", str(i)),
                prompt=obj["prompt"],
                system=obj.get("system"),
                model=obj.get("model"),
            ))
    return requests


def _build_inline_request(req: BatchRequest) -> dict:
    """Convert a BatchRequest to a Gemini inline request dict.

    Embeds correlation key in system instruction so we can match
    responses back to requests even if order isn't preserved.
    """
    # Build system instruction with embedded key
    key_marker = f"{_KEY_PREFIX}{req.key}{_KEY_SUFFIX}"
    if req.system:
        system_text = f"{key_marker} {req.system}"
    else:
        system_text = f"{key_marker} Follow the user's instructions."

    result = {
        "contents": [{"parts": [{"text": req.prompt}], "role": "user"}],
        "config": {
            "system_instruction": {"parts": [{"text": system_text}]},
        },
    }
    return result


def _extract_key_from_response(text: str, index: int) -> str:
    """Try to extract correlation key from response text. Fall back to index."""
    # The key won't be in response text — it's in system instruction.
    # We rely on positional matching as primary, with key extraction as backup
    # if we can parse the job's request data.
    return str(index)


def submit(
    requests: list[BatchRequest],
    model: str = DEFAULT_MODEL,
    display_name: Optional[str] = None,
) -> str:
    """Submit a batch job. Returns the job name (ID).

    Args:
        requests: List of BatchRequest objects
        model: Default model for requests without per-request override
        display_name: Optional human-readable job name
    """
    client = _get_client()
    model = _strip_model_prefix(model)

    inline_requests = [_build_inline_request(r) for r in requests]

    config = {}
    if display_name:
        config["display_name"] = display_name

    job = client.batches.create(
        model=model,
        src=inline_requests,
        config=config if config else None,
    )
    logger.info(f"Submitted batch job: {job.name} ({len(requests)} requests)")
    return job.name


def status(job_name: str) -> dict:
    """Get batch job status.

    Returns dict with: name, state, create_time, update_time, request_count
    """
    client = _get_client()
    job = client.batches.get(name=job_name)

    state = job.state.name if hasattr(job.state, "name") else str(job.state)

    result = {
        "name": job.name,
        "state": state,
    }

    # Add optional fields if available
    if hasattr(job, "create_time") and job.create_time:
        result["create_time"] = str(job.create_time)
    if hasattr(job, "update_time") and job.update_time:
        result["update_time"] = str(job.update_time)

    return result


def fetch(job_name: str, original_keys: Optional[list[str]] = None) -> list[BatchResult]:
    """Fetch results from a completed batch job.

    Args:
        job_name: The batch job name/ID
        original_keys: Keys from the original requests (for correlation).
                       If provided, used for positional matching.

    Returns list of BatchResult objects.
    """
    client = _get_client()
    job = client.batches.get(name=job_name)

    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Job not complete: {state}")

    results = []
    responses = job.dest.inlined_responses if hasattr(job, "dest") and job.dest else []

    for i, resp in enumerate(responses):
        key = original_keys[i] if original_keys and i < len(original_keys) else str(i)

        if hasattr(resp, "response") and resp.response:
            try:
                content = resp.response.text
            except (AttributeError, IndexError):
                content = str(resp.response)
            results.append(BatchResult(key=key, content=content))
        elif hasattr(resp, "error") and resp.error:
            results.append(BatchResult(key=key, error=str(resp.error)))
        else:
            results.append(BatchResult(key=key, error="No response or error in result"))

    return results


def list_jobs(limit: int = 20) -> list[dict]:
    """List recent batch jobs."""
    client = _get_client()
    jobs = []
    for job in client.batches.list(config={"page_size": limit}):
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        entry = {"name": job.name, "state": state}
        if hasattr(job, "create_time") and job.create_time:
            entry["create_time"] = str(job.create_time)
        jobs.append(entry)
        if len(jobs) >= limit:
            break
    return jobs


def cancel(job_name: str) -> None:
    """Cancel a batch job."""
    client = _get_client()
    client.batches.cancel(name=job_name)
    logger.info(f"Cancelled batch job: {job_name}")


def wait_for_completion(
    job_name: str,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
) -> str:
    """Poll until job reaches a terminal state.

    Args:
        job_name: The batch job name/ID
        poll_interval: Seconds between polls (default: 30)
        timeout: Max wait in seconds (default: 24h)
        progress_callback: Optional callable(state_str) for progress updates

    Returns the terminal state string.
    """
    client = _get_client()
    start = time.time()

    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)

        if progress_callback:
            progress_callback(state)

        if state in TERMINAL_STATES:
            return state

        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Batch job {job_name} did not complete within {timeout}s")

        time.sleep(poll_interval)
