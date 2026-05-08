"""Gemini Batch API support via google.genai SDK.

50% cost discount on async processing (24h target, usually faster).
Uses inline requests (not GCS file upload) — sufficient for <20MB batches.

Inline responses don't preserve input order (googleapis/python-genai#1909).
Each InlinedRequest carries a `metadata: dict[str, str]` slot that's mirrored
onto the matching InlinedResponse — that's the supported correlation channel.
We stuff the caller's key under metadata['llmx_key'] on submit and read it
back on fetch. Falls back to positional matching only when metadata is absent
(older SDK / API versions), and warns loudly so the corruption surface is
visible.
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

# Metadata key used to round-trip the caller's request key.
_METADATA_KEY = "llmx_key"


def _get_api_key() -> str:
    """Resolve Gemini API key via env vars + macOS Keychain."""
    from .providers import check_api_key as _provider_check, _get_api_key as _provider_get
    _provider_check("google")  # raises RuntimeError with hint if missing
    key = _provider_get("google")
    if not key:
        raise RuntimeError("API key not found for google after Keychain fallback")
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

    Round-trips the caller's request key through the InlinedRequest.metadata
    slot, which the API mirrors onto the matching InlinedResponse so we can
    correlate even when responses come back out of order.
    """
    config: dict = {}
    if req.system:
        config["system_instruction"] = {"parts": [{"text": req.system}]}

    result: dict = {
        "contents": [{"parts": [{"text": req.prompt}], "role": "user"}],
        "metadata": {_METADATA_KEY: req.key},
    }
    if config:
        result["config"] = config
    if req.model:
        result["model"] = req.model
    return result


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

    Correlates each response back to its request via the
    InlinedResponse.metadata['llmx_key'] slot that submit() populates.
    Falls back to positional matching only if the metadata channel is
    missing (older API/SDK), and warns — that path can corrupt results
    when the API doesn't preserve order.

    Args:
        job_name: The batch job name/ID
        original_keys: Optional keys from the original requests, used only
                       to size the positional fallback. New callers should
                       not need this — keys round-trip via metadata.

    Returns list of BatchResult objects (caller's original key on each).
    """
    client = _get_client()
    job = client.batches.get(name=job_name)

    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Job not complete: {state}")

    results = []
    responses = job.dest.inlined_responses if hasattr(job, "dest") and job.dest else []
    fell_back_to_positional = False

    for i, resp in enumerate(responses):
        # Prefer metadata round-trip; fall back to positional with a loud warning.
        meta = getattr(resp, "metadata", None)
        key: Optional[str] = None
        if isinstance(meta, dict):
            key = meta.get(_METADATA_KEY)
        if key is None:
            fell_back_to_positional = True
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

    if fell_back_to_positional and responses:
        logger.warn(
            f"[batch:WARN] {job_name} responses missing metadata['{_METADATA_KEY}']; "
            "fell back to positional matching. If response order isn't preserved "
            "(googleapis/python-genai#1909), correlations may be wrong."
        )
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
