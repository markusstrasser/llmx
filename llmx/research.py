"""Deep research using OpenAI Responses API"""

import os
import sys
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown

from .logger import logger

console = Console()

MODELS = {
    "o3": "o3-deep-research",
    "o4-mini": "o4-mini-deep-research",
}


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as mm:ss"""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _extract_citations(output) -> list[dict]:
    """Extract citation annotations from response output."""
    citations = []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            for ann in getattr(content, "annotations", []) or []:
                url = getattr(ann, "url", None)
                title = getattr(ann, "title", None)
                if url:
                    citations.append({"url": url, "title": title or url})
    return citations


def _count_tool_calls(output) -> dict[str, int]:
    """Count tool calls by type from response output."""
    counts: dict[str, int] = {}
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type and item_type != "message":
            counts[item_type] = counts.get(item_type, 0) + 1
    return counts


def _extract_agent_text(d: dict) -> str:
    """Pull assistant text from a Perplexity Agent API response dict."""
    if d.get("output_text"):
        return d["output_text"]
    for o in reversed(d.get("output", []) or []):
        if o.get("type") == "message":
            try:
                return o["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                continue
    return ""


def _extract_agent_citations(d: dict) -> list[dict]:
    """Gather {url,title} from a Perplexity Agent API response (search_results blocks)."""
    cites: list[dict] = []
    for o in d.get("output", []) or []:
        for r in o.get("results", []) or []:
            url = r.get("url")
            if url:
                cites.append({"url": url, "title": r.get("title") or url})
    return cites


def research_perplexity_agent(
    prompt: str,
    preset: str = "deep-research",
    output_file: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Run deep research via the Perplexity Agent API (Search-as-Code architecture).

    SaC exposes Perplexity's search stack as code-orchestrated primitives — but only
    inside Perplexity's PRIVATE internal harness. Eval (evals/sac_bakeoff, 2026-06-10)
    found the public Agent API has NO measured advantage in ANY regime: it ties the
    cheap Exa /answer path on bounded structured pulls (~3x cost/latency) AND returns
    recall 0 on the exhaustive CVE flagship (4 configs all abstain) where a single
    $0.005 Exa call gets precision 1.0. This lane is PARITY-ONLY (availability, not a
    recommendation). For real work, DIY code fan-out->verify over Exa/Brave + an
    authoritative registry (NVD/FDA/SEC/eutils) beats it.
    `deep-research` preset ~$0.7/call & 1-5 min; `pro-search` ~$0.2 & ~25s.
    """
    import json as _json
    import urllib.error
    import urllib.request

    from .usage_log import log_usage

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PERPLEXITY_API_KEY not set. Export it to use Perplexity Agent research.\n"
            "Example: export PERPLEXITY_API_KEY=pplx-..."
        )

    payload = {"preset": preset, "input": prompt}
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.perplexity.ai/v1/agent",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    console.print(f"[dim]Perplexity Agent ({preset}) — synchronous; deep-research can take 1-5 min[/dim]")
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = _json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        raise RuntimeError(f"Perplexity Agent API failed: HTTP {e.code}: {body}") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Perplexity Agent API request failed: {e}") from e

    total_elapsed = time.time() - start_time
    if d.get("status") not in (None, "completed"):
        raise RuntimeError(f"Unexpected status: {d.get('status')} — {d.get('error')}")

    report_text = _extract_agent_text(d)
    citations = _extract_agent_citations(d)
    usage = d.get("usage", {}) or {}
    cost = (usage.get("cost", {}) or {}).get("total_cost")

    log_usage(
        provider="perplexity", model=d.get("model") or f"agent:{preset}", transport="agent-api",
        reasoning_effort=preset, prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        reasoning_tokens=(usage.get("output_tokens_details", {}) or {}).get("reasoning_tokens"),
        cached_tokens=(usage.get("input_tokens_details", {}) or {}).get("cached_tokens"),
        latency_s=total_elapsed,
    )

    elapsed_str = _format_elapsed(total_elapsed)
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    logger.info("Research complete",
                {"elapsed": elapsed_str, "cost": cost_str, "report_length": len(report_text)})

    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
            if citations:
                f.write("\n\n---\n\n## Sources\n\n")
                seen = set()
                for c in citations:
                    if c["url"] not in seen:
                        f.write(f"- [{c['title']}]({c['url']})\n")
                        seen.add(c["url"])
        console.print(f"\nReport saved to [bold]{output_file}[/bold] ({elapsed_str}, {cost_str})")
    else:
        console.print()
        console.print(Markdown(report_text))
        seen = set()
        unique = [c for c in citations if c["url"] not in seen and not seen.add(c["url"])]
        tail = f"\n[dim]{elapsed_str} | {cost_str}"
        if unique:
            tail += f" | {len(unique)} sources"
        console.print(tail + "[/dim]")


def research(
    prompt: str,
    model: str = "o3",
    max_tool_calls: Optional[int] = None,
    code_interpreter: bool = False,
    output_file: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Run deep research query via OpenAI Responses API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it to use deep research.\n"
            "Example: export OPENAI_API_KEY=sk-..."
        )

    model_id = MODELS.get(model, model)
    client = OpenAI(timeout=3600)

    # Build tools list
    tools: list[dict] = [{"type": "web_search_preview"}]
    if code_interpreter:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

    logger.debug(
        "Starting deep research",
        {"model": model_id, "tools": [t["type"] for t in tools], "max_tool_calls": max_tool_calls},
    )

    # Build request kwargs
    create_kwargs: dict = {
        "model": model_id,
        "input": prompt,
        "background": True,
        "tools": tools,
    }
    if max_tool_calls is not None:
        create_kwargs["max_tool_calls"] = max_tool_calls

    # Fire off the request
    start_time = time.time()

    try:
        resp = client.responses.create(**create_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to start deep research: {e}") from e

    logger.debug(f"Response ID: {resp.id}, status: {resp.status}")

    # Poll with spinner
    with Live(
        Spinner("dots", text=f"Researching... [0:00]"),
        console=console,
        transient=True,
    ) as live:
        while resp.status in {"queued", "in_progress"}:
            time.sleep(5)
            elapsed = _format_elapsed(time.time() - start_time)

            # Show tool call progress if available
            tool_counts = _count_tool_calls(resp.output) if resp.output else {}
            progress_parts = [f"Researching... [{elapsed}]"]
            if tool_counts:
                counts_str = ", ".join(f"{k}: {v}" for k, v in tool_counts.items())
                progress_parts.append(f"({counts_str})")
            live.update(Spinner("dots", text=" ".join(progress_parts)))

            try:
                resp = client.responses.retrieve(resp.id)
            except Exception as e:
                logger.warn(f"Poll failed (retrying): {e}")
                time.sleep(5)
                continue

    total_elapsed = time.time() - start_time

    # Check for failure
    if resp.status == "failed":
        error_detail = getattr(resp, "error", None)
        raise RuntimeError(f"Deep research failed: {error_detail or 'unknown error'}")

    if resp.status == "cancelled":
        raise RuntimeError("Deep research was cancelled")

    if resp.status != "completed":
        raise RuntimeError(f"Unexpected status: {resp.status}")

    # Extract results
    report_text = resp.output_text or ""
    citations = _extract_citations(resp.output)
    tool_counts = _count_tool_calls(resp.output)

    # Stats
    elapsed_str = _format_elapsed(total_elapsed)
    counts_str = ", ".join(f"{k}: {v}" for k, v in tool_counts.items()) if tool_counts else "none"

    logger.info(
        f"Research complete",
        {"elapsed": elapsed_str, "tool_calls": counts_str, "report_length": len(report_text)},
    )

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
            if citations:
                f.write("\n\n---\n\n## Sources\n\n")
                seen = set()
                for c in citations:
                    if c["url"] not in seen:
                        f.write(f"- [{c['title']}]({c['url']})\n")
                        seen.add(c["url"])
        console.print(f"\nReport saved to [bold]{output_file}[/bold] ({elapsed_str})")
    else:
        # Render to terminal
        console.print()
        console.print(Markdown(report_text))

        if citations:
            seen = set()
            unique = [c for c in citations if c["url"] not in seen and not seen.add(c["url"])]
            if unique:
                console.print(f"\n[dim]{elapsed_str} | {counts_str} | {len(unique)} sources[/dim]")
