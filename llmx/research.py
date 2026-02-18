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
