"""CLI commands for Gemini Batch API.

Usage:
    llmx batch submit prompts.jsonl -m gemini-3-flash-preview
    llmx batch submit prompts.jsonl --wait -o results.jsonl
    llmx batch status batches/abc123
    llmx batch get batches/abc123 -o results.jsonl
    llmx batch list
    llmx batch cancel batches/abc123
"""

import sys
import json

import click

from .logger import configure_logger, logger


@click.group("batch")
def batch_group():
    """Gemini Batch API — 50% cost discount on async processing."""
    pass


@batch_group.command("submit")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-m", "--model", default="gemini-3-flash-preview", help="Model (default: gemini-3-flash-preview)")
@click.option("--wait", is_flag=True, help="Wait for completion and print results")
@click.option("-o", "--output", help="Output file for results (JSONL)")
@click.option("--name", "display_name", help="Human-readable job name")
@click.option("--poll-interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
@click.option("--debug", is_flag=True, help="Debug logging")
def submit_cmd(input_file, model, wait, output, display_name, poll_interval, debug):
    """Submit a batch job from a JSONL file.

    Input JSONL format (one per line):
        {"key": "req-1", "prompt": "...", "system": "...", "model": "..."}

    Only "prompt" is required. "key" defaults to line index.

    Examples:
        llmx batch submit prompts.jsonl
        llmx batch submit prompts.jsonl -m gemini-3.1-pro-preview --wait
        llmx batch submit prompts.jsonl --wait -o results.jsonl
    """
    configure_logger(debug=debug)

    from . import gemini_batch as gb

    requests = gb.parse_input_jsonl(input_file)
    if not requests:
        click.echo("Error: No requests found in input file.", err=True)
        sys.exit(1)

    click.echo(f"Submitting {len(requests)} requests with model {model}...", err=True)
    job_name = gb.submit(requests, model=model, display_name=display_name)
    click.echo(f"Submitted: {job_name}", err=True)

    if not wait:
        # Print job name to stdout for scripting
        click.echo(job_name)
        return

    # Wait for completion
    def _progress(state):
        click.echo(f"  State: {state}", err=True)

    click.echo("Waiting for completion...", err=True)
    final_state = gb.wait_for_completion(
        job_name, poll_interval=poll_interval, progress_callback=_progress
    )

    if final_state != "JOB_STATE_SUCCEEDED":
        click.echo(f"Job ended with state: {final_state}", err=True)
        sys.exit(1)

    # Fetch results
    keys = [r.key for r in requests]
    results = gb.fetch(job_name, original_keys=keys)
    _output_results(results, output)


@batch_group.command("status")
@click.argument("job_name")
@click.option("--debug", is_flag=True, help="Debug logging")
def status_cmd(job_name, debug):
    """Check batch job status.

    Example:
        llmx batch status batches/abc123
    """
    configure_logger(debug=debug)

    from . import gemini_batch as gb

    info = gb.status(job_name)
    click.echo(json.dumps(info, indent=2))


@batch_group.command("get")
@click.argument("job_name")
@click.option("-o", "--output", help="Output file (JSONL). Default: stdout")
@click.option("--debug", is_flag=True, help="Debug logging")
def get_cmd(job_name, output, debug):
    """Fetch results from a completed batch job.

    Example:
        llmx batch get batches/abc123
        llmx batch get batches/abc123 -o results.jsonl
    """
    configure_logger(debug=debug)

    from . import gemini_batch as gb

    results = gb.fetch(job_name)
    _output_results(results, output)


@batch_group.command("list")
@click.option("-n", "--limit", type=int, default=20, help="Max jobs to show (default: 20)")
@click.option("--debug", is_flag=True, help="Debug logging")
def list_cmd(limit, debug):
    """List recent batch jobs.

    Example:
        llmx batch list
        llmx batch list -n 5
    """
    configure_logger(debug=debug)

    from . import gemini_batch as gb

    jobs = gb.list_jobs(limit=limit)
    if not jobs:
        click.echo("No batch jobs found.", err=True)
        return

    for job in jobs:
        line = f"{job['name']}  {job['state']}"
        if "create_time" in job:
            line += f"  {job['create_time']}"
        click.echo(line)


@batch_group.command("cancel")
@click.argument("job_name")
@click.option("--debug", is_flag=True, help="Debug logging")
def cancel_cmd(job_name, debug):
    """Cancel a batch job.

    Example:
        llmx batch cancel batches/abc123
    """
    configure_logger(debug=debug)

    from . import gemini_batch as gb

    gb.cancel(job_name)
    click.echo(f"Cancelled: {job_name}", err=True)


def _output_results(results, output_path=None):
    """Write batch results as JSONL to file or stdout."""
    lines = []
    for r in results:
        obj = {"key": r.key}
        if r.content is not None:
            obj["content"] = r.content
        if r.error is not None:
            obj["error"] = r.error
        lines.append(json.dumps(obj, ensure_ascii=False))

    text = "\n".join(lines) + "\n" if lines else ""

    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
        click.echo(f"Results written to {output_path} ({len(results)} items)", err=True)
    else:
        click.echo(text, nl=False)
