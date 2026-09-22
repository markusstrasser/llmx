"""CLI boundary for typed live subscription probes."""

from __future__ import annotations

import json

import click

from .probe import DEFAULT_CACHE_TTL_SECONDS, run_subscription_probe


@click.command("probe")
@click.option("--provider", default="anthropic", show_default=True)
@click.option("--model", default="claude-opus-5-5", show_default=True)
@click.option("--timeout", type=click.IntRange(1, 600), default=120, show_default=True)
@click.option(
    "--cache-ttl",
    "cache_ttl_seconds",
    type=click.IntRange(0, 86_400),
    default=DEFAULT_CACHE_TTL_SECONDS,
    show_default=True,
    help="Reuse an identical provider/model/route result for this many seconds.",
)
@click.option("--refresh", is_flag=True, help="Bypass an unexpired cached result.")
@click.option("--json", "as_json", is_flag=True, help="Emit the typed result as JSON.")
def probe_cmd(
    provider: str,
    model: str,
    timeout: int,
    cache_ttl_seconds: int,
    refresh: bool,
    as_json: bool,
) -> None:
    """Make one bounded live call through a subscription CLI route."""
    try:
        result = run_subscription_probe(
            provider=provider,
            model=model,
            timeout=timeout,
            cache_ttl_seconds=cache_ttl_seconds,
            refresh=refresh,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        click.echo(
            f"[llmx:PROBE] verdict={result.verdict} cached={str(result.cached).lower()} "
            f"auth={result.auth} transport={result.transport} model={result.model} "
            f"exit={result.exit_code}"
        )
        if result.detail:
            click.echo(result.detail, err=True)
    if result.exit_code:
        raise click.exceptions.Exit(result.exit_code)
