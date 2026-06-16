"""llmx info — routing facts for agents (mirror of transport truth, not model-guide judgment)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from . import __version__
from .dispatch_plan import SCHEMA_VERSION, collect_routing_mirror


DEFAULT_MIRROR = Path.home() / ".claude" / "cache" / "llmx-routing.json"


@click.command("info")
@click.option("--json", "as_json", is_flag=True, help="JSON output (default when piping).")
@click.option(
    "--write-mirror",
    is_flag=True,
    default=False,
    help=f"Write routing mirror to {DEFAULT_MIRROR} for agent Read.",
)
@click.option(
    "--mirror-path",
    type=click.Path(),
    default=None,
    help="Override mirror output path (used with --write-mirror).",
)
def info_cmd(as_json: bool, write_mirror: bool, mirror_path: str | None):
    """Show installed CLIs, subscription routes, and effort aliases.

    Judgment-layer economics (Fable path, cosigner policy) stay in model-guide/SKILL.md.
    """
    payload = collect_routing_mirror()
    payload["llmx_version"] = __version__
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()

    if write_mirror:
        path = Path(mirror_path) if mirror_path else DEFAULT_MIRROR
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        click.echo(f"Wrote {path}", err=True)

    if as_json or not sys.stdout.isatty():
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"llmx info ({SCHEMA_VERSION})")
    click.echo(f"  version: {__version__}")
    for name, st in payload["cli_providers"].items():
        mark = "yes" if st["installed"] else "no"
        click.echo(f"  {name}: binary={st['binary']} installed={mark}")
    click.echo("  lite models: " + ", ".join(payload["lite_allowed_models"]))
    click.echo("  policy: see model-guide/SKILL.md (not duplicated here)")
