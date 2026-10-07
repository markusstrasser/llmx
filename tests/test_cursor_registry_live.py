"""Hermetic parser coverage plus an opt-in live Cursor registry contract."""

import os
import shutil
import subprocess

import pytest

from llmx.model_ids import CURSOR_GROK47_MODELS


def parse_cursor_model_ids(output: str) -> set[str]:
    """Parse the stable `<id> - <label>` rows from `cursor-agent models`."""
    return {
        line.split(" - ", 1)[0].strip()
        for line in output.splitlines()
        if " - " in line and line.split(" - ", 1)[0].strip()
    }


def test_cursor_registry_parser_is_hermetic() -> None:
    output = """Available models

cursor-grok-4.6-high - Cursor Grok 4.6
cursor-grok-4.6-high-fast - Cursor Grok 4.6 Fast
grok-4.7-low - Grok 4.7 Low
"""
    assert parse_cursor_model_ids(output) == {
        "cursor-grok-4.6-high",
        "cursor-grok-4.6-high-fast",
        "grok-4.7-low",
    }


@pytest.mark.skipif(
    os.environ.get("LLMX_LIVE_CURSOR_REGISTRY") != "1",
    reason="set LLMX_LIVE_CURSOR_REGISTRY=1 for the live cursor-agent registry contract",
)
def test_live_cursor_registry_contains_configured_grok47_slugs() -> None:
    binary = shutil.which("cursor-agent")
    assert binary, "cursor-agent is required for the live registry contract"
    completed = subprocess.run(
        [binary, "models"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    live_models = parse_cursor_model_ids(completed.stdout)
    missing = set(CURSOR_GROK47_MODELS) - live_models
    assert not missing, (
        f"configured Cursor Grok slugs absent from live registry: {sorted(missing)}"
    )
