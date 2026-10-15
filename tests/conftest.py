import pytest


@pytest.fixture(autouse=True)
def _refused_stream_dir(tmp_path, monkeypatch):
    """Keep refused-reply diagnostics written by tests out of the real ~/.cache/llmx."""
    monkeypatch.setattr("llmx.cli_backends.CLAUDE_REFUSED_STREAM_DIR", tmp_path / "refused-claude-streams")
