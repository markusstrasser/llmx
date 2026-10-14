"""llmx image --auth subscription: codex image_gen route (regression for -i eating the prompt)."""
import subprocess
from pathlib import Path

import pytest

from llmx import image


def test_codex_cmd_ends_multivalue_image_flag(tmp_path, monkeypatch):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    out = tmp_path / "out.png"
    seen = {}

    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "codex":  # usage logging may shell out (e.g. git)
            return real_run(cmd, **kw)
        seen["cmd"] = cmd
        out.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(__import__("shutil"), "which", lambda _: "/bin/codex")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("llmx.usage_log.log_usage", lambda **kw: None)
    assert image.generate_codex_image("a cell", str(out), [str(ref)]) == [out.resolve()]
    cmd = seen["cmd"]
    i = cmd.index("-i")
    assert cmd[i + 1] == str(ref.resolve())
    assert cmd[-2] == "--" and "a cell" in cmd[-1]


def test_codex_no_image_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("shutil"), "which", lambda _: "/bin/codex")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    monkeypatch.setattr(image, "CODEX_GENERATED_DIR", tmp_path / "none")
    monkeypatch.setattr("llmx.usage_log.log_usage", lambda **kw: None)
    with pytest.raises(RuntimeError, match="no image"):
        image.generate_codex_image("a cell", str(tmp_path / "o.png"))
