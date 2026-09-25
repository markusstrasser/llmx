"""Regression test for the grandchild-pipe wedge (cli_backends b8e26b0).

A codex-cli descendant that setsid()s out of the process group before the pipe
fds close escapes killpg and keeps stdout open, so proc.communicate() never sees
EOF: pre-fix, cli_chat hung unboundedly past --timeout (observed 2h28m; arc-agi
row llmx-codex-timeout-not-enforced, 2026-07-18). The fix bounds the wait via
thread-join + force-closing our pipe ends. This test runs both polarities
against the LIVE package with a fake `codex` binary on PATH:

  (i)  fast fake codex — completes normally, cli_chat returns real text quickly.
  (ii) wedge fake codex — spawns a setsid grandchild holding stdout; cli_chat
       must return a typed (non-str) failure within timeout + 2*grace + margin.

Scope note: the fix bounds llmx's own WAIT. It does not reap the escaped
grandchild (no pgid tracking survives setsid); the test kills any survivor in
cleanup and does not gate on its absence. ~30-60s wall for the wedge case.

One-shot verification of the original patch (applies it to a pre-fix copy):
arc-agi experiments/composed_sol_match/killswitch/llmx_fix/.
"""

import os
import stat
import subprocess
import time

from llmx import cli_backends as cb

TIMEOUT_S = 5
GRACE_S = 20  # matches _COMMUNICATE_GRACE_S in cli_backends
WEDGE_BOUND_S = TIMEOUT_S + 2 * GRACE_S + 10  # kill-join + force-close-join + margin

FAST_CODEX = (
    "#!/bin/bash\n"
    'echo \'{"type":"item.completed","item":{"type":"agent_message",'
    '"text":"hello from fake codex"}}\'\n'
    "exit 0\n"
)

# Forks a grandchild that re-sessions (setsid) before the parent's pipe fds
# close, escaping killpg while holding a dup of fd 1 — the pipe cli_chat's
# communicate() is blocked reading from.
WEDGE_CODEX = """#!/usr/bin/env python3
import os, sys, time

marker = os.environ.get("WEDGE_MARKER_FILE")
pid = os.fork()
if pid == 0:
    os.setsid()
    if marker:
        with open(marker, "w") as f:
            f.write(str(os.getpid()))
    time.sleep(600)
    sys.exit(0)
time.sleep(600)
"""


def _write_fake_codex(bindir, script: str) -> None:
    path = os.path.join(bindir, "codex")
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(timeout: int):
    start = time.time()
    result = cb.cli_chat(
        provider="codex-cli",
        prompt="say hi",
        model="gpt-6-sol",
        timeout=timeout,
        lite="bare",
    )
    return time.time() - start, result


def test_fast_codex_completes_normally(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_fake_codex(str(bindir), FAST_CODEX)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    elapsed, result = _run(TIMEOUT_S)

    assert elapsed < TIMEOUT_S
    assert isinstance(result, str)
    assert "hello from fake codex" in result


def test_wedge_grandchild_cannot_hang_cli_chat(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_fake_codex(str(bindir), WEDGE_CODEX)
    marker = tmp_path / "wedge_grandchild.pid"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("WEDGE_MARKER_FILE", str(marker))

    try:
        elapsed, result = _run(TIMEOUT_S)
    finally:
        _kill_escaped_grandchild(str(marker))

    assert elapsed < WEDGE_BOUND_S, (
        f"cli_chat took {elapsed:.1f}s — the grandchild-pipe wedge regressed "
        f"(pre-fix behavior: unbounded hang past --timeout)"
    )
    assert not isinstance(result, str), (
        f"expected a typed timeout failure, got text: {result!r}"
    )


def _kill_escaped_grandchild(marker_path: str) -> None:
    """Exact-PID cleanup of the deliberately-escaped grandchild (never pgrep)."""
    if not os.path.exists(marker_path):
        return
    try:
        pid = int(open(marker_path).read().strip())
    except (ValueError, OSError):
        return
    if subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
