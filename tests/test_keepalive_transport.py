"""Transport-level keepalive for long streaming responses (offline)."""

from __future__ import annotations

import errno
import json
import socket
from unittest.mock import patch

import pytest

from llmx import keepalive, providers


def _sse(events: list[dict]) -> bytes:
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


class _StreamingResponse:
    """Minimal urllib response: line-iterable, closable, with headers."""

    code = 200
    msg = "OK"
    reason = "OK"

    def __init__(self, lines: list[bytes], headers: dict | None = None):
        self._lines = lines
        self.headers = headers or {}

    def info(self):
        return self.headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


class _BrokenResponse(_StreamingResponse):
    def __iter__(self):
        yield self._lines[0]
        yield self._lines[1]
        raise OSError(errno.ETIMEDOUT, "Operation timed out")


def _message_lines() -> list[bytes]:
    payload = _sse([
        {"type": "message_start",
         "message": {"model": "served-opus", "usage": {"input_tokens": 10}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ])
    return payload.splitlines(keepends=True)


def test_helper_sets_kernel_keepalive_on_a_real_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        keepalive.enable_tcp_keepalive(sock)
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
        idle_option = getattr(socket, "TCP_KEEPALIVE", None)
        if idle_option is None:
            idle_option = getattr(socket, "TCP_KEEPIDLE", None)
        if idle_option is not None:
            assert sock.getsockopt(socket.IPPROTO_TCP, idle_option) == 60
        if hasattr(socket, "TCP_KEEPINTVL"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 15
        if hasattr(socket, "TCP_KEEPCNT"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 4
    finally:
        sock.close()


@pytest.mark.parametrize("stream", [False, True])
def test_native_claude_transport_opens_a_keepalive_connection(stream):
    instances = []

    class FakeConnection:
        sock = None

        def __init__(self, host, timeout=None, **kwargs):
            instances.append({"host": host, "timeout": timeout, **kwargs})
            self._response = _StreamingResponse(
                _message_lines(), headers={"request-id": "req_keepalive-1"})

        def set_debuglevel(self, level):
            pass

        def request(self, method, url, body=None, headers=None, **kwargs):
            pass

        def getresponse(self):
            return self._response

        def close(self):
            pass

    with patch.object(keepalive, "KeepaliveHTTPSConnection", FakeConnection), \
            patch("llmx.providers._get_api_key", return_value="fake"), \
            patch("llmx.usage_log.log_usage"):
        result = providers._openai_chat(
            "hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30,
            stream, None, None, "max")

    assert result == "ok"
    assert len(instances) == 1
    assert instances[0]["host"] == "api.anthropic.com"
    assert instances[0]["timeout"] == 30


@pytest.mark.parametrize("stream", [False, True])
def test_keepalive_timeout_mid_stream_is_a_transient_transport_error(stream):
    broken = _BrokenResponse([
        b'data: {"type":"message_start","message":{"model":"served","usage":{}}}\n',
        b"\n",
    ])
    with patch("llmx.keepalive.open_url", return_value=broken), \
            patch("llmx.providers._get_api_key", return_value="fake"), \
            patch("llmx.usage_log.log_usage"), \
            pytest.raises(providers.ServiceUnavailableError) as caught:
        providers._openai_chat(
            "hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30,
            stream, None, None, "max")

    assert "keepalive" in str(caught.value)
    assert caught.value.error_type == "service_unavailable"
    assert isinstance(caught.value, providers.RateLimitError)
