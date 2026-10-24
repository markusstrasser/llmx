"""TCP keepalive for long-lived streaming HTTP transports.

A per-read socket timeout cannot tell a slow model from a dead network. The
native Claude profile sets that timeout to the whole call budget (3600 s), so a
link that dies mid-stream stalls for the rest of the hour. Kernel keepalive
probes detect a dead peer in roughly two minutes no matter how often the API
sends events, which is why this lives at the transport instead of behind an
application-level idle timer (a long thinking phase can legitimately be silent).

Every long-response transport in llmx must open its connection through
``open_url`` so its HTTPS socket carries these options.
"""

from __future__ import annotations

import http.client
import socket
import urllib.request

# 60 s idle + 4 probes 15 s apart ≈ 2 minutes to notice a dead peer.
KEEPALIVE_IDLE_SECONDS = 60
KEEPALIVE_INTERVAL_SECONDS = 15
KEEPALIVE_PROBE_COUNT = 4


def enable_tcp_keepalive(sock: socket.socket) -> None:
    """Enable SO_KEEPALIVE and the platform's probe knobs on a connected socket.

    macOS exposes the idle time as ``TCP_KEEPALIVE``; Linux as ``TCP_KEEPIDLE``.
    Options are only set where the platform defines them.
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    idle_option = "TCP_KEEPALIVE" if hasattr(socket, "TCP_KEEPALIVE") else "TCP_KEEPIDLE"
    for option, value in (
        (idle_option, KEEPALIVE_IDLE_SECONDS),
        ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL_SECONDS),
        ("TCP_KEEPCNT", KEEPALIVE_PROBE_COUNT),
    ):
        if hasattr(socket, option):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)


class KeepaliveHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that turns on TCP keepalive as soon as it is connected."""

    def connect(self) -> None:
        super().connect()
        enable_tcp_keepalive(self.sock)


class _KeepaliveHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        kwargs = {"context": self._context}
        check_hostname = getattr(self, "_check_hostname", None)
        if check_hostname is not None:
            kwargs["check_hostname"] = check_hostname
        return self.do_open(KeepaliveHTTPSConnection, req, **kwargs)


_OPENER = urllib.request.build_opener(_KeepaliveHTTPSHandler())


def open_url(request, *, timeout):
    """Open an HTTP(S) URL like ``urlopen``, with TCP keepalive on the socket.

    Accepts a URL string or ``urllib.request.Request``. The caller's per-call
    ``timeout`` is unchanged: it still bounds each read.
    """
    return _OPENER.open(request, timeout=timeout)
