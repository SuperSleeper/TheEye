"""HTTPS and an http -> https redirect on the same port.

uvicorn is started without TLS and with this as its HTTP protocol. Each new connection is peeked at (without
consuming anything): a TLS handshake (first byte 0x16) is wrapped in TLS and handed to uvicorn's normal HTTP
protocol; anything else is plain HTTP and gets a 301 to the same address over https. So http://host:8080 just
works instead of failing with a protocol error. The visitor's address is kept (login lockouts, local-only setup).
Needs uvicorn's loop="asyncio": uvloop reads a new connection even while it's paused, so the peek sees nothing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import ssl
from typing import Optional

from uvicorn.config import HTTP_PROTOCOLS
from uvicorn.importer import import_from_string

log = logging.getLogger("theeye.tlsredirect")

TLS_HANDSHAKE = 0x16
FIRST_BYTE_TIMEOUT = 15.0          # a connection that sends nothing is dropped
MAX_HEAD = 16384                   # plain-HTTP request head we'll read before redirecting


def make_protocol(certfile: str, keyfile: str):
    """The protocol class to pass to uvicorn as http=..."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    ctx.set_alpn_protocols(["http/1.1"])
    inner_cls = import_from_string(HTTP_PROTOCOLS["auto"])

    class SniffTLS(asyncio.Protocol):
        def __init__(self, config, server_state, app_state, _loop=None):
            self._args = (config, server_state, app_state)
            self._loop = _loop or asyncio.get_event_loop()
            self._tr: Optional[asyncio.Transport] = None
            self._plain = False
            self._buf = b""

        def connection_made(self, transport) -> None:
            self._tr = transport
            transport.pause_reading()                 # peek first; read nothing yet
            self._loop.create_task(self._sniff())

        async def _sniff(self) -> None:
            tr = self._tr
            sock = tr.get_extra_info("socket")
            if sock is None:
                tr.close()
                return
            peek = socket.socket(fileno=os.dup(sock.fileno()))   # same connection, separate fd for waiting
            ready = self._loop.create_future()
            try:
                self._loop.add_reader(peek.fileno(), lambda: ready.done() or ready.set_result(None))
                try:
                    await asyncio.wait_for(ready, FIRST_BYTE_TIMEOUT)
                finally:
                    self._loop.remove_reader(peek.fileno())
                first = peek.recv(1, socket.MSG_PEEK)
            except (asyncio.TimeoutError, OSError):
                first = b""
            finally:
                peek.close()              # closes only the duplicate fd; the connection stays open
            if not first or tr.is_closing():
                tr.close()
                return
            if first[0] == TLS_HANDSHAKE:
                # The browser's request often arrives with the end of the handshake, before start_tls returns:
                # hold it in a stand-in until uvicorn's protocol has its connection_made.
                hold = _Hold()
                try:
                    tls_tr = await self._loop.start_tls(tr, hold, ctx, server_side=True, ssl_handshake_timeout=15)
                except Exception:  # noqa: BLE001 - failed handshake (bad client, cert not trusted): drop it
                    tr.close()
                    return
                if tls_tr is None or hold.lost:
                    tr.close()
                    return
                inner = inner_cls(*self._args, _loop=self._loop)
                inner.connection_made(tls_tr)
                tls_tr.set_protocol(inner)
                for d in hold.data:
                    inner.data_received(d)
                if hold.eof:
                    inner.eof_received()
            else:
                self._plain = True
                tr.resume_reading()

        def data_received(self, data: bytes) -> None:     # plain HTTP only
            if not self._plain:
                return
            self._buf += data
            if b"\r\n\r\n" not in self._buf and len(self._buf) < MAX_HEAD:
                return
            self._redirect()

        def _redirect(self) -> None:
            head = self._buf.split(b"\r\n\r\n", 1)[0].decode("latin-1", "replace").split("\r\n")
            parts = head[0].split(" ")
            path = parts[1] if len(parts) >= 2 and parts[1].startswith("/") else "/"
            host = next((h.split(":", 1)[1].strip() for h in head[1:] if h.lower().startswith("host:")), "")
            if not host or any(c in host for c in " \t/\\@"):
                sock = self._tr.get_extra_info("sockname")
                host = f"localhost:{sock[1]}" if sock else "localhost"
            loc = f"https://{host}{path}"
            body = f'<a href="{loc}">{loc}</a>\n'.encode()
            self._tr.write(
                b"HTTP/1.1 301 Moved Permanently\r\nLocation: " + loc.encode("latin-1", "replace") +
                b"\r\nContent-Type: text/html\r\nContent-Length: " + str(len(body)).encode() +
                b"\r\nConnection: close\r\n\r\n" + body)
            self._tr.close()

        def connection_lost(self, exc) -> None:
            pass

    return SniffTLS


class _Hold(asyncio.Protocol):
    """Collects what arrives over TLS between the handshake and the hand-over to uvicorn's protocol."""

    def __init__(self) -> None:
        self.data: list[bytes] = []
        self.eof = self.lost = False

    def data_received(self, data: bytes) -> None:
        self.data.append(data)

    def eof_received(self):
        self.eof = True
        return True                     # keep the connection open; the real protocol decides

    def connection_lost(self, exc) -> None:
        self.lost = True
