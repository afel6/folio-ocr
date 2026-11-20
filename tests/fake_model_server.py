"""A local fake of an OpenAI-compatible model server, bound to 127.0.0.1 on a free port."""

from __future__ import annotations

import http
import socket
import socketserver
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class Seen:
    method: str
    target: str
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class Answer:
    status: int = 200
    body: bytes = b""
    headers: tuple[tuple[str, str], ...] = ()
    delay_s: float = 0.0
    byte_delay_s: float = 0.0
    close_after: int | None = None
    raw: bytes | None = None
    trickle: bytes = b""


def _read_request(rfile) -> Seen | None:
    line = rfile.readline(65537)
    parts = line.decode("latin-1").split()
    if len(parts) < 2:
        return None
    headers: dict[str, str] = {}
    while True:
        raw = rfile.readline(65537)
        if raw in (b"\r\n", b"\n", b""):
            break
        name, _, value = raw.decode("latin-1").partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0") or 0)
    return Seen(parts[0], parts[1], headers, rfile.read(length))


def _head(answer: Answer) -> bytes:
    reason = http.HTTPStatus(answer.status).phrase
    lines = [
        f"HTTP/1.1 {answer.status} {reason}",
        "Content-Type: application/json",
        f"Content-Length: {len(answer.body)}",
        "Connection: close",
        *(f"{k}: {v}" for k, v in answer.headers),
    ]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        fake: FakeModelServer = self.server.fake  # type: ignore[attr-defined]
        try:
            if self.request.recv(1, socket.MSG_PEEK) == b"":
                # An empty peek means the client closed without sending anything; drain
                # what is left and hang up. (A TLS hello is not empty: it is read as a
                # request line, found malformed and dropped.)
                self.request.recv(65536)
                return
            seen = _read_request(self.rfile)
            if seen is None:
                return
            fake.seen.append(seen)
            self._reply(fake.answer_for(seen))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The client went away mid-reply; there is nobody left to answer.
            return

    def _reply(self, answer: Answer) -> None:
        time.sleep(answer.delay_s)
        if answer.raw is not None:
            self.wfile.write(answer.raw)
            self._trickle(answer)
            return
        self.wfile.write(_head(answer))
        body = answer.body
        if answer.close_after is not None:
            body = body[: answer.close_after]
        if answer.byte_delay_s > 0:
            for i in range(len(body)):
                self.wfile.write(body[i : i + 1])
                self.wfile.flush()
                time.sleep(answer.byte_delay_s)
        else:
            self.wfile.write(body)
        self._trickle(answer)

    def _trickle(self, answer: Answer) -> None:
        pause = answer.byte_delay_s or 0.05
        for i in range(len(answer.trickle)):
            self.wfile.write(answer.trickle[i : i + 1])
            self.wfile.flush()
            time.sleep(pause)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class FakeModelServer:
    def __init__(self, answer: Answer | Callable[[Seen], Answer] = Answer()) -> None:
        self._answer = answer
        self.seen: list[Seen] = []
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.fake = self  # type: ignore[attr-defined]
        self.port: int = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def answer_for(self, seen: Seen) -> Answer:
        return self._answer(seen) if callable(self._answer) else self._answer

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
