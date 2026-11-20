"""One HTTP request to an OpenAI-compatible model server, under an overall deadline."""

from __future__ import annotations

import base64
import http.client
import socket
import ssl
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

_CHUNK = 65536
# The longest wait a socket timeout and a timer accept on every platform (about 23 days);
# a longer configured limit acts as this one.
MAX_WAIT_S = 2_000_000.0
_TIMED_OUT = "the model server did not answer in time"
_REFUSED = "the model server refused the connection; check that it is running"
_UNRESOLVED = "the model server's host name could not be resolved"
_TLS = "the secure connection to the model server failed (certificate or TLS error)"
_BROKEN = "the model server closed the connection or sent a broken HTTP reply"
_NETWORK = "the model server could not be reached (network error)"


class ModelCallError(Exception):
    """Why a call failed, as one line that is safe to show: never the URL, host, port, key,
    user name, password or the server's body."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


@dataclass(frozen=True)
class Target:
    scheme: str
    hostport: str
    path: str
    headers: tuple[tuple[str, bytes], ...] = field(repr=False)


@dataclass(frozen=True)
class Reply:
    status: int
    body: bytes


class _Late(Exception):
    """The overall time limit has run out."""


def _ascii_only(text: str) -> str:
    return "".join(c if c.isascii() else urllib.parse.quote(c) for c in text)


def _decode(part: str) -> str:
    try:
        return urllib.parse.unquote(part, errors="strict")
    except UnicodeDecodeError:
        return part


def _authorization(netloc: str, api_key: str | None) -> tuple[tuple[str, bytes], ...]:
    if api_key is not None:
        return (("Authorization", b"Bearer " + api_key.encode("utf-8")),)
    if "@" not in netloc:
        return ()
    user, _, password = netloc.rpartition("@")[0].partition(":")
    pair = f"{_decode(user)}:{_decode(password)}".encode()
    return (("Authorization", b"Basic " + base64.b64encode(pair)),)


def _with_query(url: str, path: str) -> str:
    """`path` made sendable, followed by the URL's query exactly as written, if it has one."""
    parts = urllib.parse.urlsplit(url)
    if "?" in url.partition("#")[0]:
        return _ascii_only(path) + "?" + _ascii_only(parts.query)
    return _ascii_only(path)


def target(url: str, api_key: str | None) -> Target:
    """Build the request target and headers for the chat completions endpoint."""
    parts = urllib.parse.urlsplit(url)
    path = parts.path if parts.path.endswith("/") else parts.path + "/"
    headers = (("Content-Type", b"application/json"), *_authorization(parts.netloc, api_key))
    return Target(
        parts.scheme,
        parts.netloc.rpartition("@")[2],
        _with_query(url, path + "chat/completions"),
        headers,
    )


def native_target(url: str, api_key: str | None, endpoint: str) -> Target | None:
    """The target of the server's own `/api/<endpoint>` next to its `/v1` base URL (Ollama),
    or None when the base URL does not end in `/v1`."""
    parts = urllib.parse.urlsplit(url)
    path = parts.path.removesuffix("/")
    if not path.endswith("/v1"):
        return None
    root = path.removesuffix("/v1")
    headers = _authorization(parts.netloc, api_key)
    return Target(
        parts.scheme,
        parts.netloc.rpartition("@")[2],
        _with_query(url, f"{root}/api/{endpoint}"),
        headers,
    )


def open_connection(target: Target, timeout_s: float) -> http.client.HTTPConnection:
    """Make an unconnected connection; no proxy is used."""
    if target.scheme == "https":
        return http.client.HTTPSConnection(
            target.hostport, timeout=timeout_s, context=ssl.create_default_context()
        )
    return http.client.HTTPConnection(target.hostport, timeout=timeout_s)


def is_late(started: float, now: float, timeout_s: float) -> bool:
    return now - started >= timeout_s


class _Watched:
    """The socket the time-limit timer shuts down. It is kept here rather than read from the
    connection, because http.client drops the connection's reference once the headers of a
    reply that ends with the connection have been read."""

    def __init__(self) -> None:
        self.sock: socket.socket | None = None


def _abort(watched: _Watched) -> None:
    sock = watched.sock
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        # The socket is already closed, so there is nothing left to shut down.
        return


def _address_by_address(
    left: Callable[[], float], watched: _Watched
) -> Callable[..., socket.socket]:
    """socket.create_connection for http.client, except that each address gets only the time
    still left (the standard one gives every address of a host the full limit)."""

    def create(address: tuple, *_args: object, **_kwargs: object) -> socket.socket:
        failure: OSError = OSError("the model server's host name has no address")
        for family, kind, proto, _, where in socket.getaddrinfo(
            address[0], address[1], 0, socket.SOCK_STREAM
        ):
            try:
                sock = socket.socket(family, kind, proto)
            except OSError as exc:  # this address family cannot be used here: try the next
                failure = exc
                continue
            watched.sock = sock
            try:
                sock.settimeout(left())
                sock.connect(where)
                return sock
            except OSError as exc:
                sock.close()
                failure = exc
            except _Late:
                sock.close()
                raise
        raise failure

    return create


def _failure(exc: Exception, late: bool) -> ModelCallError:
    if late or isinstance(exc, _Late | TimeoutError):
        return ModelCallError(_TIMED_OUT, timed_out=True)
    if isinstance(exc, ConnectionRefusedError):
        return ModelCallError(_REFUSED)
    if isinstance(exc, socket.gaierror | UnicodeError):
        return ModelCallError(_UNRESOLVED)
    if isinstance(exc, ssl.SSLError):
        return ModelCallError(_TLS)
    if isinstance(exc, http.client.HTTPException | ConnectionError):
        return ModelCallError(_BROKEN)
    return ModelCallError(_NETWORK)


def _exchange(
    conn: http.client.HTTPConnection, target: Target, body: bytes | None, left: Callable[[], float]
) -> Reply:
    sock = conn.sock  # the response may close the connection's own reference early
    sock.settimeout(left())
    method = "GET" if body is None else "POST"
    conn.request(method, target.path, body=body, headers=dict(target.headers))
    sock.settimeout(left())
    response = conn.getresponse()
    chunks = []
    while not response.isclosed():
        sock.settimeout(left())
        chunk = response.read1(_CHUNK)
        if not chunk:
            break
        chunks.append(chunk)
    if response.length:  # the server hung up before sending the announced length
        raise http.client.IncompleteRead(b"".join(chunks), response.length)
    return Reply(response.status, b"".join(chunks))


def post(
    target: Target,
    body: bytes,
    timeout_s: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
    connect: Callable[[Target, float], http.client.HTTPConnection] = open_connection,
) -> Reply:
    """Send one POST and return the reply; the limit covers connect to the last body byte."""
    return _send(target, body, timeout_s, clock, connect)


def get(
    target: Target,
    timeout_s: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
    connect: Callable[[Target, float], http.client.HTTPConnection] = open_connection,
) -> Reply:
    """Send one GET and return the reply, under the same overall limit as `post`."""
    return _send(target, None, timeout_s, clock, connect)


def _send(
    target: Target,
    body: bytes | None,
    timeout_s: float,
    clock: Callable[[], float],
    connect: Callable[[Target, float], http.client.HTTPConnection],
) -> Reply:
    timeout_s = min(timeout_s, MAX_WAIT_S)
    started = clock()

    def left() -> float:
        remaining = timeout_s - (clock() - started)
        if remaining <= 0:
            raise _Late
        return remaining

    conn = None
    watched = _Watched()
    watchdog = threading.Timer(timeout_s, _abort, args=(watched,))
    watchdog.daemon = True
    try:
        conn = connect(target, left())
        # http.client opens its socket through this attribute; replacing it bounds each
        # address by the time left.
        conn._create_connection = _address_by_address(left, watched)
        watchdog.start()
        conn.connect()
        watched.sock = conn.sock  # a TLS socket replaces the plain one
        reply = _exchange(conn, target, body, left)
        if is_late(started, clock(), timeout_s):
            raise _Late
        return reply
    except (_Late, OSError, http.client.HTTPException, ValueError) as exc:
        # ValueError covers a bad host name (UnicodeError) and a TLS socket that the
        # timer shut down in the middle of a read.
        raise _failure(exc, is_late(started, clock(), timeout_s)) from None
    finally:
        watchdog.cancel()
        if conn is not None:
            conn.close()
        if watched.sock is not None:
            watched.sock.close()
