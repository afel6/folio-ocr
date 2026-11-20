"""Request guard of the local web service: host and origin checks, security headers and a
friendly one-line answer to an unexpected failure."""

import json
import sys
from collections.abc import Callable, Iterable

from starlette.requests import ClientDisconnect

from folio_ocr.config import is_loopback

SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"content-security-policy", b"default-src 'self'"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"cache-control", b"no-store"),
)
STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
BAD_HOST = "this address is not served here; open the address folio serve printed"
FOREIGN_ORIGIN = "refused: the request came from another site"
UNEXPECTED = "unexpected failure; please report it"


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    """The Host header values the service answers: its own address, plus the loopback names
    when it listens on the loopback."""
    names = {f"[{host}]" if ":" in host else host}
    if is_loopback(host):
        names |= {"localhost", "127.0.0.1", "[::1]"}
    allowed = {f"{name}:{port}" for name in names}
    return frozenset(allowed | names) if port == 80 else frozenset(allowed)


def error_body(message: str) -> bytes:
    return json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")


def error_headers(body: bytes) -> list[tuple[bytes, bytes]]:
    """The headers of a JSON error reply holding `body`."""
    return [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *SECURITY_HEADERS,
    ]


async def send_error(send, status: int, message: str) -> None:
    body = error_body(message)
    headers = error_headers(body)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def stderr_line(line: str) -> None:
    sys.stderr.write(line + "\n")


def _values(scope, name: bytes) -> list[bytes]:
    return [v for k, v in scope["headers"] if k == name]


class Guard:
    """ASGI middleware in front of the app."""

    def __init__(self, app, *, hosts: Iterable[str], log: Callable[[str], None] = stderr_line):
        self.app = app
        self.hosts = frozenset(hosts)
        self.log = log

    def _refusal(self, scope) -> tuple[int, str] | None:
        hosts = _values(scope, b"host")
        if len(hosts) != 1 or hosts[0].decode("latin-1") not in self.hosts:
            return 400, BAD_HOST
        if scope["method"] in STATE_CHANGING:
            origins = _values(scope, b"origin")
            if origins and origins != [b"http://" + hosts[0]]:
                return 403, FOREIGN_ORIGIN
        return None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        refusal = self._refusal(scope)
        if refusal:
            await send_error(send, *refusal)
            return
        started = False

        async def guarded_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = [*message.get("headers", ()), *SECURITY_HEADERS]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, guarded_send)
            if not started:
                self.log("error: unexpected failure (no response) while answering a request")
                await send_error(send, 500, UNEXPECTED)
        except ClientDisconnect:
            return  # the client went away mid-upload; nobody is listening
        except Exception as exc:  # noqa: BLE001 - logged, then answered with a 500
            name = type(exc).__name__
            self.log(f"error: unexpected failure ({name}) while answering a request")
            if not started:
                await send_error(send, 500, UNEXPECTED)
