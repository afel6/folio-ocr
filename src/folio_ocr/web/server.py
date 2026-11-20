"""Start the local web service: bind the port, print where it listens, run until Ctrl-C."""

import errno
import socket
from collections.abc import Callable, Mapping

import h11
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from folio_ocr.config import Settings, find_tesseract, is_loopback
from folio_ocr.model_reader import ModelConfig, ModelReader
from folio_ocr.web import guard
from folio_ocr.web.guard import stderr_line

UNREADABLE_REQUEST = "the request could not be read"
_DENIED = {errno.EACCES, errno.EPERM, getattr(errno, "WSAEACCES", 10013)}


def bind(host: str, port: int) -> socket.socket:
    """Listen on host and port (port 0 picks any free port)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return socket.create_server((host, port), family=family)


def bind_error(exc: OSError, host: str, port: int) -> str:
    """One friendly line for a failure to listen, chosen by the error kind only."""
    if isinstance(exc, socket.gaierror):
        return f"cannot find the address {host}; check FOLIO_HOST"
    if exc.errno == errno.EADDRINUSE:
        return f"port {port} is already in use; stop the other program or set FOLIO_PORT to another port"
    if exc.errno in _DENIED:
        return f"permission denied to listen on port {port}; set FOLIO_PORT to another port"
    if exc.errno == errno.EADDRNOTAVAIL:
        return f"{host} is not an address of this machine; check FOLIO_HOST"
    return f"cannot listen on {host} port {port}"


def address(host: str, port: int) -> str:
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/"


class _Http(H11Protocol):
    """uvicorn's HTTP/1.1, except that a request it cannot parse gets the service's one-line
    JSON error with the security headers."""

    def send_400_response(self, msg: str) -> None:
        if self.transport.is_closing():  # already answered; the rest of the bytes are ignored
            return
        body = guard.error_body(UNREADABLE_REQUEST)
        headers = [*guard.error_headers(body), (b"connection", b"close")]
        for event in (
            h11.Response(status_code=400, headers=headers, reason=b"Bad Request"),
            h11.Data(data=body),
            h11.EndOfMessage(),
        ):
            self.transport.write(self.conn.send(event))
        self.transport.close()


def build_server(app, sock: socket.socket) -> uvicorn.Server:
    return uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="warning",
            access_log=False,
            server_header=False,
            timeout_graceful_shutdown=5,
            lifespan="on",
            http=_Http,
        )
    )


def _announce(settings: Settings, port: int, write: Callable[[str], None]) -> None:
    url = address(settings.host, port)
    write(
        f"folio serve: open {url} in a browser; press Ctrl-C to stop (open documents are then cleared)"
    )
    if not is_loopback(settings.host):
        write(
            "warning: listening beyond this machine: anyone on the network who can reach "
            f"{url} can list, read and delete documents"
        )


def serve(
    settings: Settings,
    environ: Mapping[str, str],
    *,
    write: Callable[[str], None] = stderr_line,
    make_app=None,
    make_server=build_server,
) -> int:
    """Run the web service until it is stopped; returns the exit code."""
    config = ModelConfig.from_settings(settings)
    try:
        sock = bind(settings.host, settings.port)
    except OSError as exc:
        write(f"error: {bind_error(exc, settings.host, settings.port)}")
        return 1
    try:
        port = sock.getsockname()[1]
        model = None if config is None else ModelReader(config)
        if make_app is None:
            from folio_ocr.web.app import create_app as make_app
        app = make_app(
            settings, port=port, tesseract=find_tesseract(settings, environ), model=model
        )
        _announce(settings, port, write)
        srv = make_server(app, sock)
        try:
            srv.run(sockets=[sock])
        except KeyboardInterrupt:
            write("folio serve stopped")
            return 130
        if srv.started:
            return 0
        write("error: the web service could not start")
        return 1
    finally:
        sock.close()
