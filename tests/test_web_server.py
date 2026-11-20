"""Starting the local web service: binding, messages, shutdown."""

import errno
import http.client
import os
import socket
import threading
import time

import pytest
from fastapi import FastAPI

from folio_ocr import cli
from folio_ocr.config import Settings
from folio_ocr.errors import ConfigError
from folio_ocr.web import server


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)


def settings(tmp_path, **kw):
    return Settings(weights_dir=tmp_path, host=kw.pop("host", "127.0.0.1"), port=0, **kw)


class FakeServer:
    def __init__(self, started=True, raises=None):
        self.started, self.raises, self.sockets = started, raises, None

    def run(self, sockets):
        self.sockets = [x.getsockname()[1] for x in sockets]
        if self.raises:
            raise self.raises


def serve_with(tmp_path, fake=None, **kw):
    lines, made = [], {}

    def make_app(settings, **kwargs):
        made.update(kwargs)
        return "app"

    fake = fake or FakeServer()
    code = server.serve(
        settings(tmp_path, **kw), {}, write=lines.append, make_app=make_app,
        make_server=lambda app, sock: fake,
    )  # fmt: skip
    return code, lines, made, fake


def test_port_in_use_is_one_friendly_line(tmp_path):
    with socket.create_server(("127.0.0.1", 0)) as held:
        p = held.getsockname()[1]
        lines = []
        s = Settings(weights_dir=tmp_path, port=p)
        code = server.serve(s, {}, write=lines.append, make_app=lambda *a, **k: None)
    assert code == 1
    expected = f"error: port {p} is already in use; stop the other program or set FOLIO_PORT"
    assert lines == [expected + " to another port"]


@pytest.mark.parametrize(
    ("exc", "line"),
    [
        (OSError(errno.EACCES, "SECRET"), "permission denied to listen on port 80; set FOLIO_PORT to another port"),
        (OSError(errno.EPERM, "SECRET"), "permission denied to listen on port 80; set FOLIO_PORT to another port"),
        (OSError(10013, "SECRET"), "permission denied to listen on port 80; set FOLIO_PORT to another port"),
        (OSError(errno.EADDRNOTAVAIL, "SECRET"), "h is not an address of this machine; check FOLIO_HOST"),
        (socket.gaierror(-2, "SECRET"), "cannot find the address h; check FOLIO_HOST"),
        (OSError(errno.EIO, "SECRET"), "cannot listen on h port 80"),
    ],
)  # fmt: skip
def test_bind_error_lines(exc, line):
    assert server.bind_error(exc, "h", 80) == line


def test_address():
    assert server.address("127.0.0.1", 8000) == "http://127.0.0.1:8000/"
    assert server.address("::1", 8000) == "http://[::1]:8000/"
    assert server.address("localhost", 80) == "http://localhost:80/"


def test_serve_runs_the_app_on_the_bound_port(tmp_path):
    code, lines, made, fake = serve_with(tmp_path)
    assert code == 0
    assert made["model"] is None and made["port"] == fake.sockets[0] != 0
    assert lines == [
        f"folio serve: open http://127.0.0.1:{made['port']}/ in a browser; press Ctrl-C to stop "
        + "(open documents are then cleared)"
    ]


def test_remote_host_gets_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "bind", lambda host, port: socket.create_server(("127.0.0.1", 0)))
    _, lines, made, _ = serve_with(tmp_path, host="192.0.2.1")
    url = f"http://192.0.2.1:{made['port']}/"
    assert lines[1:] == [
        "warning: listening beyond this machine: anyone on the network who can reach "
        + f"{url} can list, read and delete documents"
    ]
    assert len(lines) == 2


def test_ctrl_c_stops_with_130(tmp_path):
    code, lines, _, _ = serve_with(tmp_path, FakeServer(raises=KeyboardInterrupt()))
    assert code == 130 and lines[-1] == "folio serve stopped"


def test_not_started_is_an_error(tmp_path):
    code, lines, _, _ = serve_with(tmp_path, FakeServer(started=False))
    assert code == 1 and lines[-1] == "error: the web service could not start"


def test_model_url_without_name_is_refused_before_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "bind", lambda *a: pytest.fail("bound"))
    s = Settings(weights_dir=tmp_path, model_url="http://127.0.0.1:9/v1", model_name=None)
    with pytest.raises(ConfigError):
        server.serve(s, {}, write=lambda line: None, make_app=lambda *a, **k: None)


def test_server_uses_the_json_error_protocol():
    sock = server.bind("127.0.0.1", 0)
    try:
        assert server.build_server(FastAPI(), sock).config.http is server._Http
    finally:
        sock.close()


def test_real_server_answers_and_stops():
    app = FastAPI()
    app.get("/ping")(lambda: {"ok": True})
    sock = server.bind("127.0.0.1", 0)
    srv = server.build_server(app, sock)
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        end = time.monotonic() + 10
        while not srv.started and time.monotonic() < end:
            time.sleep(0.02)
        assert srv.started
        conn = http.client.HTTPConnection("127.0.0.1", sock.getsockname()[1], timeout=5)
        conn.request("GET", "/ping")
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        srv.should_exit = True
        thread.join(10)
        sock.close()
    assert not thread.is_alive()


ENV = {"FOLIO_HOST": "0.0.0.0"}


def test_cli_refuses_remote_host_without_flag(tmp_path, capsys):
    env = {**ENV, "FOLIO_WEIGHTS_DIR": str(tmp_path)}
    assert cli.main(["serve"], env) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: FOLIO_HOST is not a loopback address") and err.count("\n") == 1


def test_cli_allow_remote_passes_settings(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(server, "serve", lambda s, e: got.append((s, e)) or 0)
    env = {**ENV, "FOLIO_WEIGHTS_DIR": str(tmp_path)}
    assert cli.main(["serve", "--allow-remote"], env) == 0
    assert got[0][0].host == "0.0.0.0" and got[0][1] is env


def test_cli_default_host_is_loopback(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(server, "serve", lambda s, e: got.append((s, e)) or 0)
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path)}
    assert cli.main(["serve"], env) == 0
    assert got[0][0].host == "127.0.0.1" and got[0][1] is env
