import asyncio
import json

import pytest
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from folio_ocr.web import guard
from folio_ocr.web.guard import Guard, allowed_hosts

HOSTS = allowed_hosts("127.0.0.1", 8000)


async def ok(request):
    return PlainTextResponse("ok")


def make_client(lines):
    inner = Starlette(routes=[Route("/", ok, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])])

    async def routed(scope, receive, send):
        if scope["path"] == "/boom":
            raise ValueError("SECRET-TEXT")
        await inner(scope, receive, send)

    app = Guard(routed, hosts=HOSTS, log=lines.append)
    return TestClient(app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False)


def test_allowed_hosts():
    assert {"127.0.0.1:8000", "localhost:8000", "[::1]:8000"} <= allowed_hosts("127.0.0.1", 8000)
    assert "localhost:8000" in allowed_hosts("localhost", 8000)
    assert "[::1]:8000" in allowed_hosts("::1", 8000)
    assert allowed_hosts("192.168.1.5", 8000) == {"192.168.1.5:8000"}
    assert "127.0.0.1" in allowed_hosts("127.0.0.1", 80)
    assert "127.0.0.1" not in allowed_hosts("127.0.0.1", 8000)


@pytest.mark.parametrize(
    ("host", "status"),
    [
        ("127.0.0.1:8001", 400),
        ("evil.example:8000", 400),
        ("localhost:8000", 200),
        ("LOCALHOST:8000", 400),
    ],
)
def test_host_check(host, status):
    r = make_client([]).get("/", headers={"host": host})
    assert r.status_code == status
    if status == 400:
        assert r.json() == {"error": guard.BAD_HOST}


@pytest.mark.parametrize(
    ("method", "origin", "status"),
    [
        ("POST", None, 200),
        ("POST", "http://127.0.0.1:8000", 200),
        ("POST", "http://evil.example", 403),
        ("PUT", "http://evil.example", 403),
        ("DELETE", "http://evil.example", 403),
        ("POST", "null", 403),
        ("POST", "https://127.0.0.1:8000", 403),
        ("POST", "http://127.0.0.1:8001", 403),
        ("GET", "http://evil.example", 200),
    ],
)
def test_origin_check(method, origin, status):
    headers = {} if origin is None else {"origin": origin}
    r = make_client([]).request(method, "/", headers=headers)
    assert r.status_code == status
    if status == 403:
        assert r.json() == {"error": guard.FOREIGN_ORIGIN}


def test_two_origin_headers_refused():
    r = make_client([]).post(
        "/", headers=[("origin", "http://127.0.0.1:8000"), ("origin", "http://127.0.0.1:8000")]
    )
    assert r.status_code == 403


def test_headers_on_every_status():
    c = make_client([])
    responses = [
        c.get("/"),
        c.get("/", headers={"host": "evil.example"}),
        c.post("/", headers={"origin": "http://evil.example"}),
        c.get("/boom"),
        c.options("/", headers={"origin": "http://evil.example"}),
    ]
    assert [r.status_code for r in responses] == [200, 400, 403, 500, 200]
    for r in responses:
        assert r.headers["content-security-policy"] == "default-src 'self'"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["cache-control"] == "no-store"
        assert not [k for k in r.headers if k.lower().startswith("access-control-allow")]
        if r.status_code >= 400:
            assert "\n" not in r.text


def test_exception_before_start_is_one_logged_line():
    lines = []
    r = make_client(lines).get("/boom")
    assert r.status_code == 500
    assert r.json() == {"error": guard.UNEXPECTED}
    assert len(lines) == 1
    assert "ValueError" in lines[0]
    assert "SECRET-TEXT" not in lines[0]
    assert "/boom" not in lines[0]


def test_default_log_is_one_stderr_line(capsys):
    async def app(scope, receive, send):
        raise ValueError("SECRET-TEXT")

    async def send(message):
        pass

    scope = {"type": "http", "method": "GET", "headers": [(b"host", b"127.0.0.1:8000")]}
    asyncio.run(Guard(app, hosts=HOSTS)(scope, None, send))
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "ValueError" in err
    assert "SECRET-TEXT" not in err


def test_exception_after_start_does_not_escape():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ValueError("SECRET-TEXT")

    lines, sent = [], []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "headers": [(b"host", b"127.0.0.1:8000")]}
    asyncio.run(Guard(app, hosts=HOSTS, log=lines.append)(scope, None, send))
    assert [m["type"] for m in sent] == ["http.response.start"]
    assert len(lines) == 1


def test_non_http_scope_passes_through():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(Guard(app, hosts=HOSTS)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]


def test_app_messages_are_not_mutated():
    start = {"type": "http.response.start", "status": 200, "headers": [(b"a", b"b")]}

    async def app(scope, receive, send):
        await send(start)

    async def send(message):
        pass

    scope = {"type": "http", "method": "GET", "headers": [(b"host", b"127.0.0.1:8000")]}
    asyncio.run(Guard(app, hosts=HOSTS)(scope, None, send))
    assert start["headers"] == [(b"a", b"b")]


async def run_raw(app, lines):
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"host", b"127.0.0.1:8000")],
    }
    await Guard(app, hosts=HOSTS, log=lines.append)(scope, receive, send)
    return sent


def test_app_returning_without_a_response_is_a_500():
    async def silent(scope, receive, send):
        return None

    lines = []
    sent = asyncio.run(run_raw(silent, lines))
    assert sent[0]["status"] == 500
    assert json.loads(sent[1]["body"]) == {"error": guard.UNEXPECTED}
    assert len(lines) == 1 and "no response" in lines[0]


def test_client_disconnect_is_quiet():
    async def gone(scope, receive, send):
        raise ClientDisconnect()

    lines = []
    assert asyncio.run(run_raw(gone, lines)) == []
    assert lines == []
