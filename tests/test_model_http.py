"""Tests for the HTTP transport that talks to an OpenAI-compatible model server."""

from __future__ import annotations

import base64
import socket
import threading
import time

import pytest
from fake_model_server import Answer, FakeModelServer

from folio_ocr import model_http
from folio_ocr.model_http import (
    ModelCallError,
    Reply,
    is_late,
    open_connection,
    post,
    target,
)

TIMED_OUT = "the model server did not answer in time"
REFUSED = "the model server refused the connection; check that it is running"
UNRESOLVED = "the model server's host name could not be resolved"
TLS = "the secure connection to the model server failed (certificate or TLS error)"
BROKEN = "the model server closed the connection or sent a broken HTTP reply"


@pytest.mark.parametrize(
    ("url", "path"),
    [
        ("http://127.0.0.1:11434/v1", "/v1/chat/completions"),
        ("http://127.0.0.1:11434/v1/", "/v1/chat/completions"),
        ("http://h:1", "/chat/completions"),
        ("http://h/a/../v1?x=1&y=%2F#frag", "/a/../v1/chat/completions?x=1&y=%2F"),
        ("http://h/v1?", "/v1/chat/completions?"),
        ("http://h/%41b/v1", "/%41b/v1/chat/completions"),
        ("http://h/\N{LATIN SMALL LETTER E WITH ACUTE}/v1", "/%C3%A9/v1/chat/completions"),
        ("http://h/v1?q=\N{LATIN SMALL LETTER E WITH ACUTE}", "/v1/chat/completions?q=%C3%A9"),
    ],
)
def test_target_path(url, path):
    assert target(url, None).path == path


@pytest.mark.parametrize(
    ("url", "hostport"),
    [
        ("http://Host.Example:8080/v1", "Host.Example:8080"),
        ("http://[::1]:8080/v1", "[::1]:8080"),
        ("http://u:p@Host.Example:8080/v1", "Host.Example:8080"),
    ],
)
def test_target_hostport(url, hostport):
    t = target(url, None)
    assert t.hostport == hostport
    assert t.scheme == "http"


def _basic(text: str) -> bytes:
    return b"Basic " + base64.b64encode(text.encode("utf-8"))


def _auth(t):
    return [v for k, v in t.headers if k == "Authorization"]


def test_content_type_is_exact_and_first():
    assert target("http://h/v1", None).headers[0] == ("Content-Type", b"application/json")


def test_key_becomes_bearer():
    assert _auth(target("http://h/v1", "sk-1")) == [b"Bearer sk-1"]


@pytest.mark.parametrize(
    ("url", "plain"),
    [
        ("http://u%40x:p%3Aw@h/v1", "u@x:p:w"),
        ("http://u:%zz@h/v1", "u:%zz"),
        ("http://u:%ff@h/v1", "u:%ff"),
    ],
)
def test_user_info_becomes_basic(url, plain):
    assert _auth(target(url, None)) == [_basic(plain)]


def test_key_wins_over_user_info():
    assert _auth(target("http://u:p@h/v1", "sk-1")) == [b"Bearer sk-1"]


def test_no_credentials_no_authorization():
    assert _auth(target("http://h/v1", None)) == []


def test_repr_hides_secrets():
    text = repr(target("http://u:PASSWORD9@h/v1", "sk-KEY9"))
    assert "sk-KEY9" not in text
    assert "PASSWORD9" not in text
    assert "Authorization" not in text


def test_post_sends_request_and_returns_reply():
    with FakeModelServer(Answer(body=b'{"ok":1}')) as server:
        t = target(f"{server.url}?q=1", "sk-1")
        reply = post(t, b'{"a": 1}', 5.0)
    assert reply == Reply(200, b'{"ok":1}')
    (seen,) = server.seen
    assert seen.method == "POST"
    assert seen.target == "/v1/chat/completions?q=1"
    assert seen.body == b'{"a": 1}'
    assert seen.headers["content-type"] == "application/json"
    assert seen.headers["authorization"] == "Bearer sk-1"
    assert seen.headers["host"] == f"127.0.0.1:{server.port}"


def _fail(server, timeout_s, **kw):
    started = time.perf_counter()
    with pytest.raises(ModelCallError) as info:
        post(target(server.url, None), b"{}", timeout_s, **kw)
    return info.value, time.perf_counter() - started


def test_trickling_body_hits_the_overall_deadline():
    with FakeModelServer(Answer(body=b"x" * 200, byte_delay_s=0.05)) as server:
        err, took = _fail(server, 0.4)
    assert err.timed_out is True
    assert str(err) == TIMED_OUT
    assert took < 1.2


def test_slow_headers_hit_the_deadline():
    with FakeModelServer(Answer(delay_s=3)) as server:
        err, took = _fail(server, 0.3)
    assert err.timed_out is True
    assert took < 1.5


def test_is_late_boundary():
    assert is_late(0.0, 5.0, 5.0) is True
    assert is_late(0.0, 4.999, 5.0) is False


def _counting_clock(late_at: int | None, late_value: float):
    calls = [0]

    def clock() -> float:
        calls[0] += 1
        return late_value if calls[0] == late_at else 0.0

    return clock, calls


def test_reply_completing_at_the_limit_is_late():
    with FakeModelServer(Answer(body=b"hello")) as server:
        t = target(server.url, None)
        clock, calls = _counting_clock(None, 0.0)
        post(t, b"{}", 5.0, clock=clock)
        n = calls[0]
        clock, _ = _counting_clock(n, 5.0)
        with pytest.raises(ModelCallError) as info:
            post(t, b"{}", 5.0, clock=clock)
        assert info.value.timed_out is True
        clock, _ = _counting_clock(n, 5.0 - 1e-6)
        assert post(t, b"{}", 5.0, clock=clock) == Reply(200, b"hello")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _raise_gaierror(_target, _timeout):
    raise socket.gaierror(-2, "Name or service not known")


def _case_refused(url_user, key):
    port = _free_port()
    return target(f"http://{url_user}127.0.0.1:{port}/v1", key), port, REFUSED, None


def _case_unresolved(url_user, key):
    return target(f"http://{url_user}nohost.invalid:1/v1", key), 1, UNRESOLVED, _raise_gaierror


def _run(case, url_user, key, answer=None):
    if answer is None:
        t, port, message, connect = case(url_user, key)
        return _post_expect(t, port, message, connect)
    with FakeModelServer(answer) as server:
        scheme = "https" if answer is TLS_PROBE else "http"
        t = target(f"{scheme}://{url_user}127.0.0.1:{server.port}/v1", key)
        return _post_expect(t, server.port, TLS if scheme == "https" else BROKEN, None)


TLS_PROBE = Answer()


def _post_expect(t, port, message, connect):
    kw = {} if connect is None else {"connect": connect}
    with pytest.raises(ModelCallError) as info:
        post(t, b"{}", 5.0, **kw)
    assert str(info.value) == message
    assert info.value.timed_out is False
    return str(info.value), port


def test_refused():
    _run(_case_refused, "", None)


def test_host_name_not_resolved():
    _run(_case_unresolved, "", None)


def test_tls_failure_against_plain_server():
    _run(None, "", None, TLS_PROBE)


def test_closed_mid_body():
    _run(None, "", None, Answer(body=b"x" * 100, close_after=10))


def test_broken_status_line():
    _run(None, "", None, Answer(raw=b"HELLO\r\n\r\n"))


def test_redirect_is_not_followed():
    with FakeModelServer(Answer(body=b"b")) as other:
        location = (("Location", other.url),)
        with FakeModelServer(Answer(302, b"moved", headers=location)) as server:
            reply = post(target(server.url, None), b"{}", 5.0)
    assert reply == Reply(302, b"moved")
    assert other.seen == []


@pytest.mark.parametrize("which", ["refused", "unresolved", "tls", "closed"])
def test_messages_leak_nothing(which):
    user = "u:SENTINEL-2@"
    key = "sk-SENTINEL-1"
    if which == "refused":
        message, port = _run(_case_refused, user, key)
    elif which == "unresolved":
        message, port = _run(_case_unresolved, user, key)
    elif which == "tls":
        message, port = _run(None, user, key, TLS_PROBE)
    else:
        message, port = _run(None, user, key, Answer(body=b"x" * 100, close_after=10))
    for secret in ("SENTINEL", "127.0.0.1", str(port), "Authorization"):
        assert secret not in message
    assert len(message.splitlines()) == 1


def test_open_connection_picks_scheme():
    assert type(open_connection(target("http://h:1/v1", None), 1.0)).__name__ == "HTTPConnection"
    assert type(open_connection(target("https://h:1/v1", None), 1.0)).__name__ == "HTTPSConnection"


def test_native_target_paths():
    from folio_ocr.model_http import native_target

    assert native_target("http://h:1/v1", None, "tags").path == "/api/tags"
    assert native_target("http://h:1/v1/", None, "tags").path == "/api/tags"
    assert native_target("http://h/proxy/v1?k=1", None, "tags").path == "/proxy/api/tags?k=1"
    assert native_target("http://h:1/v2", None, "tags") is None
    assert native_target("http://h:1", None, "tags") is None


def test_native_target_headers_and_host():
    from folio_ocr.model_http import native_target

    plain = native_target("http://h:1/v1", None, "version")
    assert plain.headers == ()
    assert plain.hostport == "h:1"
    keyed = native_target("http://h:1/v1", "sk-1", "version")
    assert keyed.headers == (("Authorization", b"Bearer sk-1"),)
    user = native_target("http://u:p@h:1/v1", None, "version")
    assert user.headers == (("Authorization", _basic("u:p")),)
    assert user.hostport == "h:1"


def test_get_sends_no_body_and_returns_the_reply():
    from folio_ocr.model_http import get, native_target

    with FakeModelServer(Answer(status=404, body=b'{"x":1}')) as server:
        reply = get(native_target(server.url, None, "tags"), 5.0)
    assert reply == Reply(404, b'{"x":1}')
    (seen,) = server.seen
    assert seen.method == "GET"
    assert seen.target == "/api/tags"
    assert seen.body == b""
    assert "content-length" not in seen.headers
    assert "content-type" not in seen.headers


def test_get_hits_the_overall_deadline():
    from folio_ocr.model_http import get, native_target

    slow = Answer(body=b"x" * 200, byte_delay_s=0.05)
    with FakeModelServer(slow) as server, pytest.raises(ModelCallError) as info:
        get(native_target(server.url, None, "tags"), 0.4)
    assert info.value.timed_out is True
    assert str(info.value) == TIMED_OUT


def _post_timed(answer, timeout_s):
    started = time.perf_counter()
    with FakeModelServer(answer) as server, pytest.raises(ModelCallError) as caught:
        post(target(server.url, None), b"{}", timeout_s)
    return caught.value, time.perf_counter() - started


_TRICKLED_HEADERS = b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * 400 + b"\r\n\r\n"


def test_trickled_headers_are_cut_by_the_timer():
    answer = Answer(raw=b"", trickle=_TRICKLED_HEADERS, byte_delay_s=0.02)
    err, elapsed = _post_timed(answer, 0.4)
    assert err.timed_out is True
    assert str(err) == TIMED_OUT
    assert elapsed < 1.5


def test_trickled_chunk_size_of_a_will_close_reply_is_cut_by_the_timer():
    head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    answer = Answer(raw=head, trickle=b"1;" + b"x" * 400, byte_delay_s=0.02)
    err, elapsed = _post_timed(answer, 0.4)
    assert err.timed_out is True
    assert elapsed < 1.5


def test_without_the_timer_the_trickle_ends_late_but_still_times_out(monkeypatch):
    started = []

    class NoTimer:
        def __init__(self, *args, **kwargs):
            self.daemon = False

        def start(self):
            started.append(True)

        def cancel(self):
            pass

    monkeypatch.setattr(model_http.threading, "Timer", NoTimer)
    answer = Answer(
        raw=b"",
        trickle=b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * 60 + b"\r\n\r\n" + b"",
        byte_delay_s=0.02,
    )
    err, elapsed = _post_timed(answer, 0.3)
    assert started == [True]
    assert err.timed_out is True
    assert elapsed > 1.0


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _three_addresses(monkeypatch, port):
    entry = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
    monkeypatch.setattr(model_http.socket, "getaddrinfo", lambda *a, **k: [entry] * 3)


def _recording_sockets(monkeypatch):
    made = []
    real = socket.socket

    def make(*args, **kwargs):
        sock = real(*args, **kwargs)
        made.append(sock)
        return sock

    monkeypatch.setattr(model_http.socket, "socket", make)
    return made


def test_every_address_is_tried_and_the_refusal_is_reported(monkeypatch):
    port = _closed_port()
    _three_addresses(monkeypatch, port)
    made = _recording_sockets(monkeypatch)
    create = model_http._address_by_address(lambda: 5.0, model_http._Watched())
    with pytest.raises(ConnectionRefusedError):
        create(("127.0.0.1", port))
    assert len(made) == 3
    assert all(s.fileno() == -1 for s in made)
    with pytest.raises(ModelCallError) as caught:
        post(target(f"http://127.0.0.1:{port}/v1", None), b"{}", 30.0)
    assert str(caught.value) == REFUSED
    assert len(made) == 6


def test_time_running_out_between_addresses_stops_and_closes_every_socket(monkeypatch):
    port = _closed_port()
    _three_addresses(monkeypatch, port)
    made = _recording_sockets(monkeypatch)
    calls = []

    def left():
        calls.append(1)
        if len(calls) == 2:
            raise model_http._Late
        return 5.0

    create = model_http._address_by_address(left, model_http._Watched())
    with pytest.raises(model_http._Late):
        create(("127.0.0.1", port))
    assert len(made) == 2
    assert all(s.fileno() == -1 for s in made)


def test_a_deep_body_is_returned_untouched():
    deep = b"[" * 6000 + b"]" * 6000
    with FakeModelServer(Answer(body=deep)) as server:
        reply = post(target(server.url, None), b"{}", 5.0)
    assert reply == Reply(200, deep)


@pytest.mark.parametrize("limit", [5e6, 1e13])
def test_a_huge_limit_cannot_crash_the_timer(monkeypatch, limit):
    errors = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    with FakeModelServer(Answer(body=b"hello")) as server:
        reply = post(target(server.url, None), b"{}", limit)
    assert reply == Reply(200, b"hello")
    assert errors == []


def test_the_limit_handed_to_connect_never_exceeds_the_cap():
    handed = []

    def connect(tgt, timeout_s):
        handed.append(timeout_s)
        return open_connection(tgt, timeout_s)

    with FakeModelServer(Answer(body=b"hello")) as server:
        post(target(server.url, None), b"{}", 1e13, connect=connect)
    assert len(handed) == 1
    assert model_http.MAX_WAIT_S - 5 < handed[0] <= model_http.MAX_WAIT_S


def test_time_running_out_after_the_first_address_stops_the_other_addresses(monkeypatch):
    port = _closed_port()
    _three_addresses(monkeypatch, port)
    made = _recording_sockets(monkeypatch)
    ticks = []

    def clock():
        ticks.append(1)
        return 0.0 if len(ticks) <= 3 else 100.0

    started = time.perf_counter()
    with pytest.raises(ModelCallError) as caught:
        post(target(f"http://127.0.0.1:{port}/v1", None), b"{}", 30.0, clock=clock)
    assert time.perf_counter() - started < 15
    assert caught.value.timed_out is True
    assert str(caught.value) == TIMED_OUT
    assert 1 <= len(made) < 3


def test_an_address_whose_socket_cannot_be_made_is_skipped(monkeypatch):
    real = socket.getaddrinfo
    bogus = (socket.AF_INET, socket.SOCK_STREAM, 9999, "", ("127.0.0.1", 1))
    with FakeModelServer(Answer(body=b"{}")) as fake:

        def two_addresses(host, port, *args, **kwargs):
            return [bogus, *real(host, port, socket.AF_INET, socket.SOCK_STREAM)]

        monkeypatch.setattr(model_http.socket, "getaddrinfo", two_addresses)
        reply = model_http.post(model_http.target(fake.url, None), b"{}", 5.0)
    assert reply.status == 200
