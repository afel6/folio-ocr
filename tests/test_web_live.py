"""The real app on a real uvicorn server: what the guard and the 500 answer look like on the wire."""

import http.client
import json
import socket
import threading
import time

import pytest
from PIL import Image
from sample_files import save

from folio_ocr.config import Settings
from folio_ocr.web import server
from folio_ocr.web.app import create_app

CSP = "default-src 'self'"
UNEXPECTED = {"error": "unexpected failure; please report it"}
PNG = save(Image.new("L", (10, 10)), "PNG")


def failing_read(data, no, **kw):
    raise RuntimeError("SENTINEL-LINE-TEXT")


@pytest.fixture(scope="module")
def port(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("live")
    static = tmp / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>home</p>")
    sock = server.bind("127.0.0.1", 0)
    number = sock.getsockname()[1]
    app = create_app(Settings(weights_dir=tmp), port=number, tesseract=None, model=None,
                     read_page=failing_read, static_dir=static)  # fmt: skip
    srv = server.build_server(app, sock)
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield number
    srv.should_exit = True
    thread.join(10)
    sock.close()
    assert not thread.is_alive()


def ask(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        return r.status, r, r.read()
    finally:
        conn.close()


def raw(port, request: bytes):
    """Status line, content-type values, whole head and body of the reply to raw request bytes."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
        data = b""
        try:  # the server may close while a large request is still being sent
            s.sendall(request)
            while chunk := s.recv(65536):
                data += chunk
        except ConnectionError:
            pass
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    types = [x.split(":", 1)[1].strip() for x in lines[1:] if x.lower().startswith("content-type")]
    return lines[0], types, body, head


def test_health_has_csp(port):
    status, r, _ = ask(port, "GET", "/api/health")
    assert status == 200 and r.getheader("content-security-policy") == CSP


def test_unexpected_failure_over_the_wire(port, capfd):
    capfd.readouterr()
    multipart = (b'--B\r\nContent-Disposition: form-data; name="languages"\r\n\r\nen\r\n'
                 b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n\r\n'
                 + PNG + b"\r\n--B--\r\n")  # fmt: skip
    status, r, body = ask(port, "POST", "/api/documents", multipart,
                          {"Content-Type": "multipart/form-data; boundary=B"})  # fmt: skip
    doc = json.loads(body)["document_id"]
    status, r, body = ask(port, "POST", f"/api/documents/{doc}/pages/1/read")
    assert status == 500 and json.loads(body) == UNEXPECTED
    assert r.getheader("content-security-policy") == CSP
    err = capfd.readouterr().err
    assert len(err.splitlines()) == 1 and "RuntimeError" in err
    assert "Traceback" not in err and "SENTINEL" not in err


def test_wrong_host_is_400_json(port):
    status, r, body = ask(port, "GET", "/api/health", headers={"Host": f"127.0.0.1:{port + 1}"})
    assert status == 400 and list(json.loads(body)) == ["error"]
    assert r.getheader("content-type") == "application/json"


def test_http_1_0_without_host_is_400_json(port):
    status, types, body, _ = raw(port, b"GET /api/health HTTP/1.0\r\n\r\n")
    assert status.startswith("HTTP/1.1 400") and types == ["application/json"]
    assert list(json.loads(body)) == ["error"]


def assert_unreadable_json(port, request: bytes):
    status, types, body, head = raw(port, request)
    head = head.lower()
    assert status == "HTTP/1.1 400 Bad Request" and types == ["application/json"]
    assert json.loads(body) == {"error": "the request could not be read"}
    assert b"content-security-policy: default-src 'self'" in head
    assert b"connection: close" in head
    assert b"Invalid HTTP request received." not in body


def test_http_1_1_without_host_gets_the_json_error(port):
    assert_unreadable_json(port, b"GET /api/health HTTP/1.1\r\n\r\n")


def test_two_host_headers_get_the_json_error(port):
    h = f"Host: 127.0.0.1:{port}\r\n"
    assert_unreadable_json(port, f"GET /api/health HTTP/1.1\r\n{h}{h}\r\n".encode())


def test_oversized_header_gets_the_json_error(port):
    h = f"Host: 127.0.0.1:{port}\r\nX-Big: {'a' * 200000}\r\n"
    assert_unreadable_json(port, f"GET /api/health HTTP/1.1\r\n{h}\r\n".encode())
