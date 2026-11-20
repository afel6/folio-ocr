"""The review page as folio serves it: every file loads under the strict content policy."""

import re
from pathlib import Path

import pytest
from starlette import responses
from starlette.testclient import TestClient

from folio_ocr.config import Settings
from folio_ocr.web.app import create_app

STATIC = Path(__file__).resolve().parents[1] / "src" / "folio_ocr" / "web" / "static"
PORT = 8000
CSP = "default-src 'self'"
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(weights_dir=tmp_path_factory.mktemp("weights"))
    app = create_app(settings, port=PORT, tesseract=None, model=None)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        yield test_client


def _static_names():
    return sorted(p.name for p in STATIC.iterdir() if p.is_file())


def test_index_is_served_at_root_with_the_content_policy(client):
    reply = client.get("/")
    assert reply.status_code == 200
    assert reply.headers["content-security-policy"] == CSP
    assert reply.headers["x-content-type-options"] == "nosniff"
    assert reply.headers["content-type"].split(";")[0] == "text/html"
    assert '<script type="module" src="/app.js"></script>' in reply.text


@pytest.mark.parametrize("name", _static_names())
def test_every_static_file_is_served_with_a_type_the_browser_runs(client, name):
    reply = client.get(f"/{name}")
    assert reply.status_code == 200
    assert reply.headers["content-security-policy"] == CSP
    expected = TYPES.get(Path(name).suffix)
    if expected:
        assert reply.headers["content-type"].split(";")[0] == expected


def test_javascript_type_is_independent_of_the_host_mime_database(client, monkeypatch):
    monkeypatch.setattr(responses, "guess_type", lambda _: ("application/javascript", None))
    reply = client.get("/app.js")
    assert reply.status_code == 200
    assert reply.headers["content-type"].split(";")[0] == "text/javascript"
    assert reply.headers["x-content-type-options"] == "nosniff"


def test_every_module_the_page_imports_is_served(client):
    pending, seen = ["app.js"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        reply = client.get(f"/{name}")
        assert reply.status_code == 200, name
        pending += re.findall(r"from '\./([\w.-]+\.js)'", reply.text)
    assert {"app.js", "api.js", "copy.js", "logic.js", "queue.js", "dom.js"} <= seen
