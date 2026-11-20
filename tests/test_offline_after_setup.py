"""After `folio setup` has filled a fresh folder, reading works with the network blocked."""

import json
import os
import socket
from pathlib import Path

import httpx
import pytest

from folio_ocr import assets, cli
from folio_ocr.config import Settings
from folio_ocr.export import to_json, to_txt
from folio_ocr.pipeline import read_path

FIXTURES = Path(__file__).parent / "fixtures"
WEIGHTS = Settings.from_env(os.environ).weights_dir

if assets.files_problem(assets.MANIFEST, WEIGHTS) is not None:
    pytest.skip("the model files are not set up (folio setup)", allow_module_level=True)

HOST = "https://objects.example/"


def mirror(seen):
    by_url = {a.url: a for a in assets.MANIFEST}

    def handler(request):
        url = str(request.url)
        seen.append(url)
        if url in by_url:
            return httpx.Response(302, headers={"location": HOST + by_url[url].name})
        if url.startswith(HOST):
            return httpx.Response(200, content=(WEIGHTS / url[len(HOST) :]).read_bytes())
        return httpx.Response(404)

    return handler


def block_sockets(monkeypatch):
    attempts = []

    def blocker(name):
        def refuse(*args, **kwargs):
            attempts.append((name, args))
            raise OSError("network blocked")

        return refuse

    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, blocker(name))
    return attempts


def test_english_read_works_offline_from_a_freshly_set_up_folder(capsys, monkeypatch, tmp_path):
    seen = []
    real_fetch = assets.fetch
    transport = httpx.MockTransport(mirror(seen))
    monkeypatch.setattr(assets, "fetch", lambda *a, **k: real_fetch(*a, transport=transport, **k))
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "assets")}
    assert cli.main(["setup"], env) == 0
    assert all(s.ok for s in assets.verify(assets.MANIFEST, tmp_path / "assets"))
    assert seen == [u for a in assets.MANIFEST for u in (a.url, HOST + a.name)]
    capsys.readouterr()

    attempts = block_sockets(monkeypatch)
    image = FIXTURES / "en-letter-scan.jpg"
    code = cli.main(["read", "--lang", "en", str(image)], env)
    captured = capsys.readouterr()
    assert code == 0
    doc = read_path(image, languages="en", weights_dir=tmp_path / "assets")
    pages = json.loads(to_json(doc))["pages"]  # lines needing review depend on the tau table
    review = sum(x["status"] == "needs_review" for p in pages for x in p["lines"])
    assert captured.err == (
        f"{image}: 14 lines found, 14 read, {review} need review, 0 unread pages\n"
    )
    assert captured.out == to_txt(doc)
    assert attempts == []
