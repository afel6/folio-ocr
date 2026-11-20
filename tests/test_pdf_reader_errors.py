"""How a reader process failure reaches the HTTP API, the command line and the doctor."""

import dataclasses
import os

import pytest
from PIL import Image
from sample_files import pdf_bytes, save
from starlette.testclient import TestClient
from test_web_api import ENGINE, make_page

from folio_ocr import cli, doctor, ingest, pdf_client, pipeline
from folio_ocr.config import Settings
from folio_ocr.errors import InputError, PdfReaderError
from folio_ocr.web import upload
from folio_ocr.web.app import create_app

PDF_503 = "PDFs cannot be read here: folio could not start its PDF reader with limits"
PNG = save(Image.new("L", (10, 10)), "PNG")
PNG_503 = "images cannot be read here: folio could not start its image reader with limits"
LIMIT = pdf_client.limit_line("pdf")


class Reader:
    """Stands in for read_page and page_image: raises `error` when set."""

    def __init__(self):
        self.error = None

    def read_page(self, data, no, **kw):
        if self.error:
            raise self.error
        return dataclasses.replace(make_page(no)), ENGINE

    def page_image(self, data, no):
        if self.error:
            raise self.error
        return Image.new("L", (40, 20))


@pytest.fixture
def reader():
    return Reader()


@pytest.fixture
def client(tmp_path, reader):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>home</p>")
    app = create_app(
        Settings(weights_dir=tmp_path), port=8000, tesseract=None, model=None,
        read_page=reader.read_page, page_image=reader.page_image,
        static_dir=static, log=lambda line: None,
    )  # fmt: skip
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        yield c


def post_file(client):
    return client.post(
        "/api/documents",
        files={"file": ("a.png", PNG, "application/x")},
        data={"languages": "en"},
    )


def refuse_open(error):
    def opened(data, kind):
        raise error

    return opened


@pytest.fixture
def doc_id(client, monkeypatch):
    monkeypatch.setattr(upload, "_page_count", lambda data, kind: 2)
    return post_file(client).json()["document_id"]


def statuses(client):
    (doc,) = client.get("/api/documents").json()
    return [p["status"] for p in doc["pages"]]


def test_read_route_answers_503_with_the_exact_line(client, reader, doc_id):
    reader.error = PdfReaderError("pdf", "x")
    reply = client.post(f"/api/documents/{doc_id}/pages/1/read")
    assert (reply.status_code, reply.json()) == (503, {"error": PDF_503})


def test_image_route_answers_503_with_the_exact_line(client, reader, doc_id):
    reader.error = PdfReaderError("pdf", "x")
    reply = client.get(f"/api/documents/{doc_id}/pages/1/image")
    assert (reply.status_code, reply.json()) == (503, {"error": PDF_503})


def test_upload_answers_503_and_stores_nothing(client, monkeypatch):
    monkeypatch.setattr(pdf_client, "opened", refuse_open(PdfReaderError("pdf", "x")))
    reply = post_file(client)
    assert (reply.status_code, reply.json()) == (503, {"error": PDF_503})
    assert client.get("/api/documents").json() == []


def test_upload_of_a_png_answers_503_and_stores_nothing(client, monkeypatch):
    monkeypatch.setattr(pdf_client, "opened", refuse_open(PdfReaderError("png", "x")))
    reply = post_file(client)
    assert (reply.status_code, reply.json()) == (503, {"error": PNG_503})
    assert client.get("/api/documents").json() == []


def test_upload_that_breaks_a_limit_answers_400_and_stores_nothing(client, monkeypatch):
    monkeypatch.setattr(pdf_client, "opened", refuse_open(InputError(LIMIT)))
    reply = post_file(client)
    assert (reply.status_code, reply.json()) == (400, {"error": LIMIT})
    assert client.get("/api/documents").json() == []


def test_read_that_breaks_a_limit_answers_400_and_keeps_the_page_status(client, reader, doc_id):
    before = statuses(client)
    reader.error = InputError(LIMIT)
    reply = client.post(f"/api/documents/{doc_id}/pages/1/read")
    assert (reply.status_code, reply.json()) == (400, {"error": LIMIT})
    assert statuses(client) == before == ["unread", "unread"]


def test_image_route_that_breaks_a_limit_answers_400(client, reader, doc_id):
    reader.error = InputError(LIMIT)
    reply = client.get(f"/api/documents/{doc_id}/pages/1/image")
    assert (reply.status_code, reply.json()) == (400, {"error": LIMIT})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def test_cli_read_shows_the_file_and_the_line(monkeypatch, capsys):
    def fail(path, **kw):
        raise PdfReaderError("pdf", "x")

    monkeypatch.setattr(pipeline, "read_path", fail)
    code = cli.main(["read", "file.pdf"], {})
    err = capsys.readouterr().err
    assert code == 1
    assert err == f"error: file.pdf: {PDF_503}" + chr(10)


def test_cli_read_of_a_png_shows_the_image_line(monkeypatch, capsys):
    def fail(path, **kw):
        raise PdfReaderError("png", "x")

    monkeypatch.setattr(pipeline, "read_path", fail)
    code = cli.main(["read", "file.png"], {})
    assert code == 1
    assert capsys.readouterr().err == f"error: file.png: {PNG_503}" + chr(10)


def no_ocr_readers(monkeypatch):
    monkeypatch.setattr(doctor.ocr_setup, "onnxtr_status", lambda *_a, **_k: "ok")
    monkeypatch.setattr(doctor.ocr_setup, "tesseract_status", lambda *_a, **_k: "ok")


def test_doctor_fails_the_pdf_reader_check(monkeypatch, tmp_path):
    no_ocr_readers(monkeypatch)

    def fail():
        raise PdfReaderError("pdf", "the reader process could not be started")

    monkeypatch.setattr(pdf_client, "self_check", fail)
    lines = []
    settings = Settings(weights_dir=tmp_path / "assets")
    code = doctor.run(settings, lines.append, environ={"PATH": ""}, assets=())
    detail = "the reader process could not be started; PDFs and images cannot be read"
    check = doctor._pdf_reader()
    assert (check.name, check.status, check.detail) == ("pdf reader", "fail", detail)
    assert code == 1
    assert any(line.startswith("fail") and detail in line for line in lines)


def test_doctor_real_pdf_reader_check_is_ok_and_names_the_limits():
    check = doctor._pdf_reader()
    assert (check.name, check.status) == ("pdf reader", "ok")
    assert f"{ingest.PDF_MEMORY_MB} MB and {ingest.PDF_SECONDS} s per file" in check.detail
    assert pdf_client.live_workers() == 0


def test_real_upload_of_a_pdf_stores_a_document(client):
    reply = client.post(
        "/api/documents",
        files={"file": ("a.pdf", pdf_bytes([(612, 792)] * 2), "application/pdf")},
        data={"languages": "en"},
    )
    assert reply.status_code == 200 and reply.json()["page_count"] == 2
