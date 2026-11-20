"""The HTTP API of the local web service, through the real app with a fake page reader."""

import dataclasses
import hashlib
import inspect
import io
import json
import re
import threading
from pathlib import Path

import docx
import pytest
from docx.enum.text import WD_COLOR_INDEX
from PIL import Image
from sample_files import OVERSIZED_IMAGE_CASES as BIG_IMAGE_CASES
from sample_files import encrypted, pdf_bytes, save
from starlette.testclient import TestClient

from folio_ocr import assets, export, ocr_setup
from folio_ocr.config import Settings
from folio_ocr.document import IDENTITY, Engine, Line, Page
from folio_ocr.errors import (
    ConfigError,
    InputError,
    ModelServerError,
    SetupNeededError,
    WeightsError,
)
from folio_ocr.web.app import BAD_BODY, create_app
from folio_ocr.web.store import BUSY

ENGINE = Engine("det", "a" * 64, "onnxtr", None, None, "untested", None)
PNG = save(Image.new("L", (10, 10)), "PNG")
MAX_BYTES = 26_214_400
CSP = "default-src 'self'"
UNEXPECTED = "unexpected failure; please report it"


def make_line(n, k, tag, status="needs_review"):
    text = f"{tag}{n}-{k}"
    return Line(f"p{n}-l{k}", (0, 0, 5, 5), text, "ltr", "latin", "onnxtr", 0.5, "reader", (),
                status, text)  # fmt: skip


def make_page(n, tag="old", rotation=0.0, size=(40, 20)):
    lines = (make_line(n, 1, tag), make_line(n, 2, tag, "auto_accepted"))
    return Page(n, size[0], size[1], 300, rotation, IDENTITY, "ocr", "ltr", "read", (), (),
                lines)  # fmt: skip


class FakeReader:
    """Stands in for read_page: waits on `release` when `gate` is set, raises `error` if set."""

    def __init__(self):
        self.tag, self.error, self.rotation, self.gate = "old", None, 0.0, False
        self.regions = ()  # unread areas of the page it returns
        self.started, self.release = threading.Event(), threading.Event()

    def __call__(self, data, no, **kw):
        if self.gate:
            self.started.set()
            assert self.release.wait(10)
        if self.error:
            raise self.error
        size = (20, 40) if self.rotation else (40, 20)
        page = make_page(no, self.tag, self.rotation, size)
        return dataclasses.replace(page, unread_regions=self.regions), ENGINE


def fake_image(data, no):
    return Image.new("L", (40, 20))


@pytest.fixture
def reader():
    return FakeReader()


@pytest.fixture
def logs():
    return []


@pytest.fixture
def static(tmp_path):
    d = tmp_path / "static"
    d.mkdir()
    (d / "index.html").write_text("<p>home</p>")
    (d / "app.js").write_text("let x = 1;")
    return d


def make_client(tmp_path, reader, static, logs, **kw):
    app = create_app(
        kw.pop("settings", Settings(weights_dir=tmp_path)), port=8000, tesseract=None,
        model=kw.pop("model", None), read_page=reader, page_image=fake_image,
        static_dir=static, log=logs.append,
    )  # fmt: skip
    return TestClient(app, base_url="http://127.0.0.1:8000", **kw)


@pytest.fixture
def c(tmp_path, reader, static, logs):
    with make_client(tmp_path, reader, static, logs) as client:
        yield client


def upload(c, data=PNG, name="a.png", languages="en"):
    form = {"languages": languages} if languages else None
    return c.post("/api/documents", files={"file": (name, data, "application/x")}, data=form)


def new_doc(c, **kw):
    return upload(c, **kw).json()["document_id"]


def read(c, doc, n=1):
    return c.post(f"/api/documents/{doc}/pages/{n}/read")


def exported(c, doc):
    return c.get(f"/api/documents/{doc}/export", params={"format": "json"}).json()


def edit(c, doc, line, text):
    return c.put(f"/api/documents/{doc}/lines/{line}", json={"final_text": text})


def in_thread(fn, *args):
    out = []
    t = threading.Thread(target=lambda: out.append(fn(*args)))
    t.start()
    return t, out


def slow_read(c, reader, doc, n=1):
    """Start a read that stays running until `reader.release` is set."""
    reader.gate = True
    t, out = in_thread(read, c, doc, n)
    assert reader.started.wait(10)
    return t, out


def finish(reader, t, out):
    reader.release.set()
    t.join(10)
    return out[0]


# 1. upload


def test_upload_returns_id_and_page_count(c):
    r = upload(c)
    assert r.status_code == 200
    assert set(r.json()) == {"document_id", "page_count"} and r.json()["page_count"] == 1


def test_upload_size_limit_is_exact(c):
    data = PNG + b"\0" * (MAX_BYTES - len(PNG))
    assert upload(c, data).status_code == 200
    r = upload(c, data + b"\0")
    assert r.status_code == 400 and r.json() == {"error": "the file is larger than 25 MB"}


@pytest.mark.parametrize(("pages", "code"), [(20, 200), (21, 400)])
def test_upload_page_limit(c, pages, code):
    r = upload(c, pdf_bytes([(100, 100)] * pages), "a.pdf")
    assert r.status_code == code
    if code == 200:
        assert r.json()["page_count"] == 20


@pytest.mark.parametrize(
    "data", [pdf_bytes([(14400, 14400)]), encrypted(), b"not a file at all"],
    ids=["huge page", "encrypted", "unknown type"],
)  # fmt: skip
def test_upload_refuses_bad_files(c, data):
    r = upload(c, data, "a.pdf")
    assert r.status_code == 400 and list(r.json()) == ["error"]


def test_upload_needs_languages(c):
    assert upload(c, languages=None).status_code == 400


# 2. unread and reading placeholders


def test_new_document_pages_are_unread(c):
    doc = exported(c, new_doc(c, data=pdf_bytes([(100, 100)] * 3), name="a.pdf"))
    assert [(p["status"], p["lines"]) for p in doc["pages"]] == [("unread", [])] * 3


def test_page_shows_reading_during_a_read(c, reader):
    doc = new_doc(c)
    t, out = slow_read(c, reader, doc)
    assert exported(c, doc)["pages"][0]["status"] == "reading"
    assert c.get("/api/documents").json()[0]["pages"] == [{"index": 1, "status": "reading"}]
    assert finish(reader, t, out).status_code == 200


# 3. listing


def test_list_documents(c):
    first = new_doc(c, name="one.png", languages="ar")
    second = new_doc(c, name="two.png")
    read(c, first)
    rows = c.get("/api/documents").json()
    assert [r["document_id"] for r in rows] == [first, second]
    assert set(rows[0]) == {"document_id", "filename", "page_count", "languages", "created_at",
                            "has_edits", "unresolved_count", "pages"}  # fmt: skip
    assert rows[0]["filename"] == "one.png" and rows[0]["languages"] == "ar"
    assert rows[0]["unresolved_count"] == 1 and rows[0]["has_edits"] is False
    assert rows[0]["pages"] == [{"index": 1, "status": "read"}]
    assert rows[1]["unresolved_count"] == 0 and rows[1]["pages"][0]["status"] == "unread"
    assert re.match(r"\d{4}-\d\d-\d\dT", rows[0]["created_at"])


# 4. reading


def test_read_returns_the_page(c):
    doc = new_doc(c)
    r = read(c, doc)
    assert r.json() == json.loads(json.dumps(dataclasses.asdict(make_page(1))))


def test_second_read_while_one_runs_is_busy(c, reader):
    doc, other = new_doc(c), new_doc(c)
    t, out = slow_read(c, reader, doc)
    for target in (doc, other):
        r = read(c, target)
        assert r.status_code == 409 and r.json() == {"error": BUSY}
    assert finish(reader, t, out).status_code == 200


def test_edit_survives_a_re_read(c, reader):
    doc = new_doc(c)
    read(c, doc)
    reader.tag = "new"
    t, out = slow_read(c, reader, doc)
    assert edit(c, doc, "p1-l1", "edited text").status_code == 200
    reply = finish(reader, t, out).json()
    for lines in (reply["lines"], exported(c, doc)["pages"][0]["lines"]):
        first, second = lines
        assert (first["final_text"], first["status"]) == ("edited text", "edited")
        assert second["text"] == "new1-2"


# 5. weights errors


@pytest.mark.parametrize(
    "err",
    [
        SetupNeededError("the OCR model files are missing; run folio setup"),
        SetupNeededError("the OCR model files are damaged; run folio setup"),
        WeightsError("the OCR model files cannot be read; check the access rights of the assets folder"),
    ],
)  # fmt: skip
def test_weights_errors_are_503_and_the_page_is_released(c, reader, err):
    doc = new_doc(c)
    reader.error = err
    r = read(c, doc)
    assert r.status_code == 503 and r.json() == {"error": str(err)}
    assert exported(c, doc)["pages"][0]["status"] == "unread"
    reader.error = None
    assert read(c, doc).status_code == 200


# 6. model server errors


def test_model_server_error_is_502(c, reader):
    reader.error = ModelServerError("the model server failed")
    r = read(c, new_doc(c))
    assert r.status_code == 502 and r.json() == {"error": "the model server failed"}


def test_no_module_raises_model_server_error_yet():
    src = Path(__file__).parent.parent / "src" / "folio_ocr"
    found = [
        str(p) for p in src.rglob("*.py") if p.name != "errors.py"
        and re.search(r"raise ModelServerError|ModelServerError\(", p.read_text(encoding="utf-8"))
    ]  # fmt: skip
    assert found == []


# 7. error bodies


def _validation(c):
    return c.put("/api/documents/x/lines/y", json={"final_text": 5})


def _not_json(c):
    return c.put(
        "/api/documents/x/lines/y", content=b"{nope", headers={"content-type": "application/json"}
    )


def _unexpected(c):
    return c.post(f"/api/documents/{new_doc(c)}/pages/1/read")


CASES = [
    (_validation, 400), (_not_json, 400),
    (lambda c: c.get("/api/nope"), 404),
    (lambda c: c.post("/api/health"), 405),
    (lambda c: c.get("/api/health", headers={"host": "127.0.0.1:8001"}), 400),
    (lambda c: c.delete("/api/documents/x", headers={"origin": "http://evil.example"}), 403),
    (_unexpected, 500),
]  # fmt: skip


@pytest.mark.parametrize(("send", "status"), CASES)
def test_every_error_body_is_one_line(c, reader, send, status):
    reader.error = RuntimeError("boom")
    r = send(c)
    body = r.json()
    assert r.status_code == status and list(body) == ["error"]
    assert body["error"] and body["error"].splitlines() == [body["error"]]
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["content-security-policy"] == CSP


def test_unexpected_failure_is_logged_once_without_its_message(c, reader, logs):
    reader.error = RuntimeError("SENTINEL-LINE-TEXT")
    r = read(c, new_doc(c))
    assert r.json() == {"error": UNEXPECTED}
    assert len(logs) == 1 and "RuntimeError" in logs[0] and "SENTINEL" not in logs[0]
    assert read(c, new_doc(c)).status_code == 500  # the page was released: no 409


def test_unexpected_failure_goes_to_stderr_by_default(tmp_path, reader, static, capsys):
    app = create_app(Settings(weights_dir=tmp_path), port=8000, tesseract=None, model=None,
                     read_page=reader, page_image=fake_image, static_dir=static)  # fmt: skip
    reader.error = RuntimeError("SENTINEL-LINE-TEXT")
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        assert read(client, new_doc(client)).status_code == 500
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and "RuntimeError" in lines[0] and "SENTINEL" not in lines[0]


def test_other_folio_errors_are_a_friendly_500(c, reader, logs):
    reader.error = ConfigError("x")
    r = read(c, new_doc(c))
    assert r.status_code == 500 and r.json() == {"error": UNEXPECTED}
    assert len(logs) == 1 and "ConfigError" in logs[0] and "x)" not in logs[0]


def test_image_route_answers_a_damaged_page_with_400(tmp_path, reader, static, logs):
    text = "page 1 of the PDF could not be read; it may be damaged"

    def damaged(data, no):
        raise InputError(text)

    app = create_app(Settings(weights_dir=tmp_path), port=8000, tesseract=None, model=None,
                     read_page=reader, page_image=damaged, static_dir=static, log=logs.append)  # fmt: skip
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        r = client.get(f"/api/documents/{new_doc(client)}/pages/1/image")
    assert r.status_code == 400 and r.json() == {"error": text}


def allow(r):
    return sorted(x.strip() for x in r.headers["allow"].split(","))


@pytest.mark.parametrize(
    ("path", "methods"),
    [
        ("/api/documents/x/pages/1/read", ["POST"]),
        ("/api/documents/x/lines/y", ["PUT"]),
        ("/api/documents/x", ["DELETE"]),
        ("/api/health", ["GET"]),
    ],
)
def test_wrong_method_on_an_api_path_is_405(c, path, methods):
    r = c.get(path) if path != "/api/health" else c.post(path)
    assert r.status_code == 405 and r.json() == {"error": "method not allowed"}
    assert allow(r) == methods


def test_delete_on_the_collection_is_405_listing_get_and_post(c):
    r = c.delete("/api/documents")
    assert r.status_code == 405 and allow(r) == ["GET", "POST"]


@pytest.mark.parametrize("path", ["/api/nope", "/api"])
def test_unknown_api_path_is_404(c, path):
    r = c.get(path)
    assert r.status_code == 404 and r.json() == {"error": "not found"}


def test_pages_are_still_served(c):
    assert c.get("/").text == "<p>home</p>" and c.get("/app.js").text == "let x = 1;"


@pytest.mark.parametrize(
    ("ctype", "status"),
    [
        ("text/plain", 400),
        ("application/merge-patch+json", 400),
        ("application/json; charset=utf-8", 200),
        ("Application/JSON", 200),
    ],
)
def test_edit_content_type_must_be_json(c, ctype, status):
    doc = new_doc(c)
    read(c, doc)
    r = c.put(f"/api/documents/{doc}/lines/p1-l1", content=b'{"final_text": "z"}',
              headers={"content-type": ctype})  # fmt: skip
    assert r.status_code == status
    if status == 400:
        assert r.json() == {"error": BAD_BODY}


# 8. headers


def test_no_cors_headers_anywhere(c):
    doc = new_doc(c)
    read(c, doc)
    replies = [
        upload(c), read(c, doc), c.get("/api/health"),
        c.get(f"/api/documents/{doc}/export?format=txt"),
        c.options("/api/documents", headers={"origin": "http://127.0.0.1:8000",
                                             "access-control-request-method": "POST"}),
    ]  # fmt: skip
    for r in replies:
        assert not [h for h in r.headers if h.lower().startswith("access-control-")]


@pytest.mark.parametrize("path", ["/api/health", "/", "/app.js", "/api/nope", "/missing.txt"])
def test_csp_on_every_kind_of_response(c, path):
    assert c.get(path).headers["content-security-policy"] == CSP


# 9. page image


def png_size(r):
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    return Image.open(io.BytesIO(r.content)).size


def test_image_before_read_is_the_rendered_page(c):
    assert png_size(c.get(f"/api/documents/{new_doc(c)}/pages/1/image?v=3")) == (40, 20)


def test_image_after_read_is_straightened(c, reader):
    doc = new_doc(c)
    reader.rotation = 90.0
    page = read(c, doc).json()
    assert png_size(c.get(f"/api/documents/{doc}/pages/1/image")) == (page["width"], page["height"])
    assert (page["width"], page["height"]) == (20, 40)


@pytest.mark.parametrize("n", ["0", "02", "2", "-1", "x"])
def test_image_of_a_missing_page_is_404(c, n):
    assert c.get(f"/api/documents/{new_doc(c)}/pages/{n}/image").status_code == 404


# 10. editing

XSS = "<img src=x onerror=alert(1)>"


def test_confirm_and_edit_statuses(c):
    doc = new_doc(c)
    read(c, doc)
    same = edit(c, doc, "p1-l1", "old1-1").json()
    assert same["status"] == "confirmed" and same["final_text"] == "old1-1"
    assert edit(c, doc, "p1-l1", "other").json()["status"] == "edited"


def test_markup_comes_back_verbatim(c):
    doc = new_doc(c)
    read(c, doc)
    r = edit(c, doc, "p1-l1", XSS)
    assert XSS.encode() in r.content and r.headers["content-type"] == "application/json"
    assert exported(c, doc)["pages"][0]["lines"][0]["final_text"] == XSS


def test_lone_surrogate_is_refused(c):
    doc = new_doc(c)
    read(c, doc)
    r = c.put(f"/api/documents/{doc}/lines/p1-l1", content=b'{"final_text": "\\ud800"}',
              headers={"content-type": "application/json"})  # fmt: skip
    assert r.status_code == 400 and list(r.json()) == ["error"]


def test_unknown_line_is_404(c):
    assert edit(c, new_doc(c), "p1-l9", "x").status_code == 404


def test_edit_without_content_type_is_accepted(c):
    doc = new_doc(c)
    read(c, doc)
    r = c.put(f"/api/documents/{doc}/lines/p1-l1", content=b'{"final_text": "z", "x": 1}')
    assert r.status_code == 200 and r.json()["final_text"] == "z"


# 11. export


def export_of(c, doc, fmt):
    return c.get(f"/api/documents/{doc}/export", params={"format": fmt})


def test_export_bodies_and_headers(c):
    doc = new_doc(c, data=pdf_bytes([(100, 100)] * 2), name="a.pdf")
    read(c, doc)
    model = c.app.state.store.get(doc).doc
    txt, js, dx = (export_of(c, doc, f) for f in ("txt", "json", "docx"))
    assert txt.content == export.to_txt(model).encode()
    assert txt.headers["content-type"] == "text/plain; charset=utf-8"
    assert js.content == export.to_json(model).encode()
    assert js.headers["content-type"] == "application/json"
    assert dx.headers["content-type"].endswith("wordprocessingml.document")
    runs = [r for p in docx.Document(io.BytesIO(dx.content)).paragraphs for r in p.runs]
    assert [r.text for r in runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW] == ["old1-1"]
    for r in (txt, js, dx):
        assert r.headers["x-unresolved-count"] == "1" and r.headers["x-unread-pages"] == "1"
        assert r.headers["content-disposition"].startswith('attachment; filename="a.')


def test_a_page_with_unread_areas_counts_as_unresolved(c, reader):
    reader.regions = ((0.0, 0.0, 5.0, 5.0), (5.0, 5.0, 9.0, 9.0))
    doc = new_doc(c)
    read(c, doc)
    assert c.get("/api/documents").json()[0]["unresolved_count"] == 2  # 1 line + 1 page
    txt = export_of(c, doc, "txt")
    assert txt.headers["x-unresolved-count"] == "2"
    assert txt.text.startswith(
        "[page 1: 2 areas were not read; the JSON export lists their boxes]\n"
    )


def test_export_filename_is_neutralised(c):
    doc = new_doc(c, name="../../a\r\nb<x>.pdf")
    disposition = export_of(c, doc, "txt").headers["content-disposition"]
    assert not set(disposition) & set("/\\\r\n<>")


def test_export_non_ascii_filename(c):
    disposition = export_of(c, new_doc(c, name="فاتورة.png"), "txt").headers["content-disposition"]
    assert disposition == (
        'attachment; filename="______.txt"; '
        "filename*=UTF-8''%D9%81%D8%A7%D8%AA%D9%88%D8%B1%D8%A9.txt"
    )


@pytest.mark.parametrize("fmt", [None, "pdf", ""])
def test_export_needs_a_known_format(c, fmt):
    r = c.get(f"/api/documents/{new_doc(c)}/export", params={} if fmt is None else {"format": fmt})
    assert r.status_code == 400 and r.json() == {"error": "format must be txt, docx or json"}


def test_export_of_unknown_document_is_404(c):
    assert export_of(c, "0" * 32, "txt").status_code == 404


# 12. eviction


def three(c):
    return [new_doc(c) for _ in range(3)]


def test_edited_oldest_blocks_a_fourth_upload(c):
    ids = three(c)
    read(c, ids[0])
    edit(c, ids[0], "p1-l1", "changed")
    r = upload(c)
    assert r.status_code == 409 and list(r.json()) == ["error"]
    assert [d["document_id"] for d in c.get("/api/documents").json()] == ids


def test_confirmed_oldest_blocks_a_fourth_upload(c):
    ids = three(c)
    read(c, ids[0])
    edit(c, ids[0], "p1-l1", "old1-1")
    assert upload(c).status_code == 409


def test_three_edited_documents_block_a_fourth_upload(c):
    ids = three(c)
    for doc in ids:
        read(c, doc)
        edit(c, doc, "p1-l1", "changed")
    assert upload(c).status_code == 409
    assert len(c.get("/api/documents").json()) == 3


def test_unedited_oldest_gives_way(c):
    first, *_ = three(c)
    assert upload(c).status_code == 200
    assert export_of(c, first, "txt").status_code == 404


# 13. delete


def test_delete_then_gone(c):
    doc = new_doc(c)
    r = c.delete(f"/api/documents/{doc}")
    assert r.status_code == 204 and r.content == b""
    assert c.delete(f"/api/documents/{doc}").status_code == 404


def test_delete_during_a_read_makes_the_read_404(c, reader):
    doc = new_doc(c)
    t, out = slow_read(c, reader, doc)
    assert c.delete(f"/api/documents/{doc}").status_code == 204
    assert finish(reader, t, out).status_code == 404
    assert read(c, new_doc(c)).status_code == 200


# 14. health


class FakeModel:
    def __init__(self, fails=None):
        self.fits, self.servers, self.done, self.fails = 0, 0, threading.Event(), fails

    def reachable(self):
        return None

    def fit_so_far(self):
        return "untested"

    def calibrated_so_far(self):
        return "uncalibrated"

    def identity_so_far(self):
        return None

    def fit(self):
        self.fits += 1
        if self.fails:
            self.done.set()
            raise self.fails

    def server(self):
        self.servers += 1
        self.done.set()


def test_health_without_a_model(c):
    body = c.get("/api/health").json()
    assert body["status"] == "ok" and body["model"]["configured"] is False
    assert body["tesseract"] == "missing" and body["weights"] == "missing"


def test_health_with_a_model_leaks_nothing(tmp_path, reader, static, logs, capsys):
    settings = Settings(weights_dir=tmp_path, model_url="http://127.0.0.1:9/v1", model_name="m",
                        model_api_key="SENTINEL-KEY")  # fmt: skip
    model = FakeModel()
    with make_client(tmp_path, reader, static, logs, settings=settings, model=model) as client:
        r = client.get("/api/health")
        assert model.done.wait(10)
    assert r.json()["model"] == {"configured": True, "reachable": None, "name": "m",
                                 "host": "127.0.0.1", "leaves_machine": False,
                                 "fit": "untested", "calibrated": "uncalibrated",
                                 "identity": None, "taus": []}  # fmt: skip
    everything = r.text + str(r.headers) + capsys.readouterr().err + "".join(logs)
    assert "SENTINEL-KEY" not in everything and "/v1" not in everything
    assert model.fits == 1 and model.servers == 1


def test_startup_asks_for_the_fit_and_the_server_once(tmp_path, reader, static, logs):
    model = FakeModel()
    with make_client(tmp_path, reader, static, logs, model=model) as client:
        assert model.done.wait(10)
        client.get("/api/health")
    assert (model.fits, model.servers, logs) == (1, 1, [])


def test_a_blocked_self_test_does_not_block_startup(tmp_path, reader, static, logs):
    release = threading.Event()
    model = FakeModel()
    model.fit = lambda: release.wait(10)
    try:
        with make_client(tmp_path, reader, static, logs, model=model) as client:
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/documents").status_code == 200
            release.set()
    finally:
        release.set()


def test_health_is_a_plain_function_route(c):
    (route,) = [r for r in c.app.routes if getattr(r, "path", None) == "/api/health"]
    assert not inspect.iscoroutinefunction(route.endpoint)


def test_self_test_failure_is_logged_not_raised(tmp_path, reader, static, logs):
    model = FakeModel(fails=RuntimeError("x"))
    with make_client(tmp_path, reader, static, logs, model=model) as client:
        assert model.done.wait(10)
        client.get("/api/health")
    assert logs == ["error: the model server self-test could not run (RuntimeError)"]
    assert model.servers == 0


def test_second_health_does_not_hash_again(c, tmp_path, monkeypatch):
    content = b"x"
    asset = assets.Asset("onnxtr/tiny.onnx", "https://example.invalid/t",
                         hashlib.sha256(content).hexdigest(), 1, "test")  # fmt: skip
    (tmp_path / "onnxtr").mkdir()
    (tmp_path / "onnxtr" / "tiny.onnx").write_bytes(content)
    monkeypatch.setattr(ocr_setup, "ONNXTR", (asset,))
    real, calls = hashlib.sha256, []

    def counting(*args):
        if not args:  # the file hasher starts empty; other hashes pass their data
            calls.append(1)
        return real(*args)

    monkeypatch.setattr(assets.hashlib, "sha256", counting)
    assert c.get("/api/health").json()["weights"] in ("ok", "uncalibrated")
    first = len(calls)
    assert first >= 1
    c.get("/api/health")
    assert len(calls) == first


# 15. static files and shutdown


def test_static_files(c):
    home = c.get("/")
    assert home.status_code == 200 and home.headers["content-type"].startswith("text/html")
    assert home.text == "<p>home</p>"
    assert c.get("/app.js").status_code == 200


def test_shutdown_empties_the_store(tmp_path, reader, static, logs):
    with make_client(tmp_path, reader, static, logs) as client:
        new_doc(client)
        store = client.app.state.store
        assert len(store.documents()) == 1
    assert store.documents() == []


@pytest.mark.parametrize(("method", "path"), [("POST", "/"), ("PUT", "/app.js")])
def test_wrong_method_on_a_static_file_keeps_its_allow(c, method, path):
    r = c.request(method, path)
    assert r.status_code == 405
    assert r.json() == {"error": "method not allowed"}
    assert r.headers["allow"] == "GET, HEAD"


@pytest.mark.parametrize("path", ["/api/health/", "/api/documents/"])
def test_a_trailing_slash_is_an_unknown_path_not_a_redirect(c, path):
    r = c.get(path, follow_redirects=False)
    assert (r.status_code, r.json()) == (404, {"error": "not found"})


def test_a_very_long_page_number_is_an_unknown_page(c):
    doc = c.post("/api/documents", files={"file": ("a.png", PNG)}, data={"languages": "en"})
    doc_id = doc.json()["document_id"]
    r = c.get(f"/api/documents/{doc_id}/pages/{'9' * 5000}/image")
    assert (r.status_code, r.json()) == (404, {"error": "no such page in this document"})


@pytest.mark.parametrize(("data", "line"), BIG_IMAGE_CASES)
def test_upload_of_a_pdf_with_an_oversized_image_is_refused(c, data, line):
    r = upload(c, data, "a.pdf")
    assert (r.status_code, r.json()) == (400, {"error": line})
