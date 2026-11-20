"""The model server path end to end: text-layer lines re-read, OCR pages read, server trouble
and the time limits, through the real pipeline against a local fake server."""

import base64
import dataclasses
import io
import json
import os
import socket
import time
from pathlib import Path

import pytest
from fake_model_server import Answer, FakeModelServer, Seen
from PIL import Image

from folio_ocr import calibration, model_pages, model_request, model_setup, ocr_setup, pipeline
from folio_ocr.config import Settings
from folio_ocr.model_calibration import ModelCalibration
from folio_ocr.model_reader import ModelConfig, ModelReader, line_time_limit, page_time_limit
from folio_ocr.model_reply import UNREADABLE
from folio_ocr.model_self_test_lines import LINES

FIXTURES = Path(__file__).parent / "fixtures"
PDF = "ar-adversarial-2.pdf"  # 14 lines on one text-layer page, 9 of them flagged visual_order
TRIGGERS = ("visual_order", "text_layer_invalid")
TEXT = "Invoice number: INV-2026-3706"
DIGEST = "ab" * 32
CHAT = "/v1/chat/completions"
WEIGHTS = Settings.from_env(os.environ).weights_dir
needs_ocr = pytest.mark.skipif(
    ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"),
    reason="the OCR model files are not set up (folio setup)",
)


def reply(text) -> bytes:
    choice = {
        "message": {"content": text},
        "finish_reason": "stop",
        "logprobs": {"content": [{"token": "x", "logprob": -0.01}]},
    }
    return json.dumps({"choices": [choice]}).encode()


def sent_image(seen: Seen) -> Image.Image:
    url = json.loads(seen.body)["messages"][0]["content"][1]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L")


def _key(image: Image.Image) -> tuple:
    return image.size, image.tobytes()


SELF_TEST = {_key(Image.open(io.BytesIO(x.png)).convert("L")): x.text for x in LINES}


def chat(fake: FakeModelServer) -> list[Seen]:
    return [s for s in fake.seen if s.target == CHAT]


def line_requests(fake: FakeModelServer) -> list[Seen]:
    """The chat requests that are not one of the two self-test lines."""
    return [s for s in chat(fake) if _key(sent_image(s)) not in SELF_TEST]


def server(text=TEXT, *, trigger=None, native=None) -> FakeModelServer:
    """A fake model server: the self-test lines are answered exactly; any other line gets
    `trigger` (an Answer) or else `text`. `native` maps "/api/version" and "/api/tags" to
    answers; they are 404 otherwise."""
    native = native or {}

    def answer(seen: Seen) -> Answer:
        if seen.method == "POST" and seen.target == CHAT:
            known = SELF_TEST.get(_key(sent_image(seen)))
            if known is not None:
                return Answer(body=reply(known))
            return Answer(body=reply(text)) if trigger is None else trigger
        return native.get(seen.target, Answer(status=404, body=b"{}"))

    return FakeModelServer(answer)


def reader(url: str, **kw) -> ModelReader:
    fields = {"name": "m", "api_key": None, "line_timeout_s": 5.0, "page_deadline_s": 900.0}
    return ModelReader(ModelConfig(url=url, two_read=False, **{**fields, **kw}))


def closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}/v1"


def read_pdf(model=None):
    return pipeline.read_bytes((FIXTURES / PDF).read_bytes(), PDF, languages="en+ar", model=model)


@pytest.fixture(scope="module")
def plain():
    return read_pdf()


def is_trigger(line) -> bool:
    return bool(set(line.flags) & set(TRIGGERS))


def triggers(doc):
    return [x for x in doc.pages[0].lines if is_trigger(x)]


def blank(page) -> Image.Image:
    return Image.new("L", (page.width, page.height), 255)


def test_text_layer_lines_are_reread_and_the_others_kept(plain):
    with server("Total due 42") as fake:
        doc = read_pdf(reader(fake.url))
    before, after = plain.pages[0].lines, doc.pages[0].lines
    assert len(before) == len(after) and 0 < len(triggers(plain)) < len(before)
    for old, new in zip(before, after, strict=True):
        if not is_trigger(old):
            assert new == old
            continue
        assert (new.reader, new.text, new.final_text) == ("model", "Total due 42", "Total due 42")
        assert new.flags[: len(old.flags)] == old.flags
        assert "low_confidence" in new.flags and new.status == "needs_review"
    sent = [sent_image(s).size for s in line_requests(fake)]
    expect = [model_request.crop(blank(plain.pages[0]), x.box).size for x in triggers(plain)]
    assert len(chat(fake)) == 2 + len(expect) and sent == expect


def test_repeated_or_script_mismatch_alone_is_never_sent(plain):
    page = plain.pages[0]
    a, b, c = (dataclasses.replace(x, status="needs_review") for x in page.lines[:3])
    a = dataclasses.replace(a, flags=("repeated",))
    b = dataclasses.replace(b, flags=("script_mismatch",))
    c = dataclasses.replace(c, flags=("visual_order", "script_mismatch"))
    page = dataclasses.replace(page, lines=(a, b, c))
    with server() as fake:
        doc = model_pages.ModelDocument(reader(fake.url), "en+ar")
        out = doc.text_page().reread(page, blank(page))
    assert out.lines[0] is a and out.lines[1] is b
    assert out.lines[2].reader == "model" and len(line_requests(fake)) == 1


def test_an_edited_line_is_not_overwritten(plain):
    page = plain.pages[0]
    edited = dataclasses.replace(page.lines[0], flags=("visual_order",), status="edited")
    page = dataclasses.replace(page, lines=(edited,))
    with server() as fake:
        doc = model_pages.ModelDocument(reader(fake.url), "en+ar")
        out = doc.text_page().reread(page, blank(page))
    assert out.lines[0] is edited and line_requests(fake) == []


def test_engine_of_a_text_only_document(plain):
    with server() as fake:
        doc = read_pdf(reader(fake.url))
    e = doc.engine
    assert (e.reader, e.model_name, e.model_host, e.model_fit) == ("none", "m", "127.0.0.1", "pass")
    assert e.prompt_sha256 == model_request.PROMPT_SHA256 and e.calibration is None
    assert (plain.engine.model_fit, plain.engine.model_name) == ("untested", None)


def _trouble(case):
    """A fake (or None for a closed port) and the model reader for one kind of server trouble."""
    if case == "down":
        return None, reader(closed_port_url())
    answers = {
        "401": Answer(status=401, body=b"{}"),
        "500": Answer(status=500, body=b"{}"),
        "malformed": Answer(body=b"not json"),
    }
    if case == "401":  # a refused key fails every request, the self-test too
        fake = FakeModelServer(lambda seen: answers[case])
    else:
        fake = server(trigger=answers[case])
    return fake, None


@pytest.mark.parametrize("case", ["down", "401", "500", "malformed"])
def test_server_trouble_keeps_the_text_layer(plain, case):
    fake, model = _trouble(case)
    if fake is None:
        doc = read_pdf(model)
    else:
        with fake:
            doc = read_pdf(reader(fake.url))
    self_test_fails = case in ("down", "401")
    n, page = len(triggers(plain)), doc.pages[0]
    assert page.error is not None and "\n" not in page.error
    assert page.error.endswith(f" ({n} lines not read)")
    if case in ("401", "500"):
        assert f"HTTP {case}" in page.error
    for old, new in zip(plain.pages[0].lines, page.lines, strict=True):
        if not is_trigger(old):
            assert new == old
            continue
        assert (new.reader, new.text) == ("text_layer", old.text)
        assert "not_read" in new.flags and new.flags[: len(old.flags)] == old.flags
        assert ("model_self_test_failed" in new.flags) == self_test_fails
    assert doc.engine.model_fit == ("fail" if self_test_fails else "pass")


def test_null_content_gives_empty_model_lines(plain):
    body = json.dumps({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]})
    with server(trigger=Answer(body=body.encode())) as fake:
        doc = read_pdf(reader(fake.url))
    assert doc.pages[0].error is None
    assert len(triggers(doc)) == len(triggers(plain))
    for new in triggers(doc):
        assert (new.reader, new.text) == ("model", "")
        assert "no_transcription" in new.flags


def test_a_server_that_answers_the_same_for_every_image_fails_the_self_test():
    with FakeModelServer(lambda seen: Answer(body=reply(TEXT))) as fake:
        doc = read_pdf(reader(fake.url))
    assert doc.engine.model_fit == "fail"
    rereads = [x for x in doc.pages[0].lines if x.reader == "model"]
    assert rereads and all("model_self_test_failed" in x.flags for x in rereads)


def test_line_time_limit():
    slow = Answer(body=reply(TEXT), byte_delay_s=0.05)
    started = time.perf_counter()
    with server(trigger=slow) as fake:
        doc = read_pdf(reader(fake.url, line_timeout_s=0.3))
    took = time.perf_counter() - started
    lines = triggers(doc)
    assert lines and all("not_read" in x.flags for x in lines)
    assert doc.pages[0].error.startswith(line_time_limit(0.3))
    assert took < 0.3 * len(lines) + 3


def test_page_time_limit(plain):
    with server(trigger=Answer(body=reply(TEXT), delay_s=0.3)) as fake:
        doc = read_pdf(reader(fake.url, page_deadline_s=0.5))
    n = len(triggers(plain))
    assert len(doc.pages[0].lines) == len(plain.pages[0].lines)
    assert page_time_limit(0.5) in doc.pages[0].error
    skipped = [x for x in doc.pages[0].lines if "not_read" in x.flags]
    assert len(skipped) >= n - 3 and len(line_requests(fake)) <= 3 < n


def native_answers() -> dict:
    tags = {"models": [{"name": "m", "model": "m", "digest": DIGEST}]}
    return {
        "/api/version": Answer(body=b'{"version":"0.40.16"}'),
        "/api/tags": Answer(body=json.dumps(tags).encode()),
    }


def test_calibrated_lines_carry_no_low_confidence(monkeypatch):
    with server(native=native_answers()) as fake:
        uncalibrated = read_pdf(reader(fake.url))
    assert triggers(uncalibrated)
    assert all("low_confidence" in x.flags for x in triggers(uncalibrated))
    info = model_setup.ServerInfo("ollama", "0.40", DIGEST)
    key = model_setup.model_setup(info, "m", "logprobs", model_pages.TEXT_LAYER_PREPROCESSING).key
    monkeypatch.setattr(calibration, "TAUS", calibration.checked_table({key: 0.0}))
    with server(native=native_answers()) as fake:
        doc = read_pdf(reader(fake.url))
    assert triggers(doc)
    for x in triggers(doc):
        assert "low_confidence" not in x.flags and x.status == "needs_review"
    assert doc.engine.calibration == ModelCalibration(key, 0.0, "package", None)


def read_image(name, languages, url, **kw):
    return pipeline.read_bytes(
        (FIXTURES / name).read_bytes(),
        name,
        languages=languages,
        weights_dir=WEIGHTS,
        model=reader(url),
        **kw,
    )


@needs_ocr
def test_ocr_page_is_read_by_the_model():
    with server() as fake:
        doc = read_image("en-invoice-clean.png", "en", fake.url)
    page = doc.pages[0]
    assert page.status == "read" and page.lines
    assert len(chat(fake)) == 2 + len(page.lines)
    for x in page.lines:
        assert (x.reader, x.confidence_method) == ("model", "logprobs")
        assert "low_confidence" in x.flags and x.status == "needs_review"
    assert doc.engine.reader == "model" and doc.engine.calibration is None


@needs_ocr
def test_arabic_without_tesseract_is_read_by_the_model():
    with server() as fake:
        doc = read_image("ar-invoice-clean.png", "ar", fake.url, tesseract_path=None)
    lines = doc.pages[0].lines
    assert lines and all(x.reader == "model" for x in lines)
    assert not any("needs_model_server" in x.flags for x in lines)


@needs_ocr
def test_arabic_with_a_broken_tesseract_is_still_read_by_the_model(tmp_path):
    junk = tmp_path / "tesseract.exe"
    junk.write_bytes(b"junk bytes, not a program")
    with server() as fake:
        doc = read_image("ar-invoice-clean.png", "ar", fake.url, tesseract_path=junk)
    page = doc.pages[0]
    assert page.status == "read" and all(x.reader == "model" for x in page.lines)


def test_a_reply_nested_too_deep_is_an_unreadable_reply_not_a_crash(plain):
    deep = b"[" * 6000 + b"]" * 6000
    with server(trigger=Answer(body=deep)) as fake:
        doc = read_pdf(reader(fake.url))
    n, page = len(triggers(plain)), doc.pages[0]
    assert page.error is not None and UNREADABLE in page.error
    assert page.error.endswith(f" ({n} lines not read)")
    assert n and all("not_read" in x.flags for x in triggers(doc))


def first_lines_answered(answered: int) -> FakeModelServer:
    """A fake that passes the self-test, answers the first `answered` line requests and fails
    every later one with HTTP 500."""
    count = iter(range(1_000_000))

    def answer(seen: Seen) -> Answer:
        if seen.method == "POST" and seen.target == CHAT:
            known = SELF_TEST.get(_key(sent_image(seen)))
            if known is not None:
                return Answer(body=reply(known))
            if next(count) < answered:
                return Answer(body=reply(TEXT))
            return Answer(status=500, body=b"{}")
        return Answer(status=404, body=b"{}")

    return FakeModelServer(answer)


@needs_ocr
@pytest.mark.parametrize(("answered", "status"), [(0, "failed"), (1, "read")])
def test_an_ocr_page_with_no_line_answered_has_failed(answered, status):
    with first_lines_answered(answered) as fake:
        doc = read_image("en-invoice-clean.png", "en", fake.url)
    page = doc.pages[0]
    assert len(page.lines) > 1 and page.status == status
    assert sum("not_read" not in x.flags for x in page.lines) == answered
    assert page.error is not None and "HTTP 500" in page.error
