"""Lines read by the model server: OCR lines, re-read text-layer lines, engine record."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from folio_ocr import calibration, model_calibration, model_request, model_setup
from folio_ocr.charsets import line_script
from folio_ocr.document import Engine, Line, Page
from folio_ocr.flags import (
    MODEL_SELF_TEST_FAILED,
    NOT_READ,
    ORIENTATION_UNCERTAIN,
    PRESENTATION_FORMS,
    REPEATED,
    SCRIPT_MISMATCH,
    TEXT_LAYER_INVALID,
    VISUAL_ORDER,
)
from folio_ocr.model_pages import ModelDocument, ocr_preprocessing, swap_line
from folio_ocr.model_reply import MIXED_ORDER, ModelReading
from folio_ocr.ocr_setup import OcrSetup, onnxtr_setup
from folio_ocr.ocr_words import LineWords

INFO = model_setup.ServerInfo("ollama", "0.12", "a" * 64)
GOOD = ModelReading("Invoice 12", (), flags=(), confidence=0.999, method="logprobs")
BOX = (10.0, 10.0, 90.0, 30.0)
PAGE = ("ltr", "en", ())
AR = "".join(chr(c) for c in (0x627, 0x644, 0x639, 0x631, 0x628))
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
CONFIG = SimpleNamespace(name="m", host="127.0.0.1", page_deadline_s=900.0)


@dataclass
class StubReader:
    replies: list = field(default_factory=list)
    fit_result: str = "pass"
    info: object = None
    calls: list = field(default_factory=list)
    clock: float = 100.0
    config: SimpleNamespace = field(default_factory=lambda: CONFIG)

    def now(self):
        return self.clock

    def fit(self):
        self.calls.append("fit")
        return self.fit_result

    def server(self):
        return self.info

    def tau(self, key):
        return model_calibration.tau_for(key, None, ())

    def read(self, image, languages, page_ends_at):
        self.calls.append(("read", image, languages, page_ends_at))
        return self.replies.pop(0)

    def images(self):
        return [c[1] for c in self.calls if isinstance(c, tuple)]


def ocr_setup(setup=None):
    return OcrSetup("none", None, None, None, 1.0, None, setup, None)


def lw(box=BOX):
    return LineWords(box, ())


def build(reader, reading, page=PAGE):
    ocr = ocr_setup()
    mp = ModelDocument(reader, "en").ocr_page(ocr)
    return mp.line(ocr, (1, 1), lw(), reading, page)


def with_tau(monkeypatch, method, tau=0.5):
    key = model_setup.model_setup(INFO, "m", method, ocr_preprocessing(ocr_setup())).key
    monkeypatch.setattr(calibration, "TAUS", calibration.checked_table({key: tau}))


def two(conf):
    return ModelReading("Invoice 12", (), flags=(), confidence=conf, method="two_read")


def reply_text(text):
    return ModelReading(text, (), confidence=0.99, method="logprobs")


def test_uncalibrated_reply_is_low_confidence():
    line = build(StubReader(), GOOD)
    assert calibration.LOW_CONFIDENCE in line.flags
    assert line.status == "needs_review"
    assert line.reader == "model"
    assert line.confidence_method == "logprobs"
    assert line.text == "Invoice 12"


def test_calibrated_reply_accepted_until_prompt_changes(monkeypatch):
    with_tau(monkeypatch, "logprobs")
    line = build(StubReader(info=INFO), GOOD)
    assert line.flags == ()
    assert line.status == "auto_accepted"
    monkeypatch.setattr(model_request, "PROMPT_SHA256", "other")
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), GOOD).flags


def test_missing_server_info_is_low_confidence(monkeypatch):
    with_tau(monkeypatch, "logprobs")
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=None), GOOD).flags


def test_two_read_without_tau_is_low():
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), two(0.99)).flags


def test_two_read_floor_and_tau(monkeypatch):
    with_tau(monkeypatch, "two_read")
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), two(0.97)).flags
    assert build(StubReader(info=INFO), two(0.99)).flags == ()


def test_logprobs_tau_does_not_accept_two_read(monkeypatch):
    with_tau(monkeypatch, "logprobs")
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), two(0.99)).flags


def test_method_none_is_low(monkeypatch):
    with_tau(monkeypatch, "logprobs")
    reading = ModelReading("Invoice 12", (), method="none")
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), reading).flags


def test_mixed_order_from_real_reply():
    path = Path(__file__).parent / "data" / "model_replies" / "mixed_visual_order.json"
    text = json.loads(path.read_text(encoding="utf-8"))["choices"][0]["message"]["content"]
    assert MIXED_ORDER in build(StubReader(), reply_text(text)).flags


@pytest.mark.parametrize("text", [AR, "Invoice", AR + " " + chr(0x661) + chr(0x662), AR + " 12"])
def test_no_mixed_order_for_single_script(text):
    assert MIXED_ORDER not in build(StubReader(), reply_text(text)).flags


def test_reply_flags_first_and_raw_text_kept():
    reading = ModelReading("```x```\nY", (), flags=("truncated", "format_noise"), method="none")
    line = build(StubReader(), reading)
    assert line.flags[:2] == ("truncated", "format_noise")
    assert line.text == "```x```\nY"
    assert len(set(line.flags)) == len(line.flags)


def test_self_test_failure_flagged_on_read_and_unread():
    reader = StubReader(fit_result="fail")
    assert MODEL_SELF_TEST_FAILED in build(reader, GOOD).flags
    line = build(reader, ModelReading("", (), error="down"))
    assert MODEL_SELF_TEST_FAILED in line.flags and NOT_READ in line.flags


def test_error_reading_makes_unread_line_with_extra_flags():
    page = ("ltr", "en", (ORIENTATION_UNCERTAIN,))
    line = build(StubReader(), ModelReading("", (), error="down"), page=page)
    assert (line.reader, line.text, line.confidence_method) == ("none", "", "none")
    assert line.flags == (NOT_READ, ORIENTATION_UNCERTAIN)
    assert line.status == "needs_review" and line.confidence is None


def test_none_reading_is_unread():
    line = build(StubReader(), None)
    assert line.reader == "none" and line.flags[0] == NOT_READ


def test_read_fits_first_crops_and_sets_deadline():
    reader = StubReader(replies=[GOOD])
    ocr = ocr_setup()
    mp = ModelDocument(reader, "en").ocr_page(ocr)
    reader.clock = 5000.0
    gray = Image.new("L", (200, 100), 255)
    assert mp.read(ocr, gray, lw()) is GOOD
    assert reader.calls[0] == "fit"
    _, image, languages, ends_at = reader.calls[1]
    assert image.size == (80 + 2 * model_request.PADDING_PX, 20 + 2 * model_request.PADDING_PX)
    assert languages == "en"
    assert ends_at == 100.0 + 900.0


def test_read_empty_box_is_error_reading():
    reader = StubReader()
    mp = ModelDocument(reader, "en").ocr_page(ocr_setup())
    got = mp.read(ocr_setup(), Image.new("L", (50, 50)), lw((500.0, 500.0, 600.0, 600.0)))
    assert got.error and got.text == ""
    assert reader.images() == []


def tl_line(n, flags, status="needs_review", text="old"):
    box = (10.0, 20.0 * n, 10.0 + 10 * n, 20.0 * n + 10)
    return Line(f"p1-l{n}", box, text, "ltr", "latin", "text_layer", None, "reader", flags,
                status, text)  # fmt: skip


def tl_page(lines, error=None):
    return Page(1, 400, 400, 300, 0.0, IDENT, "text_layer", "ltr", "read", (), (),
                tuple(lines), None, error)  # fmt: skip


def routed():
    return [
        tl_line(1, (VISUAL_ORDER,)),
        tl_line(2, (TEXT_LAYER_INVALID, PRESENTATION_FORMS)),
        tl_line(3, (REPEATED,)),
        tl_line(4, (SCRIPT_MISMATCH,)),
        tl_line(5, (VISUAL_ORDER, SCRIPT_MISMATCH)),
        tl_line(6, (VISUAL_ORDER,), "edited"),
        tl_line(7, (TEXT_LAYER_INVALID,), "confirmed"),
        tl_line(8, (), "auto_accepted"),
    ]


def test_reread_routing():
    lines = routed()
    reader = StubReader(replies=[reply_text(f"new{i}") for i in range(3)])
    page = tl_page(lines)
    out = ModelDocument(reader, "en").text_page().reread(page, Image.new("RGB", (400, 400)))
    gray = Image.new("L", (400, 400))
    assert len(reader.images()) == 3
    for img, i in zip(reader.images(), (0, 1, 4), strict=True):
        assert img.size == model_request.crop(gray, lines[i].box).size
    for k, i in enumerate((0, 1, 4)):
        got = out.lines[i]
        assert (got.reader, got.text, got.status) == ("model", f"new{k}", "needs_review")
        assert got.flags[: len(lines[i].flags)] == lines[i].flags
        assert (got.id, got.box) == (lines[i].id, lines[i].box)
    for i in (2, 3, 5, 6, 7):
        assert out.lines[i] is lines[i]
    assert out.error is None


def test_reread_failures():
    lines = routed()
    bad = [ModelReading("", (), error="down"), ModelReading("", (), error="other")]
    reader = StubReader(replies=[reply_text("ok"), *bad], fit_result="fail")
    page = tl_page(lines, error="earlier")
    out = ModelDocument(reader, "en").text_page().reread(page, Image.new("L", (400, 400)))
    for i in (1, 4):
        got = out.lines[i]
        assert (got.reader, got.text) == ("text_layer", "old")
        assert NOT_READ in got.flags and MODEL_SELF_TEST_FAILED in got.flags
        assert got.flags[: len(lines[i].flags)] == lines[i].flags
        assert got.status == "needs_review"
    assert out.error == "earlier; down (2 lines not read)"


def test_single_reread_failure_is_bare_message():
    reader = StubReader(replies=[ModelReading("", (), error="down")])
    page = tl_page([tl_line(1, (VISUAL_ORDER,))])
    out = ModelDocument(reader, "en").text_page().reread(page, Image.new("L", (400, 400)))
    assert out.error == "down"


def test_reread_without_failure_keeps_page_error():
    reader = StubReader(replies=[reply_text("ok")])
    page = tl_page([tl_line(1, (VISUAL_ORDER,))], error="kept")
    out = ModelDocument(reader, "en").text_page().reread(page, Image.new("L", (400, 400)))
    assert out.error == "kept"


def test_swap_line():
    cur, new = tl_line(1, (VISUAL_ORDER,)), tl_line(1, (), "auto_accepted", "new")
    for status in ("edited", "confirmed"):
        kept = tl_line(1, (), status)
        assert swap_line(kept, new) is kept
    assert swap_line(cur, new) is new


def test_engine_unchanged_when_nothing_asked():
    base = Engine("d", None, "onnxtr", None, None, "untested", None)
    assert ModelDocument(StubReader(), "en").engine(base) is base


@pytest.mark.parametrize("info", [None, INFO])
def test_engine_after_ocr_read(info):
    base = Engine("d", None, "onnxtr", None, None, "untested", None)
    ocr = ocr_setup()
    reader = StubReader(replies=[GOOD], info=info, fit_result="fail")
    doc = ModelDocument(reader, "en")
    page = doc.ocr_page(ocr)
    page.read(ocr, Image.new("L", (200, 100)), lw())
    page.line(ocr, (1, 1), lw(), GOOD, PAGE)
    eng = doc.engine(base)
    assert (eng.reader, eng.model_name, eng.model_host) == ("model", "m", "127.0.0.1")
    assert eng.model_fit == "fail"
    assert eng.prompt_sha256 == model_request.PROMPT_SHA256
    if info is None:
        assert eng.calibration is None
    else:
        key = model_setup.model_setup(INFO, "m", "logprobs", ocr_preprocessing(ocr)).key
        assert eng.calibration == model_calibration.ModelCalibration(key, None)


def test_engine_calibration_carries_tau(monkeypatch):
    with_tau(monkeypatch, "logprobs", 0.4)
    doc = ModelDocument(StubReader(replies=[GOOD], info=INFO), "en")
    ocr = ocr_setup()
    page = doc.ocr_page(ocr)
    page.read(ocr, Image.new("L", (200, 100)), lw())
    page.line(ocr, (1, 1), lw(), GOOD, PAGE)
    base = Engine("d", None, "onnxtr", None, None, "untested", None)
    assert doc.engine(base).calibration.tau == 0.4


def test_engine_after_only_rereads_keeps_reader():
    base = Engine("d", None, "text_layer", None, None, "untested", None)
    doc = ModelDocument(StubReader(replies=[reply_text("ok")]), "en")
    page = tl_page([tl_line(1, (VISUAL_ORDER,))])
    doc.text_page().reread(page, Image.new("L", (400, 400)))
    eng = doc.engine(base)
    assert eng.reader == "text_layer" and eng.model_name == "m"


def test_ocr_preprocessing():
    assert ocr_preprocessing(ocr_setup()) == onnxtr_setup(None).preprocessing
    setup = onnxtr_setup("5.3")
    assert ocr_preprocessing(ocr_setup(setup)) == setup.preprocessing
    assert setup.preprocessing != onnxtr_setup(None).preprocessing


@pytest.mark.parametrize(
    "text",
    [
        "abc " + chr(0x434) + chr(0x430),  # Latin and Cyrillic
        chr(0x628) + chr(0x627) + " " + chr(0x3B1),  # Arabic and Greek
    ],
)
def test_mixed_order_needs_arabic_and_latin_letters_not_just_two_scripts(text):
    assert line_script(text) == "mixed"
    assert MIXED_ORDER not in build(StubReader(), reply_text(text)).flags


def _logprobs(conf):
    return ModelReading("Invoice 12", (), flags=(), confidence=conf, method="logprobs")


def test_score_below_tau_is_low_and_equal_to_tau_is_accepted(monkeypatch):
    with_tau(monkeypatch, "logprobs", 0.5)
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), _logprobs(0.3)).flags
    assert build(StubReader(info=INFO), _logprobs(0.5)).flags == ()


def test_two_read_floor_is_a_strict_less_than(monkeypatch):
    with_tau(monkeypatch, "two_read", 0.5)
    assert build(StubReader(info=INFO), two(0.98)).flags == ()
    assert calibration.LOW_CONFIDENCE in build(StubReader(info=INFO), two(0.9799)).flags
