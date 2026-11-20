"""Reading a page with OCR: lines, confidence, flags, errors and order, tested with doubles for
the orientation, word search, recogniser and Tesseract."""

import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from folio_ocr import ocr, tesseract
from folio_ocr.geometry import straighten
from folio_ocr.ocr import _line, ocr_page
from folio_ocr.ocr_setup import OcrSetup
from folio_ocr.ocr_words import LineWords, Word
from folio_ocr.orientation import Orientation
from folio_ocr.reading import Reading

AR1, AR2 = "اب", "جد"
LEFT, MID, RIGHT = (0.0, 0.0, 20.0, 10.0), (25.0, 0.0, 45.0, 10.0), (50.0, 0.0, 70.0, 10.0)
FAR_LEFT, FAR_RIGHT = (0.0, 0.0, 20.0, 10.0), (100.0, 0.0, 120.0, 10.0)
LOW_ROW = (0.0, 30.0, 20.0, 40.0)
TESS, TESSDATA = Path("/opt/tesseract"), Path("/opt/data")


def setup(kind="onnxtr", tau=0.5, problem=None, osd=False):
    tess = kind == "tesseract" or osd
    return OcrSetup(
        kind, None, TESS if tess else None, TESSDATA if tess else None, 7.5, problem, None, tau, osd
    )


class Doubles:
    """Orientation, word search, word recogniser and Tesseract answering from fixed tables."""

    def __init__(
        self, monkeypatch, boxes, texts=None, uncertain=False, angle=0.0, orientation_error=None
    ):
        self.reads = []
        self.osd = []  # the osd callable handed to the orientation of each page
        self.texts = texts or {}
        self.tesseract = {}  # nth Tesseract call (from 1) -> Reading
        monkeypatch.setattr(ocr, "page_orientation", self.orientation)
        self.angle, self.uncertain, self.orientation_error = angle, uncertain, orientation_error
        monkeypatch.setattr(ocr, "find_words", lambda models, gray, rgb, detection: list(boxes))
        monkeypatch.setattr(ocr, "recognise_words", self.recognise)
        monkeypatch.setattr(tesseract, "read_line", self.read_line)

    def orientation(self, models, image, osd=None):
        self.osd.append(osd)
        return Orientation(self.angle, self.uncertain, None, "vote", self.orientation_error)

    def recognise(self, models, rgb, boxes):
        return [Word(b, *self.texts.get(b, ("", math.nan))) for b in boxes]

    def read_line(self, binary, tessdata, png, timeout_s):
        self.reads.append((binary, tessdata, png, timeout_s))
        return self.tesseract[len(self.reads)]


def read(ocr_setup, image=None, languages="en", **kw):
    image = image or Image.new("L", (200, 100), 255)
    return ocr_page(1, image, ocr=ocr_setup, languages=languages, render_dpi=None, **kw)


def page_of(
    ocr_setup, boxes, monkeypatch, languages="en", image=None, reasons=(), invalid=None, **doubles
):
    """(doubles, page) of a page whose word search finds `boxes`."""
    found = Doubles(monkeypatch, boxes, **doubles)
    page = read(ocr_setup, image, languages, reasons=reasons, invalid_lines=invalid)
    return found, page


def read_en(monkeypatch, scores, tau=0.5):
    """Words 'hello' (left), blank (middle), 'world' (right), found in scrambled order."""
    texts = {RIGHT: ("world", scores[1]), LEFT: ("hello", scores[0]), MID: ("  ", 0.95)}
    return page_of(setup(tau=tau), [RIGHT, LEFT, MID], monkeypatch, texts=texts)[1]


# --- onnxtr lines --------------------------------------------------------------------------


def test_onnxtr_line_text_confidence_and_acceptance(monkeypatch):
    page = read_en(monkeypatch, (0.7, 0.6))
    (line,) = page.lines
    assert line.text == line.final_text == "hello world"
    assert (line.reader, line.confidence_method) == ("onnxtr", "reader")
    assert line.confidence == 0.6  # the lowest word score
    assert line.flags == () and line.status == "auto_accepted"
    assert line.id == "p1-l1" and line.box == (0.0, 0.0, 70.0, 10.0)
    assert page.status == "read" and page.error is None and page.origin == "ocr"


def test_a_score_equal_to_the_threshold_is_not_low(monkeypatch):
    (line,) = read_en(monkeypatch, (0.5, 0.9)).lines
    assert "low_confidence" not in line.flags


def test_a_score_below_the_threshold_is_low_confidence(monkeypatch):
    (line,) = read_en(monkeypatch, (0.9, 0.4)).lines
    assert line.flags == ("low_confidence",) and line.status == "needs_review"
    assert line.confidence == 0.4


def test_without_a_threshold_every_line_is_low_confidence(monkeypatch):
    (line,) = read_en(monkeypatch, (0.9, 0.9), tau=None).lines
    assert line.flags == ("low_confidence",)
    assert line.confidence == 0.9


def test_an_unusable_score_gives_no_confidence_and_low_confidence(monkeypatch):
    (line,) = read_en(monkeypatch, (0.9, math.nan)).lines
    assert line.confidence is None
    assert line.flags == ("low_confidence",)


def test_a_line_without_words_has_no_confidence():
    line = _line(setup(), (1, 1), LineWords(LEFT, ()), Reading("x", ()), ("ltr", "en", ()))
    assert line.confidence is None
    assert "low_confidence" in line.flags


def test_a_lone_surrogate_is_replaced_and_flagged(monkeypatch):
    texts = {LEFT: ("ab" + chr(0xD800), 0.9)}
    _, page = page_of(setup(), [LEFT], monkeypatch, texts=texts)
    (line,) = page.lines
    assert line.text == "ab�"
    assert line.flags == ("invalid_text",)


def test_line_flags_come_from_the_text_and_the_page_languages(monkeypatch):
    texts = {LEFT: (AR1, 0.9)}
    _, page = page_of(setup(), [LEFT], monkeypatch, languages="en", texts=texts)
    assert page.lines[0].flags == ("script_mismatch",)
    _, page = page_of(setup(), [LEFT], monkeypatch, languages="ar", texts=texts)
    assert page.lines[0].flags == ()


# --- no reader -----------------------------------------------------------------------------


def test_with_no_reader_every_line_waits_for_a_model_server(monkeypatch):
    doubles, page = page_of(setup("none", tau=None), [LEFT, LOW_ROW], monkeypatch)
    assert len(page.lines) == 2
    for line in page.lines:
        assert line.reader == "none" and line.text == "" and line.confidence is None
        assert line.confidence_method == "none"
        assert line.flags == ("needs_model_server",)
        assert line.status == "needs_review"
    assert page.status == "read" and page.error is None
    assert doubles.reads == []


def test_with_no_reader_an_uncertain_orientation_is_flagged_after_the_need(monkeypatch):
    _, page = page_of(setup("none", None), [LEFT], monkeypatch, uncertain=True)
    assert page.lines[0].flags == ("needs_model_server", "orientation_uncertain")


# --- Tesseract -----------------------------------------------------------------------------


def tesseract_page(monkeypatch, boxes, readings, **doubles):
    found = Doubles(monkeypatch, boxes, **doubles)
    found.tesseract = readings
    return found, read(setup("tesseract"))


def test_tesseract_reads_each_line_with_the_documents_settings(monkeypatch):
    readings = {1: Reading(AR1, (0.9, 0.8)), 2: Reading("ab", (0.9,))}
    doubles, page = tesseract_page(monkeypatch, [LEFT, LOW_ROW], readings)
    first, second = page.lines
    assert (first.reader, first.text, first.confidence) == ("tesseract", AR1, 0.8)
    assert (second.reader, second.text) == ("tesseract", "ab")
    assert len(doubles.reads) == 2
    for binary, tessdata, png, timeout in doubles.reads:
        assert (binary, tessdata, timeout) == (TESS, TESSDATA, 7.5)
        assert png.startswith(b"\x89PNG")


def test_a_failed_tesseract_line_is_not_read_and_the_page_says_why(monkeypatch):
    readings = {1: Reading("", (), error="Tesseract ran too long")}
    _, page = tesseract_page(monkeypatch, [LEFT], readings, uncertain=True)
    (line,) = page.lines
    assert line.reader == "none" and line.text == "" and line.confidence is None
    assert line.flags == ("not_read", "orientation_uncertain")
    assert line.status == "needs_review"
    assert page.error == "Tesseract ran too long" and page.status == "read"


def test_the_page_error_counts_the_lines_that_failed(monkeypatch):
    readings = {
        1: Reading("", (), error="first failure"),
        2: Reading("ok", (0.9,)),
        3: Reading("", (), error="second failure"),
    }
    rows = [LEFT, LOW_ROW, (0.0, 60.0, 20.0, 70.0)]
    _, page = tesseract_page(monkeypatch, rows, readings)
    assert page.error == "first failure (2 lines not read)"


def test_a_tesseract_problem_fails_the_page_without_running_the_program(monkeypatch):
    problem = "Tesseract language data is missing; run folio setup"
    doubles, page = page_of(setup("tesseract", problem=problem), [LEFT, LOW_ROW], monkeypatch)
    assert page.status == "failed" and page.error == problem
    assert [ln.flags for ln in page.lines] == [("not_read",)] * 2
    assert [ln.reader for ln in page.lines] == ["none"] * 2
    assert doubles.reads == []


@pytest.mark.parametrize("kind", ["none", "tesseract"])
def test_readers_other_than_onnxtr_do_not_recognise_words(monkeypatch, kind):
    doubles = Doubles(monkeypatch, [LEFT])
    doubles.tesseract = {1: Reading("x", (0.9,))}
    monkeypatch.setattr(ocr, "recognise_words", lambda *a: pytest.fail("recognised"))
    assert len(read(setup(kind, None)).lines) == 1


# --- orientation by Tesseract OSD ----------------------------------------------------------


def test_the_osd_function_asks_tesseract_for_the_page_with_the_documents_settings(monkeypatch):
    doubles = Doubles(monkeypatch, [LEFT])
    asked = []

    def answer(binary, tessdata, png, timeout_s):
        asked.append((binary, tessdata, png, timeout_s))
        return tesseract.Osd(180, 3.0)

    monkeypatch.setattr(tesseract, "osd", answer)
    image = Image.new("L", (200, 100), 255)
    read(setup(osd=True), image)
    (osd,) = doubles.osd
    assert osd(image) == tesseract.Osd(180, 3.0)
    assert asked == [(TESS, TESSDATA, tesseract.page_png(image), 7.5)]


def test_without_osd_the_orientation_gets_no_osd_function(monkeypatch):
    doubles = Doubles(monkeypatch, [LEFT])
    read(setup())
    assert doubles.osd == [None]


def test_an_osd_failure_becomes_the_page_error_and_flags_the_page(monkeypatch):
    why = "Tesseract did not report the page orientation within 60 s"
    texts = {LEFT: ("a", 0.9), LOW_ROW: ("b", 0.9)}
    _, page = page_of(
        setup(osd=True),
        [LEFT, LOW_ROW],
        monkeypatch,
        texts=texts,
        uncertain=True,
        orientation_error=why,
    )
    assert page.error == why and len(page.error.splitlines()) == 1 and page.status == "read"
    assert page.flags == ("orientation_uncertain",)
    assert [ln.flags for ln in page.lines] == [("orientation_uncertain",)] * 2


def test_the_orientation_error_comes_first_when_lines_failed_too(monkeypatch):
    why = "Tesseract did not report the page orientation within 60 s"
    readings = {1: Reading("", (), error="line failure")}
    _, page = tesseract_page(monkeypatch, [LEFT], readings, uncertain=True, orientation_error=why)
    assert page.error == f"{why}; 1 line not read"


def test_the_orientation_error_comes_first_with_several_lines_not_read(monkeypatch):
    why = "Tesseract did not report the page orientation within 60 s"
    readings = {1: Reading("", (), error="line failure"), 2: Reading("", (), error="line failure")}
    _, page = tesseract_page(
        monkeypatch, [LEFT, LOW_ROW], readings, uncertain=True, orientation_error=why
    )
    assert page.error == f"{why}; 2 lines not read"


def test_a_tesseract_problem_outranks_the_orientation_error(monkeypatch):
    problem = "Tesseract language data is missing; run folio setup"
    _, page = page_of(
        setup("tesseract", problem=problem), [LEFT], monkeypatch, orientation_error="x"
    )
    assert page.error == problem


# --- page level ----------------------------------------------------------------------------


def test_an_uncertain_orientation_flags_the_page_and_every_line(monkeypatch):
    texts = {LEFT: ("a", 0.9), LOW_ROW: ("b", 0.9)}
    _, page = page_of(setup(), [LEFT, LOW_ROW], monkeypatch, texts=texts, uncertain=True)
    assert page.flags == ("orientation_uncertain",)
    assert [ln.flags for ln in page.lines] == [("orientation_uncertain",)] * 2
    assert all(ln.status == "needs_review" for ln in page.lines)


def test_a_certain_orientation_adds_no_flag(monkeypatch):
    _, page = page_of(setup(), [LEFT], monkeypatch, texts={LEFT: ("a", 0.9)})
    assert page.flags == () and page.lines[0].flags == ()


def test_page_flags_run_reasons_then_orientation_then_unread_regions(monkeypatch):
    image = Image.new("L", (200, 100), 255)
    ImageDraw.Draw(image).rectangle((100, 50, 190, 90), fill=0)
    _, page = page_of(
        setup(),
        [LEFT],
        monkeypatch,
        image=image,
        reasons=("empty_with_ink",),
        invalid=3,
        texts={LEFT: ("a", 0.9)},
        uncertain=True,
    )
    assert page.flags == ("empty_with_ink", "orientation_uncertain", "unread_regions")
    assert len(page.unread_regions) == 1
    assert page.invalid_lines == 3 and page.index == 1


def test_the_page_is_turned_by_the_orientation_angle(monkeypatch):
    image = Image.new("L", (200, 100), 255)
    _, page = page_of(setup("none", None), [LEFT], monkeypatch, image=image, angle=180.0)
    assert page.rotation_applied == 180.0
    assert page.transform == straighten(image, 180.0)[1]
    assert (page.width, page.height) == (200, 100)


def test_an_arabic_page_reads_right_to_left_within_a_row(monkeypatch):
    texts = {FAR_LEFT: (AR1, 0.9), FAR_RIGHT: (AR2, 0.9)}
    _, page = page_of(setup(), [FAR_LEFT, FAR_RIGHT], monkeypatch, languages="ar", texts=texts)
    assert page.direction == "rtl"
    assert [ln.text for ln in page.lines] == [AR2, AR1]
    assert [ln.direction for ln in page.lines] == ["rtl", "rtl"]
    assert [ln.id for ln in page.lines] == ["p1-l1", "p1-l2"]


def test_an_english_page_reads_left_to_right_within_a_row(monkeypatch):
    texts = {FAR_LEFT: ("one", 0.9), FAR_RIGHT: ("two", 0.9)}
    _, page = page_of(setup(), [FAR_RIGHT, FAR_LEFT], monkeypatch, texts=texts)
    assert page.direction == "ltr"
    assert [ln.text for ln in page.lines] == ["one", "two"]
