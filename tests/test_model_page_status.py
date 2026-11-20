"""The status of a page read by a model server: failed only when no line was answered."""

from test_ocr_page_unit import LEFT, LOW_ROW, Doubles, read, setup

from folio_ocr.document import Line
from folio_ocr.reading import Reading

EMPTY = ("no_transcription", "empty", "low_confidence")


class StubModel:
    """A model page whose lines carry the flags given per line, in reading order."""

    def __init__(self, flags):
        self.flags = list(flags)

    def read(self, ocr, gray, line):
        return None

    def line(self, ocr, n, line, reading, page):
        flags = self.flags[n[1] - 1]
        return Line(
            f"p1-l{n[1]}", line.box, "" if flags else "x", "ltr", "unknown",
            "none" if "not_read" in flags else "model", None, "none", flags,
            "needs_review", "",
        )  # fmt: skip


def status(monkeypatch, boxes, flags):
    Doubles(monkeypatch, boxes)
    page = read(setup("none", None), model=StubModel(flags))
    return page.status


def test_every_line_not_read_fails_the_page(monkeypatch):
    assert status(monkeypatch, [LEFT, LOW_ROW], [("not_read",)] * 2) == "failed"


def test_one_answered_line_keeps_the_page_read(monkeypatch):
    assert status(monkeypatch, [LEFT, LOW_ROW], [("not_read",), ()]) == "read"


def test_empty_answers_are_answers(monkeypatch):
    assert status(monkeypatch, [LEFT, LOW_ROW], [EMPTY, EMPTY]) == "read"


def test_a_page_without_lines_is_read(monkeypatch):
    assert status(monkeypatch, [], []) == "read"


def test_without_a_model_failing_tesseract_lines_leave_the_page_read(monkeypatch):
    doubles = Doubles(monkeypatch, [LEFT, LOW_ROW])
    doubles.tesseract = {1: Reading("", (), error="boom"), 2: Reading("", (), error="boom")}
    page = read(setup("tesseract"))
    assert [ln.flags for ln in page.lines] == [("not_read",)] * 2
    assert page.status == "read"
