"""Areas of a page that were not read: how the exports and the command line report them."""

import errno
import io
import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import docx
import pytest
from docx.enum.text import WD_COLOR_INDEX

from folio_ocr import cli, pipeline
from folio_ocr.document import IDENTITY, Document, Engine, Line, Page, Source
from folio_ocr.export import (
    review_line_count,
    to_docx,
    to_json,
    to_txt,
    unresolved_count,
)

BOX = (0.0, 0.0, 5.0, 5.0)
NOTE_1 = "[page 1: 3 areas were not read; the JSON export lists their boxes]"
NOTE_3 = "[page 3: 1 area was not read; the JSON export lists its box]"
AREA_1 = "page 1: 3 areas were not read (see --format json for their boxes)"
AREA_3 = "page 3: 1 area was not read (see --format json for their boxes)"
SUMMARY = "a.png: 3 lines found, 3 read, 1 need review, 1 unread pages"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def line(text, status="auto_accepted", n=1):
    return Line(
        f"p1-l{n}", (0, 0, 1, 1), text, "ltr", "latin", "text_layer", None, "none",
        ("low_confidence",) if status == "needs_review" else (), status, text,
    )  # fmt: skip


def page(index, lines=(), regions=0, status="read", error=None):
    return Page(
        index, 10, 10, 300, 0.0, IDENTITY, "ocr", "ltr", status, (), (BOX,) * regions,
        tuple(lines), error=error,
    )  # fmt: skip


def doc(pages, filename="a.png"):
    engine = Engine("none", None, "none", None, None, "untested", None)
    return Document(
        "id", Source(filename, "0" * 64, "image", len(pages)), "en", engine, tuple(pages),
        datetime(2026, 1, 1, tzinfo=UTC), False,
    )  # fmt: skip


def three_pages(error=None):
    return doc([
        page(1, [line("a", "needs_review"), line("b", n=2)], regions=3),
        page(2, [line("c")]),
        page(3, regions=1, status="unread", error=error),
    ])  # fmt: skip


def test_review_line_count_counts_only_lines():
    assert review_line_count(three_pages()) == 1


def test_unresolved_count_adds_pages_with_unread_areas():
    assert unresolved_count(three_pages()) == 3


def test_unresolved_count_counts_pages_not_regions():
    d = doc([page(1, [line("a")], regions=5), page(2, [line("b")], regions=2), page(3)])
    assert unresolved_count(d) == 2


def test_unresolved_count_includes_regions_on_unread_pages():
    assert unresolved_count(doc([page(1, regions=2, status="unread")])) == 1


def test_to_txt_marks_pages_with_unread_areas():
    assert to_txt(three_pages()) == f"{NOTE_1}\na\nb\n\fc\n\f{NOTE_3}\n"


def test_to_txt_without_unread_areas_is_just_the_lines():
    d = doc([page(1, [line("a"), line("b", n=2)]), page(2, [line("c")])])
    assert to_txt(d) == "a\nb\n\fc\n"


def test_to_json_ends_with_unresolved_count():
    data = json.loads(to_json(three_pages()))
    assert data["unresolved_count"] == 3
    assert list(data)[-1] == "unresolved_count"


def read_docx(d):
    return docx.Document(io.BytesIO(to_docx(d))).paragraphs


def yellow(paragraphs):
    return [
        r for p in paragraphs for r in p.runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW
    ]


def test_docx_first_paragraph_counts_lines_only():
    first = read_docx(three_pages())[0].text
    assert first == "Lines that need review: 1 (highlighted)."


def test_docx_note_follows_the_page_heading():
    paragraphs = read_docx(three_pages())
    at = [p.text for p in paragraphs].index("Page 1")
    note = paragraphs[at + 1]
    assert note.text == NOTE_1
    assert [r.font.highlight_color for r in note.runs] == [WD_COLOR_INDEX.YELLOW]


def test_docx_page_without_unread_areas_has_no_note():
    texts = [p.text for p in read_docx(three_pages())]
    assert texts[texts.index("Page 2") + 1] == "c"


def test_docx_singular_note_follows_the_not_read_paragraph():
    texts = [p.text for p in read_docx(three_pages())]
    at = texts.index("Page 3")
    assert texts[at + 1 : at + 3] == ["This page was not read.", NOTE_3]


def test_docx_yellow_runs_equal_unresolved_count():
    assert len(yellow(read_docx(three_pages()))) == 3


def test_docx_review_line_stays_highlighted():
    runs = [r for p in read_docx(three_pages()) for r in p.runs if r.text == "a"]
    assert [r.font.highlight_color for r in runs] == [WD_COLOR_INDEX.YELLOW]


@pytest.fixture
def reader(monkeypatch):
    table = {}

    def fake(path, *, languages, **_kw):
        result = table[path.name]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(pipeline, "read_path", fake)
    return SimpleNamespace(table=table)


def run(capsys, *argv):
    code = cli.main(list(argv), {})
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_read_reports_each_page_with_unread_areas(reader, capsys):
    reader.table["a.png"] = d = three_pages()
    code, out, err = run(capsys, "read", "a.png")
    assert (code, out) == (0, to_txt(d))
    assert err == f"{SUMMARY}\na.png: {AREA_1}\na.png: {AREA_3}\n"


def test_read_puts_a_page_error_before_its_area_line(reader, capsys):
    reader.table["a.png"] = three_pages(error="Tesseract did not answer")
    code, _, err = run(capsys, "read", "a.png")
    expected = (
        f"{SUMMARY}\na.png: {AREA_1}\na.png: page 3: Tesseract did not answer\n"
        f"a.png: {AREA_3}\nhint: run folio doctor to check the readers\n"
    )
    assert (code, err) == (1, expected)


def test_read_saves_text_then_lists_areas(reader, capsys, tmp_path):
    reader.table["a.png"] = d = three_pages()
    code, _, err = run(capsys, "read", "a.png", "--out", str(tmp_path), "--format", "txt")
    target = tmp_path / "a.txt"
    assert code == 0 and target.read_bytes() == to_txt(d).encode()
    assert err == f"{SUMMARY} (saved to {target})\na.png: {AREA_1}\na.png: {AREA_3}\n"


def test_read_json_to_stdout_counts_unread_areas(reader, capsys):
    reader.table["a.png"] = three_pages()
    code, out, err = run(capsys, "read", "a.png", "--format", "json")
    assert code == 0 and json.loads(out)["unresolved_count"] == 3
    assert err.endswith(f"a.png: {AREA_1}\na.png: {AREA_3}\n")


def test_read_lists_areas_after_a_failed_save(reader, capsys, tmp_path, monkeypatch):
    def refuse(target, data):
        raise PermissionError(errno.EACCES, "x")

    monkeypatch.setattr(cli, "_write_atomic", refuse)
    reader.table["a.png"] = three_pages()
    code, _, err = run(capsys, "read", "a.png", "--out", str(tmp_path))
    expected = f"error: permission denied: {tmp_path / 'a.txt'}\na.png: {AREA_1}\na.png: {AREA_3}\n"
    assert (code, err) == (1, expected)


def test_read_shows_a_control_character_in_the_source_as_a_question_mark(reader, capsys):
    reader.table["a\x07.png"] = doc([page(1, [line("a")], regions=3)])
    _, _, err = run(capsys, "read", "a\x07.png")
    assert err.endswith(f"a?.png: {AREA_1}\n")


def test_read_without_unread_areas_prints_only_the_summary(reader, capsys):
    reader.table["a.png"] = doc([page(1, [line("a")])])
    _, _, err = run(capsys, "read", "a.png")
    assert err == "a.png: 1 lines found, 1 read, 0 need review, 0 unread pages\n"
