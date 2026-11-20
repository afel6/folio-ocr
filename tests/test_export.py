"""Tests for the TXT, JSON and DOCX exports."""

import dataclasses
import io
import json
from datetime import UTC, datetime

import docx
import pytest
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn

from folio_ocr.document import IDENTITY, Document, Engine, Line, Page, Source
from folio_ocr.export import to_docx, to_json, to_txt, unresolved_count, visible

RAW = "a\x00b\x01\t\n\r\x1f\x7f\x80\x9f￾￿z"
LONE_SURROGATE = chr(0xD800)
ARABIC = "المبلغ 350.75 ريال"


def make_line(k, text, final=None, status="auto_accepted", direction="ltr", conf=0.9, y=0.0):
    return Line(
        id=f"p1-l{k}",
        box=(0.0, y, 10.0, y + 5.0),
        text=text,
        direction=direction,
        script="arabic" if direction == "rtl" else "latin",
        reader="model",
        confidence=conf,
        confidence_method="reader",
        flags=("low_confidence",) if status == "needs_review" else (),
        status=status,
        final_text=text if final is None else final,
    )


def make_page(index, status, lines):
    return Page(
        index=index,
        width=100,
        height=200,
        render_dpi=None,
        rotation_applied=0.0,
        transform=IDENTITY,
        origin="ocr",
        direction="ltr",
        status=status,
        flags=(),
        unread_regions=(),
        lines=tuple(lines),
    )


def make_doc(lines=None):
    if lines is None:
        lines = (
            make_line(3, "third", y=50.0),
            make_line(1, ARABIC, status="needs_review", direction="rtl", y=90.0),
            make_line(2, "old", final="edited text", status="edited", y=10.0),
        )
    return Document(
        id="doc1",
        source=Source("a.pdf", "ab" * 32, "pdf", 2),
        languages="en+ar",
        engine=Engine("det", None, "reader", "m", "h", "pass", None),
        pages=(make_page(1, "read", lines), make_page(2, "unread", ())),
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        has_edits=True,
    )


def with_final(text):
    return make_doc((make_line(1, "x", final=text),))


def test_unresolved_count_only_needs_review():
    assert unresolved_count(make_doc()) == 1
    other = make_doc(
        (
            make_line(1, "a", status="edited"),
            make_line(2, "b", status="confirmed"),
            make_line(3, "c"),
        )
    )
    assert unresolved_count(other) == 0


def test_unresolved_count_spans_pages():
    doc = make_doc()
    extra = make_page(2, "read", (make_line(1, "z", status="needs_review"),))
    doc = dataclasses.replace(doc, pages=(doc.pages[0], extra))
    assert unresolved_count(doc) == 2


def test_to_txt_exact_uses_final_text_and_stored_order():
    assert to_txt(make_doc()) == f"third\n{ARABIC}\nedited text\n\f"


def test_to_txt_pages_and_empty_page():
    doc = make_doc((make_line(1, "a"), make_line(2, "b")))
    assert to_txt(doc) == "a\nb\n\f"
    p2 = make_page(2, "read", (make_line(1, "c"),))
    two = dataclasses.replace(doc, pages=(doc.pages[0], p2))
    assert to_txt(two) == "a\nb\n\fc\n"


def test_to_txt_keeps_control_characters():
    assert to_txt(with_final(RAW)) == RAW + "\n\f"


def test_to_json_keeps_control_characters():
    data = json.loads(to_json(with_final(RAW)))
    assert data["pages"][0]["lines"][0]["final_text"] == RAW
    assert data["pages"][0]["lines"][0]["text"] == "x"


def test_to_json_round_trip_and_fields():
    doc = make_doc()
    data = json.loads(to_json(doc))
    assert list(data) == [f.name for f in dataclasses.fields(Document)] + ["unresolved_count"]
    assert data["unresolved_count"] == 1
    assert data["created_at"] == "2026-01-02T03:04:05+00:00"
    assert data["has_edits"] is True
    assert data["engine"]["model_fit"] == "pass"
    assert data["source"]["page_count"] == 2
    line = data["pages"][0]["lines"][1]
    assert list(line) == [f.name for f in dataclasses.fields(Line)]
    assert line["flags"] == ["low_confidence"]
    assert line["box"] == [0.0, 90.0, 10.0, 95.0]
    assert line["confidence"] == 0.9
    assert [ln["id"] for ln in data["pages"][0]["lines"]] == ["p1-l3", "p1-l1", "p1-l2"]
    assert data["pages"][0]["transform"][0] == [1.0, 0.0, 0.0]
    assert ARABIC in to_json(doc)


def test_to_json_deterministic_and_compact():
    doc = make_doc()
    assert to_json(doc) == to_json(doc)
    compact = to_json(doc, indent=None)
    assert "\n" not in compact
    assert '"id":"doc1","source":{' in compact
    assert json.loads(compact) == json.loads(to_json(doc))


def test_to_json_rejects_nan():
    doc = make_doc((make_line(1, "a", conf=float("nan")),))
    with pytest.raises(ValueError):
        to_json(doc)


def read_back(doc):
    return docx.Document(io.BytesIO(to_docx(doc)))


def test_to_docx_structure():
    paras = read_back(make_doc()).paragraphs
    assert paras[0].text == "Lines that need review: 1 (highlighted)."
    headings = [(p.text, p.style.name) for p in paras if p.style.name.startswith("Heading")]
    assert headings == [("Page 1", "Heading 2"), ("Page 2", "Heading 2")]
    texts = [p.text for p in paras]
    assert texts[1:5] == ["Page 1", "third", ARABIC, "edited text"]
    assert texts[5:] == ["Page 2", "This page was not read."]
    assert paras[-1].runs[0].italic is True


def test_to_docx_read_page_has_no_unread_note():
    texts = [p.text for p in read_back(make_doc()).paragraphs]
    assert texts.count("This page was not read.") == 1


def test_to_docx_direction_and_highlight():
    ltr, rtl, edited = read_back(make_doc()).paragraphs[2:5]
    assert rtl._p.pPr.find(qn("w:bidi")) is not None
    assert ltr._p.pPr is None or ltr._p.pPr.find(qn("w:bidi")) is None
    assert edited._p.pPr is None or edited._p.pPr.find(qn("w:bidi")) is None
    assert rtl.runs[0].font.highlight_color == WD_COLOR_INDEX.YELLOW
    assert ltr.runs[0].font.highlight_color is None
    assert edited.runs[0].font.highlight_color is None
    assert len(rtl.runs) == 1


def test_to_docx_control_characters_are_visible():
    raw = RAW + LONE_SURROGATE
    p = read_back(with_final(raw)).paragraphs[2]
    assert p.text == visible(raw)
    assert p._p.findall(".//" + qn("w:br")) == []
    assert p._p.findall(".//" + qn("w:tab")) == []


def test_visible_classes():
    assert visible("\x00") == "␀"
    assert visible("\n") == "␊"
    assert visible("\t\r\x1f") == "␉␍␟"
    assert visible("\x7f") == "␡"
    assert visible("\x80\x9f") == "[U+0080][U+009F]"
    assert visible(chr(0xD800) + chr(0xDFFF)) == "[U+D800][U+DFFF]"
    assert visible("￾￿") == "[U+FFFE][U+FFFF]"


def test_visible_leaves_ordinary_text():
    s = "abc " + ARABIC + " " + chr(0x200F) + " " + chr(0xFFFD) + "   é " + chr(0x1F600)
    assert visible(s) == s
