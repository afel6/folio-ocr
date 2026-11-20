from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from folio_ocr.document import IDENTITY, Document, Engine, Line, Page, Source, line_id


def make_line() -> Line:
    return Line(
        id=line_id(1, 1), box=(0.0, 0.0, 10.0, 5.0), text="hi", direction="ltr",
        script="latin", reader="text_layer", confidence=None, confidence_method="none",
        flags=(), status="auto_accepted", final_text="hi",
    )


def make_document() -> Document:
    page = Page(
        index=1, width=100, height=200, render_dpi=None, rotation_applied=0.0,
        transform=IDENTITY,
        origin="text_layer", direction="ltr", status="read", flags=(),
        unread_regions=(), lines=(make_line(),),
    )
    engine = Engine("d", None, "r", None, None, "untested", None)
    return Document(
        id="doc1", source=Source("a.pdf", "ab" * 32, "pdf", 1), languages="en",
        engine=engine, pages=(page,), created_at=datetime(2026, 1, 1, tzinfo=UTC),
        has_edits=False,
    )


def test_construction_keeps_values():
    doc = make_document()
    assert doc.pages[0].lines[0].text == "hi"
    assert doc.source.page_count == 1
    assert doc.engine.model_fit == "untested"


def test_frozen():
    doc = make_document()
    with pytest.raises(FrozenInstanceError):
        doc.id = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        doc.pages[0].lines[0].text = "x"  # type: ignore[misc]


def test_replace_returns_new_object():
    line = make_line()
    other = replace(line, text="yo")
    assert line.text == "hi" and other.text == "yo"


@pytest.mark.parametrize("page,k,expected", [(1, 1, "p1-l1"), (12, 345, "p12-l345")])
def test_line_id_format(page, k, expected):
    assert line_id(page, k) == expected


@pytest.mark.parametrize("page,k", [(0, 1), (1, 0), (-1, 3), (2, -5)])
def test_line_id_rejects_below_one(page, k):
    with pytest.raises(ValueError):
        line_id(page, k)


def test_identity_is_the_3x3_identity_matrix():
    assert IDENTITY == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_page_keeps_its_transform():
    page = make_document().pages[0]
    assert page.transform == IDENTITY
    moved = ((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.0, 0.0, 1.0))
    assert replace(page, transform=moved).transform == moved
    assert page.transform == IDENTITY
