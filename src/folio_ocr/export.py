"""Export a document as plain text, JSON or a Word file."""

import dataclasses
import io
import json

import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement

from folio_ocr.document import Document, Line, Page


def review_line_count(doc: Document) -> int:
    """Number of lines still waiting for review, across all pages."""
    return sum(line.status == "needs_review" for page in doc.pages for line in page.lines)


def unresolved_count(doc: Document) -> int:
    """Lines still waiting for review, plus the pages that have areas nobody read."""
    return review_line_count(doc) + sum(bool(page.unread_regions) for page in doc.pages)


def unread_note(page: Page) -> str | None:
    """A one-line note saying how many areas of `page` were not read; None when there are none."""
    count = len(page.unread_regions)
    if count == 0:
        return None
    if count == 1:
        return f"[page {page.index}: 1 area was not read; the JSON export lists its box]"
    return f"[page {page.index}: {count} areas were not read; the JSON export lists their boxes]"


def _page_txt(page: Page) -> str:
    note = unread_note(page)
    head = "" if note is None else note + "\n"
    return head + "".join(line.final_text + "\n" for line in page.lines)


def to_txt(doc: Document) -> str:
    """The final text of every line, one line per row, pages separated by a form feed. A page
    with areas that were not read starts with a bracketed note saying how many."""
    return "\f".join(_page_txt(page) for page in doc.pages)


def to_json(doc: Document, *, indent: int | None = 2) -> str:
    """The whole document as JSON; its last key, `unresolved_count`, is the lines needing review
    plus the pages with areas that were not read."""
    obj = dataclasses.asdict(doc)
    obj["created_at"] = doc.created_at.isoformat()
    obj["unresolved_count"] = unresolved_count(doc)
    separators = (",", ":") if indent is None else None
    return json.dumps(
        obj, ensure_ascii=False, allow_nan=False, indent=indent, separators=separators
    )


def visible(text: str) -> str:
    """Replace characters a Word file cannot hold, and control characters, with visible marks."""
    return "".join(_mark(ch) for ch in text)


def _mark(ch: str) -> str:
    cp = ord(ch)
    if cp <= 0x1F:
        return chr(0x2400 + cp)
    if cp == 0x7F:
        return "␡"
    if 0x80 <= cp <= 0x9F or 0xD800 <= cp <= 0xDFFF or cp in (0xFFFE, 0xFFFF):
        return f"[U+{cp:04X}]"
    return ch


# Elements that follow w:bidi in a paragraph's properties, in schema order.
_AFTER_BIDI = (
    "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
    "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
    "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
    "w:pPrChange",
)  # fmt: skip


def _mark_rtl(paragraph) -> None:
    pr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    pr.insert_element_before(bidi, *_AFTER_BIDI)


def _add_line(document, line: Line) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(visible(line.final_text))
    if line.direction == "rtl":
        _mark_rtl(paragraph)
    if line.status == "needs_review":
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def to_docx(doc: Document) -> bytes:
    """A Word file with a review summary, then each page's lines in reading order. A page with
    areas that were not read gets a highlighted note under its heading."""
    document = docx.Document()
    document.add_paragraph(f"Lines that need review: {review_line_count(doc)} (highlighted).")
    for page in doc.pages:
        document.add_heading(f"Page {page.index}", level=2)
        if page.status != "read":
            document.add_paragraph().add_run("This page was not read.").italic = True
        note = unread_note(page)
        if note is not None:
            document.add_paragraph().add_run(note).font.highlight_color = WD_COLOR_INDEX.YELLOW
        for line in page.lines:
            _add_line(document, line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
