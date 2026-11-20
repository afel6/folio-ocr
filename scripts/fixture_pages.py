"""Page drawing for the fixtures: layout of text cells, shaped and visual-order PDFs, line strips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fixture_fonts import font_key, font_path
from fixture_images import SCALE, ink_box, render_pdf
from fixture_text import DocText, TextLine
from fpdf import FPDF
from PIL import Image

PAGE_W, PAGE_H, MARGIN, FONT_PT, PITCH = 420, 595, 30, 11, 22
TABLE_WIDTHS, GUTTER = (140, 50, 70, 70), 10
FIXED_DATE = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Placed:
    line: TextLine
    x: float
    y: float
    w: float


def _table_spans(scale: float) -> list[tuple[float, float]]:
    spans, x = [], MARGIN * scale
    for width in TABLE_WIDTHS:
        spans.append((x, width * scale))
        x += (width + GUTTER) * scale
    return spans


def layout_document(doc: DocText, scale: float = 1.0) -> list[Placed]:
    """Place every cell of every row in reading order (table rows mirrored for rtl).

    `scale` multiplies the page size, margin, row pitch, column widths and gutter.
    """
    placed, page_w = [], PAGE_W * scale
    for index, row in enumerate(doc.rows):
        if len(row.cells) == 1:
            spans = [(MARGIN * scale, page_w - 2 * MARGIN * scale)]
        elif len(row.cells) == len(TABLE_WIDTHS):
            spans = _table_spans(scale)
        else:
            raise ValueError(f"a row has {len(row.cells)} cells; expected 1 or 4")
        if doc.direction == "rtl":
            spans = [(page_w - x - w, w) for x, w in spans]
        y = (MARGIN + index * PITCH) * scale
        placed += [Placed(c, x, y, w) for c, (x, w) in zip(row.cells, spans, strict=True)]
    return placed


def blank_pdf(width: float, height: float) -> FPDF:
    """A PDF with the fixed creation date, no automatic page breaks and no pages yet."""
    pdf = FPDF(unit="pt", format=(width, height))
    pdf.set_auto_page_break(False)
    pdf.set_creation_date(FIXED_DATE)
    pdf.c_margin = 2
    return pdf


def new_pdf(arabic_key: str, shaping: bool = True, scale: float = 1.0) -> FPDF:
    pdf = blank_pdf(PAGE_W * scale, PAGE_H * scale)
    pdf.c_margin = 2 * scale
    for key in ("sans", arabic_key):
        pdf.add_font(key, fname=str(font_path(key)))
    pdf.set_text_shaping(shaping)
    if arabic_key == "naskh":
        pdf.set_fallback_fonts(["sans"])
    pdf.add_page()
    return pdf


def draw_cells(
    pdf: FPDF,
    placed: list[Placed],
    arabic_key: str,
    scale: float = 1.0,
    texts: list[str] | None = None,
    fonts: list[str] | None = None,
) -> None:
    """Draw each placed cell on the current page; raise ValueError if a text does not fit its cell.

    `texts` and `fonts` replace the drawn text and its font per cell (the ground truth keeps `line.text`).
    """
    for i, p in enumerate(placed):
        text = p.line.text if texts is None else texts[i]
        pdf.set_font(fonts[i] if fonts else font_key(text, arabic_key), size=FONT_PT * scale)
        if pdf.get_string_width(text) > p.w - 2 * scale:
            raise ValueError(f"line is wider than its cell: {text!r}")
        pdf.set_xy(p.x, p.y)
        pdf.cell(p.w, PITCH * scale, text, align="R" if p.line.direction == "rtl" else "L")


def draw_pdf(placed: list[Placed], arabic_key: str, scale: float = 1.0) -> bytes:
    """Draw shaped text into a PDF; raise ValueError if a line does not fit its cell."""
    pdf = new_pdf(arabic_key, scale=scale)
    draw_cells(pdf, placed, arabic_key, scale)
    return bytes(pdf.output())


def draw_chars(pdf: FPDF, text: str, x: float, baseline: float, rtl: bool) -> None:
    """Draw `text` one character at a time from `x`, leftwards when `rtl`, in logical order."""
    for ch in text:
        step = pdf.get_string_width(ch)
        x -= step if rtl else 0
        pdf.text(x, baseline, ch)
        x += 0 if rtl else step


def draw_visual_order_pdf(p: Placed) -> bytes:
    """Draw a one-line page one unshaped character at a time, right to left for rtl text.

    pdfium extracts such a page in visual order (letters reversed) only when the page holds a single line.
    """
    pdf = new_pdf("amiri", shaping=False)
    pdf.set_font(font_key(p.line.text, "amiri"), size=FONT_PT)
    baseline = p.y + PITCH / 2 + 0.35 * FONT_PT
    rtl = p.line.direction == "rtl"
    draw_chars(pdf, p.line.text, p.x + p.w - 1 if rtl else p.x + 1, baseline, rtl)
    return bytes(pdf.output())


def cell_boxes(img: Image.Image, placed: list[Placed], scale: float = 1.0) -> list[list[int]]:
    pixels = np.asarray(img)
    return [
        ink_box(
            pixels,
            (
                round(p.x * SCALE),
                round(p.y * SCALE),
                round((p.x + p.w) * SCALE),
                round((p.y + PITCH * scale) * SCALE),
            ),
        )
        for p in placed
    ]


def render_line_strip(
    text: str, font_file: Path, fallback_file: Path | None = None
) -> tuple[bytes, Image.Image]:
    """Draw one shaped line (24 pt, right aligned) and return its PDF bytes and 300 dpi render."""
    pdf = blank_pdf(220, 50)
    pdf.add_font("main", fname=str(font_file))
    if fallback_file is not None:
        pdf.add_font("fallback", fname=str(fallback_file))
    pdf.set_text_shaping(True)
    if fallback_file is not None:
        pdf.set_fallback_fonts(["fallback"])
    pdf.add_page()
    pdf.set_font("main", size=24)
    pdf.set_xy(10, 5)
    pdf.cell(200, 40, text, align="R")
    data = bytes(pdf.output())
    return data, render_pdf(data)
