"""Page layout and drawing for the development-variant documents (A4, shaped text, 300 dpi)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import uharfbuzz as hb
from dev_fonts import dev_font_path
from fixture_images import SCALE, ink_box, render_pdf
from fixture_pages import blank_pdf
from fixture_text import DocText, TextLine
from fontTools.ttLib import TTFont
from fpdf import FPDF
from PIL import Image

PAGE_W, PAGE_H, MARGIN = 595, 842, 40
PITCH = 1.6  # line pitch in font sizes for Latin-only pages
# Arabic fonts reach further above and below the baseline (Scheherazade New spans about 2.1 em),
# so pages that may draw Arabic get a taller line for their Arabic font.
ARABIC_PITCH = {"scheherazade": 2.4, "plexarabic": 2.0, "lateef": 2.0}
MIN_SIZE = 7
RECEIPT_WEIGHTS = (0.7, 0.3)
ARABIC_RUN = re.compile("[؀-ۿ]+")
# A detected word box is about this many font sizes tall; narrow table gaps are set in box heights.
BOX_HEIGHT = 1.25


class DoesNotFit(ValueError):
    """A text is wider than its cell, or the rows run past the page."""


@dataclass(frozen=True)
class Cell:
    line: TextLine
    x: float
    y: float
    w: float
    h: float


@cache
def _cmap(key: str) -> frozenset[int]:
    return frozenset(TTFont(dev_font_path(key)).getBestCmap())


@cache
def _hb_font(key: str) -> hb.Font:
    return hb.Font(hb.Face(dev_font_path(key).read_bytes()))


def has_arabic(text: str) -> bool:
    return ARABIC_RUN.search(text) is not None


def font_key(text: str, latin_key: str, arabic_key: str) -> str:
    """The Arabic font for a line with Arabic-block characters, else the Latin font."""
    return arabic_key if has_arabic(text) else latin_key


def check_text(text: str, latin_key: str, arabic_key: str) -> None:
    """Raise ValueError if a character has no glyph or an Arabic run shapes to glyph id 0."""
    main = font_key(text, latin_key, arabic_key)
    allowed = _cmap(main) | (_cmap(latin_key) if main == arabic_key else frozenset())
    missing = sorted({c for c in text if ord(c) not in allowed})
    if missing:
        raise ValueError(f"{main} has no glyph for {missing!r} in {text!r}")
    for run in ARABIC_RUN.findall(text):
        buf = hb.Buffer()
        buf.add_str(run)
        buf.direction, buf.script, buf.language = "rtl", "Arab", "ar"
        hb.shape(_hb_font(arabic_key), buf, {})
        if any(info.codepoint == 0 for info in buf.glyph_infos):
            raise ValueError(f"{arabic_key} shapes {run!r} to a missing glyph")


def _spans(layout: str, count: int, size: float) -> list[tuple[float, float]]:
    """(x, width) of each column of a row with `count` cells, left to right."""
    width, gap = PAGE_W - 2 * MARGIN, 2 * size
    if count == 1:
        return [(MARGIN, width)]
    if layout == "grid":
        stride = (width + gap) / count
        return [(MARGIN + i * stride, stride - gap) for i in range(count)]
    weights = RECEIPT_WEIGHTS if layout == "receipt" else (1 / count,) * count
    usable, x, spans = width - gap * (count - 1), float(MARGIN), []
    for weight in weights:
        spans.append((x, usable * weight))
        x += usable * weight + gap
    return spans


def line_pitch(language: str, arabic_key: str) -> float:
    """Line pitch in font sizes: Latin-only pages keep PITCH, others the Arabic font's pitch."""
    return PITCH if language == "en" else ARABIC_PITCH[arabic_key]


def _fitted_spans(
    doc: DocText, size: float, gap: float, measure: Callable[[str], float]
) -> list[tuple[float, float]]:
    """Columns as wide as their longest text, `gap` box heights apart (a tight table)."""
    count = max(len(row.cells) for row in doc.rows)
    widths = [0.4 * size + 2] * count
    for row in doc.rows:
        for k, cell in enumerate(row.cells):
            widths[k] = max(widths[k], measure(cell.text) + 0.4 * size + 2)
    space, x, spans = gap * BOX_HEIGHT * size, float(MARGIN), []
    for width in widths:
        spans.append((x, width))
        x += width + space
    if x - space > PAGE_W - MARGIN:
        raise DoesNotFit(f"a tight table of {size} pt is wider than the page")
    return spans


def layout_cells(
    doc: DocText,
    size: float,
    pitch_factor: float = PITCH,
    table_gap: float | None = None,
    measure: Callable[[str], float] | None = None,
) -> list[Cell]:
    """Place every cell of every row in reading order (columns mirrored for rtl documents).
    A table with `table_gap` gets columns fitted to their text, that many box heights apart."""
    pitch, cells = pitch_factor * size, []
    tight = None
    if doc.layout == "table" and table_gap is not None and measure is not None:
        tight = _fitted_spans(doc, size, table_gap, measure)
    for index, row in enumerate(doc.rows):
        y = MARGIN + index * pitch
        if y + pitch > PAGE_H - MARGIN:
            raise DoesNotFit(f"{len(doc.rows)} rows of {size} pt do not fit the page")
        spans = tight[: len(row.cells)] if tight else _spans(doc.layout, len(row.cells), size)
        if doc.direction == "rtl":
            spans = [(PAGE_W - x - w, w) for x, w in spans]
        cells += [Cell(c, x, y, w, pitch) for c, (x, w) in zip(row.cells, spans, strict=True)]
    return cells


def _family(key: str) -> str:
    """PDF family name; the prefix keeps "courier" from meaning the built-in core font."""
    return f"dev-{key}"


def _new_pdf(latin_key: str, arabic_key: str) -> FPDF:
    pdf = blank_pdf(PAGE_W, PAGE_H)
    for key in (latin_key, arabic_key):
        pdf.add_font(_family(key), fname=str(dev_font_path(key)))
    pdf.set_text_shaping(True)
    pdf.set_fallback_fonts([_family(latin_key)])
    pdf.add_page()
    return pdf


def draw_pdf(cells: list[Cell], latin_key: str, arabic_key: str, size: float) -> bytes:
    """Draw shaped text into a PDF; raise DoesNotFit if a text is wider than its cell."""
    pdf = _new_pdf(latin_key, arabic_key)
    pdf.c_margin = 0.2 * size
    for cell in cells:
        text = cell.line.text
        pdf.set_font(_family(font_key(text, latin_key, arabic_key)), size=size)
        if pdf.get_string_width(text) > cell.w - 2 * pdf.c_margin:
            raise DoesNotFit(f"text is wider than its cell: {text!r}")
        pdf.set_xy(cell.x, cell.y)
        pdf.cell(cell.w, cell.h, text, align="R" if cell.line.direction == "rtl" else "L")
    return bytes(pdf.output())


def cell_boxes(img: Image.Image, cells: list[Cell]) -> list[list[int]]:
    """Ink box of each cell in the clean render."""
    pixels = np.asarray(img)
    return [
        ink_box(
            pixels,
            (
                round(c.x * SCALE),
                round(c.y * SCALE),
                round((c.x + c.w) * SCALE),
                round((c.y + c.h) * SCALE),
            ),
        )
        for c in cells
    ]


def _measure(latin_key: str, arabic_key: str, size: float) -> Callable[[str], float]:
    pdf = _new_pdf(latin_key, arabic_key)

    def width(text: str) -> float:
        pdf.set_font(_family(font_key(text, latin_key, arabic_key)), size=size)
        return pdf.get_string_width(text)

    return width


def render_page(
    doc: DocText, latin_key: str, arabic_key: str, size: int, table_gap: float | None = None
) -> tuple[bytes, Image.Image, list[list[int]], int]:
    """Check the text, then draw it at `size` pt or the next size down that fits.

    Returns the PDF, its 300 dpi render, the ink box of every cell in reading order and the size
    used. Raises DoesNotFit when even the smallest size is too big, ValueError for a missing glyph.
    """
    for row in doc.rows:
        for cell in row.cells:
            check_text(cell.text, latin_key, arabic_key)
    for used in range(size, MIN_SIZE - 1, -1):
        try:
            measure = _measure(latin_key, arabic_key, used) if table_gap is not None else None
            pitch = line_pitch(doc.language, arabic_key)
            cells = layout_cells(doc, used, pitch, table_gap, measure)
            pdf = draw_pdf(cells, latin_key, arabic_key, used)
        except DoesNotFit:
            continue
        img = render_pdf(pdf)
        return pdf, img, cell_boxes(img, cells), used
    raise DoesNotFit(f"{doc.layout} document does not fit at {MIN_SIZE} pt or more")
