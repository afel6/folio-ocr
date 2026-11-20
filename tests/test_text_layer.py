import json
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from fpdf import FPDF
from fpdf.enums import TextMode

from folio_ocr.document import TextChar
from folio_ocr.text_layer import (
    GAP_FACTOR,
    _combine_surrogates,
    line_box,
    line_text,
    page_chars,
    read_text_layer,
    segment_lines,
)

FIXTURES = Path(__file__).parent / "fixtures"
FONTS = FIXTURES / "fonts"
SHY = chr(0xAD)
AR_LINE = "فاتورة رقم ١٢٣٤ بتاريخ ٢٠٢٤/٠٣/١٥"
EN_LINE = "Invoice 42 for Ahmed Hassan"


def new_pdf(font_file: str, shaping: bool = True) -> FPDF:
    pdf = FPDF(unit="pt", format=(420, 595))
    pdf.set_auto_page_break(False)
    pdf.add_font("F", fname=str(FONTS / font_file))
    pdf.set_font("F", size=16)
    if shaping:
        pdf.set_text_shaping(True)
    pdf.add_page()
    return pdf


def open_page(pdf: FPDF, tmp_path):
    path = tmp_path / "t.pdf"
    pdf.output(str(path))
    doc = pdfium.PdfDocument(str(path))
    return doc, doc[0]


def glyphs(text, x, y0=0.0, y1=10.0, width=5.0):
    """One boxed character per letter of `text`, side by side from `x`."""
    return [TextChar(c, (x + i * width, y0, x + (i + 1) * width, y1)) for i, c in enumerate(text)]


def texts(lines):
    return [line_text(line) for line in lines]


def table_row(scale):
    """Two words 0.3 x height apart, then a cell break 1.0 x height away, at `scale`."""
    s = scale
    return (
        glyphs("ab", 0, y1=10 * s, width=5 * s)
        + [TextChar(" ", None)]
        + glyphs("cd", 13 * s, y1=10 * s, width=5 * s)
        + [TextChar(" ", None)]
        + glyphs("ef", 33 * s, y1=10 * s, width=5 * s)
    )


# segment_lines on hand-made characters
def test_rows_run_top_to_bottom_whatever_the_index_order():
    chars = glyphs("cd", 0, 30, 40) + glyphs("ab", 0, 0, 10)
    assert texts(segment_lines(chars, "ltr")) == ["ab", "cd"]


def test_gap_wider_than_factor_splits_a_row():
    width = 10 * GAP_FACTOR + 1
    chars = glyphs("a", 0) + glyphs("b", 5 + width)
    assert texts(segment_lines(chars, "ltr")) == ["a", "b"]


def test_gap_narrower_than_factor_keeps_a_row():
    width = 10 * GAP_FACTOR - 1
    chars = glyphs("a", 0) + glyphs("b", 5 + width)
    assert texts(segment_lines(chars, "ltr")) == ["ab"]


def test_explicit_factor_decides_the_split():
    chars = glyphs("a", 0) + glyphs("b", 5 + 5)
    assert texts(segment_lines(chars, "ltr", factor=0.4)) == ["a", "b"]
    assert texts(segment_lines(chars, "ltr", factor=0.6)) == ["ab"]


@pytest.mark.parametrize("scale", [1, 4])
def test_same_layout_splits_the_same_at_every_scale(scale):
    assert texts(segment_lines(table_row(scale), "ltr")) == ["ab cd", "ef"]


def test_whitespace_and_boxless_inside_a_segment_stay():
    chars = glyphs("a", 0) + [TextChar(SHY, None), TextChar(" ", None)] + glyphs("b", 5)
    assert texts(segment_lines(chars, "ltr")) == ["a" + SHY + " b"]


def test_whitespace_and_boxless_between_segments_are_dropped():
    chars = glyphs("a", 0) + [TextChar(SHY, None), TextChar(" ", (5.0, 0.0, 30.0, 10.0))]
    chars += glyphs("b", 100)
    assert texts(segment_lines(chars, "ltr")) == ["a", "b"]


def test_segments_are_trimmed():
    chars = [TextChar(" ", None)] + glyphs("ab", 0) + [TextChar(" ", None)]
    assert texts(segment_lines(chars, "ltr")) == ["ab"]


def test_rtl_orders_segments_right_to_left_and_ltr_left_to_right():
    chars = glyphs("ab", 100) + glyphs("cd", 0)
    assert texts(segment_lines(chars, "rtl")) == ["ab", "cd"]
    assert texts(segment_lines(chars, "ltr")) == ["cd", "ab"]


def test_characters_in_a_segment_keep_index_order():
    chars = [
        TextChar("a", (5.0, 0.0, 10.0, 10.0)),
        TextChar("b", (0.0, 0.0, 5.0, 10.0)),
    ]
    assert texts(segment_lines(chars, "ltr")) == ["ab"]


def test_short_glyph_joins_the_row_band_not_just_the_previous_glyph():
    tall = TextChar("T", (0.0, -10.0, 5.0, 12.0))
    dot = TextChar(".", (5.0, 0.0, 10.0, 3.0))
    short = TextChar("s", (10.0, 9.0, 15.0, 12.0))
    assert texts(segment_lines([tall, dot, short], "ltr")) == ["T.s"]


def test_empty_input_gives_no_lines():
    assert segment_lines([], "ltr") == ()


# line_box
def test_line_box_unions_tight_boxes_of_non_whitespace_chars():
    chars = [
        TextChar("a", (0.0, 0.0, 10.0, 10.0), tight_box=(1.0, 2.0, 8.0, 9.0)),
        TextChar(" ", (10.0, 0.0, 20.0, 10.0), tight_box=(10.0, 0.0, 20.0, 10.0)),
        TextChar("b", (20.0, 0.0, 30.0, 10.0), tight_box=(22.0, 1.0, 29.0, 8.0)),
    ]
    assert line_box(chars) == (1.0, 1.0, 29.0, 9.0)


def test_line_box_falls_back_to_loose_box_without_a_tight_box():
    chars = [
        TextChar("a", (0.0, 0.0, 10.0, 10.0), tight_box=(1.0, 2.0, 8.0, 9.0)),
        TextChar("b", (20.0, 0.0, 30.0, 12.0)),
    ]
    assert line_box(chars) == (1.0, 0.0, 30.0, 12.0)


def test_line_box_without_boxes_is_none():
    assert line_box([TextChar("a", None), TextChar(" ", (0.0, 0.0, 5.0, 5.0))]) is None
    assert line_box([]) is None


# real PDFs
def test_tight_box_lies_inside_loose_box_horizontally(tmp_path):
    pdf = new_pdf("NotoSans-Regular.ttf")
    pdf.text(20, 60, EN_LINE)
    doc, page = open_page(pdf, tmp_path)
    try:
        boxed = [c for c in page_chars(page) if c.box]
        assert boxed
        for c in boxed:
            assert c.box[0] - 1e-3 <= c.tight_box[0] <= c.tight_box[2] <= c.box[2] + 1e-3
    finally:
        doc.close()


@pytest.mark.parametrize(
    ("mode", "invisible"),
    [(TextMode.FILL, False), (TextMode.INVISIBLE, True), (TextMode.CLIP, True)],
)
def test_render_mode_decides_invisible(tmp_path, mode, invisible):
    pdf = new_pdf("NotoSans-Regular.ttf")
    pdf.text_mode = mode
    pdf.text(20, 60, EN_LINE)
    doc, page = open_page(pdf, tmp_path)
    try:
        chars = [c for c in page_chars(page) if c.text.strip() and c.box]
        assert chars
        assert {c.invisible for c in chars} == {invisible}
    finally:
        doc.close()


def test_mixed_page_two_lines(tmp_path):
    pdf = new_pdf("Amiri-Regular.ttf")
    pdf.set_xy(10, 60)
    pdf.cell(400, 24, AR_LINE, align="R")
    pdf.add_font("N", fname=str(FONTS / "NotoSans-Regular.ttf"))
    pdf.set_font("N", size=16)
    pdf.set_xy(10, 140)
    pdf.cell(400, 24, EN_LINE, align="R")
    doc, page = open_page(pdf, tmp_path)
    try:
        lines = read_text_layer(page)
        assert texts(lines) == [AR_LINE, EN_LINE]
        for ch in (c for line in lines for c in line if c.box):
            x0, y0, x1, y1 = ch.box
            assert 0 <= x0 < x1 <= 1750 and 0 <= y0 < y1 <= 2479
        assert read_text_layer(page, dpi=72) != ()
    finally:
        doc.close()


def test_combine_surrogates_joins_pairs_and_keeps_lone_halves():
    pair = [(3, 0xD83D, False), (4, 0xDE00, False)]
    assert _combine_surrogates(pair) == [(3, 0x1F600, False)]
    assert _combine_surrogates([(0, 0xD83D, False), (1, 65, False)]) == [
        (0, 0xD83D, False),
        (1, 65, False),
    ]
    assert _combine_surrogates([(0, 0xDE00, True)]) == [(0, 0xDE00, True)]


def test_unmapped_glyphs_are_marked(tmp_path):
    pdf = new_pdf("NotoNaskhArabic-Regular.ttf")
    pdf.set_xy(10, 60)
    pdf.cell(400, 24, "بتاريخ أحمد", align="R")
    doc, page = open_page(pdf, tmp_path)
    try:
        assert any(c.unmapped for c in page_chars(page))
    finally:
        doc.close()


def test_blank_page_has_no_lines(tmp_path):
    doc, page = open_page(new_pdf("NotoSans-Regular.ttf"), tmp_path)
    try:
        assert read_text_layer(page) == ()
    finally:
        doc.close()


@pytest.mark.parametrize("dpi", [0, -300, 1.5, "300", True])
def test_invalid_dpi(tmp_path, dpi):
    doc, page = open_page(new_pdf("NotoSans-Regular.ttf"), tmp_path)
    try:
        with pytest.raises(ValueError):
            page_chars(page, dpi)
    finally:
        doc.close()


def test_english_letter_reads_fourteen_lines_starting_with_ground_truth():
    gt = json.loads((FIXTURES / "en-letter.gt.json").read_text(encoding="utf-8"))
    doc = pdfium.PdfDocument(str(FIXTURES / "en-letter.pdf"))
    try:
        lines = read_text_layer(doc[0])
    finally:
        doc.close()
    assert len(lines) == 14
    assert line_text(lines[0]) == gt["lines"][0]["text"]
