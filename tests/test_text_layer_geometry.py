"""Box mapping for rotated, offset and cropped pages, image boxes, and row grouping."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
import pytest
from pdf_variants import open_fixture, variant

from folio_ocr.document import TextChar
from folio_ocr.text_layer import (
    device_map,
    image_boxes,
    line_box,
    line_text,
    read_text_layer,
    segment_lines,
)

DPI = 300
S = DPI / 72
NAMES = ("en-invoice.pdf", "ar-invoice.pdf", "mixed-letter.pdf")
# Mean absolute grey-level difference allowed between a variant render and the original:
# resampling the same glyphs on a shifted pixel grid only changes anti-aliased edges.
PIXEL_BOUND = 1.0


def render(page: pdfium.PdfPage) -> np.ndarray:
    return np.asarray(page.render(scale=S, grayscale=True).to_pil(), dtype=np.float64)


def lines_of(page: pdfium.PdfPage) -> list[tuple[str, tuple]]:
    return [(line_text(s), line_box(s)) for s in read_text_layer(page, DPI)]


def assert_same_lines(got, want, shift=(0.0, 0.0)):
    assert [t for t, _ in got] == [t for t, _ in want]
    for (_, gb), (_, wb) in zip(got, want, strict=True):
        for g, w, d in zip(gb, wb, shift * 2, strict=True):
            assert abs(g - (w + d)) <= 1.5


@pytest.fixture(scope="module", params=NAMES)
def original(request):
    doc = open_fixture(request.param)
    page = doc[0]
    yield request.param, render(page), lines_of(page)
    doc.close()


PDFIUM_ARABIC_RESORT = (
    "pdfium re-sorts the Arabic runs of line 0 of ar-invoice.pdf at /Rotate 270: "
    "its last four characters (end of the supplier name) come out in front of the line"
)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_variant_displays_and_reads_like_original(original, rotation, request):
    name, want_img, want_lines = original
    if (name, rotation) == ("ar-invoice.pdf", 270):
        request.applymarker(pytest.mark.xfail(strict=True, reason=PDFIUM_ARABIC_RESORT))
    doc, page = variant(name, rotation)
    try:
        img = render(page)
        assert img.shape == want_img.shape
        assert np.abs(img - want_img).mean() < PIXEL_BOUND
        assert_same_lines(lines_of(page), want_lines)
    finally:
        doc.close()


def test_offset_variant_reads_like_original(original):
    name, _, want_lines = original
    doc, page = variant(name, 0, 150, 250)
    try:
        assert_same_lines(lines_of(page), want_lines)
    finally:
        doc.close()


def test_offset_rotated_variant_reads_like_original(original):
    name, _, want_lines = original
    doc, page = variant(name, 90, 150, 250)
    try:
        assert_same_lines(lines_of(page), want_lines)
    finally:
        doc.close()


def test_smaller_crop_box_shifts_boxes(original):
    name, _, want_lines = original
    doc = open_fixture(name)
    try:
        page = doc[0]
        x0, y0, x1, y1 = page.get_bbox()
        page.set_cropbox(x0 + 20, y0 + 20, x1 - 20, y1 - 20)
        assert_same_lines(lines_of(page), want_lines, shift=(-20 * S, -20 * S))
    finally:
        doc.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_line_boxes_hold_ink_and_cover_it(original, rotation):
    name = original[0]
    doc, page = variant(name, rotation, 150, 250)
    try:
        img = render(page)
        ink = img < 128
        covered = np.zeros_like(ink)
        for _, (x0, y0, x1, y1) in lines_of(page):
            region = (
                slice(max(0, math.floor(y0)), math.ceil(y1)),
                slice(max(0, math.floor(x0)), math.ceil(x1)),
            )
            assert ink[region].any()
            covered[region] = True
        assert (ink & covered).sum() >= 0.99 * ink.sum()
    finally:
        doc.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_device_map_matches_pdfium_page_to_device(rotation):
    doc, page = variant("en-invoice.pdf", rotation, 150, 250)
    try:
        width, height = math.ceil(page.get_width() * S), math.ceil(page.get_height() * S)
        to_box = device_map(page, DPI)
        x0, y0, x1, y1 = page.get_bbox()
        for x, y in ((x0 + 10, y0 + 20), (x1 - 30, y1 - 5), ((x0 + x1) / 2, (y0 + y1) / 2)):
            dx, dy = pdfium_raw.c_int(), pdfium_raw.c_int()
            pdfium_raw.FPDF_PageToDevice(
                page.raw, 0, 0, round(page.get_width() * S), round(page.get_height() * S),
                0, x, y, dx, dy,
            )  # fmt: skip
            got = to_box(x, y, x, y)
            assert abs(got[0] - dx.value) <= 1 and abs(got[1] - dy.value) <= 1
        assert width > 0 and height > 0
    finally:
        doc.close()


def test_device_map_returns_ordered_box_for_any_rotation():
    doc, page = variant("en-invoice.pdf", 270, 150, 250)
    try:
        x0, y0, x1, y1 = device_map(page, DPI)(160, 260, 200, 280)
        assert x0 < x1 and y0 < y1
    finally:
        doc.close()


def test_image_boxes_cover_the_scan_under_the_invisible_text():
    doc = open_fixture("ar-letter-invisible_ocr.pdf")
    try:
        boxes = image_boxes(doc[0])
        assert len(boxes) == 1
        x0, y0, x1, y1 = boxes[0]
        assert (x0, y0) == (0, 0)
        assert abs(x1 - 420 * S) <= 1.5 and abs(y1 - 595 * S) <= 1.5
    finally:
        doc.close()


def test_image_boxes_empty_for_text_only_page():
    doc = open_fixture("en-invoice.pdf")
    try:
        assert image_boxes(doc[0]) == ()
    finally:
        doc.close()


def test_image_boxes_find_image_inside_a_form_and_clip_to_the_page():
    doc = open_fixture("ar-letter-invisible_ocr.pdf")
    try:
        page = doc[0]
        image = next(o for o in page.get_objects(max_depth=1) if o.type == 3)
        image.set_matrix(pdfium.PdfMatrix(100, 0, 0, 100, -50, 500))  # sticks out left and top
        page.gen_content()
        x0, y0, x1, y1 = image_boxes(page)[0]
        assert x0 == 0 and y0 == 0
        assert abs(x1 - 50 * S) <= 1.5 and abs(y1 - 95 * S) <= 1.5
        image.set_matrix(pdfium.PdfMatrix(100, 0, 0, 100, -500, 500))  # fully outside
        page.gen_content()
        assert image_boxes(page) == ()
    finally:
        doc.close()


def test_image_boxes_use_form_bounds():
    src = open_fixture("ar-letter-invisible_ocr.pdf")
    dest = pdfium.PdfDocument.new()
    try:
        form = src.page_as_xobject(0, dest).as_pageobject()
        form.set_matrix(pdfium.PdfMatrix(0.5, 0, 0, 0.5, 10, 20))
        page = dest.new_page(420, 595)
        page.insert_obj(form)
        page.gen_content()
        ((x0, y0, x1, y1),) = image_boxes(page)
        assert abs(x0 - 10 * S) <= 1.5 and abs(x1 - 220 * S) <= 1.5
        assert abs(y1 - (595 - 20) * S) <= 1.5 and abs(y0 - (595 - 20 - 297.5) * S) <= 1.5
    finally:
        dest.close()
        src.close()


# --- rows from an overlap graph ---------------------------------------------------------


def glyph(x: float, y0: float, y1: float, w: float = 10.0, text: str = "n") -> TextChar:
    return TextChar(text, (x, y0, x + w, y1))


def row_texts(chars: list[TextChar]) -> list[str]:
    return [line_text(s) for s in segment_lines(chars, "ltr", factor=100.0)]


def test_mark_above_the_row_joins_it_through_a_tall_glyph():
    chars = [glyph(10 * i, 100, 140, text=t) for i, t in enumerate("abc")]
    chars.append(glyph(30, 80, 160, text="T"))  # tall glyph overlapping the row
    chars.append(glyph(32, 84, 96, w=4, text="M"))  # mark inside T only, above the row glyphs
    assert row_texts(chars) == ["abcTM"]


def test_rows_with_normal_leading_stay_apart():
    top = [glyph(10 * i, 100, 140, text="a") for i in range(3)]
    bottom = [glyph(10 * i, 150, 190, text="b") for i in range(3)]
    assert row_texts(top + bottom) == ["aaa", "bbb"]


def test_rows_with_ten_percent_overlap_stay_apart():
    top = [glyph(10 * i, 100, 140, text="a") for i in range(3)]
    bottom = [glyph(10 * i, 136, 176, text="b") for i in range(3)]
    assert row_texts(top + bottom) == ["aaa", "bbb"]


def test_two_row_tall_glyph_merges_both_rows():
    """Known risk: a glyph spanning two rows links them into one row (then split by gaps)."""
    top = [glyph(10 * i, 100, 140, text="a") for i in range(3)]
    bottom = [glyph(10 * i, 150, 190, text="b") for i in range(3)]
    tall = [glyph(40, 100, 190, text="|")]
    assert len(segment_lines(top + bottom + tall, "ltr", factor=100.0)) == 1


def test_rows_run_top_to_bottom_regardless_of_input_order():
    chars = [glyph(0, 150, 190, text="b"), glyph(0, 100, 140, text="a")]
    assert [line_text(s) for s in segment_lines(chars, "ltr")] == ["a", "b"]


def test_five_thousand_glyphs_in_fifty_rows_are_fast():
    chars = [glyph(10 * c, 40 * r, 40 * r + 30) for r in range(50) for c in range(100)]
    start = time.perf_counter()
    lines = segment_lines(chars, "ltr")
    assert time.perf_counter() - start < 2.0  # pairwise checks would take over 10 s
    assert len(lines) == 50


def test_real_edge_harakat_row_stays_one_line():
    """Boxes pdfium gave an Edge-printed Arabic line with harakat: lam and fatha get 5.7 px
    boxes above the row and must not split off as a line of their own."""
    data = json.loads((Path(__file__).parent / "data" / "edge_harakat_row.json").read_text("utf-8"))
    chars = [
        TextChar(
            c["text"],
            tuple(c["box"]) if c["box"] else None,
            False,
            tuple(c["tight_box"]) if c["tight_box"] else None,
        )
        for c in data["chars"]
    ]
    small = [c for c in chars if c.box and c.text.strip() and c.box[3] - c.box[1] < 10]
    assert len(small) == 2
    segments = segment_lines(chars, "rtl")
    assert len(segments) == 1
    assert line_text(segments[0]) == data["text"]
