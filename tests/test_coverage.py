"""Tests for finding ink that no text box covers."""

import math

import pytest
from PIL import Image, ImageDraw

from folio_ocr import coverage, ocr_setup
from folio_ocr.coverage import has_ink, uncovered_ink


def page(size=(200, 200), value=255):
    return Image.new("L", size, value)


def blot(img, box, value=0):
    ImageDraw.Draw(img).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=value)
    return img


def test_blank_has_no_ink():
    img = page()
    assert not has_ink(img)
    assert uncovered_ink(img, []) == ()


def test_single_pixel_is_ink():
    img = page()
    img.putpixel((5, 5), 0)
    assert has_ink(img)
    assert uncovered_ink(img, []) == ((5.0, 5.0, 6.0, 6.0),)


def test_ink_threshold():
    assert has_ink(page(value=127))
    assert not has_ink(page(value=128))
    assert uncovered_ink(page((10, 10), 127), []) == ((0.0, 0.0, 10.0, 10.0),)
    assert uncovered_ink(page((10, 10), 128), []) == ()


def test_square_exact_bounds():
    img = blot(page(), (40, 50, 50, 60))
    assert uncovered_ink(img, []) == ((40.0, 50.0, 50.0, 60.0),)
    assert uncovered_ink(img, [(0.0, 0.0, 5.0, 5.0)]) == ((40.0, 50.0, 50.0, 60.0),)


def test_covered_square_ignored():
    img = blot(page(), (40, 50, 50, 60))
    assert uncovered_ink(img, [(40.0, 50.0, 50.0, 60.0)]) == ()


def test_margin():
    img = blot(page(), (40, 50, 50, 60))
    box = [(41.0, 50.0, 50.0, 60.0)]  # ink column x=40 is 1 px outside
    assert uncovered_ink(img, box) == ()
    assert uncovered_ink(img, box, margin=0) == ((40.0, 50.0, 41.0, 60.0),)
    far = [(43.0, 50.0, 50.0, 60.0)]  # columns 40..42 lie 1..3 px outside
    assert uncovered_ink(img, far) == ((40.0, 50.0, 41.0, 60.0),)


def test_two_blobs_sorted_by_y_then_x():
    img = blot(blot(page(), (150, 10, 155, 15)), (10, 120, 15, 125))
    assert uncovered_ink(img, []) == ((150.0, 10.0, 155.0, 15.0), (10.0, 120.0, 15.0, 125.0))
    img = blot(blot(page(), (100, 10, 105, 15)), (10, 10, 15, 15))
    assert uncovered_ink(img, []) == ((10.0, 10.0, 15.0, 15.0), (100.0, 10.0, 105.0, 15.0))


def test_neighbouring_squares_merge():
    img = blot(blot(page(), (28, 5, 31, 8)), (33, 5, 36, 8))
    assert uncovered_ink(img, []) == ((28.0, 5.0, 36.0, 8.0),)


def test_diagonal_squares_merge():
    img = blot(blot(page(), (28, 28, 31, 31)), (33, 33, 36, 36))
    assert uncovered_ink(img, []) == ((28.0, 28.0, 36.0, 36.0),)


def test_two_squares_apart_split():
    img = blot(blot(page(), (5, 5, 8, 8)), (70, 5, 73, 8))
    assert uncovered_ink(img, []) == ((5.0, 5.0, 8.0, 8.0), (70.0, 5.0, 73.0, 8.0))


def test_modes():
    rgb = blot(Image.new("RGB", (100, 100), "white"), (40, 50, 50, 60), (0, 0, 0))
    assert uncovered_ink(rgb, []) == ((40.0, 50.0, 50.0, 60.0),)
    one = Image.new("1", (100, 100), 1)
    ImageDraw.Draw(one).rectangle((40, 50, 49, 59), fill=0)
    assert uncovered_ink(one, []) == ((40.0, 50.0, 50.0, 60.0),)
    assert has_ink(one)


def test_light_grey_is_not_ink():
    img = blot(page(), (40, 50, 50, 60), 200)
    assert not has_ink(img)
    assert uncovered_ink(img, []) == ()


def test_input_unchanged():
    img = blot(page(), (40, 50, 50, 60))
    before = img.tobytes()
    assert uncovered_ink(img, [(40.0, 50.0, 50.0, 60.0)]) == ()
    assert img.tobytes() == before


def test_cover_past_edge():
    img = blot(page((100, 100)), (90, 90, 100, 100))
    assert uncovered_ink(img, [(80.0, 80.0, 500.0, 500.0)]) == ()
    assert uncovered_ink(img, [(-50.0, -50.0, -10.0, -10.0)]) == ((90.0, 90.0, 100.0, 100.0),)


def test_empty_cover_box_is_ignored():
    img = blot(page(), (40, 50, 50, 60))
    assert uncovered_ink(img, [(45.0, 55.0, 45.0, 55.0)], margin=0) == ((40.0, 50.0, 50.0, 60.0),)


def test_fractional_cover_floors_and_ceils_outward():
    img = blot(page(), (40, 50, 50, 60))
    assert uncovered_ink(img, [(40.4, 50.4, 49.6, 59.6)], margin=0) == ()


@pytest.mark.parametrize("margin", [-1.0, math.inf, math.nan, True, "2"])
def test_bad_margin(margin):
    with pytest.raises(ValueError):
        uncovered_ink(page(), [], margin=margin)


@pytest.mark.parametrize("tile", [0, -3, 2.5, True, "8"])
def test_bad_tile(tile):
    with pytest.raises(ValueError):
        uncovered_ink(page(), [], tile=tile)


def covered_page():
    img = page((2480, 3508))
    boxes = []
    for i in range(60):
        x, y = 100 + (i % 3) * 700, 100 + (i // 3) * 160
        blot(img, (x, y, x + 600, y + 140))
        boxes.append((float(x), float(y), float(x + 600), float(y + 140)))
    return img, boxes


def test_a_full_covered_page_has_no_uncovered_ink():
    img, boxes = covered_page()
    assert uncovered_ink(img, boxes) == ()


def test_specks_far_apart_stay_separate_regions():
    img, boxes = covered_page()
    for i in range(200):
        img.putpixel((2150 + (i % 5) * 64, 10 + (i // 5) * 64), 0)
    assert len(uncovered_ink(img, boxes)) == 200


def photographed(size=(300, 400), border=20, value=60):
    """A light page lying on a dark table: a dark band all round the image."""
    img = page(size)
    w, h = size
    for box in ((0, 0, w, border), (0, h - border, w, h), (0, 0, border, h), (w - border, 0, w, h)):
        blot(img, box, value)
    return img


def test_the_dark_surround_of_a_photo_is_not_ink():
    assert uncovered_ink(photographed(), []) == ()


def test_text_inside_a_photographed_page_is_still_found():
    img = blot(photographed(), (120, 150, 180, 170))
    assert uncovered_ink(img, []) == ((120.0, 150.0, 180.0, 170.0),)
    assert uncovered_ink(img, [(120.0, 150.0, 180.0, 170.0)]) == ()


def test_text_touching_the_surround_is_left_out_with_it():
    # Ink joined to the surround belongs to it: the table edge, not a line on the page.
    img = blot(photographed(), (20, 150, 60, 170))
    assert uncovered_ink(img, []) == ()


def test_a_dark_band_along_one_edge_stays_ink():
    w, h = 300, 400
    img = blot(page((w, h)), (0, 0, w, 20))  # a dark bar along the top: full width, 5 % high
    assert uncovered_ink(img, []) == ((0.0, 0.0, 300.0, 20.0),)
    left = blot(page((w, h)), (0, 0, 20, h))  # a dark bar down the left: full height, 7 % wide
    assert uncovered_ink(left, []) == ((0.0, 0.0, 20.0, 400.0),)
    top_and_half_left = blot(blot(page((w, h)), (0, 0, w, 20)), (0, 0, 20, h // 2))
    assert uncovered_ink(top_and_half_left, []) == ((0.0, 0.0, 300.0, 200.0),)  # half as high


def test_two_bars_along_two_edges_are_a_surround():
    # The paper pushed into a corner of the photo: the table shows along the top and the left.
    img = blot(blot(page((300, 400)), (0, 0, 300, 20)), (0, 0, 20, 400))
    assert uncovered_ink(img, []) == ()
    blot(img, (120, 150, 180, 170))
    assert uncovered_ink(img, []) == ((120.0, 150.0, 180.0, 170.0),)


def test_a_turned_photo_with_white_corners_still_loses_its_surround():
    # A straightened photo: the dark ring is tilted inside a canvas whose corners are white.
    img = photographed((300, 400)).rotate(5, expand=True, fillcolor=255)
    blot(img, (140, 200, 200, 220))
    assert uncovered_ink(img, []) == ((140.0, 200.0, 200.0, 220.0),)


def ring_at(size, ring_size, offset=0, bar=5):
    """A dark ring `bar` px thick of `ring_size`, its top-left corner at (offset, offset)."""
    img = page(size)
    rw, rh = ring_size
    for x0, y0, x1, y1 in (
        (0, 0, rw, bar),
        (0, rh - bar, rw, rh),
        (0, 0, bar, rh),
        (rw - bar, 0, rw, rh),
    ):
        blot(img, (x0 + offset, y0 + offset, x1 + offset, y1 + offset))
    return img


def test_a_ring_spanning_exactly_nine_tenths_is_a_surround():
    assert uncovered_ink(ring_at((300, 400), (270, 360)), []) == ()


@pytest.mark.parametrize("ring_size", [(269, 360), (270, 359), (269, 359)])
def test_a_ring_short_of_nine_tenths_stays_ink(ring_size):
    found = uncovered_ink(ring_at((300, 400), ring_size), [])
    assert found == ((0.0, 0.0, float(ring_size[0]), float(ring_size[1])),)


def test_a_ring_joined_only_at_diagonal_pixels_is_one_surround():
    # Two halves, each half the page wide; at the top and the bottom they meet corner to corner.
    img = page((300, 400))
    for box in ((0, 0, 150, 5), (0, 0, 5, 400), (0, 395, 150, 400)):  # left half
        blot(img, box)
    for box in ((150, 5, 300, 10), (295, 5, 300, 395), (150, 390, 300, 395)):  # right half
        blot(img, box)
    assert uncovered_ink(img, []) == ()


def test_a_patch_covering_half_the_page_stays_ink():
    # Top bar 300 x 100 plus left bar 100 x 300: 60000 of the 120000 pixels.
    img = blot(blot(page((300, 400)), (0, 0, 300, 100)), (0, 100, 100, 400))
    assert uncovered_ink(img, []) == ((0.0, 0.0, 300.0, 400.0),)
    img.putpixel((99, 399), 255)  # one pixel less: under half
    assert uncovered_ink(img, []) == ()


def test_a_dark_page_with_one_light_speck_keeps_its_ink():
    # Dark paper (grey 100) with a black text block and one white speck (a punched hole).
    img = blot(page((300, 400), 100), (100, 150, 200, 170))
    img.putpixel((5, 5), 255)
    assert uncovered_ink(img, []) == ((0.0, 0.0, 300.0, 400.0),)
    assert uncovered_ink(img, [(100.0, 150.0, 200.0, 170.0)]) == ((0.0, 0.0, 300.0, 400.0),)


def test_a_white_on_black_page_keeps_its_ink():
    img = blot(page((300, 400), 0), (100, 150, 200, 170), 255)
    assert uncovered_ink(img, []) == ((0.0, 0.0, 300.0, 400.0),)


def frame(img, inset, bar):
    """Draw a dark frame `bar` px thick, `inset` px inside the image edge; return its bars."""
    w, h = img.size
    bars = [
        (inset, inset, w - inset, inset + bar),
        (inset, h - inset - bar, w - inset, h - inset),
        (inset, inset, inset + bar, h - inset),
        (w - inset - bar, inset, w - inset, h - inset),
    ]
    for box in bars:
        blot(img, box)
    return [tuple(float(v) for v in box) for box in bars]


def test_a_stamp_across_a_printed_frame_stays_ink():
    img = page((300, 400))
    bars = frame(img, 10, 3)
    blot(img, (20, 200, 120, 220))  # a text line inside the frame
    blot(img, (250, 300, 295, 320))  # a stamp across the right side of the frame
    assert uncovered_ink(img, []) == ((10.0, 10.0, 295.0, 390.0),)
    assert uncovered_ink(img, [*bars, (20.0, 200.0, 120.0, 220.0)]) == (
        (250.0, 300.0, 295.0, 320.0),
    )


def test_handwriting_on_a_form_grid_stays_ink():
    img = page((300, 400))
    lines = [*frame(img, 10, 2), (10.0, 200.0, 290.0, 202.0), (150.0, 10.0, 152.0, 390.0)]
    for box in lines[4:]:
        blot(img, tuple(int(v) for v in box))
    blot(img, (30, 190, 120, 200))  # handwriting sitting on the middle grid line
    blot(img, (30, 100, 120, 110))  # an entry clear of the lines
    assert uncovered_ink(img, []) == ((10.0, 10.0, 290.0, 390.0),)
    assert uncovered_ink(img, lines) == ((30.0, 100.0, 120.0, 110.0), (30.0, 190.0, 120.0, 198.0))


@pytest.mark.parametrize(("offset", "found"), [(2, ()), (3, ((3.0, 3.0, 297.0, 397.0),))])
def test_a_ring_must_reach_within_three_pixels_of_the_edge(offset, found):
    img = ring_at((300, 400), (300 - 2 * offset, 400 - 2 * offset), offset)
    assert uncovered_ink(img, []) == found


def cross(side, gap):
    """A dark cross of two 5 px bars spanning 270 x 360 on a 300 x 400 page, clear of every
    edge except that the arm towards `side` ends `gap` px from it; return it and its bounds."""
    x0, x1, y0, y1 = 15, 285, 20, 380
    x0, x1, y0, y1 = {
        "left": (gap, x1, y0, y1),
        "right": (x0, 300 - gap, y0, y1),
        "top": (x0, x1, gap, y1),
        "bottom": (x0, x1, y0, 400 - gap),
    }[side]
    img = blot(blot(page((300, 400)), (x0, 200, x1, 205)), (150, y0, 155, y1))
    return img, (float(x0), float(y0), float(x1), float(y1))


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_each_edge_counts_for_a_surround(side):
    img, bounds = cross(side, 2)
    assert uncovered_ink(img, []) == ()
    img, bounds = cross(side, 3)
    assert uncovered_ink(img, []) == (bounds,)


def test_the_setup_key_names_every_surround_rule(monkeypatch):
    parts = (
        f"surround-{coverage.SURROUND_CONNECTIVITY}-way",
        f"span>={coverage.SURROUND_SPAN}",
        f"share<{coverage.SURROUND_SHARE}",
        f"edge<{coverage.SURROUND_EDGE_PX}px",
    )
    assert all(part in coverage.METHOD for part in parts)
    before = ocr_setup.onnxtr_setup().key
    monkeypatch.setattr(coverage, "METHOD", coverage.METHOD + ";other")
    assert ocr_setup.onnxtr_setup().key != before
