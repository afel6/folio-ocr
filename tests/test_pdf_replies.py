"""Every check on what the file reader process sends back."""

import base64
import math

import pytest

from folio_ocr import ingest, pdf_replies
from folio_ocr.pdf_frames import Ended

BOX = [0.0, 0.0, 1.0, 1.0]
GOOD_CHAR = [65, BOX, False, None, False]
PAGE_LINES = (
    "page {} is too large to read (over 50 megapixels at 300 dpi)",
    "page {} holds an image too large to read (over 50 megapixels)",
    "page {} of the PDF could not be read; it may be damaged",
)


@pytest.mark.parametrize("line", sorted(ingest.REFUSALS))
def test_every_known_refusal_passes_unchanged(line):
    assert pdf_replies.refusal(line) == line


@pytest.mark.parametrize("template", PAGE_LINES)
@pytest.mark.parametrize("page", [1, 20])
def test_page_refusals_pass_at_the_edges(template, page):
    line = template.format(page)
    assert pdf_replies.refusal(line) == line


@pytest.mark.parametrize("template", PAGE_LINES)
@pytest.mark.parametrize("page", [0, 21, 100, "01"])
def test_page_refusals_outside_the_page_range_are_refused(template, page):
    with pytest.raises(Ended):
        pdf_replies.refusal(template.format(page))


@pytest.mark.parametrize("line", ["anything else", "", None, 5, ["x"], "page 1 is damaged"])
def test_unknown_refusal_is_refused(line):
    with pytest.raises(Ended):
        pdf_replies.refusal(line)


def test_page_sizes_are_computed_by_folio():
    assert pdf_replies.page_sizes({"count": 2, "sizes": [[72.0, 36], [612, 792.5]]}) == (
        ingest.render_size(72, 36),
        ingest.render_size(612, 792.5),
    )


@pytest.mark.parametrize(
    "reply",
    [
        {"count": 1, "sizes": [[1.0, 1.0]] * 2},
        {"count": 21, "sizes": [[1.0, 1.0]] * 21},
        {"count": 1, "sizes": "x"},
        {"sizes": [[1.0, 1.0]]},
        {"count": 1, "sizes": [[1.0]]},
        {"count": 1, "sizes": [[1.0, "2"]]},
        {"count": 1, "sizes": [[True, 1.0]]},
        {"count": 1, "sizes": [[-1.0, 1.0]]},
        {"count": 1, "sizes": [[math.nan, 1.0]]},
        {"count": 1, "sizes": [[float("nan"), 1]]},
        {"count": 1, "sizes": [[math.inf, 1.0]]},
        {"count": 1, "sizes": [[1700.0, 1700.0]]},
        {"count": 1, "sizes": [[1e308, 1.0]]},
    ],
)
def test_bad_page_sizes_are_refused(reply):
    with pytest.raises(Ended):
        pdf_replies.page_sizes(reply)


def test_twenty_pages_are_accepted():
    assert len(pdf_replies.page_sizes({"count": 20, "sizes": [[1.0, 1.0]] * 20})) == 20


@pytest.mark.parametrize("count", [1, 20])
def test_frame_count_accepts_the_range(count):
    assert pdf_replies.frame_count({"count": count}) == count


@pytest.mark.parametrize("count", [0, 21, True, 1.0, "1", None, -1])
def test_frame_count_refuses_the_rest(count):
    with pytest.raises(Ended):
        pdf_replies.frame_count({"count": count})


def test_image_size_accepts_the_cap_and_zero():
    assert pdf_replies.image_size({"mode": "L", "size": [10000, 5000]}, "L") == (10000, 5000)
    assert pdf_replies.image_size({"mode": "L", "size": [0, 5]}, "L") == (0, 5)


@pytest.mark.parametrize(
    "reply",
    [
        {"mode": "RGB", "size": [1, 1]},
        {"mode": "L", "size": [1.0, 1]},
        {"mode": "L", "size": [True, 1]},
        {"mode": "L", "size": [1]},
        {"mode": "L", "size": None},
        {"mode": "L", "size": [-1, 1]},
        {"mode": "L", "size": [10001, 5000]},
        {"mode": "L", "size": [2**31, 0]},
        {"size": [1, 1]},
    ],
)
def test_bad_image_sizes_are_refused(reply):
    with pytest.raises(Ended):
        pdf_replies.image_size(reply, "L")


def test_byte_count_per_band():
    assert pdf_replies.byte_count((4, 5), "L") == 20
    assert pdf_replies.byte_count((4, 5), "RGB") == 60


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_frame_mode_accepts_l_and_rgb(mode):
    assert pdf_replies.frame_mode({"mode": mode}) == mode


@pytest.mark.parametrize("mode", ["RGBA", "P", None, 1, []])
def test_frame_mode_refuses_the_rest(mode):
    with pytest.raises(Ended):
        pdf_replies.frame_mode({"mode": mode})


def test_frame_info_decodes_the_profile_and_transparency():
    icc = base64.b64encode(b"profile").decode()
    assert pdf_replies.frame_info({"icc_profile": icc, "transparency": 7}, "L") == {
        "icc_profile": b"profile",
        "transparency": 7,
    }
    assert pdf_replies.frame_info({"transparency": [1, 2, 3]}, "RGB") == {"transparency": (1, 2, 3)}
    assert pdf_replies.frame_info({}, "L") == {}


@pytest.mark.parametrize(
    ("reply", "mode"),
    [
        ({"icc_profile": "!!!"}, "L"),
        ({"icc_profile": 5}, "L"),
        ({"transparency": [1, 2]}, "RGB"),
        ({"transparency": 1}, "RGB"),
        ({"transparency": [1]}, "RGB"),
        ({"transparency": [1, 2]}, "L"),
        ({"transparency": "1"}, "L"),
        ({"transparency": True}, "L"),
        ({"transparency": [-1, 2, 3]}, "RGB"),
        ({"transparency": [1, 2, 65536]}, "RGB"),
    ],
)
def test_bad_frame_info_is_refused(reply, mode):
    with pytest.raises(Ended):
        pdf_replies.frame_info(reply, mode)


def test_lines_keep_lone_surrogates_and_control_characters():
    (line,) = pdf_replies.text_lines(
        {
            "chars": [[0xD800, BOX, False, None, False], [0x0, None, True, BOX, True]],
            "lines": [[0, 1]],
        }
    )
    assert [c.text for c in line] == [chr(0xD800), "\x00"]
    assert line[0].box == tuple(BOX) and line[1].box is None
    assert line[1].unmapped and line[1].invisible and line[1].tight_box == tuple(BOX)


def test_lines_group_characters_by_position():
    chars = [[65 + i, BOX, False, None, False] for i in range(3)]
    lines = pdf_replies.text_lines({"chars": chars, "lines": [[2, 0], [1]]})
    assert [[c.text for c in line] for line in lines] == [["C", "A"], ["B"]]
    assert isinstance(lines, tuple) and all(isinstance(line, tuple) for line in lines)


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_nonfinite_boxes_are_refused(coordinate):
    box = [0.0, 0.0, coordinate, 1.0]
    with pytest.raises(Ended):
        pdf_replies.text_lines({"chars": [[65, box, False, None, False]], "lines": [[0]]})


@pytest.mark.parametrize(
    "char",
    [
        [0x110000, BOX, False, None, False],
        [-1, BOX, False, None, False],
        [True, BOX, False, None, False],
        [65.0, BOX, False, None, False],
        [65, [0, 0, 1, 1], False, None, False],
        [65, [0.0, 0.0, 1.0], False, None, False],
        [65, BOX, 0, None, False],
        [65, BOX, False, None, 1],
        [65, BOX, False, None],
        "A",
    ],
)
def test_bad_characters_are_refused(char):
    with pytest.raises(Ended):
        pdf_replies.text_lines({"chars": [char], "lines": []})


@pytest.mark.parametrize(
    "lines",
    [[[]], [[1]], [[-1]], [[True]], [[0.0]], [["0"]], [0], ["0"], [None], "x", None],
)
def test_bad_lines_are_refused(lines):
    with pytest.raises(Ended):
        pdf_replies.text_lines({"chars": [GOOD_CHAR], "lines": lines})


def test_a_position_needs_a_character():
    with pytest.raises(Ended):
        pdf_replies.text_lines({"chars": [], "lines": [[0]]})


def test_a_text_line_needs_a_box():
    with pytest.raises(Ended):
        pdf_replies.text_lines({"chars": [[65, None, False, None, False]], "lines": [[0]]})


def test_text_layer_must_be_lists():
    for reply in ({"chars": "x", "lines": []}, {"chars": []}, {"lines": []}, {}):
        with pytest.raises(Ended):
            pdf_replies.text_lines(reply)


def test_image_boxes():
    assert pdf_replies.image_boxes({"images": None}) is None
    assert pdf_replies.image_boxes({}) is None
    assert pdf_replies.image_boxes({"images": [BOX]}) == (tuple(BOX),)
    for bad in ("x", [[1, 2, 3, 4]], [[1.0]]):
        with pytest.raises(Ended):
            pdf_replies.image_boxes({"images": bad})
