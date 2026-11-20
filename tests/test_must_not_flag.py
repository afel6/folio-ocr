"""The order check must not flag correct lines laid out by a real browser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from folio_ocr import bidi, brackets
from folio_ocr.brackets import faces_away, swap_pairs
from folio_ocr.document import TextChar
from folio_ocr.text_layer_check import (
    check_text_layer,
    line_direction,
    order_holds,
    page_direction,
)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import must_not_flag as mnf

LAYOUT = json.loads((HERE / "data" / "edge_must_not_flag_layout.json").read_text("utf-8"))["lines"]
BY_TEXT = {(line["text"], line["dir"]): line for line in LAYOUT}
GATED = [(t, "rtl") for t in mnf.RTL] + [(t, "ltr") for t in mnf.LTR]


def _chars(line: dict, mirror: bool = False) -> list[TextChar]:
    def box(b):
        l, t, r, bt = b
        return (-r, t, -l, bt) if mirror else (l, t, r, bt)

    return [TextChar(c, box(b)) for c, b in zip(line["text"], line["boxes"], strict=True)]


def _boxed(text: str, spans: list[tuple[float, float]]) -> list[TextChar]:
    return [TextChar(c, (a, 0, b, 10)) for c, (a, b) in zip(text, spans, strict=True)]


def test_layout_holds_each_gated_line_once_and_every_measured_line():
    keys = [(line["text"], line["dir"]) for line in LAYOUT]
    for key in GATED + [(t, d) for t, d in mnf.MEASURE_ONLY]:
        assert keys.count(key) == 1, ascii(key)
    assert all(len(line["boxes"]) == len(line["text"]) for line in LAYOUT)


@pytest.mark.parametrize(("text", "direction"), GATED, ids=ascii)
def test_correct_line_is_not_flagged(text, direction):
    line = BY_TEXT[(text, direction)]
    chars = _chars(line)
    assert line_direction(chars, direction) == (direction, True)
    assert order_holds(chars, direction)
    if page_direction(chars) == direction:
        result = check_text_layer([chars], has_ink=True)
        assert "visual_order" not in result.lines[0].flags
    assert faces_away(text) is False
    assert swap_pairs(text) is False
    assert not brackets.mirrored(text)


def _compared_xs(chars):
    return {
        c.box[0]
        for c in chars
        if c.text.strip() and bidi.bidi_class(c.text) not in ("NSM", "BN") and c.box[2] > c.box[0]
    }


@pytest.mark.parametrize("text", mnf.RTL, ids=ascii)
def test_mirrored_rtl_layout_fails_the_order_check(text):
    chars = _chars(BY_TEXT[(text, "rtl")])
    if len(_compared_xs(chars)) < 2:
        pytest.skip("fewer than two compared characters")
    assert not order_holds(_chars(BY_TEXT[(text, "rtl")], mirror=True), "rtl")


def test_measure_only_lines_parse():
    for text, direction in mnf.MEASURE_ONLY:
        assert direction in ("ltr", "rtl")
        assert len(BY_TEXT[(text, direction)]["boxes"]) == len(text)


@pytest.mark.parametrize(
    "text",
    [
        ")مثال(",
        "المادة)٣(من العقد",
        "(a) )b(",
        "Friday) and send (copy",
        "1234) x (y",
    ],
    ids=ascii,
)
def test_swapped_brackets_are_detected(text):
    assert swap_pairs(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "1) Pay the invoice (by",
        "a) first item",
        "١) البند",
        "(مثال)",
        "x)y",
        "see (page",
        "123) x (y",
        "[a]",
        "{b}",
    ],
    ids=ascii,
)
def test_legitimate_brackets_are_not_swapped(text):
    assert swap_pairs(text) is False


@pytest.mark.parametrize("text", [")مثال(", "١( البند الأول"], ids=ascii)
def test_brackets_facing_away_are_detected(text):
    assert faces_away(text) is True


@pytest.mark.parametrize("text", ["1) item", "a)", "( x )"])
def test_brackets_facing_their_word_are_accepted(text):
    assert faces_away(text) is False


def test_reversed_dots_on_an_rtl_line_pass():
    text = "انتظر..."
    spans = [(90 - 10 * i, 100 - 10 * i) for i in range(5)]
    spans += [(20, 30), (30, 40), (40, 50)]
    assert order_holds(_boxed(text, spans), "rtl")


def test_swapped_zeros_on_an_ltr_line_pass():
    assert order_holds(_boxed("500", [(0, 10), (20, 30), (10, 20)]), "ltr")


def test_swapped_distinct_digits_fail():
    assert not order_holds(_boxed("50", [(10, 20), (0, 10)]), "ltr")
