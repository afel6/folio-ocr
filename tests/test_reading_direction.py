"""Page and line reading direction (display direction only).

Page.direction = D, the page majority of strong letters (tie -> rtl). A line whose strong
letters all run one way takes that direction, unless its order check holds only under the
other base: then it takes that base. A mixed-direction or zero-strong line takes D.
"""

import json
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from folio_ocr import bidi
from folio_ocr.document import TextChar
from folio_ocr.flags import VISUAL_ORDER
from folio_ocr.text_layer import line_text, read_text_layer
from folio_ocr.text_layer_check import check_text_layer, line_direction, order_holds

FIXTURES = Path(__file__).parent / "fixtures"
ARABIC = "شكرا جزيلا لكم على التعاون"
ENGLISH = "Thank you very much for the help"
MIXED = "إلى: Tomas Okafor"  # 3 Arabic letters, 11 Latin letters


def laid_out(text, visual, y=0.0):
    """Characters of `text`; character visual[k] is drawn k-th from the left (10 px apart)."""
    pos = {i: k for k, i in enumerate(visual)}
    return tuple(
        TextChar(c, (10.0 * pos[i], y, 10.0 * pos[i] + 8.0, y + 10.0)) for i, c in enumerate(text)
    )


def displayed(text, base, y=0.0):
    """`text` drawn in the bidi display order for `base` ('ltr' or 'rtl')."""
    return laid_out(text, bidi.visual_order(bidi.levels(text, 1 if base == "rtl" else 0)), y)


def line_on_page(line, page):
    """Line check of `line` on a page whose majority script gives direction `page`."""
    majority = displayed(ARABIC if page == "rtl" else ENGLISH, page, 0.0)
    report = check_text_layer([majority, line], has_ink=True)
    assert report.direction == page
    return report.lines[1]


# Page direction


def test_page_direction_is_the_letter_majority_not_the_first_strong_letter():
    first_latin = [displayed("ab", "ltr"), displayed(ARABIC, "rtl", 50.0)]
    assert check_text_layer(first_latin, has_ink=True).direction == "rtl"
    first_arabic = [displayed("حضر", "rtl"), displayed(ENGLISH, "ltr", 50.0)]
    assert check_text_layer(first_arabic, has_ink=True).direction == "ltr"


def test_page_direction_tie_is_rtl():
    lines = [displayed("ab", "ltr"), displayed("حض", "rtl", 50.0)]
    assert check_text_layer(lines, has_ink=True).direction == "rtl"


def test_private_use_bullets_do_not_vote_for_the_page_direction():
    lines = [displayed("", "ltr"), displayed("حض", "rtl", 50.0)]
    assert check_text_layer(lines, has_ink=True).direction == "rtl"


def test_western_digits_do_not_vote_for_the_page_direction():
    lines = [displayed("1234567", "ltr"), displayed("حض", "rtl", 50.0)]
    assert check_text_layer(lines, has_ink=True).direction == "rtl"


def test_arabic_indic_digits_do_not_vote_for_the_page_direction():
    lines = [displayed("ab", "ltr"), displayed("٣٤٥٦٧", "rtl", 50.0)]
    assert check_text_layer(lines, has_ink=True).direction == "ltr"


# One-direction lines take their letters' direction


@pytest.mark.parametrize("page", ["rtl", "ltr"])
def test_english_line_is_ltr_on_any_page(page):
    assert line_on_page(displayed("Total amount due.", "ltr", 50.0), page).direction == "ltr"


@pytest.mark.parametrize("page", ["rtl", "ltr"])
def test_arabic_line_is_rtl_on_any_page(page):
    assert line_on_page(displayed("حضر الاجتماع.", "rtl", 50.0), page).direction == "rtl"


def test_arabic_indic_digits_do_not_make_an_english_line_rtl():
    assert line_on_page(displayed("Total ٣٥٠", "ltr", 50.0), "rtl").direction == "ltr"


def test_private_use_bullet_does_not_make_an_arabic_line_ltr():
    assert line_on_page(displayed(" مرحبا", "rtl", 50.0), "ltr").direction == "rtl"


def test_one_direction_line_holding_under_both_bases_keeps_its_letters_direction():
    latin = laid_out("Total amount", range(12))
    arabic = laid_out("حضر الاجتماع", reversed(range(12)))
    for chars in (latin, arabic):
        assert order_holds(chars, "ltr") and order_holds(chars, "rtl")
    assert line_direction(latin, "rtl") == ("ltr", True)
    assert line_direction(arabic, "ltr") == ("rtl", True)


def test_one_direction_line_failing_under_both_bases_keeps_its_letters_direction():
    latin, arabic = laid_out("abc", [2, 1, 0]), laid_out("حضر", [0, 1, 2])
    for chars in (latin, arabic):
        assert not order_holds(chars, "ltr") and not order_holds(chars, "rtl")
    assert line_direction(latin, "rtl") == ("ltr", False)
    assert line_direction(arabic, "ltr") == ("rtl", False)


# A one-direction line whose order check holds only under the other base takes that base


@pytest.mark.parametrize("page", ["ltr", "rtl"])
def test_one_direction_line_holding_only_under_the_other_base_takes_that_base(page):
    latin = laid_out("abc !", [4, 3, 0, 1, 2], 50.0)  # drawn "! abc", as an RTL base shows it
    arabic = laid_out("حضر !", [2, 1, 0, 3, 4], 50.0)  # "!" on the right, as an LTR base shows it
    assert not order_holds(latin, "ltr") and order_holds(latin, "rtl")
    assert order_holds(arabic, "ltr") and not order_holds(arabic, "rtl")
    assert line_direction(latin, page) == ("rtl", True)
    assert line_direction(arabic, page) == ("ltr", True)
    for chars, want in ((latin, "rtl"), (arabic, "ltr")):
        line = line_on_page(chars, page)
        assert line.direction == want
        assert VISUAL_ORDER not in line.flags


# Mixed-direction and zero-strong lines take D


@pytest.mark.parametrize("page", ["rtl", "ltr"])
def test_mixed_line_takes_the_page_direction_not_its_letter_majority(page):
    assert line_on_page(displayed(MIXED, page, 50.0), page).direction == page


@pytest.mark.parametrize("page", ["rtl", "ltr"])
@pytest.mark.parametrize("text", ["50 123", "٣٥٠٫٧٥", "+966 50 123 4567", ""])
def test_line_without_strong_letters_takes_the_page_direction(text, page):
    assert line_on_page(displayed(text, page, 50.0), page).direction == page


@pytest.mark.parametrize("page", ["rtl", "ltr"])
@pytest.mark.parametrize("text", [MIXED, "+966 50 123 4567"])
def test_line_drawn_for_the_other_base_still_takes_the_page_direction(text, page):
    other = "ltr" if page == "rtl" else "rtl"
    assert line_on_page(displayed(text, other, 50.0), page).direction == page


# Fixtures


def _letter_directions(chars):
    found = set()
    for ch in chars:
        if unicodedata.category(ch.text).startswith("L"):
            found.add({"L": "ltr", "R": "rtl", "AL": "rtl"}.get(bidi.bidi_class(ch.text)))
    return found - {None}


def _expected_direction(chars, page):
    letters = _letter_directions(chars)
    if len(letters) != 1:
        return page
    own = letters.pop()
    other = "ltr" if own == "rtl" else "rtl"
    return other if not order_holds(chars, own) and order_holds(chars, other) else own


PDFS = [
    e["file"]
    for e in json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
    if e["file"].endswith(".pdf")
]


@pytest.mark.parametrize("name", PDFS)
def test_fixture_lines_follow_the_reading_direction_rule(name):
    doc = pdfium.PdfDocument(str(FIXTURES / name))
    try:
        segs = read_text_layer(doc[0])
    finally:
        doc.close()
    report = check_text_layer(segs, has_ink=True)
    wrong = []
    for seg, line in zip(segs, report.lines, strict=True):
        want = _expected_direction(seg, report.direction)
        if line.direction != want:
            wrong.append((line_text(seg), line.direction, want))
    assert wrong == []
