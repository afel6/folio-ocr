"""Mirrored brackets and guillemets in one text-layer line."""

import pytest

from folio_ocr import brackets
from folio_ocr.document import TextChar
from folio_ocr.text_layer_check import order_holds

OPEN_Q, CLOSE_Q = chr(0xAB), chr(0xBB)

CORRECT = [
    "1) Pay the invoice (by",
    "١) يُدفع المبلغ (خلال",
    "Options a) and b) apply",
    "أ) البند ب) الشرط",
    "x) and y)",
    "(a) (b)",
    "f(x) = y",
    "see (page 3) and (page 4",
    "المادة(٣)من العقد",
    "السيد(Tomas)حاضر",
    "( x )",
    "قال " + OPEN_Q + "نعم" + CLOSE_Q + " للعرض",
    "نهاية الاقتباس" + CLOSE_Q,
]


@pytest.mark.parametrize("text", CORRECT)
def test_correct_lines_show_no_mirroring(text):
    assert not brackets.mirrored(text)


def test_mirrored_tight_pair_is_found_by_swapping_both():
    assert brackets.swap_pairs("المادة)٣(من العقد")


def test_half_mirrored_pair_is_found_by_swapping_one():
    # swapping both gives `(Tomas(`, which pairs nothing
    assert brackets.swap_pairs("السيد)Tomas)حاضر")


def test_list_markers_never_count_anywhere_in_the_line():
    assert not brackets.swap_pairs("Options a) and b) apply")
    assert not brackets.swap_pairs("123) x (y")
    assert brackets.swap_pairs("1234) x (y")  # four digits is not a list marker


def test_inside_out_pair():
    assert brackets.inside_out_pair("١( يُدفع المبلغ )خلال")
    assert not brackets.inside_out_pair("(مثال)")
    assert not brackets.inside_out_pair("( x )")
    assert not brackets.inside_out_pair("f(x) y")


def test_guillemets_swapped():
    assert brackets.guillemets_swapped("قال " + CLOSE_Q + "نعم" + OPEN_Q + " للعرض")
    assert not brackets.guillemets_swapped("قال " + OPEN_Q + "نعم" + CLOSE_Q + " للعرض")
    assert not brackets.guillemets_swapped(CLOSE_Q + " end of a wrapped quotation")
    assert not brackets.guillemets_swapped(OPEN_Q + "a" + CLOSE_Q + " " + CLOSE_Q + "b" + OPEN_Q)


def test_faces_away():
    assert brackets.faces_away(")مثال(")
    assert brackets.faces_away("١( البند الأول")
    assert not brackets.faces_away("1) item")


@pytest.mark.parametrize(
    "text",
    [")مثال(", "المادة)٣(من", "السيد)Tomas)حاضر", "١( يُدفع المبلغ )خلال",
     "قال " + CLOSE_Q + "نعم" + OPEN_Q],
)
def test_each_pdfium_mirroring_is_caught(text):
    assert brackets.mirrored(text)


def _at(text: str, xs: list[float]) -> list[TextChar]:
    return [TextChar(ch, (x, 0.0, x + 10.0, 10.0)) for ch, x in zip(text, xs, strict=True)]


def test_neighbouring_repeats_in_reverse_index_order_pass():
    # the three dots come back right to left: same text, so no order error
    assert order_holds(_at("Wait...", [0, 10, 20, 30, 60, 50, 40]), "ltr")


def test_different_neighbours_reversed_still_fail():
    assert not order_holds(_at("ab", [10, 0]), "ltr")
