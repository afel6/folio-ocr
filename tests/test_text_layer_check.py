"""Order and content checks of the embedded text layer."""

import functools

import pytest
from unicode_samples import character_cases

from folio_ocr import bidi
from folio_ocr.brackets import faces_away
from folio_ocr.document import TextChar
from folio_ocr.flags import (
    CID_CODES,
    EMPTY_WITH_INK,
    INVISIBLE_TEXT,
    PRESENTATION_FORMS,
    PRIVATE_USE,
    TEXT_LAYER_INVALID,
    VISUAL_ORDER,
)
from folio_ocr.text_layer import segment_lines
from folio_ocr.text_layer_check import (
    check_text_layer,
    content_failures,
    line_direction,
    order_holds,
    page_direction,
)


def laid_out(text, visual_indices, width=10.0):
    """Characters of `text` boxed left to right in the given order of logical indices;
    characters not listed get no box."""
    boxes = {i: (p * width, 0.0, p * width + width, 10.0) for p, i in enumerate(visual_indices)}
    return [TextChar(c, boxes.get(i)) for i, c in enumerate(text)]


def plain(text, **kwargs):
    """Characters in logical order, laid out left to right."""
    return [
        TextChar(c, (10.0 * i, 0.0, 10.0 * i + 10.0, 10.0), **kwargs) for i, c in enumerate(text)
    ]


@functools.cache
def _sample_cases():
    return tuple(
        (text, "rtl" if direction == 1 else "ltr", list(order))
        for text, direction, _para, _levels, order in character_cases()
    )


def _mirrored(chars):
    return [
        TextChar(c.text, None if c.box is None else (-c.box[2], c.box[1], -c.box[0], c.box[3]))
        for c in chars
    ]


def _compared_count(chars):
    return sum(
        1
        for c in chars
        if c.box is not None and c.text.strip() and bidi.bidi_class(c.text) not in ("NSM", "BN")
    )


# 1. conformance layout


def test_conformance_samples_hold_in_their_unicode_display_order():
    assert len(_sample_cases()) == 2332
    failing = [t for t, d, o in _sample_cases() if not order_holds(laid_out(t, o), d)]
    assert failing == []


def test_mirrored_conformance_layout_fails_wherever_two_characters_are_compared():
    checked = 0
    passing = []
    for text, direction, order in _sample_cases():
        chars = laid_out(text, order)
        if _compared_count(chars) < 2:
            continue
        checked += 1
        if order_holds(_mirrored(chars), direction):
            passing.append(text)
    assert checked > 2000
    assert passing == []


# 2. neutrals


def test_percent_sign_is_compared_after_its_digits():
    assert order_holds(laid_out("50%", [0, 1, 2]), "ltr")
    assert not order_holds(laid_out("50%", [2, 0, 1]), "ltr")


# 3. mixed line uses the page direction


def test_mixed_line_laid_out_for_rtl_passes_on_rtl_page():
    chars = laid_out("Tomas " + "حضر", [8, 7, 6, 5, 0, 1, 2, 3, 4])
    assert line_direction(chars, "rtl") == ("rtl", True)


def test_mixed_line_laid_out_for_ltr_fails_on_rtl_page():
    chars = laid_out("Tomas " + "حضر", [0, 1, 2, 3, 4, 5, 8, 7, 6])
    assert line_direction(chars, "rtl") == ("rtl", False)


# 4. no letters


def test_digits_only_line_follows_the_page_direction_rtl():
    assert not line_direction(laid_out("50 123", [0, 1, 2, 3, 4, 5]), "rtl")[1]
    assert line_direction(laid_out("50 123", [3, 4, 5, 2, 0, 1]), "rtl") == ("rtl", True)


def test_digits_only_line_follows_the_page_direction_ltr():
    assert line_direction(laid_out("50 123", [0, 1, 2, 3, 4, 5]), "ltr") == ("ltr", True)


# 5. one-direction line


def test_latin_line_on_rtl_page_is_checked_as_ltr():
    assert line_direction(laid_out("Total", [0, 1, 2, 3, 4]), "rtl") == ("ltr", True)


def test_one_direction_line_passing_only_under_the_other_base_takes_that_base():
    # "abc !" shown with an RTL base: "!" left of "abc"; under its letters' LTR base it fails
    chars = laid_out("abc !", [4, 3, 0, 1, 2])
    assert line_direction(chars, "ltr") == ("rtl", True)
    assert VISUAL_ORDER not in check_text_layer([chars], has_ink=True).lines[0].flags


def test_reversed_latin_line_fails_under_both_bases():
    assert line_direction(laid_out("abc", [2, 1, 0]), "rtl") == ("ltr", False)
    assert line_direction(laid_out("abc", [2, 1, 0]), "ltr") == ("ltr", False)


def test_arabic_line_laid_out_left_to_right_passes_on_ltr_page_as_rtl():
    chars = laid_out("حضر", [2, 1, 0])
    assert line_direction(chars, "ltr") == ("rtl", True)


# 6. per segment


def _cell(text, x0):
    """Glyphs side by side from x0; spaces get no box."""
    chars, x = [], x0
    for c in text:
        if c == " ":
            chars.append(TextChar(c, None))
            continue
        chars.append(TextChar(c, (x, 0.0, x + 10.0, 10.0)))
        x += 10.0
    return chars


def _two_cell_row():
    return _cell("Item A", 300.0) + [TextChar(" ", None)] + _cell("Qty 2", 0.0)


def test_row_cells_are_checked_one_segment_at_a_time():
    chars = _two_cell_row()
    segments = segment_lines(chars, "rtl")
    assert len(segments) == 2
    result = check_text_layer(segments, has_ink=True)
    assert all(VISUAL_ORDER not in line.flags for line in result.lines)
    assert not order_holds(chars, "rtl")


# 7. tolerance


def _arabic_pair(box_first, box_second):
    return [TextChar("ا", box_first), TextChar("ب", box_second)]


def test_step_of_exactly_half_the_width_passes():
    assert order_holds(_arabic_pair((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)), "rtl")


def test_step_beyond_half_the_width_fails():
    assert not order_holds(_arabic_pair((0.0, 0.0, 10.0, 10.0), (5.5, 0.0, 15.5, 10.0)), "rtl")


def test_tolerance_uses_the_smaller_width_when_the_wider_glyph_comes_first():
    # visual order is the second character, then the first
    wide_first = (80.0, 0.0, 120.0, 10.0)
    assert order_holds(_arabic_pair((90.0, 0.0, 100.0, 10.0), wide_first), "rtl")
    assert not order_holds(_arabic_pair((89.0, 0.0, 99.0, 10.0), wide_first), "rtl")


# 8. explicit formatting


@pytest.mark.parametrize("mark", [0x202B, 0x2067])
def test_explicit_embedding_or_isolate_fails_closed(mark):
    assert not order_holds(plain("ab" + chr(mark) + "c"), "ltr")


def test_implicit_mark_does_not_fail_a_correct_line():
    assert order_holds(plain("abc" + chr(0x200F)), "ltr")


# 9. skipped characters


def _line_with_odd_middle(middle):
    return [
        TextChar("a", (0.0, 0.0, 10.0, 10.0)),
        middle,
        TextChar("b", (10.0, 0.0, 20.0, 10.0)),
    ]


@pytest.mark.parametrize(
    "middle",
    [
        TextChar(chr(0x064E), (-500.0, 0.0, -490.0, 10.0)),
        TextChar("x", None),
        TextChar("x", (-500.0, 0.0, -500.0, 10.0)),
        TextChar(chr(0x200C), (-500.0, 0.0, -490.0, 10.0)),
    ],
    ids=["nsm", "boxless", "zero-width", "bn"],
)
def test_skipped_characters_do_not_fail_the_line(middle):
    assert order_holds(_line_with_odd_middle(middle), "ltr")


def test_a_far_box_on_a_compared_character_does_fail():
    assert not order_holds(_line_with_odd_middle(TextChar("x", (-500.0, 0.0, -490.0, 10.0))), "ltr")


# 10. bracket rule


@pytest.mark.parametrize(
    "text",
    [
        ")" + "مثال" + "(",
        "السيد )Tomas )حاضر",
        "البند )٣( من العقد",
        ")x(",
    ],
)
def test_bracket_facing_away_from_its_word_is_detected(text):
    assert faces_away(text)


@pytest.mark.parametrize(
    "text",
    [
        "1) item",
        "١) البند",
        "a)",
        "(" + "مثال" + ")",
        "see (page",
        "x)y",
        "( x )",
        "1) Pay the invoice (by",
        "١) يُدفع المبلغ (خلال",
        "[a]",
        "{b}",
    ],
)
def test_bracket_facing_its_word_is_accepted(text):
    assert not faces_away(text)


def test_bracket_rule_flags_a_line_whose_positions_hold():
    text = ")" + "مثال" + "("
    chars = laid_out(text, [5, 4, 3, 2, 1, 0])
    assert order_holds(chars, "rtl")
    assert check_text_layer([chars], has_ink=True).lines[0].flags == (VISUAL_ORDER,)


# 11. content failures


@pytest.mark.parametrize(
    "text",
    ["(cid:12)", "a" + chr(0x01), "a" + chr(0x1F), "a" + chr(0x7F), "a" + chr(0x85)]
    + ["a" + chr(0x9F), "a" + chr(0xFFFD)],
)
def test_cid_codes_reason(text):
    assert content_failures(plain(text)) == (CID_CODES,)


def test_unmapped_glyph_is_a_cid_code():
    assert content_failures([TextChar("a", (0.0, 0.0, 10.0, 10.0), unmapped=True)]) == (CID_CODES,)


def test_tab_carriage_return_and_line_feed_are_allowed():
    assert content_failures(plain("a\tb\rc\nd")) == ()


@pytest.mark.parametrize("cp", [0xE000, 0xF0B7, 0xF0000])
def test_private_use_reason(cp):
    assert content_failures(plain("a" + chr(cp))) == (PRIVATE_USE,)


@pytest.mark.parametrize("cp", [0xFEFB, 0xFB50])
def test_presentation_forms_reason(cp):
    assert content_failures(plain("a" + chr(cp))) == (PRESENTATION_FORMS,)


def test_all_content_failures_come_in_fixed_order():
    text = chr(0xFEFB) + chr(0xE000) + "(cid:1)"
    assert content_failures(plain(text)) == (CID_CODES, PRIVATE_USE, PRESENTATION_FORMS)


def test_content_failure_marks_the_line_text_layer_invalid():
    line = check_text_layer([plain("a" + chr(0xE000))], has_ink=True).lines[0]
    assert line.reasons == (PRIVATE_USE,)
    assert line.flags == (TEXT_LAYER_INVALID,)


# 12. page reasons


def test_invisible_text_flags_every_line_but_does_not_send_the_page_to_ocr():
    lines = [plain("abc", invisible=True), plain("def"), plain("ghi")]
    result = check_text_layer(lines, has_ink=True)
    assert result.reasons == (INVISIBLE_TEXT,)
    assert result.invalid_lines == 0
    assert not result.to_ocr
    assert all(TEXT_LAYER_INVALID in line.flags for line in result.lines)


def test_more_than_half_of_the_lines_with_content_failures_send_the_page_to_ocr():
    bad, good = plain("ab" + chr(0xE000)), plain("abc")
    result = check_text_layer([bad, bad, bad, good, good], has_ink=True)
    assert TEXT_LAYER_INVALID in result.reasons
    assert result.invalid_lines == 3
    assert result.to_ocr


def test_exactly_half_of_the_lines_with_content_failures_keep_the_text_layer():
    bad, good = plain("ab" + chr(0xE000)), plain("abc")
    result = check_text_layer([bad, bad, good, good], has_ink=True)
    assert result.reasons == ()
    assert result.invalid_lines == 2
    assert not result.to_ocr


@pytest.mark.parametrize("lines", [[], [plain("  ")], [plain(" "), plain("\t")]])
def test_no_text_with_ink_goes_to_ocr(lines):
    result = check_text_layer(lines, has_ink=True)
    assert result.reasons == (EMPTY_WITH_INK,)
    assert result.to_ocr


@pytest.mark.parametrize("lines", [[], [plain("  ")]])
def test_no_text_without_ink_has_no_reason(lines):
    result = check_text_layer(lines, has_ink=False)
    assert result.reasons == ()
    assert not result.to_ocr


def test_page_reasons_come_in_fixed_order():
    result = check_text_layer([plain("a" + chr(0xE000), invisible=True)], has_ink=True)
    assert result.reasons == (INVISIBLE_TEXT, TEXT_LAYER_INVALID)


# 13. page direction


def test_latin_majority_is_ltr():
    assert page_direction(plain("abc " + "ب")) == "ltr"


def test_arabic_majority_is_rtl():
    assert page_direction(plain("a " + "بتث")) == "rtl"


def test_equal_counts_are_rtl():
    assert page_direction(plain("ab" + "بت")) == "rtl"


def test_no_letters_is_rtl():
    assert page_direction(plain("123 45")) == "rtl"
    assert page_direction([]) == "rtl"


def test_private_use_and_digits_are_not_letters():
    assert page_direction(plain(chr(0xF0B7) * 5 + "12345" + "ب")) == "rtl"
    assert page_direction(plain(chr(0xF0B7) * 5 + "12345" + "a")) == "ltr"
