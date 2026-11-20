import subprocess
import sys

import pytest

from folio_ocr import flags as f
from folio_ocr.document import Line

AR = "مرحبا بكم"
BLOCK = "abcdefghijklmnopqrst"  # 20 distinct characters


def make_line(flags=(), status="auto_accepted") -> Line:
    return Line(
        id="p1-l1", box=(0.0, 0.0, 10.0, 5.0), text="hi", direction="ltr",
        script="latin", reader="text_layer", confidence=None, confidence_method="none",
        flags=flags, status=status, final_text="hi",
    )


# clean_text
def test_clean_text_replaces_lone_surrogates():
    assert f.clean_text("\ud800x") == ("\uFFFDx", ("invalid_text",))
    assert f.clean_text("a\udc00") == ("a\uFFFD", ("invalid_text",))


def test_clean_text_combines_a_valid_pair_given_as_two_surrogates():
    # replacing each surrogate on its own would give two U+FFFD and a flag
    pair = chr(0xD83D) + chr(0xDE00)
    assert f.clean_text(pair) == (chr(0x1F600), ())
    assert f.clean_text(pair + chr(0xD800)) == (chr(0x1F600) + chr(0xFFFD), ("invalid_text",))


def test_clean_text_does_not_flag_an_existing_replacement_char():
    assert f.clean_text("\uFFFD") == ("\uFFFD", ())
    assert f.clean_text("abc") == ("abc", ())


def test_flags_module_does_not_load_pdf_library():
    code = "import sys, folio_ocr.flags; print('pypdfium2' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# R2
def test_is_empty_means_whitespace_only():
    assert f.is_empty("") and f.is_empty("  \n\t")
    assert not f.is_empty(" a ")


def test_is_repeated_counts_overlapping_windows():
    # 22 equal letters hold the same 20-char window 3 times; non-overlapping counting misses it
    assert f.is_repeated("a" * 22)
    assert not f.is_repeated("a" * 21)


def test_is_repeated_on_blocks_and_normal_text():
    assert f.is_repeated(BLOCK * 3)
    assert not f.is_repeated(BLOCK * 2)
    assert not f.is_repeated("Invoice INV-2026-0417 for Daniel Mercer, total 1,250.00")


# R3 script share
def test_arabic_letter_share_counts_letters_only():
    assert f.arabic_letter_share("123 !") is None
    assert f.arabic_letter_share("ab 12") == 0
    assert f.arabic_letter_share(AR + " ١٢") == 100
    assert f.arabic_letter_share("ab" + "سم") == 50


def test_script_mismatch_cases():
    assert f.script_mismatch("Invoice 42", "ar")
    assert not f.script_mismatch("فاتورة", "ar")
    assert f.script_mismatch("فاتورة", "en")
    assert not f.script_mismatch("فاتورة Invoice", "en+ar")
    assert not f.script_mismatch("١٢٣", "ar")


def test_script_mismatch_en_ar_accepts_pure_lines_of_either_script():
    assert not f.script_mismatch("Invoice total", "en+ar")  # 0 % Arabic
    assert not f.script_mismatch("فاتورة", "en+ar")  # 100 % Arabic


def test_script_mismatch_exactly_thirty_points_off_is_not_flagged():
    assert not f.script_mismatch("سسسسسسس" + "abc", "ar")  # 70 % Arabic
    assert f.script_mismatch("سسسسسسسسسسسسس" + "abcdefg", "ar")  # 65 %
    assert not f.script_mismatch("سسس" + "abcdefg", "en")  # 30 %
    assert f.script_mismatch("سسسس" + "abcdef", "en")  # 40 %


def test_script_mismatch_rejects_unknown_languages():
    with pytest.raises(ValueError):
        f.script_mismatch("a", "fr")


# R3 code points
def test_presentation_forms():
    assert f.has_presentation_forms("\uFEFB")
    assert f.has_presentation_forms("\uFB50")
    assert f.has_presentation_forms("\uFEFF")  # inside the stated U+FE70-FEFF range
    assert not f.has_presentation_forms("لا")
    assert not f.has_presentation_forms("\uFB01")  # Latin ligature block, not Arabic forms


def test_presentation_form_range_ends():
    assert not f.has_presentation_forms(chr(0xFE0F))  # variation selector, outside both ranges
    assert f.has_presentation_forms(chr(0xFDFF))
    assert f.has_presentation_forms(chr(0xFE70))


def test_lookalikes():
    assert f.has_lookalikes("B\u0430nk")  # Cyrillic a
    assert not f.has_lookalikes("Bank")
    assert f.has_lookalikes("\u06CC")  # Farsi yeh
    assert not f.has_lookalikes("ي")
    assert f.has_lookalikes("\u06F4")  # extended Arabic-Indic four
    assert not f.has_lookalikes("٤")
    assert f.has_lookalikes("\uFF21")


def test_line_flags_rejects_unknown_languages_even_for_empty_text():
    with pytest.raises(ValueError):
        f.line_flags("", "fr")


def test_line_flags_order_and_empty_stop():
    assert f.line_flags("  ", "en") == ("empty",)
    assert f.line_flags("", "ar", handwritten=True) == ("handwritten", "empty")
    text = "B\u0430nk " + "x" * 22 + " \uFEFB"
    assert f.line_flags(text, "ar") == (
        "repeated", "script_mismatch", "presentation_forms", "lookalike_chars",
    )
    assert f.line_flags("hello", "en", handwritten=True) == ("handwritten",)
    assert f.line_flags("hello", "en") == ()


# model server and review status
@pytest.mark.parametrize(
    ("languages", "model", "tesseract", "expected"),
    [
        ("ar", False, False, True),
        ("en+ar", False, False, True),
        ("en", False, False, False),
        ("ar", True, False, False),
        ("ar", False, True, False),
        ("en+ar", True, True, False),
    ],
)
def test_needs_model_server(languages, model, tesseract, expected):
    got = f.needs_model_server(languages, model_configured=model, tesseract_configured=tesseract)
    assert got is expected


def test_needs_model_server_rejects_unknown_languages():
    with pytest.raises(ValueError):
        f.needs_model_server("fr", model_configured=False, tesseract_configured=False)


def test_review_status():
    assert f.review_status((), "auto_accepted") == "auto_accepted"
    assert f.review_status((), "needs_review") == "auto_accepted"
    assert f.review_status(("empty",), "auto_accepted") == "needs_review"
    assert f.review_status(("empty",), "edited") == "edited"
    assert f.review_status(("empty",), "confirmed") == "confirmed"


def test_with_flags_merges_without_duplicates_and_returns_new_line():
    line = make_line(flags=("empty",))
    out = f.with_flags(line, ["empty", "repeated"])
    assert out.flags == ("empty", "repeated") and out.status == "needs_review"
    assert line.flags == ("empty",) and line.status == "auto_accepted"


def test_with_flags_keeps_reviewer_status():
    assert f.with_flags(make_line(status="edited"), ["empty"]).status == "edited"
    assert f.with_flags(make_line(status="confirmed"), ["empty"]).status == "confirmed"
    assert f.with_flags(make_line(), []).status == "auto_accepted"


# script_mismatch with letters from other scripts
@pytest.mark.parametrize("lang", ["en", "ar", "en+ar"])
@pytest.mark.parametrize("cp", [0x3B1, 0x430, 0x5D0, 0x4E2D])
def test_other_script_letter_flags_in_every_mode(lang, cp):
    assert f.script_mismatch(chr(cp), lang)


def test_one_cyrillic_letter_inside_latin_word_flags_in_en_ar():
    assert f.script_mismatch("B" + chr(0x0430) + "nk", "en+ar")
    assert f.script_mismatch("B" + chr(0x0430) + "nk", "en")


def test_one_greek_letter_in_arabic_text_flags():
    assert f.script_mismatch(AR + " " + chr(0x3B1), "ar")


@pytest.mark.parametrize("text", [
    "caf" + chr(0xE9), "Stra" + chr(0xDF) + "e", chr(0xFB01) + "sh", chr(0xB5) + "m",
])
def test_accented_latin_does_not_flag_in_en_ar(text):
    assert not f.script_mismatch(text, "en+ar")
    assert not f.script_mismatch(text, "en")


def test_arabic_with_tatweel_and_arabic_indic_digits_does_not_flag():
    text = AR[:2] + chr(0x0640) + AR[2:] + " " + chr(0x0661) + chr(0x0662)
    assert not f.script_mismatch(text, "en+ar")
    assert not f.script_mismatch(text, "ar")


@pytest.mark.parametrize("lang", ["en", "ar", "en+ar"])
def test_digits_and_punctuation_alone_never_flag(lang):
    assert not f.script_mismatch("12.5 - (7) %", lang)
