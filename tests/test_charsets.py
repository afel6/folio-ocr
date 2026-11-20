import pytest

from folio_ocr import charsets


def test_in_ranges():
    assert charsets.in_ranges("\ufefb", charsets.PRESENTATION_FORMS)
    assert not charsets.in_ranges("ل", charsets.PRESENTATION_FORMS)


@pytest.mark.parametrize(
    "ch,expected",
    [
        ("a", "latin"),
        ("ب", "arabic"),
        ("\u064b", "arabic"),
        ("\u0301", "latin"),
        ("5", None),
        (" ", None),
        (".", None),
    ],
)
def test_script_of(ch, expected):
    assert charsets.script_of(ch) == expected


@pytest.mark.parametrize(
    "ch,expected",
    [
        ("ب", True),
        ("a", False),
        ("\u064b", False),
        ("٣", False),
        ("\ufefb", True),
    ],
)
def test_is_arabic_letter(ch, expected):
    assert charsets.is_arabic_letter(ch) is expected


@pytest.mark.parametrize(
    "cp",
    [
        0x41,
        0x7A,
        0xAA,
        0xB5,
        0xBA,
        0xC0,
        0xE9,
        0xDF,
        0x24F,
        0x250,
        0x2AF,
        0x1D00,
        0x1E00,
        0x2071,
        0x207F,
        0x2090,
        0x2C60,
        0xA722,
        0xAB30,
        0xFB01,
        0xFF21,
        0xFF5A,
    ],
)
def test_is_latin_letter_true(cp):
    assert charsets.is_latin_letter(chr(cp))


@pytest.mark.parametrize(
    "cp",
    [
        0x30,
        0x20,
        0x2E,
        0xD7,  # digit, space, full stop, multiplication sign
        0x3B1,
        0x430,
        0x5D0,
        0x628,
        0x4E2D,  # Greek, Cyrillic, Hebrew, Arabic, CJK
        0x300,  # combining mark, not a letter
    ],
)
def test_is_latin_letter_false(cp):
    assert not charsets.is_latin_letter(chr(cp))


@pytest.mark.parametrize("cp", [0x3B1, 0x430, 0x5D0, 0x4E2D, 0x3042, 0xAC00, 0xE01])
def test_is_other_script_letter_true(cp):
    assert charsets.is_other_script_letter(chr(cp))


@pytest.mark.parametrize(
    "cp",
    [
        0x61,
        0xE9,
        0xB5,
        0xFB01,
        0x628,
        0x640,
        0x30,
        0x66B,
        0x20,
        0x2E,
        0x301,
    ],
)
def test_is_other_script_letter_false(cp):
    # tatweel (0x640) is a modifier letter inside the Arabic block
    assert not charsets.is_other_script_letter(chr(cp))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice 2026", "latin"),
        ("فاتورة ٢٠٢٦", "arabic"),
        ("الاسم: Tomas", "mixed"),
        ("12/05/2026 - 350.75", "unknown"),
        ("", "unknown"),
        ("Привет", "unknown"),
        ("Tomas Привет", "mixed"),
        ("َُ", "unknown"),
    ],
)
def test_line_script(text, expected):
    assert charsets.line_script(text) == expected
