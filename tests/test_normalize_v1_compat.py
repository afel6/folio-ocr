"""The current normalisation gives the same text as the frozen first version on its pins."""

import unicodedata

import pytest

from folio_ocr.metrics import cer, wer
from folio_ocr.normalize import lenient_v2, norm_v2

_V1_MAP = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x2032: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x2033: '"',
    0x2010: "-", 0x2011: "-", 0x2013: "-", 0x2014: "-", 0x2212: "-",
    0x2026: "...",
    0x200B: None, 0x200C: None, 0x200D: None, 0xFEFF: None,
}  # fmt: skip


def norm_v1(text):
    return " ".join(unicodedata.normalize("NFC", text).translate(_V1_MAP).split())


def lenient_v1(text):
    folded = norm_v1(text).casefold()
    kept = "".join(c for c in folded if not unicodedata.category(c).startswith("P"))
    return " ".join(kept.split())


PINS = [
    "“Hi”— x", "abc", "abd", "a b", "ab", "the cat sat", "the sat",
    *[f"a{chr(cp)}b" for cp in _V1_MAP],
    "a   \n\tb", "  lead and trail  ", "été",
]  # fmt: skip


@pytest.mark.parametrize("text", PINS)
def test_current_rules_equal_first_version_rules(text):
    assert norm_v2(text) == norm_v1(text)
    assert lenient_v2(text) == lenient_v1(text)


def test_single_quote_variants_become_apostrophes():
    assert norm_v2("a‘b’c‚d′e") == "a'b'c'd'e"


def test_double_quote_variants_become_straight_quotes():
    assert norm_v2("a“b”c„d″e") == 'a"b"c"d"e'


def test_dash_variants_become_hyphens():
    assert norm_v2("a‐b‑c–d—e−f") == "a-b-c-d-e-f"


def test_ellipsis_becomes_three_dots():
    assert norm_v2("a…") == "a..."


def test_zero_width_characters_are_removed():
    assert norm_v2("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"


def test_quoted_dash_pin_value():
    assert norm_v2("“Hi”— x") == '"Hi"- x'


@pytest.mark.parametrize("pair", [("abc", "abd"), ("a b", "ab")])
def test_character_error_rate_is_the_same_under_both_versions(pair):
    assert cer([pair]).rate == cer([pair], normalize=norm_v1).rate == 1 / 3


def test_word_error_rate_is_the_same_under_both_versions():
    pair = ("the cat sat", "the sat")
    assert wer([pair]).rate == wer([pair], normalize=norm_v1).rate == 1 / 3
