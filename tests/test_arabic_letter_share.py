"""The Arabic share of a line counts letters only; vowel marks are not letters."""

import pytest

from folio_ocr.flags import arabic_letter_share, script_mismatch

VOCALISED = (
    "شُكْرًا "
    "جَزِيلًا "
    "لَكُمْ"
)


def test_fully_vocalised_arabic_is_all_arabic():
    assert arabic_letter_share(VOCALISED) == 100.0


def test_vocalised_arabic_matches_an_arabic_page():
    assert script_mismatch(VOCALISED, "ar") is False


def test_vocalised_arabic_mismatches_an_english_page():
    assert script_mismatch(VOCALISED, "en") is True


def test_vocalised_arabic_matches_a_mixed_page():
    assert script_mismatch(VOCALISED, "en+ar") is False


def test_marks_do_not_count_towards_the_share():
    text = "Tomas شُكْرًا"
    assert arabic_letter_share(text) == pytest.approx(400 / 9)


def test_latin_accents_are_not_arabic():
    assert arabic_letter_share("été") == 0.0
    assert script_mismatch("été", "en") is False
