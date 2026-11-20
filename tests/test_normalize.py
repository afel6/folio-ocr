import unicodedata

from folio_ocr.normalize import lenient_v2, norm_v2


def test_quotes_and_dash_pin():
    assert norm_v2("“Hi”— x") == '"Hi"- x'


def test_tatweel_removed():
    assert norm_v2("مـــحمد") == "محمد"


def test_short_vowels_removed():
    assert norm_v2("شُكْرًا") == "شكرا"


def test_lam_alef_presentation_form_expanded():
    assert norm_v2("\uFEFB") == "لا"


def test_arabic_indic_digits_and_decimal_separator():
    assert norm_v2("١٢٣٫٥") == "123.5"


def test_extended_digits():
    assert norm_v2("\u06F4\u06F5") == "45"


def test_hamza_kept_strict_folded_lenient():
    assert norm_v2("أحمد") == "أحمد"
    assert lenient_v2("أحمد") == "احمد"


def test_isolated_tashkeel_deleted_before_nfkc():
    assert norm_v2("ب\uFE70ت") == "بت"


def test_english_not_nfkc_normalised():
    assert norm_v2("\uFB01le") == "\uFB01le"
    assert norm_v2("\uFF21") == "\uFF21"
    assert norm_v2("x²") == "x²"


def test_madda_kept_strict_dropped_lenient():
    assert norm_v2("مٓ") == "مٓ"
    assert lenient_v2("مٓ") == "م"


def test_presentation_form_folded_before_tatweel_and_vowel_deletion():
    assert norm_v2("\uFCF2") == ""


def test_nfc_before_tatweel_deletion_keeps_hamza_off_alef():
    assert norm_v2("\u0627\u0640\u0654") == "\u0627\u0654"


def test_range_ends_of_each_step():
    assert norm_v2("a\u200Db") == "ab"
    assert norm_v2("a\u202Eb") == "ab"
    assert norm_v2("a\uFE7Fb") == "ab"
    assert norm_v2("\uFEFC") == unicodedata.normalize("NFKC", "\uFEFC")
    assert norm_v2("\uFB50") == unicodedata.normalize("NFKC", "\uFB50")
    assert norm_v2("\u0628\u0670") == "\u0628"


def test_remaining_range_ends():
    assert norm_v2(chr(0x0669)) == "9"
    assert norm_v2(chr(0x06F0)) == "0"
    for cp in (0x061C, 0x200E, 0x2066):
        assert norm_v2("a" + chr(cp) + "b") == "ab"
    for cp in (0xFDFC, 0xFE80):
        assert norm_v2(chr(cp)) == unicodedata.normalize("NFKC", chr(cp))
    assert norm_v2(chr(0xFDFC)) == chr(0x0631) + chr(0x06CC) + chr(0x0627) + chr(0x0644)
    assert norm_v2("a" + chr(0x0655)) == "a" + chr(0x0655)  # kept by the strict form


def test_percent_and_thousands_separator():
    assert norm_v2("٥٪") == "5%"
    assert norm_v2("١٬٢") == "1,2"


def test_invisible_marks_removed():
    assert norm_v2("a\u200Bb\u200F") == "ab"
    assert norm_v2("\uFEFFa\u202Ab\u2069") == "ab"


def test_whitespace_collapsed():
    assert norm_v2(" a \u00A0\n b ") == "a b"


def test_case_kept_strict():
    assert norm_v2("Hi") == "Hi"


def test_arabic_comma_strict_vs_lenient():
    assert norm_v2("،") == "،"
    assert lenient_v2("،") == ""


def test_lenient_folds_letters():
    assert lenient_v2("\u0649") == "ي"
    assert lenient_v2("\u06CC") == "ي"
    assert lenient_v2("ة") == "ه"
    assert lenient_v2("\u06A9") == "ك"
    assert lenient_v2("ٱ") == "ا"
    assert lenient_v2("إ") == "ا"
    assert lenient_v2("آ") == "ا"


def test_lenient_removes_punctuation_and_case():
    assert lenient_v2("Hi, there!") == "hi there"


def test_ellipsis():
    assert norm_v2("…") == "..."
    assert lenient_v2("…") == ""


def test_empty():
    assert norm_v2("") == ""
    assert lenient_v2("") == ""
