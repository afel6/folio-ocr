"""Tests for the per-line measures of model replies against ground-truth lines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import model_run_lines as sl

from folio_ocr.normalize import norm_v2

VOWELLED = "كَتَبَ"


def test_constants():
    assert len(sl.TASHKEEL) == 9 and "ٰ" in sl.TASHKEEL and "ً" in sl.TASHKEEL
    assert len(sl.ORDER_CLASSES) == 5
    assert sl.MIN_ORDER_LINES == 40


def test_digit_script():
    assert sl.digit_script("5") == "european"
    assert sl.digit_script("٣") == "arabic-indic"
    assert sl.digit_script("۷") == "arabic-indic"
    assert sl.digit_script("a") is None


def test_digit_runs():
    assert sl.digit_runs("a12-٣٤5") == ("12", "٣٤", "5")
    assert sl.digit_runs("12٣٤") == ("12", "٣٤")
    assert sl.digit_runs("abc") == ()


def test_order_classes():
    assert sl.order_classes("hello 12") == ()
    assert sl.order_classes("اسم Tomas 12") == ("arabic+latin",)
    assert sl.order_classes("اسم") == ()
    assert sl.order_classes("رقم 12") == ("arabic+1run/european",)
    assert sl.order_classes("رقم ١٢") == ("arabic+1run/arabic-indic",)
    assert sl.order_classes("رقم 1 و 2") == ("arabic+2runs/european",)
    assert sl.order_classes("رقم ١ و ٢") == ("arabic+2runs/arabic-indic",)
    assert sl.order_classes("رقم 1 ٢") == ("arabic+2runs/european", "arabic+2runs/arabic-indic")


def test_script_runs():
    assert sl.script_runs("اسم العميل: Tomas Okafor") == (
        ("arabic", "اسمالعميل"),
        ("latin", "TomasOkafor"),
    )
    assert sl.script_runs("ab 12 3 cd") == (
        ("latin", "ab"),
        ("digit", "12"),
        ("digit", "3"),
        ("latin", "cd"),
    )
    assert sl.script_runs("") == ()


def test_order_error_cases():
    gt = "اسم العميل: Tomas Okafor"
    assert sl.order_error(gt, "Tomas Okafor : اسم العميل") is True
    assert sl.order_error("رقم الفاتورة ١٢٣٤", "١٢٣٤ رقم الفاتورة") is True
    assert sl.order_error(gt, gt) is False
    assert sl.order_error("رقم الفاتورة ١٢٣٤", "رقم الفاتورة 1234") is False
    assert sl.order_error("المبلغ 1234", "المبلغ 4321") is True
    assert sl.order_error(gt, "اسم العميل: Tomas Okafcr") is False
    assert sl.order_error("رقم الفاتورة ١٢٣٤", "رقم الفاتورة") is False
    assert sl.order_error(gt, "") is False


def test_order_error_when_a_reply_joins_two_runs_of_one_kind():
    assert sl.order_error("المبلغ 1234 ريال", "المبلغ ريال 1234") is True
    assert sl.order_error("اسم Tomas العميل", "اسم العميل Tomas") is True


def test_the_second_arabic_run_does_not_reuse_the_first_runs_window():
    # the two Arabic runs must not share a window: the second one is not placed over the first
    assert sl.order_error("اب 1 اب", "اب 1") is False


def test_order_error_ignores_a_garbage_reply():
    assert sl.order_error("اسم العميل: Tomas Okafor", "qwxzv kjhgf") is False


def test_a_window_with_exactly_half_the_characters_wrong_is_placed():
    # "abcd" against "abxy" is 2 errors in 4 (0.5): placed first in the reply, so the order is wrong
    assert sl.order_error("اسم abcd", "abxy اسم") is True


def test_a_window_with_more_than_half_the_characters_wrong_is_not_placed():
    # "abcdefg" against "abcwxyz" is 4 errors in 7 (just above 0.5): no place, so no order error
    assert sl.order_error("اسم abcdefg", "abcwxyz اسم") is False


def test_a_window_is_not_placed_when_it_starts_on_a_character_of_another_kind():
    # the only close window for the latin "ab" starts on an Arabic letter; it must be skipped
    assert sl.order_error("ab اب", "اب اب b") is False


def test_with_tashkeel():
    assert norm_v2(VOWELLED) != VOWELLED
    assert sl.with_tashkeel(VOWELLED) == VOWELLED
    assert sl.with_tashkeel("  كَتَبَ   ١٢ ") == VOWELLED + " 12"


def test_tashkeel_count():
    assert sl.tashkeel_count(VOWELLED) == 3
    assert sl.tashkeel_count("كتب") == 0


def measures(gt, reply, vocab=()):
    return sl.line_measures(gt, reply, frozenset(vocab))


def test_line_measures_keys_and_cer():
    m = measures("abcd", "abcx")
    assert set(m) == {
        "cer",
        "cer_with_tashkeel",
        "tashkeel_gt",
        "tashkeel_reply",
        "digit_inserts",
        "digit_deletes",
        "latin_in_arabic",
        "real_word_misreads",
        "digits_kept",
        "digits_total",
        "order_error",
    }
    assert m["cer"] == 0.25
    assert measures("", "x")["cer"] is None


def test_line_measures_tashkeel():
    m = measures(VOWELLED, "كتب")
    assert m["cer"] == 0.0
    assert m["cer_with_tashkeel"] == 0.5
    assert (m["tashkeel_gt"], m["tashkeel_reply"]) == (3, 0)


def test_line_measures_digits():
    m = measures("a12", "a1 345")
    assert (m["digit_inserts"], m["digit_deletes"]) == (2, 0)
    m = measures("a123", "a1")
    assert (m["digit_inserts"], m["digit_deletes"]) == (0, 2)
    m = measures("abc", "abc")
    assert (m["digit_inserts"], m["digit_deletes"]) == (0, 0)


def test_line_measures_latin_in_arabic():
    assert measures("اسم", "اسم ab")["latin_in_arabic"] == 2
    assert measures("اسم ab", "اسم abc")["latin_in_arabic"] == 0
    assert measures("ab", "abc")["latin_in_arabic"] == 0


def test_line_measures_real_word_misreads():
    m = measures("the cat sat", "the cot sat sat", {"cot", "sat", "xyz"})
    assert m["real_word_misreads"] == 2  # cot and the extra sat
    assert measures("the cat", "the kat", {"cot"})["real_word_misreads"] == 0


def test_line_measures_digit_script_and_order():
    m = measures("رقم ١٢", "رقم 12")
    assert (m["digits_kept"], m["digits_total"]) == (0, 2)
    assert m["order_error"] is False
    assert measures("hello", "hello")["order_error"] is None


def test_pooled():
    recs = [
        {"gt": "abcd", "text": "abcx"},
        {"gt": "ab", "text": None},
        {"gt": "", "text": "zz"},
    ]
    assert sl.pooled(recs) == {"lines": 2, "reference_chars": 6, "errors": 3, "cer": 0.5}
    assert sl.pooled([])["cer"] is None


def rec(classes=("arabic+latin",), err=False, flags=(), reader="model", conf=0.9):
    return {
        "classes": classes,
        "reader": reader,
        "flags": flags,
        "order_error": err,
        "confidence": conf,
    }


def test_order_summary():
    recs = [rec() for _ in range(sl.MIN_ORDER_LINES)]
    recs += [rec(err=True, flags=("mixed_order",)), rec(reader="cnn", err=True)]
    out = sl.order_summary(recs)
    c = out["classes"]["arabic+latin"]
    assert c["lines"] == 41 and c["order_errors"] == 1
    assert c["unflagged_order_errors"] == 0 and c["status"] == "ok"
    assert c["unflagged_by_any_flag"] == 0
    assert out["classes"]["arabic+1run/european"]["status"] == "short"
    assert out["verdict"] == "stop" and out["min_lines"] == 40
    recs.append(rec(err=True, flags=("low_confidence",)))
    recs.append(rec(err=True, flags=("other",)))
    recs.append(rec(err=True, conf=None))
    c = sl.order_summary(recs)["classes"]["arabic+latin"]
    assert c["unflagged_order_errors"] == 3 and c["status"] == "widen"
    assert c["unflagged_by_any_flag"] == 1


def test_order_summary_frozen():
    recs = [rec(classes=(k,)) for k in sl.ORDER_CLASSES for _ in range(40)]
    assert sl.order_summary(recs)["verdict"] == "frozen"


def mrec(group="ar", **kw):
    base = {
        "script_group": group,
        "reader": "model",
        "gt": "ab",
        "text": "ab",
        "flags": (),
        "tashkeel_gt": 0,
        "tashkeel_reply": 0,
        "digit_inserts": 0,
        "digit_deletes": 0,
        "latin_in_arabic": 0,
        "real_word_misreads": 0,
        "digits_kept": 0,
        "digits_total": 0,
        "completion_tokens": None,
        "max_tokens": None,
    }
    return {**base, **kw}


def test_measure_summary():
    recs = [
        mrec(
            tashkeel_reply=2,
            digits_kept=1,
            digits_total=2,
            latin_in_arabic=3,
            real_word_misreads=2,
            digit_inserts=1,
            flags=("truncated",),
            completion_tokens=10,
            max_tokens=10,
        ),
        mrec(
            tashkeel_gt=1,
            digit_deletes=4,
            flags=("truncated",),
            completion_tokens=5,
            max_tokens=10,
            text="ax",
        ),
        mrec(reader="cnn"),
        mrec(group="en"),
    ]
    out = sl.measure_summary(recs)
    ar = out["ar"]
    assert ar["lines"] == 2
    assert ar["cer"] == 0.25 and ar["cer_with_tashkeel"] == 0.25
    assert (ar["tashkeel_added_lines"], ar["tashkeel_dropped_lines"]) == (1, 1)
    assert (ar["digit_inserts"], ar["digit_deletes"]) == (1, 4)
    assert (ar["latin_in_arabic_lines"], ar["latin_in_arabic_letters"]) == (1, 3)
    assert (ar["real_word_misreads"], ar["real_word_misread_lines"]) == (2, 1)
    assert (ar["digits_kept"], ar["digits_total"], ar["digit_script_kept_share"]) == (1, 2, 0.5)
    assert ar["truncated_lines"] == 2 and ar["truncated_share"] == 1.0
    assert ar["truncated_at_max_tokens"] == 1
    assert out["en"]["digit_script_kept_share"] is None
    assert out["en"]["truncated_share"] == 0.0
