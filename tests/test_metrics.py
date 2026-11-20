import jiwer
import pytest

from folio_ocr.metrics import (
    EntityMatch,
    ErrorCounts,
    Ratio,
    cer,
    cer_by_script,
    char_errors,
    compose,
    digit_script_preserved,
    entity_exact_match,
    gt_line_boxes,
    iou,
    line_recall,
    map_box,
    match_entities,
    match_lines,
    missed_error_rate,
    reading_order_errors,
    wer,
)


def test_cer_pins():
    assert cer([("abc", "abd")]).rate == 1 / 3
    assert cer([("a b", "ab")]).rate == 1 / 3


def test_wer_pin():
    assert wer([("the cat sat", "the sat")]).rate == 1 / 3


def test_cer_pooled_not_averaged():
    assert cer([("a", "b"), ("abcdefghij", "abcdefghij")]).rate == 1 / 11


def test_wer_pooled_not_averaged():
    pairs = [("a", "b"), ("a b c d e f g h i j", "a b c d e f g h i j")]
    assert wer(pairs).rate == 1 / 11


PAIRS = [
    ("hello world", "hallo word"),
    ("the quick brown fox", "the quick fox"),
    ("abc def", "abc def ghi"),
    ("one two three", "on two thre"),
]


def test_jiwer_oracle():
    refs = [p[0] for p in PAIRS]
    hyps = [p[1] for p in PAIRS]
    assert cer(PAIRS).rate == pytest.approx(jiwer.cer(refs, hyps))
    assert wer(PAIRS).rate == pytest.approx(jiwer.wer(refs, hyps))


def test_sdi_split():
    c = char_errors("abc", "xbcd")
    assert (c.substitutions, c.deletions, c.insertions) == (1, 0, 1)
    assert c.reference_length == 3
    assert c.edits == 2


def test_empty_hypothesis_is_all_deletions():
    c = cer([("abc", "")])
    assert (c.substitutions, c.deletions, c.insertions) == (0, 3, 0)


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        cer([("", "a")])
    with pytest.raises(ValueError):
        wer([(" ", "a")])
    with pytest.raises(ValueError):
        cer_by_script([("", "a")])


def test_normalisation_applied():
    assert cer([("١٢٣", "123")]).rate == 0
    assert cer([("مـحمد", "محمد")]).rate == 0
    assert cer([("شُكرا", "شكرا")]).rate == 0


def test_custom_normalize():
    assert cer([("AB", "ab")], normalize=str.lower).rate == 0


def test_counts_add_and_rate_none():
    s = ErrorCounts(1, 2, 3, 10) + ErrorCounts(1, 0, 0, 5)
    assert s == ErrorCounts(2, 2, 3, 15)
    assert ErrorCounts(0, 0, 0, 0).rate is None
    assert Ratio(0, 0).rate is None
    assert Ratio(1, 4).rate == 0.25


def test_script_only_latin_part_wrong():
    out = cer_by_script([("مرحبا hello", "مرحبا hallo")])
    assert set(out) == {"arabic", "latin"}
    assert out["arabic"].rate == 0
    assert out["latin"].edits == 1


def test_script_insertion_of_latin_letter_counts_to_latin():
    out = cer_by_script([("مرحبا", "مرحبxا")])
    assert out["latin"].insertions == 1
    assert out["arabic"].edits == 0


def test_script_digit_error_on_latin_line_in_arabic_batch_counts_to_latin():
    pairs = [("\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0627\u0644\u0645\u0643\u062a\u0628",) * 2,
             ("total 123", "total 124")]
    out = cer_by_script(pairs)
    assert out["latin"].substitutions == 1
    assert out["arabic"].edits == 0


def test_script_digit_errors_on_arabic_line_count_to_arabic():
    out = cer_by_script([("رقم 123", "رقم 124")])
    assert out["arabic"].substitutions == 1
    assert out["latin"].edits == 0
    assert out["arabic"].reference_length == 7
    assert out["latin"].reference_length == 0


def test_script_tie_uses_global_majority_then_latin():
    # first line is digits only: tie, global majority is arabic
    out = cer_by_script([("12", "13"), ("مرحبا", "مرحبا")])
    assert out["arabic"].substitutions == 1
    # no letters anywhere: latin
    out = cer_by_script([("12", "13")])
    assert out["latin"].substitutions == 1


def test_script_deletions_by_reference_char():
    out = cer_by_script([("abc مرحبا", "abc مرح")])
    assert out["arabic"].deletions == 2
    assert out["latin"].edits == 0


def test_entities_exact():
    ref = "pay 100 now"
    m = match_entities(ref, ref, [(4, 7)])
    assert m == (EntityMatch("100", "100", True, True),)
    assert entity_exact_match(m, by="raw") == Ratio(1, 1)


def test_entities_digit_script_raw_false_value_true():
    m = match_entities("pay ١٢٣ now", "pay 123 now", [(4, 7)])
    assert (m[0].raw_exact, m[0].value_exact) == (False, True)
    assert entity_exact_match(m, by="value") == Ratio(1, 1)
    assert entity_exact_match(m, by="raw") == Ratio(0, 1)


def test_entities_insertion_inside_vs_before():
    inside = match_entities("pay 100 now", "pay 1x00 now", [(4, 7)])
    assert inside[0].hypothesis == "1x00"
    assert not inside[0].raw_exact
    before = match_entities("pay 100 now", "pay x100 now", [(4, 7)])
    assert before[0].hypothesis == "100"
    assert before[0].raw_exact


def test_entities_deleted():
    m = match_entities("abc XYZ def", "abc  def", [(4, 7)])
    assert m[0].hypothesis == ""
    assert not m[0].raw_exact and not m[0].value_exact


def test_entities_replace_followed_by_delete():
    m = match_entities("a 12 b", "a 9 b", [(2, 4)])
    assert m[0].hypothesis == "9"


@pytest.mark.parametrize("span", [(-1, 2), (2, 2), (3, 2), (0, 99), (0.0, 2)])
def test_entities_bad_span(span):
    with pytest.raises(ValueError):
        match_entities("abc", "abc", [span])


def test_entity_exact_match_bad_by():
    with pytest.raises(ValueError):
        entity_exact_match([], by="other")


def test_digit_script_arabic_vs_western():
    assert digit_script_preserved("١٢٣", "123") == Ratio(0, 3)


def test_digit_script_same_script_wrong_value():
    assert digit_script_preserved("١٢٣", "١٢٤") == Ratio(3, 3)


def test_digit_script_deleted_counts_in_denominator():
    assert digit_script_preserved("١٢٣", "١") == Ratio(1, 3)


def test_digit_script_extended_and_none():
    assert digit_script_preserved("\u06F4\u06F5", "\u06F4\u06F5") == Ratio(2, 2)
    r = digit_script_preserved("abc", "abc")
    assert r == Ratio(0, 0) and r.rate is None


def test_iou_basics():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert iou((0, 0, 0, 5), (0, 0, 0, 5)) == 0.0


def test_one_big_box_matches_only_one_line():
    gt = [(0, 0, 10, 5), (0, 5, 10, 10)]
    pred = [(0, 0, 10, 10)]
    assert iou(gt[0], pred[0]) == 0.5
    assert match_lines(gt, pred) == ((0, 0),)


def test_matching_maximum_cardinality_beats_greedy():
    gt = [(0, 0, 10, 10), (0, 10, 10, 20)]
    pred = [(0, 4, 10, 14), (0, 0, 10, 3)]
    thr = 0.2
    assert iou(gt[0], pred[0]) == pytest.approx(60 / 140)
    assert iou(gt[1], pred[0]) == pytest.approx(40 / 160)
    assert iou(gt[0], pred[1]) == pytest.approx(0.3)
    assert iou(gt[1], pred[1]) == 0.0
    # greedy highest-IoU-first would take (gt0, pred0) and leave gt1 unmatched
    assert match_lines(gt, pred, threshold=thr) == ((0, 1), (1, 0))
    assert line_recall(gt, pred, threshold=thr) == Ratio(2, 2)


def test_iou_exactly_half_counts():
    assert match_lines([(0, 0, 2, 1)], [(0, 0, 1, 1)]) == ((0, 0),)


def test_reading_order():
    gt = [(0, 0, 1, 1), (0, 2, 1, 3), (0, 4, 1, 5)]
    assert reading_order_errors(gt, gt) == Ratio(0, 2)
    assert reading_order_errors(gt, [gt[0], gt[2], gt[1]]) == Ratio(1, 2)
    assert reading_order_errors(gt, [gt[0]]).rate is None
    assert reading_order_errors([], []) == Ratio(0, 0)


def test_missed_error_rate():
    lines = [
        ("abcdefghij", "abcdefghix", "auto_accepted"),
        ("abcdefghij", "abcdefghix", "needs_review"),
        ("abcdefghij", "abcdefghij", "auto_accepted"),
    ]
    assert missed_error_rate(lines) == Ratio(1, 2)


def test_missed_error_rate_boundary_and_none():
    exact = [("a" * 50, "a" * 49 + "b", "auto_accepted")]  # exactly 2 %
    assert missed_error_rate(exact) == Ratio(0, 0)
    assert missed_error_rate(exact).rate is None


def test_missed_error_rate_empty_ref():
    with pytest.raises(ValueError):
        missed_error_rate([("", "a", "auto_accepted")])


def test_entities_insertion_right_after_span_is_outside():
    after = match_entities("pay 100 now", "pay 100x now", [(4, 7)])
    assert after[0].hypothesis == "100"
    assert after[0].raw_exact


# box mapping
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
SHIFT = ((1.0, 0.0, 10.0), (0.0, 1.0, 20.0), (0.0, 0.0, 1.0))
TURN = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # (x, y) -> (-y, x)
SHEAR = ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # x' = x + y
PERSPECTIVE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.0, 1.0))  # w = 0.1 x + 1


def test_map_box_identity():
    assert map_box(IDENT, (1.5, 2.5, 3.5, 9.0)) == (1.5, 2.5, 3.5, 9.0)


def test_map_box_translation():
    assert map_box(SHIFT, (0.0, 0.0, 4.0, 2.0)) == (10.0, 20.0, 14.0, 22.0)


def test_map_box_quarter_turn_reorders_the_corners():
    assert map_box(TURN, (1.0, 2.0, 3.0, 5.0)) == (-5.0, 1.0, -2.0, 3.0)


def test_map_box_takes_the_bounds_of_all_four_corners():
    assert map_box(SHEAR, (0.0, 0.0, 2.0, 1.0)) == (0.0, 0.0, 3.0, 1.0)


def test_map_box_perspective_uses_all_four_corners():
    # Corners map to (0,0), (5,0), (5,5), (0,10); two corners alone give height 5.
    assert map_box(PERSPECTIVE, (0.0, 0.0, 10.0, 10.0)) == (0.0, 0.0, 5.0, 10.0)


@pytest.mark.parametrize("w_row", [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
def test_map_box_rejects_corner_behind_the_image_plane(w_row):
    matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), w_row)
    with pytest.raises(ValueError, match="box maps outside the image plane"):
        map_box(matrix, (1.0, 1.0, 2.0, 2.0))


def test_compose_applies_the_right_matrix_first():
    point = (1.0, 0.0, 1.0, 0.0)
    assert map_box(compose(SHIFT, TURN), point) == (10.0, 21.0, 10.0, 21.0)
    assert map_box(compose(TURN, SHIFT), point) == (-20.0, 11.0, -20.0, 11.0)


def test_compose_with_identity_is_unchanged():
    assert compose(SHIFT, IDENT) == SHIFT
    assert compose(IDENT, SHIFT) == SHIFT


def test_gt_line_boxes_maps_through_transform_after_homography():
    boxes = [(1.0, 2.0, 3.0, 5.0), (0.0, 0.0, 1.0, 1.0)]
    assert gt_line_boxes(boxes, TURN, SHIFT) == (
        (5.0, 21.0, 8.0, 23.0),
        (9.0, 20.0, 10.0, 21.0),
    )
    assert gt_line_boxes([], TURN, SHIFT) == ()
