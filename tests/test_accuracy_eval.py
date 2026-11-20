"""Tests for scripts/accuracy_eval.py: outcome of each ground-truth line of a read page."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import accuracy_eval as ae

from folio_ocr.document import IDENTITY, Line, Page

REF = "a" * 50  # 1 edit = exactly 2 %


def _line(box, text=REF, *, conf=0.9, reader="onnxtr", flags=(), status="auto_accepted"):
    return Line("p1-l1", box, text, "ltr", "latin", reader, conf, "reader", flags, status, text)


def _page(lines=(), unread=(), transform=IDENTITY):
    return Page(1, 1000, 1000, None, 0.0, transform, "ocr", "ltr", "read", (), tuple(unread),
                tuple(lines))


def _gt(boxes, texts=None, homography=None):
    texts = texts or [REF] * len(boxes)
    return {
        "file": "x.png",
        "homography": homography or [list(r) for r in IDENTITY],
        "lines": [{"clean_box": list(b), "text": t} for b, t in zip(boxes, texts, strict=True)],
    }


B = (0, 0, 100, 10)


def test_exactly_two_percent_is_good():
    out = ae.page_outcome(_gt([B]), _page([_line(B, "a" * 49 + "b")]))
    g = out.gt[0]
    assert (g.kind, g.bad, g.cer) == ("matched", False, pytest.approx(0.02))
    assert g.line.good is True


def test_above_two_percent_is_bad():
    out = ae.page_outcome(_gt([B]), _page([_line(B, "a" * 48 + "bb")]))
    assert out.gt[0].bad and out.gt[0].line.good is False


def test_not_read_is_excluded():
    out = ae.page_outcome(_gt([B]), _page([_line(B, "zzz", reader="none")]))
    g = out.gt[0]
    assert (g.kind, g.bad) == ("not_read", False)
    assert not ae.missed_at(g, 0.0)
    assert out.lines == ()


def test_empty_reference_raises_naming_file():
    with pytest.raises(ValueError, match="x.png"):
        ae.page_outcome(_gt([B], [" "]), _page([_line(B)]))


def test_unread_cover_60_percent_is_caught():
    out = ae.page_outcome(_gt([B]), _page(unread=[(0, 0, 60, 10)]))
    g = out.gt[0]
    assert (g.kind, g.bad) == ("caught_unread", True)
    assert not ae.missed_at(g, 1.0)


def test_unread_cover_40_percent_is_pure_deletion():
    g = ae.page_outcome(_gt([B]), _page(unread=[(0, 0, 40, 10)])).gt[0]
    assert g.kind == "pure_deletion"
    assert ae.missed_at(g, 0.0) and ae.missed_at(g, 1.0)


def test_covered_share_counts_overlap_once():
    boxes = [(0, 0, 30, 10), (20, 0, 40, 10)]
    assert ae.covered_share(B, boxes) == pytest.approx(0.4)
    assert ae.covered_share((5, 5, 5, 9), boxes) == 0.0
    assert ae.covered_share(B, []) == 0.0


def test_unread_overlap_not_double_counted():
    unread = [(0, 0, 30, 10), (10, 0, 40, 10)]  # union 40 %, sum 60 %
    assert ae.page_outcome(_gt([B]), _page(unread=unread)).gt[0].kind == "pure_deletion"


def _grouping(covers):
    # a tall GT box covered by two short lines; IoU with each < 0.5
    gt_box = (0, 0, 100, 20)
    return ae.page_outcome(_gt([gt_box]), _page(covers)).gt[0]


def test_grouping_missed_only_when_all_covers_auto():
    top, bottom = (0, 0, 100, 8), (0, 12, 100, 20)
    both = _grouping([_line(top), _line(bottom)])
    assert both.kind == "grouping" and len(both.covers) == 2
    assert ae.missed_at(both, 0.5)
    one_flagged = _grouping([_line(top), _line(bottom, flags=("x",))])
    assert not ae.missed_at(one_flagged, 0.5)
    assert not ae.missed_at(_grouping([_line(top), _line(bottom, conf=0.1)]), 0.5)


def test_matched_missed_needs_bad_and_auto():
    bad = ae.page_outcome(_gt([B]), _page([_line(B, "b" * 50)])).gt[0]
    assert ae.missed_at(bad, 0.9) and not ae.missed_at(bad, 0.95)
    good = ae.page_outcome(_gt([B]), _page([_line(B)])).gt[0]
    assert not ae.missed_at(good, 0.0)


def test_boxes_mapped_through_homography_and_transform():
    shift = ((1.0, 0, 50), (0, 1.0, 0), (0, 0, 1.0))
    scale = ((2.0, 0, 0), (0, 2.0, 0), (0, 0, 1.0))
    # clean (0,0,100,10) -> homography +50 x -> (50,0,150,10) -> x2 -> (100,0,300,20)
    page = _page([_line((100, 0, 300, 20))], transform=scale)
    out = ae.page_outcome(_gt([B], homography=[list(r) for r in shift]), page)
    assert out.gt[0].kind == "matched"
    assert ae.page_outcome(_gt([B]), page).gt[0].kind != "matched"


def test_auto_at_edges():
    ln = ae.page_outcome(_gt([B]), _page([_line(B, conf=0.8)])).lines[0]
    assert ae.auto_at(ln, 0.8) and not ae.auto_at(ln, 0.8000001)
    none = ae.page_outcome(_gt([B]), _page([_line(B, conf=None)])).lines[0]
    assert not ae.auto_at(none, 0.0)
    flagged = ae.page_outcome(_gt([B]), _page([_line(B, flags=("x",))])).lines[0]
    assert flagged.other_flags and not ae.auto_at(flagged, 0.0)
    low = ae.page_outcome(_gt([B]), _page([_line(B, flags=("low_confidence",))])).lines[0]
    assert low.low_confidence and not low.other_flags and ae.auto_at(low, 0.5)


def test_status_agrees():
    ln = ae.page_outcome(_gt([B]), _page([_line(B, conf=0.8)])).lines[0]
    assert ae.status_agrees(ln, 0.8) and not ae.status_agrees(ln, 0.9)


def test_unmatched_output_line_listed_and_unmatched_flag():
    out = ae.page_outcome(_gt([B]), _page([_line((500, 500, 600, 510))]))
    assert len(out.lines) == 1 and not out.lines[0].matched and out.lines[0].good is None


def _split_flag(split_boxes):
    return ae.page_outcome(_gt([B]), _page([_line(B)]), split_boxes).lines[0].split_word


def test_split_word_centre_inside_grown_line_box():
    assert _split_flag([(40, 4, 50, 8)])
    assert _split_flag([(101, 2, 103, 4)])  # centre (102, 3): on the 2 px growth edge
    assert _split_flag([(50, -2, 60, -2)])  # centre y = -2: on the growth edge


def test_split_word_centre_outside_grown_line_box():
    assert not _split_flag([(101, 2, 105, 4)])  # centre x = 103
    assert not _split_flag([(40, 13, 50, 15)])  # centre y = 14
    assert not _split_flag([(-50, 0, 40, 10)])  # overlaps the line but its centre is outside


def test_split_word_default_is_false():
    assert not ae.page_outcome(_gt([B]), _page([_line(B)])).lines[0].split_word


def _rot(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _turn(transform, homography=IDENTITY):
    return ae.page_outcome(
        _gt([B], homography=[list(r) for r in homography]), _page(transform=transform)
    ).turn_wrong


@pytest.mark.parametrize("deg", [0, 44, -44])
def test_turn_within_45_degrees_is_right(deg):
    assert _turn(_rot(deg)) is False


@pytest.mark.parametrize("deg", [46, -46, 180, -90, 90])
def test_turn_beyond_45_degrees_is_wrong(deg):
    assert _turn(_rot(deg)) is True


def test_turn_uses_page_transform_and_homography_together():
    assert _turn(_rot(-90), homography=_rot(90)) is False  # net 0 degrees
    assert _turn(_rot(90), homography=_rot(90)) is True  # net 180 degrees
    assert _turn(_rot(0), homography=_rot(20)) is False


def test_orientation_fields_default_and_come_from_flags():
    plain = ae.page_outcome(_gt([B]), _page([_line(B)]))
    assert (plain.orientation_source, plain.orientation_uncertain) == (None, False)
    page = replace(_page([_line(B)]), flags=("orientation_uncertain",))
    out = ae.page_outcome(_gt([B]), page, orientation_source="osd")
    assert (out.orientation_source, out.orientation_uncertain) == ("osd", True)


@pytest.mark.parametrize("deg", [46, -46, 134, -134, 136, -136, 180])
def test_turn_off_is_the_signed_rotation_the_classification_reads(deg):
    out = ae.page_outcome(_gt([B]), _page(transform=_rot(deg)))
    assert out.turn_wrong is True
    assert out.turn_off == pytest.approx(-180.0 if deg == 180 else deg)


def test_turn_off_of_an_upright_page_is_zero():
    assert ae.page_outcome(_gt([B]), _page(transform=_rot(0))).turn_off == pytest.approx(0.0)


def test_orientation_record_comes_through_page_outcome():
    out = ae.page_outcome(
        _gt([B]), _page([_line(B)]),
        orientation_source="vote", orientation_branch="four", orientation_margin=0.25,
        orientation_failed=True,
    )
    assert (out.orientation_source, out.orientation_branch) == ("vote", "four")
    assert (out.orientation_margin, out.orientation_failed) == (0.25, True)
    plain = ae.page_outcome(_gt([B]), _page([_line(B)]))
    assert (plain.orientation_branch, plain.orientation_margin, plain.orientation_failed) == (
        None, None, False)


def test_unread_cover_of_exactly_half_is_caught():
    g = ae.page_outcome(_gt([B]), _page(unread=[(0, 0, 50, 10)])).gt[0]
    assert g.kind == "caught_unread"


def test_line_cover_of_exactly_half_is_grouping():
    lines = [_line((0, 0, 25, 10)), _line((25, 0, 50, 10))]
    g = ae.page_outcome(_gt([B]), _page(lines)).gt[0]
    assert g.kind == "grouping" and len(g.covers) == 2


def test_line_cover_below_half_is_pure_deletion():
    lines = [_line((0, 0, 25, 10)), _line((25, 0, 49, 10))]
    g = ae.page_outcome(_gt([B]), _page(lines)).gt[0]
    assert g.kind == "pure_deletion"


def test_unread_wins_over_line_cover_when_both_reach_half():
    tall = (0, 0, 100, 20)
    lines = [_line((0, 0, 100, 8)), _line((0, 12, 100, 20))]  # 80 % line cover
    g = ae.page_outcome(_gt([tall]), _page(lines, unread=[(0, 0, 50, 20)])).gt[0]
    assert g.kind == "caught_unread"
