"""Tests for scripts/accuracy_stats.py: document resampling, tau choice and the cell verdict."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import accuracy_stats as st
from accuracy_eval import GtOutcome, OutputLine, PageOutcome


def _ol(conf=0.9, *, flags=False, matched=True, good=None, read=True):
    return OutputLine(conf, flags, False, "needs_review", matched, read, good)


def _bad(conf):
    return GtOutcome("matched", True, _ol(conf, good=False), (), 0.5)


def _good(conf):
    return GtOutcome("matched", False, _ol(conf, good=True), (), 0.0)


def _unit(name, gts, extra_lines=()):
    lines = tuple(g.line for g in gts if g.line is not None and g.line.ocr_read)
    return st.Unit(name, (PageOutcome(tuple(gts), lines + tuple(extra_lines)),))


PURE = GtOutcome("pure_deletion", True, None, (), None)
CAUGHT = GtOutcome("caught_unread", True, None, (), None)
NOT_READ = GtOutcome("not_read", False, _ol(0.9, read=False), (), None)


def test_unit_counts_exclude_not_read():
    u = _unit("a", [_bad(0.9), _good(0.9), NOT_READ, PURE, CAUGHT])
    assert st.unit_counts(u, 0.5) == (3, 2)
    assert st.unit_counts(u, 0.95) == (3, 1)


def test_resample_weights_integer_rows_sum_to_n():
    w = st.resample_weights(7, 50)
    assert w.shape == (50, 7) and np.issubdtype(w.dtype, np.integer)
    assert (w.sum(axis=1) == 7).all()
    assert (st.resample_weights(7, 50) == w).all()


def test_interval_resamples_documents_not_lines():
    big = _unit("big", [_bad(0.9)] * 50)
    small = [_unit(f"s{i}", [_bad(0.1)]) for i in range(49)]
    iv = st.missed_interval([big, *small], 0.5)
    assert iv.point == pytest.approx(50 / 99)
    assert iv.high - iv.low > 0.3


def test_interval_empty_and_no_bad():
    assert st.missed_interval([], 0.5) == st.Interval(None, None, None, None)
    iv = st.missed_interval([_unit("a", [_good(0.9)])], 0.5)
    assert iv.point is None and iv.bound_95 == 1.0


def test_resamples_without_bad_line_count_one():
    units = [_unit("a", [_bad(0.9)] * 5)] + [_unit(f"g{i}", [_good(0.9)]) for i in range(9)]
    iv = st.missed_interval(units, 0.5)
    assert iv.point == 1.0 and iv.high == 1.0 and iv.low == 1.0


def _cell(n_docs, bad_per_doc, conf_for):
    return [
        _unit(f"d{i}", [_bad(conf_for(i, k)) for k in range(bad_per_doc)]) for i in range(n_docs)
    ]


def _linear(units):
    for tau in st.TAU_GRID:
        if st.missed_interval(units, tau).bound_95 <= st.MAX_MISSED:
            return tau
    return None


def test_choose_tau_minimal_and_matches_linear_scan():
    units = _cell(40, 5, lambda i, k: (i * 5 + k) / 200)
    choice = st.choose_cell_tau(units)
    assert choice.status == "calibrated" and choice.reason is None
    assert choice.tau == _linear(units)
    assert choice.tau > 0.5
    assert choice.interval.bound_95 <= st.MAX_MISSED
    assert st.missed_interval(units, choice.tau - 0.001).bound_95 > st.MAX_MISSED
    assert choice.bad_lines == 200


def test_bound_95_is_the_0_95_quantile_not_the_0_975_one():
    # 100 documents with two bad lines each, confidences 0.00 .. 0.99: at tau 0.94 six percent
    # are missed, the 0.95 quantile of the resampled rate is 0.10 and the 0.975 one is 0.11.
    units = [_unit(f"d{i}", [_bad(i / 100)] * 2) for i in range(100)]
    iv = st.missed_interval(units, 0.94)
    assert iv.bound_95 == pytest.approx(0.10) and iv.high == pytest.approx(0.11)
    assert iv.bound_95 <= st.MAX_MISSED < iv.high
    choice = st.choose_cell_tau(units)
    assert choice.tau == pytest.approx(0.931)  # a 0.975 bound would need tau 0.95 or more
    assert st.missed_interval(units, 0.95).high <= st.MAX_MISSED


def test_choose_tau_too_few_bad_lines():
    c = st.choose_cell_tau(_cell(5, 10, lambda i, k: 0.5))
    assert (c.status, c.tau, c.reason) == ("not measured", None, "not measured (50 bad lines)")


def test_choose_tau_miss_floor():
    units = _cell(40, 5, lambda i, k: 0.5) + [_unit(f"p{i}", [PURE] * 10) for i in range(5)]
    c = st.choose_cell_tau(units)
    assert c.status == "not measured" and c.tau is None
    assert c.reason == "not measured (miss floor 20.0 %)"
    assert c.interval.point == pytest.approx(0.2)


def test_choose_tau_no_units_not_measured():
    assert st.choose_cell_tau([]).status == "not measured"


def _choice(tau):
    iv = st.Interval(None, None, None, None)
    return st.TauChoice(tau, "calibrated" if tau is not None else "not measured", None, iv, 0)


def test_setup_taus_takes_highest_calibrated():
    got = st.setup_taus(
        [("a", _choice(0.3)), ("a", _choice(0.7)), ("a", _choice(None)), ("b", _choice(None))]
    )
    assert got == {"a": 0.7}


def test_cell_summary_hand_computed():
    grouping = GtOutcome("grouping", True, None, (_ol(0.9),), None)
    u1 = _unit(
        "a",
        [_bad(0.9), _good(0.9), _good(0.3), PURE, CAUGHT, NOT_READ],
        [_ol(0.9, matched=False), _ol(0.2, matched=False), _ol(None, matched=False)],
    )
    u2 = _unit("b", [_good(0.9), grouping])
    s = st.cell_summary([u1, u2], 0.5)
    assert s == {
        "pages": 2,
        "orientation_uncertain_pages": 0,
        "orientation_uncertain_share": 0.0,
        "orientation": {g: NO_PAGES for g in st.ORIENTATION_GROUPS},
        "documents": 2,
        "documents_with_bad_lines": 2,
        "ocr_read_lines": 7,
        "good_lines": 3,
        "bad_lines": 4,
        "missed": 3,
        "fail_closed_lines": 1,
        "fail_closed_share": 1 / 7,
        "flag_rate_all": 3 / 7,
        "flag_rate_good": 1 / 3,
        "low_confidence_share": 1.0,
        "pure_deletions": 1,
        "grouping": 1,
        "caught_unread": 1,
        "pure_deletion_share": 1 / 3,
        "auto_accepted_unmatched": 1,
        "not_read_lines": 1,
        "split_word_lines": 0,
        "split_word_good_flag_rate": None,
        "split_word_missed": 0,
    }


def test_cell_summary_rates_none_without_denominator():
    s = st.cell_summary([], 0.5)
    for k in ("fail_closed_share", "flag_rate_all", "flag_rate_good", "low_confidence_share",
              "pure_deletion_share"):
        assert s[k] is None
    assert s["documents"] == 0 and s["missed"] == 0


def test_low_confidence_share_excludes_other_flag_above_tau():
    lines = [_ol(0.9, flags=True, matched=False), _ol(0.1, matched=False)]
    assert st.cell_summary([_unit("a", [], lines)], 0.5)["low_confidence_share"] == 0.5


GOOD = {
    "documents": 40, "documents_with_bad_lines": 35, "ocr_read_lines": 1000,
    "good_lines": 800, "bad_lines": 150, "missed": 10, "fail_closed_share": 0.01,
}
IV = st.Interval(0.07, 0.03, 0.12, 0.11)
NM = "not measured"


def _verdict(summary=GOOD, interval=IV, key_ok=True, tau_present=True):
    return st.grade_verdict(summary, interval, key_ok=key_ok, tau_present=tau_present)


def test_grade_verdict_key_and_tau():
    assert _verdict(key_ok=False, tau_present=False) == (NM, "not measured (setup key mismatch)")
    assert _verdict(tau_present=False) == (NM, "not measured (no frozen tau)")


def test_grade_verdict_fail_closed_and_minimums():
    assert _verdict({**GOOD, "fail_closed_share": 0.05}) == (
        NM, "not measured (fail-closed lines 5.0% > 2 %)")
    assert _verdict({**GOOD, "bad_lines": 99}) == (NM, "not measured (99 bad lines from 35 documents)")
    assert _verdict({**GOOD, "documents_with_bad_lines": 29}) == (
        NM, "not measured (150 bad lines from 29 documents)")
    assert _verdict({**GOOD, "bad_lines": 100, "documents_with_bad_lines": 30})[0] == "pass"


def test_grade_verdict_pass_fail_edge():
    assert _verdict() == ("pass", None)
    assert _verdict(interval=st.Interval(0.1, 0, 0.2, 0.2)) == ("pass", None)
    assert _verdict(interval=st.Interval(0.101, 0, 0.2, 0.2)) == ("fail", None)


def test_grade_verdict_order():
    s = {**GOOD, "fail_closed_share": 0.03, "bad_lines": 1}
    assert _verdict(s)[1].startswith("not measured (fail-closed")
    assert _verdict(s, key_ok=False)[1] == "not measured (setup key mismatch)"
    assert _verdict({**GOOD, "fail_closed_share": 0.02})[0] == "pass"


def _sw(conf, *, good=None, matched=True):
    return OutputLine(conf, False, False, "needs_review", matched, True, good, True)


def _split_good(c):
    return GtOutcome("matched", False, _sw(c, good=True), (), 0.0)


def _split_bad(c):
    return GtOutcome("matched", True, _sw(c, good=False), (), 0.5)


def test_cell_summary_split_word_keys():
    cover = _sw(0.9, matched=False)
    grouping_split = GtOutcome("grouping", True, None, (_ol(0.9, matched=False), cover), None)
    grouping_plain = GtOutcome("grouping", True, None, (_ol(0.9, matched=False),), None)
    gts = [_split_good(0.9), _split_good(0.3), _split_good(0.2), _split_bad(0.9),
           _split_bad(0.3), _bad(0.9), _good(0.9)]
    s = st.cell_summary([_unit("a", gts), _unit("b", [grouping_split, grouping_plain], [cover])], 0.5)
    assert s["split_word_lines"] == 6
    assert s["split_word_good_flag_rate"] == 2 / 3
    assert s["split_word_missed"] == 2  # the 0.9 bad split line and the grouping with a split cover


def test_cell_summary_split_word_rate_none_without_good_split_lines():
    s = st.cell_summary([_unit("a", [_good(0.9), _bad(0.9)])], 0.5)
    assert s["split_word_lines"] == 0 and s["split_word_good_flag_rate"] is None
    assert s["split_word_missed"] == 0


def _docs(n, bad_each, start=0):
    return [_unit(f"d{start + i}", [_bad(0.5)] * bad_each) for i in range(n)]


def test_choose_tau_needs_both_dev_minimums():
    few_docs = st.choose_cell_tau(_docs(39, 7))  # 273 bad lines from 39 documents
    assert few_docs.status == "not measured" and few_docs.reason == "not measured (273 bad lines)"
    few_lines = st.choose_cell_tau(_docs(60, 3))  # 180 bad lines
    assert few_lines.reason == "not measured (180 bad lines)"
    assert st.DEV_MIN_DOCS == 40 and st.DEV_CAP_DOCS == 300


NO_PAGES = {
    "pages": 0, "flagged": 0, "flagged_share": None, "wrong": 0, "wrong_180": 0,
    "wrong_90_270": 0, "wrong_flagged": 0, "wrong_unflagged": 0, "largest_wrong_margin": None,
}


def _page_with(source, uncertain, wrong, branch="pair", margin=None, failed=False, off=None):
    off = (90.0 if wrong else 0.0) if off is None else off
    return PageOutcome((), (), wrong, source, uncertain, branch, margin, failed, off)


def _groups(*pages):
    return st.cell_summary([st.Unit("a", pages)], 0.5)["orientation"]


def test_cell_summary_orientation_counts():
    pages = (
        _page_with("osd", True, True),
        _page_with("osd", False, True),
        _page_with("osd", False, False),
        _page_with("vote", True, False),
        _page_with("vote", True, True),
        _page_with("vote", False, True),
        _page_with(None, True, True, None),  # nothing recorded: in no group
    )
    s = st.cell_summary([st.Unit("a", pages[:4]), st.Unit("b", pages[4:])], 0.5)
    assert s["pages"] == 7
    assert s["orientation_uncertain_pages"] == 4
    assert s["orientation_uncertain_share"] == pytest.approx(4 / 7)
    groups = s["orientation"]
    assert set(groups) == {"osd", "vote-pair", "vote-four", "vote-after-osd-failure", "none"}
    assert (groups["osd"]["pages"], groups["osd"]["wrong_flagged"]) == (3, 1)
    assert (groups["osd"]["wrong_unflagged"], groups["osd"]["flagged"]) == (1, 1)
    assert groups["vote-pair"]["pages"] == 3 and groups["vote-pair"]["wrong"] == 2
    assert groups["vote-pair"]["flagged_share"] == pytest.approx(2 / 3)
    assert groups["vote-four"] == groups["none"] == NO_PAGES


def test_cell_summary_orientation_empty():
    s = st.cell_summary([], 0.5)
    assert (s["pages"], s["orientation_uncertain_pages"], s["orientation_uncertain_share"]) == (
        0, 0, None)
    assert s["orientation"] == {g: NO_PAGES for g in st.ORIENTATION_GROUPS}


def test_a_page_whose_osd_failed_is_grouped_apart_whatever_its_branch():
    groups = _groups(
        _page_with("vote", True, True, "pair", 0.1, failed=True),
        _page_with("vote", True, False, "four", 0.3, failed=True),
        _page_with("vote", False, False, "pair", 0.5),
        _page_with("vote", False, False, "four", 0.5),
    )
    assert groups["vote-after-osd-failure"]["pages"] == 2
    assert groups["vote-after-osd-failure"]["wrong_flagged"] == 1
    assert groups["vote-pair"]["pages"] == groups["vote-four"]["pages"] == 1


def test_a_page_with_no_word_is_in_the_none_group_with_no_margin():
    groups = _groups(_page_with("vote", False, False, "none"), _page_with("vote", True, True, "none", 0.9))
    assert groups["none"]["pages"] == 2 and groups["none"]["largest_wrong_margin"] is None
    assert groups["vote-pair"]["pages"] == groups["vote-four"]["pages"] == 0


@pytest.mark.parametrize(
    ("off", "wrong_180", "wrong_90_270"),
    [(46.0, 0, 1), (-134.0, 0, 1), (135.0, 0, 1), (136.0, 1, 0), (-136.0, 1, 0), (180.0, 1, 0)],
)
def test_a_wrong_page_is_upside_down_beyond_135_degrees_else_sideways(off, wrong_180, wrong_90_270):
    group = _groups(_page_with("osd", True, True, off=off))["osd"]
    assert (group["wrong"], group["wrong_180"], group["wrong_90_270"]) == (
        1, wrong_180, wrong_90_270)


def test_the_largest_wrong_margin_is_over_wrong_vote_pages_only():
    groups = _groups(
        _page_with("vote", True, True, "four", 0.25),
        _page_with("vote", True, True, "four", 0.375),
        _page_with("vote", False, False, "four", 0.5),  # right page: its margin does not count
        _page_with("vote", True, True, "pair", 0.0625),
        _page_with("vote", False, False, "pair", 0.5),
        _page_with("osd", True, True, "pair", None),
    )
    assert groups["vote-four"]["largest_wrong_margin"] == 0.375
    assert groups["vote-pair"]["largest_wrong_margin"] == 0.0625
    assert groups["osd"]["largest_wrong_margin"] is None
    assert groups["vote-after-osd-failure"]["largest_wrong_margin"] is None
