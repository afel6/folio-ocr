"""Document resampling, confidence-threshold choice and the graded cell verdict."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from accuracy_eval import auto_at, missed_at

RESAMPLES = 2000
RNG_SEED = 20261119
MAX_MISSED = 0.10
DEV_MIN_BAD = 200
DEV_MIN_DOCS = 40
DEV_CAP_DOCS = 300
TEST_MIN_BAD = 100
TEST_MIN_DOCS = 30
VOID_SHARE = 0.02
TAU_GRID = tuple(i / 1000 for i in range(1001))


@dataclass(frozen=True)
class Unit:
    """One resampling unit: a generated document with all its images."""

    name: str
    pages: tuple


@dataclass(frozen=True)
class Interval:
    point: float | None
    low: float | None
    high: float | None
    bound_95: float | None


@dataclass(frozen=True)
class TauChoice:
    tau: float | None
    status: Literal["calibrated", "not measured"]
    reason: str | None
    interval: Interval
    bad_lines: int


_EMPTY = Interval(None, None, None, None)


def _outcomes(unit: Unit):
    return (g for page in unit.pages for g in page.gt if g.kind != "not_read")


def unit_counts(unit: Unit, tau: float) -> tuple[int, int]:
    """(bad, missed) ground-truth lines of one unit at `tau`; unread-by-reader lines excluded."""
    bad = missed = 0
    for g in _outcomes(unit):
        bad += g.bad
        missed += g.bad and missed_at(g, tau)
    return bad, missed


def resample_weights(n: int, resamples: int = RESAMPLES, seed: int = RNG_SEED) -> np.ndarray:
    """Multinomial draw counts of `n` units, one row per resample."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=resamples)


def missed_interval(units: Sequence[Unit], tau: float) -> Interval:
    """Missed-error rate with its document-resampled interval and one-sided 95 % bound."""
    if not units:
        return _EMPTY
    counts = np.array([unit_counts(u, tau) for u in units], dtype=np.int64)
    bad, missed = counts[:, 0], counts[:, 1]
    weights = resample_weights(len(units))
    bad_sum = weights @ bad
    rates = np.where(bad_sum > 0, (weights @ missed) / np.maximum(bad_sum, 1), 1.0)
    point = float(missed.sum() / bad.sum()) if bad.sum() else None
    low, high, bound = (float(q) for q in np.quantile(rates, [0.025, 0.975, 0.95]))
    return Interval(point, low, high, bound)


def _total_bad(units: Sequence[Unit]) -> int:
    return sum(unit_counts(u, 0.0)[0] for u in units)


def choose_cell_tau(units: Sequence[Unit]) -> TauChoice:
    """Smallest grid threshold whose one-sided 95 % bound on the missed rate is within limit."""
    bad = _total_bad(units)
    bad_docs = sum(unit_counts(u, 0.0)[0] > 0 for u in units)
    if bad < DEV_MIN_BAD or bad_docs < DEV_MIN_DOCS:
        return TauChoice(None, "not measured", f"not measured ({bad} bad lines)", _EMPTY, bad)
    top = missed_interval(units, TAU_GRID[-1])
    if top.bound_95 > MAX_MISSED:
        reason = f"not measured (miss floor {top.point * 100:.1f} %)"
        return TauChoice(None, "not measured", reason, top, bad)
    # Missed counts never rise with tau, so the first passing grid index can be bisected.
    lo, hi = 0, len(TAU_GRID) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if missed_interval(units, TAU_GRID[mid]).bound_95 <= MAX_MISSED:
            hi = mid
        else:
            lo = mid + 1
    return TauChoice(TAU_GRID[lo], "calibrated", None, missed_interval(units, TAU_GRID[lo]), bad)


def setup_taus(cells: Sequence[tuple[str, TauChoice]]) -> dict[str, float]:
    """Highest calibrated threshold per setup key."""
    out: dict[str, float] = {}
    for key, choice in cells:
        if choice.tau is not None and choice.status == "calibrated":
            out[key] = max(out.get(key, choice.tau), choice.tau)
    return out


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _split_missed(units: Sequence[Unit], tau: float) -> int:
    """Missed bad lines whose matched line, or any grouping cover, holds a split word."""
    count = 0
    for u in units:
        for g in _outcomes(u):
            if g.bad and missed_at(g, tau):
                count += bool(g.line and g.line.split_word) or any(c.split_word for c in g.covers)
    return count


ORIENTATION_GROUPS = ("osd", "vote-pair", "vote-four", "vote-after-osd-failure", "none")
TURN_180 = 135.0  # degrees: a wrong page turned further than this is upside down


def _orientation_group(page) -> str | None:
    """The one group a page belongs to; None when its orientation was not recorded."""
    if page.orientation_failed:
        return "vote-after-osd-failure"
    if page.orientation_branch == "none":
        return "none"
    if page.orientation_source == "osd":
        return "osd"
    if page.orientation_source == "vote" and page.orientation_branch in ("pair", "four"):
        return f"vote-{page.orientation_branch}"
    return None


def _group_summary(group: str, mine: Sequence) -> dict:
    wrong = [p for p in mine if p.turn_wrong]
    margins = [p.orientation_margin for p in wrong if p.orientation_margin is not None]
    flagged = sum(p.orientation_uncertain for p in mine)
    return {
        "pages": len(mine),
        "flagged": flagged,
        "flagged_share": rate(flagged, len(mine)),
        "wrong": len(wrong),
        "wrong_180": sum(abs(p.turn_off) > TURN_180 for p in wrong),
        "wrong_90_270": sum(abs(p.turn_off) <= TURN_180 for p in wrong),
        "wrong_flagged": sum(p.orientation_uncertain for p in wrong),
        "wrong_unflagged": sum(not p.orientation_uncertain for p in wrong),
        "largest_wrong_margin": max(margins) if group.startswith("vote") and margins else None,
    }


def _orientation_summary(pages: Sequence) -> dict:
    """Page counts by orientation uncertainty, and wrong turns by how the way up was decided."""
    uncertain = sum(p.orientation_uncertain for p in pages)
    groups = {
        g: _group_summary(g, [p for p in pages if _orientation_group(p) == g])
        for g in ORIENTATION_GROUPS
    }
    return {
        "pages": len(pages),
        "orientation_uncertain_pages": uncertain,
        "orientation_uncertain_share": rate(uncertain, len(pages)),
        "orientation": groups,
    }


def cell_summary(units: Sequence[Unit], tau: float) -> dict[str, Any]:
    """Counts and rates of one cell at `tau`."""
    gts = [g for u in units for p in u.pages for g in p.gt]
    lines = [ln for u in units for p in u.pages for ln in p.lines]
    kinds = [g.kind for g in gts]
    counts = [unit_counts(u, tau) for u in units]
    missed = sum(m for _, m in counts)
    good = [ln for ln in lines if ln.good]
    flagged = [ln for ln in lines if not auto_at(ln, tau)]
    fail_closed = sum(ln.confidence is None for ln in lines)
    pure = kinds.count("pure_deletion")
    split_good = [ln for ln in good if ln.split_word]
    return {
        **_orientation_summary([p for u in units for p in u.pages]),
        "documents": len(units),
        "documents_with_bad_lines": sum(b > 0 for b, _ in counts),
        "ocr_read_lines": len(lines),
        "good_lines": len(good),
        "bad_lines": sum(b for b, _ in counts),
        "missed": missed,
        "fail_closed_lines": fail_closed,
        "fail_closed_share": rate(fail_closed, len(lines)),
        "flag_rate_all": rate(len(flagged), len(lines)),
        "flag_rate_good": rate(sum(not auto_at(ln, tau) for ln in good), len(good)),
        "low_confidence_share": rate(
            sum(ln.confidence is None or ln.confidence < tau for ln in flagged), len(flagged)
        ),
        "pure_deletions": pure,
        "grouping": kinds.count("grouping"),
        "caught_unread": kinds.count("caught_unread"),
        "pure_deletion_share": rate(pure, missed),
        "auto_accepted_unmatched": sum(not ln.matched and auto_at(ln, tau) for ln in lines),
        "not_read_lines": kinds.count("not_read"),
        "split_word_lines": sum(ln.split_word for ln in lines),
        "split_word_good_flag_rate": rate(
            sum(not auto_at(ln, tau) for ln in split_good), len(split_good)
        ),
        "split_word_missed": _split_missed(units, tau),
    }


def grade_verdict(
    summary: Mapping, interval: Interval, *, key_ok: bool, tau_present: bool
) -> tuple[str, str | None]:
    """Verdict of a graded cell and the one-line reason when it is not measured."""
    if not key_ok:
        return "not measured", "not measured (setup key mismatch)"
    if not tau_present:
        return "not measured", "not measured (no frozen tau)"
    share = summary["fail_closed_share"]
    if share is not None and share > VOID_SHARE:
        return "not measured", f"not measured (fail-closed lines {share:.1%} > 2 %)"
    bad, docs = summary["bad_lines"], summary["documents_with_bad_lines"]
    if bad < TEST_MIN_BAD or docs < TEST_MIN_DOCS:
        return "not measured", f"not measured ({bad} bad lines from {docs} documents)"
    if interval.point is not None and interval.point <= MAX_MISSED:
        return "pass", None
    return "fail", None
