"""Outcome of each ground-truth line of one read page: matched, grouped, deleted or unread."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from folio_ocr.calibration import LOW_CONFIDENCE
from folio_ocr.document import Box, Line, Matrix, Page
from folio_ocr.flags import ORIENTATION_UNCERTAIN
from folio_ocr.metrics import char_errors, compose, gt_line_boxes, match_lines
from folio_ocr.normalize import norm_v2

OCR_READERS = frozenset({"onnxtr", "tesseract"})
BAD_CER = 0.02  # a line is bad when its NORM_V2 CER is above this
COVER_SHARE = 0.5
IOU_THRESHOLD = 0.5
TURN_LIMIT = 45.0  # degrees: a page turned further than this from upright is turned wrong
SPLIT_GROW = 2.0  # pixels the line box grows by when testing a split-word box centre


@dataclass(frozen=True)
class OutputLine:
    """One line the pipeline returned."""

    confidence: float | None
    other_flags: bool  # carries a flag other than "low_confidence"
    low_confidence: bool
    status: str
    matched: bool
    ocr_read: bool
    good: bool | None  # matched OCR-read line: CER <= BAD_CER; else None
    split_word: bool = False  # a word of the line was read in more than one OnnxTR part


@dataclass(frozen=True)
class GtOutcome:
    kind: Literal["matched", "grouping", "pure_deletion", "caught_unread", "not_read"]
    bad: bool
    line: OutputLine | None
    covers: tuple[OutputLine, ...]
    cer: float | None


@dataclass(frozen=True)
class PageOutcome:
    gt: tuple[GtOutcome, ...]
    lines: tuple[OutputLine, ...]  # every OCR-read output line of the page
    turn_wrong: bool = False  # the page is turned more than TURN_LIMIT from the ground truth
    orientation_source: str | None = None  # "osd" or "vote"; None when not recorded
    orientation_uncertain: bool = False  # the page carries the orientation_uncertain flag
    orientation_branch: str | None = None  # "pair", "four" or "none"; None when not recorded
    orientation_margin: float | None = None  # the vote's best score minus the next one
    orientation_failed: bool = False  # Tesseract's orientation answer failed
    turn_off: float = 0.0  # signed degrees the page is turned from the ground truth


def covered_share(box: Box, boxes: Sequence[Box]) -> float:
    """Exact share of `box` area inside the union of `boxes` (overlaps counted once)."""
    x0, y0, x1, y1 = box
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if area <= 0:
        return 0.0
    clipped = [
        (max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)) for b in boxes
    ]
    clipped = [b for b in clipped if b[2] > b[0] and b[3] > b[1]]
    xs = sorted({x for b in clipped for x in (b[0], b[2])})
    covered = 0.0
    for xa, xb in itertools.pairwise(xs):
        spans = sorted((b[1], b[3]) for b in clipped if b[0] <= xa and b[2] >= xb)
        length, end = 0.0, y0
        for ya, yb in spans:
            ya = max(ya, end)
            if yb > ya:
                length += yb - ya
                end = yb
        covered += (xb - xa) * length
    return covered / area


def auto_at(line: OutputLine, tau: float) -> bool:
    """Whether the line is auto accepted at threshold `tau`."""
    return not line.other_flags and line.confidence is not None and line.confidence >= tau


def status_agrees(line: OutputLine, tau: float) -> bool:
    return (line.status == "auto_accepted") == auto_at(line, tau)


def missed_at(outcome: GtOutcome, tau: float) -> bool:
    """Whether a bad ground-truth line would be auto accepted at `tau`."""
    if outcome.kind == "matched":
        return outcome.bad and auto_at(outcome.line, tau)
    if outcome.kind == "grouping":
        return all(auto_at(c, tau) for c in outcome.covers)
    return outcome.kind == "pure_deletion"


def _has_split_word(box: Box, split_boxes: Sequence[Box]) -> bool:
    """Whether the centre of some split-word box lies in `box` grown by SPLIT_GROW."""
    x0, y0, x1, y1 = box
    return any(
        x0 - SPLIT_GROW <= (b[0] + b[2]) / 2 <= x1 + SPLIT_GROW
        and y0 - SPLIT_GROW <= (b[1] + b[3]) / 2 <= y1 + SPLIT_GROW
        for b in split_boxes
    )


def _output_line(
    line: Line,
    *,
    matched: bool,
    good: bool | None,
    split_boxes: Sequence[Box] = (),
    readers: frozenset[str] = OCR_READERS,
) -> OutputLine:
    others = [f for f in line.flags if f != LOW_CONFIDENCE]
    return OutputLine(
        line.confidence,
        bool(others),
        LOW_CONFIDENCE in line.flags,
        line.status,
        matched,
        line.reader in readers,
        good,
        _has_split_word(line.box, split_boxes),
    )


def _overlaps(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _matched_outcome(
    gt_line: Mapping[str, Any],
    line: Line,
    name: str,
    split_boxes: Sequence[Box],
    readers: frozenset[str] = OCR_READERS,
) -> GtOutcome:
    if line.reader not in readers:
        out = _output_line(line, matched=True, good=None, split_boxes=split_boxes, readers=readers)
        return GtOutcome("not_read", False, out, (), None)
    ref = norm_v2(gt_line["text"])
    if not ref:
        raise ValueError(f"ground-truth line is empty after normalisation in {name}")
    cer = char_errors(ref, norm_v2(line.text)).rate
    bad = cer > BAD_CER
    out = _output_line(
        line, matched=True, good=not bad, split_boxes=split_boxes, readers=readers
    )
    return GtOutcome("matched", bad, out, (), cer)


def _unmatched_outcome(box: Box, page: Page, outputs: Sequence[OutputLine]) -> GtOutcome:
    if covered_share(box, page.unread_regions) >= COVER_SHARE:
        return GtOutcome("caught_unread", True, None, (), None)
    if covered_share(box, [ln.box for ln in page.lines]) >= COVER_SHARE:
        covers = tuple(o for o, ln in zip(outputs, page.lines, strict=True) if _overlaps(box, ln.box))
        return GtOutcome("grouping", True, None, covers, None)
    return GtOutcome("pure_deletion", True, None, (), None)


def turn_off(homography: Matrix, transform: Matrix) -> float:
    """The net rotation of `transform` after `homography`, in degrees from -180 to 180."""
    m = compose(transform, homography)
    degrees = math.degrees(math.atan2(m[1][0], m[0][0]))
    return (degrees + 180.0) % 360.0 - 180.0


def turn_wrong(homography: Matrix, transform: Matrix) -> bool:
    """Whether the net rotation of `transform` after `homography` is beyond TURN_LIMIT."""
    return abs(turn_off(homography, transform)) > TURN_LIMIT


def page_outcome(
    gt: Mapping[str, Any],
    page: Page,
    split_boxes: Sequence[Box] = (),
    orientation_source: str | None = None,
    orientation_branch: str | None = None,
    orientation_margin: float | None = None,
    orientation_failed: bool = False,
    readers: frozenset[str] = OCR_READERS,
) -> PageOutcome:
    """Outcome of every ground-truth line of `gt` against the lines read on `page`;
    `split_boxes` are the boxes of words read in more than one OnnxTR part and
    `orientation_source`, `orientation_branch`, `orientation_margin` and `orientation_failed` are
    what the pipeline recorded about the way up of the page; lines whose reader is in `readers`
    count as OCR-read."""
    homography: Matrix = tuple(tuple(float(v) for v in row) for row in gt["homography"])  # type: ignore[assignment]
    gt_boxes = gt_line_boxes([ln["clean_box"] for ln in gt["lines"]], homography, page.transform)
    pairs = dict(match_lines(gt_boxes, [ln.box for ln in page.lines], IOU_THRESHOLD))
    matched_to = {j: g for g, j in pairs.items()}
    outcomes: dict[int, GtOutcome] = {}
    for g, j in pairs.items():
        outcomes[g] = _matched_outcome(gt["lines"][g], page.lines[j], str(gt["file"]), split_boxes, readers
        )
    outputs = tuple(
        outcomes[matched_to[j]].line if j in matched_to and outcomes[matched_to[j]].line
        else _output_line(ln, matched=False, good=None, split_boxes=split_boxes, readers=readers)
        for j, ln in enumerate(page.lines)
    )
    ordered = tuple(
        outcomes[g] if g in outcomes else _unmatched_outcome(gt_boxes[g], page, outputs)
        for g in range(len(gt_boxes))
    )
    off = turn_off(homography, page.transform)
    return PageOutcome(
        ordered,
        tuple(o for o in outputs if o.ocr_read),
        abs(off) > TURN_LIMIT,
        orientation_source,
        ORIENTATION_UNCERTAIN in page.flags,
        orientation_branch,
        orientation_margin,
        orientation_failed,
        off,
    )
