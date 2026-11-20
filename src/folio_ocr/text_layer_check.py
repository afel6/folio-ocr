"""Checks on the lines of an embedded PDF text layer (review rule R4): base direction,
display-order consistency of each line, per-line content failures and page reasons."""

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from folio_ocr import bidi, brackets
from folio_ocr.document import Box, TextChar
from folio_ocr.flags import (
    CID_CODES,
    EMPTY_WITH_INK,
    INVISIBLE_TEXT,
    PRESENTATION_FORMS,
    PRIVATE_USE,
    TEXT_LAYER_INVALID,
    VISUAL_ORDER,
    has_presentation_forms,
)

_LETTER_CLASSES = {"L": "ltr", "R": "rtl", "AL": "rtl"}
_ALLOWED_CONTROLS = {0x09, 0x0A, 0x0D}
_REPLACEMENT = chr(0xFFFD)


@dataclass(frozen=True)
class LineCheck:
    reasons: tuple[str, ...]  # content failures: cid_codes, private_use, presentation_forms
    flags: tuple[str, ...]  # text_layer_invalid and/or visual_order
    direction: str  # "ltr" or "rtl"


@dataclass(frozen=True)
class TextLayerCheck:
    direction: str  # page base direction D
    lines: tuple[LineCheck, ...]
    reasons: tuple[str, ...]  # invisible_text, empty_with_ink, text_layer_invalid
    invalid_lines: int  # lines with at least one content failure

    @property
    def to_ocr(self) -> bool:
        """The page is read by OCR instead of its text layer."""
        return EMPTY_WITH_INK in self.reasons or TEXT_LAYER_INVALID in self.reasons


def _letter_direction(ch: str) -> str | None:
    """'ltr' or 'rtl' for a strong letter, None for anything else."""
    if not unicodedata.category(ch).startswith("L"):
        return None
    return _LETTER_CLASSES.get(bidi.bidi_class(ch))


def page_direction(chars: Sequence[TextChar]) -> str:
    """Base direction D: the majority of strong letters by count; a tie is 'rtl'."""
    return majority_direction(ch.text for ch in chars)


def majority_direction(texts: Iterable[str]) -> str:
    """'ltr' when strong left-to-right letters outnumber right-to-left ones in `texts`, else
    'rtl' (a tie, no letters included, is 'rtl')."""
    counts = {"ltr": 0, "rtl": 0}
    for text in texts:
        for ch in text:
            direction = _letter_direction(ch)
            if direction:
                counts[direction] += 1
    return "ltr" if counts["ltr"] > counts["rtl"] else "rtl"


def text_direction(text: str, page: str) -> str:
    """Direction of a line known only by its text (OCR and model lines): the direction of its
    letters when they all run one way, else the page direction `page`."""
    seen = {_letter_direction(ch) for ch in text} - {None}
    return seen.pop() if len(seen) == 1 else page


def _compared(ch: TextChar) -> bool:
    """Boxed characters other than whitespace, NSM and BN take part in the order check."""
    if ch.box is None or ch.box[2] <= ch.box[0] or not ch.text.strip():
        return False
    return bidi.bidi_class(ch.text) not in ("NSM", "BN")


def _steps_left(a: Box, b: Box) -> bool:
    centre_a, centre_b = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
    return centre_b < centre_a - 0.5 * min(a[2] - a[0], b[2] - b[0])


def order_holds(chars: Sequence[TextChar], direction: str) -> bool:
    """True when the characters, displayed by the bidi algorithm with base `direction`,
    run left to right on the page (within half the smaller glyph width)."""
    text = "".join(ch.text for ch in chars)
    if bidi.has_explicit_formatting(text):
        return False
    levels = bidi.levels(text, 1 if direction == "rtl" else 0)
    shown = [i for i in bidi.visual_order(levels) if _compared(chars[i])]
    return not any(_steps_left(a, b) for a, b in pairwise(_merge_repeats(chars, shown)))


def _merge_repeats(chars: Sequence[TextChar], shown: Sequence[int]) -> list[Box]:
    """Boxes in display order, with neighbouring copies of one character (`...`, `00`)
    joined into one box: their order cannot change the text, so it is not checked."""
    boxes: list[Box] = []
    last: int | None = None
    for i in shown:
        box = chars[i].box
        if last is not None and abs(last - i) == 1 and chars[last].text == chars[i].text:
            prev = boxes[-1]
            box = (min(prev[0], box[0]), prev[1], max(prev[2], box[2]), prev[3])
            boxes[-1] = box
        else:
            boxes.append(box)
        last = i
    return boxes


def line_direction(chars: Sequence[TextChar], page: str) -> tuple[str, bool]:
    """(direction, order holds) of one line on a page with base direction `page`.

    A line with letters of both directions, or with no letters, is checked under `page`
    only. A line whose letters all run one way passes under either base and takes the
    first that holds: its letters' direction, then the other.
    """
    seen = {_letter_direction(ch.text) for ch in chars} - {None}
    if len(seen) != 1:
        return page, order_holds(chars, page)
    first = seen.pop()
    for base in (first, "ltr" if first == "rtl" else "rtl"):
        if order_holds(chars, base):
            return base, True
    return first, False


def _bad_control(ch: str) -> bool:
    cp = ord(ch)
    return (cp < 0x20 and cp not in _ALLOWED_CONTROLS) or 0x7F <= cp <= 0x9F


def content_failures(chars: Sequence[TextChar]) -> tuple[str, ...]:
    """Content failures of one line, in a fixed order."""
    text = "".join(ch.text for ch in chars)
    checks = (
        (
            CID_CODES,
            "(cid:" in text
            or any(ch.unmapped for ch in chars)
            or any(_bad_control(c) or c == _REPLACEMENT for c in text),
        ),
        (PRIVATE_USE, any(unicodedata.category(c) == "Co" for c in text)),
        (PRESENTATION_FORMS, has_presentation_forms(text)),
    )
    return tuple(name for name, hit in checks if hit)


def _line_check(chars: Sequence[TextChar], page: str, invisible: bool) -> LineCheck:
    reasons = content_failures(chars)
    direction, holds = line_direction(chars, page)
    text = "".join(ch.text for ch in chars)
    holds = holds and not brackets.mirrored(text)
    flags = ((TEXT_LAYER_INVALID,) if reasons or invisible else ()) + (
        () if holds else (VISUAL_ORDER,)
    )
    return LineCheck(reasons, flags, direction)


def check_text_layer(lines: Sequence[Sequence[TextChar]], *, has_ink: bool) -> TextLayerCheck:
    """Line flags and page reasons for the text-layer lines of one page."""
    every = [ch for line in lines for ch in line]
    direction = page_direction(every)
    invisible = any(ch.invisible and ch.text.strip() for ch in every)
    checks = tuple(_line_check(line, direction, invisible) for line in lines)
    invalid = sum(1 for check in checks if check.reasons)
    page_checks = (
        (INVISIBLE_TEXT, invisible),
        (EMPTY_WITH_INK, has_ink and not any(ch.text.strip() for ch in every)),
        (TEXT_LAYER_INVALID, 2 * invalid > len(checks)),
    )
    reasons = tuple(name for name, hit in page_checks if hit)
    return TextLayerCheck(direction, checks, reasons, invalid)
