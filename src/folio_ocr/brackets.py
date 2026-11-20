"""Signs that a text layer holds mirrored brackets or guillemets (PDF readers often return
the mirrored glyph of a bracket shown right to left, so `(٣)` comes back as `)٣(`)."""

import re

from folio_ocr import bidi

_SWAP = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}
# a list marker: up to three digits or one letter and `)`, at the line start or after whitespace
_LIST_MARKER = re.compile(r"(?:^|(?<=\s))(?:\d{1,3}|[^\W\d_])\)")
_OPEN_QUOTE, _CLOSE_QUOTE = chr(0xAB), chr(0xBB)


def _paired(text: str) -> set[int]:
    return {i for pair in bidi.bracket_pairs(text) for i in pair}


def _space(text: str, i: int) -> bool:
    """Whitespace, or outside the line."""
    return i < 0 or i >= len(text) or text[i].isspace()


def faces_away(text: str) -> bool:
    """A bracket outside every bracket pair faces away from its word: a closing bracket
    after whitespace (or the line start) and before a non-space, or an opening bracket
    after a non-space and before whitespace (or the line end)."""
    paired = _paired(text)
    for i, ch in enumerate(text):
        kind = bidi.bracket_kind(ch)
        if kind is None or i in paired:
            continue
        before, after = _space(text, i - 1), _space(text, i + 1)
        if (kind == "close" and before and not after) or (kind == "open" and not before and after):
            return True
    return False


def inside_out_pair(text: str) -> bool:
    """A bracket pair turned inside out: `x( word )y`."""
    return any(
        not _space(text, a - 1)
        and _space(text, a + 1)
        and _space(text, b - 1)
        and not _space(text, b + 1)
        for a, b in bidi.bracket_pairs(text)
    )


def _swapped(text: str, positions: list[int]) -> str:
    chars = list(text)
    for i in positions:
        chars[i] = _SWAP[text[i]]
    return "".join(chars)


def swap_pairs(text: str) -> bool:
    """Swapping the unpaired brackets among ( ) [ ] { }, all together or one at a time,
    pairs one of them (`)٣(`, `)Tomas)`). The `)` of a list marker never counts."""
    markers = {m.end() - 1 for m in _LIST_MARKER.finditer(text)}
    paired = _paired(text)
    loose = [i for i, ch in enumerate(text) if ch in _SWAP and i not in paired | markers]
    trials = [loose] + [[i] for i in loose] if loose else []
    return any(any(i in _paired(_swapped(text, trial)) for i in trial) for trial in trials)


def guillemets_swapped(text: str) -> bool:
    """A `»` before a later `«` with no `«` before it: `»نعم«`."""
    first_close = text.find(_CLOSE_QUOTE)
    if first_close < 0 or _OPEN_QUOTE in text[:first_close]:
        return False
    return _OPEN_QUOTE in text[first_close + 1 :]


def mirrored(text: str) -> bool:
    """Any sign of mirrored brackets or guillemets in one line of a text layer."""
    return faces_away(text) or inside_out_pair(text) or swap_pairs(text) or guillemets_swapped(text)
