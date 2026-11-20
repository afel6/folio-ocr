"""Character classes shared by the metrics and the review flags."""

import unicodedata

ARABIC_BLOCKS = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
LATIN_LETTERS = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00AA, 0x00AA),
    (0x00B5, 0x00B5),
    (0x00BA, 0x00BA),
    (0x00C0, 0x024F),
    (0x0250, 0x02FF),
    (0x1D00, 0x1DBF),
    (0x1E00, 0x1EFF),
    (0x2071, 0x2071),
    (0x207F, 0x207F),
    (0x2090, 0x209C),
    (0x2C60, 0x2C7F),
    (0xA720, 0xA7FF),
    (0xAB30, 0xAB6F),
    (0xFB00, 0xFB06),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
)
PRESENTATION_FORMS = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
LOOKALIKES = (
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x052F),  # Cyrillic
    (0x06A9, 0x06A9),  # keheh, looks like kaf
    (0x06BE, 0x06BE),  # heh doachashmee, looks like heh
    (0x06C1, 0x06C1),  # heh goal, looks like heh
    (0x06CC, 0x06CC),  # Farsi yeh, looks like yeh / alef maksura
    (0x06D5, 0x06D5),  # ae, looks like teh marbuta / heh
    (0x06F0, 0x06F9),  # extended Arabic-Indic digits
    (0xFF01, 0xFF5E),  # full-width Latin, digits and punctuation
    (0x1D400, 0x1D7FF),  # mathematical alphanumerics
)


def in_ranges(ch: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    """True when the single character `ch` lies in one of the inclusive ranges."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_of(ch: str) -> str | None:
    """'arabic' or 'latin' for letters and marks, None for digits, punctuation, spaces."""
    if not unicodedata.category(ch).startswith(("L", "M")):
        return None
    return "arabic" if in_ranges(ch, ARABIC_BLOCKS) else "latin"


def is_arabic_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L") and in_ranges(ch, ARABIC_BLOCKS)


def is_latin_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L") and in_ranges(ch, LATIN_LETTERS)


def is_other_script_letter(ch: str) -> bool:
    """A letter that is neither Latin nor Arabic."""
    return (
        unicodedata.category(ch).startswith("L")
        and not is_latin_letter(ch)
        and not is_arabic_letter(ch)
    )


def line_script(text: str) -> str:
    """Script of a line from its letters: 'latin', 'arabic', 'mixed' (letters of two or more
    scripts) or 'unknown' (no letters, or letters of one script other than Latin and Arabic)."""
    scripts = {
        "arabic" if is_arabic_letter(c) else "latin" if is_latin_letter(c) else "other"
        for c in text
        if unicodedata.category(c).startswith("L")
    }
    if len(scripts) > 1:
        return "mixed"
    return scripts.pop() if scripts & {"latin", "arabic"} else "unknown"
