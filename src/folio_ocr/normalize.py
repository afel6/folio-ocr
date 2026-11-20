"""Text normalisation applied before scoring Arabic and Latin OCR output."""

import unicodedata

_INVISIBLE = (
    {0xFEFF, 0x061C, 0x200E, 0x200F}
    | set(range(0x200B, 0x200E))
    | set(range(0x202A, 0x202F))
    | set(range(0x2066, 0x206A))
)
_ISOLATED_MARKS = set(range(0xFE70, 0xFE80))
_SHORT_VOWELS = set(range(0x064B, 0x0653)) | {0x0670}

_DELETE_FIRST = {cp: None for cp in _INVISIBLE | _ISOLATED_MARKS}
_PUNCT_MAP = {
    **{cp: "'" for cp in (0x2018, 0x2019, 0x201A, 0x2032)},
    **{cp: '"' for cp in (0x201C, 0x201D, 0x201E, 0x2033)},
    **{cp: "-" for cp in (0x2010, 0x2011, 0x2013, 0x2014, 0x2212)},
    0x2026: "...",
}
_DELETE_LATE = {cp: None for cp in _SHORT_VOWELS | {0x0640}}
_DIGITS = {
    **{0x0660 + i: str(i) for i in range(10)},
    **{0x06F0 + i: str(i) for i in range(10)},
    0x066B: ".",
    0x066C: ",",
    0x066A: "%",
}

_LENIENT_FOLD = {
    **{cp: "ا" for cp in (0x0622, 0x0623, 0x0625, 0x0671)},
    **{cp: "ي" for cp in (0x0649, 0x06CC)},
    0x0629: "ه",
    0x06A9: "ك",
    **{cp: None for cp in (0x0653, 0x0654, 0x0655)},
}


def _nfkc_presentation_forms(text: str) -> str:
    """Apply NFKC only to Arabic presentation-form characters."""
    return "".join(
        unicodedata.normalize("NFKC", ch)
        if 0xFB50 <= ord(ch) <= 0xFDFF or 0xFE80 <= ord(ch) <= 0xFEFC
        else ch
        for ch in text
    )


def norm_v2(text: str) -> str:
    """Strict normalisation: invisible marks, presentation forms, quotes, dashes,
    tatweel, short vowels and digit variants are unified; case and letters stay."""
    text = text.translate(_DELETE_FIRST)
    text = _nfkc_presentation_forms(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_PUNCT_MAP)
    text = text.translate(_DELETE_LATE)
    text = text.translate(_DIGITS)
    return " ".join(text.split())


def lenient_v2(text: str) -> str:
    """Strict normalisation plus case folding, punctuation removal and
    Arabic letter-variant folding."""
    text = norm_v2(text).casefold()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return " ".join(text.translate(_LENIENT_FOLD).split())
