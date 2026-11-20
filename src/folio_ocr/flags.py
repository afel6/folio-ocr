"""Review flags for lines of text, and the review status they lead to."""

import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace

from folio_ocr import charsets
from folio_ocr.charsets import in_ranges, is_arabic_letter, is_other_script_letter
from folio_ocr.document import Line

HANDWRITTEN = "handwritten"
EMPTY = "empty"
REPEATED = "repeated"
SCRIPT_MISMATCH = "script_mismatch"
PRESENTATION_FORMS = "presentation_forms"
LOOKALIKE_CHARS = "lookalike_chars"
VISUAL_ORDER = "visual_order"
TEXT_LAYER_INVALID = "text_layer_invalid"
TRUNCATED = "truncated"
NO_TRANSCRIPTION = "no_transcription"
FORMAT_NOISE = "format_noise"
NOT_READ = "not_read"
NEEDS_MODEL_SERVER = "needs_model_server"
MODEL_SELF_TEST_FAILED = "model_self_test_failed"
INVALID_TEXT = "invalid_text"
CID_CODES = "cid_codes"
PRIVATE_USE = "private_use"
EMPTY_WITH_INK = "empty_with_ink"
INVISIBLE_TEXT = "invisible_text"
UNREAD_REGIONS = "unread_regions"
ORIENTATION_UNCERTAIN = "orientation_uncertain"

LANGUAGES = ("en", "ar", "en+ar")
_EXPECTED_SHARE = dict(zip(LANGUAGES, ((0, 0), (100, 100), (0, 100)), strict=True))
_TOLERANCE = 30


def clean_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Replace lone surrogates with U+FFFD; flag only when that added replacements."""
    fixed = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    added = fixed.count("\ufffd") > text.count("\ufffd")
    return fixed, ((INVALID_TEXT,) if added else ())


def is_empty(text: str) -> bool:
    return not text.strip()


def is_repeated(text: str, size: int = 20, times: int = 3) -> bool:
    """True when some `size`-character window occurs `times` or more times, overlaps counted."""
    counts = Counter(text[i : i + size] for i in range(len(text) - size + 1))
    return any(n >= times for n in counts.values())


def arabic_letter_share(text: str) -> float | None:
    """Percentage (0-100) of letters that are Arabic, or None without letters."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return None
    return 100 * sum(is_arabic_letter(c) for c in letters) / len(letters)


def _check_languages(languages: str) -> None:
    if languages not in _EXPECTED_SHARE:
        raise ValueError("languages must be 'en', 'ar' or 'en+ar'")


def script_mismatch(text: str, languages: str) -> bool:
    """True for any letter outside Latin and Arabic, or an unexpected Arabic share."""
    _check_languages(languages)
    if any(is_other_script_letter(c) for c in text):
        return True
    share = arabic_letter_share(text)
    if share is None:
        return False
    low, high = _EXPECTED_SHARE[languages]
    return share < low - _TOLERANCE or share > high + _TOLERANCE


def has_presentation_forms(text: str) -> bool:
    return any(in_ranges(c, charsets.PRESENTATION_FORMS) for c in text)


def has_lookalikes(text: str) -> bool:
    return any(in_ranges(c, charsets.LOOKALIKES) for c in text)


def line_flags(text: str, languages: str, *, handwritten: bool = False) -> tuple[str, ...]:
    _check_languages(languages)
    out = [HANDWRITTEN] if handwritten else []
    if is_empty(text):
        return tuple(out + [EMPTY])
    checks = (
        (REPEATED, is_repeated(text)),
        (SCRIPT_MISMATCH, script_mismatch(text, languages)),
        (PRESENTATION_FORMS, has_presentation_forms(text)),
        (LOOKALIKE_CHARS, has_lookalikes(text)),
    )
    return tuple(out + [name for name, hit in checks if hit])


def needs_model_server(
    languages: str, *, model_configured: bool, tesseract_configured: bool
) -> bool:
    _check_languages(languages)
    return not model_configured and languages != "en" and not tesseract_configured


def review_status(flags: Iterable[str], status: str) -> str:
    if status in ("edited", "confirmed"):
        return status
    return "needs_review" if tuple(flags) else "auto_accepted"


def with_flags(line: Line, flags: Iterable[str]) -> Line:
    merged = tuple(dict.fromkeys((*line.flags, *flags)))
    return replace(line, flags=merged, status=review_status(merged, line.status))
