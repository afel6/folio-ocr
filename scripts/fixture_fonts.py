"""Fonts used by the fixtures, plus checks that the text has glyphs, joined forms and one glyph per cluster."""

from __future__ import annotations

import unicodedata
from functools import cache
from pathlib import Path

import uharfbuzz as hb
from fixture_text import TextLine
from fontTools.ttLib import TTFont

from folio_ocr.charsets import in_ranges

FONT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fonts"
FONT_FILES = {
    "sans": "NotoSans-Regular.ttf",
    "naskh": "NotoNaskhArabic-Regular.ttf",
    "amiri": "Amiri-Regular.ttf",
}


def font_path(key: str) -> Path:
    return FONT_DIR / FONT_FILES[key]


@cache
def _font_bytes(key: str) -> bytes:
    return font_path(key).read_bytes()


@cache
def _cmap(key: str) -> frozenset[int]:
    return frozenset(TTFont(font_path(key)).getBestCmap())


def _is_arabic_block(ch: str) -> bool:
    return in_ranges(ch, ((0x0600, 0x06FF),))


def _is_arabic_letter(ch: str) -> bool:
    return _is_arabic_block(ch) and unicodedata.category(ch) == "Lo"


def font_key(text: str, arabic_key: str) -> str:
    """Font drawing a line: the Arabic font if it has any Arabic-block character, else Noto Sans."""
    return arabic_key if any(_is_arabic_block(c) for c in text) else "sans"


def check_coverage(lines: list[TextLine], arabic_key: str) -> None:
    """Raise ValueError if a character of a line is missing from every font that draws it."""
    fallback = _cmap("sans") if arabic_key == "naskh" else frozenset()
    for line in lines:
        key = font_key(line.text, arabic_key)
        missing = sorted({c for c in line.text if ord(c) not in _cmap(key) | fallback})
        if missing:
            raise ValueError(f"{key} has no glyph for {missing!r} in line {line.text!r}")


def _shape_arabic(text: str, font: hb.Font) -> hb.Buffer:
    """Shape `text` right to left as Arabic and return the buffer of glyphs."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction, buf.script, buf.language = "rtl", "Arab", "ar"
    hb.shape(font, buf, {})
    return buf


def _has_joined_form(text: str, key: str) -> bool:
    font = hb.Font(hb.Face(_font_bytes(key)))
    buf = _shape_arabic(text, font)
    for info in buf.glyph_infos:
        ch = text[info.cluster]
        if _is_arabic_letter(ch) and info.codepoint != font.get_nominal_glyph(ord(ch)):
            return True
    return False


def check_shaping(lines: list[TextLine], arabic_key: str) -> None:
    """Raise ValueError if a line with Arabic letters shapes to isolated forms only."""
    for line in lines:
        if any(_is_arabic_letter(c) for c in line.text) and not _has_joined_form(
            line.text, arabic_key
        ):
            raise ValueError(f"no joined letter forms in {arabic_key} for {line.text!r}")


def check_single_glyph_clusters(lines: list[TextLine]) -> None:
    """Raise ValueError if Amiri shapes a character of an Arabic line to more than one glyph.

    The PDF writer gives the extra glyph no Unicode mapping, so text extraction would show a stray character.
    """
    font = hb.Font(hb.Face(_font_bytes("amiri")))
    for line in lines:
        if not any(_is_arabic_block(c) for c in line.text):
            continue
        clusters = [info.cluster for info in _shape_arabic(line.text, font).glyph_infos]
        if len(clusters) != len(set(clusters)):
            raise ValueError(f"a character shapes to several glyphs in amiri for {line.text!r}")
