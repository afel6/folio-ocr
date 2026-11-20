"""Fixtures beyond the six documents: second type size, adversarial pages, per-character page, bulleted and invisible-text PDFs."""

from __future__ import annotations

import io
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from fixture_fonts import FONT_DIR, FONT_FILES, font_key
from fixture_images import SCAN_QUALITY, jpeg_bytes, render_pdf
from fixture_output import document, emit, rendered
from fixture_pages import (
    FONT_PT,
    MARGIN,
    PAGE_H,
    PAGE_W,
    PITCH,
    Placed,
    cell_boxes,
    draw_cells,
    layout_document,
    new_pdf,
)
from fixture_text import (
    AR_ADVERSARIAL,
    AR_BULLETED,
    BULLET_ROWS,
    EN_ADVERSARIAL,
    EN_BULLETED,
    DocText,
    Row,
    TextLine,
    build_line,
)
from fontTools.ttLib import TTFont
from fpdf.enums import TextMode
from must_not_flag import LTR, RTL

TOKENS = (("en", "en"), ("ar", "ar"), ("mixed", "en+ar"))
SECOND_SCALE = 16 / 11
BULLET_CHAR = chr(0xF0B7)  # the private-use bullet a word processor writes
DIGIT_JOINERS = "/.," + chr(0x066B) + chr(0x066C)
PER_CHAR_NAME = "ar-letter-per_char.pdf"
INVISIBLE_NAME = "ar-letter-invisible_ocr.pdf"
FIXTURE_TAGS = {
    **{f"{t}-invoice-16pt.pdf": ["second_size"] for t, _ in TOKENS},
    "ar-adversarial-1.pdf": ["adversarial"],
    "ar-adversarial-2.pdf": ["adversarial"],
    "en-adversarial.pdf": ["adversarial"],
    "ar-adversarial-3.pdf": ["adversarial"],
    "en-adversarial-2.pdf": ["adversarial"],
    PER_CHAR_NAME: ["adversarial"],
    "en-bulleted.pdf": ["bulleted"],
    "ar-bulleted.pdf": ["bulleted"],
    INVISIBLE_NAME: ["planted_errors"],
}
IDENT = np.eye(3)
INDENT_BULLET, INDENT_TEXT, BULLET_CELL, CELL_MARGIN = 18, 36, 14, 2


def build_second_size(seed: int, out_dir: Path) -> tuple[Path, ...]:
    """The three invoices drawn again with the whole layout scaled to a 16 pt type size."""
    files: tuple[Path, ...] = ()
    for token, language in TOKENS:
        doc, placed = document(token, language, "invoice", seed, SECOND_SCALE)
        pdf, img, boxes = rendered(placed, "amiri", SECOND_SCALE)
        meta = (language, "invoice", doc.direction)
        name = f"{token}-invoice-16pt.pdf"
        files += emit(out_dir, name, pdf, "born_digital", meta, img.size, IDENT, placed, boxes)
    return files


def _single_cell_doc(language: str, direction: str, layout: str, texts) -> DocText:
    rows = tuple(Row((build_line([(unicodedata.normalize("NFC", t), None)]),)) for t in texts)
    return DocText(language, layout, direction, rows)


def build_adversarial(out_dir: Path) -> tuple[Path, ...]:
    """Pages of numbers, dates, punctuation, harakat, tatweel and mixed scripts, one line per row."""
    half = len(AR_ADVERSARIAL) // 2  # 28 lines do not fit one page at 11 pt
    pages = (
        ("ar-adversarial-1.pdf", "en+ar", "rtl", AR_ADVERSARIAL[:half]),
        ("ar-adversarial-2.pdf", "en+ar", "rtl", AR_ADVERSARIAL[half:]),
        ("en-adversarial.pdf", "en+ar", "ltr", EN_ADVERSARIAL),
        # lines whose order must not be flagged: tight brackets, quotes, dashes, ellipses
        ("ar-adversarial-3.pdf", "en+ar", "rtl", (*RTL[24:], RTL[9])),
        ("en-adversarial-2.pdf", "en+ar", "ltr", LTR),
    )
    files: tuple[Path, ...] = ()
    for name, language, direction, texts in pages:
        placed = layout_document(_single_cell_doc(language, direction, "adversarial", texts))
        # harakat shape to mark glyphs on purpose, so the single-glyph-cluster check is not run here
        pdf, img, boxes = rendered(placed, "amiri", clusters=False)
        meta = (language, "adversarial", direction)
        files += emit(out_dir, name, pdf, "born_digital", meta, img.size, IDENT, placed, boxes)
    return files


def _digit_like(text: str, i: int) -> bool:
    """A digit, or a separator lying between two digits."""
    if unicodedata.category(text[i]) == "Nd":
        return True
    inside = 0 < i < len(text) - 1
    return (
        inside
        and text[i] in DIGIT_JOINERS
        and unicodedata.category(text[i - 1]) == "Nd"
        and unicodedata.category(text[i + 1]) == "Nd"
    )


def visual_order(text: str) -> list[int]:
    """Logical indices of `text` in left-to-right visual order for right-to-left text.

    The characters read in reverse, except that each run of digits (with `/ . , ٫ ٬` between two
    digits) keeps its logical left-to-right order.
    """
    groups: list[list[int]] = []
    for i in range(len(text)):
        if groups and _digit_like(text, i) and _digit_like(text, groups[-1][-1]):
            groups[-1].append(i)
        else:
            groups.append([i])
    return [i for group in reversed(groups) for i in group]


def _per_char_pdf(placed: list[Placed]) -> bytes:
    """Unshaped Amiri, one character at a time, right aligned, emitted in logical order."""
    pdf = new_pdf("amiri", shaping=False)
    for p in placed:
        text = p.line.text
        pdf.set_font(font_key(text, "amiri"), size=FONT_PT)
        xs, right = {}, p.x + p.w - 1
        for i in reversed(visual_order(text)):
            right -= pdf.get_string_width(text[i])
            xs[i] = right
        if right < p.x:
            raise ValueError(f"line is wider than its cell: {text!r}")
        baseline = p.y + PITCH / 2 + 0.35 * FONT_PT
        for i, ch in enumerate(text):
            pdf.text(xs[i], baseline, ch)
    return bytes(pdf.output())


def build_per_char(seed: int, out_dir: Path) -> tuple[Path, ...]:
    """The Arabic letter drawn one unshaped character at a time on every line."""
    doc, placed = document("ar", "ar", "letter", seed)
    if any(c.isascii() and c.isalpha() for p in placed for c in p.line.text):
        raise ValueError("the per-character page holds a Latin letter")
    data = _per_char_pdf(placed)
    img = render_pdf(data)
    meta = ("ar", "letter", doc.direction)
    boxes = cell_boxes(img, placed)
    return emit(out_dir, PER_CHAR_NAME, data, "per_character", meta, img.size, IDENT, placed, boxes)


def bullet_font_bytes() -> bytes:
    """Noto Sans with U+F0B7 added to every Unicode cmap, drawing the same glyph as U+2022."""
    font = TTFont(FONT_DIR / FONT_FILES["sans"], recalcTimestamp=False)
    for table in font["cmap"].tables:
        if table.isUnicode():
            table.cmap[0xF0B7] = table.cmap[0x2022]
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def _bullet_cell(direction: str, y: float) -> Placed:
    """The bullet's cell: its glyph sits 18 pt from the column start (mirrored for rtl)."""
    bullet = TextLine("\N{BULLET}", direction, ())
    if direction == "rtl":
        edge = PAGE_W - MARGIN - INDENT_BULLET + CELL_MARGIN
        return Placed(bullet, edge - BULLET_CELL, y, BULLET_CELL)
    return Placed(bullet, MARGIN + INDENT_BULLET - CELL_MARGIN, y, BULLET_CELL)


def _item_cell(line: TextLine, direction: str, y: float) -> Placed:
    """An item's cell: its text starts 36 pt from the column start (right aligned for rtl)."""
    right = PAGE_W - MARGIN
    if direction == "rtl":
        return Placed(line, MARGIN, y, right - INDENT_TEXT + CELL_MARGIN - MARGIN)
    x = MARGIN + INDENT_TEXT - CELL_MARGIN
    return Placed(line, x, y, right - x)


def _bullet_rows(direction: str, texts) -> tuple[list[Placed], list[str], list[str]]:
    """Placed cells in reading order, the text drawn in each and its font key."""
    placed, drawn, fonts = [], [], []
    for index, (kind, text) in enumerate(zip(BULLET_ROWS, texts, strict=True)):
        y = MARGIN + index * PITCH
        line = build_line([(unicodedata.normalize("NFC", text), None)])
        if kind == "bullet":
            placed += [_bullet_cell(direction, y), _item_cell(line, direction, y)]
            drawn += [BULLET_CHAR, line.text]
            fonts += ["bullet", font_key(line.text, "amiri")]
        else:
            placed.append(Placed(line, MARGIN, y, PAGE_W - 2 * MARGIN))
            drawn.append(line.text)
            fonts.append(font_key(line.text, "amiri"))
    return placed, drawn, fonts


def _bulleted_pdf(placed: list[Placed], drawn: list[str], fonts: list[str]) -> bytes:
    """Draw the page with the bullet font loaded from a temporary file that is removed afterwards."""
    with tempfile.TemporaryDirectory() as tmp:
        font_file = Path(tmp) / "bullet.ttf"
        font_file.write_bytes(bullet_font_bytes())
        pdf = new_pdf("amiri")
        pdf.add_font("bullet", fname=str(font_file))
        draw_cells(pdf, placed, "amiri", texts=drawn, fonts=fonts)
        return bytes(pdf.output())


def build_bulleted(out_dir: Path) -> tuple[Path, ...]:
    """Word-style pages: bullets in a private-use code point with a hanging indent."""
    files: tuple[Path, ...] = ()
    for name, language, direction, texts in (
        ("en-bulleted.pdf", "en", "ltr", EN_BULLETED),
        ("ar-bulleted.pdf", "ar", "rtl", AR_BULLETED),
    ):
        placed, drawn, fonts = _bullet_rows(direction, texts)
        data = _bulleted_pdf(placed, drawn, fonts)
        img = render_pdf(data)
        meta = (language, "bulleted", direction)
        boxes = cell_boxes(img, placed)
        files += emit(out_dir, name, data, "born_digital", meta, img.size, IDENT, placed, boxes)
    return files


def _swap_digits(text: str, start: int, end: int) -> str:
    """Swap the first two different digits inside text[start:end]."""
    pos = [i for i in range(start, end) if unicodedata.category(text[i]) == "Nd"]
    for a in pos:
        for b in pos:
            if b > a and unicodedata.digit(text[a]) != unicodedata.digit(text[b]):
                chars = list(text)
                chars[a], chars[b] = chars[b], chars[a]
                return "".join(chars)
    raise ValueError(f"no two different digits in {text[start:end]!r}")


def _bump_digit(text: str, start: int, end: int) -> str:
    """Raise the first digit inside text[start:end] by one (9 becomes 0)."""
    i = next(i for i in range(start, end) if unicodedata.category(text[i]) == "Nd")
    d = unicodedata.digit(text[i])
    return text[:i] + chr(ord(text[i]) - d + (d + 1) % 10) + text[i + 1 :]


def _first_entity(lines: list[TextLine], kind: str, used: set[int]):
    for i, line in enumerate(lines):
        for e in line.entities:
            if e.type == kind and i not in used:
                return i, e
    raise ValueError(f"no line with a {kind} entity to edit")


def plant_errors(lines: list[TextLine]) -> list[str]:
    """The texts of `lines` with four planted recognition errors.

    Swapped date digits, one changed amount digit, one Arabic letter replaced by another and
    one word deleted, each on a different line.
    """
    texts, used = [ln.text for ln in lines], set()
    i, e = _first_entity(lines, "date", used)
    texts[i] = _swap_digits(texts[i], e.start, e.end)
    used.add(i)
    i, e = _first_entity(lines, "amount", used)
    texts[i] = _bump_digit(texts[i], e.start, e.end)
    used.add(i)
    j = next(k for k, t in enumerate(texts) if k not in used and "ر" in t)
    texts[j] = texts[j].replace("ر", "ز", 1)
    used.add(j)
    k = next(k for k, t in enumerate(texts) if k not in used and " مقدمة" in t)
    texts[k] = texts[k].replace(" مقدمة", "", 1)
    return texts


def invisible_layer_pdf(placed: list[Placed], page_jpeg: bytes, texts: list[str]) -> bytes:
    """A page image with an invisible (render mode 3) text layer drawn over it."""
    pdf = new_pdf("amiri")
    pdf.image(io.BytesIO(page_jpeg), x=0, y=0, w=PAGE_W, h=PAGE_H)
    pdf.text_mode = TextMode.INVISIBLE
    draw_cells(pdf, placed, "amiri", texts=texts)
    return bytes(pdf.output())


def build_invisible(seed: int, out_dir: Path) -> tuple[Path, ...]:
    """The Arabic letter as a page image with an invisible text layer holding planted errors."""
    doc, placed = document("ar", "ar", "letter", seed)
    _, img, boxes = rendered(placed, "amiri")
    texts = plant_errors([p.line for p in placed])
    data = invisible_layer_pdf(placed, jpeg_bytes(img, SCAN_QUALITY), texts)
    meta = ("ar", "letter", doc.direction)
    return emit(
        out_dir, INVISIBLE_NAME, data, "invisible_text", meta, img.size, IDENT, placed, boxes
    )


def build_special(seed: int, out_dir: Path) -> tuple[Path, ...]:
    """Write the second-size, adversarial, per-character, bulleted and invisible-layer fixtures."""
    return (
        *build_second_size(seed, out_dir),
        *build_adversarial(out_dir),
        *build_per_char(seed, out_dir),
        *build_bulleted(out_dir),
        *build_invisible(seed, out_dir),
    )
