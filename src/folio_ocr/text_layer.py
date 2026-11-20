"""Read the embedded text of a PDF page as characters with pixel boxes, split into lines."""

import statistics
from collections.abc import Callable, Sequence

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from folio_ocr.document import Box, TextChar
from folio_ocr.ingest import RENDER_DPI, render_size
from folio_ocr.text_layer_check import page_direction

# A row splits where the horizontal gap between glyphs exceeds this many times the row's
# median glyph height (calibrated on the born-digital fixtures at 11 and 16 pt).
GAP_FACTOR = 0.7
_INVISIBLE_MODES = (
    pdfium_raw.FPDF_TEXTRENDERMODE_INVISIBLE,
    pdfium_raw.FPDF_TEXTRENDERMODE_CLIP,
)


def _read_code_points(tp, count: int) -> list[tuple[int, int, bool]]:
    """(index, code unit, generated) for every character of the text page."""
    return [
        (
            i,
            pdfium_raw.FPDFText_GetUnicode(tp.raw, i),
            pdfium_raw.FPDFText_IsGenerated(tp.raw, i) == 1,
        )
        for i in range(count)
    ]


def _combine_surrogates(
    units: Sequence[tuple[int, int, bool]],
) -> list[tuple[int, int, bool]]:
    """Join each high+low surrogate pair into one code point at the first index.

    A lone high or lone low surrogate stays as its own entry.
    """
    out: list[tuple[int, int, bool]] = []
    i = 0
    while i < len(units):
        idx, code, generated = units[i]
        i += 1
        if 0xD800 <= code <= 0xDBFF and i < len(units) and 0xDC00 <= units[i][1] <= 0xDFFF:
            code = 0x10000 + ((code - 0xD800) << 10) + (units[i][1] - 0xDC00)
            i += 1
        out.append((idx, code, generated))
    return out


def _invisible(tp, index: int) -> bool:
    obj = pdfium_raw.FPDFText_GetTextObject(tp.raw, index)
    return bool(obj) and pdfium_raw.FPDFTextObj_GetTextRenderMode(obj) in _INVISIBLE_MODES


def device_map(page: pdfium.PdfPage, dpi: int) -> Callable[[float, float, float, float], Box]:
    """Map a user-space rect (left, bottom, right, top) to a pixel box (x0, y0, x1, y1) in
    the frame of `page.render(scale=dpi / 72)`, which applies /Rotate and the page box."""
    s = dpi / 72
    bx0, by0, bx1, by1 = page.get_bbox()
    rotation = page.get_rotation() % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("page rotation must be a multiple of 90 degrees")
    point = {
        0: lambda x, y: ((x - bx0) * s, (by1 - y) * s),
        90: lambda x, y: ((y - by0) * s, (x - bx0) * s),
        180: lambda x, y: ((bx1 - x) * s, (y - by0) * s),
        270: lambda x, y: ((by1 - y) * s, (bx1 - x) * s),
    }[rotation]

    def to_pixels(left: float, bottom: float, right: float, top: float) -> Box:
        (xa, ya), (xb, yb) = point(left, bottom), point(right, top)
        return (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    return to_pixels


def _holds_image(obj) -> bool:
    """True when the page object is an image or a form with an image at any depth."""
    kind = pdfium_raw.FPDFPageObj_GetType(obj)
    if kind == pdfium_raw.FPDF_PAGEOBJ_IMAGE:
        return True
    if kind != pdfium_raw.FPDF_PAGEOBJ_FORM:
        return False
    return any(
        _holds_image(pdfium_raw.FPDFFormObj_GetObject(obj, i))
        for i in range(pdfium_raw.FPDFFormObj_CountObjects(obj))
    )


def image_boxes(page: pdfium.PdfPage, dpi: int = RENDER_DPI) -> tuple[Box, ...]:
    """Pixel boxes of the images drawn on `page` (top-level images and forms holding one),
    clipped to the rendered page, sorted top to bottom then left to right."""
    to_pixels = device_map(page, dpi)
    width, height = render_size(*page.get_size(), dpi)
    boxes = []
    for obj in page.get_objects(max_depth=1):
        if not _holds_image(obj.raw):
            continue
        x0, y0, x1, y1 = to_pixels(*obj.get_bounds())
        box = (max(x0, 0.0), max(y0, 0.0), min(x1, float(width)), min(y1, float(height)))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return tuple(sorted(boxes, key=lambda b: (b[1], b[0])))


def page_chars(page: pdfium.PdfPage, dpi: int = RENDER_DPI) -> tuple[TextChar, ...]:
    """Characters of `page` in pdfium's index order, with boxes in pixels at `dpi`."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    to_pixels = device_map(page, dpi)
    tp = page.get_textpage()
    try:
        units = _read_code_points(tp, tp.count_chars())
        out: list[TextChar] = []
        for idx, code, generated in _combine_surrogates(units):
            text = chr(code)
            if generated and text in "\r\n":
                continue
            box = tight = None
            if not generated:
                box = to_pixels(*tp.get_charbox(idx, loose=True))
                tight = to_pixels(*tp.get_charbox(idx, loose=False))
            unmapped = pdfium_raw.FPDFText_HasUnicodeMapError(tp.raw, idx) == 1
            out.append(TextChar(text, box, unmapped, tight, _invisible(tp, idx)))
        return tuple(out)
    finally:
        tp.close()


def _has_height(box: Box | None) -> bool:
    return box is not None and box[3] > box[1]


def _rows(glyphs: list[tuple[int, TextChar]]) -> list[list[tuple[int, TextChar]]]:
    """Group glyphs into rows: connected components of glyphs whose vertical overlap is at
    least half the smaller of their heights. Rows are ordered top to bottom by the median
    glyph centre, ties by the smallest character index."""
    parent = list(range(len(glyphs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tops = [item[1].box[1] for item in glyphs]
    bottoms = [item[1].box[3] for item in glyphs]
    active: list[int] = []
    for k in sorted(range(len(glyphs)), key=tops.__getitem__):
        top, bottom = tops[k], bottoms[k]
        active = [a for a in active if bottoms[a] > top]
        height = bottom - top
        for a in [
            a
            for a in active
            if min(bottoms[a], bottom) - top >= 0.5 * min(bottoms[a] - tops[a], height)
        ]:
            parent[find(a)] = find(k)
        active.append(k)
    groups: dict[int, list[tuple[int, TextChar]]] = {}
    for k, item in enumerate(glyphs):
        groups.setdefault(find(k), []).append(item)

    def order(row: list[tuple[int, TextChar]]) -> tuple[float, int]:
        centres = [(ch.box[1] + ch.box[3]) / 2 for _, ch in row]
        return statistics.median(centres), min(i for i, _ in row)

    return sorted(groups.values(), key=order)


def _split_row(row: list[tuple[int, TextChar]], factor: float) -> list[list[tuple[int, TextChar]]]:
    """Cut a row where the gap between glyphs exceeds `factor` x its median glyph height."""
    limit = factor * statistics.median(ch.box[3] - ch.box[1] for _, ch in row)
    parts: list[list[tuple[int, TextChar]]] = []
    right = None
    for item in sorted(row, key=lambda t: (t[1].box[0], t[1].box[2], t[0])):
        box = item[1].box
        if right is None or box[0] - right > limit:
            parts.append([])
            right = box[2]
        parts[-1].append(item)
        right = max(right, box[2])
    return parts


def _attach(
    chars: Sequence[TextChar], segment_of: dict[int, int], count: int
) -> list[list[TextChar]]:
    """Every character in index order, in the segment of the glyphs on both its sides.

    Whitespace and boxless characters between two segments are dropped, so a segment
    never starts or ends with whitespace.
    """
    previous: list[int | None] = []
    last = None
    for i in range(len(chars)):
        last = segment_of.get(i, last)
        previous.append(last)
    nxt = None
    segments: list[list[TextChar]] = [[] for _ in range(count)]
    for i in range(len(chars) - 1, -1, -1):
        nxt = segment_of.get(i, nxt)
        if previous[i] is not None and previous[i] == nxt:
            segments[nxt].append(chars[i])
    return [seg[::-1] for seg in segments]


def segment_lines(
    chars: Sequence[TextChar], direction: str, factor: float = GAP_FACTOR
) -> tuple[tuple[TextChar, ...], ...]:
    """Lines of the text layer: rows of vertically overlapping glyphs cut at wide gaps.

    Rows run top to bottom; the segments of a row run left to right for 'ltr' and right
    to left for 'rtl'. Inside a segment characters keep pdfium's index order.
    """
    glyphs = [(i, ch) for i, ch in enumerate(chars) if ch.text.strip() and _has_height(ch.box)]
    ordered = _segments(glyphs, direction, factor)
    segment_of = {i: n for n, part in enumerate(ordered) for i, _ in part}
    return tuple(tuple(seg) for seg in _attach(chars, segment_of, len(ordered)) if seg)


def _segments(
    glyphs: list[tuple[int, TextChar]], direction: str, factor: float
) -> list[list[tuple[int, TextChar]]]:
    """Rows top to bottom, each cut at wide gaps; the parts of a row run left to right for
    'ltr' and right to left for 'rtl'. Inside a part, glyphs run left to right."""
    ordered: list[list[tuple[int, TextChar]]] = []
    for row in _rows(glyphs):
        parts = _split_row(row, factor)
        if direction == "rtl":
            parts.sort(key=lambda p: -max(ch.box[2] for _, ch in p))
        ordered += parts
    return ordered


def group_boxes(boxes: Sequence[Box], direction: str, factor: float) -> list[list[int]]:
    """Indices of word boxes grouped into lines the way `segment_lines` groups glyphs; inside a
    line the indices run left to right. Boxes without height are left out."""
    glyphs = [(i, TextChar("□", box)) for i, box in enumerate(boxes) if _has_height(box)]
    return [[i for i, _ in part] for part in _segments(glyphs, direction, factor)]


def read_text_layer(
    page: pdfium.PdfPage, dpi: int = RENDER_DPI
) -> tuple[tuple[TextChar, ...], ...]:
    chars = page_chars(page, dpi)
    return segment_lines(chars, page_direction(chars))


def line_text(chars: Sequence[TextChar]) -> str:
    return "".join(c.text for c in chars)


def line_box(chars: Sequence[TextChar]) -> Box | None:
    """Bounds of the glyph outlines of a line (advance boxes where a glyph has none)."""
    boxes = [
        ch.tight_box if _has_height(ch.tight_box) else ch.box
        for ch in chars
        if ch.text.strip() and (_has_height(ch.tight_box) or _has_height(ch.box))
    ]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
