"""Checks on what the file reader process sends back. A hostile file may control that process,
so every reply is checked against what folio expects before anything is built from it: types,
counts, page sizes from folio's own arithmetic, and pixel lengths. A reply that fails a check
raises `pdf_frames.Ended`."""

import base64
import binascii
import math
import re
from dataclasses import dataclass

from PIL import Image

from folio_ocr import ingest
from folio_ocr.document import Box, TextChar
from folio_ocr.pdf_frames import Ended
from folio_ocr.text_layer import line_box

_BANDS = {"L": 1, "RGB": 3}
_MAX_CODE_POINT = 0x10FFFF
# Every line the file checks can refuse a file with; a refusal outside this list is not shown.
_PAGE_LINES = (
    re.compile(r"page ([1-9][0-9]?) is too large to read \(over 50 megapixels at 300 dpi\)"),
    re.compile(r"page ([1-9][0-9]?) holds an image too large to read \(over 50 megapixels\)"),
    re.compile(r"page ([1-9][0-9]?) of the PDF could not be read; it may be damaged"),
)


@dataclass(frozen=True)
class PdfPage:
    """A page as the reader process saw it: its gray render, its text layer as lines of
    characters (`text_layer.text_lines`, grouped in the reader process, inside its time limit)
    and the boxes of the images it draws (None when PDFium could not list them)."""

    image: Image.Image
    lines: tuple[tuple[TextChar, ...], ...]
    images: tuple[Box, ...] | None


def _check(condition: bool, what: str) -> None:
    if not condition:
        raise Ended(f"the reader sent {what}")


def refusal(line: object) -> str:
    """The refusal line, when it is one the file checks can give."""
    _check(isinstance(line, str), "a refusal that is not text")
    if line in ingest.REFUSALS:
        return line
    for pattern in _PAGE_LINES:
        found = pattern.fullmatch(line)
        if found and int(found.group(1)) <= ingest.MAX_PDF_PAGES:
            return line
    raise Ended("the reader sent an unknown refusal")


def _number(value: object) -> float:
    _check(type(value) is float or type(value) is int, "a value that is not a number")
    return value


def page_sizes(reply: dict) -> tuple[tuple[int, int], ...]:
    """Pixel sizes of the pages at RENDER_DPI, from the page sizes in points."""
    sizes, count = reply.get("sizes"), reply.get("count")
    _check(isinstance(sizes, list) and len(sizes) <= ingest.MAX_PDF_PAGES, "a bad page list")
    _check(count == len(sizes), "a page count that does not match")
    longest = ingest.MAX_PIXELS * 72 / ingest.RENDER_DPI  # points; bounds the arithmetic below
    pixels = []
    for size in sizes:
        _check(isinstance(size, list) and len(size) == 2, "a bad page size")
        width, height = (_number(v) for v in size)
        _check(all(0 <= v <= longest for v in (width, height)), "a bad page size")
        pixels.append(ingest.render_size(width, height))
        _check(pixels[-1][0] * pixels[-1][1] <= ingest.MAX_PIXELS, "a page over the pixel cap")
    return tuple(pixels)


def frame_count(reply: dict) -> int:
    count = reply.get("count")
    _check(type(count) is int and 1 <= count <= ingest.MAX_PDF_PAGES, "a bad frame count")
    return count


def image_size(reply: dict, mode: str) -> tuple[int, int]:
    """The (width, height) of a reply that carries pixels in `mode`."""
    size = reply.get("size")
    _check(reply.get("mode") == mode, "pixels in another mode")
    _check(isinstance(size, list) and len(size) == 2, "a bad image size")
    width, height = size
    _check(type(width) is int and type(height) is int, "a bad image size")
    _check(all(0 <= v <= ingest.MAX_PIXELS for v in size), "a bad image size")
    _check(width * height <= ingest.MAX_PIXELS, "a bad image size")
    return width, height


def byte_count(size: tuple[int, int], mode: str) -> int:
    return size[0] * size[1] * _BANDS[mode]


def frame_mode(reply: dict) -> str:
    mode = reply.get("mode")
    _check(isinstance(mode, str) and mode in _BANDS, "a frame in another mode")
    return mode


def frame_info(reply: dict, mode: str) -> dict:
    """The frame's ICC profile and transparency colour, which a served PNG keeps."""
    info = {}
    icc, transparency = reply.get("icc_profile"), reply.get("transparency")
    if icc is not None:
        _check(isinstance(icc, str), "a bad colour profile")
        try:
            info["icc_profile"] = base64.b64decode(icc, validate=True)
        except (binascii.Error, ValueError) as err:
            raise Ended("the reader sent a bad colour profile") from err
    if transparency is not None:
        bands = _BANDS[mode]
        values = transparency if isinstance(transparency, list) else [transparency]
        _check(
            len(values) == bands and all(type(v) is int and 0 <= v <= 65535 for v in values),
            "a bad colour",
        )
        info["transparency"] = tuple(values) if bands > 1 else values[0]
    return info


def _box(value: object) -> Box:
    _check(isinstance(value, list) and len(value) == 4, "a bad box")
    _check(all(type(v) is float and math.isfinite(v) for v in value), "a bad box")
    return tuple(value)


def _optional_box(value: object) -> Box | None:
    return None if value is None else _box(value)


def _char(value: object) -> TextChar:
    _check(isinstance(value, list) and len(value) == 5, "a bad character")
    code, box, unmapped, tight, invisible = value
    _check(type(code) is int and 0 <= code <= _MAX_CODE_POINT, "a bad character")
    _check(type(unmapped) is bool and type(invisible) is bool, "a bad character")
    return TextChar(chr(code), _optional_box(box), unmapped, _optional_box(tight), invisible)


def text_lines(reply: dict) -> tuple[tuple[TextChar, ...], ...]:
    """The page's lines, each sent as positions in the page's character list."""
    chars, lines = reply.get("chars"), reply.get("lines")
    _check(isinstance(chars, list) and isinstance(lines, list), "a bad text layer")
    built = [_char(c) for c in chars]
    out = []
    for line in lines:
        _check(isinstance(line, list), "a bad line")
        _check(all(type(i) is int and 0 <= i < len(built) for i in line), "a bad line")
        segment = tuple(built[i] for i in line)
        _check(line_box(segment) is not None, "a line without a box")
        out.append(segment)
    return tuple(out)


def image_boxes(reply: dict) -> tuple[Box, ...] | None:
    boxes = reply.get("images")
    if boxes is None:
        return None
    _check(isinstance(boxes, list), "a bad image list")
    return tuple(_box(b) for b in boxes)
