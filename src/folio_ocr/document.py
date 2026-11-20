"""The document model: a source file read into pages of text lines."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# x0, y0, x1, y1 in pixels of the straightened page image; x1 and y1 are exclusive.
Box = tuple[float, float, float, float]
Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
Languages = Literal["en", "ar", "en+ar"]


@dataclass(frozen=True)
class TextChar:
    text: str  # one character (a surrogate pair from pdfium is combined)
    box: Box | None  # advance width x font height; None for characters pdfium generated itself
    unmapped: bool = False  # pdfium could not map this glyph to Unicode
    tight_box: Box | None = None  # the glyph's own outline bounds
    invisible: bool = False  # drawn with text render mode 3 (invisible) or 7 (clip only)


@dataclass(frozen=True)
class Source:
    filename: str
    sha256: str
    kind: Literal["pdf", "image"]
    page_count: int


@dataclass(frozen=True)
class Calibration:
    setup_key: str  # the OCR reader setup that judged the lines (folio_ocr.calibration.Setup.key)
    tau: float | None  # its frozen confidence threshold; None when none is recorded


@dataclass(frozen=True)
class Engine:
    detector: str
    weights_sha256: str | None
    reader: str
    model_name: str | None
    model_host: str | None
    model_fit: Literal["pass", "fail", "untested"]
    prompt_sha256: str | None
    calibration: Calibration | None = None


@dataclass(frozen=True)
class Line:
    id: str
    box: Box
    text: str
    direction: Literal["ltr", "rtl"]
    script: Literal["latin", "arabic", "mixed", "unknown"]
    reader: Literal["text_layer", "onnxtr", "tesseract", "model", "none"]  # none: nobody read it
    confidence: float | None
    confidence_method: Literal["logprobs", "two_read", "reader", "none"]
    flags: tuple[str, ...]
    status: Literal["auto_accepted", "needs_review", "edited", "confirmed"]
    final_text: str


@dataclass(frozen=True)
class Page:
    index: int
    width: int
    height: int
    render_dpi: int | None
    rotation_applied: float
    transform: Matrix  # post-EXIF frame -> straightened page image frame
    origin: Literal["text_layer", "ocr"]
    direction: Literal["ltr", "rtl"]
    status: Literal["unread", "reading", "read", "failed"]
    flags: tuple[str, ...]
    unread_regions: tuple[Box, ...]
    lines: tuple[Line, ...]
    # Count of text-layer lines with a content failure when the page went to OCR (0 when it had
    # no text), None otherwise.
    invalid_lines: int | None = None
    error: str | None = None  # one line saying why lines of this page could not be read


@dataclass(frozen=True)
class Document:
    id: str
    source: Source
    languages: Languages
    engine: Engine
    pages: tuple[Page, ...]
    created_at: datetime
    has_edits: bool


def line_id(page: int, k: int) -> str:
    """Identifier of line `k` on page `page`, both counted from 1."""
    if page < 1 or k < 1:
        raise ValueError("page and line numbers start at 1")
    return f"p{page}-l{k}"
