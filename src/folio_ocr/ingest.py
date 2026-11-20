"""Checks on an uploaded file before any page is read: type by magic bytes, size, page count,
pixel count and a safe display filename. Nothing here renders a PDF page."""

import io
import math
import re
import struct
import threading
import warnings

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image, ImageOps, UnidentifiedImageError

from folio_ocr.errors import InputError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 20  # also the cap on the frames (pages) of a TIFF
MAX_PIXELS = 50_000_000
RENDER_DPI = 300
# PDFium is not thread-safe, not even across documents: hold this lock around every call.
PDFIUM_LOCK = threading.Lock()

_NOT_SUPPORTED = "the file is not a PDF, PNG, JPEG, TIFF or WebP file"
_ENCRYPTED = "the PDF is encrypted; remove its password and try again"
_DAMAGED_PDF = "the PDF could not be opened; it may be damaged"
_IMAGE_TOO_BIG = "the image is too large to read (over 50 megapixels)"
_DECODE = "the image could not be decoded; it may be damaged"

_PILLOW_FORMAT = {"png": "PNG", "jpeg": "JPEG", "tiff": "TIFF", "webp": "WEBP"}
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
_PILLOW_ERRORS = (
    UnidentifiedImageError,
    OSError,
    SyntaxError,
    ValueError,
    EOFError,
    struct.error,
    TypeError,
    IndexError,
    KeyError,
)
# Images over the 50 MP cap are refused anyway, so Pillow's own warning for them is silenced once
# here: a per-call catch_warnings() is not thread-safe and images are read in parallel.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
_UNSAFE_NAME_CHARS = re.compile(
    r'[\x00-\x1f\x7f-\x9f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069\ud800-\udfff<>:"/\\|?*]'
)


def sniff(data: bytes) -> str:
    """Return "pdf", "png", "jpeg", "tiff" or "webp" from the first bytes of the file."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(_TIFF_MAGIC):
        return "tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise InputError(_NOT_SUPPORTED)


def check_size(size: int) -> None:
    if size > MAX_UPLOAD_BYTES:
        raise InputError("the file is larger than 25 MB")


def sanitise_filename(name: str) -> str:
    """A display-safe file name: no folders, control characters or reserved characters."""
    base = re.split(r"[/\\]", name)[-1]
    cleaned = _UNSAFE_NAME_CHARS.sub("_", base)[:255].strip(" .")
    return cleaned or "document"


def render_size(width_pt: float, height_pt: float, dpi: int = RENDER_DPI) -> tuple[int, int]:
    """Pixel size of a page of the given size in points when rendered at `dpi`."""
    scale = dpi / 72  # the same float step pypdfium2 uses, so sizes agree to the pixel
    return math.ceil(width_pt * scale), math.ceil(height_pt * scale)


def _check_pdf(doc: pdfium.PdfDocument) -> None:
    if pdfium_raw.FPDF_GetSecurityHandlerRevision(doc.raw) != -1:
        raise InputError(_ENCRYPTED)
    if len(doc) > MAX_PDF_PAGES:
        raise InputError(f"the PDF has more than {MAX_PDF_PAGES} pages")
    for i in range(len(doc)):
        try:
            width, height = render_size(*doc.get_page_size(i))
        except pdfium.PdfiumError as err:
            raise InputError(_DAMAGED_PDF) from err
        if width * height > MAX_PIXELS:
            raise InputError(f"page {i + 1} is too large to read (over 50 megapixels at 300 dpi)")


def open_pdf(data: bytes) -> pdfium.PdfDocument:
    """Open a PDF after checking encryption, page count and page sizes. The caller closes it."""
    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as err:
        if err.err_code in (pdfium_raw.FPDF_ERR_PASSWORD, pdfium_raw.FPDF_ERR_SECURITY):
            raise InputError(_ENCRYPTED) from err
        raise InputError(_DAMAGED_PDF) from err
    try:
        _check_pdf(doc)
    except InputError:
        doc.close()
        raise
    return doc


def _check_frames(image: Image.Image, kind: str) -> None:
    """Check the header size of every page frame before any pixel is decoded. TIFF frames
    are pages and count against the page cap; other formats are read as one picture."""
    index = 0
    while True:
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise InputError(_IMAGE_TOO_BIG)
        if kind != "tiff":
            break
        try:
            image.seek(index + 1)
        except EOFError:
            break
        index += 1
        if index >= MAX_PDF_PAGES:
            raise InputError(f"the TIFF has more than {MAX_PDF_PAGES} pages")
    image.seek(0)


def open_image(data: bytes, kind: str) -> Image.Image:
    """Open an image lazily after checking the pixel count of every page frame."""
    try:
        image = Image.open(io.BytesIO(data), formats=[_PILLOW_FORMAT[kind]])
        _check_frames(image, kind)
    except Image.DecompressionBombError as err:
        raise InputError(_IMAGE_TOO_BIG) from err
    except _PILLOW_ERRORS as err:
        raise InputError(_DECODE) from err
    return image


def frame_count(image: Image.Image) -> int:
    """Pages in the image: the frames of a TIFF; every other format is one page."""
    if image.format == "TIFF":
        return getattr(image, "n_frames", 1)
    return 1


def load_frame(image: Image.Image, index: int) -> Image.Image:
    """Decode one frame, apply its EXIF orientation and return an 8-bit gray (L) or RGB copy
    that owns its pixels (see `page_pixels`)."""
    try:
        image.seek(index)
        image.load()
        frame = ImageOps.exif_transpose(image)
        return page_pixels(image.copy() if frame is image else frame)
    except _PILLOW_ERRORS as err:
        raise InputError(_DECODE) from err


_ALPHA_MODES = frozenset({"RGBA", "RGBa", "LA", "La", "PA"})


def page_pixels(frame: Image.Image) -> Image.Image:
    """`frame` as a page looks on white paper, in mode L or RGB: transparent parts are laid
    over white, 16- and 32-bit gray is scaled to 8 bits, every other mode is converted.
    L and RGB frames are returned unchanged."""
    if frame.mode in ("L", "RGB"):
        return frame
    if frame.mode == "1":
        return frame.convert("L")
    if frame.mode in ("I", "F") or frame.mode.startswith("I;16"):
        values = np.asarray(frame, dtype=np.float64)
        sixteen_bit = frame.mode.startswith("I;16") or values.max(initial=0.0) > 255
        top = 65535.0 if sixteen_bit else 255.0
        return Image.fromarray(np.clip(values * (255.0 / top), 0, 255).astype(np.uint8), "L")
    if frame.mode in _ALPHA_MODES or "transparency" in frame.info:
        rgba = frame.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return frame.convert("RGB")
