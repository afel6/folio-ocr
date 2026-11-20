"""Builders for PDF variants of the fixtures: rotated, offset, with a rectangle or an image."""

import io
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


def open_fixture(name: str) -> pdfium.PdfDocument:
    return pdfium.PdfDocument(str(FIXTURES / name))


def move_content(page: pdfium.PdfPage, rotation: int, dx: float = 0.0, dy: float = 0.0) -> None:
    """Rewrite the page so that with /Rotate `rotation` it displays like the unrotated page,
    shifted in user space by (dx, dy) with the media and crop boxes following."""
    w, h = page.get_size()
    # (a, b, c, d, e, f): new user space of the page content before the shift.
    m = {
        0: (1, 0, 0, 1, 0, 0),
        90: (0, 1, -1, 0, h, 0),
        180: (-1, 0, 0, -1, w, h),
        270: (0, -1, 1, 0, 0, w),
    }[rotation]
    for obj in list(page.get_objects(max_depth=1)):
        a, b, c, d, e, f = obj.get_matrix().get()
        mat = pdfium.PdfMatrix(a, b, c, d, e, f).multiply(pdfium.PdfMatrix(*m))
        obj.set_matrix(mat.translate(dx, dy))
    page.gen_content()
    bw, bh = (h, w) if rotation in (90, 270) else (w, h)
    page.set_mediabox(dx, dy, dx + bw, dy + bh)
    page.set_cropbox(dx, dy, dx + bw, dy + bh)
    page.set_rotation(rotation)


def variant(name: str, rotation: int = 0, dx: float = 0.0, dy: float = 0.0):
    doc = open_fixture(name)
    page = doc[0]
    move_content(page, rotation, dx, dy)
    return doc, page


def variant_bytes(name: str, rotation: int = 0, dx: float = 0.0, dy: float = 0.0) -> bytes:
    doc, _ = variant(name, rotation, dx, dy)
    try:
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    finally:
        doc.close()


def add_rect(page: pdfium.PdfPage, left: float, bottom: float, width: float, height: float) -> None:
    """Draw a filled black rectangle (PDF user space, origin bottom left)."""
    rect = pdfium_raw.FPDFPageObj_CreateNewRect(left, bottom, width, height)
    pdfium_raw.FPDFPageObj_SetFillColor(rect, 0, 0, 0, 255)
    pdfium_raw.FPDFPath_SetDrawMode(rect, 1, 0)
    pdfium_raw.FPDFPage_InsertObject(page.raw, rect)
    page.gen_content()


def add_image(
    pdf: pdfium.PdfDocument,
    page: pdfium.PdfPage,
    pil_image: Image.Image,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    """Draw `pil_image` scaled to width x height points."""
    obj = pdfium.PdfImage.new(pdf)
    obj.set_bitmap(pdfium.PdfBitmap.from_pil(pil_image))
    obj.set_matrix(pdfium.PdfMatrix(width, 0, 0, height, left, bottom))
    page.insert_obj(obj)
    page.gen_content()
