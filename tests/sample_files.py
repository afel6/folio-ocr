"""Small PDF and image files built in memory for the ingest and pipeline tests."""

import io
import struct
import sys
import time
import zlib

import fpdf
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image


def pdf_bytes(sizes):
    """A PDF with one blank page per (width, height) in points."""
    doc = pdfium.PdfDocument.new()
    for w, h in sizes:
        doc.new_page(w, h)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def encrypted(user=""):
    """A one-page PDF with an owner password and the given user password."""
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_encryption(owner_password="o", user_password=user)
    return bytes(pdf.output())


def png_header(width: int, height: int) -> bytes:
    """A tiny PNG whose header claims `width` x `height` pixels but holds no pixel data."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    chunk += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    end = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    return b"\x89PNG\r\n\x1a\n" + chunk + end


def save(img, fmt, **kw):
    """Encode `img` as `fmt` and return the bytes."""
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def tiff(sizes):
    """A multi-frame greyscale TIFF with one frame per (width, height)."""
    first, *rest = [Image.new("L", s) for s in sizes]
    return save(first, "TIFF", save_all=True, append_images=rest)


def _write_objects(out, bodies):
    """Append objects 1.. with the given bodies to `out`; return their byte offsets."""
    offsets = []
    for number, body in enumerate(bodies, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    return offsets


def raw_pdf(*bodies):
    """A PDF whose object i + 1 has body `bodies[i]`, with a correct cross-reference table.
    Object 1 must be the catalog."""
    out = bytearray(b"%PDF-1.7\n")
    offsets = _write_objects(out, bodies)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(bodies) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(bodies) + 1,
        xref,
    )
    return bytes(out)


def pdf_stream(entries, data=b""):
    """The body of a stream object with dictionary entries `entries` and contents `data`."""
    return b"<< %s /Length %d >>\nstream\n%s\nendstream" % (entries, len(data), data)


def zero_rows(width, height):
    """`width` x `height` zero bytes, Flate-compressed one row at a time to keep memory low."""
    packer = zlib.compressobj(9)
    row = bytes(width)
    return b"".join(packer.compress(row) for _ in range(height)) + packer.flush()


_DRAW = b"q 612 0 0 792 0 0 cm /Im0 Do Q"
_FORM = (
    b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /XObject << /Im0 %d 0 R >> >>"
)


def _image_stream(size, pixels, extra=b""):
    """An 8-bit gray image stream whose dictionary states `size`; `pixels` = (width, height)
    fills it with zero pixels, otherwise it holds one byte."""
    entries = b"/Type /XObject /Subtype /Image /ColorSpace /DeviceGray /BitsPerComponent 8 "
    if pixels:
        return pdf_stream(entries + size + extra + b" /Filter /FlateDecode", zero_rows(*pixels))
    return pdf_stream(entries + size + extra, b"\0")


def image_pdf(width, height, *, place="page", mask=None, pixels=False, size=None, objects=()):
    """A one-page 612 x 792 pt PDF that draws one gray image whose dictionary states
    `width` x `height` (or the entries `size`). `place`: "page", "form" (inside a Form
    XObject), "annot" (an annotation's appearance), "pattern" (a tiling pattern) or "inline".
    `mask` = (key, w, h) gives the image a /SMask or /Mask (object 6) of that size and draws a
    10 x 10 image. `pixels` fills the streams with real zero pixels (Flate) instead of one
    byte. `objects` are added as objects 6 on (7 on with a mask)."""
    size = size or b"/Width %d /Height %d" % (width, height)
    if place == "inline":
        head = b"q 612 0 0 792 0 0 cm BI /W %d /H %d /CS /G /BPC 8 /F /Fl ID " % (width, height)
        content = head + zlib.compress(bytes(10)) + b" EI Q"
        return raw_pdf(*_page_tree(b"<< >>", b""), pdf_stream(b"", content))
    extra, objects = b"", list(objects)
    if mask:
        key, mw, mh = mask
        extra, size, (width, height) = (
            b" /%s 6 0 R" % key.encode(),
            b"/Width 10 /Height 10",
            (10, 10),
        )
        objects.insert(0, _image_stream(b"/Width %d /Height %d" % (mw, mh), pixels and (mw, mh)))
    image = _image_stream(size, pixels and (width, height), extra)
    return raw_pdf(*_placed(place, image, objects))


def _page_tree(resources, page_extra):
    return (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources %s /Contents 4 0 R%s >>"
        % (resources, page_extra),
    )


def _placed(place, image, objects):
    """Objects 1-5 draw object 5 (`image`) from `place`; `objects` follow from object 6, then
    any form or annotation the place needs."""
    n = 6 + len(objects)
    if place == "page":
        tree, content, tail = _page_tree(b"<< /XObject << /Im0 5 0 R >> >>", b""), _DRAW, []
    elif place == "form":
        tree = _page_tree(b"<< /XObject << /F0 %d 0 R >> >>" % n, b"")
        content, tail = b"/F0 Do", [pdf_stream(_FORM % 5, _DRAW)]
    elif place == "annot":
        tree, content = _page_tree(b"<< >>", b" /Annots [%d 0 R]" % n), b""
        stamp = b"<< /Type /Annot /Subtype /Stamp /Rect [0 0 612 792] /AP << /N %d 0 R >> >>"
        tail = [stamp % (n + 1), pdf_stream(_FORM % 5, _DRAW)]
    else:  # pattern
        tree = _page_tree(b"<< /Pattern << /P0 %d 0 R >> >>" % n, b"")
        content = b"/Pattern cs /P0 scn 0 0 612 792 re f"
        tiling = b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 612 792] /XStep 612 "
        tiling += b"/YStep 792 /Resources << /XObject << /Im0 5 0 R >> >>"
        tail = [pdf_stream(tiling, _DRAW)]
    return (*tree, pdf_stream(b"", content), image, *objects, *tail)


def objstm_pdf(width_ref=b"/Width 7 0 R /Height 200", packed=b"300"):
    """A PDF 1.5 one-page file with a cross-reference stream whose image stream (object 5)
    states `width_ref`; object 7 (`packed`) lives in an uncompressed object stream (object 6)."""
    tree = _page_tree(b"<< /XObject << /Im0 5 0 R >> >>", b"")
    bodies = [*tree, pdf_stream(b"", _DRAW), _image_stream(width_ref, None)]
    out = bytearray(b"%PDF-1.5\n")
    offsets = _write_objects(out, bodies)
    header = b"7 0 "
    offsets.append(len(out))
    out += b"6 0 obj\n" + pdf_stream(b"/Type /ObjStm /N 1 /First %d" % len(header), header + packed)
    out += b"\nendobj\n"
    xref_at = len(out)
    rows = [(0, 0, 65535)] + [(1, o, 0) for o in offsets] + [(2, 6, 0), (1, xref_at, 0)]
    table = b"".join(struct.pack(">BIH", *row) for row in rows)
    out += b"8 0 obj\n<< /Type /XRef /Size 9 /W [1 4 2] /Root 1 0 R /Length %d >>\n" % len(table)
    out += b"stream\n" + table + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(out)


PAGE_IMAGE_LINE = "page 1 holds an image too large to read (over 50 megapixels)"
PDF_IMAGE_LINE = "the PDF holds an image too large to read (over 50 megapixels)"
UNCHECKED_LINE = "the PDF's images could not be checked safely; it was not read"
OVERSIZED_IMAGE_CASES = [
    (image_pdf(20000, 20000), PAGE_IMAGE_LINE),
    (image_pdf(0, 0, mask=("SMask", 20000, 20000)), PDF_IMAGE_LINE),
    (image_pdf(0, 0, size=b"/Width 9 0 R /Height 200"), UNCHECKED_LINE),
]


def pdfium_image_sizes(data):
    """The pixel sizes PDFium reports for the images on page 1 of the PDF `data`."""
    doc = pdfium.PdfDocument(data)
    page = doc[0]
    try:
        return [o.get_px_size() for o in page.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,))]
    finally:
        page.close()
        doc.close()


def assert_within_a_second(started):
    """A read that began at perf_counter() `started` ended within a second. A tracer
    (coverage, a debugger) slows Python several-fold, so the limit holds only without one."""
    assert sys.gettrace() is not None or time.perf_counter() - started < 1.0
