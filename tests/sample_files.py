"""Small PDF and image files built in memory for the ingest and pipeline tests."""

import io
import struct
import zlib

import fpdf
import pypdfium2 as pdfium
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
