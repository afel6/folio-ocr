"""Generators for PDFs that are small as files and costly to read: a huge page image, a huge
/SMask packed in an object stream, a content stream of 400 MB and a Form XObject fan-out.
Each is deterministic, built in memory and at most 25 MB."""

import struct
import zlib

from sample_files import _DRAW, _image_stream, _page_tree, _write_objects, pdf_stream, raw_pdf

_CHUNK = 4 * 2**20


def flate_repeated(unit: bytes, total: int) -> bytes:
    """`unit` repeated up to `total` bytes (a whole number of units), Flate-compressed in
    chunks so that building it never holds more than a few MB."""
    packer = zlib.compressobj(1)
    block = unit * max(1, _CHUNK // len(unit))
    parts = [packer.compress(block) for _ in range(total // len(block))]
    parts.append(packer.compress(unit * ((total % len(block)) // len(unit))))
    return b"".join(parts) + packer.flush()


def _gray_image(side: int, extra: bytes = b"") -> bytes:
    entries = b"/Type /XObject /Subtype /Image /ColorSpace /DeviceGray /BitsPerComponent 8 "
    size = b"/Width %d /Height %d" % (side, side)
    data = flate_repeated(b"\0" * side, side * side)
    return pdf_stream(entries + size + extra + b" /Filter /FlateDecode", data)


def page_image_bomb(side: int = 40000) -> bytes:
    """One page drawing a `side` x `side` gray image of zeros."""
    return raw_pdf(
        *_page_tree(b"<< /XObject << /Im0 5 0 R >> >>", b""),
        pdf_stream(b"", _DRAW),
        _gray_image(side),
    )


def packed_smask_bomb(side: int = 40000) -> bytes:
    """A 10 x 10 image whose /SMask (object 6) is a `side` x `side` stream that sits inside a
    compressed object stream (object 7), found through a type-2 cross-reference entry."""
    bodies = [
        *_page_tree(b"<< /XObject << /Im0 5 0 R >> >>", b""),
        pdf_stream(b"", _DRAW),
        _image_stream(b"/Width 10 /Height 10", (10, 10), b" /SMask 6 0 R"),
    ]
    out = bytearray(b"%PDF-1.5\n")
    offsets = _write_objects(out, bodies)
    packed = b"6 0 " + _gray_image(side)
    first = 4
    offsets.append(len(out))
    members = zlib.compress(packed)
    out += b"7 0 obj\n" + pdf_stream(
        b"/Type /ObjStm /N 1 /First %d /Filter /FlateDecode" % first, members
    )
    out += b"\nendobj\n"
    xref_at = len(out)
    rows = [(0, 0, 65535)] + [(1, o, 0) for o in offsets[:5]] + [(2, 7, 0), (1, offsets[5], 0)]
    rows.append((1, xref_at, 0))
    table = b"".join(struct.pack(">BIH", *row) for row in rows)
    out += b"8 0 obj\n<< /Type /XRef /Size 9 /W [1 4 2] /Root 1 0 R /Length %d >>\n" % len(table)
    out += b"stream\n" + table + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(out)


def content_stream_bomb(total: int = 400 * 2**20) -> bytes:
    """A normal A4 page whose Flate content stream inflates to about `total` bytes of drawing."""
    unit = b"0 0 m 1 1 l S\n"
    packed = flate_repeated(unit, total - total % len(unit))
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>"
    return raw_pdf(
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page,
        pdf_stream(b"/Filter /FlateDecode", packed),
    )


def form_fanout_bomb(levels: int = 22) -> bytes:
    """A normal A4 page drawing Form 0; Form k draws Form k + 1 twice, the last one a line."""
    first = 5
    forms = []
    for k in range(levels):
        last = k == levels - 1
        draw = b"0 0 m 1 1 l S" if last else b"/F Do /F Do"
        res = b"" if last else b"/Resources << /XObject << /F %d 0 R >> >>" % (first + k + 1)
        forms.append(pdf_stream(b"/Type /XObject /Subtype /Form /BBox [0 0 595 842] " + res, draw))
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /XObject << /F %d 0 R >> >> /Contents 4 0 R >>" % first
    )
    return raw_pdf(
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page,
        pdf_stream(b"", b"/F Do"),
        *forms,
    )
