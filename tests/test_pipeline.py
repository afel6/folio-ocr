import errno
import hashlib
import io
import json
import math
import pathlib
import re
import socket
from datetime import timedelta
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
import pytest
from PIL import Image
from sample_files import encrypted, pdf_bytes, png_header, tiff

from folio_ocr import document, ingest
from folio_ocr.document import Engine, TextChar
from folio_ocr.errors import InputError
from folio_ocr.ingest import MAX_UPLOAD_BYTES
from folio_ocr.pipeline import read_bytes, read_path
from folio_ocr.text_layer import image_boxes
from folio_ocr.text_layer_check import check_text_layer
from folio_ocr.text_layer_page import text_layer_line

pytestmark = pytest.mark.usefixtures("ocr_stub")

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
IMAGES = [f["file"] for f in MANIFEST if f["file"].endswith((".png", ".jpg"))]
BORN_DIGITAL = [
    f"{lang}-{kind}" for lang in ("en", "ar", "mixed") for kind in ("invoice", "letter")
]

NOT_SUPPORTED = "the file is not a PDF, PNG, JPEG, TIFF or WebP file"
TOO_BIG = "the file is larger than 25 MB"
LANGS = "languages must be en, ar or en+ar"
PIXELS_PAGE = "page 1 is too large to read (over 50 megapixels at 300 dpi)"
IMG_BIG = "the image is too large to read (over 50 megapixels)"
DAMAGED = "the PDF could not be opened; it may be damaged"
PAGE_DAMAGED = "page 1 of the PDF could not be read; it may be damaged"
ENCRYPTED = "the PDF is encrypted; remove its password and try again"


def _gt(name):
    return json.loads((FIXTURES / f"{name}.gt.json").read_text(encoding="utf-8"))


def _refused(data, message, languages="en", name="x.pdf"):
    with pytest.raises(InputError) as info:
        read_bytes(data, name, languages=languages)
    assert str(info.value) == message


@pytest.fixture(scope="module")
def reads(ocr_stub_module):
    out = {}
    for name in BORN_DIGITAL:
        gt = _gt(name)
        out[name] = read_path(FIXTURES / f"{name}.pdf", languages=gt["language"])
    return out


@pytest.fixture(scope="module")
def scanned(ocr_stub_module):
    return read_path(FIXTURES / "ar-invoice-scanned.pdf", languages="ar")


# born-digital pages


@pytest.mark.parametrize("name", BORN_DIGITAL)
def test_born_digital_page_fields(reads, name):
    doc = reads[name]
    gt = _gt(name)
    assert doc.source.kind == "pdf"
    assert doc.source.page_count == len(doc.pages) == 1
    page = doc.pages[0]
    assert page.index == 1
    assert page.origin == "text_layer"
    assert page.status == "read"
    assert page.render_dpi == 300
    assert page.rotation_applied == 0.0
    assert page.transform == document.IDENTITY
    assert page.flags == ()
    assert page.unread_regions == ()
    assert page.invalid_lines is None
    assert (page.width, page.height) == (gt["width"], gt["height"])


@pytest.mark.parametrize("name", BORN_DIGITAL)
def test_born_digital_lines(reads, name):
    lines = reads[name].pages[0].lines
    assert len(lines) > 0
    assert [ln.id for ln in lines] == [f"p1-l{k}" for k in range(1, len(lines) + 1)]
    for ln in lines:
        assert ln.reader == "text_layer"
        assert ln.confidence is None
        assert ln.confidence_method == "none"
        assert ln.final_text == ln.text
        assert (ln.status == "needs_review") == bool(ln.flags)


def test_en_invoice_all_lines_ltr(reads):
    assert {ln.direction for ln in reads["en-invoice"].pages[0].lines} == {"ltr"}


def test_ar_letter_arabic_lines_rtl(reads):
    arabic = [ln for ln in reads["ar-letter"].pages[0].lines if ln.script == "arabic"]
    assert arabic
    assert all(ln.direction == "rtl" for ln in arabic)


def test_mixed_letter_has_both_scripts(reads):
    scripts = {ln.script for ln in reads["mixed-letter"].pages[0].lines}
    assert "mixed" in scripts or {"latin", "arabic"} <= scripts


# routing


def test_scanned_pdf_goes_to_ocr(scanned):
    page = scanned.pages[0]
    assert page.origin == "ocr"
    assert page.status == "unread"
    assert page.flags == ("empty_with_ink", "unread_regions")
    assert page.invalid_lines == 0
    assert page.lines == ()
    assert page.unread_regions == ((0.0, 0.0, float(page.width), float(page.height)),)
    assert page.render_dpi == 300


def test_broken_unmapped_goes_to_ocr():
    page = read_path(FIXTURES / "ar-letter-broken_unmapped.pdf", languages="ar").pages[0]
    assert page.origin == "ocr"
    assert page.flags == ("text_layer_invalid", "unread_regions")
    assert page.invalid_lines == 13
    assert page.lines == ()


def test_invisible_ocr_keeps_text_layer_and_flags():
    path = FIXTURES / "ar-letter-invisible_ocr.pdf"
    page = read_path(path, languages="ar").pages[0]
    assert page.origin == "text_layer"
    assert page.flags == ("invisible_text", "unread_regions")
    pdf = pdfium.PdfDocument(str(path))
    try:
        pdf_page = pdf[0]
        boxes = image_boxes(pdf_page, 300)
        pdf_page.close()
    finally:
        pdf.close()
    assert boxes
    assert page.unread_regions == boxes
    assert page.lines
    for ln in page.lines:
        assert "text_layer_invalid" in ln.flags
        assert ln.status == "needs_review"


@pytest.mark.parametrize("name", ["en-bulleted", "ar-bulleted"])
def test_bulleted_lines_are_invalid_private_use(name):
    page = read_path(FIXTURES / f"{name}.pdf", languages=_gt(name)["language"]).pages[0]
    assert page.origin == "text_layer"
    bullets = [ln for ln in page.lines if ln.text == ""]
    assert len(bullets) == 5
    assert all("text_layer_invalid" in ln.flags for ln in bullets)
    assert sum("text_layer_invalid" in ln.flags for ln in page.lines) == 5


# images


@pytest.fixture(scope="module")
def image_reads(ocr_stub_module):
    return {
        name: read_path(FIXTURES / name, languages=_gt(name.rsplit(".", 1)[0])["language"])
        for name in IMAGES
    }


@pytest.mark.parametrize("name", IMAGES)
def test_image_page(image_reads, name):
    doc = image_reads[name]
    gt = _gt(name.rsplit(".", 1)[0])
    assert doc.source.kind == "image"
    assert doc.source.page_count == 1
    page = doc.pages[0]
    assert page.origin == "ocr"
    assert page.status == "unread"
    assert page.render_dpi is None
    assert page.flags == ("unread_regions",)
    assert page.unread_regions == ((0.0, 0.0, float(gt["width"]), float(gt["height"])),)
    assert (page.width, page.height) == (gt["width"], gt["height"])


def test_exif_fixtures_are_transposed(image_reads):
    swapped = 0
    for name in IMAGES:
        if "-exif" not in name:
            continue
        with Image.open(FIXTURES / name) as img:
            stored = img.size
        page = image_reads[name].pages[0]
        swapped += stored == (page.height, page.width) and stored != (page.width, page.height)
    assert swapped >= 1


# refusals


def test_wrong_magic():
    _refused(b"hello world", NOT_SUPPORTED)


def test_size_checked_before_type():
    _refused(b"%PDF-" + bytes(MAX_UPLOAD_BYTES + 1 - 5), TOO_BIG)
    _refused(bytes(MAX_UPLOAD_BYTES + 1), TOO_BIG)


def test_twenty_one_pages_refused():
    _refused(pdf_bytes([(100, 100)] * 21), "the PDF has more than 20 pages")


def test_owner_password_only_pdf_refused():
    _refused(encrypted(), ENCRYPTED)


def test_huge_page_refused_without_rendering(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("rendered")

    monkeypatch.setattr(pdfium.PdfPage, "render", boom)
    monkeypatch.setattr(pdfium_raw, "FPDF_RenderPageBitmap", boom)
    _refused(pdf_bytes([(14400, 14400)]), PIXELS_PAGE)


def test_crafted_png_header_refused():
    data = png_header(8000, 8000)
    assert len(data) < 100
    _refused(data, IMG_BIG, name="x.png")


def test_garbage_pdf_refused():
    _refused(b"%PDF-1.7\nnot really a pdf at all", DAMAGED)


@pytest.mark.parametrize("languages", ["fr", "EN", ""])
def test_bad_languages(languages):
    _refused(pdf_bytes([(100, 100)]), LANGS, languages=languages)


@pytest.mark.parametrize("languages", ["fr", "EN", ""])
def test_read_path_looks_at_the_file_before_the_languages(tmp_path, languages):
    with pytest.raises(FileNotFoundError):
        read_path(tmp_path / "does-not-exist.pdf", languages=languages)
    target = tmp_path / "ok.pdf"
    target.write_bytes(pdf_bytes([(100, 100)]))
    with pytest.raises(InputError) as info:
        read_path(target, languages=languages)
    assert str(info.value) == LANGS


# read_path


def test_size_refused_from_stat_before_reading(tmp_path, monkeypatch):
    big = tmp_path / "big.pdf"
    with big.open("wb") as f:
        f.truncate(MAX_UPLOAD_BYTES + 1)

    def boom(*_a, **_k):
        raise AssertionError("file was opened")

    monkeypatch.setattr(pathlib.Path, "open", boom)
    with pytest.raises(InputError) as info:
        read_path(big, languages="en")
    assert str(info.value) == TOO_BIG


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_path(tmp_path / "nope.pdf", languages="en")


def test_folder_is_eisdir(tmp_path):
    with pytest.raises(IsADirectoryError) as info:
        read_path(tmp_path, languages="en")
    assert info.value.errno == errno.EISDIR


def test_filename_is_sanitised(tmp_path):
    data = pdf_bytes([(100, 100)])
    try:
        target = tmp_path / "fac<t>ure.pdf"
        target.write_bytes(data)
        expected = "fac_t_ure.pdf"
    except OSError:
        target = tmp_path / "facture .pdf."
        target.write_bytes(data)
        expected = "facture .pdf"
    assert read_path(target, languages="en").source.filename == expected


# no network


def test_no_socket_needed(monkeypatch):
    class Blocked:
        def __init__(self, *_a, **_k):
            raise AssertionError("socket used")

    def boom(*_a, **_k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", Blocked)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(AssertionError):
        socket.socket()
    for name in ("en-invoice.pdf", "en-invoice-clean.png", "ar-invoice-scanned.pdf"):
        language = "ar" if name.startswith("ar") else "en"
        assert read_path(FIXTURES / name, languages=language).pages


# document fields


def test_document_fields():
    data = pdf_bytes([(100, 100)])
    first = read_bytes(data, "../secret/фاتورة.pdf", languages="en+ar")
    second = read_bytes(data, "a.pdf", languages="en+ar")
    assert first.source.sha256 == hashlib.sha256(data).hexdigest()
    assert first.source.filename == "фاتورة.pdf"
    assert re.fullmatch(r"[0-9a-f]{32}", first.id)
    assert first.id != second.id
    assert first.created_at.utcoffset() == timedelta(0)
    assert first.engine == Engine("none", None, "none", None, None, "untested", None)
    assert first.has_edits is False
    assert first.languages == "en+ar"
    assert first.source.kind == "pdf"


def test_arabic_filename_kept():
    name = "../secret/فاتورة.pdf"
    doc = read_bytes(pdf_bytes([(100, 100)]), name, languages="ar")
    assert doc.source.filename == "فاتورة.pdf"


# multi-frame images


def test_tiff_frames_are_pages():
    doc = read_bytes(tiff([(10, 10), (20, 10), (30, 10)]), "s.tiff", languages="en")
    assert doc.source.page_count == 3
    assert [(p.index, p.width, p.height) for p in doc.pages] == [
        (1, 10, 10),
        (2, 20, 10),
        (3, 30, 10),
    ]


def test_animated_webp_is_one_page():
    frames = [Image.new("RGB", (16, 16), c) for c in ("red", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=50)
    doc = read_bytes(buf.getvalue(), "a.webp", languages="en")
    assert doc.source.page_count == 1


# text_layer_line


def _chars(text):
    return [
        TextChar(text=c, box=(10.0 * i, 0.0, 10.0 * i + 10.0, 12.0)) for i, c in enumerate(text)
    ]


def _line(text, languages="en"):
    chars = _chars(text)
    report = check_text_layer([chars], has_ink=True)
    return text_layer_line(1, 1, chars, report.lines[0], languages)


def test_lone_surrogate_becomes_replacement_char():
    line = _line("ab" + chr(0xD800) + "cd")
    assert line.text == "ab�cd"
    assert "invalid_text" in line.flags
    assert line.status == "needs_review"


def test_clean_latin_line_is_accepted():
    line = _line("hello world")
    assert line.flags == ()
    assert line.status == "auto_accepted"
    assert line.script == "latin"
    assert line.direction == "ltr"


def test_presentation_form_flag_appears_once():
    line = _line("abﻻ", "en+ar")
    assert line.flags.count("presentation_forms") == 1


def test_line_flags_never_repeat():
    line = _line("ab" + chr(0xD800) + "ﻻﻻﻻ", "en+ar")
    assert len(set(line.flags)) == len(line.flags)


# damaged pages


def test_page_render_failure(monkeypatch):
    def boom(*_a, **_k):
        raise pdfium.PdfiumError("x")

    monkeypatch.setattr(pdfium.PdfPage, "render", boom)
    _refused((FIXTURES / "en-invoice.pdf").read_bytes(), PAGE_DAMAGED, name="a.pdf")


def test_page_load_failure(monkeypatch):
    def boom(*_a, **_k):
        raise pdfium.PdfiumError("x")

    monkeypatch.setattr(pdfium.PdfDocument, "__getitem__", boom)
    _refused((FIXTURES / "en-invoice.pdf").read_bytes(), PAGE_DAMAGED, name="a.pdf")


# --- PDFium is not thread-safe: every call runs under one lock --------------------------


def test_pdfium_runs_under_the_lock(monkeypatch):
    from folio_ocr.ingest import PDFIUM_LOCK

    calls = []
    open_doc, render = pdfium.PdfDocument, pdfium.PdfPage.render

    def open_spy(*args, **kwargs):
        calls.append(PDFIUM_LOCK.locked())
        return open_doc(*args, **kwargs)

    def render_spy(self, *args, **kwargs):
        calls.append(PDFIUM_LOCK.locked())
        return render(self, *args, **kwargs)

    monkeypatch.setattr(pdfium, "PdfDocument", open_spy)
    monkeypatch.setattr(pdfium.PdfPage, "render", render_spy)
    read_path(FIXTURES / "en-invoice.pdf", languages="en")
    assert calls and all(calls)
    assert not PDFIUM_LOCK.locked()


def test_concurrent_reads_give_the_same_pages():
    from concurrent.futures import ThreadPoolExecutor

    names = [
        "en-invoice.pdf",
        "ar-invoice.pdf",
        "mixed-letter.pdf",
        "ar-invoice-scanned.pdf",
        "en-invoice-clean.png",
        "ar-invoice-photo-exif.jpg",
    ] * 2
    data = {n: (FIXTURES / n).read_bytes() for n in names}
    alone = {n: read_bytes(data[n], n, languages="en+ar").pages for n in set(names)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        docs = list(pool.map(lambda n: read_bytes(data[n], n, languages="en+ar"), names))
    assert [d.pages for d in docs] == [alone[n] for n in names]


def test_rendered_page_image_owns_its_pixels(monkeypatch):
    closed = []
    close = pdfium.PdfBitmap.close

    def close_spy(self):
        closed.append(True)
        return close(self)

    monkeypatch.setattr(pdfium.PdfBitmap, "close", close_spy)
    doc = read_path(FIXTURES / "ar-invoice-scanned.pdf", languages="ar")
    assert closed and doc.pages[0].width > 0


def test_mpo_second_picture_is_not_a_page():
    first, second = Image.new("RGB", (16, 8), "white"), Image.new("RGB", (16, 8), "black")
    buf = io.BytesIO()
    first.save(buf, "MPO", save_all=True, append_images=[second])
    assert Image.open(io.BytesIO(buf.getvalue())).n_frames == 2
    doc = read_bytes(buf.getvalue(), "photo.jpg", languages="en")
    assert doc.source.page_count == 1 and doc.pages[0].width == 16


# page direction and page numbering


def test_page_direction(reads, scanned, image_reads):
    assert reads["ar-letter"].pages[0].direction == "rtl"
    assert reads["en-invoice"].pages[0].direction == "ltr"
    assert image_reads["en-invoice-clean.png"].pages[0].direction == "rtl"
    assert scanned.pages[0].direction == "rtl"


def test_second_text_layer_page_is_numbered_from_two():
    merged = pdfium.PdfDocument.new()
    for name in ("en-invoice.pdf", "ar-invoice.pdf"):
        src = pdfium.PdfDocument(str(FIXTURES / name))
        merged.import_pages(src)
        src.close()
    buf = io.BytesIO()
    merged.save(buf)
    merged.close()
    doc = read_bytes(buf.getvalue(), "both.pdf", languages="en+ar")
    first, second = doc.pages
    assert (first.index, second.index) == (1, 2)
    assert [ln.id for ln in second.lines] == [f"p2-l{k}" for k in range(1, len(second.lines) + 1)]
    assert second.lines and second.direction == "rtl"


# the 50-megapixel cap, to the pixel


def test_image_pixel_cap_boundary():
    assert ingest.open_image(png_header(10000, 5000), "png").size == (10000, 5000)
    with pytest.raises(InputError) as info:
        ingest.open_image(png_header(10000, 5001), "png")
    assert str(info.value) == IMG_BIG


def test_pdf_page_pixel_cap_boundary():
    exact, one_row_more = (2400, 1200), (2400, 1200.24)
    assert math.prod(ingest.render_size(*exact)) == 50_000_000
    assert math.prod(ingest.render_size(*one_row_more)) == 50_000_000 + 10_000
    ingest.open_pdf(pdf_bytes([exact])).close()
    with pytest.raises(InputError) as info:
        ingest.open_pdf(pdf_bytes([one_row_more]))
    assert str(info.value) == PIXELS_PAGE


# --- OCR pages are read one at a time, outside the lock ---------------------------------


def _merged(*names):
    merged = pdfium.PdfDocument.new()
    for name in names:
        src = pdfium.PdfDocument(str(FIXTURES / name))
        merged.import_pages(src)
        src.close()
    buf = io.BytesIO()
    merged.save(buf)
    merged.close()
    return buf.getvalue()


def _trace(monkeypatch):
    """Record "render n" / "ocr n" in order, and whether the PDFium lock was held at each."""
    from folio_ocr import pipeline

    events, held = [], []
    render, ocr_page = pipeline._render, pipeline.ocr_page

    def render_spy(page):
        events.append("render")
        return render(page)

    def ocr_spy(page_no, *args, **kwargs):
        events.append(f"ocr {page_no}")
        held.append(ingest.PDFIUM_LOCK.locked())
        return ocr_page(page_no, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_render", render_spy)
    monkeypatch.setattr(pipeline, "ocr_page", ocr_spy)
    return events, held


def test_each_ocr_page_is_read_before_the_next_page_is_rendered(monkeypatch):
    events, held = _trace(monkeypatch)
    doc = read_bytes(
        _merged("ar-invoice-scanned.pdf", "ar-invoice-scanned.pdf"), "s.pdf", languages="ar"
    )
    assert events == ["render", "ocr 1", "render", "ocr 2"]
    assert held == [False, False] and not ingest.PDFIUM_LOCK.locked()
    assert [p.index for p in doc.pages] == [1, 2]


def test_pages_stay_in_order_when_a_middle_page_keeps_its_text_layer(monkeypatch):
    events, held = _trace(monkeypatch)
    data = _merged("ar-invoice-scanned.pdf", "en-invoice.pdf", "ar-invoice-scanned.pdf")
    doc = read_bytes(data, "m.pdf", languages="en+ar")
    assert [(p.index, p.origin) for p in doc.pages] == [(1, "ocr"), (2, "text_layer"), (3, "ocr")]
    assert events == ["render", "ocr 1", "render", "render", "ocr 3"]
    assert held == [False, False]
