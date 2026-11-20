"""One page at a time: `read_page` gives the page `read_bytes` gives, and `page_image` the
picture its boxes are drawn on."""

import io
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageOps
from sample_files import save

from folio_ocr import pipeline
from folio_ocr.errors import InputError, WeightsError

FIXTURES = Path(__file__).parent / "fixtures"


def two_page_pdf() -> bytes:
    """en-invoice.pdf then ar-invoice.pdf, both kept from their text layer."""
    doc = pdfium.PdfDocument.new()
    for name in ("en-invoice.pdf", "ar-invoice.pdf"):
        src = pdfium.PdfDocument(str(FIXTURES / name))
        doc.import_pages(src)
        src.close()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture(scope="module")
def pdf() -> bytes:
    return two_page_pdf()


@pytest.mark.parametrize("n", [1, 2])
def test_read_page_equals_the_page_of_the_whole_read(pdf, n):
    whole = pipeline.read_bytes(pdf, "two.pdf", languages="en+ar")
    page, engine = pipeline.read_page(pdf, n, languages="en+ar")
    assert page == whole.pages[n - 1]
    assert page.lines  # a page with text, so the comparison means something
    assert engine == whole.engine


@pytest.mark.parametrize("n", [1, 2])
def test_page_image_is_the_frame_of_the_boxes(pdf, n):
    page, _ = pipeline.read_page(pdf, n, languages="en+ar")
    image = pipeline.page_image(pdf, n)
    assert image.size == (page.width, page.height)
    assert image.mode == "L"


def test_page_image_of_a_photo_has_its_exif_turn_applied():
    data = (FIXTURES / "en-invoice-photo-exif.jpg").read_bytes()
    with Image.open(io.BytesIO(data)) as raw:
        expected = ImageOps.exif_transpose(raw).size
        assert expected != raw.size  # the fixture is stored sideways
    assert pipeline.page_image(data, 1).size == expected


@pytest.mark.parametrize("n", [0, 3, -1])
def test_pages_outside_the_file_are_refused(pdf, n):
    with pytest.raises(ValueError):
        pipeline.read_page(pdf, n, languages="en")
    with pytest.raises(ValueError):
        pipeline.page_image(pdf, n)


def test_image_pages_outside_the_file_are_refused():
    png = save(Image.new("L", (8, 8), 255), "PNG")
    with pytest.raises(ValueError):
        pipeline.page_image(png, 2)
    with pytest.raises(ValueError):
        pipeline.read_page(png, 2, languages="en", weights_dir=None)


def test_an_ocr_page_without_model_files_fails_as_the_whole_read_does():
    png = save(Image.new("L", (64, 64), 255), "PNG")
    with pytest.raises(WeightsError) as whole:
        pipeline.read_bytes(png, "x.png", languages="en", weights_dir=None)
    with pytest.raises(WeightsError) as one:
        pipeline.read_page(png, 1, languages="en", weights_dir=None)
    assert (type(one.value), str(one.value)) == (type(whole.value), str(whole.value))


def test_bad_languages_are_refused_before_reading(pdf):
    with pytest.raises(InputError, match="languages"):
        pipeline.read_page(pdf, 1, languages="fr")


# --- PDFium is not thread-safe: every call runs under one lock --------------------------


def spy_pdfium(monkeypatch):
    from folio_ocr.ingest import PDFIUM_LOCK

    calls = []
    open_doc, render, length = pdfium.PdfDocument, pdfium.PdfPage.render, pdfium.PdfDocument.__len__

    def open_spy(*args, **kwargs):
        calls.append(PDFIUM_LOCK.locked())
        return open_doc(*args, **kwargs)

    def render_spy(self, *args, **kwargs):
        calls.append(PDFIUM_LOCK.locked())
        return render(self, *args, **kwargs)

    def len_spy(self):
        calls.append(PDFIUM_LOCK.locked())
        return length(self)

    monkeypatch.setattr(pdfium, "PdfDocument", open_spy)
    monkeypatch.setattr(pdfium.PdfPage, "render", render_spy)
    monkeypatch.setattr(open_doc, "__len__", len_spy)
    return calls, PDFIUM_LOCK


def test_read_page_runs_pdfium_under_the_lock(monkeypatch):
    calls, lock = spy_pdfium(monkeypatch)
    pipeline.read_page((FIXTURES / "en-invoice.pdf").read_bytes(), 1, languages="en")
    assert len(calls) >= 2 and all(calls) and not lock.locked()


def test_page_image_runs_pdfium_under_the_lock(monkeypatch):
    calls, lock = spy_pdfium(monkeypatch)
    pipeline.page_image((FIXTURES / "en-invoice.pdf").read_bytes(), 1)
    assert len(calls) >= 3 and all(calls) and not lock.locked()


def test_new_document_runs_pdfium_under_the_lock(monkeypatch):
    from folio_ocr.web.upload import Upload, new_document

    calls, lock = spy_pdfium(monkeypatch)
    new_document(Upload("a.pdf", (FIXTURES / "en-invoice.pdf").read_bytes(), "en"))
    assert len(calls) >= 2 and all(calls) and not lock.locked()


def test_page_image_of_an_unloadable_pdf_page_is_an_input_error(pdf, monkeypatch):
    def broken(self, index):
        raise pdfium.PdfiumError("no page")

    monkeypatch.setattr(pdfium.PdfDocument, "__getitem__", broken)
    with pytest.raises(InputError, match="page 1 of the PDF could not be read"):
        pipeline.page_image(pdf, 1)
