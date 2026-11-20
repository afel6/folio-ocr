"""The folio process never calls PDFium and never decodes an image: a user's file is opened
only in the reader process, so a crash or a runaway there cannot reach folio."""

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from folio_ocr import pipeline
from folio_ocr.web.upload import Upload, new_document

pytestmark = pytest.mark.usefixtures("ocr_stub")

FIXTURES = Path(__file__).parent / "fixtures"
PDF = (FIXTURES / "en-invoice.pdf").read_bytes()
PNG = (FIXTURES / "en-invoice-clean.png").read_bytes()


@pytest.fixture(autouse=True)
def folio_cannot_open_files(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the folio process opened a user's file")

    monkeypatch.setattr(pdfium, "PdfDocument", boom)
    monkeypatch.setattr(Image, "open", boom)


def test_read_bytes_of_a_pdf_with_a_text_layer():
    doc = pipeline.read_bytes(PDF, "a.pdf", languages="en")
    assert doc.pages[0].origin == "text_layer"


def test_read_bytes_of_a_png():
    doc = pipeline.read_bytes(PNG, "a.png", languages="en")
    assert len(doc.pages) == 1


def test_read_page_of_a_pdf_page():
    page, _engine = pipeline.read_page(PDF, 1, languages="en")
    assert page.index == 1


def test_page_image_of_a_pdf_page():
    assert pipeline.page_image(PDF, 1).mode == "L"


def test_page_image_of_an_image_frame():
    assert pipeline.page_image(PNG, 1).size[0] > 0


def test_new_document_of_a_pdf():
    new_document(Upload("a.pdf", PDF, "en"))
