"""How many reader processes a document costs: one per document unless a page goes to OCR."""

import io
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from folio_ocr import pdf_client
from folio_ocr.pipeline import read_bytes, read_page

pytestmark = pytest.mark.usefixtures("ocr_stub")

FIXTURES = Path(__file__).parent / "fixtures"


def joined(*names):
    """One PDF made of the pages of the named fixtures, in order."""
    dst = pdfium.PdfDocument.new()
    for name in names:
        src = pdfium.PdfDocument(FIXTURES / name)
        dst.import_pages(src)
        src.close()
    out = io.BytesIO()
    dst.save(out)
    dst.close()
    return out.getvalue()


@pytest.fixture
def starts(monkeypatch):
    calls = []
    real = pdf_client._Worker.start

    def counted(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(pdf_client._Worker, "start", counted)
    return calls


def test_read_page_closes_its_reader_before_ocr(ocr_stub):
    data = (FIXTURES / "ar-invoice-scanned.pdf").read_bytes()
    page, _ = read_page(data, 1, languages="ar")
    assert page.origin == "ocr"
    assert [call.workers_live for call in ocr_stub.pages] == [0]


def test_a_one_page_scanned_pdf_starts_one_reader(starts):
    data = (FIXTURES / "ar-invoice-scanned.pdf").read_bytes()
    doc = read_bytes(data, "s.pdf", languages="ar")
    assert doc.pages[0].origin == "ocr"
    assert len(starts) == 1


def test_a_two_page_pdf_read_by_ocr_starts_two_readers(starts):
    data = joined("ar-invoice-scanned.pdf", "ar-invoice-scanned.pdf")
    doc = read_bytes(data, "s.pdf", languages="ar")
    assert [p.origin for p in doc.pages] == ["ocr", "ocr"]
    assert len(starts) == 2


def test_a_text_layer_pdf_of_three_pages_starts_one_reader(starts):
    data = joined("en-invoice.pdf", "en-invoice.pdf", "en-invoice.pdf")
    doc = read_bytes(data, "t.pdf", languages="en")
    assert len(doc.pages) == 3
    assert len(starts) == 1
