"""A page whose image list could not be read: a kept text-layer page is refused, an OCR page
is still read."""

import dataclasses
from pathlib import Path

import pytest

from folio_ocr import ingest, pdf_client, pipeline
from folio_ocr.errors import InputError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def unlisted_images(monkeypatch):
    real = pdf_client.OpenFile.page

    def page(self, index):
        return dataclasses.replace(real(self, index), images=None)

    monkeypatch.setattr(pdf_client.OpenFile, "page", page)


def test_a_text_layer_page_without_an_image_list_is_refused(unlisted_images):
    data = (FIXTURES / "en-invoice.pdf").read_bytes()
    with pytest.raises(InputError) as err:
        pipeline.read_bytes(data, "en-invoice.pdf", languages="en")
    assert str(err.value) == ingest.page_damaged(1)


def test_an_ocr_page_without_an_image_list_is_still_read(unlisted_images, ocr_stub):
    data = (FIXTURES / "ar-invoice-scanned.pdf").read_bytes()
    doc = pipeline.read_bytes(data, "ar-invoice-scanned.pdf", languages="en")
    assert doc.pages[0].origin == "ocr"
