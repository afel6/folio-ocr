"""A PDF mixing a text-layer page and a scanned page is read page by page: the text page from
its text layer, the scanned page flagged for a model server when no reader for its language is
set up."""

import os
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from folio_ocr import ocr_setup
from folio_ocr.config import Settings
from folio_ocr.errors import SetupNeededError
from folio_ocr.pipeline import read_path

FIXTURES = Path(__file__).parent / "fixtures"
WEIGHTS = Settings.from_env(os.environ).weights_dir

if ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"):
    pytest.skip("the OCR model files are not set up (folio setup)", allow_module_level=True)


def read(path, **kwargs):
    return read_path(path, languages="en+ar", **kwargs)


@pytest.fixture(scope="module")
def mixed_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mixed") / "mixed.pdf"
    dst = pdfium.PdfDocument.new()
    for name in ("en-letter.pdf", "ar-invoice-scanned.pdf"):
        src = pdfium.PdfDocument(FIXTURES / name)
        dst.import_pages(src)
        src.close()
    dst.save(path)
    dst.close()
    return path


@pytest.fixture(scope="module")
def mixed(mixed_path):
    return read(mixed_path, weights_dir=WEIGHTS, tesseract_path=None)


def line_key(line):
    return (line.id, line.text, line.box, line.flags, line.status)


def scanned_key(line):
    fields = (line.text, line.box, line.flags, line.status)
    return (*fields, line.reader, line.confidence, line.confidence_method)


def test_pages_come_from_the_text_layer_and_ocr(mixed):
    assert [p.origin for p in mixed.pages] == ["text_layer", "ocr"]
    assert [p.index for p in mixed.pages] == [1, 2]


def test_text_page_matches_reading_that_page_alone(mixed):
    alone = read(FIXTURES / "en-letter.pdf", weights_dir=WEIGHTS)
    assert mixed.pages[0].lines
    assert [line_key(x) for x in mixed.pages[0].lines] == [
        line_key(x) for x in alone.pages[0].lines
    ]


def test_scanned_page_is_flagged_for_a_model_server(mixed):
    alone = read(FIXTURES / "ar-invoice-scanned.pdf", weights_dir=WEIGHTS)
    lines = mixed.pages[1].lines
    page_wide = [f for f in ("orientation_uncertain",) if f in mixed.pages[1].flags]
    assert lines
    assert [x.id for x in lines] == [f"p2-l{k}" for k in range(1, len(lines) + 1)]
    assert len(lines) == len(alone.pages[0].lines)
    for line, other in zip(lines, alone.pages[0].lines, strict=True):
        assert scanned_key(line) == scanned_key(other)
        assert line.text == ""
        assert line.reader == "none"
        assert line.confidence is None
        assert line.confidence_method == "none"
        assert line.status == "needs_review"
        assert line.flags == ("needs_model_server", *page_wide)


def test_without_model_files_the_scanned_page_cannot_be_read(mixed_path):
    with pytest.raises(SetupNeededError):
        read(mixed_path, weights_dir=None)
