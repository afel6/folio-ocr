"""The fixture PDFs read and render exactly as pinned in tests/data/pdf_pins.json."""

import json

import pdf_pins as pins
import pypdfium2 as pdfium
import pytest

from folio_ocr import calibration, pipeline

PINNED = json.loads(pins.PIN_FILE.read_text(encoding="utf-8"))
PDFS = pins.fixture_pdfs()
IDS = [p.name for p in PDFS]


def _skip_other_build() -> None:
    if pins.build_info() != PINNED["build"]:
        pytest.skip(f"pins were made on {PINNED['build']}; this build is {pins.build_info()}")


@pytest.fixture
def empty_taus():
    """The Document pins were made with an empty confidence threshold table: run under it."""
    assert PINNED["ocr_env"]["taus"] == calibration.table_sha256({})
    with pins.empty_taus():
        yield


def test_pins_cover_every_fixture():
    assert sorted(PINNED["files"]) == sorted(IDS)


@pytest.mark.parametrize("path", PDFS, ids=IDS)
def test_renders_match_pins(path):
    _skip_other_build()
    data = path.read_bytes()
    expected = PINNED["files"][path.name]["renders"]
    assert pins.renders(data, pins.page_count(data)) == expected


@pytest.mark.parametrize("path", PDFS, ids=IDS)
def test_served_png_match_pins(path):
    _skip_other_build()
    data = path.read_bytes()
    expected = PINNED["files"][path.name]["png"]
    assert pins.served_pngs(data, path.name, pins.page_count(data)) == expected


@pytest.mark.parametrize("languages", pins.LANGUAGES)
@pytest.mark.parametrize("path", PDFS, ids=IDS)
def test_documents_match_pins(path, languages, empty_taus):
    if not pins.ocr_env_ready():
        pytest.skip("the OCR models or Tesseract are not set up")
    _skip_other_build()
    if pins.tesseract_decided() != PINNED["ocr_env"]["tesseract"]:
        pytest.skip("pins were made with another Tesseract")
    got, weights = pins.document_pin(path.read_bytes(), path.name, languages)
    if weights is not None and weights != PINNED["ocr_env"]["weights"][languages]:
        pytest.skip("pins were made with other OCR weights")
    want = PINNED["files"][path.name]["documents"][languages]
    assert got == want, (
        f"{path.name} ({languages}): pages now {got['pages']}, pinned {want['pages']}"
    )


def _raw_render(data: bytes, index: int):
    doc = pdfium.PdfDocument(data)
    try:
        page = doc[index]
        try:
            bitmap = page.render(scale=300 / 72, grayscale=True)
            try:
                return bitmap.to_pil()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        doc.close()


@pytest.mark.parametrize("path", PDFS, ids=IDS)
def test_render_equals_raw_pdfium(path):
    data = path.read_bytes()
    for n in range(1, pins.page_count(data) + 1):
        raw, got = _raw_render(data, n - 1), pipeline.page_image(data, n)
        assert (got.mode, got.size, got.tobytes()) == (raw.mode, raw.size, raw.tobytes()), (
            f"{path.name} page {n}"
        )
