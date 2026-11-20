"""Fixture and crafted image pixels survive preview encoding and match pins on the reference build."""

import json
from io import BytesIO

import image_pins as ipins
import pdf_pins as pins
import pytest
from PIL import Image

from folio_ocr import calibration, pipeline

PINNED = json.loads(ipins.PIN_FILE.read_text(encoding="utf-8"))
INPUTS = ipins.inputs()
IDS = list(INPUTS)
DOC_NAMES = ("en-invoice-clean.png", "en-letter-photo-exif.jpg", "en-invoice-scan.jpg")


def _skip_other_build() -> None:
    if pins.build_info() != PINNED["build"]:
        pytest.skip(f"pins were made on {PINNED['build']}; this build is {pins.build_info()}")


@pytest.fixture
def empty_taus():
    """The Document pins were made with an empty confidence threshold table: run under it."""
    assert PINNED["ocr_env"]["taus"] == calibration.table_sha256({})
    with pins.empty_taus():
        yield


def test_pins_cover_every_input():
    assert sorted(PINNED["files"]) == sorted(IDS)


@pytest.mark.parametrize("name", IDS)
def test_image_frames_match_pins(name):
    _skip_other_build()
    data = INPUTS[name]
    assert ipins.frames(data, ipins.frame_count(data)) == PINNED["files"][name]["frame"]


@pytest.mark.parametrize("name", IDS)
def test_image_served_png_match_pins(name):
    data = INPUTS[name]
    got = pins.served_pngs(data, name, ipins.frame_count(data))
    if pins.build_info() == PINNED["build"]:
        assert got == PINNED["files"][name]["png"]


@pytest.mark.parametrize("name", DOC_NAMES)
def test_image_documents_match_pins(name, empty_taus):
    if not pins.ocr_env_ready():
        pytest.skip("the OCR models or Tesseract are not set up")
    _skip_other_build()
    if pins.tesseract_decided() != PINNED["ocr_env"]["tesseract"]:
        pytest.skip("pins were made with another Tesseract")
    got, weights = pins.document_pin(INPUTS[name], name, "en")
    if weights is not None and weights != PINNED["ocr_env"]["weights"]["en"]:
        pytest.skip("pins were made with other OCR weights")
    assert got == PINNED["files"][name]["documents"]["en"]


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("crafted-l-trns.png", "transparency"),
        ("crafted-rgb-trns.png", "transparency"),
        ("crafted-icc.jpg", "icc_profile"),
    ],
)
def test_worker_keeps_crafted_preview_metadata(name, key):
    data = INPUTS[name]
    with Image.open(BytesIO(data)) as source, pipeline.page_image(data, 1) as frame:
        assert frame.info[key] == source.info[key]
