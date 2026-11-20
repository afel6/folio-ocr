"""The fixture and crafted images decode and are served exactly as pinned in
tests/data/image_pins.json."""

import json

import image_pins as ipins
import pdf_pins as pins
import pytest

from folio_ocr import calibration

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
    _skip_other_build()
    data = INPUTS[name]
    got = pins.served_pngs(data, name, ipins.frame_count(data))
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


def test_crafted_images_cover_info_keys():
    _skip_other_build()
    empty = pins.sha(repr({}).encode())
    for key in ("transparency", "icc_profile"):
        carriers = [
            n for n, d in INPUTS.items()
            if any(key in ipins.info_keys(d, k) for k in range(1, ipins.frame_count(d) + 1))
        ]  # fmt: skip
        assert carriers, key
        for n in carriers:
            assert all(f["info"] != empty for f in PINNED["files"][n]["frame"].values()), n
