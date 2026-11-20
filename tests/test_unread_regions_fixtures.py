"""Unread regions on the built-in image fixtures: a photo's dark surround is not an unread area,
and clean and scanned pages keep the areas they had (the ink nothing read)."""

import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from folio_ocr import ocr, ocr_setup
from folio_ocr.config import Settings
from folio_ocr.metrics import gt_line_boxes, match_lines
from folio_ocr.pipeline import read_path

FIXTURES = Path(__file__).parent / "fixtures"
PINNED = Path(__file__).parent / "data" / "unread_regions_clean_scan.json"
WEIGHTS = Settings.from_env(os.environ).weights_dir
if ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"):
    pytest.skip("the OCR model files are not set up (folio setup)", allow_module_level=True)

DOCUMENTS = [f"{t}-{layout}" for t in ("en", "ar", "mixed") for layout in ("invoice", "letter")]
PHOTOS = [f"{document}-{kind}.jpg" for document in DOCUMENTS for kind in ("photo", "photo-exif")]


def _page(path: Path):
    return read_path(path, languages="en", weights_dir=WEIGHTS).pages[0]


@pytest.mark.parametrize("name", PHOTOS)
def test_no_photo_has_an_unread_region_over_half_the_page(name):
    page = _page(FIXTURES / name)
    for x0, y0, x1, y1 in page.unread_regions:
        assert (x1 - x0) * (y1 - y0) < 0.5 * page.width * page.height


@pytest.mark.parametrize("name", sorted(json.loads(PINNED.read_text("utf-8"))))
def test_clean_and_scanned_pages_keep_their_unread_regions(name):
    pinned = json.loads(PINNED.read_text("utf-8"))[name]
    assert [list(box) for box in _page(FIXTURES / name).unread_regions] == pinned


def test_a_white_on_black_page_is_flagged_unread(tmp_path):
    path = tmp_path / "white-on-black.png"
    img = Image.new("L", (1240, 1754), 0)
    ImageDraw.Draw(img).rectangle((200, 300, 1000, 500), fill=255)
    img.save(path)
    page = _page(path)
    assert "unread_regions" in page.flags
    assert page.unread_regions


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def test_a_line_missed_on_a_photo_gives_a_region_around_that_line_only(monkeypatch):
    name = "en-invoice-photo.jpg"
    gt = json.loads((FIXTURES / "en-invoice-photo.gt.json").read_text("utf-8"))
    clean = [tuple(line["clean_box"]) for line in gt["lines"]]
    transform = _page(FIXTURES / name).transform
    target = gt_line_boxes(clean, gt["homography"], transform)[5]
    original = ocr.find_words

    def without_target(models, gray, rgb, detection):
        found = original(models, gray, rgb, detection)
        return [b for b in found if not _inside(_centre(b), target)]

    monkeypatch.setattr(ocr, "find_words", without_target)
    page = _page(FIXTURES / name)
    assert page.transform == transform
    boxes = gt_line_boxes(clean, gt["homography"], page.transform)
    assert 5 not in [g for g, _ in match_lines(boxes, [line.box for line in page.lines])]
    hits = [b for b in page.unread_regions if _inside(_centre(target), b)]
    assert len(hits) == 1
    x0, y0, x1, y1 = hits[0]
    pad = 32  # one ink square
    assert target[0] - pad <= x0 and target[1] - pad <= y0
    assert x1 <= target[2] + pad and y1 <= target[3] + pad
