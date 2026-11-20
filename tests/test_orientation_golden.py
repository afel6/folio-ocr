"""Golden orientation answers on five generated development pages: the vote alone, and
Tesseract's orientation answer with the vote behind it. Set FOLIO_REGEN_ORIENTATION_GOLDEN=1 to
rewrite tests/data/orientation_golden.json from the current code."""

import json
import math
import os
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from folio_ocr import ocr_setup, onnxtr_models, tesseract
from folio_ocr.assets import TESSERACT_OSD, files_problem
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.flags import ORIENTATION_UNCERTAIN
from folio_ocr.geometry import rotation
from folio_ocr.ingest import page_pixels
from folio_ocr.metrics import compose
from folio_ocr.orientation import page_orientation
from folio_ocr.pipeline import read_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import dev_text
import make_dev_variants as mdv

GOLDEN = Path(__file__).parent / "data" / "orientation_golden.json"
REGEN = os.environ.get("FOLIO_REGEN_ORIENTATION_GOLDEN") == "1"
WEIGHTS = Settings.from_env(os.environ).weights_dir
TESSERACT = find_tesseract(Settings.from_env(os.environ))

if ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"):
    pytest.skip("the OCR model files are not set up (folio setup)", allow_module_level=True)

HAS_OSD = TESSERACT is not None and files_problem((TESSERACT_OSD,), WEIGHTS) is None
needs_osd = pytest.mark.skipif(not HAS_OSD, reason="Tesseract or the pinned osd file is not set up")

PAGES = (
    (6, "ar", "dev-00006-ar-report-lowres"),
    (11, "en+ar", "dev-00011-mixed-report-scan"),
    (8, "en", "dev-00008-en-grid-lowres"),
    (1, "en", "dev-00001-en-report-rotated"),
    (1, "en+ar", "dev-00001-mixed-grid-rotated"),
)
STEMS = [p[2] for p in PAGES]


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    out = tmp_path_factory.mktemp("orientation_golden")
    found = {}
    for seed, language, stem in PAGES:
        entry = mdv.build_document(seed, language, dev_text.LANGUAGES.index(language), out)
        assert entry["unit"] == stem
        gt = json.loads((out / f"{stem}.gt.json").read_text("utf-8"))
        path = out / gt["file"]
        found[stem] = (path, gt)
    return found


@pytest.fixture(scope="module")
def models():
    return onnxtr_models.load(WEIGHTS)


def _image(path: Path) -> Image.Image:
    return page_pixels(ImageOps.exif_transpose(Image.open(path)))


def _true_turn(gt: dict) -> int:
    h = tuple(tuple(float(v) for v in row) for row in gt["homography"])
    for q in (0, 90, 180, 270):
        m = compose(rotation(gt["width"], gt["height"], q)[0], h)
        deg = math.degrees(math.atan2(m[1][0], m[0][0]))
        if abs((deg + 180.0) % 360.0 - 180.0) <= 45.0:
            return q
    raise AssertionError("no quarter turn within 45 degrees")


def _quarter(angle: float) -> int:
    return round(angle / 90.0) % 4 * 90


def _osd(im: Image.Image):
    return tesseract.osd(TESSERACT, WEIGHTS / "tessdata", tesseract.page_png(im), 60.0)


def _row(o) -> dict:
    return {"angle": o.angle, "source": o.source, "uncertain": o.uncertain}


@pytest.fixture(scope="module")
def results(pages, models):
    out = {}
    for stem, (path, gt) in pages.items():
        image = _image(path)
        row = {"vote": _row(page_orientation(models, image))}
        if HAS_OSD:
            row["osd"] = _row(page_orientation(models, image, osd=_osd))
        out[stem] = row
    if REGEN and HAS_OSD:
        GOLDEN.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    return out


def _golden() -> dict:
    return json.loads(GOLDEN.read_text("utf-8"))


@pytest.mark.parametrize("stem", STEMS)
def test_vote_matches_golden(stem, pages, results):
    got = results[stem]["vote"]
    assert got == _golden()[stem]["vote"]
    if _quarter(got["angle"]) != _true_turn(pages[stem][1]):
        assert got["uncertain"] is True


@needs_osd
@pytest.mark.parametrize("stem", STEMS)
def test_osd_matches_golden_and_true_turn(stem, pages, results):
    got = results[stem]["osd"]
    assert got == _golden()[stem]["osd"]
    assert _quarter(got["angle"]) == _true_turn(pages[stem][1])


def _flagged(doc) -> bool:
    return ORIENTATION_UNCERTAIN in doc.pages[0].flags


@needs_osd
def test_read_path_with_osd(pages, results):
    doc = read_path(
        pages[STEMS[0]][0], languages="ar", weights_dir=WEIGHTS, tesseract_path=TESSERACT
    )
    assert doc.pages[0].rotation_applied == results[STEMS[0]]["osd"]["angle"]
    assert not _flagged(doc)


def test_read_path_without_tesseract(pages, results):
    doc = read_path(pages[STEMS[0]][0], languages="en", weights_dir=WEIGHTS, tesseract_path=None)
    vote = results[STEMS[0]]["vote"]
    assert doc.pages[0].rotation_applied == vote["angle"]
    assert _flagged(doc) == vote["uncertain"]
