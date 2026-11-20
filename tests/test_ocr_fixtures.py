"""Reading the fixture files with the real OCR models: line finding through the page transform,
orientation, reader flags, confidence scores and the offline guarantees."""

import functools
import inspect
import json
import math
import os
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from folio_ocr import calibration, cli, ocr, ocr_setup, onnxtr_models
from folio_ocr.assets import ONNXTR, TESSERACT_OSD, files_problem
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.document import Document
from folio_ocr.errors import WeightsError
from folio_ocr.metrics import cer, compose, gt_line_boxes, map_box, match_lines
from folio_ocr.pipeline import read_path
from folio_ocr.tesseract import tessdata_problem

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "data" / "ocr_golden.json"
WEIGHTS = Settings.from_env(os.environ).weights_dir
TESSERACT = find_tesseract(Settings.from_env(os.environ))


def _major_minor() -> str | None:
    """major.minor of the installed Tesseract, or None when there is none."""
    if TESSERACT is None:
        return None
    from folio_ocr.tesseract import tesseract_version

    return calibration.version_major_minor(tesseract_version(TESSERACT))


TESSERACT_MAJOR_MINOR = _major_minor()

if ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"):
    pytest.skip("the OCR model files are not set up (folio setup)", allow_module_level=True)

needs_tesseract = pytest.mark.skipif(
    TESSERACT is None or tessdata_problem(WEIGHTS) is not None,
    reason="Tesseract or its language data is not set up",
)

needs_osd = pytest.mark.skipif(
    TESSERACT is None or files_problem((TESSERACT_OSD,), WEIGHTS) is not None,
    reason="Tesseract or the pinned osd.traineddata is not set up",
)

IMAGES = sorted(p.name for p in FIXTURES.iterdir() if p.suffix in (".png", ".jpg"))
CONDITIONS = {
    "clean": ("-clean.png",),
    "scan": ("-scan.jpg",),
    "photo": ("-photo.jpg", "-photo-exif.jpg"),
}
MIN_RECALL = {"clean": 0.98, "scan": 0.98, "photo": 0.95}
WHITE = (255, 255, 255)


def _languages(name: str) -> str:
    return {"ar": "ar", "mixed": "en+ar"}.get(name.split("-")[0], "en")


@functools.cache
def _read(name: str, languages: str | None = None, tesseract: bool = False) -> Document:
    return read_path(
        FIXTURES / name,
        languages=languages or _languages(name),
        weights_dir=WEIGHTS,
        tesseract_path=TESSERACT if tesseract else None,
    )


def _gt(name: str) -> dict:
    return json.loads((FIXTURES / (name.rsplit(".", 1)[0] + ".gt.json")).read_text("utf-8"))


def _matched(doc: Document, gt: dict, base=None) -> tuple[list, tuple]:
    """(ground-truth lines, matches) of a page read, in the straightened page frame."""
    page = doc.pages[0]
    homography = gt["homography"] if base is None else compose(base, gt["homography"])
    boxes = gt_line_boxes([tuple(g["clean_box"]) for g in gt["lines"]], homography, page.transform)
    return boxes, match_lines(boxes, [line.box for line in page.lines])


# 1. line recall through the page transform
@pytest.mark.parametrize("condition", CONDITIONS)
def test_line_recall_by_condition(condition):
    hit = total = 0
    for name in IMAGES:
        if not name.endswith(CONDITIONS[condition]):
            continue
        boxes, matches = _matched(_read(name), _gt(name))
        hit, total = hit + len(matches), total + len(boxes)
    assert total > 0
    assert hit / total >= MIN_RECALL[condition], (hit, total)


def test_transform_is_needed_for_scans():
    stored = misses = 0
    for name in (n for n in IMAGES if n.endswith("-scan.jpg")):
        gt = _gt(name)
        page = _read(name).pages[0]
        boxes = [tuple(g["box"]) for g in gt["lines"]]
        stored += len(match_lines(boxes, [line.box for line in page.lines]))
        misses += len(boxes)
    mapped = sum(len(_matched(_read(n), _gt(n))[1]) for n in IMAGES if n.endswith("-scan.jpg"))
    assert stored < mapped <= misses


# 2. upside down
def test_upside_down_page_reads_like_the_clean_one():
    def rate(name: str) -> float:
        doc, gt = _read(name), _gt(name)
        _, matches = _matched(doc, gt)
        pairs = [(gt["lines"][g]["text"], doc.pages[0].lines[j].text) for g, j in matches]
        return cer(pairs).rate

    assert _read("en-invoice-upside_down.png").pages[0].rotation_applied == 180.0
    assert abs(rate("en-invoice-upside_down.png") - rate("en-invoice-clean.png")) <= 0.01


# 3. rotations
def _turn_matrix(kind: str, w: int, h: int):
    if kind == "90":
        return ((0, 1, 0), (-1, 0, w), (0, 0, 1))
    if kind == "180":
        return ((-1, 0, w), (0, -1, h), (0, 0, 1))
    return ((0, -1, h), (1, 0, 0), (0, 0, 1))


def _tilt_matrix(degrees: float, size: tuple[int, int], new: tuple[int, int]):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    cx, cy, nx, ny = size[0] / 2, size[1] / 2, new[0] / 2, new[1] / 2
    return (
        (c, s, nx - c * cx - s * cy),
        (-s, c, ny + s * cx - c * cy),
        (0, 0, 1),
    )


TURNS = {
    "90": (Image.Transpose.ROTATE_90, 270.0),
    "180": (Image.Transpose.ROTATE_180, 180.0),
    "270": (Image.Transpose.ROTATE_270, 90.0),
}


@pytest.mark.parametrize(
    "with_tesseract", [False, pytest.param(True, marks=needs_osd)], ids=["vote", "osd"]
)
@pytest.mark.parametrize("kind", [*TURNS, "+3", "-3"])
def test_turned_invoice_keeps_every_line(kind, with_tesseract, tmp_path):
    source = Image.open(FIXTURES / "en-invoice-clean.png").convert("RGB")
    if kind in TURNS:
        turned = source.transpose(TURNS[kind][0])
        base, expected = _turn_matrix(kind, *source.size), TURNS[kind][1]
    else:
        degrees = float(kind)
        turned = source.rotate(degrees, expand=True, fillcolor=WHITE)
        base, expected = _tilt_matrix(degrees, source.size, turned.size), (-degrees) % 360
    path = tmp_path / "turned.png"
    turned.save(path)
    doc = read_path(
        path,
        languages="en",
        weights_dir=WEIGHTS,
        tesseract_path=TESSERACT if with_tesseract else None,
    )
    boxes, matches = _matched(doc, _gt("en-invoice-clean.png"), base)
    upright = _matched(_read("en-invoice-clean.png", "en"), _gt("en-invoice-clean.png"))[1]
    # Turning the page loses no line the upright page keeps (at gap factor 1.7 the upright
    # invoice joins one item with its quantity, so 23 of the 24 ground-truth lines match).
    assert len(boxes) == 24 and len(matches) == len(upright) == 23
    assert [g for g, _ in matches] == [g for g, _ in upright]
    gap = abs(doc.pages[0].rotation_applied - expected)
    assert min(gap, 360 - gap) <= (0 if kind in TURNS else 1)


# 4. a line the detector misses is reported as unread ink
def test_missed_line_becomes_an_unread_region(monkeypatch):
    gt = _gt("en-invoice-clean.png")
    target = tuple(gt["lines"][5]["clean_box"])
    original = ocr.find_words

    def without_target(models, gray, rgb, detection):
        def inside(b):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return target[0] <= cx <= target[2] and target[1] <= cy <= target[3]

        return [b for b in original(models, gray, rgb, detection) if not inside(b)]

    monkeypatch.setattr(ocr, "find_words", without_target)
    doc = read_path(FIXTURES / "en-invoice-clean.png", languages="en", weights_dir=WEIGHTS)
    page = doc.pages[0]
    mapped = map_box(page.transform, target)
    centre = ((mapped[0] + mapped[2]) / 2, (mapped[1] + mapped[3]) / 2)
    assert 5 not in [g for g, _ in _matched(doc, gt)[1]]
    assert any(b[0] <= centre[0] <= b[2] and b[1] <= centre[1] <= b[3] for b in page.unread_regions)
    assert "unread_regions" in page.flags


# 5. Arabic and mixed pages without a reader, and with Tesseract
OCR_ONLY = [n for n in IMAGES if not n.startswith("en-") and n != "blank-scan.jpg"]


@pytest.mark.parametrize("name", [*OCR_ONLY, "ar-invoice-scanned.pdf"])
def test_arabic_pages_wait_for_a_model_server_without_tesseract(name):
    doc = _read(name)
    page = doc.pages[0]
    assert page.origin == "ocr" and page.lines
    assert doc.engine.reader == "none" and doc.engine.calibration is None
    assert doc.engine.detector == "fast_base"
    for line in page.lines:
        assert line.reader == "none" and line.text == ""
        assert line.flags[:1] == ("needs_model_server",)
        assert line.status != "auto_accepted"


@needs_tesseract
@pytest.mark.parametrize("name", ["ar-letter-clean.png", "mixed-invoice-clean.png"])
def test_tesseract_lines_are_accepted_only_at_or_above_the_frozen_tau(name):
    doc = _read(name, tesseract=True)
    assert doc.engine.reader == "tesseract"
    assert doc.engine.calibration.setup_key.startswith("tesseract|lstm|")
    tau = doc.engine.calibration.tau
    assert tau == calibration.lookup(doc.engine.calibration.setup_key)
    read = [line for line in doc.pages[0].lines if line.reader == "tesseract"]
    assert read
    for line in read:
        low = tau is None or line.confidence is None or line.confidence < tau
        assert ("low_confidence" in line.flags) == low
        assert (line.status == "auto_accepted") == (not line.flags)


# 6. 7. no network
@pytest.fixture
def no_sockets(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)


def test_english_read_without_network(no_sockets):
    onnxtr_models._load.cache_clear()
    doc = read_path(FIXTURES / "en-invoice-clean.png", languages="en", weights_dir=WEIGHTS)
    assert doc.pages[0].lines


@needs_tesseract
def test_arabic_read_with_tesseract_without_network(no_sockets):
    onnxtr_models._load.cache_clear()
    doc = read_path(
        FIXTURES / "ar-letter-clean.png",
        languages="ar",
        weights_dir=WEIGHTS,
        tesseract_path=TESSERACT,
    )
    assert doc.pages[0].lines and doc.engine.reader == "tesseract"


def test_model_folder_named_http_loads_from_disk(tmp_path, no_sockets):
    folder = tmp_path / "httpd assets"
    for asset in ONNXTR:
        target = folder / asset.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WEIGHTS / asset.name, target)
    onnxtr_models._load.cache_clear()
    doc = read_path(FIXTURES / "en-invoice-clean.png", languages="en", weights_dir=folder)
    assert doc.pages[0].lines


# 8. OnnxTR 0.9.0 word-score semantics
def test_onnxtr_word_score_is_the_least_certain_step():
    import onnxtr
    from onnxtr.models import ocr_predictor
    from onnxtr.models.recognition.models.crnn import CRNNPostProcessor

    assert onnxtr.__version__ == "v0.9.0"
    assert "confidence_aggregation" not in inspect.signature(ocr_predictor).parameters
    logits = np.full((1, 3, 3), -10.0)  # steps x (two characters + blank)
    logits[0, 0, 0] = logits[0, 2, 1] = 10.0
    logits[0, 1] = [0.2, 0.0, 0.1]  # one unsure step
    ((_, score),) = CRNNPostProcessor("ab")(logits)
    uncertain = float(np.exp(0.2) / np.exp([0.2, 0.0, 0.1]).sum())
    assert score == pytest.approx(uncertain)
    assert score < 0.5  # the mean over steps would be about 0.78


# 8b. split words
def _invoice_id_crop(models):
    """The crop of the word `INV-2026-3706` of en-invoice-clean.png and its relative box."""
    from onnxtr.utils.geometry import extract_crops

    rgb = np.array(Image.open(FIXTURES / "en-invoice-clean.png").convert("RGB"))
    line = next(
        g for g in _gt("en-invoice-clean.png")["lines"] if g["text"].endswith("INV-2026-3706")
    )
    x0, y0, x1, y1 = line["clean_box"]
    boxes = onnxtr_models.detect(models, rgb).boxes
    height, width = rgb.shape[:2]
    inside = [
        b
        for b in boxes
        if x0 <= (b[0] + b[2]) / 2 * width <= x1 and y0 <= (b[1] + b[3]) / 2 * height <= y1
    ]
    wide = [b for b in inside if (b[2] - b[0]) * width / ((b[3] - b[1]) * height) > 8]
    assert len(wide) == 1
    return rgb, wide[0], extract_crops(rgb, np.array([wide[0]]))[0]


def test_a_split_word_is_read_in_one_pass_and_scored_by_its_lowest_part():
    models = onnxtr_models.load(WEIGHTS)
    rgb, box, crop = _invoice_id_crop(models)
    assert crop.shape[1] / crop.shape[0] > 8
    ((text, score, split),) = onnxtr_models.recognise(models, rgb, np.array([box]))
    parts, _, remapped = models.split_crops(
        [crop], onnxtr_models.SPLIT_RATIO, onnxtr_models.SPLIT_TARGET, onnxtr_models.SPLIT_OVERLAP
    )
    part_scores = [s for _, s in models.recogniser(parts)]
    models.recogniser.split_wide_crops = True  # OnnxTR's own splitting, for the comparison only
    try:
        ((stock_text, stock_score),) = models.recogniser([crop])
    finally:
        models.recogniser.split_wide_crops = False
    assert remapped and len(part_scores) > 1 and split is True
    assert text == stock_text == "INV-2026-3706"
    assert sum(part_scores) / len(part_scores) == pytest.approx(stock_score, abs=1e-6)
    assert score == min(part_scores)


def test_the_crop_split_is_pinned_to_what_the_recogniser_was_calibrated_with():
    models = onnxtr_models.load(WEIGHTS)
    assert (onnxtr_models.SPLIT_RATIO, onnxtr_models.SPLIT_TARGET) == (8, 6)
    assert onnxtr_models.SPLIT_OVERLAP == 0.5
    reader = models.recogniser
    assert (reader.critical_ar, reader.target_ar, reader.overlap_ratio) == (8, 6, 0.5)
    assert reader.split_wide_crops is False


@pytest.mark.parametrize("name", ["critical_ar", "target_ar", "overlap_ratio"])
def test_a_recogniser_with_another_split_setting_is_refused(monkeypatch, name):
    import onnxtr.models as models_module

    real = models_module.recognition_predictor

    def other(model):
        predictor = real(model)
        setattr(predictor, name, getattr(predictor, name) + 1)
        return predictor

    monkeypatch.setattr(models_module, "recognition_predictor", other)
    with pytest.raises(WeightsError) as raised:
        onnxtr_models._load.__wrapped__(str(WEIGHTS))
    assert len(str(raised.value).splitlines()) == 1 and name in str(raised.value)


# 9. word scores reach the line confidence
def test_line_confidence_is_the_lowest_word_score(monkeypatch):
    seen = []
    real = onnxtr_models.recognise

    def spy(models, rgb, boxes):
        result = real(models, rgb, boxes)
        seen.append((rgb.shape, np.asarray(boxes), result))
        return result

    monkeypatch.setattr(onnxtr_models, "recognise", spy)
    doc = read_path(FIXTURES / "en-invoice-clean.png", languages="en", weights_dir=WEIGHTS)
    (height, width, _), boxes, result = seen[-1]
    page = doc.pages[0]
    assert doc.engine.reader == "onnxtr"
    checked = 0
    for line in page.lines:
        scores = []
        for (x0, y0, x1, y1), (_, score, _) in zip(boxes, result, strict=True):
            cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
            if line.box[0] <= cx <= line.box[2] and line.box[1] <= cy <= line.box[3]:
                scores.append(score)
        assert line.reader == "onnxtr" and line.confidence_method == "reader"
        if all(math.isfinite(s) for s in scores):
            assert line.confidence == min(scores)
            checked += 1
        else:
            assert line.confidence is None and "low_confidence" in line.flags
    assert checked > 10


# 10. golden confidence
@pytest.mark.parametrize(
    "with_tesseract", [False, pytest.param(True, marks=needs_osd)], ids=["vote", "osd"]
)
def test_golden_lines_keep_their_text_and_confidence(with_tesseract):
    # Both OnnxTR setups read the upright invoice the same way; each is judged at its own tau.
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    doc = _read("en-invoice-clean.png", "en", with_tesseract)
    assert doc.engine.calibration.setup_key == (
        ocr_setup.onnxtr_setup(TESSERACT_MAJOR_MINOR if with_tesseract else None).key
    )
    by_id = {line.id: line for line in doc.pages[0].lines}
    tau = doc.engine.calibration.tau
    assert tau == calibration.lookup(doc.engine.calibration.setup_key)
    assert len(golden["lines"]) == 7
    for pinned in golden["lines"]:
        line = by_id[pinned["id"]]
        assert line.text == pinned["text"]
        assert abs(line.confidence - pinned["confidence"]) <= 0.01
        low = "low_confidence" in line.flags
        assert low == (tau is None or line.confidence < tau)
        # The flag decision at the frozen tau is the one the pinned score gives.
        assert low == (tau is None or pinned["confidence"] < tau)


def test_a_pinned_golden_line_holds_a_split_word(monkeypatch):
    seen = []
    real = onnxtr_models.recognise
    monkeypatch.setattr(onnxtr_models, "recognise", lambda *a: seen.append(real(*a)) or seen[-1])
    read_path(FIXTURES / "en-invoice-clean.png", languages="en", weights_dir=WEIGHTS)
    pinned = [ln for ln in json.loads(GOLDEN.read_text("utf-8"))["lines"] if "INV-" in ln["text"]]
    assert [w[0] for w in seen[-1] if w[2]] == ["INV-2026-3706"]
    assert len(pinned) == 1 and pinned[0]["text"].endswith("INV-2026-3706")


# 11. the command line
def test_folio_read_prints_one_json_document(capsys):
    path = FIXTURES / "en-invoice-clean.png"
    code = cli.main(
        ["read", "--lang", "en", "--format", "json", str(path)],
        {"FOLIO_WEIGHTS_DIR": str(WEIGHTS)},
    )
    out, err = capsys.readouterr()
    page = json.loads(out)["pages"][0]
    assert code == 0 and page["origin"] == "ocr" and page["lines"]
    assert "Invoice: Northwind Office Supply Ltd" in [line["text"] for line in page["lines"]]
    review = sum(line["status"] != "auto_accepted" for line in page["lines"])
    assert f"{path}: 23 lines found, 23 read, {review} need review, 0 unread pages" in err


# 12. orientation uncertainty
def test_without_tesseract_only_the_arabic_letters_have_an_uncertain_orientation():
    # The vote (no orientation answer from Tesseract) scores Arabic letters closely both ways up;
    # with the pair margin 0.13 the clean letter is flagged as well as the two photos.
    uncertain = sorted(n for n in IMAGES if "orientation_uncertain" in _read(n).pages[0].flags)
    assert uncertain == ["ar-letter-clean.png", "ar-letter-photo-exif.jpg", "ar-letter-photo.jpg"]
    for name in uncertain:
        assert all("orientation_uncertain" in ln.flags for ln in _read(name).pages[0].lines)


# 13. Tesseract reads mixed-language pages
@needs_tesseract
def test_default_languages_use_tesseract():
    doc = _read("en-invoice-clean.png", "en+ar", True)
    assert doc.engine.reader == "tesseract"
    assert any(line.reader == "tesseract" for line in doc.pages[0].lines)


# 14. Tesseract time limit
@needs_tesseract
def test_tesseract_timeout_is_reported_without_a_crash():
    doc = read_path(
        FIXTURES / "en-invoice-clean.png",
        languages="en+ar",
        weights_dir=WEIGHTS,
        tesseract_path=TESSERACT,
        line_timeout_s=0.001,
    )
    page = doc.pages[0]
    # The orientation answer runs out of time first; the vote then turns the page.
    assert (
        page.error
        == "Tesseract did not report the page orientation within 0.001 s; "
        f"{len(page.lines)} lines not read"
    )
    assert len(page.lines) == 23 and page.status == "read"  # the version check has its own limit
    for line in page.lines:
        assert (line.text, line.reader, line.flags[0], line.status) == (
            "",
            "none",
            "not_read",
            "needs_review",
        )


# Reads from worker threads give the same pages as one at a time.
def _page_lines(doc: Document) -> list[tuple]:
    return [
        (line.id, line.text, line.box, line.flags, line.confidence)
        for page in doc.pages
        for line in page.lines
    ]


def test_parallel_reads_equal_serial_reads():
    jobs = [
        ("en-invoice-clean.png", "en"),
        ("en-letter-scan.jpg", "en"),
        ("ar-invoice-scanned.pdf", "ar"),
    ]

    def read(job):
        return read_path(FIXTURES / job[0], languages=job[1], weights_dir=WEIGHTS)

    serial = {job: _page_lines(read(job)) for job in jobs}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda job: (job, _page_lines(read(job))), jobs * 2))
    for job, lines in results:
        assert len(lines) == len(serial[job]) > 0
        for got, want in zip(lines, serial[job], strict=True):
            assert got[:4] == want[:4]
            assert (got[4] is None) == (want[4] is None)
            assert got[4] is None or abs(got[4] - want[4]) <= 1e-6
