"""Line grouping, text direction and the OnnxTR model wrappers, tested with doubles."""

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from onnxtr.models.recognition.predictor._utils import remap_preds, split_crops

from folio_ocr import onnxtr_models
from folio_ocr.assets import Asset
from folio_ocr.errors import SetupNeededError, WeightsError
from folio_ocr.onnxtr_models import Detection, Models, _LocalPath
from folio_ocr.text_layer import group_boxes
from folio_ocr.text_layer_check import majority_direction, text_direction

# --- group_boxes ---------------------------------------------------------------------------

# Row 1 (height 10): boxes 1 and 0 are 10 apart (kept together at factor 1), 2 is 10.5 further.
BOXES = [
    (30.0, 0.0, 50.0, 10.0),  # 0
    (0.0, 0.0, 20.0, 10.0),  # 1
    (60.5, 0.0, 80.0, 10.0),  # 2
    (0.0, 30.0, 20.0, 40.0),  # 3: second row
    (0.0, 50.0, 20.0, 50.0),  # 4: no height
]


def test_group_boxes_cuts_a_row_only_at_gaps_above_the_factor():
    assert group_boxes(BOXES, "ltr", 1.0) == [[1, 0], [2], [3]]


def test_group_boxes_keeps_a_gap_equal_to_the_limit():
    assert group_boxes(BOXES[:2], "ltr", 1.0) == [[1, 0]]
    assert group_boxes(BOXES[:2], "ltr", 0.99) == [[1], [0]]


def test_group_boxes_orders_the_parts_of_a_row_by_direction():
    assert group_boxes(BOXES, "rtl", 1.0) == [[2], [1, 0], [3]]


def test_group_boxes_leaves_out_boxes_without_height():
    assert 4 not in [i for line in group_boxes(BOXES, "ltr", 1.0) for i in line]
    assert group_boxes([(0.0, 5.0, 9.0, 5.0)], "ltr", 1.3) == []


def test_group_boxes_reads_rows_top_to_bottom_in_both_directions():
    low_first = [(0.0, 30.0, 20.0, 40.0), (0.0, 0.0, 20.0, 10.0)]
    assert group_boxes(low_first, "ltr", 1.3) == [[1], [0]]
    assert group_boxes(low_first, "rtl", 1.3) == [[1], [0]]


# --- directions ----------------------------------------------------------------------------

AR = "\u0627\u0628\u062a"


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["abc"], "ltr"),
        ([AR], "rtl"),
        (["abc", "a" + AR[:2]], "ltr"),
        (["a", AR[0]], "rtl"),  # a tie
        (["1234567 " + AR[0]], "rtl"),  # digits do not vote
        (["1234567"], "rtl"),  # no letters
        ([], "rtl"),
    ],
)
def test_majority_direction(texts, expected):
    assert majority_direction(texts) == expected


@pytest.mark.parametrize(
    ("text", "page", "expected"),
    [
        ("abc", "rtl", "ltr"),
        (AR, "ltr", "rtl"),
        ("abc " + AR, "ltr", "ltr"),
        ("abc " + AR, "rtl", "rtl"),
        ("123", "rtl", "rtl"),
        ("123", "ltr", "ltr"),
        ("", "ltr", "ltr"),
    ],
)
def test_text_direction(text, page, expected):
    assert text_direction(text, page) == expected


# --- model loading -------------------------------------------------------------------------

GOOD = {"onnxtr/d.onnx": b"abc", "onnxtr/r.onnx": b"xyz"}


def fake_assets(tmp_path, monkeypatch, files):
    """Two tiny pinned model files; `files` maps names to the bytes written (absent: missing)."""
    pinned = tuple(
        Asset(name, "https://x.example/" + name, hashlib.sha256(data).hexdigest(), 3, "Apache-2.0")
        for name, data in GOOD.items()
    )
    monkeypatch.setattr(onnxtr_models, "ONNXTR", pinned)
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)


def test_load_without_a_folder_asks_for_setup():
    with pytest.raises(SetupNeededError) as info:
        onnxtr_models.load(None)
    assert str(info.value) == "the OCR model files are not set up; run folio setup"


def test_load_reports_a_missing_file(tmp_path, monkeypatch):
    fake_assets(tmp_path, monkeypatch, {"onnxtr/d.onnx": b"abc"})
    with pytest.raises(SetupNeededError) as info:
        onnxtr_models.load(tmp_path)
    assert str(info.value) == "the OCR model files are missing; run folio setup"


@pytest.mark.parametrize("wrong", [b"abd", b"abcd"])
def test_load_reports_wrong_bytes_as_damaged(tmp_path, monkeypatch, wrong):
    fake_assets(tmp_path, monkeypatch, {"onnxtr/d.onnx": wrong, "onnxtr/r.onnx": b"xyz"})
    with pytest.raises(SetupNeededError) as info:
        onnxtr_models.load(tmp_path)
    assert str(info.value) == "the OCR model files are damaged; run folio setup"


def test_load_reports_an_unreadable_file(tmp_path, monkeypatch):
    fake_assets(tmp_path, monkeypatch, {"onnxtr/r.onnx": b"xyz"})
    (tmp_path / "onnxtr" / "d.onnx").mkdir()  # a folder where a model file should be
    with pytest.raises(WeightsError) as info:
        onnxtr_models.load(tmp_path)
    assert not isinstance(info.value, SetupNeededError)
    assert str(info.value) == (
        "the OCR model files cannot be read; check the access rights of the assets folder"
    )


def test_load_prefers_missing_over_damaged(tmp_path, monkeypatch):
    fake_assets(tmp_path, monkeypatch, {"onnxtr/d.onnx": b"abd"})
    with pytest.raises(SetupNeededError) as info:
        onnxtr_models.load(tmp_path)
    assert str(info.value) == "the OCR model files are missing; run folio setup"


def test_load_hands_intact_files_to_the_loader(tmp_path, monkeypatch):
    fake_assets(tmp_path, monkeypatch, GOOD)
    seen = []
    monkeypatch.setattr(onnxtr_models, "_load", lambda folder: seen.append(folder) or "models")
    assert onnxtr_models.load(tmp_path) == "models"
    assert seen == [str(tmp_path)]


def test_local_path_never_contains_http_but_keeps_other_substrings():
    path = _LocalPath("C:/x/httpd/m.onnx")
    assert "http" not in path
    assert "httpd" in path
    assert "x/httpd" in path
    assert "ftp" not in path
    assert str(path) == "C:/x/httpd/m.onnx"
    assert Path(path).name == "m.onnx"


# --- recognise / detect --------------------------------------------------------------------


class FakeRecogniser:
    def __init__(self, scores=None):
        self.calls = 0
        self.scores = scores  # per call: the scores of the parts, in order

    def __call__(self, crops):
        self.calls += 1
        scores = self.scores or [0.9] * len(crops)
        assert len(scores) == len(crops)
        return [("w", s) for s in scores]


def models_with(recogniser):
    return Models(None, recogniser, split_crops, remap_preds)


PAGE = np.full((100, 200, 3), 255, dtype=np.uint8)  # 200 wide, 100 high


def rel_box(x1):
    """A crop 10 rows high and round(x1 * 200) + 1 columns wide."""
    return [0.0, 0.1, x1, 0.19]


def test_recognise_scores_a_wide_crop_by_its_lowest_part_and_marks_it_split():
    recogniser = FakeRecogniser([0.9, 0.9, 0.7, 0.8])
    boxes = np.array([rel_box(0.395), rel_box(0.4), rel_box(0.2)])
    read = onnxtr_models.recognise(models_with(recogniser), PAGE, boxes)
    assert read[0] == ("w", 0.9, False)  # 80 x 10: exactly 8:1, one part
    assert read[1][0] == "ww" and read[1][1:] == (
        0.7,
        True,
    )  # OnnxTR joins the parts  # 81 x 10: two parts
    assert read[2][1:] == (0.8, False)
    assert recogniser.calls == 1


def test_recognise_gives_a_word_with_an_unusable_part_score_no_score():
    boxes = np.array([rel_box(0.4)])
    for bad in (math.nan, math.inf, -0.1, 1.5):
        read = onnxtr_models.recognise(models_with(FakeRecogniser([0.9, bad])), PAGE, boxes)
        assert read[0][0] == "ww" and math.isnan(read[0][1]) and read[0][2] is True


def test_recognise_reads_nothing_for_no_boxes():
    class Never:
        def __call__(self, crops):
            raise AssertionError("called")

    assert onnxtr_models.recognise(models_with(Never()), PAGE, np.empty((0, 4))) == []


def fake_detector(found, out_map, thresh=0.3):
    def run(images, return_maps):
        assert return_maps and len(images) == 1
        return [found], [out_map]

    run.model = SimpleNamespace(postprocessor=SimpleNamespace(bin_thresh=thresh))
    return run


def test_detect_returns_relative_boxes_and_a_binary_mask():
    found = np.array([[0.1, 0.2, 0.5, 0.6, 0.9], [0.2, 0.2, 0.3, 0.3, 0.8]])
    out_map = np.array([[0.2, 0.31], [0.3, 1.0]])
    result = onnxtr_models.detect(Models(fake_detector(found, out_map), None, None, None), PAGE)
    assert result.boxes.shape == (2, 4)
    assert result.boxes[0].tolist() == [0.1, 0.2, 0.5, 0.6]
    assert result.seg_map.dtype == np.uint8
    assert result.seg_map.tolist() == [[0, 255], [0, 255]]  # strictly above the threshold


def test_detect_without_words_gives_an_empty_box_array():
    result = onnxtr_models.detect(
        Models(fake_detector([], np.zeros((2, 2))), None, None, None), PAGE
    )
    assert result.boxes.shape == (0, 4)


def test_pixel_boxes_scale_x_by_width_and_y_by_height():
    detection = Detection(np.array([[0.1, 0.2, 0.5, 0.6]]), np.zeros((1, 1), dtype=np.uint8))
    (found,) = onnxtr_models.pixel_boxes(detection, 200, 100)
    assert found == pytest.approx((20.0, 20.0, 100.0, 60.0))


def test_recognise_skips_a_crop_with_no_pixels():
    class Counting(FakeRecogniser):
        def __call__(self, crops):
            assert all(c.shape[0] > 0 and c.shape[1] > 0 for c in crops)
            return super().__call__(crops)

    recogniser = Counting()
    edge = [0.0, 0.998, 0.2, 1.0]  # starts on the page's last half row: no pixels
    read = onnxtr_models.recognise(models_with(recogniser), PAGE, np.array([edge, rel_box(0.2)]))
    assert read[0][0] == "" and math.isnan(read[0][1]) and read[0][2] is False
    assert read[1] == ("w", 0.9, False) and recogniser.calls == 1


def test_recognise_with_only_empty_crops_never_calls_the_recogniser():
    class Never:
        def __call__(self, crops):
            raise AssertionError("called")

    read = onnxtr_models.recognise(models_with(Never()), PAGE, np.array([[0.0, 0.998, 0.2, 1.0]]))
    assert len(read) == 1 and read[0][0] == "" and math.isnan(read[0][1])
