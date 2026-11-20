"""The OnnxTR text detector and printed-Latin word recogniser, loaded from the pinned model files
in the assets folder and nowhere else: nothing is downloaded while a page is read."""

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from folio_ocr.assets import ONNXTR, ONNXTR_DETECTOR, ONNXTR_RECOGNISER, files_problem
from folio_ocr.document import Box
from folio_ocr.errors import SetupNeededError, WeightsError

DETECTOR = "fast_base"
RECOGNISER = "crnn_vgg16_bn"
# OnnxTR scores a part of a word by its least certain CTC step (the largest softmax probability of
# a step, blanks included, minimised over the steps). A word wider than SPLIT_RATIO times its
# height is read in overlapping parts and scored by its least certain part.
AGGREGATION = "min;split=min"
# The crop split the recogniser was calibrated with (OnnxTR's critical, target aspect ratio and
# overlap); a recogniser configured otherwise is refused.
SPLIT_RATIO = 8
SPLIT_TARGET = 6
SPLIT_OVERLAP = 0.5


class _LocalPath(str):
    """A model file path. OnnxTR downloads any model path that contains "http"; this one never
    reports containing it, so a folder such as `C:\\Users\\httpd` still loads from disk."""

    __slots__ = ()

    def __contains__(self, item: object) -> bool:
        return False if item == "http" else super().__contains__(item)  # type: ignore[operator]


_PROBLEM = {
    "missing": "the OCR model files are missing; run folio setup",
    "cannot read": "the OCR model files cannot be read; check the access rights of the assets "
    "folder",
    "damaged": "the OCR model files are damaged; run folio setup",
}


@dataclass(frozen=True)
class Models:
    detector: Any  # onnxtr DetectionPredictor
    recogniser: Any  # onnxtr RecognitionPredictor, reading one crop or one part per call
    split_crops: Any  # onnxtr's crop splitter
    remap_preds: Any  # onnxtr's joiner of the texts read from the parts of a crop


@dataclass(frozen=True)
class Detection:
    boxes: np.ndarray  # (N, 4) relative x0, y0, x1, y1 on the page that was searched
    seg_map: np.ndarray  # the detector's text mask (uint8, 0 or 255), square, padded


def load(weights_dir: Path | None) -> Models:
    """The detector and recogniser from `weights_dir`, after checking both files against their
    pinned size and sha256. Raises SetupNeededError when there is no folder or a file is missing
    or damaged, and plain WeightsError when a file cannot be read."""
    if weights_dir is None:
        raise SetupNeededError("the OCR model files are not set up; run folio setup")
    problem = files_problem(ONNXTR, weights_dir)
    if problem in ("missing", "damaged"):
        raise SetupNeededError(_PROBLEM[problem])
    if problem is not None:
        raise WeightsError(_PROBLEM[problem])
    return _load(str(weights_dir))


def _check_split(recogniser: Any) -> None:
    pinned = {"critical_ar": SPLIT_RATIO, "target_ar": SPLIT_TARGET, "overlap_ratio": SPLIT_OVERLAP}
    for name, value in pinned.items():
        if getattr(recogniser, name) != value:
            raise WeightsError(f"the OCR recogniser splits wide words differently ({name})")


@functools.lru_cache(maxsize=2)
def _load(folder: str) -> Models:
    from onnxtr.models import detection_predictor, recognition_predictor
    from onnxtr.models.detection import fast_base
    from onnxtr.models.recognition import crnn_vgg16_bn
    from onnxtr.models.recognition.predictor._utils import remap_preds, split_crops

    base = Path(folder)
    detector = fast_base(_LocalPath(base / ONNXTR_DETECTOR.name))
    recogniser = recognition_predictor(crnn_vgg16_bn(_LocalPath(base / ONNXTR_RECOGNISER.name)))
    _check_split(recogniser)
    recogniser.split_wide_crops = False  # words are split and joined by `recognise`
    return Models(detection_predictor(detector), recogniser, split_crops, remap_preds)


def detect(models: Models, rgb: np.ndarray) -> Detection:
    """Word boxes found on the page `rgb` (H, W, 3 uint8) and the detector's text mask."""
    (found,), (out_map,) = models.detector([rgb], return_maps=True)
    threshold = models.detector.model.postprocessor.bin_thresh
    mask = (out_map > threshold).astype(np.uint8) * 255
    return Detection(np.asarray(found, dtype=np.float64).reshape(-1, 5)[:, :4], mask)


def pixel_boxes(detection: Detection, width: int, height: int) -> list[Box]:
    """The detected boxes in pixels of a `width` x `height` page."""
    return [
        (float(x0 * width), float(y0 * height), float(x1 * width), float(y1 * height))
        for x0, y0, x1, y1 in detection.boxes
    ]


def _word_score(scores: list[float]) -> float:
    """The least of a word's part scores; NaN when any is not a probability."""
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in scores):
        return math.nan
    return min(scores)


def _read_crops(models: Models, crops: list[np.ndarray]) -> list[tuple[str, float, bool]]:
    parts, crop_map, _ = models.split_crops(crops, SPLIT_RATIO, SPLIT_TARGET, SPLIT_OVERLAP, True)
    results = models.recogniser(parts)
    texts = models.remap_preds(results, crop_map, SPLIT_OVERLAP)
    words = []
    for (text, _), item in zip(texts, crop_map, strict=True):
        first, end = (item, item + 1) if isinstance(item, int) else item[:2]
        scores = [float(results[k][1]) for k in range(first, end)]
        words.append((text, _word_score(scores), end - first > 1))
    return words


def recognise(models: Models, rgb: np.ndarray, boxes: np.ndarray) -> list[tuple[str, float, bool]]:
    """(text, score, split) of each word crop, read in one pass. A crop wider than SPLIT_RATIO is
    read in overlapping parts (split is True): its text is OnnxTR's join of the parts and its
    score the least part score. A score that is not a probability, or of a crop with no pixels
    (a box on the page's last half pixel, which is not read), is NaN: not usable."""
    from onnxtr.utils.geometry import extract_crops

    if len(boxes) == 0:
        return []
    crops = extract_crops(rgb, np.asarray(boxes, dtype=np.float64))
    kept = [i for i, crop in enumerate(crops) if crop.shape[0] > 0 and crop.shape[1] > 0]
    read = (
        dict(zip(kept, _read_crops(models, [crops[i] for i in kept]), strict=True)) if kept else {}
    )
    return [read.get(i, ("", math.nan, False)) for i in range(len(crops))]
