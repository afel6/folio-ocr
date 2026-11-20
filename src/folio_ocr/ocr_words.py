"""Words and lines of a straightened OCR page: the detector's word boxes, a closer second search of
ink that no word box covers, and lines grouped from the word boxes with tight ink boxes."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from folio_ocr import onnxtr_models
from folio_ocr.coverage import INK_LEVEL, uncovered_ink
from folio_ocr.document import Box
from folio_ocr.ocr_setup import (
    LINE_INK_MARGIN,
    OCR_GAP_FACTOR,
    RESCUE_REGIONS,
    WINDOW_HEIGHTS,
    WINDOW_MIN_PX,
)
from folio_ocr.onnxtr_models import Detection, Models
from folio_ocr.text_layer import group_boxes


@dataclass(frozen=True)
class Word:
    box: Box
    text: str  # "" when the words are not recognised on their own (Tesseract reads lines)
    score: float  # NaN when not scored
    split: bool = False  # read in more than one part (a word wider than the split ratio)


@dataclass(frozen=True)
class LineWords:
    box: Box  # the ink of the line
    words: tuple[Word, ...]  # left to right


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _search_again(models: Models, rgb: np.ndarray, region: Box, known: list[Box]) -> list[Box]:
    """Word boxes the detector finds when it looks only at the window around `region`."""
    pad = max(WINDOW_MIN_PX, WINDOW_HEIGHTS * (region[3] - region[1]))
    height, width = rgb.shape[:2]
    x0, y0 = max(0, math.floor(region[0] - pad)), max(0, math.floor(region[1] - pad))
    x1, y1 = min(width, math.ceil(region[2] + pad)), min(height, math.ceil(region[3] + pad))
    window = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    found = onnxtr_models.detect(models, window)
    added: list[Box] = []
    for bx0, by0, bx1, by1 in onnxtr_models.pixel_boxes(found, x1 - x0, y1 - y0):
        box = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
        if inside and not any(_overlaps(box, other) for other in (*known, *added)):
            added.append(box)
    return added


def find_words(
    models: Models, gray: Image.Image, rgb: np.ndarray, detection: Detection | None
) -> list[Box]:
    """Word boxes of the straightened page (`gray` and `rgb` hold the same pixels). The largest
    RESCUE_REGIONS regions of ink outside every box are searched again at a closer scale."""
    found = detection if detection is not None else onnxtr_models.detect(models, rgb)
    boxes = onnxtr_models.pixel_boxes(found, gray.width, gray.height)
    regions = sorted(uncovered_ink(gray, boxes), key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))[
        :RESCUE_REGIONS
    ]
    for region in regions:
        boxes += _search_again(models, rgb, region, boxes)
    return boxes


def recognise_words(models: Models, rgb: np.ndarray, boxes: Sequence[Box]) -> list[Word]:
    """Each box read as one printed word by the OnnxTR recogniser."""
    height, width = rgb.shape[:2]
    relative = np.array(
        [(b[0] / width, b[1] / height, b[2] / width, b[3] / height) for b in boxes],
        dtype=np.float64,
    ).reshape(-1, 4)
    read = onnxtr_models.recognise(models, rgb, relative)
    return [Word(box, text, score, split) for box, (text, score, split) in zip(boxes, read, strict=True)]


def _ink_box(gray: Image.Image, box: Box) -> Box:
    """`box` shrunk to the ink inside it (grown by LINE_INK_MARGIN); unchanged when blank."""
    x0 = max(0, math.floor(box[0] - LINE_INK_MARGIN))
    y0 = max(0, math.floor(box[1] - LINE_INK_MARGIN))
    x1 = min(gray.width, math.ceil(box[2] + LINE_INK_MARGIN))
    y1 = min(gray.height, math.ceil(box[3] + LINE_INK_MARGIN))
    ink = gray.crop((x0, y0, x1, y1)).point(lambda v: 255 if v < INK_LEVEL else 0).getbbox()
    if ink is None:
        return box
    return (float(x0 + ink[0]), float(y0 + ink[1]), float(x0 + ink[2]), float(y0 + ink[3]))


def group_lines(gray: Image.Image, words: Sequence[Word], direction: str) -> list[LineWords]:
    """Lines in reading order for a page of base `direction` ('ltr' or 'rtl')."""
    lines = []
    for group in group_boxes([w.box for w in words], direction, OCR_GAP_FACTOR):
        members = tuple(words[i] for i in group)
        union = (
            min(w.box[0] for w in members),
            min(w.box[1] for w in members),
            max(w.box[2] for w in members),
            max(w.box[3] for w in members),
        )
        lines.append(LineWords(_ink_box(gray, union), members))
    return lines
