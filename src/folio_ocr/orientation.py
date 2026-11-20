"""Which way up a page is, and how far it is skewed.

Tesseract's orientation answer decides when it is confident enough and its quarter is one the
sideways test allows; otherwise the vote below decides.

Sideways or not comes from the shape of the long detected words (text lines turned by 90 degrees
give long boxes taller than wide); boxes of a few characters are not long either way and do not
vote. Between the two remaining ways up, the page is read both ways and the way whose words
the recogniser reads more confidently wins: printed words read upside down score low whatever
their script. A page with no long word (a grid of single characters) is read all four ways. The
small skew is OnnxTR's estimate from the detector's text mask."""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from PIL import Image

from folio_ocr import onnxtr_models
from folio_ocr.geometry import straighten
from folio_ocr.onnxtr_models import Detection, Models
from folio_ocr.tesseract import Osd, TesseractFailure

SIDEWAYS_SHARE = 0.5  # share of long-box area in boxes taller than wide
# A box votes on sideways when its long side is at least this many times its short side.
LONG_RATIO = 3.0
VOTE_WORDS = 40  # the largest word crops read for the vote
VOTE_QUANTILE = 0.75
# Below this score margin between the best and the next way up, the way up is reported as uncertain:
# one value for a vote between two ways (a page with long words), one for a vote over all four
# (set on the development pages: above every margin by which that vote picked a wrong way up).
UNCERTAIN_MARGIN_PAIR = 0.13
UNCERTAIN_MARGIN_FOUR = 0.41
# Tesseract's orientation answer is used only from this confidence on (set on the development
# pages: above the confidence of every wrong answer among the allowed turns).
OSD_FLOOR = 1.5


@dataclass(frozen=True)
class Orientation:
    angle: float  # degrees counterclockwise that straighten the page
    uncertain: bool  # the best two ways up scored within the vote's margin, or the orientation
    # answer failed
    detection: Detection | None  # words of the straightened page when no skew was applied
    source: Literal["osd", "vote"] = "vote"  # what decided the way up
    error: str | None = None  # one line: why Tesseract's orientation answer could not be had
    # Which ways up were possible: two (the page has long words), four (it has none), or none (no
    # word was found, nothing to turn); taken from the sideways test only.
    branch: Literal["pair", "four", "none"] = "none"
    margin: float | None = None  # vote only: best score minus the next way's score


def _sideways(boxes: np.ndarray, width: int, height: int) -> bool | None:
    """Most long-box area lies in boxes taller than wide (`boxes` are relative to a page of
    `width` x `height` pixels); None when the page has no long box."""
    widths = (boxes[:, 2] - boxes[:, 0]) * width
    heights = (boxes[:, 3] - boxes[:, 1]) * height
    long_ = np.maximum(widths, heights) >= LONG_RATIO * np.minimum(widths, heights)
    areas = widths * heights * long_
    total = float(areas.sum())
    if total <= 0:
        return None
    return float(areas[heights > widths].sum()) / total > SIDEWAYS_SHARE


def _vote(models: Models, rgb: np.ndarray, detection: Detection) -> float:
    """The VOTE_QUANTILE word score over the largest word crops; 0.0 when none is usable."""
    boxes = detection.boxes
    h, w = rgb.shape[:2]
    areas = (boxes[:, 2] - boxes[:, 0]) * w * (boxes[:, 3] - boxes[:, 1]) * h
    largest = boxes[np.argsort(-areas, kind="stable")[:VOTE_WORDS]]
    scores = [s for _, s, _ in onnxtr_models.recognise(models, rgb, largest) if math.isfinite(s)]
    return float(np.quantile(scores, VOTE_QUANTILE)) if scores else 0.0


def _skew(detection: Detection) -> int:
    from onnxtr.models._utils import estimate_orientation

    return int(estimate_orientation(detection.seg_map, (0, 1.0)))


def _turned(models: Models, image: Image.Image, rgb: np.ndarray, first: Detection, quarter: int):
    """The page turned by `quarter` as an array, and its detection (`first` when upright)."""
    if quarter == 0:
        return rgb, first
    turned = np.asarray(straighten(image, quarter)[0].convert("RGB"))
    return turned, onnxtr_models.detect(models, turned)


def _branch(ways: tuple[int, ...]) -> Literal["pair", "four"]:
    return "four" if len(ways) == 4 else "pair"


def _result(
    quarter: int, found: Detection, uncertain: bool, branch: Literal["pair", "four"]
) -> Orientation:
    skew = _skew(found)
    angle = float((quarter + skew) % 360)
    return Orientation(angle, uncertain, found if skew == 0 else None, "osd", None, branch)


def _by_vote(models, image, rgb, first, ways) -> Orientation:
    seen = []
    for quarter in ways:
        turned, found = _turned(models, image, rgb, first, quarter)
        seen.append((_vote(models, turned, found), quarter, found))
    (best, quarter, found), (other, _, _), *_ = sorted(seen, key=lambda s: -s[0])
    limit = UNCERTAIN_MARGIN_FOUR if len(ways) == 4 else UNCERTAIN_MARGIN_PAIR
    voted = _result(quarter, found, best - other < limit, _branch(ways))
    return replace(voted, source="vote", margin=best - other)


def page_orientation(
    models: Models,
    image: Image.Image,
    osd: Callable[[Image.Image], Osd | None] | None = None,
) -> Orientation:
    """How to turn `image` (post-EXIF page, mode L or RGB) so its text reads upright. An `osd`
    answer with confidence from OSD_FLOOR on and a quarter among the possible ways up decides
    without a vote; with no answer, a weak or impossible one, the ways up are voted on. When
    `osd` fails, the vote decides, the result is uncertain and `error` says why."""
    rgb = np.asarray(image.convert("RGB"))
    first = onnxtr_models.detect(models, rgb)
    if len(first.boxes) == 0:
        return Orientation(0.0, False, first)
    sideways = _sideways(first.boxes, image.width, image.height)
    ways = (0, 90, 180, 270) if sideways is None else (90, 270) if sideways else (0, 180)
    failure = None
    try:
        answer = None if osd is None else osd(image)
    except TesseractFailure as caught:
        answer, failure = None, str(caught)
    if answer is not None and answer.confidence >= OSD_FLOOR and answer.quarter in ways:
        _, found = _turned(models, image, rgb, first, answer.quarter)
        return _result(answer.quarter, found, False, _branch(ways))
    voted = _by_vote(models, image, rgb, first, ways)
    if failure is None:
        return voted
    return replace(voted, uncertain=True, error=failure)
