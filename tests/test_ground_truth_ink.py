"""Ground-truth line boxes hold their line's ink on scanned and photographed fixtures."""

import functools
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from folio_ocr.document import IDENTITY
from folio_ocr.metrics import compose, gt_line_boxes

FIXTURES = Path(__file__).parent / "fixtures"
CONDITIONS = {"scan", "photo", "photo_exif", "upside_down"}
SHIFT_DOWN = ((1, 0, 0), (0, 1, 25.0), (0, 0, 1))


def _clean_name(name):
    return re.sub(r"-(scan|photo-exif|photo|upside_down)\.(jpg|png)$", "-clean.png", name)


@functools.cache
def _gt(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _targets():
    entries = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
    return [
        (e["file"], e["gt"])
        for e in entries
        if e["condition"] in CONDITIONS and _gt(e["gt"])["lines"]
    ]


TARGETS = _targets()
SCANS = [t for t in TARGETS if t[0].endswith("-scan.jpg")]
CLEANS = sorted({_clean_name(f) for f, _ in TARGETS})


@functools.cache
def _gray(name):
    with Image.open(FIXTURES / name) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("L"))


def _homography(gt):
    return tuple(tuple(float(v) for v in row) for row in gt["homography"])


def _project(matrix, xs, ys):
    h = np.asarray(matrix, dtype=float)
    w = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
    return (h[0, 0] * xs + h[0, 1] * ys + h[0, 2]) / w, (h[1, 0] * xs + h[1, 1] * ys + h[1, 2]) / w


def _line_ink(clean, box):
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(clean[y0:y1, x0:x1] < 128)
    return xs + x0 + 0.5, ys + y0 + 0.5


def _hit_fractions(clean, target, boxes, matrix, mapped_boxes=None):
    fractions = []
    for i, box in enumerate(boxes):
        xs, ys = _line_ink(clean, box)
        assert len(xs) > 0, f"no ink in line box {box}"
        u, v = _project(matrix, xs, ys)
        if mapped_boxes is not None:
            m = mapped_boxes[i]
            eps = 1e-6
            assert np.all(
                (u >= m[0] - eps) & (u <= m[2] + eps) & (v >= m[1] - eps) & (v <= m[3] + eps)
            )
        cols = np.clip(np.floor(u).astype(int), 0, target.shape[1] - 1)
        rows = np.clip(np.floor(v).astype(int), 0, target.shape[0] - 1)
        fractions.append(float(np.mean(target[rows, cols] < 160)))
    return fractions


def _setup(name, gt_name):
    gt = _gt(gt_name)
    clean_name = _clean_name(name)
    clean = _gray(clean_name)
    boxes = [tuple(line["clean_box"]) for line in gt["lines"]]
    clean_gt = _gt(clean_name.replace(".png", ".gt.json"))
    assert boxes == [tuple(line["clean_box"]) for line in clean_gt["lines"]]
    return gt, clean, _homography(gt), boxes


@pytest.mark.parametrize(("name", "gt_name"), TARGETS, ids=[t[0] for t in TARGETS])
def test_target_frame_is_the_gt_frame_after_exif_transpose(name, gt_name):
    gt = _gt(gt_name)
    assert _gray(name).shape == (gt["height"], gt["width"])


@pytest.mark.parametrize("clean_name", CLEANS)
def test_clean_boxes_hold_all_ink_of_the_clean_render(clean_name):
    gt = _gt(clean_name.replace(".png", ".gt.json"))
    clean = _gray(clean_name)
    covered = np.zeros(clean.shape, dtype=bool)
    for line in gt["lines"]:
        x0, y0, x1, y1 = line["clean_box"]
        assert (clean[y0:y1, x0:x1] < 128).any(), line["text"]
        covered[y0:y1, x0:x1] = True
    assert int(np.count_nonzero((clean < 128) & ~covered)) == 0


@pytest.mark.parametrize(("name", "gt_name"), TARGETS, ids=[t[0] for t in TARGETS])
def test_mapped_line_ink_lands_on_ink(name, gt_name):
    _, clean, h, boxes = _setup(name, gt_name)
    mapped = gt_line_boxes(boxes, h, IDENTITY)
    fractions = _hit_fractions(clean, _gray(name), boxes, h, mapped)
    assert min(fractions) >= 0.9


@pytest.mark.parametrize(("name", "gt_name"), TARGETS, ids=[t[0] for t in TARGETS])
def test_no_ink_outside_the_mapped_boxes_on_the_page(name, gt_name):
    _, clean, h, boxes = _setup(name, gt_name)
    target = _gray(name)
    rows, cols = np.mgrid[0 : target.shape[0], 0 : target.shape[1]]
    cx, cy = _project(np.linalg.inv(np.asarray(h)), cols + 0.5, rows + 0.5)
    ch, cw = clean.shape
    on_page = (cx >= 10) & (cx <= cw - 10) & (cy >= 10) & (cy <= ch - 10)
    allowed = np.zeros(target.shape, dtype=bool)
    for x0, y0, x1, y1 in gt_line_boxes(boxes, h, IDENTITY):
        allowed[
            max(math.floor(y0 - 4), 0) : math.ceil(y1 + 4),
            max(math.floor(x0 - 4), 0) : math.ceil(x1 + 4),
        ] = True
    dark = (target < 100) & on_page
    assert int(dark.sum()) > 0
    assert int((dark & ~allowed).sum()) == 0


@pytest.mark.parametrize(("name", "gt_name"), TARGETS, ids=[t[0] for t in TARGETS])
def test_mapping_shifted_by_a_line_misses_the_ink(name, gt_name):
    _, clean, h, boxes = _setup(name, gt_name)
    fractions = _hit_fractions(clean, _gray(name), boxes, compose(SHIFT_DOWN, h))
    assert sum(fractions) / len(fractions) < 0.2


def _straightening(h, width, height):
    ri = np.linalg.inv(np.asarray(h)[:2, :2])
    t = np.array([width / 2, height / 2]) - ri @ np.array([width / 2, height / 2])
    return tuple(
        tuple(float(v) for v in row)
        for row in ((ri[0, 0], ri[0, 1], t[0]), (ri[1, 0], ri[1, 1], t[1]), (0, 0, 1))
    )


@pytest.mark.parametrize(("name", "gt_name"), SCANS, ids=[t[0] for t in SCANS])
def test_straightened_scan_boxes_hold_their_ink(name, gt_name):
    gt, clean, h, boxes = _setup(name, gt_name)
    arr = _gray(name)
    t = _straightening(h, gt["width"], gt["height"])
    straight = np.asarray(
        Image.fromarray(arr).transform(
            arr.shape[::-1],
            Image.Transform.AFFINE,
            data=tuple(np.linalg.inv(t)[:2].ravel()),
            resample=Image.Resampling.BILINEAR,
            fillcolor=255,
        )
    )
    for box, mapped in zip(boxes, gt_line_boxes(boxes, h, t), strict=True):
        assert mapped[2] - mapped[0] == pytest.approx(box[2] - box[0], abs=0.5)
        assert mapped[3] - mapped[1] == pytest.approx(box[3] - box[1], abs=0.5)
    assert min(_hit_fractions(clean, straight, boxes, compose(t, h))) >= 0.9
