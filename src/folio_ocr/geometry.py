"""Turn a page image by an angle and report the exact input-to-output matrix.

Coordinates are continuous: pixel (i, j) covers [i, i+1) x [j, j+1) and y grows downwards.
Pillow samples AFFINE transforms at output pixel centres (x + 0.5, y + 0.5) and reads the
source in the same continuous convention, so the inverse matrix is passed through unchanged.
"""

import math

from PIL import Image

from folio_ocr.document import IDENTITY, Matrix

_QUADRANTS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}
_WHITE = {"L": 255, "RGB": (255, 255, 255)}


def _reduce(angle: float) -> float:
    if not math.isfinite(angle):
        raise ValueError("angle must be a finite number of degrees")
    return angle % 360.0


def _cos_sin(angle: float) -> tuple[float, float]:
    exact = {0.0: (1.0, 0.0), 90.0: (0.0, 1.0), 180.0: (-1.0, 0.0), 270.0: (0.0, -1.0)}
    if angle in exact:
        return exact[angle]
    rad = math.radians(angle)
    return math.cos(rad), math.sin(rad)


def rotation(width: int, height: int, angle: float) -> tuple[Matrix, tuple[int, int]]:
    """Matrix (input -> output) and output size for a counterclockwise turn by `angle`."""
    if width < 1 or height < 1:
        raise ValueError("image size must be at least 1x1")
    angle = _reduce(angle)
    c, s = _cos_sin(angle)
    if angle in (0.0, 180.0):
        w2, h2 = width, height
    elif angle in (90.0, 270.0):
        w2, h2 = height, width
    else:
        w2 = math.ceil(width * abs(c) + height * abs(s) - 1e-9)
        h2 = math.ceil(width * abs(s) + height * abs(c) - 1e-9)
    cx, cy, cx2, cy2 = width / 2, height / 2, w2 / 2, h2 / 2
    matrix: Matrix = (
        (c, s, cx2 - c * cx - s * cy),
        (-s, c, cy2 + s * cx - c * cy),
        (0.0, 0.0, 1.0),
    )
    return matrix, (w2, h2)


def straighten(image: Image.Image, angle: float) -> tuple[Image.Image, Matrix]:
    """Turn `image` counterclockwise by `angle` degrees, growing the canvas to fit.

    Returns the new image and the matrix mapping input points to output points.
    Multiples of 90 are lossless; other angles are resampled once with white fill.
    """
    if image.mode not in _WHITE:
        raise ValueError(f"unsupported image mode {image.mode!r}; use L or RGB")
    reduced = _reduce(angle)
    matrix, size = rotation(image.width, image.height, reduced)
    if reduced == 0.0:
        return image, IDENTITY
    if reduced in _QUADRANTS:
        return image.transpose(_QUADRANTS[reduced]), matrix
    (a, b, _), (d, e, _), _ = matrix
    # Inverse of the rotation (its transpose), mapping output points back to the input.
    cx, cy, cx2, cy2 = image.width / 2, image.height / 2, size[0] / 2, size[1] / 2
    data = (
        a,
        d,
        cx - a * cx2 - d * cy2,
        b,
        e,
        cy - b * cx2 - e * cy2,
    )
    out = image.transform(
        size,
        Image.Transform.AFFINE,
        data,
        resample=Image.Resampling.BICUBIC,
        fillcolor=_WHITE[image.mode],
    )
    return out, matrix
