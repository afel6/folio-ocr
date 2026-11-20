"""Image handling for the fixtures: rendering, ink boxes, homographies, scan and photo conditions."""

from __future__ import annotations

import io
import math

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter

DPI = 300
SCALE = DPI / 72
INK_LEVEL = 160
SCAN_QUALITY, SCAN_NOISE = 60, 6.0
PHOTO_QUALITY, PHOTO_NOISE = 85, 3.0
PHOTO_BACKGROUND = (88, 80, 72)
PHOTO_TINT = np.array([1.0, 0.97, 0.90], np.float32)


def render_pdf(data: bytes) -> Image.Image:
    """Rasterise page 1 of a PDF at 300 dpi as a grayscale image."""
    doc = pdfium.PdfDocument(data)
    try:
        return doc[0].render(scale=SCALE, grayscale=True).to_pil().copy()
    finally:
        doc.close()


def ink_box(pixels: np.ndarray, rect: tuple[int, int, int, int]) -> list[int]:
    """Bounds [x0, y0, x1, y1] (exclusive end) of dark pixels inside rect."""
    x0, y0, x1, y1 = rect
    ink = pixels[y0:y1, x0:x1] < INK_LEVEL
    if not ink.any():
        raise ValueError(f"no ink inside cell {rect}")
    rows, cols = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    if rows[0] == 0 or cols[0] == 0 or rows[-1] == ink.shape[0] - 1 or cols[-1] == ink.shape[1] - 1:
        raise ValueError(f"ink touches the edge of cell {rect}")
    return [x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1]


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def _rotation(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _project(h: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    q = h @ np.array([[x, y, 1.0] for x, y in points]).T
    return np.stack([q[0] / q[2], q[1] / q[2]], axis=1)


def _homography_from_points(src, dst) -> np.ndarray:
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    return np.append(np.linalg.solve(np.array(rows, float), np.array(rhs, float)), 1.0).reshape(
        3, 3
    )


def warp(img: Image.Image, h: np.ndarray, size: tuple[int, int], fill) -> Image.Image:
    """Apply homography h (input pixels to output pixels) and return the output image."""
    inverse = np.linalg.inv(h)
    inverse = inverse / inverse[2, 2]
    return img.transform(
        size,
        Image.Transform.PERSPECTIVE,
        tuple(inverse.flatten()[:8]),
        Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def transform_box(h: np.ndarray, box: list[int], size: tuple[int, int]) -> list[int]:
    """Axis-aligned bounds of the transformed box corners, clipped to the image."""
    x0, y0, x1, y1 = box
    pts = np.round(_project(h, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), 6)
    return [
        max(0, math.floor(pts[:, 0].min())),
        max(0, math.floor(pts[:, 1].min())),
        min(size[0], math.ceil(pts[:, 0].max())),
        min(size[1], math.ceil(pts[:, 1].max())),
    ]


def _signed(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(low, high))


def _noisy(img: Image.Image, rng: np.random.Generator, sigma: float, gain=None) -> Image.Image:
    arr = np.asarray(img, np.float32)
    if gain is not None:
        arr = arr * gain
    arr = arr + rng.standard_normal(arr.shape, dtype=np.float32) * np.float32(sigma)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def scan_condition(clean: Image.Image, rng: np.random.Generator):
    """Slightly rotated, blurred, noisy grayscale page: returns (image, H)."""
    w, h = clean.size
    rad = math.radians(_signed(rng, 0.5, 2.0))
    size = (
        math.ceil(abs(w * math.cos(rad)) + abs(h * math.sin(rad))),
        math.ceil(abs(w * math.sin(rad)) + abs(h * math.cos(rad))),
    )
    homography = (
        _translation(size[0] / 2, size[1] / 2) @ _rotation(rad) @ _translation(-w / 2, -h / 2)
    )
    page = warp(clean.convert("L"), homography, size, 255)
    page = page.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0)))
    return _noisy(page, rng, SCAN_NOISE), homography


def photo_condition(clean: Image.Image, rng: np.random.Generator):
    """Keystoned, rotated page on a dark table with uneven warm light: returns (image, H)."""
    w, h = clean.size
    k = rng.uniform(0.05, 0.10)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    keystone = _homography_from_points(
        corners, [(k * w / 2, 0), (w - k * w / 2, 0), (w, h), (0, h)]
    )
    rotate = _translation(w / 2, h / 2) @ _rotation(math.radians(_signed(rng, 2, 5)))
    tilted = rotate @ _translation(-w / 2, -h / 2) @ keystone
    pts = _project(tilted, corners)
    (lo_x, lo_y), (hi_x, hi_y) = pts.min(axis=0), pts.max(axis=0)
    margin_x, margin_y = 0.04 * (hi_x - lo_x), 0.04 * (hi_y - lo_y)
    homography = _translation(margin_x - lo_x, margin_y - lo_y) @ tilted
    size = (math.ceil(hi_x - lo_x + 2 * margin_x), math.ceil(hi_y - lo_y + 2 * margin_y))
    page = warp(clean.convert("RGB"), homography, size, PHOTO_BACKGROUND)
    low, angle = rng.uniform(0.70, 0.80), rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float32)
    along = (xx - size[0] / 2) * math.cos(angle) + (yy - size[1] / 2) * math.sin(angle)
    along = (along - along.min()) / (along.max() - along.min())
    gain = (low + (1 - low) * along)[..., None] * PHOTO_TINT
    return _noisy(page, rng, PHOTO_NOISE, gain), homography


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(img: Image.Image, quality: int, exif: Image.Exif | None = None) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, **({"exif": exif} if exif else {}))
    return buf.getvalue()
