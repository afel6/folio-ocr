"""Image degradations for the development variants: each returns the image, its homography and params."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from fixture_images import PHOTO_BACKGROUND, PHOTO_TINT, jpeg_bytes, png_bytes, warp
from PIL import Image, ImageFilter

CONDITION_MIX = {
    "clean": 0.10,
    "scan": 0.25,
    "photo": 0.20,
    "lowres": 0.15,
    "fax": 0.10,
    "faded": 0.10,
    "rotated": 0.10,
}
GRID_ROTATED = 0.50
Result = tuple[Image.Image, np.ndarray, dict, str]


def condition_mix(template: str) -> dict[str, float]:
    """Condition probabilities for a template; grids are mostly turned by a quarter."""
    if template != "grid":
        return dict(CONDITION_MIX)
    rest = (1 - GRID_ROTATED) / (1 - CONDITION_MIX["rotated"])
    return {k: GRID_ROTATED if k == "rotated" else v * rest for k, v in CONDITION_MIX.items()}


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def _rotation(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _perspective(src: list, dst: list) -> np.ndarray:
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    return np.append(np.linalg.solve(np.array(rows, float), np.array(rhs, float)), 1.0).reshape(
        3, 3
    )


def _project(h: np.ndarray, points: list) -> np.ndarray:
    q = h @ np.array([[x, y, 1.0] for x, y in points]).T
    return np.stack([q[0] / q[2], q[1] / q[2]], axis=1)


def _signed(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(low, high))


def _tilt(img: Image.Image, keystone: float, radians: float, fill, margin: float):
    """Keystone then rotate the page on a canvas that holds it: returns (image, homography)."""
    w, h = img.size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    top = [(keystone * w / 2, 0), (w - keystone * w / 2, 0), (w, h), (0, h)]
    tilted = _rotation(radians) @ _translation(-w / 2, -h / 2) @ _perspective(corners, top)
    tilted = _translation(w / 2, h / 2) @ tilted
    pts = _project(tilted, corners)
    (lo_x, lo_y), (hi_x, hi_y) = pts.min(axis=0), pts.max(axis=0)
    mx, my = margin * (hi_x - lo_x), margin * (hi_y - lo_y)
    homography = _translation(mx - lo_x, my - lo_y) @ tilted
    size = (math.ceil(hi_x - lo_x + 2 * mx), math.ceil(hi_y - lo_y + 2 * my))
    return warp(img, homography, size, fill), homography


def _noisy(img: Image.Image, rng: np.random.Generator, sigma: float, gain=None) -> Image.Image:
    arr = np.asarray(img, np.float32)
    if gain is not None:
        arr = arr * gain
    arr = arr + rng.standard_normal(arr.shape, dtype=np.float32) * np.float32(sigma)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _speckled(img: Image.Image, rng: np.random.Generator, prob: float) -> Image.Image:
    """Salt and pepper: each pixel turns black or white with probability prob/2 each."""
    arr = np.array(img)
    draw = rng.random(arr.shape[:2], dtype=np.float32)
    arr[draw < prob / 2] = 0
    arr[(draw >= prob / 2) & (draw < prob)] = 255
    return Image.fromarray(arr)


def _blurred(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(sigma)) if sigma > 0.05 else img


def _scaled(img: Image.Image, scale: float, resample: Image.Resampling):
    size = (round(img.size[0] * scale), round(img.size[1] * scale))
    return img.resize(size, resample), np.diag([scale, scale, 1.0])


def rotate_quarter(img: Image.Image, degrees: int) -> tuple[Image.Image, np.ndarray]:
    """Exact counter-clockwise quarter turn (90, 180 or 270) and its homography."""
    w, h = img.size
    turns = {
        90: (Image.Transpose.ROTATE_90, [[0, 1, 0], [-1, 0, w], [0, 0, 1]]),
        180: (Image.Transpose.ROTATE_180, [[-1, 0, w], [0, -1, h], [0, 0, 1]]),
        270: (Image.Transpose.ROTATE_270, [[0, -1, h], [1, 0, 0], [0, 0, 1]]),
    }
    if degrees not in turns:
        raise ValueError(f"not a quarter turn: {degrees}")
    method, matrix = turns[degrees]
    return img.transpose(method), np.array(matrix, float)


def _scan(clean, rng) -> Result:
    deg = _signed(rng, 0.3, 4.0)
    params = {"rotation_deg": deg, "blur_sigma": float(rng.uniform(0.3, 1.8))}
    params["noise_sigma"] = float(rng.uniform(3, 18))
    params["speckle"] = float(rng.uniform(0, 0.004)) if rng.random() < 0.5 else 0.0
    params["jpeg_quality"] = int(rng.integers(25, 81))
    page, h = _tilt(clean.convert("L"), 0.0, math.radians(deg), 255, 0.0)
    page = _noisy(_blurred(page, params["blur_sigma"]), rng, params["noise_sigma"])
    if params["speckle"]:
        page = _speckled(page, rng, params["speckle"])
    return page, h, params, "jpg"


def _photo(clean, rng) -> Result:
    params = {
        "keystone": float(rng.uniform(0.03, 0.16)),
        "rotation_deg": _signed(rng, 1, 8),
        "light_low": float(rng.uniform(0.5, 0.85)),
        "light_angle": float(rng.uniform(0, 2 * math.pi)),
        "noise_sigma": float(rng.uniform(3, 14)),
        "blur_sigma": float(rng.uniform(0, 1.2)),
        "jpeg_quality": int(rng.integers(35, 86)),
    }
    rotation = math.radians(params["rotation_deg"])
    page, h = _tilt(clean.convert("RGB"), params["keystone"], rotation, PHOTO_BACKGROUND, 0.04)
    w, ht = page.size
    yy, xx = np.mgrid[0:ht, 0:w].astype(np.float32)
    angle = params["light_angle"]
    along = (xx - w / 2) * math.cos(angle) + (yy - ht / 2) * math.sin(angle)
    along = (along - along.min()) / (along.max() - along.min())
    gain = (params["light_low"] + (1 - params["light_low"]) * along)[..., None] * PHOTO_TINT
    page = _noisy(_blurred(page, params["blur_sigma"]), rng, params["noise_sigma"], gain)
    return page, h, params, "jpg"


def _lowres(clean, rng) -> Result:
    box = bool(rng.random() < 0.5)
    params = {"scale": float(rng.uniform(0.33, 0.70)), "resample": "box" if box else "bilinear"}
    params["blur_sigma"] = float(rng.uniform(0, 0.8)) if rng.random() < 0.5 else 0.0
    params["jpeg_quality"] = int(rng.integers(40, 91))
    resample = Image.Resampling.BOX if box else Image.Resampling.BILINEAR
    page, h = _scaled(clean.convert("L"), params["scale"], resample)
    return _blurred(page, params["blur_sigma"]), h, params, "jpg"


def _fax(clean, rng) -> Result:
    params = {
        "scale": float(rng.uniform(0.5, 0.7)),
        "blur_sigma": float(rng.uniform(0.4, 1.2)),
        "threshold": int(rng.integers(110, 171)),
        "speckle": float(rng.uniform(0.0005, 0.003)),
    }
    page, h = _scaled(clean.convert("L"), params["scale"], Image.Resampling.BILINEAR)
    page = _blurred(page, params["blur_sigma"]).point(
        lambda v: 255 if v >= params["threshold"] else 0
    )
    return _speckled(page.convert("L"), rng, params["speckle"]), h, params, "png"


def _faded(clean, rng) -> Result:
    params = {
        "ink_strength": float(rng.uniform(0.25, 0.6)),
        "noise_sigma": float(rng.uniform(2, 8)),
        "jpeg_quality": int(rng.integers(50, 91)),
    }
    arr = 255 - (255 - np.asarray(clean.convert("L"), np.float32)) * params["ink_strength"]
    page = _noisy(Image.fromarray(np.rint(arr).astype(np.uint8)), rng, params["noise_sigma"])
    return page, np.eye(3), params, "jpg"


def _rotated(clean, rng) -> Result:
    quarter = int(rng.choice([90, 180, 270]))
    skew = _signed(rng, 0.2, 1.5) if rng.random() < 0.5 else 0.0
    params = {
        "quarter_deg": quarter,
        "skew_deg": skew,
        "noise_sigma": float(rng.uniform(0.5, 4)),
        "jpeg_quality": int(rng.integers(60, 91)),
    }
    page, h = rotate_quarter(clean.convert("L"), quarter)
    if skew:
        page, extra = _tilt(page, 0.0, math.radians(skew), 255, 0.0)
        h = extra @ h
    return _noisy(page, rng, params["noise_sigma"]), h, params, "jpg"


def _clean(clean, rng) -> Result:
    return clean, np.eye(3), {}, "png"


_CONDITIONS: dict[str, Callable[[Image.Image, np.random.Generator], Result]] = {
    "clean": _clean,
    "scan": _scan,
    "photo": _photo,
    "lowres": _lowres,
    "fax": _fax,
    "faded": _faded,
    "rotated": _rotated,
}


def degrade(clean: Image.Image, condition: str, rng: np.random.Generator) -> Result:
    """Apply one condition: returns (image, homography clean px -> output px, params, extension)."""
    if condition not in _CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    return _CONDITIONS[condition](clean, rng)


def encode_image(img: Image.Image, ext: str, params: dict) -> bytes:
    """PNG bytes, or JPEG bytes at the recorded quality."""
    return png_bytes(img) if ext == "png" else jpeg_bytes(img, params["jpeg_quality"])
