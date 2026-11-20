import math

import pytest
from PIL import Image

from folio_ocr.document import IDENTITY
from folio_ocr.geometry import rotation, straighten
from folio_ocr.metrics import map_box

W, H = 240, 120
BLOCK = (150, 30, 155, 35)  # right of centre, above it
ANGLES = [3, -3, 10, 30, 178, 182, 359.5]


def apply(m, x, y):
    return (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])


def block_image(mode="L"):
    white = 255 if mode == "L" else (255, 255, 255)
    black = 0 if mode == "L" else (0, 0, 0)
    img = Image.new(mode, (W, H), white)
    img.paste(black, BLOCK)
    return img


def dark_pixels(img):
    g = img.convert("L")
    w, h = g.size
    px = g.load()
    return [(x, y) for y in range(h) for x in range(w) if px[x, y] < 128]


def centroid(pts):
    return (
        sum(x + 0.5 for x, _ in pts) / len(pts),
        sum(y + 0.5 for _, y in pts) / len(pts),
    )


@pytest.mark.parametrize("angle", [90, 180, 270, -90, 450, 720])
def test_quadrants_are_lossless_and_matrix_is_exact(angle):
    img = Image.new("L", (3, 2))
    img.putdata([10, 20, 30, 40, 50, 60])
    out, m = straighten(img, angle)
    k = (angle % 360) // 90
    expected = [
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    ][k]
    if expected is None:
        assert out.tobytes() == img.tobytes()
        assert m == IDENTITY
        return
    assert out.tobytes() == img.transpose(expected).tobytes()
    assert out.size == ((2, 3) if k != 2 else (3, 2))
    for y in range(2):
        for x in range(3):
            u, v = apply(m, x + 0.5, y + 0.5)
            assert u == int(u) + 0.5 and v == int(v) + 0.5
            assert out.getpixel((int(u), int(v))) == img.getpixel((x, y))


def test_quadrant_formulas():
    _, m = straighten(Image.new("L", (3, 2)), 90)
    assert apply(m, 1.0, 0.5) == (0.5, 2.0)  # (x, y) -> (y, W - x)


@pytest.mark.parametrize("angle", ANGLES)
def test_block_lands_where_matrix_says(angle):
    img = block_image()
    out, m = straighten(img, angle)
    assert out.size == rotation(W, H, angle)[1]
    pts = dark_pixels(out)
    cx, cy = apply(m, 152.5, 32.5)
    mx, my = centroid(pts)
    assert abs(mx - cx) < 0.75 and abs(my - cy) < 0.75
    x0, y0, x1, y1 = map_box(m, tuple(float(v) for v in BLOCK))
    assert all(x0 - 1 <= x < x1 + 1 and y0 - 1 <= y < y1 + 1 for x, y in pts)


@pytest.mark.parametrize("angle", ANGLES)
def test_corners_inside_and_touch_every_side(angle):
    out, m = straighten(block_image(), angle)
    w2, h2 = out.size
    cs = [apply(m, x, y) for x, y in ((0, 0), (W, 0), (W, H), (0, H))]
    eps = 1e-6
    assert all(-eps <= u <= w2 + eps and -eps <= v <= h2 + eps for u, v in cs)
    assert min(u for u, _ in cs) < 1 and max(u for u, _ in cs) > w2 - 1
    assert min(v for _, v in cs) < 1 and max(v for _, v in cs) > h2 - 1


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_fill_is_white_mode_kept_input_untouched(mode):
    img = block_image(mode)
    before = img.tobytes()
    out, _ = straighten(img, 30)
    assert out.mode == mode and img.tobytes() == before
    white = 255 if mode == "L" else (255, 255, 255)
    w2, h2 = out.size
    for pt in ((0, 0), (w2 - 1, 0), (0, h2 - 1), (w2 - 1, h2 - 1)):
        assert out.getpixel(pt) == white


@pytest.mark.parametrize("angle", [90, 10])
def test_turns_counterclockwise(angle):
    _, m = straighten(block_image(), angle)
    _, y = apply(m, 152.5, H / 2)
    assert y < rotation(W, H, angle)[1][1] / 2  # right of centre moves up


def test_size_rounding():
    assert rotation(W, H, 30)[1] == (
        math.ceil(W * math.cos(math.pi / 6) + H * 0.5 - 1e-9),
        math.ceil(W * 0.5 + H * math.cos(math.pi / 6) - 1e-9),
    )
    assert rotation(10, 10, 360.0)[1] == (10, 10)


def test_errors_and_identity():
    img = block_image()
    out, m = straighten(img, 0)
    assert m == IDENTITY and out.tobytes() == img.tobytes()
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            straighten(img, bad)
    with pytest.raises(ValueError):
        straighten(Image.new("P", (4, 4)), 10)
    with pytest.raises(ValueError):
        rotation(0, 5, 3)
    with pytest.raises(ValueError):
        rotation(5, 0, 3)
