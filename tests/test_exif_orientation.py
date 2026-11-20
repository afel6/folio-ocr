"""Photos tagged with an EXIF orientation are read upright, whatever the tag says."""

import io

import pytest
from PIL import Image, ImageOps

from folio_ocr.ingest import load_frame, open_image

T = Image.Transpose
COLOURS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 0, 0),
    (255, 255, 255),
]
# orientation tag -> the transpose that undoes how the camera stored the pixels
UPRIGHTING = {
    2: T.FLIP_LEFT_RIGHT,
    3: T.ROTATE_180,
    4: T.FLIP_TOP_BOTTOM,
    5: T.TRANSPOSE,
    6: T.ROTATE_270,
    7: T.TRANSVERSE,
    8: T.ROTATE_90,
}
INVERSE = {T.ROTATE_270: T.ROTATE_90, T.ROTATE_90: T.ROTATE_270}


def blocks() -> Image.Image:
    img = Image.new("RGB", (64, 32))
    for i, colour in enumerate(COLOURS):
        img.paste(colour, ((i % 4) * 16, (i // 4) * 16, (i % 4) * 16 + 16, (i // 4) * 16 + 16))
    return img


def stored_as(orientation: int) -> bytes:
    img = blocks()
    if orientation in UPRIGHTING:
        method = UPRIGHTING[orientation]
        img = img.transpose(INVERSE.get(method, method))
    exif = Image.Exif()
    exif[274] = orientation
    out = io.BytesIO()
    img.save(out, "JPEG", quality=100, exif=exif)
    return out.getvalue()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_every_orientation_is_read_upright(orientation):
    data = stored_as(orientation)
    assert ImageOps.exif_transpose(Image.open(io.BytesIO(data))).size == (64, 32)
    frame = load_frame(open_image(data, "jpeg"), 0)
    assert frame.size == (64, 32)
    for i, colour in enumerate(COLOURS):
        pixel = frame.convert("RGB").getpixel(((i % 4) * 16 + 8, (i // 4) * 16 + 8))
        assert all(abs(a - b) <= 8 for a, b in zip(pixel, colour, strict=True)), (i, pixel)
