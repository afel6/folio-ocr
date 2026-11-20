"""Tests for the model request: prompt, token budget, line crop and JSON body."""

import base64
import hashlib
import io
import json
import math

import pytest
from PIL import Image

from folio_ocr import model_request as mr

PROMPT = (
    "Transcribe the text in this image exactly as written. Do not correct, translate or "
    "explain. Keep digits in the script they are written in. Write ? for an unreadable "
    "character. Reply with an empty message if there is no text."
)
PROMPT_HEX = "9f48cab462e9a79d9f1c8d27e81cb8b9cbdf1b8dddcd0b5f2ec6a12e4696c6d1"


def test_prompt_and_hash_are_pinned():
    assert mr.PROMPT == PROMPT
    assert mr.PROMPT_SHA256 == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    assert mr.PROMPT_SHA256 == PROMPT_HEX


def test_max_tokens_latin_rate():
    assert mr.max_tokens(886, 70, "en") == 16 + math.ceil(3 * 886 / 70)


@pytest.mark.parametrize("languages", ["ar", "en+ar"])
def test_max_tokens_arabic_rate(languages):
    assert mr.max_tokens(886, 70, languages) == 16 + math.ceil(5 * 886 / 70)


def test_max_tokens_zero_height_and_monotonic_width():
    assert mr.max_tokens(100, 0, "en") == 16 + 300
    counts = [mr.max_tokens(w, 40, "en") for w in range(0, 500, 7)]
    assert counts == sorted(counts)
    assert counts[0] == 16


def _page(width=100, height=80):
    return Image.new("L", (width, height), 0)


def test_crop_grows_by_padding_on_every_side():
    out = mr.crop(_page(), (30, 30, 50, 40))
    assert out.size == (20 + 2 * mr.PADDING_PX, 10 + 2 * mr.PADDING_PX)
    assert out.mode == "L"


def test_crop_clamps_at_the_corners():
    assert mr.crop(_page(), (0, 0, 10, 10)).size == (10 + mr.PADDING_PX, 10 + mr.PADDING_PX)
    far = mr.crop(_page(), (90, 70, 100, 80))
    assert far.size == (10 + mr.PADDING_PX, 10 + mr.PADDING_PX)


def test_crop_float_box_rounds_outward():
    out = mr.crop(_page(), (30.5, 30.5, 50.2, 40.2))
    x0 = math.floor(30.5 - mr.PADDING_PX)
    x1 = math.ceil(50.2 + mr.PADDING_PX)
    y1 = math.ceil(40.2 + mr.PADDING_PX)
    assert out.size == (x1 - x0, y1 - x0)


def test_crop_takes_the_right_pixels():
    page = _page()
    page.putpixel((40, 35), 200)
    out = mr.crop(page, (30, 30, 50, 40))
    origin = 30 - mr.PADDING_PX
    assert out.getpixel((40 - origin, 35 - origin)) == 200


def test_crop_converts_to_grey():
    rgb = Image.new("RGB", (50, 50), (255, 0, 0))
    assert mr.crop(rgb, (10, 10, 20, 20)).mode == "L"


@pytest.mark.parametrize(
    "box", [(200, 10, 210, 20), (10, 200, 20, 210), (-50, 10, -20, 20), (10, -50, 20, -20)]
)
def test_crop_outside_the_image_raises(box):
    with pytest.raises(ValueError):
        mr.crop(_page(), box)


def test_altered_adds_a_white_margin():
    base = Image.new("L", (30, 21), 7)
    out = mr.altered(base)
    margin = math.ceil(0.5 * 21)
    assert out.size == (30 + 2 * margin, 21 + 2 * margin)
    assert out.getpixel((0, 0)) == 255
    assert out.getpixel((out.width - 1, out.height - 1)) == 255
    assert out.getpixel((margin - 1, margin + 3)) == 255
    assert out.getpixel((margin + 30, margin + 3)) == 255
    assert out.crop((margin, margin, margin + 30, margin + 21)).tobytes() == base.tobytes()


def test_png_round_trips_as_grey():
    base = Image.new("L", (9, 5), 0)
    base.putpixel((3, 2), 99)
    back = Image.open(io.BytesIO(mr.png(base)))
    assert back.mode == "L"
    assert back.size == (9, 5)
    assert back.tobytes() == base.tobytes()


def test_request_body_shape():
    data = b"\x89PNGfake"
    body = json.loads(mr.request_body("m", data, 77))
    assert set(body) == {
        "model",
        "messages",
        "temperature",
        "seed",
        "max_tokens",
        "logprobs",
        "stream",
    }
    assert (body["model"], body["temperature"], body["seed"], body["max_tokens"]) == (
        "m",
        0,
        42,
        77,
    )
    assert body["logprobs"] is True
    assert body["stream"] is False
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


def test_request_body_sends_non_ascii_names_as_utf8():
    name = "mod\N{LATIN SMALL LETTER E WITH ACUTE}l"
    raw = mr.request_body(name, b"x", 5)
    assert name.encode("utf-8") in raw
    assert b"\\u" not in raw


def test_arguments_hold_every_constant_without_separators():
    args = mr.arguments()
    for item in (
        f"prompt_sha256={PROMPT_HEX}",
        "temperature=0",
        "seed=42",
        "logprobs=true",
        "stream=false",
        "max_tokens_base=16",
        "max_tokens_latin=3",
        "max_tokens_arabic=5",
    ):
        assert item in args
    assert all(" " not in a and "|" not in a for a in args)


def test_crop_constants():
    assert mr.crop_constants() == {
        "model_padding_px": 6,
        "model_scale": 1,
        "model_image": "png-8bit-grey",
        "model_two_read_margin": 0.5,
    }
