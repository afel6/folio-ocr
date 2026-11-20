"""What the model server is sent for one text line: the fixed prompt, the line crop and the
request constants. Every constant here is part of the model's calibration setup key."""

import base64
import hashlib
import io
import json
import math

from PIL import Image, ImageOps

from folio_ocr.document import Box

PROMPT = (
    "Transcribe the text in this image exactly as written. Do not correct, translate or "
    "explain. Keep digits in the script they are written in. Write ? for an unreadable "
    "character. Reply with an empty message if there is no text."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
TEMPERATURE = 0
SEED = 42
# max_tokens = BASE_TOKENS + ceil(rate x crop width / crop height); the rate is per expected
# script: Arabic needs about one token per letter, Latin text fewer.
BASE_TOKENS = 16
LATIN_TOKENS_PER_HEIGHT = 3
ARABIC_TOKENS_PER_HEIGHT = 5
PADDING_PX = 6  # grey pixels kept around the line box
SCALE = 1  # crops are sent at the page image's own resolution
PNG_ENCODING = "png-8bit-grey"
# The second read of FOLIO_TWO_READ sees the crop inside a white margin of this many line
# heights on every side.
TWO_READ_MARGIN = 0.5


def max_tokens(width: int, height: int, languages: str) -> int:
    """Token budget of one reply: English documents get the Latin rate, others the Arabic."""
    rate = LATIN_TOKENS_PER_HEIGHT if languages == "en" else ARABIC_TOKENS_PER_HEIGHT
    return BASE_TOKENS + math.ceil(rate * width / max(height, 1))


def crop(gray: Image.Image, box: Box) -> Image.Image:
    """The line's pixels grown by PADDING_PX and clamped to the image, as 8-bit grey.
    Raises ValueError for a box with no pixels inside the image."""
    x0 = max(0, math.floor(box[0] - PADDING_PX))
    y0 = max(0, math.floor(box[1] - PADDING_PX))
    x1 = min(gray.width, math.ceil(box[2] + PADDING_PX))
    y1 = min(gray.height, math.ceil(box[3] + PADDING_PX))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the line box holds no pixels of the page")
    return gray.crop((x0, y0, x1, y1)).convert("L")


def altered(image: Image.Image) -> Image.Image:
    """The crop inside a white margin of TWO_READ_MARGIN line heights (the second read)."""
    border = max(1, math.ceil(TWO_READ_MARGIN * image.height))
    return ImageOps.expand(image.convert("L"), border=border, fill=255)


def png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("L").save(out, format="PNG")
    return out.getvalue()


def request_body(model_name: str, image_png: bytes, tokens: int) -> bytes:
    """The JSON body of one /chat/completions request for one line image."""
    data_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
    content = [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": tokens,
        "logprobs": True,
        "stream": False,
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def arguments() -> tuple[str, ...]:
    """The request constants as setup-key arguments."""
    return (
        f"prompt_sha256={PROMPT_SHA256}",
        f"temperature={TEMPERATURE}",
        f"seed={SEED}",
        "logprobs=true",
        "stream=false",
        f"max_tokens_base={BASE_TOKENS}",
        f"max_tokens_latin={LATIN_TOKENS_PER_HEIGHT}",
        f"max_tokens_arabic={ARABIC_TOKENS_PER_HEIGHT}",
    )


def crop_constants() -> dict[str, str | int | float]:
    """The crop constants as preprocessing-hash entries."""
    return {
        "model_padding_px": PADDING_PX,
        "model_scale": SCALE,
        "model_image": PNG_ENCODING,
        "model_two_read_margin": TWO_READ_MARGIN,
    }
