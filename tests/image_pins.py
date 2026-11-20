"""Byte-level pins of how folio-ocr decodes every fixture image and a set of crafted ones: the
frames, the PNGs the web API serves, and the no-server Document JSON for a few of them.
`main()` rewrites tests/data/image_pins.json."""

import io
import json
from pathlib import Path

import pdf_pins as pins
from PIL import Image, ImageCms

from folio_ocr import calibration, pipeline

PIN_FILE = Path(__file__).parent / "data" / "image_pins.json"
EXTENSIONS = ("png", "jpg", "jpeg", "tif", "tiff", "webp")
DOCUMENT_NAMES = ("en-invoice-clean.png", "en-letter-photo-exif.jpg", "en-invoice-scan.jpg")
SIZE = (300, 200)


def _save(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _pattern(mode: str = "RGB") -> Image.Image:
    base = Image.linear_gradient("L").resize(SIZE)
    rgb = Image.merge(
        "RGB", (base, base.transpose(Image.Transpose.FLIP_LEFT_RIGHT), base.rotate(90))
    )
    rgb.paste((255, 255, 255), (10, 10, 120, 90))
    return rgb.convert(mode)


def _frames_tiff() -> bytes:
    frames = [_pattern().rotate(a) for a in (0, 90, 180)]
    return _save(frames[0], "TIFF", save_all=True, append_images=frames[1:])


def _gray16_tiff() -> bytes:
    img = Image.new("I;16", SIZE)
    img.putdata([(x * 300 + y * 7) % 65536 for y in range(SIZE[1]) for x in range(SIZE[0])])
    return _save(img, "TIFF")


def _half_transparent() -> bytes:
    img = _pattern("RGBA")
    img.putalpha(Image.linear_gradient("L").resize(SIZE))
    return _save(img, "PNG")


def crafted() -> dict[str, bytes]:
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    icc = icc[:24] + bytes(12) + icc[36:]  # the header's creation date is the clock
    exif = Image.Exif()
    exif[0x0112] = 6
    return {
        "crafted-rgb-trns.png": _save(_pattern(), "PNG", transparency=(255, 255, 255)),
        "crafted-l-trns.png": _save(_pattern("L"), "PNG", transparency=0),
        "crafted-icc.jpg": _save(_pattern(), "JPEG", icc_profile=icc),
        "crafted-exif6.jpg": _save(_pattern(), "JPEG", exif=exif.tobytes()),
        "crafted-3frames.tif": _frames_tiff(),
        "crafted-gray16.tif": _gray16_tiff(),
        "crafted-rgba-half.png": _half_transparent(),
        "crafted-1bit.png": _save(_pattern("1"), "PNG"),
        "crafted-lossless.webp": _save(_pattern(), "WEBP", lossless=True),
    }


def inputs() -> dict[str, bytes]:
    paths = sorted(
        (p for e in EXTENSIONS for p in pins.FIXTURES.glob(f"*.{e}")), key=lambda p: p.name
    )
    return {p.name: p.read_bytes() for p in paths} | crafted()


def frame_count(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as img:
        return getattr(img, "n_frames", 1)


def frames(data: bytes, count: int) -> dict:
    out = {}
    for n in range(1, count + 1):
        img = pipeline.page_image(data, n)
        info = {k: img.info[k] for k in pins.PREVIEW_INFO_KEYS if k in img.info}
        out[str(n)] = {
            "mode": img.mode,
            "size": list(img.size),
            "sha256": pins.sha(img.tobytes()),
            "info": pins.sha(repr(info).encode("utf-8")),
        }
    return out


def compute() -> dict:
    with pins.empty_taus():
        return _compute()


def _compute() -> dict:
    files, weights = {}, {}
    for name, data in inputs().items():
        count = frame_count(data)
        entry = {
            "frame": frames(data, count),
            "png": pins.served_pngs(data, name, count),
            "documents": {},
        }
        if name in DOCUMENT_NAMES:
            entry["documents"]["en"], found = pins.document_pin(data, name, "en")
            if found:
                weights["en"] = found
        files[name] = entry
    ocr_env = {
        "tesseract": pins.tesseract_decided(),
        "weights": weights,
        "taus": calibration.table_sha256({}),
    }
    return {"build": pins.build_info(), "ocr_env": ocr_env, "files": files}


def main() -> None:
    PIN_FILE.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
