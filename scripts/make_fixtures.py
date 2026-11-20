"""Writes the seeded English and Arabic OCR fixtures, their ground truth and a manifest.

Every page is drawn by fpdf2 (with text shaping) into a PDF; images are rasterised from
that PDF at 300 dpi. Clean PNGs are plain renders; scans get rotation, blur, noise and
JPEG 60; photos get keystone, rotation, a light gradient, noise and JPEG. The blank scan
starts from a white page. The visual-order PDF is drawn unshaped, one character at a time.
Usage: python scripts/make_fixtures.py [--seed N] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path

import fixture_extra
import numpy as np
from fixture_extra import TOKENS
from fixture_fonts import FONT_DIR, FONT_FILES
from fixture_images import (
    DPI,
    PHOTO_QUALITY,
    SCALE,
    SCAN_QUALITY,
    jpeg_bytes,
    photo_condition,
    png_bytes,
    render_pdf,
    scan_condition,
    transform_box,
)
from fixture_output import document, emit, gt_dict, json_bytes, rendered, write_pair
from fixture_pages import (
    PAGE_H,
    PAGE_W,
    Placed,
    blank_pdf,
    cell_boxes,
    draw_pdf,
    draw_visual_order_pdf,
)
from fixture_text import AR
from PIL import Image
from script_errors import describe_os_error

DEFAULT_SEED = 20261119
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tests" / "fixtures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_rng(seed: int, stem: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(stem.encode("utf-8"))])


def build_document_files(
    token: str, language: str, layout: str, seed: int, out_dir: Path
) -> tuple[Path, ...]:
    """Write the PDF, clean, scan, photo and EXIF-photo files of one document, with ground truth."""
    stem, out_dir = f"{token}-{layout}", Path(out_dir)
    doc, placed = document(token, language, layout, seed)
    meta, ident = (language, layout, doc.direction), np.eye(3)
    born_pdf, born_img, born_boxes = rendered(placed, "amiri")
    clean, boxes = born_img, born_boxes
    if layout == "letter":
        _, clean, boxes = rendered(placed, "naskh")
    rng = _image_rng(seed, stem)
    scan, h_scan = scan_condition(clean, rng)
    photo, h_photo = photo_condition(clean, rng)
    exif = Image.Exif()
    exif[0x0112] = 6
    sideways = jpeg_bytes(photo.transpose(Image.Transpose.ROTATE_90), PHOTO_QUALITY, exif)
    rows = (
        (f"{stem}.pdf", born_pdf, "born_digital", born_img.size, ident, born_boxes),
        (f"{stem}-clean.png", png_bytes(clean), "clean", clean.size, ident, boxes),
        (f"{stem}-scan.jpg", jpeg_bytes(scan, SCAN_QUALITY), "scan", scan.size, h_scan, boxes),
        (
            f"{stem}-photo.jpg",
            jpeg_bytes(photo, PHOTO_QUALITY),
            "photo",
            photo.size,
            h_photo,
            boxes,
        ),
        (f"{stem}-photo-exif.jpg", sideways, "photo_exif", photo.size, h_photo, boxes),
    )
    files: tuple[Path, ...] = ()
    for name, data, condition, size, homography, row_boxes in rows:
        files += emit(out_dir, name, data, condition, meta, size, homography, placed, row_boxes)
    return files


def _read_gt(out_dir: Path, name: str) -> dict:
    return json.loads((out_dir / name).read_text(encoding="utf-8"))


def _scanned_pdf(jpeg: bytes, size: tuple[int, int]) -> bytes:
    # 0.01 pt under the exact size keeps the renderer's round-up from adding a pixel
    width, height = size[0] * 72 / DPI - 0.01, size[1] * 72 / DPI - 0.01
    pdf = blank_pdf(width, height)
    pdf.add_page()
    pdf.image(io.BytesIO(jpeg), x=0, y=0, w=width, h=height)
    return bytes(pdf.output())


def _upside_down(out_dir: Path) -> tuple[Path, ...]:
    gt = _read_gt(out_dir, "en-invoice-clean.gt.json")
    w, h = gt["width"], gt["height"]
    flip = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1.0]])
    with Image.open(out_dir / "en-invoice-clean.png") as clean:
        flipped = clean.transpose(Image.Transpose.ROTATE_180)
    lines = [{**ln, "box": transform_box(flip, ln["box"], (w, h))} for ln in gt["lines"]]
    meta = (gt["language"], gt["layout"], gt["direction"])
    return write_pair(
        out_dir,
        "en-invoice-upside_down.png",
        png_bytes(flipped),
        gt_dict("upside_down", meta, (w, h), flip, lines),
    )


def _blank_scan(seed: int, out_dir: Path) -> tuple[Path, ...]:
    page = Image.new("L", (round(PAGE_W * SCALE), round(PAGE_H * SCALE)), 255)
    blank, homography = scan_condition(page, _image_rng(seed, "blank"))
    meta = ("en+ar", "blank", "ltr")
    return write_pair(
        out_dir,
        "blank-scan.jpg",
        jpeg_bytes(blank, SCAN_QUALITY),
        gt_dict("blank", meta, blank.size, homography, []),
    )


def _meeting_line(placed: list[Placed]) -> Placed:
    prefix = AR["meeting"].split("{}")[0]
    return next(p for p in placed if p.line.text.startswith(prefix))


def _broken_pdfs(seed: int, out_dir: Path) -> tuple[Path, ...]:
    """Write the one-line visual-order PDF (the meeting sentence) and the unmapped-text PDF."""
    doc, placed = document("ar", "ar", "letter", seed)
    meta, files = ("ar", "letter", doc.direction), ()
    one = [_meeting_line(placed)]
    for name, condition, data, cells in (
        (
            "ar-letter-broken_visual_order.pdf",
            "broken_visual_order",
            draw_visual_order_pdf(one[0]),
            one,
        ),
        ("ar-letter-broken_unmapped.pdf", "broken_unmapped", draw_pdf(placed, "naskh"), placed),
    ):
        img = render_pdf(data)
        boxes = cell_boxes(img, cells)
        files += emit(out_dir, name, data, condition, meta, img.size, np.eye(3), cells, boxes)
    return files


def _scanned_fixture(out_dir: Path) -> tuple[Path, ...]:
    gt = _read_gt(out_dir, "ar-invoice-scan.gt.json")
    pdf = _scanned_pdf((out_dir / "ar-invoice-scan.jpg").read_bytes(), (gt["width"], gt["height"]))
    kept = {k: v for k, v in gt.items() if k not in ("file", "dpi", "condition")}
    return write_pair(out_dir, "ar-invoice-scanned.pdf", pdf, {**kept, "condition": "scanned_pdf"})


def build_extras(seed: int, out_dir: Path) -> tuple[Path, ...]:
    """Write the upside-down, blank, broken-PDF, scanned-PDF and tagged fixtures (after the six documents)."""
    return (
        *_upside_down(out_dir),
        *_blank_scan(seed, out_dir),
        *_broken_pdfs(seed, out_dir),
        *_scanned_fixture(out_dir),
        *fixture_extra.build_special(seed, out_dir),
    )


def _write_manifest(seed: int, out_dir: Path, files: list[Path]) -> Path:
    entries = []
    for gt_path in (f for f in files if f.name.endswith(".gt.json")):
        gt = _read_gt(out_dir, gt_path.name)
        entries.append(
            {
                "file": gt["file"],
                "sha256": _sha256(out_dir / gt["file"]),
                "gt": gt_path.name,
                "gt_sha256": _sha256(gt_path),
                "condition": gt["condition"],
                "language": gt["language"],
                "layout": gt["layout"],
                "tags": sorted(fixture_extra.FIXTURE_TAGS.get(gt["file"], [])),
            }
        )
    fonts = [
        {"file": f"fonts/{n}", "sha256": _sha256(FONT_DIR / n)} for n in sorted(FONT_FILES.values())
    ]
    path = out_dir / "manifest.json"
    path.write_bytes(
        json_bytes(
            {"seed": seed, "fonts": fonts, "fixtures": sorted(entries, key=lambda e: e["file"])}
        )
    )
    return path


def build_all(seed: int, out_dir: Path) -> Path:
    """Write every fixture, its ground truth and manifest.json; return the manifest path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    files: tuple[Path, ...] = ()
    for token, language in TOKENS:
        for layout in ("invoice", "letter"):
            files += build_document_files(token, language, layout, seed, out_dir)
    files += build_extras(seed, out_dir)
    return _write_manifest(seed, out_dir, files)


def _parse_seed(text: str) -> int | None:
    """The seed as a non-negative integer, or None when `text` is not one."""
    if not (text.isascii() and text.isdigit()):
        return None
    try:
        return int(text)
    except ValueError:  # more digits than int() converts
        return None


def _fail(message: str) -> int:
    sys.stderr.write(f"error: {message}\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the OCR test fixtures.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    seed = _parse_seed(args.seed)
    if seed is None:
        return _fail("--seed must be a non-negative integer")
    try:
        build_all(seed, args.out)
    except OSError as exc:
        return _fail(describe_os_error(exc, args.out))
    except ValueError as exc:
        return _fail(f"cannot build fixtures: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
