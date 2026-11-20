"""Writes seeded development-variant documents, their ground truth and a manifest.

The documents differ from the fixtures in fonts, vocabulary, templates and degradations; the
ground truth has the same schema, so one evaluator reads both. Seeds 0..99999 only.
Usage: python scripts/make_dev_variants.py --first N --count K --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from pathlib import Path

import dev_degrade
import dev_fonts
import dev_pages
import dev_text
import numpy as np
from fixture_images import DPI, transform_box
from fixture_output import json_bytes
from fixture_text import DocText
from script_errors import describe_os_error

DEV_SEED_RANGE = range(100000)
SCHEMA = "folio-dev/1"
TEMPLATE_WEIGHTS = (0.22, 0.22, 0.22, 0.22, 0.12)
TIGHT_TABLES = 0.7
TIGHT_GAP = (1.2, 2.5)  # box heights between table columns
LANGUAGE_TOKENS = {"en": "en", "ar": "ar", "en+ar": "mixed"}
GENERATOR_FILES = (
    "dev_degrade.py",
    "dev_fonts.py",
    "dev_pages.py",
    "dev_pools.py",
    "dev_text.py",
    "make_dev_variants.py",
)
ATTEMPTS = 8
SCRIPTS = Path(__file__).resolve().parent


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _draw_document(rng: random.Random, language: str, template: str):
    """Pick fonts and a size, then build and render a document; redraw it when it cannot fit."""
    latin, arabic = rng.choice(dev_fonts.LATIN_KEYS), rng.choice(dev_fonts.ARABIC_KEYS)
    # Most tables are tight: columns fitted to their text, 1.2 to 2.5 box heights apart.
    tight = template == "table" and rng.random() < TIGHT_TABLES
    table_gap = rng.uniform(*TIGHT_GAP) if tight else None
    for _ in range(ATTEMPTS):
        size = rng.randint(12, 20) if template == "grid" else rng.randint(8, 16)
        doc = dev_text.make_dev_document(language, template, rng)
        try:
            return (doc, *dev_pages.render_page(doc, latin, arabic, size, table_gap))
        except dev_pages.DoesNotFit:
            continue
    raise ValueError(f"no {template} document fit the page after {ATTEMPTS} tries")


def _lines_json(doc: DocText, boxes, homography, size) -> list[dict]:
    lines = [c for row in doc.rows for c in row.cells]
    return [
        {
            "text": line.text,
            "box": transform_box(homography, box, size),
            "clean_box": [int(v) for v in box],
            "direction": line.direction,
            "entities": [{"type": e.type, "start": e.start, "end": e.end} for e in line.entities],
        }
        for line, box in zip(lines, boxes, strict=True)
    ]


def build_document(seed: int, language: str, index: int, out_dir: Path) -> dict:
    """Write one document and its ground truth; return its manifest entry."""
    rng = random.Random(f"folio-dev/{seed}/{language}")
    np_rng = np.random.default_rng([0x0DE7, seed, index])
    template = rng.choices(dev_text.TEMPLATES, weights=TEMPLATE_WEIGHTS)[0]
    mix = dev_degrade.condition_mix(template)
    condition = rng.choices(list(mix), weights=list(mix.values()))[0]
    doc, _, clean, boxes, _ = _draw_document(rng, language, template)
    image, homography, params, ext = dev_degrade.degrade(clean, condition, np_rng)
    stem = f"dev-{seed:05d}-{LANGUAGE_TOKENS[language]}-{template}-{condition}"
    data = dev_degrade.encode_image(image, ext, params)
    scale = params.get("scale")
    gt = {
        "file": f"{stem}.{ext}",
        "condition": condition,
        "language": language,
        "layout": template,
        "direction": doc.direction,
        "width": image.size[0],
        "height": image.size[1],
        "dpi": DPI if scale is None else round(DPI * scale, 2),
        "homography": np.asarray(homography).tolist(),
        "lines": _lines_json(doc, boxes, homography, image.size),
        "params": params,
        "unit": stem,
    }
    gt_bytes = json_bytes(gt)
    (out_dir / gt["file"]).write_bytes(data)
    (out_dir / f"{stem}.gt.json").write_bytes(gt_bytes)
    return {
        "file": gt["file"],
        "sha256": _sha256(data),
        "gt": f"{stem}.gt.json",
        "gt_sha256": _sha256(gt_bytes),
        "condition": condition,
        "language": language,
        "layout": template,
        "unit": stem,
    }


def _generator_hashes() -> dict[str, str]:
    """SHA-256 of each generator file, line endings read as LF."""
    return {
        f"scripts/{name}": _sha256((SCRIPTS / name).read_bytes().replace(b"\r\n", b"\n"))
        for name in GENERATOR_FILES
    }


def build_all(first: int, count: int, out_dir: Path) -> Path:
    """Write every document of seeds first..first+count-1 and manifest.json; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    documents = [
        build_document(seed, language, index, out_dir)
        for seed in range(first, first + count)
        for index, language in enumerate(dev_text.LANGUAGES)
    ]
    manifest = {
        "schema": SCHEMA,
        "seeds": {"first": first, "last": first + count - 1},
        "generator_files": _generator_hashes(),
        "fonts": dev_fonts.font_sha256s(),
        "documents": sorted(documents, key=lambda d: d["file"]),
    }
    path = out_dir / "manifest.json"
    path.write_bytes(json_bytes(manifest))
    return path


def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    sign_free = text[1:] if text[:1] == "-" else text
    return int(text) if sign_free.isascii() and sign_free.isdigit() and len(text) < 12 else None


def _fail(message: str) -> int:
    sys.stderr.write(f"error: {message}\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the development-variant documents.")
    parser.add_argument("--first")
    parser.add_argument("--count")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    # fpdf2's font subsetter logs every font table it cannot subset (Graphite tables); not errors.
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    first, count = _parse_int(args.first), _parse_int(args.count)
    if first is None or count is None or count < 1 or first not in DEV_SEED_RANGE:
        return _fail("--first/--count must stay within 0..99999")
    if first + count - 1 not in DEV_SEED_RANGE:
        return _fail("--first/--count must stay within 0..99999")
    if args.out is None:
        return _fail("--out is required")
    try:
        build_all(first, count, args.out)
    except OSError as exc:
        return _fail(describe_os_error(exc, args.out))
    except ValueError as exc:
        return _fail(f"cannot build documents: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
