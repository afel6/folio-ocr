"""Shared steps for writing a fixture: ground truth lines, JSON files, rendering with text checks."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from fixture_fonts import check_coverage, check_shaping, check_single_glyph_clusters
from fixture_images import DPI, render_pdf, transform_box
from fixture_pages import Placed, cell_boxes, draw_pdf, layout_document
from fixture_text import make_document


def json_bytes(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def lines_json(placed: list[Placed], boxes, homography, size) -> list[dict]:
    """Ground truth lines; `clean_box` is the box in the clean render, `box` is after the homography."""
    return [
        {
            "text": p.line.text,
            "box": transform_box(homography, box, size),
            "clean_box": [int(v) for v in box],
            "direction": p.line.direction,
            "entities": [{"type": e.type, "start": e.start, "end": e.end} for e in p.line.entities],
        }
        for p, box in zip(placed, boxes, strict=True)
    ]


def write_pair(out_dir: Path, name: str, data: bytes, gt: dict) -> tuple[Path, Path]:
    """Write a fixture and its '<name without extension>.gt.json'."""
    path, gt_path = out_dir / name, out_dir / f"{Path(name).stem}.gt.json"
    path.write_bytes(data)
    gt_path.write_bytes(json_bytes({"file": name, **gt, "dpi": DPI}))
    return path, gt_path


def gt_dict(condition, meta, size, homography, lines) -> dict:
    language, layout, direction = meta
    return {
        "condition": condition,
        "language": language,
        "layout": layout,
        "direction": direction,
        "width": size[0],
        "height": size[1],
        "homography": np.asarray(homography).tolist(),
        "lines": lines,
    }


def emit(out_dir, name, data, condition, meta, image_size, homography, placed, boxes):
    """Write one fixture with ground truth whose boxes are carried through the homography."""
    lines = lines_json(placed, boxes, homography, image_size)
    return write_pair(out_dir, name, data, gt_dict(condition, meta, image_size, homography, lines))


def rendered(placed: list[Placed], arabic_key: str, scale: float = 1.0, clusters: bool = True):
    """Check the text, draw it, render it at 300 dpi; return the PDF, the image and the ink boxes.

    The single-glyph-cluster check is skipped for pages that hold harakat on purpose.
    """
    lines = [p.line for p in placed]
    check_coverage(lines, arabic_key)
    check_shaping(lines, arabic_key)
    if arabic_key == "amiri" and clusters:
        check_single_glyph_clusters(lines)
    pdf = draw_pdf(placed, arabic_key, scale)
    img = render_pdf(pdf)
    return pdf, img, cell_boxes(img, placed, scale)


def document(token: str, language: str, layout: str, seed: int, scale: float = 1.0):
    doc = make_document(language, layout, random.Random(f"{seed}-{token}-{layout}"))
    return doc, layout_document(doc, scale)
