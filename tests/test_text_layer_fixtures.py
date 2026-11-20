"""Text-layer reading and checks on the committed fixtures."""

import json
import math
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from folio_ocr.flags import (
    EMPTY_WITH_INK,
    INVISIBLE_TEXT,
    PRIVATE_USE,
    TEXT_LAYER_INVALID,
    VISUAL_ORDER,
    line_flags,
)
from folio_ocr.metrics import Ratio, char_errors, gt_line_boxes, line_recall, match_lines
from folio_ocr.normalize import norm_v2
from folio_ocr.text_layer import GAP_FACTOR, line_box, line_text, read_text_layer
from folio_ocr.text_layer_check import TextLayerCheck, check_text_layer

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
PDF_ENTRIES = [e for e in MANIFEST if e["file"].endswith(".pdf")]
ELEVEN_PT = {
    f"{lang}-{kind}.pdf" for lang in ("en", "ar", "mixed") for kind in ("invoice", "letter")
}
SECOND_SIZE = {e["file"] for e in PDF_ENTRIES if "second_size" in e["tags"]}
CER_LIMIT = 0.02


def load_gt(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class GtLine:
    text: str
    box: tuple
    segment: int | None  # index of the matched extracted line
    cer: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PageRead:
    segs: tuple
    report: TextLayerCheck
    gt_boxes: tuple
    gt_lines: tuple[GtLine, ...]


def read_page(entry: dict) -> PageRead:
    gt = load_gt(entry["gt"])
    doc = pdfium.PdfDocument(str(FIXTURES / entry["file"]))
    try:
        page = doc[0]
        segs = read_text_layer(page)
        has_ink = page.render(scale=300 / 72, grayscale=True).to_pil().getextrema()[0] < 128
    finally:
        doc.close()
    report = check_text_layer(segs, has_ink=has_ink)
    gt_boxes = tuple(tuple(line["box"]) for line in gt["lines"])
    matched = dict(match_lines(gt_boxes, [line_box(s) or (0, 0, 0, 0) for s in segs]))
    lines = []
    for i, line in enumerate(gt["lines"]):
        j = matched.get(i)
        text = line_text(segs[j]) if j is not None else ""
        rate = char_errors(norm_v2(line["text"]), norm_v2(text)).rate if j is not None else 1.0
        flags = ()
        if j is not None:
            flags = report.lines[j].flags + line_flags(text, gt["language"])
        lines.append(GtLine(line["text"], gt_boxes[i], j, rate, flags))
    return PageRead(segs, report, gt_boxes, tuple(lines))


@pytest.fixture(scope="module")
def pages() -> dict[str, PageRead]:
    return {e["file"]: read_page(e) for e in PDF_ENTRIES}


def recall(pages, names) -> Ratio:
    hits = total = 0
    for name in names:
        p = pages[name]
        r = line_recall(p.gt_boxes, [line_box(s) or (0, 0, 0, 0) for s in p.segs])
        hits, total = hits + r.numerator, total + r.denominator
    return Ratio(hits, total)


def test_segment_recall_on_eleven_point_documents(pages):
    assert recall(pages, ELEVEN_PT).rate >= 0.98


def test_segment_recall_on_second_size_documents(pages):
    assert len(SECOND_SIZE) == 3
    assert recall(pages, SECOND_SIZE).rate >= 0.98


def glyph_gaps(p: PageRead):
    """(within, between) gaps in glyph heights for the glyphs of one page, grouped by GT line."""
    glyphs = [c for s in p.segs for c in s if c.text.strip() and c.box]
    groups: list[list] = [[] for _ in p.gt_boxes]
    for c in glyphs:
        cx, cy = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
        for i, b in enumerate(p.gt_boxes):
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                groups[i].append(c)
                break
    heights = [statistics.median(c.box[3] - c.box[1] for c in g) if g else 0 for g in groups]
    within = []
    for g, h in zip(groups, heights, strict=True):
        ordered = sorted(g, key=lambda c: c.box[0])
        within += [(b.box[0] - a.box[2]) / h for a, b in pairwise(ordered)]
    between = []
    for i, a in enumerate(p.gt_boxes):
        for j in range(i + 1, len(p.gt_boxes)):
            b = p.gt_boxes[j]
            overlap = min(a[3], b[3]) - max(a[1], b[1])
            if not groups[i] or not groups[j] or overlap < 0.5 * min(a[3] - a[1], b[3] - b[1]):
                continue
            left, right = sorted((groups[i], groups[j]), key=lambda g: min(c.box[0] for c in g))
            gap = min(c.box[0] for c in right) - max(c.box[2] for c in left)
            between.append(gap / max(heights[i], heights[j]))
    return within, between


def test_gap_factor_is_the_calibrated_midpoint(pages):
    within, between = [], []
    for name in ELEVEN_PT | SECOND_SIZE:
        w, b = glyph_gaps(pages[name])
        within += w
        between += b
    assert within and between
    assert 2 * max(within) <= GAP_FACTOR <= min(between) / 1.5
    assert GAP_FACTOR == round((max(within) + min(between)) / 2, 1)


def kept_pages(pages):
    return [(n, p) for n, p in pages.items() if not p.report.to_ocr]


def test_no_visual_order_flag_on_lines_that_equal_ground_truth(pages):
    flagged = [
        (name, g.text)
        for name, p in kept_pages(pages)
        for g in p.gt_lines
        if g.segment is not None and g.cer == 0 and VISUAL_ORDER in g.flags
    ]
    assert flagged == []


def test_every_wrong_line_of_a_kept_page_is_flagged(pages):
    wrong = [(n, g) for n, p in kept_pages(pages) for g in p.gt_lines if g.cer > CER_LIMIT]
    unflagged = [(n, g) for n, g in wrong if not g.flags]
    listing = [(n, ascii(g.text), round(g.cer, 3)) for n, g in unflagged]
    assert len(wrong) >= 10
    assert unflagged == [], f"unflagged lines: {listing}"


def test_bulleted_pdfs_flag_only_the_bullet_lines(pages):
    names = [e["file"] for e in PDF_ENTRIES if "bulleted" in e["tags"]]
    assert names
    for name in names:
        p = pages[name]
        assert p.report.reasons == () and not p.report.to_ocr
        assert p.report.invalid_lines == 5
        for seg, check in zip(p.segs, p.report.lines, strict=True):
            if check.reasons:
                assert line_text(seg) == chr(0xF0B7)
                assert check.reasons == (PRIVATE_USE,)
                assert TEXT_LAYER_INVALID in check.flags
            else:
                assert not set(check.flags) & {TEXT_LAYER_INVALID, VISUAL_ORDER}
        assert sum(1 for c in p.report.lines if c.reasons) == 5


def test_invisible_layer_pdf_is_flagged_but_kept(pages):
    names = [e["file"] for e in PDF_ENTRIES if "planted_errors" in e["tags"]]
    assert names
    for name in names:
        p = pages[name]
        assert p.report.reasons == (INVISIBLE_TEXT,)
        assert not p.report.to_ocr
        assert all(TEXT_LAYER_INVALID in c.flags for c in p.report.lines)
        assert all(g.segment is not None for g in p.gt_lines)
        assert sum(1 for g in p.gt_lines if g.cer > CER_LIMIT) >= 4


def test_unmapped_glyph_pdf_goes_to_ocr(pages):
    p = pages["ar-letter-broken_unmapped.pdf"]
    assert TEXT_LAYER_INVALID in p.report.reasons
    assert p.report.to_ocr
    assert 2 * p.report.invalid_lines > len(p.report.lines)


def test_image_only_pdf_goes_to_ocr(pages):
    p = pages["ar-invoice-scanned.pdf"]
    assert p.report.reasons == (EMPTY_WITH_INK,)
    assert p.report.to_ocr


def test_visual_order_pdf_line_is_flagged(pages):
    p = pages["ar-letter-broken_visual_order.pdf"]
    assert len(p.report.lines) == 1
    assert VISUAL_ORDER in p.report.lines[0].flags


@pytest.mark.parametrize(
    ("name", "limit"),
    [("ar-invoice", 0.01), ("ar-letter", 0.01), ("en-invoice", 0.005), ("en-letter", 0.005)],
)
def test_born_digital_text_layer_is_valid_and_accurate(pages, name, limit):
    p = pages[f"{name}.pdf"]
    reference = " ".join(g.text for g in p.gt_lines)
    hypothesis = " ".join(line_text(s) for s in p.segs)
    assert p.report.reasons == ()
    assert all(c.flags == () for c in p.report.lines)
    assert char_errors(norm_v2(reference), norm_v2(hypothesis)).rate <= limit


# GT boxes of a scan map through Page.transform
SCANS = [
    e["file"] for e in MANIFEST if e["file"].endswith("-scan.jpg") and load_gt(e["gt"])["lines"]
]


def straightening(gt: dict):
    h = gt["homography"]
    angle = math.atan2(h[1][0], h[0][0])
    c, s = math.cos(-angle), math.sin(-angle)
    cx, cy = gt["width"] / 2, gt["height"] / 2
    return (
        (c, -s, cx - c * cx + s * cy),
        (s, c, cy - s * cx - c * cy),
        (0.0, 0.0, 1.0),
    )


def clean_predictions(name: str, gt: dict):
    clean = load_gt(name.replace("-scan.jpg", "-clean.gt.json"))
    dx = gt["width"] / 2 - clean["width"] / 2
    dy = gt["height"] / 2 - clean["height"] / 2
    return [
        (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        for b in (ln["clean_box"] for ln in gt["lines"])
    ]


@pytest.mark.parametrize("name", SCANS)
def test_clean_boxes_map_onto_the_straightened_scan(name):
    gt = load_gt(name.replace(".jpg", ".gt.json"))
    mapped = gt_line_boxes(
        [tuple(ln["clean_box"]) for ln in gt["lines"]],
        tuple(map(tuple, gt["homography"])),
        straightening(gt),
    )
    assert line_recall(mapped, clean_predictions(name, gt)).rate == 1.0


def test_stored_scan_boxes_do_not_match_the_straightened_frame():
    gt = load_gt("ar-invoice-scan.gt.json")
    stored = [tuple(ln["box"]) for ln in gt["lines"]]
    assert line_recall(stored, clean_predictions("ar-invoice-scan.jpg", gt)).rate < 1.0
