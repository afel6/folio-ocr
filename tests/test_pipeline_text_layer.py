"""End-to-end reading of the fixture PDFs: every wrong text-layer line flagged,
pooled accuracy, unread regions and rotated or offset pages."""

import io
import json
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from pdf_variants import add_image, add_rect, open_fixture, variant_bytes
from PIL import Image

from folio_ocr.document import Document
from folio_ocr.flags import UNREAD_REGIONS, VISUAL_ORDER
from folio_ocr.metrics import cer, char_errors, match_lines
from folio_ocr.normalize import norm_v2
from folio_ocr.pipeline import read_bytes, read_path
from folio_ocr.text_layer import image_boxes

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
PDF_ENTRIES = [e for e in MANIFEST if e["file"].endswith(".pdf")]
BORN_DIGITAL = [
    f"{lang}-{kind}.pdf" for lang in ("en", "ar", "mixed") for kind in ("invoice", "letter")
]
CER_LIMIT = 0.02
S = 300 / 72


def load_gt(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reads(ocr_stub_module) -> dict[str, tuple[dict, Document]]:
    out = {}
    for e in PDF_ENTRIES:
        gt = load_gt(e["gt"])
        out[e["file"]] = (gt, read_path(FIXTURES / e["file"], languages=gt["language"]))
    return out


def covered(box, regions) -> bool:
    area = (box[2] - box[0]) * (box[3] - box[1])
    for r in regions:
        w = min(box[2], r[2]) - max(box[0], r[0])
        h = min(box[3], r[3]) - max(box[1], r[1])
        if w > 0 and h > 0 and w * h >= 0.5 * area:
            return True
    return False


def gate_rows(reads):
    """(file, gt line, matched Line or None, cer, covered) for every GT line of kept pages."""
    rows = []
    for name, (gt, doc) in reads.items():
        page = doc.pages[0]
        if page.origin != "text_layer":
            continue
        boxes = [tuple(ln["box"]) for ln in gt["lines"]]
        matched = dict(match_lines(boxes, [ln.box for ln in page.lines]))
        for i, g in enumerate(gt["lines"]):
            j = matched.get(i)
            if j is None:
                rows.append((name, g, None, 1.0, covered(boxes[i], page.unread_regions)))
                continue
            line = page.lines[j]
            rate = char_errors(norm_v2(g["text"]), norm_v2(line.final_text)).rate
            rows.append((name, g, line, rate, False))
    return rows


def test_every_wrong_line_is_flagged_needs_review(reads):
    wrong = [r for r in gate_rows(reads) if r[2] is not None and r[3] > CER_LIMIT]
    bad = [
        (r[0], ascii(r[1]["text"]), round(r[3], 3)) for r in wrong if r[2].status != "needs_review"
    ]
    assert len(wrong) >= 10
    assert bad == []


def test_no_uncovered_missed_line_on_kept_pages(reads):
    missed = [(r[0], ascii(r[1]["text"])) for r in gate_rows(reads) if r[2] is None and not r[4]]
    assert missed == []


def test_lines_equal_to_ground_truth_carry_no_visual_order(reads):
    equal = [r for r in gate_rows(reads) if r[2] is not None and r[3] == 0]
    assert equal
    assert [(r[0], ascii(r[1]["text"])) for r in equal if VISUAL_ORDER in r[2].flags] == []


def test_born_digital_fixtures_keep_their_text_layer(reads):
    assert [n for n in BORN_DIGITAL if reads[n][1].pages[0].origin != "text_layer"] == []


def pooled(reads, names):
    pairs = []
    for n in names:
        gt, doc = reads[n]
        ref = " ".join(ln["text"] for ln in gt["lines"])
        hyp = " ".join(ln.final_text for ln in doc.pages[0].lines)
        pairs.append((ref, hyp))
    return cer(pairs)


@pytest.mark.parametrize(
    ("names", "limit"),
    [(("en-invoice.pdf", "en-letter.pdf"), 0.005), (("ar-invoice.pdf", "ar-letter.pdf"), 0.01)],
)
def test_pooled_born_digital_cer(reads, names, limit):
    assert pooled(reads, names).rate <= limit


# --- unread regions -----------------------------------------------------------------------


def line_view(page) -> list[tuple]:
    return [(ln.text, ln.box, ln.flags, ln.status) for ln in page.lines]


def read_variant_doc(doc: pdfium.PdfDocument) -> Document:
    buf = io.BytesIO()
    doc.save(buf)
    return read_bytes(buf.getvalue(), "en-invoice.pdf", languages="en")


def test_black_rectangle_is_one_unread_region_and_lines_stay(reads):
    plain = reads["en-invoice.pdf"][1].pages[0]
    doc = open_fixture("en-invoice.pdf")
    try:
        add_rect(doc[0], 100, 100, 100, 60)  # empty area below the text
        page = read_variant_doc(doc).pages[0]
    finally:
        doc.close()
    assert page.flags[-1] == UNREAD_REGIONS
    assert len(page.unread_regions) == 1
    x0, y0, x1, y1 = page.unread_regions[0]
    want = (100 * S, (595 - 160) * S, 200 * S, (595 - 100) * S)
    assert x0 <= want[0] + 2 and y0 <= want[1] + 2 and x1 >= want[2] - 2 and y1 >= want[3] - 2
    assert line_view(page) == line_view(plain)


def test_image_object_is_an_unread_region():
    doc = open_fixture("en-invoice.pdf")
    try:
        add_image(doc, doc[0], Image.new("RGB", (50, 50), (20, 20, 20)), 100, 100, 100, 60)
        boxes = image_boxes(doc[0])
        page = read_variant_doc(doc).pages[0]
    finally:
        doc.close()
    assert len(boxes) == 1
    assert boxes[0] in page.unread_regions
    assert UNREAD_REGIONS in page.flags


# --- rotated and offset pages ---------------------------------------------------------------

NAMES = ("en-invoice.pdf", "ar-invoice.pdf", "mixed-letter.pdf")
VARIANTS = [(90, 0, 0), (180, 0, 0), (270, 0, 0), (0, 150, 250)]
RESORT = ("ar-invoice.pdf", 270, 0, 0)


def has_ink(data: bytes, box) -> bool:
    doc = pdfium.PdfDocument(data)
    try:
        img = doc[0].render(scale=S, grayscale=True).to_pil()
    finally:
        doc.close()
    x0, y0, x1, y1 = (int(v) for v in (box[0], box[1], box[2] + 1, box[3] + 1))
    return img.crop((x0, y0, x1, y1)).getextrema()[0] < 128


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(("rotation", "dx", "dy"), VARIANTS)
def test_rotated_and_offset_pdfs_read_like_the_upright_one(reads, name, rotation, dx, dy):
    gt, upright = reads[name]
    data = variant_bytes(name, rotation, dx, dy)
    page = read_bytes(data, name, languages=gt["language"]).pages[0]
    want = upright.pages[0]
    assert (page.width, page.height) == (want.width, want.height)
    assert page.unread_regions == ()
    assert all(has_ink(data, ln.box) for ln in page.lines)
    differing = [(a, b) for a, b in zip(page.lines, want.lines, strict=True) if a.text != b.text]
    assert len(page.lines) == len(want.lines)
    if (name, rotation, dx, dy) == RESORT:
        assert all(a.status == "needs_review" for a, _ in differing)
    else:
        assert differing == []
