"""Checks for the committed OCR fixtures and for the generator that writes them."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from fixture_checks import assert_same_ground_truth, assert_same_png
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixture_extra as fe
import fixture_fonts as ff
import fixture_images as fi
import fixture_output as fo
import fixture_pages as fp
import fixture_text as ft
import make_fixtures as mf
import must_not_flag as mnf

FIX = HERE / "fixtures"
AMIRI = FIX / "fonts" / "Amiri-Regular.ttf"
NASKH = FIX / "fonts" / "NotoNaskhArabic-Regular.ttf"
DOCS = [
    (token, language, layout)
    for token, language in (("en", "en"), ("ar", "ar"), ("mixed", "en+ar"))
    for layout in ("invoice", "letter")
]
ENTITY_TYPES = {"name", "date", "amount", "id", "phone"}
WESTERN = set("0123456789")
INDIC = set("٠١٢٣٤٥٦٧٨٩")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))


def _gt(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _all_gts() -> list[dict]:
    return [_gt(e["gt"]) for e in _manifest()["fixtures"]]


def _lines(doc: ft.DocText) -> list[ft.TextLine]:
    return [cell for row in doc.rows for cell in row.cells]


def _doc(language: str, layout: str, seed: int = 1) -> ft.DocText:
    return ft.make_document(language, layout, random.Random(seed))


# --- committed fixtures -----------------------------------------------------


def test_manifest_hashes_match_disk():
    manifest = _manifest()
    assert manifest["seed"] == mf.DEFAULT_SEED
    for font in manifest["fonts"]:
        assert _sha(FIX / font["file"]) == font["sha256"]
    for entry in manifest["fixtures"]:
        assert _sha(FIX / entry["file"]) == entry["sha256"]
        assert _sha(FIX / entry["gt"]) == entry["gt_sha256"]


def test_every_fixture_file_is_listed():
    listed = {"manifest.json"}
    for entry in _manifest()["fixtures"]:
        listed |= {entry["file"], entry["gt"]}
    actual = {
        p.relative_to(FIX).as_posix()
        for p in FIX.rglob("*")
        if p.is_file() and p.relative_to(FIX).parts[0] != "fonts"
    }
    assert actual == listed


def test_fixture_folder_is_small():
    total = sum(p.stat().st_size for p in FIX.rglob("*") if p.is_file())
    assert total < 10_000_000


def _frame_size(file: str) -> tuple[int, int]:
    path = FIX / file
    if path.suffix == ".pdf":
        page = pdfium.PdfDocument(str(path))[0]
        return page.render(scale=300 / 72, grayscale=True).to_pil().size
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).size


def test_ground_truth_is_consistent():
    for gt in _all_gts():
        assert _frame_size(gt["file"]) == (gt["width"], gt["height"]), gt["file"]
        assert gt["dpi"] == 300
        for line in gt["lines"]:
            assert unicodedata.is_normalized("NFC", line["text"])
            x0, y0, x1, y1 = line["box"]
            assert 0 <= x0 < x1 <= gt["width"] and 0 <= y0 < y1 <= gt["height"], gt["file"]
            for ent in line["entities"]:
                assert ent["type"] in ENTITY_TYPES
                assert 0 <= ent["start"] < ent["end"] <= len(line["text"])


def test_documents_have_entities_and_both_digit_systems():
    manifest = _manifest()
    seen = set()
    born = [e for e in manifest["fixtures"] if e["condition"] == "born_digital" and not e["tags"]]
    assert len(born) == 6
    for entry in manifest["fixtures"]:
        gt = _gt(entry["gt"])
        spans = [
            line["text"][e["start"] : e["end"]] for line in gt["lines"] for e in line["entities"]
        ]
        if entry["condition"] == "born_digital" and not entry["tags"]:
            assert len(spans) >= 8
        for span in spans:
            seen |= set(span)
    assert seen & WESTERN and seen & INDIC


# --- determinism, text content ----------------------------------------------


def test_build_is_byte_identical(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    out1.mkdir()
    out2.mkdir()
    files1 = mf.build_document_files("ar", "ar", "invoice", mf.DEFAULT_SEED, out1)
    files2 = mf.build_document_files("ar", "ar", "invoice", mf.DEFAULT_SEED, out2)
    assert len(files1) == 10
    for a, b in zip(files1, files2, strict=True):
        assert a.name == b.name and a.read_bytes() == b.read_bytes()


def test_build_all_reproduces_committed_manifest(tmp_path):
    out = tmp_path / "fx"
    manifest = mf.build_all(mf.DEFAULT_SEED, out)
    regenerated = json.loads(manifest.read_bytes())
    for entry in regenerated["fixtures"]:
        name = entry["file"]
        if name.endswith(".png"):
            assert entry["sha256"] == _sha(out / name)
            assert_same_png((out / name).read_bytes(), (FIX / name).read_bytes())
            # Compare the manifest using the committed digest once the PNG content matches.
            entry["sha256"] = _sha(FIX / name)
        gt_name = entry["gt"]
        current_gt_sha = _sha(out / gt_name)
        assert entry["gt_sha256"] == current_gt_sha
        committed_gt_sha = _sha(FIX / gt_name)
        if current_gt_sha != committed_gt_sha:
            assert_same_ground_truth(json.loads((out / gt_name).read_bytes()), _gt(gt_name))
            entry["gt_sha256"] = committed_gt_sha
    assert regenerated == _manifest()


@pytest.mark.parametrize("token,language,layout", DOCS)
def test_document_text_shape(token, language, layout):
    doc = _doc(language, layout)
    assert doc == _doc(language, layout)
    assert doc.direction == ("ltr" if language == "en" else "rtl")
    types = [e.type for line in _lines(doc) for e in line.entities]
    assert len(types) >= 8
    assert types.count("name") >= 2 and types.count("date") >= 2
    assert types.count("amount") >= 3 and "id" in types and "phone" in types
    for line in _lines(doc):
        assert unicodedata.is_normalized("NFC", line.text)
        for e in line.entities:
            assert line.text[e.start : e.end].strip() == line.text[e.start : e.end]
    if language == "ar":
        assert not any("a" <= c.lower() <= "z" for line in _lines(doc) for c in line.text)
    if layout == "invoice":
        assert [len(r.cells) for r in doc.rows].count(4) == 4


def test_build_line_and_direction():
    line = ft.build_line([("Name: ", None), ("Bob", "name")])
    assert line.text == "Name: Bob" and line.direction == "ltr"
    assert line.entities == (ft.Entity("name", 6, 9),)
    assert ft.line_direction("123") == "ltr"
    assert ft.line_direction("١٢ سلم") == "rtl"
    with pytest.raises(ValueError):
        ft.build_line([("é", None)])


def test_make_document_rejects_unknown_values():
    with pytest.raises(ValueError):
        ft.make_document("fr", "invoice", random.Random(1))
    with pytest.raises(ValueError):
        ft.make_document("en", "poster", random.Random(1))


# --- shaping and rendering --------------------------------------------------


def _components(img: Image.Image) -> int:
    ink = np.asarray(img) < 160
    seen = np.zeros_like(ink)
    count = 0
    for y, x in zip(*np.nonzero(ink), strict=True):
        if seen[y, x]:
            continue
        count += _flood(ink, seen, y, x) >= 4
    return count


def _flood(ink, seen, y: int, x: int) -> int:
    """Mark the 8-connected ink region holding (y, x) as seen and return its size."""
    stack, size = [(y, x)], 0
    seen[y, x] = True
    while stack:
        cy, cx = stack.pop()
        size += 1
        for ny, nx in _ink_neighbours(ink, seen, cy, cx):
            seen[ny, nx] = True
            stack.append((ny, nx))
    return size


def _ink_neighbours(ink, seen, cy: int, cx: int):
    height, width = ink.shape
    cells = ((cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return [
        (ny, nx)
        for ny, nx in cells
        if 0 <= ny < height and 0 <= nx < width and ink[ny, nx] and not seen[ny, nx]
    ]


@pytest.mark.parametrize("font", [AMIRI, NASKH], ids=["amiri", "naskh"])
def test_shaping_joins_letters(font):
    assert _components(fp.render_line_strip("سلم", font)[1]) == 1
    assert _components(fp.render_line_strip("س ل م", font)[1]) == 3


def test_rtl_placement_in_text_layer():
    pdf_bytes, _ = fp.render_line_strip("سلم", AMIRI)
    textpage = pdfium.PdfDocument(pdf_bytes)[0].get_textpage()
    text = textpage.get_text_range()
    sin = textpage.get_charbox(text.index("س"))
    meem = textpage.get_charbox(text.index("م"))
    assert sin[0] > meem[0]


def test_strip_with_fallback_font():
    _, img = fp.render_line_strip("سلم ab", NASKH, FIX / "fonts" / "NotoSans-Regular.ttf")
    assert _components(img) >= 3


@pytest.mark.parametrize("key", ["amiri", "naskh"])
@pytest.mark.parametrize("token,language,layout", DOCS)
def test_no_tofu_and_joined_forms(token, language, layout, key):
    lines = _lines(_doc(language, layout))
    ff.check_coverage(lines, key)
    ff.check_shaping(lines, key)


def test_check_coverage_raises_for_missing_character():
    line = ft.build_line([("漢", None)])
    with pytest.raises(ValueError):
        ff.check_coverage([line], "amiri")


def test_check_shaping_raises_when_letters_are_not_joined():
    with pytest.raises(ValueError):
        ff.check_shaping([ft.build_line([("ا", None)])], "amiri")


def test_single_glyph_clusters_raises_for_split_cluster_word():
    with pytest.raises(ValueError, match="والحبر"):
        ff.check_single_glyph_clusters([ft.build_line([("العقد والحبر", None)])])


@pytest.mark.parametrize("token,language,layout", DOCS)
def test_document_lines_have_one_glyph_per_cluster(token, language, layout):
    ff.check_single_glyph_clusters(_lines(_doc(language, layout)))


def test_visual_order_fixture_is_a_single_line_page():
    gt = _gt("ar-letter-broken_visual_order.gt.json")
    assert len(gt["lines"]) == 1
    assert gt["lines"][0]["text"].startswith(ft.AR["meeting"].split("{}")[0])
    page = pdfium.PdfDocument(str(FIX / gt["file"]))[0]
    assert len(page.get_textpage().get_text_range().split()) >= 4


def test_draw_pdf_rejects_line_wider_than_cell():
    placed = fp.layout_document(
        ft.DocText("en", "letter", "ltr", (ft.Row((ft.build_line([("x" * 120, None)]),)),))
    )
    with pytest.raises(ValueError):
        fp.draw_pdf(placed, "amiri")


def test_ink_box_bounds_and_errors():
    arr = np.full((40, 40), 255, np.uint8)
    with pytest.raises(ValueError):
        fi.ink_box(arr, (0, 0, 40, 40))
    arr[10:20, 12:30] = 0
    assert fi.ink_box(arr, (0, 0, 40, 40)) == [12, 10, 30, 20]
    arr[0, 5] = 0
    with pytest.raises(ValueError):
        fi.ink_box(arr, (0, 0, 40, 40))


# --- conditions -------------------------------------------------------------


def test_exif_copy_matches_photo():
    for token, _, layout in DOCS:
        photo = Image.open(FIX / f"{token}-{layout}-photo.jpg")
        copy = Image.open(FIX / f"{token}-{layout}-photo-exif.jpg")
        assert copy.getexif()[0x0112] == 6
        back = ImageOps.exif_transpose(copy)
        assert back.size == photo.size
        diff = np.abs(np.asarray(back, np.int16) - np.asarray(photo.convert(back.mode), np.int16))
        assert diff.mean() < 8
        assert (
            _gt(f"{token}-{layout}-photo-exif.gt.json")["lines"]
            == (_gt(f"{token}-{layout}-photo.gt.json")["lines"])
        )


def test_upside_down_boxes_are_rotated_clean_boxes():
    clean = _gt("en-invoice-clean.gt.json")
    flipped = _gt("en-invoice-upside_down.gt.json")
    w, h = clean["width"], clean["height"]
    assert (flipped["width"], flipped["height"]) == (w, h)
    for a, b in zip(clean["lines"], flipped["lines"], strict=True):
        x0, y0, x1, y1 = a["box"]
        assert b["box"] == [w - x1, h - y1, w - x0, h - y0]


def test_blank_and_scanned_pdf_ground_truth():
    assert _gt("blank-scan.gt.json")["lines"] == []
    scan, pdf = _gt("ar-invoice-scan.gt.json"), _gt("ar-invoice-scanned.gt.json")
    assert scan["lines"] == pdf["lines"] and scan["homography"] == pdf["homography"]


def test_warp_and_box_transform():
    img = Image.new("L", (20, 10), 255)
    ident = np.eye(3)
    assert list(np.asarray(fi.warp(img, ident, (20, 10), 255))[0]) == [255] * 20
    shift = np.array([[1, 0, 5], [0, 1, 2], [0, 0, 1.0]])
    assert fi.transform_box(shift, [1, 1, 4, 3], (20, 10)) == [6, 3, 9, 5]
    assert fi.transform_box(shift, [14, 6, 19, 9], (20, 10)) == [19, 8, 20, 10]


# --- command line -----------------------------------------------------------


def _one_line_error(capsys) -> str:
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err
    return err


def test_cli_missing_parent_folder(tmp_path, capsys):
    target = tmp_path / "no" / "such"
    assert mf.main(["--out", str(target)]) == 1
    assert _one_line_error(capsys) == f"error: not found: {target}\n"


def test_cli_permission_error(tmp_path, capsys, monkeypatch):
    def deny(seed, out):
        raise PermissionError(13, "Permission denied", str(out / "x.png"))

    monkeypatch.setattr(mf, "build_all", deny)
    assert mf.main(["--out", str(tmp_path)]) == 1
    assert _one_line_error(capsys) == f"error: permission denied: {tmp_path / 'x.png'}\n"


def test_cli_output_is_a_file(tmp_path, capsys):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    assert mf.main(["--out", str(target)]) == 1
    assert _one_line_error(capsys) == f"error: not a folder: {target}\n"


def test_cli_missing_font_file(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "no-fonts"
    monkeypatch.setattr(ff, "FONT_DIR", missing)
    ff._cmap.cache_clear()
    ff._font_bytes.cache_clear()
    try:
        assert mf.main(["--out", str(tmp_path / "out")]) == 1
    finally:
        ff._cmap.cache_clear()
        ff._font_bytes.cache_clear()
    assert _one_line_error(capsys) == f"error: not found: {missing / 'NotoSans-Regular.ttf'}\n"


@pytest.mark.parametrize("seed", ["-1", "abc", "1.5", "", "1" * 4301])
def test_cli_rejects_bad_seed_before_writing(tmp_path, capsys, seed):
    out = tmp_path / "out"
    assert mf.main(["--seed", seed, "--out", str(out)]) == 1
    assert _one_line_error(capsys) == "error: --seed must be a non-negative integer\n"
    assert not out.exists()


def test_cli_reports_build_value_error(tmp_path, capsys, monkeypatch):
    def refuse(seed, out):
        raise ValueError("line is wider than its cell")

    monkeypatch.setattr(mf, "build_all", refuse)
    assert mf.main(["--out", str(tmp_path)]) == 1
    assert _one_line_error(capsys) == "error: cannot build fixtures: line is wider than its cell\n"


def test_cli_passes_seed_and_out_to_build(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mf, "build_all", lambda seed, out: calls.append((seed, Path(out))))
    assert mf.main(["--seed", "7", "--out", str(tmp_path)]) == 0
    assert calls == [(7, tmp_path)]


# --- visual order of the per-character page ---------------------------------


def test_visual_order_keeps_digit_runs_in_reading_order():
    text = "ب ١٢/٠٥ ج"
    visual = "".join(text[i] for i in fe.visual_order(text))
    assert visual == "ج ١٢/٠٥ ب"
    assert fe.visual_order("") == []
    # a separator not between two digits is reversed like a letter
    assert "".join("1/ا"[i] for i in fe.visual_order("1/ا")) == "ا/1"


# --- tagged fixtures ---------------------------------------------------------

TAGGED = {
    "en-invoice-16pt.pdf": ["second_size"],
    "ar-invoice-16pt.pdf": ["second_size"],
    "mixed-invoice-16pt.pdf": ["second_size"],
    "ar-adversarial-1.pdf": ["adversarial"],
    "ar-adversarial-2.pdf": ["adversarial"],
    "en-adversarial.pdf": ["adversarial"],
    "ar-adversarial-3.pdf": ["adversarial"],
    "en-adversarial-2.pdf": ["adversarial"],
    "ar-letter-per_char.pdf": ["adversarial"],
    "en-bulleted.pdf": ["bulleted"],
    "ar-bulleted.pdf": ["bulleted"],
    "ar-letter-invisible_ocr.pdf": ["planted_errors"],
}


def _page_text(name: str) -> str:
    doc = pdfium.PdfDocument(str(FIX / name))
    return doc[0].get_textpage().get_text_range()


def _text_modes(name: str) -> list[int]:
    from pypdfium2 import raw

    page = pdfium.PdfDocument(str(FIX / name))[0]
    objs = (raw.FPDFPage_GetObject(page.raw, i) for i in range(raw.FPDFPage_CountObjects(page.raw)))
    return [
        raw.FPDFTextObj_GetTextRenderMode(o)
        for o in objs
        if raw.FPDFPageObj_GetType(o) == raw.FPDF_PAGEOBJ_TEXT
    ]


def test_new_fixtures_are_listed_with_tags():
    tags = {e["file"]: e["tags"] for e in _manifest()["fixtures"]}
    for name, expected in TAGGED.items():
        assert tags[name] == expected
    assert all(t == [] for n, t in tags.items() if n not in TAGGED)
    assert all(t == sorted(t) for t in tags.values())


def test_every_line_has_clean_box_in_key_order():
    for gt in _all_gts():
        for line in gt["lines"]:
            assert list(line) == ["text", "box", "clean_box", "direction", "entities"]
            assert all(isinstance(v, int) for v in line["clean_box"])
            if gt["condition"] in ("scan", "photo", "photo_exif"):
                size = (gt["width"], gt["height"])
                h = np.array(gt["homography"])
                assert fi.transform_box(h, line["clean_box"], size) == line["box"], gt["file"]
            if gt["condition"] not in ("scan", "photo", "photo_exif", "upside_down", "scanned_pdf"):
                assert line["clean_box"] == line["box"], gt["file"]


def test_upside_down_and_scanned_pdf_keep_clean_frame_boxes():
    clean = _gt("en-invoice-clean.gt.json")["lines"]
    down = _gt("en-invoice-upside_down.gt.json")["lines"]
    assert [ln["clean_box"] for ln in down] == [ln["box"] for ln in clean]
    scan = _gt("ar-invoice-scan.gt.json")["lines"]
    scanned = _gt("ar-invoice-scanned.gt.json")["lines"]
    assert [ln["clean_box"] for ln in scanned] == [ln["clean_box"] for ln in scan]


@pytest.mark.parametrize("token", ["en", "ar", "mixed"])
def test_16pt_invoice_holds_the_same_text(token):
    small, big = _gt(f"{token}-invoice.gt.json"), _gt(f"{token}-invoice-16pt.gt.json")
    assert [ln["text"] for ln in big["lines"]] == [ln["text"] for ln in small["lines"]]
    assert big["condition"] == "born_digital" and big["layout"] == "invoice"
    assert (big["language"], big["direction"]) == (small["language"], small["direction"])
    assert big["width"] > small["width"]
    assert _page_text(f"{token}-invoice-16pt.pdf") == _page_text(f"{token}-invoice.pdf")


def test_adversarial_pages():
    ar = _gt("ar-adversarial-1.gt.json")["lines"] + _gt("ar-adversarial-2.gt.json")["lines"]
    assert [ln["text"] for ln in ar] == [unicodedata.normalize("NFC", t) for t in ft.AR_ADVERSARIAL]
    en = _gt("en-adversarial.gt.json")
    assert (en["language"], en["direction"], en["layout"]) == ("en+ar", "ltr", "adversarial")
    assert [ln["direction"] for ln in en["lines"]].count("rtl") == 1
    assert all(ln["entities"] == [] for ln in ar + en["lines"])
    assert all(ln["clean_box"] == ln["box"] for ln in ar)


def test_must_not_flag_pages():
    for name, direction, lines in (
        ("ar-adversarial-3", "rtl", (*mnf.RTL[24:], mnf.RTL[9])),
        ("en-adversarial-2", "ltr", mnf.LTR),
    ):
        gt = _gt(f"{name}.gt.json")
        assert (gt["language"], gt["direction"], gt["layout"]) == ("en+ar", direction, "adversarial")
        assert [ln["text"] for ln in gt["lines"]] == list(lines)


def test_per_char_page_is_emitted_in_logical_order():
    gt = _gt("ar-letter-per_char.gt.json")
    assert gt["condition"] == "per_character" and len(gt["lines"]) > 10
    doc = pdfium.PdfDocument(str(FIX / gt["file"]))
    tp = doc[0].get_textpage()
    chars = [tp.get_text_range(i, 1) for i in range(tp.count_chars())]
    letters = "".join(c for c in chars if unicodedata.category(c) == "Lo")
    expected = "".join(
        c for ln in gt["lines"] for c in ln["text"] if unicodedata.category(c) == "Lo"
    )
    assert letters == expected


def test_visual_order_of_a_dated_line():
    text = "بتاريخ ١٢/٠٥/٢٠٢٦"
    visual = "".join(text[i] for i in fe.visual_order(text))
    assert visual.startswith("١٢/٠٥/٢٠٢٦ ")
    assert visual.endswith("خيرات" + chr(0x0628))
    assert sorted(fe.visual_order(text)) == list(range(len(text)))


@pytest.mark.parametrize("name", ["en-bulleted", "ar-bulleted"])
def test_bulleted_pdf(name):
    gt = _gt(f"{name}.gt.json")
    assert _page_text(f"{name}.pdf").count(chr(0xF0B7)) == 5
    texts = [ln["text"] for ln in gt["lines"]]
    assert texts.count(chr(0x2022)) == 5 and len(texts) == 14
    for i, text in enumerate(texts):
        if text == chr(0x2022):
            assert texts[i + 1] != chr(0x2022)
            assert gt["lines"][i]["direction"] == gt["direction"]
    assert gt["layout"] == "bulleted" and gt["condition"] == "born_digital"


def test_bullet_font_adds_private_use_code_point():
    import io

    from fontTools.ttLib import TTFont

    font = TTFont(io.BytesIO(fe.bullet_font_bytes()))
    cmap = font.getBestCmap()
    assert cmap[0xF0B7] == cmap[0x2022]
    assert 0xF0B7 not in TTFont(FIX / "fonts" / "NotoSans-Regular.ttf").getBestCmap()


def test_invisible_layer_has_render_mode_3_and_planted_errors():
    name = "ar-letter-invisible_ocr.pdf"
    modes = _text_modes(name)
    assert modes and set(modes) == {3}
    gt = _gt("ar-letter-invisible_ocr.gt.json")
    assert gt["condition"] == "invisible_text"
    extracted = "".join(_page_text(name).split())
    wrong = [ln for ln in gt["lines"] if "".join(ln["text"].split()) not in extracted]
    assert len(wrong) >= 4


def test_plant_errors_changes_four_lines_and_keeps_the_rest():
    _, placed = fo.document("ar", "ar", "letter", mf.DEFAULT_SEED)
    lines = [p.line for p in placed]
    planted = fe.plant_errors(lines)
    changed = [i for i, (a, b) in enumerate(zip(lines, planted, strict=True)) if a.text != b]
    assert len(changed) == 4
    assert all(unicodedata.is_normalized("NFC", t) for t in planted)
    assert fe.plant_errors(lines) == planted


def test_plant_errors_needs_a_date_line():
    with pytest.raises(ValueError):
        fe.plant_errors([ft.build_line([("x", None)])])


def test_scale_one_leaves_layout_unchanged_and_scale_multiplies_it():
    doc = _doc("en", "invoice")
    base, big = fp.layout_document(doc), fp.layout_document(doc, 2.0)
    assert [(p.x * 2, p.y * 2, p.w * 2) for p in base] == [(p.x, p.y, p.w) for p in big]
    assert fp.layout_document(doc, 1.0) == base


def test_cli_real_os_errors_are_one_line(tmp_path, capsys):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    assert mf.main(["--out", str(blocker)]) == 1
    assert _one_line_error(capsys) == f"error: not a folder: {blocker}\n"
    missing = tmp_path / "a" / "b"
    assert mf.main(["--out", str(missing)]) == 1
    assert _one_line_error(capsys) == f"error: not found: {missing}\n"
