"""Checks for the development-variant document generator (fonts, text, pages, degradations, CLI)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import dev_degrade as dd
import dev_fonts as df
import dev_pages as dp
import dev_text as dt
import make_dev_variants as mdv
from fixture_images import ink_box, transform_box

ARABIC_MARKS = set("ًٌٍَُِّْـ")
SAMPLE_SEED = 7


def _text_only(pool_text: str) -> str:
    return pool_text.replace("{}", "")


# --- fonts ---------------------------------------------------------------------------------


def test_font_table_is_pinned_and_verified():
    assert set(df.LATIN_KEYS) | set(df.ARABIC_KEYS) == set(df.DEV_FONTS)
    for key in df.DEV_FONTS:
        assert df.dev_font_path(key).read_bytes()
    assert set(df.font_sha256s()) == {f.file for f in df.DEV_FONTS.values()}


def test_font_sha_mismatch_names_the_file(monkeypatch):
    bad = dataclasses.replace(df.DEV_FONTS["crimson"], sha256="0" * 64)
    monkeypatch.setattr(df, "DEV_FONTS", {**df.DEV_FONTS, "crimson": bad})
    df.dev_font_path.cache_clear()
    try:
        with pytest.raises(ValueError, match="CrimsonText-Regular.ttf"):
            df.dev_font_path("crimson")
    finally:
        df.dev_font_path.cache_clear()


def test_missing_font_file_names_the_file(monkeypatch, tmp_path):
    monkeypatch.setattr(df, "DEV_FONT_DIR", tmp_path)
    df.dev_font_path.cache_clear()
    try:
        with pytest.raises(ValueError, match="Lateef-Regular.ttf"):
            df.dev_font_path("lateef")
    finally:
        df.dev_font_path.cache_clear()


# --- text pools ----------------------------------------------------------------------------


def test_pool_sizes_and_arabic_hygiene():
    pools = dt.POOLS
    for lang in ("en", "ar"):
        assert len(pools[f"{lang}_names"]) >= 12
        assert len(pools[f"{lang}_orgs"]) >= 6
        assert len(pools[f"{lang}_items"]) >= 12
        assert len(pools[f"{lang}_sentences"]) >= 12
    for name, strings in pools.items():
        for s in strings:
            assert unicodedata.is_normalized("NFC", s), s
            if name.startswith("ar_"):
                assert not set(s) & ARABIC_MARKS, s


def _fonts_for(name: str) -> tuple[str, ...]:
    return df.ARABIC_KEYS if name.startswith("ar_") else df.LATIN_KEYS


def test_every_pool_character_is_in_every_font_of_its_script():
    from fontTools.ttLib import TTFont

    for name, strings in dt.POOLS.items():
        chars = {c for s in strings for c in _text_only(s)}
        if name.startswith("ar_"):
            chars = {c for c in chars if ord(c) > 127}
        for key in _fonts_for(name):
            cmap = TTFont(df.dev_font_path(key)).getBestCmap()
            missing = sorted(c for c in chars if ord(c) not in cmap)
            assert not missing, f"{key} lacks {missing!r} from pool {name}"


def test_arabic_pool_shapes_without_notdef_in_every_font_pair():
    for name, strings in dt.POOLS.items():
        if not name.startswith("ar_"):
            continue
        for s in strings:
            for arabic in df.ARABIC_KEYS:
                dp.check_text(_text_only(s), "crimson", arabic)


def test_check_text_rejects_a_missing_glyph():
    with pytest.raises(ValueError, match="no glyph"):
        dp.check_text("abc 中", "crimson", "lateef")


def test_documents_per_template_and_language():
    for language in dt.LANGUAGES:
        for template in dt.TEMPLATES:
            doc = dt.make_dev_document(language, template, random.Random(f"{language}{template}"))
            assert doc.layout == template and doc.language == language
            assert doc.direction == ("ltr" if language == "en" else "rtl")
            assert doc.rows
            for row in doc.rows:
                for cell in row.cells:
                    for e in cell.entities:
                        assert e.type in {"name", "date", "amount", "id", "phone"}
                        assert 0 <= e.start < e.end <= len(cell.text)


def test_unknown_language_or_template_is_rejected():
    with pytest.raises(ValueError):
        dt.make_dev_document("fr", "table", random.Random(1))
    with pytest.raises(ValueError):
        dt.make_dev_document("en", "invoice", random.Random(1))


def test_dates_use_new_formats_and_years():
    for seed in range(20):
        doc = dt.make_dev_document("en", "report", random.Random(seed))
        text = " ".join(c.text for r in doc.rows for c in r.cells)
        assert not any(m in text for m in ("January", "March", "2026"))


def test_arabic_documents_use_one_digit_system():
    ascii_digits, indic_digits = set("0123456789"), {chr(c) for c in range(0x660, 0x66A)}
    seen = set()
    for seed in range(30):
        for template in dt.TEMPLATES:
            doc = dt.make_dev_document("ar", template, random.Random(seed))
            chars = {c for r in doc.rows for cell in r.cells for c in cell.text}
            systems = (bool(chars & ascii_digits), bool(chars & indic_digits))
            assert systems != (True, True), (seed, template)
            seen.add(systems)
    assert (True, False) in seen and (False, True) in seen


def test_table_form_and_receipt_rows_are_separate_gt_lines():
    table = dt.make_dev_document("en", "table", random.Random(3))
    assert 7 <= len(table.rows) <= 11
    assert {len(r.cells) for r in table.rows} <= {4, 5} and len(
        {len(r.cells) for r in table.rows}
    ) == 1
    form = dt.make_dev_document("ar", "form", random.Random(3))
    assert all(len(r.cells) in (1, 2) for r in form.rows) and len(form.rows[0].cells) == 2
    receipt = dt.make_dev_document("en+ar", "receipt", random.Random(3))
    assert any(len(r.cells) == 2 for r in receipt.rows)


def test_grid_cells_are_single_characters():
    for language in dt.LANGUAGES:
        for seed in range(6):
            doc = dt.make_dev_document(language, "grid", random.Random(seed))
            assert 8 <= len(doc.rows) <= 12
            assert len({len(r.cells) for r in doc.rows}) == 1 and 6 <= len(doc.rows[0].cells) <= 10
            assert all(len(c.text) == 1 for r in doc.rows for c in r.cells)


# --- pages ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table_page():
    doc = dt.make_dev_document("en", "table", random.Random(11))
    return doc, dp.render_page(doc, "ptsans", "lateef", 11)


def test_page_has_one_box_per_cell_and_clear_gaps(table_page):
    doc, (pdf, img, boxes, size) = table_page
    assert (
        img.size[0] == 2480
        and img.size[1] in (3508, 3509)
        and size <= 11
        and pdf.startswith(b"%PDF")
    )
    cells = [c for r in doc.rows for c in r.cells]
    assert len(boxes) == len(cells)
    first_row = boxes[: len(doc.rows[0].cells)]
    for left, right in itertools.pairwise(first_row):
        assert right[0] - left[2] >= 2 * size * 300 / 72 - 1


def test_text_wider_than_its_cell_raises():
    doc = dt.make_dev_document("en", "table", random.Random(11))
    cells = dp.layout_cells(doc, 9)
    with pytest.raises(dp.DoesNotFit):
        dp.draw_pdf(cells, "courier", "lateef", 40)


# --- degradations --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_page(table_page):
    return table_page[1][1], table_page[1][2]


def _measured(img: Image.Image, box: list[int], pad: int = 10) -> list[int]:
    w, h = img.size
    rect = (max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad))
    return ink_box(np.asarray(img.convert("L")), rect)


def _assert_boxes_match(clean_boxes, img, h, tol=3):
    for cb in clean_boxes[:: max(1, len(clean_boxes) // 6)]:
        mapped = transform_box(h, cb, img.size)
        got = _measured(img, mapped)
        assert max(abs(a - b) for a, b in zip(mapped, got, strict=True)) <= tol, (mapped, got)


def test_clean_is_unchanged(clean_page):
    img, boxes = clean_page
    out, h, _params, ext = dd.degrade(img, "clean", np.random.default_rng(1))
    assert out.tobytes() == img.tobytes() and np.allclose(h, np.eye(3)) and ext == "png"
    _assert_boxes_match(boxes, out, h)


@pytest.mark.parametrize("quarter", [90, 180, 270])
def test_quarter_turn_boxes_match_ink(clean_page, quarter):
    img, boxes = clean_page
    out, h = dd.rotate_quarter(img, quarter)
    assert out.size == (img.size if quarter == 180 else img.size[::-1])
    _assert_boxes_match(boxes, out, h)


def test_lowres_size_homography_and_boxes(clean_page):
    img, boxes = clean_page
    out, h, params, ext = dd.degrade(img, "lowres", np.random.default_rng(5))
    s = params["scale"]
    assert 0.33 <= s <= 0.70 and ext == "jpg" and 40 <= params["jpeg_quality"] <= 90
    assert out.size == (round(img.size[0] * s), round(img.size[1] * s))
    assert np.allclose(h, np.diag([s, s, 1.0]))
    _assert_boxes_match(boxes, out, h)


@pytest.mark.parametrize("condition", sorted(dd.CONDITION_MIX))
def test_every_condition_records_params_and_homography(clean_page, condition):
    img, _ = clean_page
    small = img.resize((620, 877))
    out, h, params, ext = dd.degrade(small, condition, np.random.default_rng(2))
    assert h.shape == (3, 3) and ext in ("png", "jpg") and isinstance(params, dict)
    json.dumps(params)
    assert dd.encode_image(out, ext, params)
    if condition == "fax":
        assert out.mode == "L" and set(np.unique(np.asarray(out))) <= {0, 255}


def test_condition_mix_sums_to_one_and_grid_favours_rotation():
    assert sum(dd.CONDITION_MIX.values()) == pytest.approx(1.0)
    grid = dd.condition_mix("grid")
    assert grid["rotated"] == pytest.approx(0.5) and sum(grid.values()) == pytest.approx(1.0)
    assert dd.condition_mix("table") == dd.CONDITION_MIX


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        dd.degrade(Image.new("L", (10, 10)), "smudged", np.random.default_rng(1))


# --- command line --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("dev")
    assert mdv.main(["--first", str(SAMPLE_SEED), "--count", "1", "--out", str(out)]) == 0
    return out


def test_output_names_gt_and_manifest(built):
    manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "folio-dev/1"
    assert manifest["seeds"] == {"first": SAMPLE_SEED, "last": SAMPLE_SEED}
    assert set(manifest["fonts"]) == {f.file for f in df.DEV_FONTS.values()}
    assert "scripts/make_dev_variants.py" in manifest["generator_files"]
    assert [d["language"] for d in manifest["documents"]] == ["ar", "en", "en+ar"]
    for doc in manifest["documents"]:
        gt = json.loads((built / doc["gt"]).read_text(encoding="utf-8"))
        stem = Path(doc["file"]).stem
        assert stem == gt["unit"] and doc["unit"] == stem
        assert stem.startswith(f"dev-{SAMPLE_SEED:05d}-")
        assert gt["file"] == doc["file"] and gt["condition"] == doc["condition"]
        with Image.open(built / doc["file"]) as img:
            assert img.size == (gt["width"], gt["height"])
        for key in ("homography", "lines", "params", "dpi", "direction", "layout", "language"):
            assert key in gt
        assert all(
            {"text", "box", "clean_box", "direction", "entities"} <= set(ln) for ln in gt["lines"]
        )


def test_same_seed_gives_identical_bytes(built, tmp_path):
    assert mdv.main(["--first", str(SAMPLE_SEED), "--count", "1", "--out", str(tmp_path)]) == 0
    names = sorted(p.name for p in built.iterdir())
    assert names == sorted(p.name for p in tmp_path.iterdir())
    for name in names:
        assert (built / name).read_bytes() == (tmp_path / name).read_bytes(), name


@pytest.mark.parametrize(
    "args",
    [
        ["--first", "-1", "--count", "1"],
        ["--first", "99999", "--count", "2"],
        ["--first", "x", "--count", "1"],
        ["--first", "0", "--count", "0"],
        ["--first", "100000", "--count", "1"],
    ],
)
def test_out_of_range_seeds_give_one_line_error(args, tmp_path, capsys):
    assert mdv.main([*args, "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert err == "error: --first/--count must stay within 0..99999\n"
    assert not (tmp_path / "o").exists()


def test_missing_parent_folder_gives_one_line_error(tmp_path, capsys):
    assert mdv.main(["--first", "0", "--count", "1", "--out", str(tmp_path / "a" / "b")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1 and "Traceback" not in err
