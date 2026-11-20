"""Tests for the cell plan hook of the accuracy run: default plan, custom plan, page_outcome."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import accuracy_eval as ae
import accuracy_reads as ar
import run_accuracy as ra
import test_run_accuracy as t
from test_run_accuracy import patched  # noqa: F401 - autouse fixture

from folio_ocr import calibration
from folio_ocr.document import IDENTITY, Line, Page

PLAN = ar.CellPlan(
    cells={"model-en": ("model", "en"), "model-ar": ("model", "ar")},
    reads={
        "en": (("model-en", "en", False),),
        "ar": (("model-ar", "ar", False),),
        "en+ar": (("model-ar", "ar", False),),
    },
    readers=frozenset({"model"}),
    keys=lambda path: {"model-en": "K-en", "model-ar": "K-ar"},
)


class ModelReader(t.FakeReader):
    """A fake reader whose lines come from the reader called "model"."""

    def __call__(self, data, name, mode, with_tesseract):
        doc, seconds, boxes, orientations = super().__call__(data, name, mode, with_tesseract)
        key = "K-en" if mode == "en" else "K-ar"
        page = replace(doc.pages[0], lines=tuple(replace(ln, reader="model") for ln in doc.pages[0].lines))
        calib = replace(doc.engine.calibration, setup_key=key)
        return replace(doc, pages=(page,), engine=replace(doc.engine, calibration=calib)), seconds, boxes, orientations


@pytest.fixture
def model_taus(monkeypatch):
    monkeypatch.setitem(calibration.TAUS, "K-en", 0.5)
    monkeypatch.setitem(calibration.TAUS, "K-ar", 0.5)


def plan_run(argv, reader, plan=PLAN):
    return ra.main(argv, reader_factory=lambda s, tess: reader, plan=plan)


def strip(record):
    return {k: v for k, v in record.items() if k not in ("started_at", "finished_at")}


def graded(tmp_path, name, **kwargs):
    root = tmp_path / name
    root.mkdir()
    held = t.write_heldout(root, 1000, 1001)
    assert ra.main(t.grade_args(root, held), reader_factory=lambda s, tess: t.FakeReader(), **kwargs) == 0
    return strip(t.results_file(root))


def test_default_plan_gives_the_same_record(tmp_path):
    a, b = graded(tmp_path, "a"), graded(tmp_path, "b", plan=ar.DEFAULT_PLAN)
    for rec in (a, b):
        rec["host"] = {k: v for k, v in rec["host"].items()}
    assert a == b


def test_cell_reads_follow_the_plan():
    assert ar.cell_reads("en", PLAN) == (("model-en", "en", False),)
    with pytest.raises(ValueError):
        ar.cell_reads("fr", PLAN)
    assert ar.DEFAULT_PLAN.cells is ar.CELLS and ar.DEFAULT_PLAN.readers == ae.OCR_READERS


def test_custom_plan_lists_exactly_its_cells_and_counts_its_reader(tmp_path, model_taus):
    held = t.write_heldout(tmp_path, 1000, 1000)
    assert plan_run(t.grade_args(tmp_path, held), ModelReader(bad_tokens=())) == 0
    cells = t.cells_of(t.results_file(tmp_path))
    assert list(cells) == ["model-en", "model-ar"]
    assert cells["model-en"]["reader"] == "model" and cells["model-en"]["setup_key"] == "K-en"
    assert cells["model-en"]["ocr_read_lines"] > 0


def test_custom_plan_grade_once_is_refused(tmp_path, capsys, model_taus):
    held = t.write_heldout(tmp_path, 1000, 1000)
    assert plan_run(t.grade_args(tmp_path, held), ModelReader()) == 0
    again = ModelReader()
    assert plan_run(t.grade_args(tmp_path, held), again) == 1
    assert "already been graded" in capsys.readouterr().err and again.calls == []


def test_custom_plan_stop_rule_closes_a_cell(tmp_path, model_taus):
    held = t.write_heldout(tmp_path, 1000, 1011)
    assert plan_run(t.grade_args(tmp_path, held), ModelReader()) == 0
    cells = t.cells_of(t.results_file(tmp_path))
    assert cells["model-ar"]["stopped_by"] == "minimums"
    assert cells["model-ar"]["seeds_used"] == [1000, 1007]
    assert cells["model-en"]["stopped_by"] == "last seed"


def test_calibrate_skips_the_arabic_measure_without_onnxtr_cells(tmp_path, model_taus, capsys):
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "manifest.json").write_text(json.dumps({"documents": []}), "utf-8")
    argv = ["calibrate", "--dev", str(dev), "--work", str(tmp_path / "w"),
            "--out", str(tmp_path / "out.json"), "--arabic-in-en"]
    assert plan_run(argv, ModelReader()) == 0
    out = json.loads((tmp_path / "out.json").read_text("utf-8"))
    assert out["measures"] == {"arabic_in_en": {"skipped": "the plan has no OnnxTR cells"}}
    assert "the plan has no OnnxTR cells" in capsys.readouterr().out


def test_page_outcome_readers_decide_what_is_ocr_read():
    ln = Line("p1-l1", (0, 0, 100, 10), "a" * 50, "ltr", "latin", "model", 0.9, "reader", (),
              "auto_accepted", "a" * 50)
    page = Page(1, 1000, 1000, None, 0.0, IDENTITY, "ocr", "ltr", "read", (), (), (ln,))
    gt = {"file": "x.png", "homography": [list(r) for r in IDENTITY],
          "lines": [{"clean_box": [0, 0, 100, 10], "text": "a" * 50}]}
    assert ae.page_outcome(gt, page).gt[0].kind == "not_read"
    got = ae.page_outcome(gt, page, readers=frozenset({"model"}))
    assert got.gt[0].kind == "matched" and len(got.lines) == 1
    other = ae.page_outcome(gt, replace(page, lines=(replace(ln, reader="onnxtr"),)),
                            readers=frozenset({"model"}))
    assert other.gt[0].kind == "not_read" and other.lines == ()
