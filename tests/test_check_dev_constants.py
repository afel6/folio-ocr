"""Tests for the dev-constants check: summary, verdict, command line and the process pool."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import check_dev_constants as cdc

from folio_ocr import ocr_setup, orientation
from folio_ocr.geometry import rotation


def page(unit="a", **over):
    base = {
        "unit": unit, "true_quarter": 0, "vote_branch": "pair", "vote_margin": 0.5,
        "vote_angle": 0.0, "vote_uncertain": False, "vote_wrong": False,
        "osd": [0, 3.0], "final_source": "osd", "final_angle": 0.0, "final_uncertain": False,
        "final_wrong": False, "recall_matched": 9, "recall_total": 10,
    }
    return {**base, **over}


def reference_for(pages):
    return {"summary": cdc.summarise(pages, None), "pages": {p["unit"]: p for p in pages}}


def test_summary_counts_branches_osd_and_recall():
    pages = [
        page("a"),
        page("b", vote_wrong=True, vote_margin=0.2, vote_uncertain=True, osd=None,
             final_source="vote", final_wrong=True, final_uncertain=True),
        page("c", vote_branch="four", vote_wrong=True, vote_margin=0.3, osd="failure: x",
             recall_matched=None, recall_total=None),
        page("d", osd=[90, 4.0], final_wrong=True),
    ]
    s = cdc.summarise(pages, None)
    assert s["vote"]["pair"] == {
        "pages": 3, "wrong": 1, "largest_wrong_margin": 0.2, "wrong_unflagged": 0}
    assert s["vote"]["four"]["wrong_unflagged"] == 1
    assert s["osd"] == {"right": 1, "no_answer": 1, "failure": 1, "wrong": 1}
    assert s["osd_used_wrong"] == 1
    assert s["final"] == {"wrong": 2, "wrong_unflagged": 1}
    assert s["recall"] == {"matched": 27, "total": 30, "share": 0.9}


def test_summary_reports_differences_from_reference():
    ref = reference_for([page("a"), page("b"), page("z")])
    s = cdc.summarise([page("a", vote_angle=90.0), page("b", osd=[90, 2.0]), page("c")], ref)
    assert s["vote_angle_differs"] == {"count": 1, "first": ["a"]}
    assert s["osd_differs"] == {"count": 1, "first": ["b"]}
    assert s["manifest_differs"] == 2


def test_verdict_holds_for_identical_run():
    pages = [page("a"), page("b")]
    ref = reference_for(pages)
    assert cdc.verdict(cdc.summarise(pages, ref), ref) == []


def test_verdict_names_each_problem():
    base = [page("a")]
    ref = reference_for(base)
    bad = [page("a", vote_wrong=True, final_wrong=True, recall_matched=5), page("b")]
    problems = cdc.verdict(cdc.summarise(bad, ref), ref)
    text = " | ".join(problems)
    assert "vote" in text and "osd" in text.lower() and "final" in text
    assert "recall" in text and "manifest" in text


def make_dev(tmp_path):
    raw = b'{"documents": []}'
    (tmp_path / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def frozen():
    return {
        "uncertain_margin_pair": orientation.UNCERTAIN_MARGIN_PAIR,
        "uncertain_margin_four": orientation.UNCERTAIN_MARGIN_FOUR,
        "osd_floor": orientation.OSD_FLOOR, "gap_factor": ocr_setup.OCR_GAP_FACTOR,
    }


def run(tmp_path, ref, capsys, *extra):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref))
    code = cdc.main(["--dev", str(tmp_path), "--reference", str(path),
                     "--out", str(tmp_path / "out.json"), *extra])
    return code, capsys.readouterr()


def test_cli_refuses_other_manifest(tmp_path, capsys):
    make_dev(tmp_path)
    code, seen = run(tmp_path, {"dev_manifest_sha256": "0", "frozen": frozen()}, capsys)
    assert code == 1 and seen.err.startswith("error: ") and "manifest" in seen.err


def test_cli_refuses_other_constants(tmp_path, capsys):
    sha = make_dev(tmp_path)
    ref = {"dev_manifest_sha256": sha, "frozen": {**frozen(), "osd_floor": 9.0}}
    code, seen = run(tmp_path, ref, capsys)
    assert code == 1 and "frozen" in seen.err


def test_cli_rejects_bad_jobs_and_missing_reference(tmp_path, capsys):
    sha = make_dev(tmp_path)
    ref = {"dev_manifest_sha256": sha, "frozen": frozen()}
    assert run(tmp_path, ref, capsys, "--jobs", "0")[0] == 1
    code = cdc.main(["--dev", str(tmp_path), "--reference", str(tmp_path / "no.json"),
                     "--out", str(tmp_path / "o.json")])
    assert code == 1 and "not found" in capsys.readouterr().err


def fake_check(path, gt):
    return page(unit=str(gt["unit"]))


def test_pool_and_serial_records_match():
    tasks = [(Path(f"{i}.png"), {"unit": f"u{i}"}) for i in range(5)]
    assert cdc.check_pages(tasks, 1, fake_check) == cdc.check_pages(tasks, 2, fake_check)


# ---------------------------------------------------------------- orientation helper

@pytest.mark.parametrize("quarter", [0, 90, 180, 270])
def test_true_quarter_is_the_turn_that_makes_the_page_upright(quarter):
    w, h = 300, 200
    homography = rotation(w, h, (360 - quarter) % 360)[0]
    assert cdc._true_quarter(w, h, homography) == quarter
    assert cdc._off_upright(w, h, quarter, homography) is False
    assert cdc._off_upright(w, h, (quarter + 180) % 360, homography) is True


# ---------------------------------------------------------------- the whole run

def make_run(tmp_path, monkeypatch, pages, *, reference_pages=None):
    """A dev folder of one page per record, a matching reference and a fake page check."""
    docs = [{"file": f"{p['unit']}.png", "gt": f"{p['unit']}.gt.json"} for p in pages]
    raw = json.dumps({"documents": docs}).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    for p in pages:
        (tmp_path / f"{p['unit']}.gt.json").write_text(json.dumps({"unit": p["unit"]}))
    ref = {"dev_manifest_sha256": hashlib.sha256(raw).hexdigest(), "frozen": frozen(),
           **reference_for(reference_pages or pages)}
    seen = {p["unit"]: p for p in pages}
    monkeypatch.setattr(cdc, "_environment", lambda: None)
    monkeypatch.setattr(cdc, "page_check", lambda path, gt: seen[gt["unit"]])
    return ref


def test_run_reports_that_the_constants_hold_and_writes_the_results(
    tmp_path, monkeypatch, capsys
):
    ref = make_run(tmp_path, monkeypatch, [page("a"), page("b")])
    code, seen = run(tmp_path, ref, capsys)
    assert code == 0 and "constants hold" in seen.out and "STOP" not in seen.out
    written = json.loads((tmp_path / "out.json").read_text())
    assert written["schema"] == cdc.SCHEMA and written["problems"] == []
    assert set(written["pages"]) == {"a", "b"} and written["frozen"] == frozen()


def test_run_stops_with_one_line_per_problem(tmp_path, monkeypatch, capsys):
    bad = [page("a", vote_wrong=True, final_wrong=True), page("b")]
    ref = make_run(tmp_path, monkeypatch, bad, reference_pages=[page("a"), page("b")])
    code, seen = run(tmp_path, ref, capsys)
    stops = [ln for ln in seen.out.splitlines() if ln.startswith("STOP:")]
    assert code == 1 and len(stops) >= 2 and "constants hold" not in seen.out
    assert json.loads((tmp_path / "out.json").read_text())["problems"] == [
        s[len("STOP: "):] for s in stops
    ]


def test_run_checks_the_out_folder_before_any_page(tmp_path, monkeypatch, capsys):
    ref = make_run(tmp_path, monkeypatch, [page("a")])
    monkeypatch.setattr(cdc, "page_check", lambda *a: pytest.fail("a page was checked"))
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref))
    code = cdc.main(["--dev", str(tmp_path), "--reference", str(path),
                     "--out", str(tmp_path / "missing" / "out.json")])
    err = capsys.readouterr().err
    assert code == 1 and err == "error: the folder for --out does not exist\n"


def test_environment_without_tesseract_is_one_line(monkeypatch):
    cdc._environment.cache_clear()
    monkeypatch.setattr(cdc, "find_tesseract", lambda settings: None)
    with pytest.raises(cdc.CheckError, match="Tesseract was not found"):
        cdc._environment()
    cdc._environment.cache_clear()


def test_osd_field_and_reference_loading_edges(tmp_path):
    assert cdc._osd_field(None, "boom") == "failure: boom"
    assert cdc._osd_field(None, None) is None
    assert cdc._osd_field(SimpleNamespace(quarter=90, confidence=2.0), None) == [90, 2.0]
    for raw, text in ((b"{", "not valid JSON"), (b"[]", "not a JSON object")):
        with pytest.raises(cdc.CheckError, match=text):
            cdc._load_json(raw, "reference file")
    sha = make_dev(tmp_path)
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"dev_manifest_sha256": sha, "frozen": frozen()}))
    with pytest.raises(cdc.CheckError, match="lacks its summary"):
        cdc._load_reference(path, sha)
    with pytest.raises(cdc.CheckError, match="not a valid manifest"):
        cdc._tasks(tmp_path, b'{"documents": [{"file": "a"}]}')


# ---------------------------------------------------------------- one page, with the models faked

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
GT_PAGE = {"unit": "a", "homography": IDENTITY, "lines": [{"clean_box": [0, 0, 10, 10]}]}


def png_file(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (40, 30), "white").save(path)
    return path


def fake_recall_models(monkeypatch, *, boxes=((0, 0, 10, 10),), skew=0):
    monkeypatch.setattr(cdc.onnxtr_models, "detect", lambda models, rgb: "first")
    found = SimpleNamespace(boxes=[1])
    monkeypatch.setattr(cdc.orientation, "_turned", lambda *a: (None, found))
    monkeypatch.setattr(cdc.orientation, "_skew", lambda f: skew)
    monkeypatch.setattr(cdc, "find_words", lambda *a: list(boxes))
    monkeypatch.setattr(
        cdc, "group_lines", lambda gray, words, d: [SimpleNamespace(box=w.box) for w in words]
    )


def test_recall_counts_matched_lines_after_straightening(tmp_path, monkeypatch):
    fake_recall_models(monkeypatch)
    image = Image.open(png_file(tmp_path))
    assert cdc._recall(None, image, GT_PAGE, IDENTITY, 0) == (1, 1)


def test_recall_is_unknown_without_a_true_quarter_or_without_words(tmp_path, monkeypatch):
    image = Image.open(png_file(tmp_path))
    assert cdc._recall(None, image, GT_PAGE, IDENTITY, None) == (None, None)
    fake_recall_models(monkeypatch, boxes=(), skew=3)
    assert cdc._recall(None, image, GT_PAGE, IDENTITY, 0) == (None, None)


def fake_page_models(monkeypatch, answer):
    monkeypatch.setattr(cdc, "_environment", lambda: (Path("tess"), Path("data"), "models"))
    monkeypatch.setattr(cdc, "_recall", lambda *a: (9, 10))

    def osd(binary, tessdata, png, timeout):
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(cdc.tesseract, "osd", osd)

    def vote(models, image, osd=None):
        source = "vote"
        if osd is not None:
            try:
                osd(image)
                source = "osd"
            except cdc.tesseract.TesseractFailure:
                pass
        return SimpleNamespace(branch="pair", margin=0.5, angle=0.0, uncertain=False,
                               source=source)

    monkeypatch.setattr(cdc.orientation, "page_orientation", vote)


def test_page_check_records_the_vote_the_osd_answer_and_the_final_choice(tmp_path, monkeypatch):
    fake_page_models(monkeypatch, SimpleNamespace(quarter=0, confidence=3.0))
    record = cdc.page_check(png_file(tmp_path), GT_PAGE)
    assert record == {
        "true_quarter": 0, "vote_branch": "pair", "vote_margin": 0.5, "vote_angle": 0.0,
        "vote_uncertain": False, "vote_wrong": False, "osd": [0, 3.0], "final_source": "osd",
        "final_angle": 0.0, "final_uncertain": False, "final_wrong": False,
        "recall_matched": 9, "recall_total": 10,
    }


def test_page_check_records_a_tesseract_failure_and_falls_back_to_the_vote(tmp_path, monkeypatch):
    fake_page_models(monkeypatch, cdc.tesseract.TesseractFailure("it hung"))
    record = cdc.page_check(png_file(tmp_path), GT_PAGE)
    assert record["osd"] == "failure: it hung" and record["final_source"] == "vote"


def test_environment_loads_the_models_once(monkeypatch):
    cdc._environment.cache_clear()
    monkeypatch.setattr(cdc, "find_tesseract", lambda settings: Path("tess"))
    monkeypatch.setattr(cdc.onnxtr_models, "load", lambda folder: "models")
    binary, tessdata, models = cdc._environment()
    assert (binary, tessdata.name, models) == (Path("tess"), "tessdata", "models")
    cdc._environment.cache_clear()
