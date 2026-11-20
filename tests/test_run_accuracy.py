"""Tests for the accuracy run: calibrate, grade and fixtures, with a fake reader."""

import hashlib
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import accuracy_dev as ad
import accuracy_eval
import accuracy_reads as ar
import accuracy_stats
import heldout
import run_accuracy as ra

from folio_ocr import calibration
from folio_ocr import ocr as ocr_module
from folio_ocr.document import (
    Calibration,
    Document,
    Engine,
    Line,
    Page,
    Source,
)
from folio_ocr.tesseract import TesseractFailure

KEYS = {"onnxtr-en-osd": "KEY-osd", "onnxtr-en-vote": "KEY-vote",
        "tesseract-en": "KEY-tess", "tesseract-ar": "KEY-tess"}
TAUS = {"KEY-osd": 0.5, "KEY-vote": 0.4, "KEY-tess": 0.6}
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ARABIC = "مرحبا"


def gt_for(name: str, n: int, text: str = "hello") -> dict:
    lines = [{"text": text, "clean_box": [0, 30 * i, 100, 30 * i + 20]} for i in range(n)]
    return {"file": name, "homography": [list(r) for r in IDENT], "lines": lines}


def make_doc(name, n, *, bad, conf, status, key, tau) -> Document:
    lines = tuple(
        Line(
            f"p1-l{i + 1}", (0.0, 30.0 * i, 100.0, 30.0 * i + 20), "xxxxx" if bad else "hello",
            "ltr", "latin", "onnxtr", conf, "reader", (), status, "hello",
        )
        for i in range(n)
    )
    page = Page(1, 100, 100, None, 0.0, IDENT, "ocr", "ltr", "read", (), (), lines)
    engine = Engine("d", None, "onnxtr", None, None, "untested", None, Calibration(key, tau))
    return Document(
        "id", Source(name, "0" * 64, "image", 1), "en", engine, (page,),
        datetime(2026, 1, 1, tzinfo=UTC), False,
    )


class FakeReader:
    """Reads by file name token and image data; records every call."""

    def __init__(self, *, n=4, calls=None, bad_tokens=("ar", "mixed"), status="auto_accepted",
                 tess_key="KEY-tess", conf=0.9, on_call=None, split_boxes=(), wrong=None):
        self.n, self.calls, self.bad_tokens = n, calls if calls is not None else [], bad_tokens
        self.status, self.tess_key, self.conf, self.on_call = status, tess_key, conf, on_call
        self.split_boxes, self.wrong = split_boxes, wrong

    def __call__(self, data, name, mode, with_tesseract):
        earlier = self.calls.count((name, mode, with_tesseract))
        self.calls.append((name, mode, with_tesseract))
        if self.on_call:
            self.on_call()
        token = name.split("-")[0]
        if mode != "en":
            key, tau = self.tess_key, 0.6
        else:
            key, tau = ("KEY-osd", 0.5) if with_tesseract else ("KEY-vote", 0.4)
        if self.wrong is not None and self.wrong(name, mode, with_tesseract, earlier):
            key = "OTHER"
        conf = self.conf
        if data.startswith(b"dev"):
            conf = (0.1 if with_tesseract else 0.15) if mode == "en" and token == "en" else (
                0.9 if mode == "en" else 0.2 if mode == "en+ar" and token == "en" else 0.3
            )
        bad = token in self.bad_tokens or data.startswith(b"dev")
        doc = make_doc(name, self.n, bad=bad, conf=conf, status=self.status, key=key, tau=tau)
        source = "osd" if with_tesseract else "vote"
        return doc, 0.25, self.split_boxes, ((source, False, "pair", None, False),)


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    monkeypatch.setattr(calibration, "TAUS", dict(TAUS))
    monkeypatch.setattr(calibration, "lookup", lambda key: calibration.TAUS.get(key))
    monkeypatch.setattr(ar, "expected_keys", lambda path: dict(KEYS))
    monkeypatch.setattr(ra, "find_tesseract", lambda settings: Path("C:/fake/tesseract.exe"))
    monkeypatch.setattr(ra, "decided_version", lambda path: "5.3.0")
    monkeypatch.setattr(ra, "tessdata_problem", lambda folder: None)

    def fake_images(seed, token, layout):
        stem = f"{token}-{layout}"
        return tuple(
            heldout.HeldoutImage(
                f"{stem}-{c}.png", f"{seed}|{token}|{layout}|{c}".encode(),
                gt_for(f"{stem}-{c}.png", 4), c, f"{seed}-{stem}",
            )
            for c in heldout.CONDITIONS
        )

    monkeypatch.setattr(heldout, "held_out_images", fake_images)


def run(argv, reader):
    return ra.main(argv, reader_factory=lambda settings, tess: reader)


def write_heldout(tmp_path, first, last, mutate=None) -> Path:
    manifest = heldout.build_manifest(first, last, "t")
    if mutate:
        mutate(manifest)
    path = tmp_path / "held.json"
    heldout.write_manifest(manifest, path)
    return path


def grade_args(tmp_path, held):
    return ["grade", "--heldout", str(held), "--work", str(tmp_path / "work"),
            "--results-dir", str(tmp_path / "results")]


def results_file(tmp_path) -> dict:
    (path,) = (tmp_path / "results").glob("grade-*.json")
    return json.loads(path.read_text("utf-8"))


def cells_of(record) -> dict:
    return {c["cell"]: c for c in record["cells"]}


# ---------------------------------------------------------------- reads and cache

def test_cell_reads_by_language():
    assert ar.cell_reads("en") == (
        ("onnxtr-en-osd", "en", True),
        ("onnxtr-en-vote", "en", False),
        ("tesseract-en", "en+ar", True),
    )
    assert ar.cell_reads("ar") == (("tesseract-ar", "ar", True),)
    assert ar.cell_reads("en+ar") == (("tesseract-ar", "en+ar", True),)
    with pytest.raises(ValueError):
        ar.cell_reads("fr")


def test_cache_round_trip_keeps_pages():
    doc = make_doc("a", 2, bad=False, conf=0.9, status="auto_accepted", key="K", tau=0.5)
    obj = json.loads(json.dumps(ar.document_to_cache(doc, 1.5)))
    assert (obj["setup_key"], obj["tau"], obj["seconds"]) == ("K", 0.5, 1.5)
    assert ar.pages_from_cache(obj) == doc.pages


def test_cache_round_trip_keeps_split_boxes():
    doc = make_doc("a", 2, bad=False, conf=0.9, status="auto_accepted", key="K", tau=0.5)
    obj = json.loads(json.dumps(ar.document_to_cache(doc, 1.5, ((1.0, 2.0, 3.0, 4.0),))))
    assert ar.split_boxes_from_cache(obj) == ((1.0, 2.0, 3.0, 4.0),)
    assert ar.split_boxes_from_cache({}) == ()


def test_cached_read_restores_split_boxes(tmp_path):
    reader = FakeReader(split_boxes=((1.0, 2.0, 3.0, 4.0),))
    args = (tmp_path, "KEY-osd", "ab" * 32, b"x", "en-a.png", "en", True, reader)
    ar.cached_read(*args)
    assert ar.split_boxes_from_cache(ar.cached_read(*args)) == ((1.0, 2.0, 3.0, 4.0),)
    assert len(reader.calls) == 1


# ---------------------------------------------------------------- split-word capture

def _fake_recognise(models, rgb, boxes):
    return [SimpleNamespace(box=b, split=b[0] >= 100) for b in boxes]


def test_split_capture_keeps_threads_apart(monkeypatch):
    monkeypatch.setattr(ocr_module, "recognise_words", _fake_recognise)
    ar.install_split_capture()
    ar.install_split_capture()  # wrapping twice must not duplicate entries
    barrier, got = threading.Barrier(2), {}

    def work(tag, boxes):
        ar.reset_split_boxes()
        barrier.wait()
        ocr_module.recognise_words(None, None, boxes)
        barrier.wait()
        got[tag] = ar.take_split_boxes()

    a = threading.Thread(target=work, args=("a", [(100, 0, 110, 5), (0, 0, 5, 5)]))
    b = threading.Thread(target=work, args=("b", [(200, 0, 210, 5)]))
    a.start()
    b.start()
    a.join()
    b.join()
    assert got == {"a": ((100, 0, 110, 5),), "b": ((200, 0, 210, 5),)}


def test_split_capture_without_split_field(monkeypatch):
    monkeypatch.setattr(
        ocr_module, "recognise_words", lambda m, r, boxes: [SimpleNamespace(box=b) for b in boxes]
    )
    ar.install_split_capture()
    ar.reset_split_boxes()
    assert ocr_module.recognise_words(None, None, [(1, 2, 3, 4)])[0].box == (1, 2, 3, 4)
    assert ar.take_split_boxes() == ()


def test_pipeline_reader_returns_split_boxes(monkeypatch):
    monkeypatch.setattr(ocr_module, "recognise_words", _fake_recognise)
    doc = make_doc("a", 1, bad=False, conf=0.9, status="auto_accepted", key="K", tau=0.5)

    def fake_read_bytes(data, name, **kwargs):
        ocr_module.recognise_words(None, None, [(100, 0, 110, 5), (0, 0, 5, 5)])
        return doc

    monkeypatch.setattr(ar, "read_bytes", fake_read_bytes)
    settings = SimpleNamespace(weights_dir=None, line_timeout_s=1)
    read = ar.pipeline_reader(settings, None)
    got_doc, seconds, boxes, orientations = read(b"x", "a.png", "en", False)
    assert got_doc is doc and seconds >= 0 and boxes == ((100, 0, 110, 5),)
    assert orientations == ()
    assert read(b"x", "a.png", "en", False)[2] == ((100, 0, 110, 5),)  # cleared before each read


def test_cached_read_reads_once(tmp_path):
    reader = FakeReader()
    args = (tmp_path, "KEY-tess", "ab" * 32, b"x", "en-a.png", "en+ar", True, reader)
    first = ar.cached_read(*args)
    assert ar.cached_read(*args) == first
    assert len(reader.calls) == 1
    (path,) = (tmp_path / "reads").rglob("*.json")
    assert path.name == f"{'ab' * 32}-en_ar.json"


# ---------------------------------------------------------------- grade

def test_grade_twice_refuses_second_run_without_reading(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    capsys.readouterr()
    again = FakeReader()
    assert run(grade_args(tmp_path, held), again) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "already been graded" in err and err.count("\n") == 1
    assert again.calls == []


def test_attempt_record_is_on_disk_before_first_read(tmp_path):
    held = write_heldout(tmp_path, 1000, 1000)
    seen = []

    def check():
        (path,) = (tmp_path / "results").glob("grade-*.json")
        seen.append(json.loads(path.read_text("utf-8"))["status"])

    assert run(grade_args(tmp_path, held), FakeReader(on_call=check)) == 0
    assert set(seen) == {"started"}
    assert results_file(tmp_path)["status"] == "finished"


def test_generator_mismatch_stops_before_any_attempt(tmp_path, capsys):
    def tamper(m):
        first = next(iter(m["generator_files"]))
        m["generator_files"][first] = "0" * 64

    held = write_heldout(tmp_path, 1000, 1001, tamper)
    reader = FakeReader()
    assert run(grade_args(tmp_path, held), reader) == 1
    assert "changed since the held-out manifest" in capsys.readouterr().err
    assert reader.calls == []
    assert not list((tmp_path / "results").glob("*.json"))


def _only(name, wrong_reads):
    """`wrong` for FakeReader: the first `wrong_reads` reads of document `name` carry OTHER."""
    return lambda n, mode, tess, earlier: n == name and earlier < wrong_reads


AR_DOC = "ar-invoice-clean.png"


def test_a_document_read_with_another_key_once_is_read_again(tmp_path):
    held = write_heldout(tmp_path, 1000, 1001)
    reader = FakeReader(wrong=_only(AR_DOC, 1))
    assert run(grade_args(tmp_path, held), reader) == 0
    assert len([c for c in reader.calls if c[0] == AR_DOC]) >= 2
    for cell in cells_of(results_file(tmp_path)).values():
        assert "mismatch" not in (cell["reason"] or "")


def test_a_document_read_twice_with_another_key_stops_the_run(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1001)
    reader = FakeReader(wrong=_only(AR_DOC, 99))
    assert run(grade_args(tmp_path, held), reader) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert AR_DOC in err and "OTHER" in err and "KEY-tess" in err
    assert results_file(tmp_path)["status"] == "started"
    assert len([c for c in reader.calls if c[0] == AR_DOC]) == 2
    assert not [p for p in (tmp_path / "work" / "reads").rglob("*.json") if "OTHER" in p.read_text("utf-8")]


def test_a_setup_key_off_the_cell_reads_not_measured():
    state = ra.CellState(setup_keys=frozenset({"OTHER"}))
    rec = ra._cell_record("tesseract-en", state, KEYS)
    assert rec["reason"] == "not measured (setup key mismatch)"


def test_grade_refuses_a_damaged_tesseract_language_data_before_any_file(
        tmp_path, capsys, monkeypatch):
    message = "Tesseract language data is damaged; run folio setup"
    monkeypatch.setattr(ra, "tessdata_problem", lambda folder: message)
    held = write_heldout(tmp_path, 1000, 1001)
    reader = FakeReader()
    assert run(grade_args(tmp_path, held), reader) == 1
    assert capsys.readouterr().err == f"error: {message}\n"
    assert not list((tmp_path / "results").glob("*.json")) and reader.calls == []


def test_grade_refuses_an_undecided_tesseract_version_before_any_file(
        tmp_path, capsys, monkeypatch):
    def timed_out(path):
        raise TesseractFailure("Tesseract did not report its version within 10 s")

    monkeypatch.setattr(ra, "decided_version", timed_out)
    held = write_heldout(tmp_path, 1000, 1001)
    reader = FakeReader()
    assert run(grade_args(tmp_path, held), reader) == 1
    assert capsys.readouterr().err == (
        "error: Tesseract did not report its version within 10 s\n")
    assert not list((tmp_path / "results").glob("*.json")) and reader.calls == []


def test_missing_tau_table_entry_reads_no_frozen_tau(tmp_path):
    del calibration.TAUS["KEY-tess"]
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    cells = cells_of(results_file(tmp_path))
    for cell in ("tesseract-en", "tesseract-ar"):
        assert cells[cell]["reason"] == "not measured (no frozen tau)"


def test_attempt_record_holds_the_seed_range(tmp_path):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    assert results_file(tmp_path)["seeds"] == {"first": 1000, "last": 1001}


def test_grade_refuses_a_range_overlapping_a_recorded_attempt(tmp_path, capsys):
    first = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, first), FakeReader()) == 0
    other = tmp_path / "elsewhere"
    other.mkdir()
    manifest = heldout.build_manifest(1001, 1002, "again")
    heldout.write_manifest(manifest, other / "again.json")
    capsys.readouterr()
    again = FakeReader()
    assert run(grade_args(tmp_path, other / "again.json"), again) == 1
    assert "overlaps an earlier held-out batch" in capsys.readouterr().err
    assert again.calls == []


def test_flags_every_line_is_false_when_a_line_reaches_tau(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    assert all(c["flags_every_line"] is False for c in results_file(tmp_path)["cells"])
    assert "flags every line" not in capsys.readouterr().out


def test_flags_every_line_is_true_when_tau_is_above_every_confidence(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader(conf=0.3, status="needs_review")) == 0
    assert all(c["flags_every_line"] is True for c in results_file(tmp_path)["cells"])
    shown = capsys.readouterr().out.splitlines()
    assert all(line.endswith(" (flags every line)") for line in shown[:4])


def _units(*confidences):
    lines = tuple(accuracy_eval.OutputLine(c, False, False, "needs_review", True, True, None)
                  for c in confidences)
    return [accuracy_stats.Unit("u", (accuracy_eval.PageOutcome((), lines),))]


def test_flags_every_line_helper_edges():
    assert ra._flags_every_line(_units(0.3, 0.5), 0.51) is True
    assert ra._flags_every_line(_units(0.3, 0.5), 0.5) is False
    assert ra._flags_every_line(_units(None, None), 0.0) is True
    assert ra._flags_every_line([], 0.5) is True
    assert ra._flags_every_line(_units(0.3), None) is False


def test_stop_rule_closes_a_cell_while_others_run_on(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1011)
    reader = FakeReader()
    assert run(grade_args(tmp_path, held), reader) == 0
    out = capsys.readouterr().out
    onnx = [c for c in reader.calls if c[1] == "en"]  # the OSD and the vote cell
    ar_reads = [c for c in reader.calls if c[1] in ("ar",) or c[0].startswith("mixed")]
    assert len(onnx) == 12 * 2 * 2 * 3
    assert len(ar_reads) == 8 * 4 * 3  # 32 documents with bad lines at the eighth seed
    cells = cells_of(results_file(tmp_path))
    assert cells["tesseract-ar"]["seeds_used"] == [1000, 1007]
    assert cells["tesseract-ar"]["stopped_by"] == "minimums"
    assert cells["tesseract-ar"]["status"] == "fail"
    for cell in ("onnxtr-en-osd", "onnxtr-en-vote"):
        assert cells[cell]["seeds_used"] == [1000, 1011]
        assert cells[cell]["stopped_by"] == "last seed"
    assert "tesseract-ar: fail 100.0 %" in out and "bad lines from 32 documents" in out
    assert out.strip().splitlines()[-1].endswith(".json")


def test_status_disagreement_is_an_error(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1000)
    assert run(grade_args(tmp_path, held), FakeReader(status="needs_review")) == 1
    assert "differs from the threshold rule" in capsys.readouterr().err


def test_results_record_names_the_orientation_margins_and_floor(tmp_path):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    record = results_file(tmp_path)
    assert record["uncertain_margins"] == {"pair": 0.13, "four": 0.41}
    assert record["osd_floor"] == 1.5


def test_results_record_has_every_listed_key(tmp_path):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    record = results_file(tmp_path)
    assert {"schema", "mode", "status", "started_at", "finished_at", "host", "versions",
            "tau_table_sha256", "heldout_manifest_sha256", "dev_manifest_sha256", "cells",
            "pages"} <= set(record)
    assert record["schema"] == "folio-accuracy/1" and record["mode"] == "grade"
    assert record["heldout_manifest_sha256"] == hashlib.sha256(held.read_bytes()).hexdigest()
    assert record["finished_at"].endswith("Z")
    assert {"python", "platform", "cpu_count", "jobs"} == set(record["host"])
    assert record["versions"]["tesseract"] == "5.3.0"
    assert record["pages"]["count"] == 60
    cell_keys = {"cell", "reader", "document_language", "setup_key", "tau", "status", "reason",
                 "documents", "documents_with_bad_lines", "ocr_read_lines", "good_lines",
                 "bad_lines", "missed", "missed_error_rate", "fail_closed_lines",
                 "fail_closed_share", "flag_rate_all", "flag_rate_good", "low_confidence_share",
                 "pure_deletions", "grouping", "caught_unread", "pure_deletion_share",
                 "auto_accepted_unmatched", "not_read_lines", "seeds_used", "stopped_by"}
    for cell in record["cells"]:
        assert cell_keys <= set(cell)
        assert {"point", "low", "high", "bound_95", "resamples", "unit"} == set(
            cell["missed_error_rate"]
        )
    assert record["pages"]["seconds_p50"] == 0.25


# ---------------------------------------------------------------- calibrate

def write_dev(root: Path, per_language=20, lines=5) -> Path:
    root.mkdir()
    docs = []
    for lang, token in (("en", "en"), ("ar", "ar"), ("en+ar", "mixed")):
        for i in range(per_language * (2 if lang == "en" else 1)):
            name = f"{token}-{i}.png"
            data = f"dev|{token}|{i}".encode()
            text = "hello" if lang == "en" else ARABIC
            (root / name).write_bytes(data)
            gt_bytes = json.dumps(gt_for(name, lines, text)).encode()
            (root / f"{token}-{i}.gt.json").write_bytes(gt_bytes)
            docs.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(),
                         "gt": f"{token}-{i}.gt.json",
                         "gt_sha256": hashlib.sha256(gt_bytes).hexdigest(), "condition": "clean",
                         "language": lang, "layout": "invoice", "unit": f"{token}-{i}"})
    (root / "manifest.json").write_text(json.dumps({"documents": docs}))
    return root


def calibrate(tmp_path, reader, *extra, name="out", work="work"):
    out = tmp_path / f"{name}.json"
    args = ["calibrate", "--dev", str(tmp_path / "dev"), "--work", str(tmp_path / work),
            "--out", str(out), *extra]
    return ra.main(args, reader_factory=lambda s, t: reader), out


def test_calibrate_setup_tau_is_highest_cell_tau(tmp_path, capsys):
    write_dev(tmp_path / "dev")
    code, out = calibrate(tmp_path, FakeReader(n=5))
    assert code == 0
    record = json.loads(out.read_text("utf-8"))
    cells = cells_of(record)
    assert cells["onnxtr-en-osd"]["cell_tau"] == pytest.approx(0.101)
    assert cells["onnxtr-en-vote"]["cell_tau"] == pytest.approx(0.151)
    assert cells["onnxtr-en-osd"]["setup_key"] == "KEY-osd"
    assert cells["onnxtr-en-vote"]["setup_key"] == "KEY-vote"
    assert cells["tesseract-en"]["cell_tau"] == pytest.approx(0.201)
    assert cells["tesseract-ar"]["cell_tau"] == pytest.approx(0.301)
    assert cells["tesseract-en"]["tau"] == pytest.approx(0.301)
    assert record["taus"] == {
        "KEY-osd": pytest.approx(0.101),
        "KEY-vote": pytest.approx(0.151),
        "KEY-tess": pytest.approx(0.301),
    }
    assert record["mode"] == "calibrate" and record["status"] == "finished"
    assert record["heldout_manifest_sha256"] is None
    assert record["dev_manifest_sha256"] == hashlib.sha256(
        (tmp_path / "dev" / "manifest.json").read_bytes()
    ).hexdigest()
    assert cells["tesseract-ar"]["status"] == "calibrated"
    assert cells["tesseract-ar"]["missed_error_rate"]["resamples"] == 2000
    assert cells["tesseract-ar"]["missed_error_rate"]["unit"] == "document"
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("onnxtr-en-osd: tau 0.101 (cell tau 0.101), dev bound 0.0 %, 200 bad")
    assert lines[-1].startswith("TAUS = {") and json.loads(lines[-1][7:])["KEY-tess"] == 0.301


def test_calibrate_dev_file_changed_is_refused(tmp_path, capsys):
    dev = write_dev(tmp_path / "dev")
    (dev / "en-0.png").write_bytes(b"changed")
    reader = FakeReader(n=5)
    code, _ = calibrate(tmp_path, reader)
    assert code == 1
    assert "en-0.png does not match the dev manifest" in capsys.readouterr().err
    assert reader.calls == []


def test_calibrate_ground_truth_changed_is_refused(tmp_path, capsys):
    dev = write_dev(tmp_path / "dev")
    (dev / "en-0.gt.json").write_text(json.dumps(gt_for("en-0.png", 5, "other")))
    reader = FakeReader(n=5)
    code, _ = calibrate(tmp_path, reader)
    err = capsys.readouterr().err
    assert code == 1 and err == "error: en-0.gt.json does not match the dev manifest\n"
    assert reader.calls == []


def test_calibrate_dev_document_without_file_names_the_field(tmp_path, capsys):
    dev = write_dev(tmp_path / "dev")
    manifest = json.loads((dev / "manifest.json").read_text())
    del manifest["documents"][0]["file"]
    (dev / "manifest.json").write_text(json.dumps(manifest))
    code, _ = calibrate(tmp_path, FakeReader(n=5))
    assert code == 1
    assert capsys.readouterr().err == "error: a dev manifest document is missing file\n"


def test_calibrate_cells_say_whether_the_tau_flags_every_line(tmp_path, capsys):
    write_dev(tmp_path / "dev")
    code, out = calibrate(tmp_path, FakeReader(n=5))
    cells = cells_of(json.loads(out.read_text("utf-8")))
    assert code == 0 and all(c["flags_every_line"] is True for c in cells.values())
    shown = capsys.readouterr().out.splitlines()[0]
    assert shown.endswith("200 bad lines (flags every line)")


def test_calibrate_a_document_read_twice_with_another_key_is_refused(tmp_path, capsys):
    write_dev(tmp_path / "dev")
    reader = FakeReader(n=5, wrong=_only("ar-0.png", 99))
    code, _ = calibrate(tmp_path, reader)
    err = capsys.readouterr().err
    assert code == 1 and err.count("\n") == 1 and err.startswith("error: ar-0.png was read twice")
    assert "OTHER" in err and "KEY-tess" in err
    again = FakeReader(n=5)
    assert calibrate(tmp_path, again)[0] == 0
    assert ("ar-0.png", "ar", True) in again.calls


def test_calibrate_second_run_uses_the_cache(tmp_path):
    write_dev(tmp_path / "dev")
    first = FakeReader(n=5)
    assert calibrate(tmp_path, first)[0] == 0
    assert first.calls
    second = FakeReader(n=5)
    assert calibrate(tmp_path, second, name="again")[0] == 0
    assert second.calls == []


def test_calibrate_jobs_do_not_change_output(tmp_path, capsys):
    write_dev(tmp_path / "dev")
    _, one = calibrate(tmp_path, FakeReader(n=5), "--jobs", "1", name="one", work="w1")
    out_one = capsys.readouterr().out
    _, three = calibrate(tmp_path, FakeReader(n=5), "--jobs", "3", name="three", work="w3")
    out_three = capsys.readouterr().out
    assert out_one == out_three

    def body(path):
        record = json.loads(path.read_text("utf-8"))
        host = {k: v for k, v in record["host"].items() if k != "jobs"}  # jobs is information
        kept = {k: v for k, v in record.items() if k not in ("started_at", "finished_at", "host")}
        return {**kept, "host": host}

    assert body(one) == body(three)


def test_calibrate_arabic_in_en_share(tmp_path):
    write_dev(tmp_path / "dev")
    code, out = calibrate(tmp_path, FakeReader(n=5), "--arabic-in-en")
    assert code == 0
    measure = json.loads(out.read_text("utf-8"))["measures"]["arabic_in_en"]
    both = {"documents": 40, "arabic_gt_lines": 40 * 5, "auto_accepted": 40 * 5, "share": 1.0}
    assert measure == {"vote": both, "osd": both, "share": 1.0}


def test_calibrate_arabic_in_en_reads_only_the_first_documents_asked_for(tmp_path):
    write_dev(tmp_path / "dev")
    reader = FakeReader(n=5)
    code, out = calibrate(tmp_path, reader, "--arabic-in-en", "--arabic-in-en-docs", "7")
    assert code == 0
    measure = json.loads(out.read_text("utf-8"))["measures"]["arabic_in_en"]
    for setup in ("vote", "osd"):
        assert measure[setup]["documents"] == 7 and measure[setup]["arabic_gt_lines"] == 7 * 5


def _record(path):
    return json.loads(path.read_text("utf-8"))


def test_calibrate_stops_at_minimums_and_does_not_read_past_it(tmp_path):
    write_dev(tmp_path / "dev", per_language=30)
    reader = FakeReader(n=5)
    code, out = calibrate(tmp_path, reader, "--jobs", "1")
    assert code == 0
    cells = cells_of(_record(out))
    for cell in ("onnxtr-en-osd", "onnxtr-en-vote", "tesseract-en", "tesseract-ar"):
        assert cells[cell]["documents_used"] == 40 and cells[cell]["stopped_by"] == "minimums"
        assert cells[cell]["documents"] == 40 and cells[cell]["bad_lines"] == 200
    names = {call[0] for call in reader.calls}
    assert "en-39.png" in names and "en-40.png" not in names
    assert "ar-29.png" in names and "mixed-9.png" in names and "mixed-10.png" not in names
    assert len(reader.calls) == 40 * 3 + 40


def test_calibrate_jobs_read_ahead_but_count_only_up_to_the_stop(tmp_path):
    write_dev(tmp_path / "dev", per_language=30)
    _, out = calibrate(tmp_path, FakeReader(n=5), "--jobs", "7")
    cells = cells_of(_record(out))
    assert cells["onnxtr-en-vote"]["documents_used"] == 40
    assert cells["onnxtr-en-osd"]["documents"] == 40
    assert cells["tesseract-ar"]["documents_used"] == 40


def test_calibrate_cap_reads_not_measured(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(accuracy_stats, "DEV_CAP_DOCS", 12)
    write_dev(tmp_path / "dev", per_language=10)
    _, out = calibrate(tmp_path, FakeReader(n=5))
    cell = cells_of(_record(out))["onnxtr-en-osd"]
    assert cell["documents_used"] == 12 and cell["stopped_by"] == "cap"
    assert cell["status"] == "not measured" and cell["reason"] == "not measured (60 bad lines)"
    assert "onnxtr-en-osd: not measured (60 bad lines)" in capsys.readouterr().out


def test_calibrate_dev_set_end_reads_not_measured(tmp_path):
    write_dev(tmp_path / "dev", per_language=5)
    _, out = calibrate(tmp_path, FakeReader(n=5))
    cell = cells_of(_record(out))["onnxtr-en-osd"]
    assert cell["documents_used"] == 10 and cell["stopped_by"] == "dev set end"
    assert cell["reason"] == "not measured (50 bad lines)"


def test_stop_reason_needs_both_minimums():
    assert ad.stop_reason(199, 60, 60) is None
    assert ad.stop_reason(250, 39, 39) is None
    assert ad.stop_reason(200, 40, 40) == "minimums"
    assert ad.stop_reason(250, 39, 300) == "cap"
    assert ad.stop_reason(200, 40, 300) == "minimums"
    assert ad.stop_reason(199, 60, 299) is None


def test_calibrate_reports_split_word_share(tmp_path, capsys):
    write_dev(tmp_path / "dev", per_language=5)
    split = ((10.0, 5.0, 20.0, 10.0),)  # the centre lies in the first line of every page
    _, out = calibrate(tmp_path, FakeReader(n=5, split_boxes=split))
    record = _record(out)
    assert cells_of(record)["onnxtr-en-osd"]["split_word_lines"] == 10
    assert record["split_word_share_en"] == pytest.approx(10 / 50)
    assert "onnxtr-en split-word lines: 20.0 %\n" in capsys.readouterr().out


# ---------------------------------------------------------------- fixtures

def test_fixtures_mode_writes_its_own_mode(tmp_path, capsys):
    out = tmp_path / "fixtures.json"
    reader = FakeReader(n=0)
    code = ra.main(["fixtures", "--work", str(tmp_path / "w"), "--out", str(out), "--jobs", "2"],
                   reader_factory=lambda s, t: reader)
    assert code == 0
    record = json.loads(out.read_text("utf-8"))
    assert record["mode"] == "fixtures" and record["status"] == "finished"
    cells = cells_of(record)
    assert cells["onnxtr-en-osd"]["documents"] == 2
    assert cells["onnxtr-en-vote"]["documents"] == 2
    assert cells["tesseract-ar"]["documents"] == 4
    assert len(reader.calls) == (2 + 2 + 2 + 4) * 3
    assert capsys.readouterr().out.count("\n") >= 3


def test_unexpected_failure_is_one_line(tmp_path, capsys):
    held = write_heldout(tmp_path, 1000, 1000)

    def boom(*_):
        raise KeyError("x")

    assert run(grade_args(tmp_path, held), boom) == 1
    assert capsys.readouterr().err == "error: unexpected failure (KeyError)\n"


def test_missing_tesseract_is_reported(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ra, "find_tesseract", lambda settings: None)
    held = write_heldout(tmp_path, 1000, 1000)
    assert run(grade_args(tmp_path, held), FakeReader()) == 1
    assert "Tesseract was not found; set FOLIO_TESSERACT_PATH" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()



def test_arabic_in_en_reads_with_and_without_tesseract_and_keeps_both_taus(tmp_path):
    write_dev(tmp_path / "dev")
    reader = FakeReader(n=5)
    code, out = calibrate(tmp_path, reader, "--arabic-in-en")
    assert code == 0
    arabic_in_en = [c for c in reader.calls if c[1] == "en" and not c[0].startswith("en-")]
    assert {c[2] for c in arabic_in_en} == {True, False}
    taus = _record(out)["taus"]
    assert taus["KEY-vote"] != taus["KEY-osd"]


def test_grade_reads_both_onnxtr_cells_into_separate_cache_files(tmp_path):
    held = write_heldout(tmp_path, 1000, 1000)
    reader = FakeReader()
    assert run(grade_args(tmp_path, held), reader) == 0
    en_reads = [c for c in reader.calls if c[0].startswith("en-") and c[1] == "en"]
    assert {c[2] for c in en_reads} == {True, False}
    assert len(en_reads) == 2 * 3 * 2
    cells = cells_of(results_file(tmp_path))
    assert cells["onnxtr-en-osd"]["setup_key"] == "KEY-osd"
    assert cells["onnxtr-en-vote"]["setup_key"] == "KEY-vote"
    osd_group = cells["onnxtr-en-osd"]["orientation"]["osd"]
    assert (osd_group["pages"], osd_group["wrong"], osd_group["flagged"]) == (6, 0, 0)
    assert cells["onnxtr-en-osd"]["orientation"]["vote-pair"]["pages"] == 0
    assert cells["onnxtr-en-vote"]["orientation"]["vote-pair"]["pages"] == 6
    assert "turn_wrong" not in cells["onnxtr-en-osd"]
    assert cells["onnxtr-en-vote"]["pages"] == 6


def test_the_results_record_the_reading_threads_as_information(tmp_path):
    write_dev(tmp_path / "dev")
    code, out = calibrate(tmp_path, FakeReader(n=5), "--jobs", "3")
    assert code == 0
    assert json.loads(out.read_text("utf-8"))["host"]["jobs"] == 3


def _arabic_measure(tmp_path, taus, limit=None):
    dev = write_dev(tmp_path / "dev")
    docs, _ = ra._load_dev(dev)
    ctx = ra.Context(FakeReader(n=5), Path("t"), dict(KEYS), tmp_path / "work", 1)
    return ra._arabic_in_en(ctx, dev, docs, taus, limit)


def test_arabic_in_en_measures_each_setup_with_its_own_tau_and_reports_the_higher(tmp_path):
    m = _arabic_measure(tmp_path, {"vote": 0.95, "osd": 0.5})
    assert m["vote"]["auto_accepted"] == 0 and m["vote"]["share"] == 0.0
    assert m["osd"]["auto_accepted"] == 200 and m["osd"]["share"] == 1.0
    assert m["share"] == 1.0


def test_arabic_in_en_share_is_the_vote_one_when_it_is_higher(tmp_path):
    m = _arabic_measure(tmp_path, {"vote": 0.5, "osd": 0.95})
    assert (m["vote"]["share"], m["osd"]["share"], m["share"]) == (1.0, 0.0, 1.0)


def test_arabic_in_en_without_one_tau_reports_the_other(tmp_path):
    m = _arabic_measure(tmp_path, {"vote": None, "osd": 0.5})
    assert m["vote"]["share"] is None and m["vote"]["reason"] == "no OnnxTR tau"
    assert m["vote"]["auto_accepted"] is None and m["share"] == 1.0


def test_arabic_in_en_share_is_none_when_both_are_none(tmp_path):
    m = _arabic_measure(tmp_path, {"vote": None, "osd": None})
    assert m["share"] is None and m["osd"]["reason"] == "no OnnxTR tau"


@pytest.mark.parametrize("bad", ["0", "-2", "x", "1.5"])
def test_arabic_in_en_docs_must_be_an_integer_of_at_least_one(tmp_path, capsys, bad):
    write_dev(tmp_path / "dev")
    reader = FakeReader(n=5)
    code, _ = calibrate(tmp_path, reader, "--arabic-in-en", "--arabic-in-en-docs", bad)
    err = capsys.readouterr().err
    assert code == 1 and err == "error: --arabic-in-en-docs must be an integer of at least 1\n"
    assert reader.calls == []


def test_calibrate_prints_both_arabic_in_en_shares(tmp_path, capsys):
    write_dev(tmp_path / "dev")
    code, _ = calibrate(tmp_path, FakeReader(n=5), "--arabic-in-en")
    assert code == 0
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("arabic in en")]
    assert line == ["arabic in en auto accepted: vote 100.0 %, osd 100.0 %"]


# ---------------------------------------------------------------- hashes and fields

def test_dev_read_hashes_the_bytes_it_uses(tmp_path):
    dev = write_dev(tmp_path / "dev")
    docs, _ = ra._load_dev(dev)
    ctx = ra.Context(FakeReader(n=5), Path("t"), dict(KEYS), tmp_path / "work", 1)
    (dev / docs[0]["file"]).write_bytes(b"changed after the manifest check")
    with pytest.raises(ValueError, match="en-0.png does not match the dev manifest"):
        ra._dev_read(ctx, dev, (docs[0], "onnxtr-en-vote", "en", False))
    (dev / docs[0]["file"]).write_bytes(b"dev|en|0")
    (dev / docs[0]["gt"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="en-0.gt.json does not match the dev manifest"):
        ra._dev_read(ctx, dev, (docs[0], "onnxtr-en-vote", "en", False))


@pytest.mark.parametrize("field", ["language", "unit"])
def test_calibrate_dev_document_without_language_or_unit_names_the_field(tmp_path, capsys, field):
    dev = write_dev(tmp_path / "dev")
    manifest = json.loads((dev / "manifest.json").read_text())
    del manifest["documents"][0][field]
    (dev / "manifest.json").write_text(json.dumps(manifest))
    code, _ = calibrate(tmp_path, FakeReader(n=5))
    assert code == 1
    assert capsys.readouterr().err == f"error: a dev manifest document is missing {field}\n"


def _fixtures_copy(tmp_path, monkeypatch):
    import shutil

    copy = tmp_path / "fixtures"
    shutil.copytree(ra.FIXTURES, copy, ignore=shutil.ignore_patterns("fonts"))
    monkeypatch.setattr(ra, "FIXTURES", copy)
    return copy


def test_fixtures_mode_refuses_a_ground_truth_file_that_differs_from_the_manifest(
    tmp_path, capsys, monkeypatch
):
    copy = _fixtures_copy(tmp_path, monkeypatch)
    token, _lang, layout = heldout.DOCUMENTS[0]
    target = copy / f"{token}-{layout}-clean.gt.json"
    target.write_bytes(target.read_bytes() + b" ")
    code = ra.main(["fixtures", "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o.json")],
                   reader_factory=lambda s, t: FakeReader(n=0))
    err = capsys.readouterr().err
    assert code == 1 and err == f"error: {target.name} does not match the fixture manifest\n"


def test_cells_carry_the_orientation_margins_and_floor(tmp_path):
    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), FakeReader()) == 0
    for cell in results_file(tmp_path)["cells"]:
        assert cell["uncertain_margins"] == {"pair": 0.13, "four": 0.41}
        assert cell["osd_floor"] == 1.5


def test_calibrated_cells_carry_the_orientation_margins_and_floor(tmp_path):
    write_dev(tmp_path / "dev")
    code, out = calibrate(tmp_path, FakeReader(n=5))
    assert code == 0
    for cell in json.loads(out.read_text("utf-8"))["cells"]:
        assert cell["uncertain_margins"] == {"pair": 0.13, "four": 0.41}
        assert cell["osd_floor"] == 1.5


def test_calibrate_creates_the_results_folder_before_reading(tmp_path):
    write_dev(tmp_path / "dev")
    reader = FakeReader(n=5)
    out = tmp_path / "eval" / "results" / "calibrate.json"
    assert not out.parent.exists()
    args = ["calibrate", "--dev", str(tmp_path / "dev"), "--work", str(tmp_path / "work"),
            "--out", str(out)]
    assert ra.main(args, reader_factory=lambda s, t: reader) == 0 and out.exists()


def test_calibrate_with_an_unusable_results_folder_reads_nothing(tmp_path, capsys):
    write_dev(tmp_path / "dev")
    (tmp_path / "taken").write_text("a file, not a folder", encoding="utf-8")
    reader = FakeReader(n=5)
    args = ["calibrate", "--dev", str(tmp_path / "dev"), "--work", str(tmp_path / "work"),
            "--out", str(tmp_path / "taken" / "calibrate.json")]
    code = ra.main(args, reader_factory=lambda s, t: reader)
    err = capsys.readouterr().err
    assert code == 1 and reader.calls == [] and err.startswith("error: ")
    assert len(err.strip().splitlines()) == 1
