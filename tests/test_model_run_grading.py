"""Tests for the calibrate, measure, grade and fixtures commands."""

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(_P(__file__).resolve().parent))

import json
from pathlib import Path
from types import SimpleNamespace

import accuracy_reads as ar
import heldout
import model_run_grading as grading
import model_run_lines as ml
import model_run_reads as mr
import model_run_vram as vram
import pytest
import run_accuracy
from model_run_support import KEYS, builder, make_session, run, sha_of, step_file, tree

from folio_ocr import calibration, pipeline
from folio_ocr.model_reply import ORDER_RULE

REAL_VRAM = vram.VramMonitor
SHA = "ab" * 32
TAUS = {"key-ar": 0.8}
GRADE_KEY = "heldout_manifest_sha256"


class FakeVram:
    """Stands in for the nvidia-smi logger: writes a small memory log."""

    def __init__(self, log, command):
        self.log = Path(log)
        self.command = command

    def __enter__(self):
        self.log.write_text("memory.used [MiB]\n1000 MiB\n2048 MiB\n", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def heldout_ok(monkeypatch):
    monkeypatch.setattr(heldout, "load_manifest", lambda p: ({"seeds": {}}, SHA))
    monkeypatch.setattr(heldout, "verify_generator", lambda m: None)
    monkeypatch.setattr(vram, "VramMonitor", FakeVram)
    monkeypatch.setattr(mr, "load_dev", lambda dev: ([], "devsha"))


def order_file(tmp_path, **patch):
    body = {"verdict": "frozen", "order_rule": ORDER_RULE, "keys": KEYS,
            "dev_manifest_sha256": "devsha", **patch}
    return step_file(tmp_path / "o.json", "order", **body)


def drift_file(tmp_path, **patch):
    body = {"keys": KEYS, "tolerance": 0.05, **patch}
    return step_file(tmp_path / "d.json", "drift", **body)


def calibrate_stub(calls, taus=TAUS, rc=0):
    def main(argv, reader_factory=None, plan=None):
        calls.append((argv, reader_factory, plan))
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(json.dumps({
            "taus": taus, "dev_manifest_sha256": "devsha",
            "cells": [{"cell": "model-en", "documents_used": 2},
                      {"cell": "model-ar", "documents_used": 1}],
            "pages": {"count": 3}}), encoding="utf-8")
        return rc

    return main


def calibrate_argv(tmp_path, **over):
    opts = {"--work": tmp_path / "w", "--dev": tmp_path / "dev", "--order": tmp_path / "o.json",
            "--drift": tmp_path / "d.json", "--heldout": tmp_path / "h.json",
            "--heldout-sha256": SHA, "--out": tmp_path / "cal.json", "--record": tmp_path / "r.json",
            "--results-dir": tmp_path / "res", "--nvidia-smi": "smi", **over}
    return ["calibrate", *[x for kv in opts.items() for x in kv]]


def test_calibrate_runs_the_accuracy_tool_and_records(tmp_path, capsys, monkeypatch, heldout_ok):
    order_file(tmp_path), drift_file(tmp_path)
    calls = []
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub(calls))
    monkeypatch.setattr(calibration, "TAUS", {"other": 0.5})
    session = make_session()
    rc, out, err = run(capsys, [*calibrate_argv(tmp_path), "--jobs", "2"], session)
    assert rc == 0, err
    argv, factory, plan = calls[0]
    assert argv[:2] == ["calibrate", "--dev"] and argv[-2:] == ["--jobs", "2"]
    assert factory is session.reader_factory and dict(plan.keys(None)) == KEYS
    assert f"MODEL TAUS = {json.dumps(TAUS, sort_keys=True)}" in out
    rec = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rec["step"] == "calibrate" and rec["keys"] == KEYS and rec["taus"] == TAUS
    assert rec["calibration"] == {"path": str(tmp_path / "cal.json"),
                                  "sha256": sha_of(tmp_path / "cal.json")}
    assert rec["heldout_fixed"] == {"path": str(tmp_path / "h.json"), "sha256": SHA}
    assert GRADE_KEY not in rec
    assert rec["dev_manifest_sha256"] == "devsha" and rec["drift_tolerance"] == 0.05
    assert rec["peak_vram_mib"] == 2048
    assert rec["tau_table_sha256_at_calibration"] == calibration.table_sha256(calibration.TAUS)


def test_calibrate_with_no_taus_stops_but_records(tmp_path, capsys, monkeypatch, heldout_ok):
    order_file(tmp_path), drift_file(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub([], taus={}))
    rc, out, _ = run(capsys, calibrate_argv(tmp_path))
    assert rc == 1 and "STOP: no model cell was calibrated; do not grade" in out
    assert (tmp_path / "r.json").exists()


def test_calibrate_passes_a_failing_exit_code_on(tmp_path, capsys, monkeypatch, heldout_ok):
    order_file(tmp_path), drift_file(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub([], rc=3))
    rc, _, _ = run(capsys, calibrate_argv(tmp_path))
    assert rc == 3 and not (tmp_path / "r.json").exists()


@pytest.mark.parametrize("case", ["no-order", "order-stop", "order-keys", "order-sha", "no-drift",
                                  "drift-keys", "heldout-sha", "graded"])
def test_calibrate_refusals_write_nothing(tmp_path, capsys, monkeypatch, heldout_ok, case):
    calls = []
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub(calls))
    patch = {"order-stop": {"verdict": "stop"}, "order-keys": {"keys": {"model-en": "x"}},
             "order-sha": {"dev_manifest_sha256": "zz"}}.get(case, {})
    if case != "no-order":
        order_file(tmp_path, **patch)
    if case != "no-drift":
        drift_file(tmp_path, **({"keys": {"model-en": "x"}} if case == "drift-keys" else {}))
    argv = calibrate_argv(tmp_path)
    if case == "heldout-sha":
        argv = calibrate_argv(tmp_path, **{"--heldout-sha256": "cd" * 32})
    if case == "graded":
        monkeypatch.setattr(heldout, "graded_attempts", lambda rd, sha: (Path("x.json"),))
    before = tree(tmp_path)
    rc, _, err = run(capsys, argv)
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert calls == [] and tree(tmp_path) == before
    if case == "graded":
        assert "already has a grade; model calibration must come before it" in err
    if case == "heldout-sha":
        assert "the held-out manifest differs from its recorded sha256; stop" in err


def test_calibrate_accepts_an_upper_case_sha(tmp_path, capsys, monkeypatch, heldout_ok):
    order_file(tmp_path), drift_file(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub([]))
    rc, _, err = run(capsys, calibrate_argv(tmp_path, **{"--heldout-sha256": SHA.upper()}))
    assert rc == 0, err


def test_calibrate_with_nvidia_smi_missing(tmp_path, capsys, monkeypatch, heldout_ok):
    monkeypatch.setattr(vram, "VramMonitor", REAL_VRAM)
    order_file(tmp_path), drift_file(tmp_path)
    calls = []
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub(calls))
    (tmp_path / "w").mkdir()
    rc, _, err = run(capsys, calibrate_argv(tmp_path, **{"--nvidia-smi": str(tmp_path / "no-smi.exe")}))
    assert rc == 1 and err.startswith("error: nvidia-smi was not found") and err.count("\n") == 1
    assert calls == [] and not (tmp_path / "r.json").exists()


def test_calibrate_with_no_vram_samples_still_records(tmp_path, capsys, monkeypatch, heldout_ok):
    order_file(tmp_path), drift_file(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", calibrate_stub([]))
    monkeypatch.setattr(FakeVram, "__enter__", lambda self: self.log.write_text("") or self)
    rc, out, err = run(capsys, calibrate_argv(tmp_path))
    rec = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 1 and err == "error: nvidia-smi gave no memory samples\n"
    assert rec["peak_vram_mib"] is None and rec["vram_error"] and rec["taus"] == TAUS
    assert "MODEL TAUS" in out


def changing_stub(change):
    inner = calibrate_stub([])

    def main(argv, reader_factory=None, plan=None):
        change(argv)
        return inner(argv, reader_factory, plan)

    return main


@pytest.mark.parametrize("case", ["heldout-changed", "graded-meanwhile", "dev-changed"])
def test_calibrate_records_nothing_when_the_state_changes_during_the_run(
        tmp_path, capsys, monkeypatch, heldout_ok, case):
    order_file(tmp_path), drift_file(tmp_path)

    def change(argv):
        if case == "heldout-changed":
            monkeypatch.setattr(heldout, "load_manifest", lambda p: ({"seeds": {}}, "cd" * 32))
        if case == "graded-meanwhile":
            monkeypatch.setattr(heldout, "graded_attempts", lambda d, s: (Path("x.json"),))

    stub = changing_stub(change)
    if case == "dev-changed":
        def main(argv, reader_factory=None, plan=None):
            rc = stub(argv, reader_factory, plan)
            out = Path(argv[argv.index("--out") + 1])
            out.write_text(json.dumps({"taus": TAUS, "dev_manifest_sha256": "otherdev"}),
                           encoding="utf-8")
            return rc
    else:
        main = stub
    monkeypatch.setattr(run_accuracy, "main", main)
    rc, _, err = run(capsys, calibrate_argv(tmp_path))
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    expected = {"heldout-changed": "differs from its recorded sha256",
                "graded-meanwhile": "already has a grade",
                "dev-changed": "dev manifest changed during the calibration"}[case]
    assert expected in err and not (tmp_path / "r.json").exists()


@pytest.mark.parametrize("written", [None, "{bad", "[]"])
def test_calibrate_with_a_missing_or_damaged_output_records_nothing(
        tmp_path, capsys, monkeypatch, heldout_ok, written):
    order_file(tmp_path), drift_file(tmp_path)

    def main(argv, reader_factory=None, plan=None):
        if written is not None:
            (tmp_path / "cal.json").write_text(written, encoding="utf-8")
        return 0

    monkeypatch.setattr(run_accuracy, "main", main)
    rc, _, err = run(capsys, calibrate_argv(tmp_path))
    assert rc == 1 and err == "error: the calibration file is missing or damaged: cal.json\n"
    assert not (tmp_path / "r.json").exists()


# ---------------------------------------------------------------- measure


def model_record(cell, **over):
    base = {"document": "a.png", "unit": "u", "cell": cell, "script_group": cell[-2:],
            "gt": "abc", "classes": [], "matched": True, "reader": "model", "text": "abc",
            "flags": [], "confidence": 0.9, "status": "read", "entries": None,
            "finish_reason": "stop", "max_tokens": 10, "completion_tokens": 3,
            **ml.line_measures("abc", "abc", frozenset())}
    return {**base, **over}


def calibrate_record(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({
        "taus": TAUS, "cells": [{"cell": "model-en", "documents_used": 1},
                                {"cell": "model-ar", "documents_used": 2}]}), encoding="utf-8")
    return step_file(tmp_path / "r.json", "calibrate", keys=KEYS, taus=TAUS, drift_tolerance=0.05,
                     dev_manifest_sha256="devsha",
                     calibration={"path": str(cal), "sha256": sha_of(cal)})


def measure_argv(tmp_path):
    return ["measure", "--work", tmp_path / "w", "--dev", tmp_path / "dev", "--record",
            tmp_path / "r.json", "--out", tmp_path / "m.json"]


def measure_setup(monkeypatch):
    docs = [{"file": f"{lang}{i}.png", "language": lang, "gt": f"{lang}{i}.gt"}
            for lang in ("en", "ar") for i in range(3)]
    monkeypatch.setattr(mr, "load_dev", lambda dev: (docs, "devsha"))
    read = lambda work, keys, reader, dev, doc: SimpleNamespace(
        doc=doc, gt={"lines": [{"text": doc["file"]}]}, exchanges=())
    monkeypatch.setattr(mr, "read_dev_doc", read)
    seen = {}

    def records(r, cell, vocab):
        seen.setdefault("vocab", vocab)
        return [model_record(cell, document=r.doc["file"])]

    monkeypatch.setattr(mr, "line_records", records)
    monkeypatch.setattr(mr, "tesseract_compare", lambda reads, pages: {"documents": len(pages),
                        "model": {"cer": 0.1}, "tesseract": {"cer": 0.2}})
    monkeypatch.setattr(ar, "pipeline_reader", lambda settings, tess: "tess-reader")
    monkeypatch.setattr(ar, "expected_keys", lambda tess: {"tesseract-ar": "tkey"})
    tess_calls = []

    def cached(work, key, sha, data, name, mode, with_tess, reader):
        tess_calls.append((key, mode, with_tess, reader))
        return {"obj": name}

    monkeypatch.setattr(ar, "cached_read", cached)
    monkeypatch.setattr(ar, "pages_from_cache", lambda obj: (obj["obj"],))
    return seen, tess_calls


def dev_dir(tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    for name in ("en0.png", "en1.png", "en2.png", "ar0.png", "ar1.png", "ar2.png"):
        (dev / name).write_bytes(name.encode())
    return dev


def test_measure_uses_the_documents_the_calibration_used(tmp_path, capsys, monkeypatch):
    dev_dir(tmp_path)
    calibrate_record(tmp_path)
    seen, tess_calls = measure_setup(monkeypatch)
    rc, out, err = run(capsys, measure_argv(tmp_path))
    assert rc == 0, err
    rec = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert rec["step"] == "measure" and rec["keys"] == KEYS
    assert [r["document"] for r in rec["dev_lines"]] == ["en0.png", "ar0.png", "ar1.png"]
    assert seen["vocab"] == mr.vocabulary_of([{"lines": [{"text": n}]} for n in
                                              ("en0.png", "ar0.png", "ar1.png")])
    assert {"measures", "order_after_calibration", "aggregation", "pins", "tesseract_compare",
            "dev_lines"} <= set(rec)
    assert set(rec["pins"]) == {"model-en", "model-ar"} and rec["pins"]["model-ar"]["tau"] == 0.8
    assert rec["pins"]["model-en"]["tau"] is None
    assert [c[:3] for c in tess_calls] == [("tkey", "ar", True)] * 2 and tess_calls[0][3] == "tess-reader"
    assert "ar: cer" in out and "tesseract" in out
    assert not any(ord(ch) > 127 for ch in out)


def test_measure_stops_when_another_aggregation_is_better(tmp_path, capsys, monkeypatch):
    dev_dir(tmp_path)
    calibrate_record(tmp_path)
    measure_setup(monkeypatch)
    monkeypatch.setattr(mr, "aggregation_check", lambda records: {
        "lines": 1, "good": 1, "bad": 1, "exp_min": 0.5, "exp_mean": 0.9, "exp_sum": 0.8,
        "best_other_margin": 0.4, "stop": True})
    rc, out, _ = run(capsys, measure_argv(tmp_path))
    assert rc == 1 and "STOP: another logprob aggregation separates good and bad dev lines" in out
    assert (tmp_path / "m.json").exists()


@pytest.mark.parametrize("case", ["no-record", "calibration-changed", "keys"])
def test_measure_refusals_write_nothing(tmp_path, capsys, monkeypatch, case):
    dev_dir(tmp_path)
    measure_setup(monkeypatch)
    if case != "no-record":
        record = calibrate_record(tmp_path)
        if case == "calibration-changed":
            (tmp_path / "cal.json").write_text("{}", encoding="utf-8")
        if case == "keys":
            body = json.loads(record.read_text(encoding="utf-8"))
            record.write_text(json.dumps({**body, "keys": {"model-en": "x"}}), encoding="utf-8")
    before = tree(tmp_path)
    rc, _, err = run(capsys, measure_argv(tmp_path))
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert tree(tmp_path) == before


def test_measure_refuses_a_changed_dev_set_before_building(tmp_path, capsys, monkeypatch):
    import model_run

    dev_dir(tmp_path)
    measure_setup(monkeypatch)
    calibrate_record(tmp_path)
    monkeypatch.setattr(mr, "load_dev", lambda dev: ([], "another"))
    before = tree(tmp_path)
    build = builder(make_session())
    rc = model_run.main([str(a) for a in measure_argv(tmp_path)], build)
    err = capsys.readouterr().err
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert "dev set changed" in err and build.calls == [] and tree(tmp_path) == before


# ---------------------------------------------------------------- grade


def grade_record(tmp_path, taus=TAUS, **patch):
    body = {"keys": KEYS, "taus": taus, "heldout_fixed": {"path": "h.json", "sha256": SHA}, **patch}
    return step_file(tmp_path / "r.json", "calibrate", **body)


def grade_argv(tmp_path, sha=SHA):
    return ["grade", "--work", tmp_path / "w", "--heldout", tmp_path / "h.json", "--heldout-sha256",
            sha, "--record", tmp_path / "r.json", "--results-dir", tmp_path / "res", "--out",
            tmp_path / "g.json", "--nvidia-smi", "smi"]


def grade_stub(calls, rc=0, write=True):
    def main(argv, reader_factory=None, plan=None):
        calls.append(argv)
        res = Path(argv[argv.index("--results-dir") + 1])
        res.mkdir(exist_ok=True)
        if write:
            for name in ("g1.json", "g2.json"):
                (res / name).write_text(json.dumps({GRADE_KEY: SHA, "cells": [name]}),
                                        encoding="utf-8")
        return rc

    return main


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(calibration, "TAUS", dict(TAUS))


def test_grade_records_the_newest_attempt(tmp_path, capsys, monkeypatch, heldout_ok, table):
    grade_record(tmp_path)
    calls = []
    monkeypatch.setattr(run_accuracy, "main", grade_stub(calls))
    rc, _, err = run(capsys, [*grade_argv(tmp_path), "--jobs", "3"])
    assert rc == 0, err
    assert calls[0][-2:] == ["--jobs", "3"]
    rec = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert rec["step"] == "grade" and rec["status"] == "finished" and rec["exit_code"] == 0
    assert rec["grade"] == {"path": str(tmp_path / "res" / "g2.json"),
                            "sha256": sha_of(tmp_path / "res" / "g2.json")}
    assert rec["peak_vram_mib"] == 2048 and rec["keys"] == KEYS and GRADE_KEY not in rec


def test_grade_failure_is_recorded_as_failed(tmp_path, capsys, monkeypatch, heldout_ok, table):
    grade_record(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", grade_stub([], rc=2))
    rc, _, _ = run(capsys, grade_argv(tmp_path))
    rec = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert rc == 2 and rec["status"] == "failed" and rec["exit_code"] == 2


def test_grade_with_no_vram_samples_still_writes_the_step(
        tmp_path, capsys, monkeypatch, heldout_ok, table):
    grade_record(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", grade_stub([]))
    monkeypatch.setattr(FakeVram, "__enter__", lambda self: self.log.write_text("") or self)
    rc, _, err = run(capsys, grade_argv(tmp_path))
    rec = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert rc == 1 and err.count("\n") == 1 and err.startswith("error: ")
    assert rec["peak_vram_mib"] is None and rec["vram_error"]
    assert rec["grade"]["path"].endswith("g2.json") and rec["exit_code"] == 0


def test_grade_that_exits_zero_without_a_grade_file_is_recorded_as_failed(
        tmp_path, capsys, monkeypatch, heldout_ok, table):
    grade_record(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", grade_stub([], rc=0, write=False))
    rc, _, err = run(capsys, grade_argv(tmp_path))
    rec = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert rec["grade"] is None and rec["grade_error"] and rec["status"] == "failed"
    assert rec["exit_code"] == 0 and not list((tmp_path / "res").glob("*.json"))


def test_grade_with_an_unhashable_attempt_still_writes_the_step(
        tmp_path, capsys, monkeypatch, heldout_ok, table):
    grade_record(tmp_path)
    monkeypatch.setattr(run_accuracy, "main", grade_stub([]))

    def broken(path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(grading.common, "file_sha256", broken)
    rc, _, err = run(capsys, grade_argv(tmp_path))
    rec = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert rc == 1 and err.count("\n") == 1 and err.startswith("error: ")
    assert rec["grade"] is None and rec["grade_error"] and rec["status"] == "failed"
    assert rec["peak_vram_mib"] == 2048


@pytest.mark.parametrize("case", ["no-record", "sha-arg", "sha-record", "keys", "no-taus", "table",
                                  "graded"])
def test_grade_refusals_read_and_write_nothing(tmp_path, capsys, monkeypatch, heldout_ok, table, case):
    calls = []
    monkeypatch.setattr(run_accuracy, "main", grade_stub(calls))
    patch = {"sha-record": {"heldout_fixed": {"path": "h", "sha256": "cd" * 32}},
             "keys": {"keys": {"model-en": "x"}}}.get(case, {})
    if case != "no-record":
        grade_record(tmp_path, taus={} if case == "no-taus" else TAUS, **patch)
    if case == "table":
        monkeypatch.setattr(calibration, "TAUS", {})
    if case == "graded":
        monkeypatch.setattr(heldout, "graded_attempts", lambda rd, sha: (Path("x.json"),))
    argv = grade_argv(tmp_path, "cd" * 32 if case == "sha-arg" else SHA)
    before = tree(tmp_path)
    rc, _, err = run(capsys, argv)
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert calls == [] and tree(tmp_path) == before
    expected = {"no-taus": "there is nothing to grade", "table": "fill them in first",
                "graded": heldout.ALREADY_GRADED, "sha-arg": "differs from its recorded"}
    if case in expected:
        assert expected[case] in err


# ---------------------------------------------------------------- fixtures


def fixture_dir(tmp_path, monkeypatch, files=("en-a.pdf", "mixed-b.pdf", "ar-c.pdf"), bad=None):
    import hashlib

    fx = tmp_path / "fx"
    fx.mkdir()
    entries = []
    for name in files:
        data = name.encode()
        (fx / name).write_bytes(data)
        (fx / (Path(name).stem + ".gt.json")).write_text(json.dumps({"lines": []}), encoding="utf-8")
        sha = hashlib.sha256(b"other" if name == bad else data).hexdigest()
        entries.append({"file": name, "sha256": sha})
    (fx / "manifest.json").write_text(json.dumps({"fixtures": entries[::-1]}), encoding="utf-8")
    monkeypatch.setattr(grading, "FIXTURES", fx)


def fixtures_argv(tmp_path, smi="smi"):
    return ["fixtures", "--work", tmp_path / "w", "--out", tmp_path / "fx.json", "--nvidia-smi", smi]


def test_fixtures_reads_each_file_with_its_languages(tmp_path, capsys, monkeypatch, heldout_ok):
    fixture_dir(tmp_path, monkeypatch)
    seen = {}

    def read_bytes(data, name, **kw):
        seen[name] = kw
        pages = (SimpleNamespace(origin="ocr"),) * (2 if name == "ar-c.pdf" else 1)
        return SimpleNamespace(pages=pages)

    monkeypatch.setattr(pipeline, "read_bytes", read_bytes)
    monkeypatch.setattr(mr, "fixture_cer", lambda gt, page: {"gt_lines": 1, "cer": 0.25})
    session = make_session()
    rc, out, err = run(capsys, [*fixtures_argv(tmp_path), "--jobs", "2"], session)
    assert rc == 0, err
    assert {n: kw["languages"] for n, kw in seen.items()} == {
        "en-a.pdf": "en", "mixed-b.pdf": "en+ar", "ar-c.pdf": "ar"}
    assert seen["en-a.pdf"]["model"] is session.model
    rec = json.loads((tmp_path / "fx.json").read_text(encoding="utf-8"))
    assert list(rec["fixtures"]) == ["ar-c.pdf", "en-a.pdf", "mixed-b.pdf"]
    ok = rec["fixtures"]["en-a.pdf"]
    assert ok["cer"] == 0.25 and ok["languages"] == "en" and ok["origin"] == "ocr" and ok["pages"] == 1
    bad = rec["fixtures"]["ar-c.pdf"]
    assert bad["cer"] is None and bad["error"] == "read as 2 pages"
    assert rec["pages"]["count"] == 3 and rec["peak_vram_mib"] == 2048
    assert "en-a.pdf: cer 25.0 %" in out and "ar-c.pdf: cer n/a" in out


def test_fixtures_with_no_vram_samples_still_writes_the_step(
        tmp_path, capsys, monkeypatch, heldout_ok):
    fixture_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "read_bytes",
                        lambda data, name, **kw: SimpleNamespace(pages=(SimpleNamespace(origin="ocr"),)))
    monkeypatch.setattr(mr, "fixture_cer", lambda gt, page: {"gt_lines": 1, "cer": 0.25})
    monkeypatch.setattr(FakeVram, "__enter__", lambda self: self.log.write_text("") or self)
    rc, out, err = run(capsys, fixtures_argv(tmp_path))
    rec = json.loads((tmp_path / "fx.json").read_text(encoding="utf-8"))
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert rec["peak_vram_mib"] is None and rec["vram_error"] and "en-a.pdf: cer" in out


def test_fixtures_sha_mismatch_is_a_one_line_error(tmp_path, capsys, monkeypatch, heldout_ok):
    fixture_dir(tmp_path, monkeypatch, bad="mixed-b.pdf")
    rc, _, err = run(capsys, fixtures_argv(tmp_path))
    assert rc == 1 and err == "error: mixed-b.pdf does not match the fixture manifest\n"
    assert not (tmp_path / "fx.json").exists()


def test_fixtures_with_nvidia_smi_missing(tmp_path, capsys, monkeypatch):
    fixture_dir(tmp_path, monkeypatch)
    (tmp_path / "w").mkdir()
    rc, _, err = run(capsys, fixtures_argv(tmp_path, str(tmp_path / "no-smi.exe")))
    assert rc == 1 and err.startswith("error: nvidia-smi was not found") and err.count("\n") == 1
    assert not (tmp_path / "fx.json").exists()


LOCAL_VARIABLES = ("FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER", "FOLIO_TAU_FILE")


@pytest.mark.parametrize("name", LOCAL_VARIABLES)
def test_grade_refuses_a_local_setup(tmp_path, capsys, monkeypatch, heldout_ok, table, name):
    calls = []
    monkeypatch.setattr(run_accuracy, "main", grade_stub(calls))
    monkeypatch.setenv(name, "")
    grade_record(tmp_path)
    before = tree(tmp_path)
    rc, _, err = run(capsys, grade_argv(tmp_path))
    assert rc == 1 and err.count("\n") == 1 and name in err and "never graded" in err
    assert calls == [] and tree(tmp_path) == before


@pytest.mark.parametrize("name", LOCAL_VARIABLES)
def test_package_calibration_refuses_a_local_setup(tmp_path, capsys, monkeypatch, heldout_ok, name):
    monkeypatch.setenv(name, "x")
    rc, _, err = run(capsys, calibrate_argv(tmp_path))
    assert rc == 1 and err.count("\n") == 1 and name in err
