"""Tests for the report command that joins every step into one results file."""

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(_P(__file__).resolve().parent))

import json

import pytest
from model_run_support import FACTS, KEYS, run, sha_of, step_file, tree

from folio_ocr import calibration, model_request

SHA = "ab" * 32

EXPECTED_KEYS = {
    "schema", "mode", "status", "started_at", "finished_at", "host", "versions", "ollama_version",
    "model_digest", "prompt_sha256", "model", "cells", "calibration_cells", "taus",
    "tau_table_sha256", "tau_table_sha256_at_calibration", "heldout_manifest_sha256",
    "graded_heldout_manifest_sha256",
    "dev_manifest_sha256", "cer_per_fixture", "pages", "pages_by_step", "peak_vram_mib",
    "vram_by_step", "vram_errors", "contract", "order", "drift", "pins", "measures", "tesseract_compare",
    "aggregation", "order_after_calibration", "dev_lines", "inputs", "label",
}


def steps(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"cells": [{"cell": "model-ar"}], "pages": {"count": 4},
                               "dev_manifest_sha256": "devsha"}), encoding="utf-8")
    grade = tmp_path / "grade.json"
    grade.write_text(json.dumps({
        "heldout_manifest_sha256": SHA, "cells": [{"cell": "model-en"}], "pages": {"count": 9},
        "host": {"python": "3.12"}, "versions": {"tesseract": "5.3"}}), encoding="utf-8")
    paths = {
        "contract": step_file(tmp_path / "c.json", "contract", checks=[{"name": "x"}], passed=True,
                              facts=dict(FACTS)),
        "order": step_file(tmp_path / "o.json", "order", keys=KEYS, classes={"a": {}},
                           verdict="frozen", order_rule="rule", documents=3, stopped_by="minimums",
                           model=FACTS),
        "drift": step_file(tmp_path / "d.json", "drift", keys=KEYS, tolerance=0.05,
                           max_spread_by_cell={"model-ar": 0.05}, text_changes=1,
                           rounds=["fresh"], repeats=3, lines=[{}, {}]),
        "record": step_file(tmp_path / "r.json", "calibrate", keys=KEYS, model=FACTS,
                            taus={"key-ar": 0.8}, peak_vram_mib=5000,
                            calibration={"path": str(cal), "sha256": sha_of(cal)},
                            tau_table_sha256_at_calibration="oldsha"),
        "measure": step_file(tmp_path / "m.json", "measure", keys=KEYS, measures={"ar": {}},
                             pins={"model-ar": {}}, tesseract_compare={"documents": 1},
                             aggregation={"stop": False}, order_after_calibration={"verdict": "frozen"},
                             dev_lines=[{"gt": "x"}]),
        "grade_record": step_file(tmp_path / "g.json", "grade", keys=KEYS, peak_vram_mib=7000,
                                  grade={"path": str(grade), "sha256": sha_of(grade)}, exit_code=0),
        "fixtures": step_file(tmp_path / "fx.json", "fixtures", keys=KEYS, peak_vram_mib=6000,
                              pages={"count": 2}, fixtures={"en-a.pdf": {"cer": 0.1},
                                                            "ar-b.pdf": {"cer": None}}),
    }
    return {"cal": cal, "grade": grade, **paths}


def report_argv(tmp_path, paths, *extra):
    return ["report", "--work", tmp_path / "w", "--contract", paths["contract"], "--order",
            paths["order"], "--drift", paths["drift"], "--record", paths["record"], "--measure",
            paths["measure"], "--grade-record", paths["grade_record"], "--fixtures",
            paths["fixtures"], "--out", tmp_path / "report.json", *extra]


def read_report(tmp_path):
    return json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def test_report_holds_every_key(tmp_path, capsys):
    paths = steps(tmp_path)
    rc, out, err = run(capsys, report_argv(tmp_path, paths))
    assert rc == 0, err
    rec = read_report(tmp_path)
    assert set(rec) == EXPECTED_KEYS
    assert rec["schema"] == "folio-accuracy/1" and rec["mode"] == "model-run"
    assert rec["status"] == "finished" and rec["started_at"] == "2026-01-01T00:00:00Z"
    assert rec["host"] == {"python": "3.12"} and rec["versions"] == {"tesseract": "5.3"}
    assert rec["ollama_version"] == "0.9.1" and rec["model_digest"] == "sha256:abc"
    assert rec["prompt_sha256"] == model_request.PROMPT_SHA256
    assert rec["model"] == {"name": "m:8b", "host": "127.0.0.1", "kind": "ollama",
                            "setup_keys": KEYS}
    assert rec["cells"] == [{"cell": "model-en"}] and rec["calibration_cells"] == [{"cell": "model-ar"}]
    assert rec["taus"] == {"key-ar": 0.8}
    assert rec["tau_table_sha256"] == calibration.table_sha256(calibration.TAUS)
    assert rec["tau_table_sha256_at_calibration"] == "oldsha"
    assert rec["heldout_manifest_sha256"] is None and rec["graded_heldout_manifest_sha256"] == SHA
    assert rec["dev_manifest_sha256"] == "devsha"
    assert rec["cer_per_fixture"] == {"en-a.pdf": 0.1, "ar-b.pdf": None}
    assert rec["pages"] == {"count": 9}
    assert rec["pages_by_step"] == {"calibrate": {"count": 4}, "grade": {"count": 9},
                                    "fixtures": {"count": 2}}
    assert rec["peak_vram_mib"] == 7000
    assert rec["vram_by_step"] == {"calibrate": 5000, "grade": 7000, "fixtures": 6000}
    assert rec["vram_errors"] == {}
    assert rec["contract"] == {"checks": [{"name": "x"}], "passed": True}
    assert rec["order"] == {"classes": {"a": {}}, "verdict": "frozen", "order_rule": "rule",
                            "documents": 3, "stopped_by": "minimums"}
    assert rec["drift"] == {"tolerance": 0.05, "max_spread_by_cell": {"model-ar": 0.05},
                            "text_changes": 1, "rounds": ["fresh"], "repeats": 3, "lines": 2}
    assert rec["dev_lines"] == [{"gt": "x"}] and rec["label"] is None
    assert set(rec["inputs"]) == {"contract", "order", "drift", "calibrate", "measure", "grade",
                                  "fixtures", "calibration", "grade_results"}
    assert rec["inputs"]["order"] == {"path": "o.json", "sha256": sha_of(paths["order"])}
    assert f"results: {tmp_path / 'report.json'}" in out
    assert f"sha256: {sha_of(tmp_path / 'report.json')}" in out


def set_step_fields(path, **fields):
    body = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**body, **fields}), encoding="utf-8")


def test_report_survives_a_grade_without_vram(tmp_path, capsys):
    paths = steps(tmp_path)
    set_step_fields(paths["grade_record"], peak_vram_mib=None, vram_error="the VRAM log failed")
    rc, _, err = run(capsys, report_argv(tmp_path, paths))
    assert rc == 0, err
    rec = read_report(tmp_path)
    assert rec["peak_vram_mib"] == 6000
    assert rec["vram_by_step"] == {"calibrate": 5000, "grade": None, "fixtures": 6000}
    assert rec["vram_errors"] == {"grade": "the VRAM log failed"}


def test_report_without_any_vram_has_null_peak(tmp_path, capsys):
    paths = steps(tmp_path)
    for key in ("record", "grade_record", "fixtures"):
        set_step_fields(paths[key], peak_vram_mib=None)
    rc, _, err = run(capsys, report_argv(tmp_path, paths))
    assert rc == 0, err
    assert read_report(tmp_path)["peak_vram_mib"] is None


def test_report_label(tmp_path, capsys):
    paths = steps(tmp_path)
    run(capsys, report_argv(tmp_path, paths, "--label", "laptop run"))
    assert read_report(tmp_path)["label"] == "laptop run"


def break_file(paths, case):
    if case == "missing":
        paths["drift"].unlink()
    elif case == "unfinished-grade":
        body = json.loads(paths["grade_record"].read_text(encoding="utf-8"))
        paths["grade_record"].write_text(json.dumps({**body, "status": "failed"}), encoding="utf-8")
    elif case == "keys":
        body = json.loads(paths["fixtures"].read_text(encoding="utf-8"))
        paths["fixtures"].write_text(json.dumps({**body, "keys": {"model-en": "x"}}),
                                     encoding="utf-8")
    elif case == "calibration-changed":
        paths["cal"].write_text("{}", encoding="utf-8")
    elif case == "grade-changed":
        paths["grade"].write_text("{}", encoding="utf-8")


@pytest.mark.parametrize("case", ["missing", "unfinished-grade", "keys", "calibration-changed",
                                  "grade-changed"])
def test_report_refusals_write_nothing(tmp_path, capsys, case):
    paths = steps(tmp_path)
    break_file(paths, case)
    before = tree(tmp_path)
    rc, _, err = run(capsys, report_argv(tmp_path, paths))
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert tree(tmp_path) == before
