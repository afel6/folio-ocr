"""Tests for the local-calibrate command: a company calibrating its own model into a tau file."""

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_run
import model_run_grading as grading
import pytest
import run_accuracy
from model_run_server import ModelRunError
from model_run_support import make_session, run, sha_of, step_file, tree

from folio_ocr import calibration, model_taus
from folio_ocr.model_reader import SelfTestReply

KEYS = {"model-en": "model|acme|en", "model-ar": "model|acme|ar"}
DEV_MANIFEST = json.dumps({"schema": "folio-dev/1", "documents": []}).encode("utf-8")
DEV_SHA = hashlib.sha256(DEV_MANIFEST).hexdigest()
REPLIES = (SelfTestReply("en", "Invoice number", 0.9), SelfTestReply("ar", "رقم", 0.8))
GOOD_CELL = {"cell": "model-en", "status": "calibrated", "reason": "calibrated", "tau": 0.5,
             "bad_lines": 250, "documents_with_bad_lines": 45,
             "missed_error_rate": {"bound_95": 0.08}}
UNMEASURED = {"cell": "model-ar", "status": "not measured", "reason": "not measured (12 bad lines)",
              "tau": None, "bad_lines": 12, "documents_with_bad_lines": 3,
              "missed_error_rate": {"bound_95": None}}
OTHER_ENTRY = {"tau": 0.7, "method": "logprobs", "bad_lines": 300, "documents": 50,
               "dev_bound": 0.05, "dev_manifest_sha256": "cd" * 32,
               "self_test": [{"line": "en", "text": "a", "score": 0.9},
                             {"line": "ar", "text": "b", "score": 0.8}],
               "drift_tolerance": 0.01}


def accuracy_stub(cells, taus, calls=None, rc=0, dev_sha=DEV_SHA, during=None):
    def main(argv, reader_factory=None, plan=None):
        if calls is not None:
            calls.append((argv, reader_factory, plan))
        out = Path(argv[argv.index("--out") + 1])
        if during is not None:
            during()
        out.write_text(json.dumps({"taus": taus, "dev_manifest_sha256": dev_sha, "cells": cells}),
                       encoding="utf-8")
        return rc

    return main


@pytest.fixture
def setup(tmp_path, monkeypatch):
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "manifest.json").write_bytes(DEV_MANIFEST)
    drift = step_file(tmp_path / "drift.json", "drift", keys=KEYS, tolerance=0.05)
    calls = []
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub(
        [GOOD_CELL, UNMEASURED], {KEYS["model-en"]: 0.5}, calls))
    model = SimpleNamespace(fingerprint=lambda: REPLIES)
    session = dataclasses.replace(make_session(keys=KEYS), model=model)
    return SimpleNamespace(tmp=tmp_path, dev=dev, drift=drift, calls=calls, session=session,
                           tau=tmp_path / "out" / "taus.json")


def argv(s, **over):
    opts = {"--dev": s.dev, "--drift": s.drift, "--tau-file": s.tau, "--out": s.tmp / "cal.json",
            "--work": s.tmp / "w", **over}
    return ["local-calibrate", *[x for kv in opts.items() for x in kv]]


def entries_of(path):
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def write_existing(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": model_taus.SCHEMA, "entries": entries}), encoding="utf-8")


def test_writes_one_entry_for_the_calibrated_cell_and_reports_it(setup, capsys):
    rc, out, err = run(capsys, argv(setup, **{"--jobs": 2}), setup.session)
    assert rc == 0, err
    entries = entries_of(setup.tau)
    assert list(entries) == [KEYS["model-en"]]
    assert entries[KEYS["model-en"]] == {
        "tau": 0.5, "method": "logprobs", "bad_lines": 250, "documents": 45, "dev_bound": 0.08,
        "dev_manifest_sha256": DEV_SHA,
        "self_test": [{"line": "en", "text": "Invoice number", "score": 0.9},
                      {"line": "ar", "text": "رقم", "score": 0.8}],
        "drift_tolerance": 0.05}
    assert "model-en: tau 0.5 written" in out
    assert "model-ar: not measured (12 bad lines); no entry written" in out
    assert f"tau file: {setup.tau} sha256 {sha_of(setup.tau)}" in out
    assert out.isascii() and model_taus.parse(setup.tau.read_bytes()).entries


def test_runs_the_accuracy_calibration_with_the_model_plan_and_no_vram_log(setup, capsys):
    run(capsys, argv(setup, **{"--jobs": 3}), setup.session)
    ((args, factory, plan),) = setup.calls
    assert args == ["calibrate", "--dev", str(setup.dev), "--work", str(setup.tmp / "w"),
                    "--out", str(setup.tmp / "cal.json"), "--jobs", "3"]
    assert factory is setup.session.reader_factory and plan.keys(None) == KEYS
    assert not list((setup.tmp / "w").glob("vram*"))


def test_a_nonzero_accuracy_exit_code_is_returned_and_writes_nothing(setup, capsys, monkeypatch):
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub([GOOD_CELL], {}, rc=3))
    rc, _, _ = run(capsys, argv(setup), setup.session)
    assert rc == 3 and not setup.tau.exists()


def test_no_calibrated_cell_stops_and_leaves_the_file_alone(setup, capsys, monkeypatch):
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub([UNMEASURED], {}))
    write_existing(setup.tau, {"model|other|key": OTHER_ENTRY})
    before = setup.tau.read_bytes()
    rc, out, _ = run(capsys, argv(setup), setup.session)
    assert rc == 1 and "STOP: no model cell was calibrated; the tau file is unchanged" in out
    assert setup.tau.read_bytes() == before


def test_a_calibrated_cell_without_a_tau_gets_no_entry(setup, capsys, monkeypatch):
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub([GOOD_CELL], {}))
    rc, out, _ = run(capsys, argv(setup), setup.session)
    assert rc == 1 and "model-en: calibrated; no entry written" in out and not setup.tau.exists()


def test_other_entries_are_kept_and_an_equal_key_is_replaced(setup, capsys):
    stale = {**OTHER_ENTRY, "tau": 0.99}
    write_existing(setup.tau, {"model|other|key": OTHER_ENTRY, KEYS["model-en"]: stale})
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 0, err
    entries = entries_of(setup.tau)
    assert entries["model|other|key"] == OTHER_ENTRY
    assert entries[KEYS["model-en"]]["tau"] == 0.5 and len(entries) == 2


@pytest.mark.parametrize("refusal", ["manifest", "no-manifest", "schema", "drift", "tau-file",
                                     "keys"])
def test_every_refusal_reads_no_model_and_writes_nothing(setup, capsys, refusal):
    if refusal == "manifest":
        (setup.dev / "manifest.json").write_text("[]")
    if refusal == "no-manifest":
        (setup.dev / "manifest.json").unlink()
    if refusal == "schema":
        (setup.dev / "manifest.json").write_text(json.dumps({"schema": "folio-heldout/1"}))
    if refusal == "drift":
        setup.drift.write_text(json.dumps({"step": "drift", "status": "started"}))
    if refusal == "tau-file":
        setup.tau.parent.mkdir()
        setup.tau.write_bytes(b"{not json")
    if refusal == "keys":
        step_file(setup.drift, "drift", keys={"model-en": "model|changed"}, tolerance=0.05)
    before = tree(setup.tmp)
    rc, out, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert not setup.calls and out == "" and tree(setup.tmp) == before


def test_refusal_messages_are_the_specified_ones(setup, capsys):
    (setup.dev / "manifest.json").write_text("[]")
    _, _, err = run(capsys, argv(setup), setup.session)
    assert err == ("error: the --dev folder holds no development manifest (schema folio-dev/1); "
                   "a held-out manifest is never calibration input\n")
    (setup.dev / "manifest.json").write_text(json.dumps({"schema": "folio-dev/1"}))
    setup.tau.parent.mkdir()
    setup.tau.write_bytes(b"x")
    _, _, err = run(capsys, argv(setup), setup.session)
    assert err == "error: the tau file exists but is refused; fix or move it first\n"


def test_a_session_without_a_key_refuses_before_any_calibration(setup, capsys):
    def no_key(work, *, warm=True):
        raise ModelRunError("no calibration key can be formed: x")

    assert model_run.main([str(a) for a in argv(setup)], no_key) == 1
    assert not setup.calls and not setup.tau.exists()
    assert capsys.readouterr().err == "error: no calibration key can be formed: x\n"


def test_a_new_file_the_package_parser_would_refuse_is_never_written(setup, capsys, monkeypatch):
    cell = {**GOOD_CELL, "bad_lines": 5}
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub([cell], {KEYS["model-en"]: 0.5}))
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err.startswith("error: the new tau file would be refused: ")
    assert err.count("\n") == 1 and not setup.tau.exists()


def test_a_write_error_is_one_error_line(setup, capsys, monkeypatch):
    def deny(src, dst):
        raise PermissionError(13, "denied", str(dst))

    monkeypatch.setattr(os, "replace", deny)
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert not setup.tau.exists() and not list(setup.tau.parent.glob("*.tmp"))


def test_the_file_holds_no_nan(setup, capsys, monkeypatch):
    cell = {**GOOD_CELL, "missed_error_rate": {"bound_95": float("nan")}}
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub([cell], {KEYS["model-en"]: 0.5}))
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err.count("\n") == 1 and not setup.tau.exists()


def test_it_ignores_the_local_setup_variables_and_never_touches_the_package_table(
        setup, capsys, monkeypatch):
    for name in ("FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER", "FOLIO_TAU_FILE"):
        monkeypatch.setenv(name, "x")
    table = dict(calibration.TAUS)
    monkeypatch.setattr(grading, "_refuse_ungradable", lambda record: pytest.fail("package gate"))
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 0, err
    assert dict(calibration.TAUS) == table


def test_entries_another_tool_wrote_during_the_calibration_are_kept(setup, capsys, monkeypatch):
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub(
        [GOOD_CELL], {KEYS["model-en"]: 0.5},
        during=lambda: write_existing(setup.tau, {"model|other|key": OTHER_ENTRY})))
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 0, err
    entries = entries_of(setup.tau)
    assert entries["model|other|key"] == OTHER_ENTRY and entries[KEYS["model-en"]]["tau"] == 0.5


def test_a_tau_file_refused_during_the_calibration_is_not_overwritten(setup, capsys, monkeypatch):
    def corrupt():
        setup.tau.parent.mkdir(parents=True, exist_ok=True)
        setup.tau.write_bytes(b"{not json")

    monkeypatch.setattr(run_accuracy, "main", accuracy_stub(
        [GOOD_CELL], {KEYS["model-en"]: 0.5}, during=corrupt))
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err == "error: the tau file exists but is refused; fix or move it first\n"
    assert setup.tau.read_bytes() == b"{not json"
    assert not list(setup.tau.parent.glob("*.tmp"))


def test_a_dev_manifest_that_changed_during_the_calibration_writes_nothing(
        setup, capsys, monkeypatch):
    monkeypatch.setattr(run_accuracy, "main", accuracy_stub(
        [GOOD_CELL], {KEYS["model-en"]: 0.5}, dev_sha="cd" * 32))
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1
    assert err == "error: the dev manifest changed during the calibration; nothing was written\n"
    assert not setup.tau.exists()


def test_a_manifest_that_is_a_folder_is_reported_as_one(setup, capsys):
    (setup.dev / "manifest.json").unlink()
    (setup.dev / "manifest.json").mkdir()
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert "holds no development manifest" not in err and "folder" in err and not setup.calls


def test_an_unreadable_manifest_is_reported_as_permission_denied(setup, capsys, monkeypatch):
    manifest = setup.dev / "manifest.json"
    real = Path.read_bytes

    def read_bytes(self):
        if self == manifest:
            raise PermissionError(13, "denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err == f"error: permission denied: {manifest}\n" and not setup.calls


def test_an_unreadable_calibration_result_is_reported_as_an_os_error(setup, capsys, monkeypatch):
    out = setup.tmp / "cal.json"
    real = Path.read_bytes

    def read_bytes(self):
        if self == out:
            raise PermissionError(13, "denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    rc, _, err = run(capsys, argv(setup), setup.session)
    assert rc == 1 and err == f"error: permission denied: {out}\n" and not setup.tau.exists()
