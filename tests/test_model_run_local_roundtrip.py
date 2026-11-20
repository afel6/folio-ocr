"""A company calibrates its own model, served by a non-Ollama server with a declared identity,
into a local tau file, and `folio read` then auto-accepts a line with that file's threshold."""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import model_run
import model_run_common as common
import model_run_dry as dry
import pytest
import run_accuracy
from fake_model_server import Answer, FakeModelServer, Seen

from folio_ocr import cli, ocr_setup
from folio_ocr.assets import TESSERACT_OSD, files_problem
from folio_ocr.config import Settings, find_tesseract

WEIGHTS = Settings.from_env(os.environ).weights_dir
TESSERACT = find_tesseract(Settings.from_env(os.environ))

if ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"):
    pytest.skip("the OCR model files are not set up (folio setup)", allow_module_level=True)
if TESSERACT is None or files_problem((TESSERACT_OSD,), WEIGHTS) is not None:
    pytest.skip("Tesseract or the pinned osd.traineddata is not set up", allow_module_level=True)

MODEL_NAME = "acme-ocr-7b"
DIGEST = "cd" * 32
REPLY_TEXT = "Invoice number: INV-2026-3706"
INVOICE = HERE / "fixtures" / "en-invoice-clean.png"


def _completion(text: str) -> Answer:
    entries = [{"token": word, "logprob": 0.0, "bytes": list(word.encode())}
               for word in text.split(" ")]
    choice = {"index": 0, "message": {"role": "assistant", "content": text},
              "finish_reason": "stop", "logprobs": {"content": entries}}
    body = json.dumps({"choices": [choice], "usage": {"prompt_tokens": 9}}, ensure_ascii=False)
    return Answer(body=body.encode("utf-8"))


def _server_answer(seen: Seen) -> Answer:
    """No identity routes; the two self-test lines get their texts, every other crop English."""
    if seen.method != "POST":
        return Answer(status=404, body=b"{}")
    body = json.loads(seen.body)
    size, pixels = dry._grey(dry._request_image(body))
    for (line_size, line_pixels), line in dry.SELF_TEST:
        if size == line_size and pixels == line_pixels:
            return _completion(line.text)
    return _completion(REPLY_TEXT)


def _accuracy_stub():
    def main(argv, reader_factory=None, plan=None):
        keys = plan.keys(None)
        out = Path(argv[argv.index("--out") + 1])
        cells = [{"cell": "model-en", "status": "calibrated", "reason": "calibrated",
                  "bad_lines": 250, "documents_with_bad_lines": 45,
                  "missed_error_rate": {"bound_95": 0.08}},
                 {"cell": "model-ar", "status": "not measured",
                  "reason": "not measured (12 bad lines)", "bad_lines": 12,
                  "documents_with_bad_lines": 3, "missed_error_rate": {"bound_95": None}}]
        taus = {keys["model-en"]: 0.5}
        dev_sha = common.file_sha256(Path(argv[argv.index("--dev") + 1]) / "manifest.json")
        out.write_text(json.dumps({"taus": taus, "dev_manifest_sha256": dev_sha, "cells": cells}),
                       encoding="utf-8")
        return 0

    return main


def test_a_declared_model_calibrates_into_a_local_file_that_folio_read_uses(
        tmp_path, monkeypatch, capsys):
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "manifest.json").write_text(json.dumps({"schema": "folio-dev/1", "documents": []}))
    monkeypatch.setattr(run_accuracy, "main", _accuracy_stub())
    with FakeModelServer(_server_answer) as fake:
        monkeypatch.setenv("FOLIO_MODEL_URL", fake.url)
        monkeypatch.setenv("FOLIO_MODEL_NAME", MODEL_NAME)
        monkeypatch.setenv("FOLIO_MODEL_DIGEST", DIGEST)
        monkeypatch.setenv("FOLIO_MODEL_SERVER", "vllm 0.9")
        keys = model_run.build_session(tmp_path / "w", warm=False).keys
        drift = tmp_path / "drift.json"
        common.write_step(drift, "drift", "now", {"keys": keys, "tolerance": 0.05})
        tau_file = tmp_path / "company" / "taus.json"
        rc = model_run.main(["local-calibrate", "--dev", str(dev), "--drift", str(drift),
                             "--tau-file", str(tau_file), "--out", str(tmp_path / "cal.json"),
                             "--work", str(tmp_path / "w")])
        assert rc == 0, capsys.readouterr().err
        entries = json.loads(tau_file.read_text(encoding="utf-8"))["entries"]
        assert list(entries) == [keys["model-en"]] and "|declared:vllm-0.9|" in keys["model-en"]
        capsys.readouterr()

        monkeypatch.setenv("FOLIO_TAU_FILE", str(tau_file))
        rc = cli.main(["read", str(INVOICE), "--lang", "en", "--format", "json"],
                      environ=os.environ)
        out = capsys.readouterr()
    assert rc == 0, out.err
    doc = json.loads(out.out)
    lines = [line for page in doc["pages"] for line in page["lines"]]
    assert any(line["reader"] == "model" and line["status"] == "auto_accepted" for line in lines)
    assert doc["engine"]["calibration"]["source"] == "local"
