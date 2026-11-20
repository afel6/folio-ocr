"""The fake model server's routing used by the dry run of the model-run pipeline."""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import accuracy_stats
import model_run_dry as dry
import model_run_vram
from fake_model_server import Seen

from folio_ocr import calibration, model_request
from folio_ocr.model_self_test_lines import LINES


def png_of(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("L").save(out, format="PNG")
    return out.getvalue()


def chat(png: bytes, tokens: int) -> Seen:
    body = model_request.request_body("m", png, tokens)
    return Seen("POST", "/v1/chat/completions", {}, body)


def reply(seen: Seen) -> dict:
    return json.loads(dry.answer(seen).body)


def content(seen: Seen) -> str:
    return reply(seen)["choices"][0]["message"]["content"]


def entries(seen: Seen) -> list:
    return reply(seen)["choices"][0]["logprobs"]["content"]


def test_version_and_tags():
    version = dry.answer(Seen("GET", "/api/version", {}, b""))
    assert json.loads(version.body) == {"version": "0.40.16"}
    tags = json.loads(dry.answer(Seen("GET", "/api/tags", {}, b"")).body)
    assert tags["models"][0]["name"] == dry.MODEL_NAME
    assert tags["models"][0]["digest"] == dry.MODEL_DIGEST


def test_self_test_lines_are_answered_exactly_even_when_reencoded():
    for line in LINES:
        again = png_of(Image.open(io.BytesIO(line.png)))
        assert content(chat(again, 128)) == line.text
        assert content(chat(line.png, 128)) == line.text


def test_one_token_budget_is_truncated():
    seen = chat(LINES[0].png, 1)
    assert reply(seen)["choices"][0]["finish_reason"] == "length"
    assert content(seen) == "Invoice"


def test_white_image_is_blank():
    seen = chat(png_of(Image.new("L", (200, 40), 255)), 60)
    assert content(seen) == ""


def test_latin_rate_gets_english_and_other_rate_gets_arabic():
    image = Image.new("L", (300, 30), 128)
    latin = 16 + math.ceil(3 * 300 / 30)
    assert content(chat(png_of(image), latin)) == "Invoice number: INV-2026-3706"
    assert content(chat(png_of(image), latin + 20)) == "رقم الفاتورة"
    assert len(entries(chat(png_of(image), latin + 20))) == 5


def test_logprobs_vary_with_the_request_counter():
    seen = chat(png_of(Image.new("L", (300, 30), 128)), 999)
    firsts = {entries(seen)[0]["logprob"] for _ in range(3)}
    assert len(firsts) == 3


def test_a_file_as_the_work_folder_gives_one_error_line(tmp_path, capsys):
    work = tmp_path / "afile"
    work.write_text("x")
    assert dry.main(["--work", str(work)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


def test_a_non_empty_work_folder_is_refused(tmp_path, capsys):
    (tmp_path / "old.txt").write_text("x")
    assert dry.main(["--work", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err == "error: the work folder is not empty; give a new or empty folder\n"


def test_the_run_clears_the_model_identity_variables_and_restores_them(monkeypatch):
    names = ("FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER", "FOLIO_TAU_FILE", "FOLIO_MODEL_API_KEY")
    for name in names:
        monkeypatch.setenv(name, f"outer-{name}")
    monkeypatch.setenv("FOLIO_MODEL_URL", "http://outer/v1")
    saved = dry._save_environment()
    dry._set_environment("http://127.0.0.1:1/v1")
    assert all(name not in os.environ for name in names)
    assert os.environ["FOLIO_MODEL_URL"] == "http://127.0.0.1:1/v1"
    dry._restore_environment(saved)
    assert all(os.environ[name] == f"outer-{name}" for name in names)
    assert os.environ["FOLIO_MODEL_URL"] == "http://outer/v1"


def test_variables_that_were_unset_stay_unset_after_the_run(monkeypatch):
    monkeypatch.delenv("FOLIO_TAU_FILE", raising=False)
    saved = dry._save_environment()
    os.environ["FOLIO_TAU_FILE"] = "set during the run"
    dry._restore_environment(saved)
    assert "FOLIO_TAU_FILE" not in os.environ


def test_the_taus_are_filled_after_measure_and_before_grade(tmp_path, monkeypatch):
    for name in ("TAUS", "lookup", "table_sha256"):
        monkeypatch.setattr(calibration, name, getattr(calibration, name))
    seen = {}

    def record(argv):
        seen[argv[0]] = "KEY" in calibration.TAUS
        return 0

    monkeypatch.setattr(dry.model_run, "main", record)
    (tmp_path / "steps").mkdir()
    (tmp_path / "steps" / "calibrate.json").write_text(json.dumps({"taus": {"KEY": 0.5}}))
    names = ("calibrate", "measure", "grade", "fixtures", "report")
    dry._run_steps([(n, [n]) for n in names], tmp_path)
    assert seen == {"calibrate": False, "measure": False, "grade": True, "fixtures": True,
                    "report": True}


def test_a_stopped_run_restores_the_process_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FOLIO_MODEL_URL", "http://outer/v1")
    monkeypatch.setattr(dry.shutil, "which", lambda name: None)  # the nvidia-smi stub goes in too
    for name in ("TAUS", "lookup", "table_sha256"):
        monkeypatch.setattr(calibration, name, getattr(calibration, name))
    before = (calibration.TAUS, calibration.lookup, calibration.table_sha256,
              model_run_vram.nvidia_smi_command)
    minimums = {n: getattr(accuracy_stats, n) for n in
                ("DEV_MIN_BAD", "DEV_MIN_DOCS", "TEST_MIN_BAD", "TEST_MIN_DOCS")}

    def stop(work, jobs):
        dry._fill_taus({"KEY": 0.5})
        raise dry.DryRunError("stopped")

    monkeypatch.setattr(dry, "_run", stop)
    assert dry.main(["--work", str(tmp_path / "new")]) == 1
    assert capsys.readouterr().err == "error: stopped\n"
    after = (calibration.TAUS, calibration.lookup, calibration.table_sha256,
             model_run_vram.nvidia_smi_command)
    assert all(a is b for a, b in zip(after, before, strict=True))
    assert {n: getattr(accuracy_stats, n) for n in minimums} == minimums
    assert os.environ["FOLIO_MODEL_URL"] == "http://outer/v1"
