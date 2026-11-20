"""The Tesseract TSV capture script reports bad input as one line and exit code 1."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import capture_tesseract_tsv as capture


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)


def test_a_bad_setting_prints_one_line_and_no_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FOLIO_PORT", "abc")
    assert capture.main(["capture", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err == "FOLIO_PORT must be a whole number from 1 to 65535\n"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("argv", [["capture"], ["capture", "a", "b"]])
def test_wrong_argument_count_prints_usage(capsys, argv):
    assert capture.main(argv) == 1
    assert capsys.readouterr().err == "usage: capture_tesseract_tsv.py OUTPUT_FOLDER\n"


def test_without_tesseract_it_says_so(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(capture, "find_tesseract", lambda settings: None)
    assert capture.main(["capture", str(tmp_path)]) == 1
    assert capsys.readouterr().err == "Tesseract was not found; set FOLIO_TESSERACT_PATH\n"
