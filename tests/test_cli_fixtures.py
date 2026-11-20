"""The command line on the real fixture files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import docx
import pytest
from docx.enum.text import WD_COLOR_INDEX
from fake_tesseract_install import install_fake_tesseract

from folio_ocr import cli, ocr_setup, tesseract
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.export import to_json, unresolved_count
from folio_ocr.metrics import cer
from folio_ocr.pipeline import read_path

FIXTURES = Path(__file__).parent / "fixtures"


ENV: dict[str, str] = {}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path_factory):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)
    weights = tmp_path_factory.mktemp("weights")
    monkeypatch.setenv("FOLIO_WEIGHTS_DIR", str(weights))
    monkeypatch.setattr(sys.modules[__name__], "ENV", {"FOLIO_WEIGHTS_DIR": str(weights)})


def main(*argv):
    return cli.main([str(a) for a in argv], ENV)


def real_read(name, lang="en+ar"):
    settings = Settings.from_env(ENV)
    return read_path(
        FIXTURES / name,
        languages=lang,
        weights_dir=settings.weights_dir,
        tesseract_path=find_tesseract(settings),
    )


def pooled(tmp_path, names, lang, limit):
    assert main("read", *[FIXTURES / n for n in names], "--lang", lang, "--out", tmp_path) == 0
    pairs = []
    for n in names:
        stem = n.removesuffix(".pdf")
        gt = json.loads((FIXTURES / f"{stem}.gt.json").read_text(encoding="utf-8"))
        ref = " ".join(ln["text"] for ln in gt["lines"])
        text = (tmp_path / f"{stem}.txt").read_text(encoding="utf-8")
        pairs.append((ref, " ".join(text.splitlines())))
    assert cer(pairs).rate <= limit


def test_english_text_layer_accuracy(tmp_path):
    pooled(tmp_path, ["en-invoice.pdf", "en-letter.pdf"], "en", 0.005)


def test_arabic_text_layer_accuracy(tmp_path):
    pooled(tmp_path, ["ar-invoice.pdf", "ar-letter.pdf"], "ar", 0.01)


def test_image_summary_matches_the_pipeline(capsys, ocr_stub):
    path = FIXTURES / "en-invoice-clean.png"
    assert main("read", path) == 0
    d = real_read(path.name)
    found = sum(len(p.lines) for p in d.pages)
    read = sum(x.reader != "none" for p in d.pages for x in p.lines)
    unread = sum(p.status != "read" for p in d.pages)
    pages = json.loads(to_json(d))["pages"]
    review = sum(x["status"] == "needs_review" for p in pages for x in p["lines"])
    expect = (
        f"{path}: {found} lines found, {read} read, {review} need review, {unread} unread pages"
    )
    err = capsys.readouterr().err
    assert expect in err
    if d.pages[0].status != "read":
        assert "1 unread pages" in err


def test_docx_output_is_readable(tmp_path):
    code = main("read", FIXTURES / "ar-invoice.pdf", "--lang", "ar", "--format", "docx",
                "--out", tmp_path)  # fmt: skip
    assert code == 0
    document = docx.Document(str(tmp_path / "ar-invoice.docx"))
    assert any(p._p.xpath("./w:pPr/w:bidi") for p in document.paragraphs)
    marked = [r for p in document.paragraphs for r in p.runs
              if r.font.highlight_color == WD_COLOR_INDEX.YELLOW]  # fmt: skip
    assert len(marked) == unresolved_count(real_read("ar-invoice.pdf", "ar"))


def test_json_to_stdout(capsys):
    assert main("read", FIXTURES / "en-invoice.pdf", "--format", "json") == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    got, want = json.loads(out), real_read("en-invoice.pdf")
    assert got["source"]["filename"] == "en-invoice.pdf"
    assert [[x["text"] for x in p["lines"]] for p in got["pages"]] == [
        [x.text for x in p.lines] for p in want.pages
    ]


def test_subprocess_stdout_is_utf8_without_crlf(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FOLIO_")} | ENV
    env.pop("PYTHONIOENCODING", None)
    src = FIXTURES / "ar-invoice.pdf"
    done = subprocess.run(
        [sys.executable, "-m", "folio_ocr", "read", str(src), "--lang", "ar"],
        check=False, capture_output=True, env=env, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0
    assert main("read", src, "--lang", "ar", "--out", tmp_path) == 0
    assert done.stdout.decode("utf-8") == (tmp_path / "ar-invoice.txt").read_bytes().decode("utf-8")
    assert b"\r\n" not in done.stdout
    err = done.stderr.decode("utf-8", "replace")
    assert "lines found" in err and "Traceback" not in err


WEIGHTS = Settings.from_env(os.environ).weights_dir
needs_models = pytest.mark.skipif(
    ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data"),
    reason="the OCR model files are not set up (folio setup)",
)
PROGRAM = "tesseract.exe" if sys.platform == "win32" else "tesseract"
DOCTOR_HINT = "hint: run folio doctor to check the readers"


def read_json_with(capsys, name, environ):
    environ = {"FOLIO_WEIGHTS_DIR": str(WEIGHTS), "PATH": "", **environ}
    path = FIXTURES / name
    code = cli.main(["read", "--lang", "ar", "--format", "json", str(path)], environ)
    cap = capsys.readouterr()
    return code, json.loads(cap.out)["pages"][0]["error"], cap.err.splitlines(), path


@needs_models
def test_a_missing_tesseract_program_is_reported(capsys, tmp_path):
    missing = tmp_path / "nowhere" / PROGRAM
    code, error, lines, path = read_json_with(
        capsys, "ar-letter-photo.jpg", {"FOLIO_TESSERACT_PATH": str(missing)}
    )
    assert code == 1 and isinstance(error, str) and error
    assert lines[1] == f"{path}: page 1: {error}"
    assert lines[-1] == DOCTOR_HINT and lines.count(DOCTOR_HINT) == 1
    if tesseract.tessdata_problem(WEIGHTS) is None:
        assert error == "Tesseract was not found at the configured path"


@needs_models
@pytest.mark.skipif(
    tesseract.tessdata_problem(WEIGHTS) is not None, reason="the Tesseract language data is missing"
)
def test_a_tesseract_line_timeout_is_reported(capsys, tmp_path, monkeypatch):
    fake = install_fake_tesseract(tmp_path / "bin" / PROGRAM)
    # a new program's first start can be slow: start it once before the timed read
    subprocess.run([str(fake), "--version"], capture_output=True, timeout=120, check=True)
    monkeypatch.setenv("FAKE_TESSERACT_SLEEP", "3")
    code, error, lines, path = read_json_with(
        capsys,
        "ar-letter-clean.png",
        {"FOLIO_TESSERACT_PATH": str(fake), "FOLIO_LINE_TIMEOUT_S": "0.5"},
    )
    assert code == 1 and "within 0.5 s" in error
    assert lines[1] == f"{path}: page 1: {error}"
    assert lines[-1] == DOCTOR_HINT and lines.count(DOCTOR_HINT) == 1
