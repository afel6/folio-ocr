"""Tesseract's page-orientation report: what is an answer, what is no answer, what is a failure."""

import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from folio_ocr import assets, orientation
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.tesseract import (
    Osd,
    TesseractFailure,
    osd,
    osd_command,
    page_png,
    parse_osd,
    tessdata_problem,
)

DATA = Path(__file__).parent / "data" / "tesseract"
FIXTURES = Path(__file__).parent / "fixtures"
BINARY = Path(sys.executable)


def pinned(name):
    """(exit code, stdout, stderr) the installed Tesseract gave for the page `name`."""
    return (
        int((DATA / f"{name}.exit").read_text()),
        (DATA / f"{name}.stdout").read_bytes(),
        (DATA / f"{name}.stderr").read_bytes(),
    )


def run_with(code, out, err=b""):
    done = SimpleNamespace(returncode=code, stdout=out, stderr=err)
    return lambda *a, **k: done


def osd_out(rot="90", conf="2.80", degrees=None):
    if degrees is None:
        degrees = str((360 - int(rot)) % 360) if rot.isdigit() else "0"
    return (
        f"Page number: 0\nOrientation in degrees: {degrees}\nRotate: {rot}\n"
        f"Orientation confidence: {conf}\nScript: Arabic\nScript confidence: 1.67\n"
    ).encode()


def test_the_pinned_report_is_an_answer():
    code, out, err = pinned("osd-report")
    assert code == 0
    assert osd(BINARY, Path("."), b"x", 1, runner=run_with(code, out, err)) == Osd(0, 17.05)


@pytest.mark.parametrize("name", ["osd-blank", "osd-two-words"])
def test_the_pinned_no_answer_output_is_none(name):
    code, out, err = pinned(name)
    assert (code, out) == (1, b"")
    assert b"Too few characters" in err
    assert osd(BINARY, Path("."), b"x", 1, runner=run_with(code, out, err)) is None


def test_empty_output_with_exit_zero_is_a_failure():
    with pytest.raises(TesseractFailure):
        osd(BINARY, Path("."), b"x", 1, runner=run_with(0, b"", b""))


def test_exit_one_without_the_no_answer_message_is_a_failure():
    with pytest.raises(TesseractFailure):
        osd(BINARY, Path("."), b"x", 1, runner=run_with(1, b"", b"Error opening data file\n"))


def test_other_exit_codes_are_failures_even_with_the_no_answer_message():
    _, _, err = pinned("osd-blank")
    for code in (2, -1):
        with pytest.raises(TesseractFailure, match="exit code"):
            osd(BINARY, Path("."), b"x", 1, runner=run_with(code, b"", err))
    with pytest.raises(TesseractFailure):
        osd(BINARY, Path("."), b"x", 1, runner=run_with(1, osd_out(), err))


@pytest.mark.parametrize("rot,quarter", [("0", 0), ("90", 270), ("180", 180), ("270", 90)])
def test_osd_rotate_gives_the_quarter_that_turns_the_page_upright(rot, quarter):
    assert parse_osd(osd_out(rot)) == Osd(quarter, 2.8)
    assert parse_osd(osd_out(rot).replace(b"\n", b"\r\n")) == Osd(quarter, 2.8)


@pytest.mark.parametrize(
    "stdout",
    [
        b"",
        b"Page number: 0\nOrientation confidence: 2.80\n",
        b"Rotate: 90\nOrientation confidence: 2.80\n",
        osd_out("45"),
        osd_out("ninety"),
        osd_out("90", "abc"),
        osd_out("90", "nan"),
        osd_out("90", "inf"),
        osd_out("90", "-0.5"),
        osd_out("90", degrees="90"),  # Rotate + Orientation is not 0 mod 360
        osd_out("0", degrees="180"),
        osd_out("90", degrees="x"),
        b"Orientation in degrees: 270\nRotate: 90\nOrientation confidence: 2\xff\n",
    ],
)
def test_a_report_that_is_not_understood_or_inconsistent_is_a_failure(stdout):
    with pytest.raises(TesseractFailure) as caught:
        parse_osd(stdout)
    assert "\n" not in str(caught.value)


def test_osd_command_is_the_pinned_argument_list(tmp_path):
    assert osd_command(BINARY, tmp_path) == [
        sys.executable, "stdin", "stdout", "--tessdata-dir", str(tmp_path), "--psm", "0"
    ]  # fmt: skip
    with pytest.raises(TesseractFailure):
        osd_command(Path("tesseract"), tmp_path)


def test_page_png_is_the_whole_page_in_grey():
    with Image.open(io.BytesIO(page_png(Image.new("RGB", (7, 5), "red")))) as png:
        assert (png.mode, png.size) == ("L", (7, 5))


def test_osd_runs_the_program_with_the_page_on_stdin(tmp_path):
    seen = {}

    def runner(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stdout=osd_out(), stderr=b"")

    assert osd(BINARY, tmp_path, b"png", 7, runner=runner) == Osd(270, 2.8)
    assert seen["argv"] == osd_command(BINARY, tmp_path)
    assert (seen["input"], seen["timeout"]) == (b"png", 7)


def test_a_program_past_the_time_limit_is_a_failure(tmp_path):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    with pytest.raises(TesseractFailure, match="within 1 s"):
        osd(BINARY, tmp_path, b"x", 1, runner=timeout)


def test_elapsed_time_is_checked_again_after_the_call(tmp_path):
    ticks = iter([0.0, 9.0])
    with pytest.raises(TesseractFailure, match="within 2 s"):
        osd(BINARY, tmp_path, b"x", 2, runner=run_with(0, osd_out()), clock=lambda: next(ticks))


def test_a_program_that_cannot_start_is_a_failure(tmp_path):
    with pytest.raises(TesseractFailure):
        osd(tmp_path / "missing.exe", tmp_path, b"x", 1)
    with pytest.raises(TesseractFailure):
        osd(Path("relative"), tmp_path, b"x", 1, runner=run_with(0, osd_out()))


def test_a_nonzero_exit_that_is_not_the_no_answer_shape_is_a_failure(tmp_path):
    with pytest.raises(TesseractFailure, match="exit code 3") as caught:
        osd(BINARY, tmp_path, b"x", 1, runner=run_with(3, osd_out(), b"C:\\secret\\path"))
    assert "secret" not in str(caught.value)


def test_unreadable_report_with_exit_zero_is_a_failure(tmp_path):
    with pytest.raises(TesseractFailure):
        osd(BINARY, tmp_path, b"x", 1, runner=run_with(0, b"garbage"))


# --- the osd file is a required Tesseract file


def test_tessdata_problem_names_the_osd_file(tmp_path, monkeypatch):
    folder = tmp_path / assets.TESSDATA_DIR
    folder.mkdir()
    monkeypatch.setattr(assets, "TESSDATA", ())
    assert tessdata_problem(tmp_path) == "Tesseract language data is missing; run folio setup"
    (folder / "osd.traineddata").write_bytes(b"not the pinned file")
    assert tessdata_problem(tmp_path) == "Tesseract language data is damaged; run folio setup"


# --- the real program

WEIGHTS = Settings.from_env(os.environ).weights_dir
REAL_BINARY = find_tesseract(Settings.from_env(os.environ))
REAL_DATA = WEIGHTS / assets.TESSDATA_DIR
real = pytest.mark.skipif(
    REAL_BINARY is None
    or not Path(REAL_BINARY).is_file()
    or assets.files_problem((assets.TESSERACT_OSD,), WEIGHTS) is not None,
    reason="Tesseract or the pinned osd.traineddata is not installed",
)


@real
@pytest.mark.parametrize(
    "turn,quarter",
    [
        (None, 0),
        (Image.Transpose.ROTATE_90, 270),  # turned 90 counterclockwise: 270 stands it up
        (Image.Transpose.ROTATE_180, 180),
        (Image.Transpose.ROTATE_270, 90),
    ],
)
def test_real_osd_quarter_of_a_turned_invoice(turn, quarter):
    with Image.open(FIXTURES / "en-invoice-clean.png") as image:
        page = image if turn is None else image.transpose(turn)
        answer = osd(Path(REAL_BINARY), REAL_DATA, page_png(page), 60)
    assert answer is not None and answer.quarter == quarter
    assert answer.confidence >= orientation.OSD_FLOOR


@real
def test_real_blank_page_is_no_answer():
    blank = page_png(Image.new("L", (1000, 1000), 255))
    assert osd(Path(REAL_BINARY), REAL_DATA, blank, 60) is None
