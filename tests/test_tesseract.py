import errno
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fake_tesseract_install import install_fake_tesseract
from PIL import Image

from folio_ocr import assets, tesseract
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.reading import Reading
from folio_ocr.tesseract import (
    TesseractFailure,
    TesseractTimeout,
    command,
    line_png,
    parse_tsv,
    read_line,
    run,
    tessdata_problem,
    tesseract_version,
)

DATA = Path(__file__).parent / "data" / "tesseract"
EN_TSV = (DATA / "en-line.tsv").read_bytes()
AR_TSV = (DATA / "ar-line.tsv").read_bytes()
AR_TEXT = "فاتورة : مؤسسة | لنخبة للتوريد ات"
SUFFIX = ".exe" if sys.platform == "win32" else ""
HEADER = EN_TSV.decode().splitlines()[0]
ROW5 = "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t{conf}\t{text}"
NOT_A_PROGRAM = "the configured Tesseract file is not a program this system can run"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    binary = install_fake_tesseract(tmp_path / "dir with spaces" / f"-fake tesseract{SUFFIX}")
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_TESSERACT_LOG", str(log))
    return binary, tmp_path / "-weights dir" / "tessdata", log


def tsv(*rows: str) -> bytes:
    return "\r\n".join([HEADER, *rows]).encode("utf-8")


def ok_runner(stdout=EN_TSV):
    done = SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
    return lambda *a, **k: done


# --- run / read_line with the fake program


def test_arguments_reach_the_program_one_by_one(fake):
    binary, tessdata, log = fake
    png = line_png(Image.new("RGB", (50, 20), "white"), (5, 5, 40, 15))
    reading = read_line(binary, tessdata, png, 30)
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["argv"] == command(binary, tessdata)[1:]
    assert entry["stdin_len"] == len(png)
    assert entry["stdin_sha256"] == hashlib.sha256(png).hexdigest()
    assert reading == parse_tsv(EN_TSV)


def test_reader_receives_the_committed_bytes_unchanged(fake):
    binary, tessdata, _ = fake
    assert run(binary, tessdata, b"x", 30) == EN_TSV


def test_command_is_the_pinned_argument_list():
    binary = Path(sys.executable).resolve()
    tessdata = Path(sys.executable).resolve().parent
    assert command(binary, tessdata) == [
        str(binary), "stdin", "stdout", "--tessdata-dir", str(tessdata), "-l", "ara+eng",
        "--psm", "13", "--oem", "1", "-c", "tessedit_create_tsv=1",
    ]  # fmt: skip


def test_relative_binary_never_starts(fake):
    _, tessdata, log = fake
    reading = read_line(Path("tesseract"), tessdata, b"x", 30)
    assert reading == Reading("", (), error="the Tesseract path must be absolute")
    assert not log.exists()


def test_slow_program_is_stopped(fake, monkeypatch):
    binary, tessdata, _ = fake
    monkeypatch.setenv("FAKE_TESSERACT_SLEEP", "5")
    reading = read_line(binary, tessdata, b"x", 0.5)
    assert reading.error == "Tesseract did not finish a line within 0.5 s"


def test_elapsed_time_is_checked_again_after_the_call(tmp_path):
    ticks = iter([0.0, 9.0])
    with pytest.raises(TesseractFailure, match="within 2 s"):
        run(Path(sys.executable), tmp_path, b"x", 2, runner=ok_runner(), clock=lambda: next(ticks))


def test_runner_is_called_without_shell_or_env(tmp_path):
    seen = {}

    def runner(argv, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=EN_TSV, stderr=b"")

    run(Path(sys.executable), tmp_path, b"png", 7, runner=runner)
    assert seen == {"input": b"png", "capture_output": True, "timeout": 7, "check": False}


def test_exit_code(fake, monkeypatch):
    binary, tessdata, _ = fake
    monkeypatch.setenv("FAKE_TESSERACT_EXIT", "3")
    assert read_line(binary, tessdata, b"x", 30).error == "Tesseract failed (exit code 3)"


def test_missing_file(tmp_path):
    reading = read_line(tmp_path / "nope" / f"tess{SUFFIX}", tmp_path, b"x", 30)
    assert reading.error == "Tesseract was not found at the configured path"


def test_folder_as_binary_is_one_line_without_path(tmp_path):
    reading = read_line(tmp_path, tmp_path, b"x", 30)
    assert reading.error in {
        "Tesseract cannot be started: permission denied",
        "Tesseract was not found at the configured path",
        "Tesseract could not be started",
    }
    assert str(tmp_path) not in reading.error


def test_text_file_is_not_a_program(tmp_path):
    bogus = tmp_path / f"x{SUFFIX}"
    bogus.write_text("hello")
    expected = (
        NOT_A_PROGRAM
        if sys.platform == "win32"
        else "Tesseract cannot be started: permission denied"
    )
    assert read_line(bogus, tmp_path, b"x", 30).error == expected


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (errno.ENOENT, "Tesseract was not found at the configured path"),
        (errno.EACCES, "Tesseract cannot be started: permission denied"),
        (errno.EPERM, "Tesseract cannot be started: permission denied"),
        (errno.ENOEXEC, NOT_A_PROGRAM),
        (errno.EIO, "Tesseract could not be started"),
    ],
)
def test_each_errno_has_its_message(tmp_path, code, message):
    def runner(*a, **k):
        raise OSError(code, "secret " + str(tmp_path))

    assert read_line(Path(sys.executable), tmp_path, b"x", 1, runner=runner).error == message


def test_winerror_193_is_not_a_program(tmp_path):
    def runner(*a, **k):
        exc = OSError(errno.EINVAL, "bad")
        exc.winerror = 193
        raise exc

    assert read_line(Path(sys.executable), tmp_path, b"x", 1, runner=runner).error == NOT_A_PROGRAM


# --- parse_tsv


def test_english_fixture():
    reading = parse_tsv(EN_TSV)
    assert reading.text == "Invoice: Northwind Office Supply Ltd"
    assert len(reading.scores) == 5
    assert all(0.9 < s <= 1 for s in reading.scores)
    assert reading.error is None


def test_arabic_fixture():
    reading = parse_tsv(AR_TSV)
    assert reading.text == AR_TEXT
    assert len(reading.scores) == 7
    assert b"\r\n" in AR_TSV


def test_bom_is_not_tsv():
    with pytest.raises(TesseractFailure, match="not TSV"):
        parse_tsv(b"\xef\xbb\xbf" + EN_TSV)


def test_invalid_utf8():
    with pytest.raises(TesseractFailure, match="not UTF-8"):
        parse_tsv(EN_TSV + b"\xff")


def test_short_row_is_not_tsv():
    with pytest.raises(TesseractFailure, match="not TSV"):
        parse_tsv(tsv("5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90"))


def test_bad_numbers_are_not_tsv():
    with pytest.raises(TesseractFailure, match="not TSV"):
        parse_tsv(tsv(ROW5.format(conf="high", text="a")))
    with pytest.raises(TesseractFailure, match="not TSV"):
        parse_tsv(tsv(ROW5.format(conf="9", text="a").replace("5\t", "x\t", 1)))


def test_wrong_header_and_empty_output():
    for blob in (b"", b"hello\r\n"):
        with pytest.raises(TesseractFailure, match="not TSV"):
            parse_tsv(blob)


def test_header_only():
    assert parse_tsv(tsv()) == Reading("", ())


def test_unusable_confidence_is_kept():
    assert parse_tsv(tsv(ROW5.format(conf="-1", text="a"))) == Reading("a", (-0.01,))


def test_empty_word_scores_but_adds_no_text():
    reading = parse_tsv(tsv(ROW5.format(conf="50", text="a"), ROW5.format(conf="40", text=" ")))
    assert reading.text == "a"
    assert reading.scores == (0.5, 0.4)


# --- line_png


def test_line_png_pads_and_clamps():
    png = line_png(Image.new("RGB", (100, 50), "white"), (10.5, 20.2, 30.1, 49))
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    out = Image.open(io.BytesIO(png))
    assert out.mode == "L"
    assert out.size == (33, 36)  # x 4..37 and y 14..50 (clamped)
    assert "dpi" not in out.info


def test_line_png_clamps_at_origin():
    out = Image.open(io.BytesIO(line_png(Image.new("L", (20, 20)), (2, 2, 10, 10))))
    assert out.size == (16, 16)


def test_line_png_empty_box():
    with pytest.raises(ValueError):
        line_png(Image.new("L", (20, 20)), (50, 50, 60, 60))


# --- version


def test_version_with_fake(fake):
    # A generous limit: the fake starts a Python interpreter, slow on a busy machine.
    assert tesseract_version(fake[0], 120.0) == "5.4.0.20240606"


def test_version_check_names_its_own_time_limit():
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10.0)

    with pytest.raises(TesseractFailure) as caught:
        tesseract_version(Path(sys.executable), runner=slow)
    assert str(caught.value) == "Tesseract did not report its version within 10 s"


def test_running_out_of_time_is_told_apart_from_other_failures():
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10.0)

    ticks = iter((0.0, 11.0))
    with pytest.raises(TesseractTimeout):
        tesseract_version(Path(sys.executable), runner=slow)
    with pytest.raises(TesseractTimeout):  # answered, but only after its time limit
        run(Path(sys.executable), Path("."), b"x", 10.0, runner=ok_runner(), clock=lambda: next(ticks))
    with pytest.raises(TesseractFailure) as other:
        tesseract_version(Path(sys.executable), runner=ok_runner(b""))
    assert not isinstance(other.value, TesseractTimeout)


def test_version_not_reported():
    with pytest.raises(TesseractFailure, match="did not report its version"):
        tesseract_version(Path(sys.executable), runner=ok_runner(b""))


def test_version_needs_absolute_path():
    with pytest.raises(TesseractFailure, match="absolute"):
        tesseract_version(Path("tesseract"))


# --- tessdata_problem


def fake_assets(tmp_path, monkeypatch):
    folder = tmp_path / assets.TESSDATA_DIR
    folder.mkdir()
    listed = []
    for name, body in (
        ("a.traineddata", b"alpha"),
        ("b.traineddata", b"beta"),
        ("osd.traineddata", b"orientation"),
    ):
        digest = hashlib.sha256(body).hexdigest()
        listed.append(
            assets.Asset(
                f"{assets.TESSDATA_DIR}/{name}", "https://x.test/f", digest, len(body), "t"
            )
        )
        (folder / name).write_bytes(body)
    monkeypatch.setattr(tesseract.assets, "TESSDATA", tuple(listed[:2]))
    monkeypatch.setattr(tesseract.assets, "TESSERACT_OSD", listed[2])
    return folder


def test_tessdata_good(tmp_path, monkeypatch):
    fake_assets(tmp_path, monkeypatch)
    assert tessdata_problem(tmp_path) is None


def test_tessdata_missing(tmp_path, monkeypatch):
    (fake_assets(tmp_path, monkeypatch) / "b.traineddata").unlink()
    assert tessdata_problem(tmp_path) == "Tesseract language data is missing; run folio setup"


def test_tessdata_damaged(tmp_path, monkeypatch):
    (fake_assets(tmp_path, monkeypatch) / "a.traineddata").write_bytes(b"ALPHA")
    assert tessdata_problem(tmp_path) == "Tesseract language data is damaged; run folio setup"


def test_tessdata_unreadable(tmp_path, monkeypatch):
    folder = fake_assets(tmp_path, monkeypatch)
    (folder / "a.traineddata").unlink()
    (folder / "a.traineddata").mkdir()  # a folder where the data file should be
    assert tessdata_problem(tmp_path) == (
        "Tesseract language data cannot be read; check the access rights of the assets folder"
    )


# --- the real program


def _real():
    settings = Settings.from_env(os.environ)
    binary = find_tesseract(settings)
    if binary is None or not Path(binary).is_file():
        return None, None, "Tesseract is not installed"
    if (problem := tessdata_problem(settings.weights_dir)) is not None:
        return None, None, problem
    return Path(binary), settings.weights_dir / assets.TESSDATA_DIR, ""


REAL_BINARY, REAL_TESSDATA, WHY = _real()
real = pytest.mark.skipif(REAL_BINARY is None, reason=WHY)
FIXTURES = Path(__file__).parent / "fixtures"


def _first_line_png(name):
    truth = json.loads((FIXTURES / f"{name}-invoice-clean.gt.json").read_text("utf-8"))
    with Image.open(FIXTURES / f"{name}-invoice-clean.png") as image:
        return line_png(image, tuple(truth["lines"][0]["box"]))


@real
def test_real_english_line():
    reading = read_line(REAL_BINARY, REAL_TESSDATA, _first_line_png("en"), 60)
    assert reading.error is None
    assert reading.text == "Invoice: Northwind Office Supply Ltd"
    assert all(0 <= s <= 1 for s in reading.scores)


@real
def test_real_arabic_line():
    reading = read_line(REAL_BINARY, REAL_TESSDATA, _first_line_png("ar"), 60)
    truth = json.loads((FIXTURES / "ar-invoice-clean.gt.json").read_text("utf-8"))
    assert reading.error is None
    # Raw-line mode keeps the colon on the first word, as printed.
    assert reading.text.split(" ")[0] == truth["lines"][0]["text"].split(" ")[0] == "فاتورة:"


@real
def test_real_version():
    assert tesseract_version(REAL_BINARY).startswith("5.")


def test_a_word_holding_a_form_feed_or_line_separator_stays_in_its_row():
    names = "level page_num block_num par_num line_num word_num left top width height conf text"
    words = ("a" + chr(12) + "b", "c" + chr(0x2028) + "d")
    rows = [
        chr(9).join(["5", "1", "1", "1", "1", str(i), "0", "0", "9", "9", "90", w])
        for i, w in enumerate(words, 1)
    ]
    data = (chr(13) + chr(10)).join([chr(9).join(names.split()), *rows, ""])
    assert parse_tsv(data.encode("utf-8")) == Reading(" ".join(words), (0.9, 0.9))
