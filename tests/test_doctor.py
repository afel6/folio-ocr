"""Tests for the environment checks of `folio doctor`."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from folio_ocr import doctor
from folio_ocr.assets import MANIFEST, Asset
from folio_ocr.config import Settings
from folio_ocr.errors import PdfReaderError

DATA = b"data"
ASSET = Asset(
    "tessdata/x.traineddata",
    "https://assets.example/x",
    hashlib.sha256(DATA).hexdigest(),
    4,
    "Apache-2.0",
)
TESS = Path(os.path.abspath(os.sep)) / "opt" / "tess" / "tesseract"
NO_TESSERACT = {"PATH": ""}
REAL_TESSERACT_STATUS = doctor.ocr_setup.tesseract_status
REAL_ONNXTR_STATUS = doctor.ocr_setup.onnxtr_status


@pytest.fixture(autouse=True)
def readers_ok(monkeypatch):
    """Both OCR readers report ok unless a test sets them, so no model file is needed."""
    monkeypatch.setattr(doctor.ocr_setup, "onnxtr_status", lambda *_a, **_k: "ok")
    monkeypatch.setattr(doctor.ocr_setup, "tesseract_status", lambda *_a, **_k: "ok")


def make_settings(tmp_path: Path, **kw) -> Settings:
    return Settings(weights_dir=tmp_path / "assets", **kw)


def put_asset(tmp_path: Path, content: bytes = DATA) -> None:
    target = tmp_path / "assets" / ASSET.name
    target.parent.mkdir(parents=True)
    target.write_bytes(content)


def done(stdout: bytes = b"", stderr: bytes = b"", code: int = 0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def by_name(result):
    return {c.name: c for c in result}


def go(settings, **kw):
    kw.setdefault("environ", NO_TESSERACT)
    kw.setdefault("assets", (ASSET,))
    return doctor.checks(settings, **kw)


def test_all_good_exit_zero_and_info_does_not_fail(tmp_path):
    put_asset(tmp_path)
    s = make_settings(tmp_path)
    lines: list[str] = []
    code = doctor.run(s, lines.append, environ=NO_TESSERACT, assets=(ASSET,))
    assert code == 0
    assert not any(line.startswith("fail") for line in lines)
    assert any(line.startswith("info") and "tesseract" in line for line in lines)


def test_line_format_and_order(tmp_path):
    put_asset(tmp_path)
    s = make_settings(
        tmp_path, model_url="http://127.0.0.1:11434/v1", model_name="m", tesseract_path=TESS
    )
    lines: list[str] = []
    runner = lambda *a, **k: done(b"tesseract 5.4.0\n")
    code = doctor.run(
        s,
        lines.append,
        version_info=(3, 13, 1),
        environ=NO_TESSERACT,
        runner=runner,
        assets=(ASSET,),
        pdf_reader=lambda: doctor.Check("pdf reader", "ok", "starts with its limits"),
    )
    import pypdfium2

    assert code == 0
    assert lines == [
        "ok   python: 3.13.1",
        f"ok   pdfium: {pypdfium2.PDFIUM_INFO.version}",
        "ok   pdf reader: starts with its limits",
        f"info assets folder: {s.weights_dir}",
        "ok   asset tessdata/x.traineddata: verified (sha256)",
        f"ok   tesseract: 5.4.0 at {TESS}",
        "ok   reader onnxtr: ok",
        "ok   reader tesseract: ok",
        "info model server: configured: model m at host 127.0.0.1",
    ]


def test_pdf_reader_line_when_the_reader_starts(monkeypatch):
    monkeypatch.setattr(doctor.pdf_client, "self_check", lambda: None)
    check = doctor._pdf_reader()
    assert (check.name, check.status) == ("pdf reader", "ok")
    assert f"{doctor.ingest.PDF_MEMORY_MB} MB" in check.detail


def test_pdf_reader_line_when_the_reader_cannot_start(monkeypatch):
    def refuse():
        raise PdfReaderError("pdf", "no job object")

    monkeypatch.setattr(doctor.pdf_client, "self_check", refuse)
    check = doctor._pdf_reader()
    assert (check.name, check.status) == ("pdf reader", "fail")
    assert check.detail == "no job object; PDFs and images cannot be read"


def test_wrong_content_asset_fails(tmp_path):
    put_asset(tmp_path, b"datb")
    lines: list[str] = []
    code = doctor.run(make_settings(tmp_path), lines.append, environ=NO_TESSERACT, assets=(ASSET,))
    assert code == 1
    assert "fail asset tessdata/x.traineddata: sha256 mismatch; run folio setup" in lines


def test_an_asset_that_cannot_be_read_points_to_access_rights(tmp_path):
    (tmp_path / "assets" / ASSET.name).mkdir(parents=True)
    check = by_name(go(make_settings(tmp_path)))["asset tessdata/x.traineddata"]
    assert (check.status, check.detail) == (
        "fail",
        "cannot read; check the access rights of the assets folder",
    )


def test_an_asset_of_the_wrong_size_points_to_setup(tmp_path):
    put_asset(tmp_path, b"dat")
    check = by_name(go(make_settings(tmp_path)))["asset tessdata/x.traineddata"]
    assert (check.status, check.detail) == ("fail", "wrong size; run folio setup")


def test_real_manifest_empty_folder(tmp_path):
    lines: list[str] = []
    code = doctor.run(make_settings(tmp_path), lines.append, environ=NO_TESSERACT)
    assert code == 1
    for a in MANIFEST:
        assert f"fail asset {a.name}: missing; run folio setup" in lines


def test_doctor_lists_the_orientation_data_in_manifest_order(tmp_path):
    lines: list[str] = []
    doctor.run(make_settings(tmp_path), lines.append, environ=NO_TESSERACT)
    assert [line.split(":")[0] for line in lines if line.startswith("fail asset ")] == [
        "fail asset tessdata/ara.traineddata",
        "fail asset tessdata/eng.traineddata",
        "fail asset tessdata/osd.traineddata",
        "fail asset onnxtr/rep_fast_base-1b89ebf9.onnx",
        "fail asset onnxtr/crnn_vgg16_bn-743599aa.onnx",
    ]
    assert "fail asset tessdata/osd.traineddata: missing; run folio setup" in lines


@pytest.mark.parametrize(
    ("version", "status", "detail"),
    [
        ((3, 11, 4), "fail", "3.11.4; folio-ocr needs Python 3.12 or newer"),
        ((3, 12, 0), "ok", "3.12.0"),
    ],
)
def test_python_version(tmp_path, version, status, detail):
    c = by_name(go(make_settings(tmp_path), version_info=version))["python"]
    assert (c.status, c.detail) == (status, detail)
    code = doctor.run(
        make_settings(tmp_path),
        lambda _l: None,
        version_info=version,
        environ=NO_TESSERACT,
        assets=(),
    )
    assert code == (1 if status == "fail" else 0)


def test_pdfium_import_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    c = by_name(go(make_settings(tmp_path)))["pdfium"]
    assert (c.status, c.detail) == ("fail", "cannot load pdfium")
    assert (
        doctor.run(make_settings(tmp_path), lambda _l: None, environ=NO_TESSERACT, assets=()) == 1
    )


def test_pdfium_os_error(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pypdfium2":
            raise OSError("no library")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert by_name(go(make_settings(tmp_path)))["pdfium"].status == "fail"


def tess_check(tmp_path, runner, **kw):
    s = make_settings(tmp_path, tesseract_path=TESS, **kw)
    return by_name(go(s, runner=runner))["tesseract"]


def test_tesseract_stdout_version_and_call_shape(tmp_path):
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return done(b"tesseract 5.4.0.20240606\n leptonica-1.84.1\n")

    c = tess_check(tmp_path, runner)
    assert (c.status, c.detail) == ("ok", f"5.4.0.20240606 at {TESS}")
    ((args, kwargs),) = calls
    assert args == ([str(TESS), "--version"],)
    assert kwargs["timeout"] == 10
    assert not kwargs.get("shell")


def test_tesseract_version_on_stderr(tmp_path):
    c = tess_check(tmp_path, lambda *a, **k: done(b"", b"tesseract 4.1.1\n"))
    assert (c.status, c.detail) == ("ok", f"4.1.1 at {TESS}")


def test_tesseract_configured_path_wins(tmp_path):
    conf = Path(os.path.abspath(os.sep)) / "conf" / "tesseract"
    seen = []

    def runner(cmd, **k):
        seen.append(cmd)
        return done(b"tesseract v5.3.0\n")

    s = make_settings(tmp_path, tesseract_path=conf)
    c = by_name(go(s, runner=runner))["tesseract"]
    assert seen == [[str(conf), "--version"]]
    assert c.detail == f"5.3.0 at {conf}"


def raiser(exc):
    def runner(*a, **k):
        raise exc

    return runner


@pytest.mark.parametrize(
    ("runner", "detail"),
    [
        (raiser(FileNotFoundError()), f"cannot run {TESS}"),
        (raiser(OSError(193, "bad exe")), f"cannot run {TESS}"),
        (raiser(subprocess.TimeoutExpired("t", 10)), f"did not answer within 10 s: {TESS}"),
        (lambda *a, **k: done(b"tesseract 5.0", code=1), f"exited with code 1: {TESS}"),
        (lambda *a, **k: done(b"\xff garbage"), f"unrecognised version output: {TESS}"),
    ],
)
def test_tesseract_failures(tmp_path, runner, detail):
    c = tess_check(tmp_path, runner)
    assert (c.status, c.detail) == ("fail", detail)


def test_tesseract_not_found(tmp_path):
    c = by_name(go(make_settings(tmp_path)))["tesseract"]
    assert c.status == "info"
    assert c.detail == (
        "not found (optional: install Tesseract or set FOLIO_TESSERACT_PATH "
        "to read Arabic without a model server)"
    )


def test_model_not_configured(tmp_path):
    c = by_name(go(make_settings(tmp_path)))["model server"]
    assert (c.status, c.detail) == ("info", "not configured (set FOLIO_MODEL_URL to use one)")


def test_model_remote_never_prints_secrets(tmp_path):
    s = make_settings(
        tmp_path,
        model_url="https://user:SENTINEL-77@models.example/v1",
        model_api_key="sk-SENTINEL-77",
    )
    lines: list[str] = []
    doctor.run(s, lines.append, environ=NO_TESSERACT, assets=(ASSET,))
    text = "\n".join(lines)
    assert "models.example" in text
    assert "(leaves this machine)" in text
    assert "SENTINEL-77" not in text
    assert "(FOLIO_MODEL_NAME not set)" in text


def test_model_loopback_stays_local(tmp_path):
    s = make_settings(
        tmp_path, model_url="http://127.0.0.1:11434/v1", model_name="m", tesseract_path=TESS
    )
    c = by_name(go(s))["model server"]
    assert "leaves this machine" not in c.detail


@pytest.mark.parametrize("windows", [True, False])
def test_tesseract_found_on_an_absolute_path_entry(tmp_path, windows):
    name = "tesseract.exe" if windows else "tesseract"
    (tmp_path / "bin").mkdir()
    program = tmp_path / "bin" / name
    program.write_bytes(b"x")
    program.chmod(0o755)
    runner = lambda *a, **k: done(b"tesseract 5.4.0")
    c = by_name(go(make_settings(tmp_path), environ={"PATH": str(tmp_path / "bin")}, runner=runner))
    detail = c["tesseract"].detail
    if windows == (os.name == "nt"):
        assert detail == f"5.4.0 at {program}"
    else:
        assert detail.startswith("not found")


# reader lines


def reader_lines(tmp_path, monkeypatch, onnxtr, tesseract):
    monkeypatch.setattr(doctor.ocr_setup, "onnxtr_status", lambda *_a, **_k: onnxtr)
    monkeypatch.setattr(doctor.ocr_setup, "tesseract_status", lambda *_a, **_k: tesseract)
    put_asset(tmp_path)
    lines: list[str] = []
    code = doctor.run(make_settings(tmp_path), lines.append, environ=NO_TESSERACT, assets=(ASSET,))
    return code, [ln for ln in lines if " reader " in ln]


@pytest.mark.parametrize(
    ("onnxtr", "start", "code"),
    [
        ("missing", "fail reader onnxtr: missing", 1),
        ("bad_data", "fail reader onnxtr: bad_data", 1),
        ("uncalibrated", "info reader onnxtr: uncalibrated (no frozen confidence", 0),
        ("ok", "ok   reader onnxtr: ok", 0),
    ],
)
def test_onnxtr_reader_line(tmp_path, monkeypatch, onnxtr, start, code):
    got_code, lines = reader_lines(tmp_path, monkeypatch, onnxtr, "ok")
    assert lines[0].startswith(start)
    assert got_code == code


@pytest.mark.parametrize(
    ("tesseract", "start", "code"),
    [
        ("missing", "info reader tesseract: missing", 0),
        ("bad_data", "fail reader tesseract: bad_data", 1),
        ("uncalibrated", "info reader tesseract: uncalibrated (no frozen confidence", 0),
        ("ok", "ok   reader tesseract: ok", 0),
    ],
)
def test_tesseract_reader_line(tmp_path, monkeypatch, tesseract, start, code):
    got_code, lines = reader_lines(tmp_path, monkeypatch, "ok", tesseract)
    assert lines[1].startswith(start)
    assert got_code == code


BAD_DATA = "bad_data (its files or program cannot be used; see the lines above)"


def test_a_bad_data_reader_line_points_to_the_lines_above(tmp_path, monkeypatch):
    _, lines = reader_lines(tmp_path, monkeypatch, "bad_data", "bad_data")
    assert lines == [f"fail reader onnxtr: {BAD_DATA}", f"fail reader tesseract: {BAD_DATA}"]


# one version question per run


def real_tesseract_reader(monkeypatch):
    """Let the Tesseract reader check really run, with every pinned file counted as present."""
    monkeypatch.setattr(doctor.ocr_setup, "tesseract_status", REAL_TESSERACT_STATUS)
    monkeypatch.setattr(doctor.ocr_setup, "files_problem", lambda assets, folder: None)


def counting(result):
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        if isinstance(result, BaseException):
            raise result
        return result

    runner.calls = calls
    return runner


def test_doctor_asks_the_tesseract_program_for_its_version_once(tmp_path, monkeypatch):
    from fake_tesseract_install import install_fake_tesseract

    real_tesseract_reader(monkeypatch)
    name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    program = install_fake_tesseract(tmp_path / "bin" / name)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_TESSERACT_LOG", str(log))
    s = make_settings(tmp_path, tesseract_path=program)
    lines: list[str] = []
    doctor.run(s, lines.append, environ=NO_TESSERACT, assets=(ASSET,))
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["argv"] for e in entries] == [["--version"]]
    assert f"ok   tesseract: 5.4.0.20240606 at {program}" in lines
    reader = next(ln for ln in lines if " reader tesseract" in ln)
    assert reader.startswith(("ok   reader tesseract: ok", "info reader tesseract: uncalibrated"))


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (subprocess.TimeoutExpired("t", 10), f"did not answer within 10 s: {TESS}"),
        (OSError(193, "bad exe"), f"cannot run {TESS}"),
    ],
)
def test_a_tesseract_that_cannot_run_is_started_once(tmp_path, monkeypatch, error, detail):
    real_tesseract_reader(monkeypatch)
    runner = counting(error)
    result = by_name(go(make_settings(tmp_path, tesseract_path=TESS), runner=runner))
    assert len(runner.calls) == 1
    assert (result["tesseract"].status, result["tesseract"].detail) == ("fail", detail)
    assert (result["reader tesseract"].status, result["reader tesseract"].detail) == (
        "info",
        "missing",
    )


def test_the_reader_line_reads_the_same_version_answer(tmp_path, monkeypatch):
    real_tesseract_reader(monkeypatch)
    runner = counting(done(b"tesseract 5\n"))
    result = by_name(go(make_settings(tmp_path, tesseract_path=TESS), runner=runner))
    assert len(runner.calls) == 1
    assert (result["tesseract"].status, result["tesseract"].detail) == ("ok", f"5 at {TESS}")
    assert result["reader tesseract"].status == "fail"
    assert result["reader tesseract"].detail.startswith("bad_data")


def onnxtr_setups_asked(monkeypatch) -> list:
    """Run the real OnnxTR reader check and record the Tesseract version it asks a setup for."""
    real_tesseract_reader(monkeypatch)
    monkeypatch.setattr(doctor.ocr_setup, "onnxtr_status", REAL_ONNXTR_STATUS)
    asked: list = []
    real_setup = doctor.ocr_setup.onnxtr_setup
    monkeypatch.setattr(
        doctor.ocr_setup, "onnxtr_setup", lambda found: asked.append(found) or real_setup(found)
    )
    return asked


@pytest.mark.parametrize(("tesseract", "version"), [(TESS, "5.4"), (None, None)])
def test_the_onnxtr_reader_line_asks_for_the_setup_of_the_found_tesseract(
    tmp_path, monkeypatch, tesseract, version
):
    asked = onnxtr_setups_asked(monkeypatch)
    runner = counting(done(b"tesseract 5.4.0\n"))
    go(make_settings(tmp_path, tesseract_path=tesseract), runner=runner)
    assert asked == [version]
    assert len(runner.calls) == (1 if tesseract else 0)  # one --version for every line


def test_the_onnxtr_reader_line_uses_the_tesseract_found_on_path(tmp_path, monkeypatch):
    asked = onnxtr_setups_asked(monkeypatch)
    (tmp_path / "bin").mkdir()
    program = tmp_path / "bin" / ("tesseract.exe" if os.name == "nt" else "tesseract")
    program.write_bytes(b"x")
    program.chmod(0o755)
    runner = counting(done(b"tesseract 5.4.0\n"))
    go(make_settings(tmp_path), environ={"PATH": str(tmp_path / "bin")}, runner=runner)
    assert asked == ["5.4"] and runner.calls == [([str(program), "--version"],)]
