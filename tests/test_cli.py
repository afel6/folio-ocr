"""Command line behaviour, run in-process with a stand-in for the page reader."""

import errno
import functools
import hashlib
import json
import os
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from sample_files import OVERSIZED_IMAGE_CASES as BIG_IMAGE_CASES

from folio_ocr import assets, cli, doctor, model_health, pipeline
from folio_ocr.document import IDENTITY, Document, Engine, Line, Page, Source
from folio_ocr.errors import InputError, SetupNeededError, WeightsError
from folio_ocr.export import to_json, to_txt
from folio_ocr.flags import NEEDS_MODEL_SERVER

HINT = "hint: use --lang en for English-only pages, or install Tesseract, or set FOLIO_MODEL_URL"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def line(text="hello", flags=(), status="auto_accepted", n=1, reader="text_layer"):
    return Line(
        f"p1-l{n}", (0, 0, 1, 1), text, "ltr", "latin", reader, None, "none",
        tuple(flags), status, text,
    )  # fmt: skip


def page(lines=(), origin="text_layer", status="read", index=1, error=None):
    return Page(
        index, 10, 10, 300, 0.0, IDENTITY, origin, "ltr", status, (), (), tuple(lines), error=error
    )  # fmt: skip


def doc(filename="a.pdf", pages=None):
    pages = [page([line()])] if pages is None else pages
    engine = Engine("none", None, "none", None, None, "untested", None)
    return Document(
        "id", Source(filename, "0" * 64, "pdf", len(pages)), "en", engine, tuple(pages),
        datetime(2026, 1, 1, tzinfo=UTC), False,
    )  # fmt: skip


@pytest.fixture
def reader(monkeypatch):
    table, calls, timeouts = {}, [], []

    def fake(
        path, *, languages, weights_dir=None, tesseract_path=None, line_timeout_s=60.0, model=None
    ):
        calls.append((path, languages, weights_dir, tesseract_path))
        timeouts.append(line_timeout_s)
        result = table[path.name]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(pipeline, "read_path", fake)
    return SimpleNamespace(table=table, calls=calls, timeouts=timeouts)


def test_read_passes_the_line_time_limit(reader, capsys):
    reader.table["a.pdf"] = doc()
    code = cli.main(["read", "a.pdf"], {"FOLIO_LINE_TIMEOUT_S": "7.5"})
    capsys.readouterr()
    assert (code, reader.timeouts) == (0, [7.5])


def run(capsys, *argv, environ=None):
    code = cli.main(list(argv), {} if environ is None else environ)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def one_error(err):
    assert err.endswith("\n") and err.count("\n") == 1 and err.startswith("error: "), err
    return err


def test_txt_to_stdout_one_file(reader, capsys):
    reader.table["a.pdf"] = d = doc()
    code, out, err = run(capsys, "read", "a.pdf")
    assert (code, out) == (0, to_txt(d))
    assert err == "a.pdf: 1 lines found, 1 read, 0 need review, 0 unread pages\n"
    assert reader.calls[0][1] == "en+ar"


def test_txt_to_stdout_two_files_have_headers(reader, capsys):
    reader.table["a.pdf"], reader.table["b.pdf"] = doc("a.pdf"), doc("b.pdf", [page([line("x")])])
    code, out, _ = run(capsys, "read", "a.pdf", "b.pdf", "--lang", "en")
    assert code == 0
    assert out == f"==> a.pdf <==\n{to_txt(reader.table['a.pdf'])}==> b.pdf <==\nx\n"


@pytest.mark.parametrize("names", [["a.pdf"], ["a.pdf", "b.pdf"]])
def test_json_to_stdout_one_line_per_document(reader, capsys, names):
    for n in names:
        reader.table[n] = doc(n)
    code, out, _ = run(capsys, "read", *names, "--format", "json")
    lines = out.split("\n")
    assert code == 0 and lines[-1] == "" and len(lines) == len(names) + 1
    assert [json.loads(x)["source"]["filename"] for x in lines[:-1]] == names
    assert lines[0] == to_json(reader.table["a.pdf"], indent=None)


@pytest.mark.parametrize("fmt", ["txt", "json", "docx"])
def test_out_writes_file_and_summary(reader, capsys, tmp_path, fmt):
    reader.table["a.pdf"] = doc("a.pdf")
    out = tmp_path / "out"
    code, so, err = run(capsys, "read", "a.pdf", "--out", str(out), "--format", fmt)
    target = out / f"a.{fmt}"
    assert (code, so) == (0, "") and target.is_file()
    expected = "a.pdf: 1 lines found, 1 read, 0 need review, 0 unread pages"
    assert err == f"{expected} (saved to {target})\n"
    data = target.read_bytes()
    if fmt == "json":
        assert data.decode() == to_json(reader.table["a.pdf"]) + "\n" and b"\n  " in data
    if fmt == "txt":
        assert data == b"hello\n"
    if fmt == "docx":
        assert data[:2] == b"PK"
    assert [p.name for p in out.iterdir()] == [target.name]


def test_docx_without_out_is_usage_error(reader, capsys):
    code, out, err = run(capsys, "read", "a.pdf", "--format", "docx")
    assert code == 1 and out == "" and reader.calls == []
    assert "--format docx needs --out DIR" in one_error(err)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["read"],
        ["read", "a.pdf", "--lang", "xx"],
        ["read", "a.pdf", "--format", "pdf"],
        ["read", "a.pdf", "--bogus"],
    ],
)
def test_usage_errors_are_one_line(reader, capsys, argv):
    code, out, err = run(capsys, *argv)
    assert code == 1 and out == ""
    one_error(err)


def test_help_exits_zero(capsys):
    code, out, _ = run(capsys, "--help")
    assert code == 0 and "read" in out


def test_input_is_never_overwritten(reader, capsys, tmp_path):
    src = tmp_path / "page.txt"
    src.write_bytes(b"original")
    reader.table["page.txt"] = doc("page.txt")
    code, _, err = run(capsys, "read", str(src), "--out", str(tmp_path))
    assert code == 0 and src.read_bytes() == b"original"
    assert (tmp_path / "page-2.txt").read_bytes() == b"hello\n"
    assert "page-2.txt" in err


def test_same_stem_gets_a_number(reader, capsys, tmp_path):
    reader.table["a.pdf"] = doc("scan.pdf", [page([line("one")])])
    reader.table["b.pdf"] = doc("scan.pdf", [page([line("two")])])
    out = tmp_path / "o"
    assert run(capsys, "read", "a.pdf", "b.pdf", "--out", str(out))[0] == 0
    assert (out / "scan.txt").read_text() == "one\n"
    assert (out / "scan-2.txt").read_text() == "two\n"


def test_earlier_run_output_is_overwritten(reader, capsys, tmp_path):
    reader.table["a.pdf"] = doc("scan.pdf")
    (tmp_path / "scan.txt").write_text("old")
    run(capsys, "read", "a.pdf", "--out", str(tmp_path))
    assert (tmp_path / "scan.txt").read_text() == "hello\n"


def test_reserved_name_is_prefixed(reader, capsys, tmp_path):
    reader.table["a.pdf"] = doc("CON.pdf")
    run(capsys, "read", "a.pdf", "--out", str(tmp_path))
    assert (tmp_path / "_CON.txt").is_file()


def test_failing_file_does_not_stop_the_others(reader, capsys):
    reader.table["a.pdf"], reader.table["c.pdf"] = doc("a.pdf"), doc("c.pdf")
    reader.table["b.pdf"] = InputError("not a PDF or image")
    code, out, err = run(capsys, "read", "a.pdf", "b.pdf", "c.pdf")
    assert code == 1 and "==> c.pdf <==" in out
    assert err.splitlines().count("error: b.pdf: not a PDF or image") == 1


def test_missing_model_files_give_one_setup_hint(reader, capsys):
    reader.table["a.pdf"] = SetupNeededError("the OCR model files are missing")
    reader.table["b.pdf"] = SetupNeededError("the OCR model files are missing")
    code, _, err = run(capsys, "read", "a.pdf", "b.pdf")
    assert code == 1
    assert err.splitlines() == [
        "error: a.pdf: the OCR model files are missing",
        "error: b.pdf: the OCR model files are missing",
        cli.SETUP_HINT,
    ]


UNREADABLE = "the OCR model files cannot be read; check the access rights of the assets folder"


def test_unreadable_model_files_give_no_setup_hint(reader, capsys):
    reader.table["a.pdf"] = WeightsError(UNREADABLE)
    code, _, err = run(capsys, "read", "a.pdf")
    assert code == 1
    assert err.splitlines() == [f"error: a.pdf: {UNREADABLE}"]


def test_a_permission_error_on_the_input_gives_no_setup_hint(reader, capsys):
    reader.table["a.pdf"] = PermissionError(errno.EACCES, "x")
    code, _, err = run(capsys, "read", "a.pdf")
    assert code == 1
    assert err.splitlines() == ["error: permission denied: a.pdf"]


IMAGE = Path(__file__).parent / "fixtures" / "en-invoice-clean.png"


def test_empty_weights_folder_gives_the_setup_hint(capsys, tmp_path):
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path), "PATH": ""}
    code, _, err = run(capsys, "read", "--lang", "en", str(IMAGE), environ=env)
    assert code == 1
    assert err.splitlines() == [
        f"error: {IMAGE}: the OCR model files are missing; run folio setup",
        cli.SETUP_HINT,
    ]


def test_unreadable_model_files_end_to_end_give_no_hint(capsys, tmp_path):
    for asset in assets.ONNXTR:
        (tmp_path / asset.name).mkdir(parents=True)
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path), "PATH": ""}
    code, _, err = run(capsys, "read", "--lang", "en", str(IMAGE), environ=env)
    assert code == 1
    assert err.splitlines() == [f"error: {IMAGE}: {UNREADABLE}"]


def test_input_error_gives_no_setup_hint(reader, capsys):
    reader.table["a.pdf"] = InputError("not a PDF or image")
    assert cli.SETUP_HINT not in run(capsys, "read", "a.pdf")[2]


def test_settings_reach_the_reader(reader, capsys, tmp_path):
    reader.table["a.pdf"] = doc()
    tess = tmp_path / "tesseract.exe"
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "w"), "FOLIO_TESSERACT_PATH": str(tess)}
    run(capsys, "read", "a.pdf", "--lang", "en", environ=env)
    assert reader.calls == [(Path("a.pdf"), "en", tmp_path / "w", tess)]


@pytest.mark.parametrize(
    ("exc", "text"),
    [
        (OSError(errno.ENOENT, "x"), "not found: a.pdf"),
        (FileNotFoundError(errno.ENOENT, "x"), "not found: a.pdf"),
        (OSError(errno.EISDIR, "x"), "is a folder, not a file: a.pdf"),
        (OSError(errno.EACCES, "x"), "permission denied: a.pdf"),
        (OSError(errno.EPERM, "x"), "permission denied: a.pdf"),
        (OSError(errno.ENOTDIR, "x"), "part of the path is not a folder: a.pdf"),
        (OSError(errno.ENOSPC, "x"), "disk full while writing: a.pdf"),
        (OSError(errno.ENAMETOOLONG, "x"), "name too long: a.pdf"),
        (OSError(errno.EIO, "x"), "cannot read or write: a.pdf"),
    ],
)
def test_os_errors_map_by_errno(reader, capsys, exc, text):
    reader.table["a.pdf"] = exc
    code, _, err = run(capsys, "read", "a.pdf")
    assert code == 1 and one_error(err) == f"error: {text}\n"


def test_real_missing_file(capsys, tmp_path):
    code, _, err = run(capsys, "read", str(tmp_path / "nope.pdf"))
    assert code == 1 and "error: not found: " in one_error(err)


def test_out_parent_missing(reader, capsys, tmp_path):
    code, _, err = run(capsys, "read", "a.pdf", "--out", str(tmp_path / "x" / "y"))
    assert code == 1 and f"error: folder not found: {tmp_path / 'x'}" in one_error(err)
    assert reader.calls == []


def test_out_is_a_file(reader, capsys, tmp_path):
    (tmp_path / "f").write_text("x")
    code, _, err = run(capsys, "read", "a.pdf", "--out", str(tmp_path / "f"))
    assert code == 1 and "error: not a folder: " in one_error(err)


def test_disk_full_while_writing_leaves_no_part_file(reader, capsys, tmp_path, monkeypatch):
    reader.table["a.pdf"] = doc()

    def full(*_a):
        raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(os, "replace", full)
    code, _, err = run(capsys, "read", "a.pdf", "--out", str(tmp_path))
    assert code == 1 and "error: disk full while writing: " in one_error(err)
    assert list(tmp_path.glob("*.part")) == [] and list(tmp_path.glob(".*")) == []


def test_permission_denied_on_temp_file(reader, capsys, tmp_path, monkeypatch):
    reader.table["a.pdf"] = doc()

    def denied(**_k):
        raise PermissionError(errno.EACCES, "no")

    monkeypatch.setattr(cli.tempfile, "NamedTemporaryFile", denied)
    code, _, err = run(capsys, "read", "a.pdf", "--out", str(tmp_path))
    assert code == 1 and "error: permission denied: " in one_error(err)


def ocr(flagged):
    flags = [[NEEDS_MODEL_SERVER] if f else [] for f in flagged]
    return page([line(flags=fl, n=i + 1) for i, fl in enumerate(flags)], "ocr")


@pytest.mark.parametrize(
    ("pages", "hinted"),
    [
        ([ocr([True, True])], True),
        ([page([line(flags=[NEEDS_MODEL_SERVER])])], False),
        ([ocr([])], False),
        ([ocr([True, False])], False),
    ],
)
def test_hint(reader, capsys, pages, hinted):
    reader.table["a.pdf"] = doc("a.pdf", pages)
    _, _, err = run(capsys, "read", "a.pdf")
    assert (HINT + "\n" in err) is hinted


def test_hint_is_written_once(reader, capsys):
    reader.table["a.pdf"] = doc("a.pdf", [ocr([True, True])])
    reader.table["b.pdf"] = doc("b.pdf", [ocr([True, True])])
    _, _, err = run(capsys, "read", "a.pdf", "b.pdf")
    assert err.count("hint:") == 1


def test_summary_counts(reader, capsys):
    pages = [
        page([line(), line(status="needs_review", n=2), line(status="needs_review", n=3)]),
        page([], "ocr", "unread", 2),
        page([], "ocr", "failed", 3),
    ]
    reader.table["a.pdf"] = doc("a.pdf", pages)
    _, _, err = run(capsys, "read", "a.pdf")
    assert err.startswith("a.pdf: 3 lines found, 3 read, 2 need review, 2 unread pages\n")


def test_summary_counts_lines_a_reader_read(reader, capsys):
    lines = [
        line("a", n=1, reader="text_layer"),
        line("b", n=2, reader="onnxtr"),
        line("c", n=3, reader="tesseract"),
        line("d", n=4, reader="model"),
        line("", n=5, reader="onnxtr"),
        line("", flags=(NEEDS_MODEL_SERVER,), status="needs_review", n=6, reader="none"),
        line("", flags=("not_read",), status="needs_review", n=7, reader="none"),
    ]
    reader.table["a.pdf"] = doc("a.pdf", [page(lines)])
    _, _, err = run(capsys, "read", "a.pdf")
    assert err.startswith("a.pdf: 7 lines found, 5 read, 2 need review, 0 unread pages\n")


def test_summary_when_no_reader_read_any_line(reader, capsys):
    lines = [
        line("", flags=(NEEDS_MODEL_SERVER,), status="needs_review", n=n, reader="none")
        for n in (1, 2, 3)
    ]
    reader.table["a.pdf"] = doc("a.pdf", [page(lines, "ocr")])
    _, _, err = run(capsys, "read", "a.pdf")
    assert err.startswith("a.pdf: 3 lines found, 0 read, 3 need review, 0 unread pages\n")


@pytest.mark.parametrize("cmd", ["read", "setup", "doctor"])
def test_config_error(capsys, cmd):
    argv = ["read", "a.pdf"] if cmd == "read" else [cmd]
    code, _, err = run(capsys, *argv, environ={"FOLIO_PORT": "abc"})
    assert code == 1 and one_error(err).startswith("error: FOLIO_PORT")


@pytest.mark.parametrize("argv", [["serve"], ["serve", "--allow-remote"]])
def test_serve_returns_the_service_exit_code(capsys, monkeypatch, argv):
    monkeypatch.setattr("folio_ocr.web.server.serve", lambda settings, environ: 7)
    code, _, err = run(capsys, *argv)
    assert (code, err) == (7, "")


BODY = b"fake asset"
ASSET = assets.Asset(
    "t/a.bin", "https://assets.example/a.bin", hashlib.sha256(BODY).hexdigest(), 10, "MIT"
)


def serve_with(monkeypatch, body):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    monkeypatch.setattr(assets, "MANIFEST", (ASSET,))
    monkeypatch.setattr(assets, "fetch", functools.partial(assets.fetch, transport=transport))


def test_setup_fetches_assets(capsys, monkeypatch, tmp_path):
    serve_with(monkeypatch, BODY)
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "w")}
    code, out, err = run(capsys, "setup", environ=env)
    assert code == 0 and out == "" and (tmp_path / "w" / "t" / "a.bin").read_bytes() == BODY
    assert "a.bin: downloading" in err and err.endswith(f"assets ready in {tmp_path / 'w'}\n")


def test_setup_digest_mismatch(capsys, monkeypatch, tmp_path):
    serve_with(monkeypatch, b"other body")
    code, _, err = run(capsys, "setup", environ={"FOLIO_WEIGHTS_DIR": str(tmp_path / "w")})
    assert code == 1 and "does not match" in err.splitlines()[-1]
    assert err.splitlines()[-1].startswith("error: ")


def test_setup_os_error(capsys, monkeypatch, tmp_path):
    def boom(*_a, **_k):
        raise OSError(errno.EACCES, "no")

    monkeypatch.setattr(assets, "fetch", boom)
    code, _, err = run(capsys, "setup", environ={"FOLIO_WEIGHTS_DIR": str(tmp_path / "w")})
    assert code == 1 and err == f"error: permission denied: {tmp_path / 'w'}\n"


def test_setup_fetches_the_whole_manifest_with_the_orientation_data(capsys, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        assets, "fetch", lambda items, directory, **_k: calls.append((tuple(items), directory))
    )
    weights = tmp_path / "w"
    code, _, err = run(capsys, "setup", environ={"FOLIO_WEIGHTS_DIR": str(weights)})
    assert code == 0 and err == f"assets ready in {weights}\n"
    assert calls == [(assets.MANIFEST, weights)]
    assert assets.TESSERACT_OSD in calls[0][0]


@pytest.mark.parametrize("rc", [0, 1])
def test_doctor_passes_exit_code_and_lines(capsys, monkeypatch, rc):
    seen = []

    def fake(settings, write, environ):
        seen.append(environ)
        write("check one")
        return rc

    monkeypatch.setattr(doctor, "run", fake)
    env = {"PATH": "x"}
    assert run(capsys, "doctor", environ=env) == (rc, "check one\n", "")
    assert seen == [env]


DOCTOR_ENV = {
    "FOLIO_MODEL_API_KEY": "sk-SENTINEL-55",
    "FOLIO_MODEL_URL": "https://u:SENTINEL-55@m.example/v1",
}


def test_doctor_without_assets_fails(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(model_health, "model_checks", lambda settings: [])
    env = {**DOCTOR_ENV, "FOLIO_WEIGHTS_DIR": str(tmp_path)}
    code, out, err = run(capsys, "doctor", environ=env)
    lines = out.splitlines()
    assert code == 1 and err == ""
    for asset in assets.MANIFEST:
        assert f"fail asset {asset.name}: missing; run folio setup" in lines
    assert (
        "info model server: configured: model (FOLIO_MODEL_NAME not set) at host "
        "m.example (leaves this machine)" in lines
    )
    assert "SENTINEL-55" not in out + err


def test_doctor_with_assets_passes(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(model_health, "model_checks", lambda settings: [])
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "a.bin").write_bytes(BODY)
    checked = functools.partial(doctor.run, assets=(ASSET,))
    monkeypatch.setattr(doctor, "run", checked)
    monkeypatch.setattr(doctor.ocr_setup, "onnxtr_status", lambda *_a, **_k: "uncalibrated")
    env = {**DOCTOR_ENV, "FOLIO_WEIGHTS_DIR": str(tmp_path)}
    code, out, err = run(capsys, "doctor", environ=env)
    assert (code, err) == (0, "")
    assert f"ok   asset {ASSET.name}: verified (sha256)" in out.splitlines()
    assert "SENTINEL-55" not in out + err


def plant_current_folder(monkeypatch, tmp_path):
    here = tmp_path / "here"
    (here / "elsewhere").mkdir(parents=True)
    for name in ("tesseract.exe", "tesseract"):
        (here / name).write_bytes(b"x")
    monkeypatch.chdir(here)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(here / "elsewhere")]))
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("FOLIO_TESSERACT_PATH", raising=False)
    return here


def test_doctor_never_runs_a_program_from_the_current_folder(capsys, monkeypatch, tmp_path):
    plant_current_folder(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "_pdfium", lambda: doctor.Check("pdfium", "ok", "x"))
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "w")}
    _, out, _ = run(capsys, "doctor", environ=env)
    assert "info tesseract: not found (optional: install Tesseract or set" in out
    assert str(tmp_path / "here") not in out


def test_read_never_passes_a_program_from_the_current_folder(reader, capsys, monkeypatch, tmp_path):
    plant_current_folder(monkeypatch, tmp_path)
    reader.table["a.pdf"] = doc()
    run(capsys, "read", "a.pdf", "--lang", "en", environ={"FOLIO_WEIGHTS_DIR": str(tmp_path / "w")})
    assert [c[3] for c in reader.calls] == [None]


def program_folder(tmp_path):
    folder = tmp_path / "bin"
    folder.mkdir()
    program = folder / ("tesseract.exe" if os.name == "nt" else "tesseract")
    program.write_bytes(b"x")
    program.chmod(0o755)
    return folder, program


def test_read_finds_the_program_through_the_environ_it_was_given(
    reader, capsys, monkeypatch, tmp_path
):
    folder, program = program_folder(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    reader.table["a.pdf"] = doc()
    env = {"PATH": str(folder), "FOLIO_WEIGHTS_DIR": str(tmp_path / "w")}
    run(capsys, "read", "a.pdf", "--lang", "en", environ=env)
    assert [c[3] for c in reader.calls] == [program]


def test_read_looks_for_the_program_once_per_run(reader, capsys, monkeypatch, tmp_path):
    asked = []

    def counting(settings, environ):
        asked.append(environ)

    monkeypatch.setattr(cli, "find_tesseract", counting)
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        reader.table[name] = doc(name)
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "w")}
    code, _, _ = run(capsys, "read", "a.pdf", "b.pdf", "c.pdf", environ=env)
    assert code == 0 and asked == [env] and len(reader.calls) == 3


def test_a_folder_that_cannot_be_searched_is_not_a_file_error(
    reader, capsys, monkeypatch, tmp_path
):
    folder, program = program_folder(tmp_path)
    real = Path.is_file

    def is_file(self):
        if self == program:
            raise PermissionError(errno.EACCES, "denied")
        return real(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    reader.table["a.pdf"] = doc()
    reader.table["b.pdf"] = doc("b.pdf")
    env = {"PATH": str(folder), "FOLIO_WEIGHTS_DIR": str(tmp_path / "w")}
    code, _, err = run(capsys, "read", "a.pdf", "b.pdf", "--lang", "en", environ=env)
    assert code == 0 and "permission denied" not in err and "error" not in err
    assert [c[3] for c in reader.calls] == [None, None]


def test_doctor_shows_the_program_found_through_its_environ(capsys, monkeypatch, tmp_path):
    folder, program = program_folder(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(
        doctor, "_tesseract_status", lambda path, runner: ("ok", f"5.4.0 at {path}")
    )
    monkeypatch.setattr(doctor.ocr_setup, "tesseract_status", lambda *_a, **_k: "ok")
    env = {"PATH": str(folder), "FOLIO_WEIGHTS_DIR": str(tmp_path / "w")}
    _, out, _ = run(capsys, "doctor", environ=env)
    assert f"ok   tesseract: 5.4.0 at {program}" in out.splitlines()


def test_read_refuses_a_relative_tesseract_path(reader, capsys, tmp_path):
    reader.table["a.pdf"] = doc()
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "w"), "FOLIO_TESSERACT_PATH": "tesseract.exe"}
    code, out, err = run(capsys, "read", "a.pdf", environ=env)
    assert code != 0 and out == "" and reader.calls == []
    assert "FOLIO_TESSERACT_PATH" in err and "tesseract.exe" not in err


def test_python_dash_m_runs_main(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["folio", "read", "--format", "docx", "x.pdf"])
    with pytest.raises(SystemExit) as ei:
        runpy.run_module("folio_ocr", run_name="__main__")
    assert ei.value.code == 1
    assert capsys.readouterr().err == "error: --format docx needs --out DIR\n"


def test_unexpected_exception_is_generic(reader, capsys):
    reader.table["a.pdf"] = RuntimeError("secret doc text")
    code, _, err = run(capsys, "read", "a.pdf")
    assert code == 1 and err == "error: unexpected failure (RuntimeError); please report it\n"


def test_keyboard_interrupt(reader, capsys):
    reader.table["a.pdf"] = KeyboardInterrupt()
    code, _, err = run(capsys, "read", "a.pdf")
    assert (code, err) == (130, "error: interrupted\n")


class _FailingBuffer:
    def __init__(self, code: int) -> None:
        self.code = code

    def write(self, data: bytes) -> int:
        raise OSError(self.code, "cannot write")

    def flush(self) -> None:
        pass


class _FailingStdout:
    def __init__(self, code: int) -> None:
        self.buffer = _FailingBuffer(code)

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return 99


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (errno.EPIPE, ""),  # pipe closed by the reader
        (errno.EINVAL, ""),  # the same on Windows
        (errno.ENOSPC, "error: disk full while writing: standard output\n"),
    ],
)
def test_stdout_failure_redirects_and_reports(reader, capsys, monkeypatch, code, message):
    reader.table["a.pdf"] = doc()
    calls = []
    monkeypatch.setattr(cli.os, "open", lambda path, flags: 7)
    monkeypatch.setattr(cli.os, "dup2", lambda a, b: calls.append((a, b)))
    monkeypatch.setattr(sys, "stdout", _FailingStdout(code))
    result = cli.main(["read", "a.pdf"], {})
    assert (result, capsys.readouterr().err, calls) == (1, message, [(7, 99)])


def stop_at_once(server, sockets=None):
    raise KeyboardInterrupt


def test_secrets_never_appear(reader, capsys, monkeypatch):
    reader.table["a.pdf"] = doc()
    monkeypatch.setattr(assets, "fetch", lambda *a, **k: None)
    monkeypatch.setattr(doctor, "run", lambda settings, write, environ, **_: 0)
    monkeypatch.setattr(uvicorn.Server, "run", stop_at_once)  # serve announces, then stops
    env = {
        "FOLIO_MODEL_API_KEY": "sk-SENTINEL-99",
        "FOLIO_MODEL_URL": "https://u:SENTINEL-99@m.example/v1",
        "FOLIO_MODEL_NAME": "m",
    }
    bad = {**env, "FOLIO_MODEL_URL": "http://u:SENTINEL-99@remote.example/"}
    cases = [(["read", "a.pdf"], env), (["setup"], env), (["doctor"], env),
             (["serve"], env), (["read", "a.pdf"], bad)]  # fmt: skip
    seen = []
    for argv, e in cases:
        _, out, err = run(capsys, *argv, environ=e)
        seen += [out, err]
    assert "error: FOLIO_MODEL_API_KEY" in seen[-1]
    assert not any("SENTINEL-99" in s for s in seen)


def test_shown_replaces_control_characters():
    assert cli._shown("a\x1b[31mb\u202ec\n") == "a?[31mb?c?"


def test_echoed_argument_is_cleaned(capsys):
    code, _, err = run(capsys, "read", "no\x1b[31m\u202e.pdf")
    assert code == 1 and "\x1b" not in err and "\u202e" not in err and "?" in one_error(err)


def test_main_defaults_to_process_arguments(reader, capsys, monkeypatch):
    reader.table["a.pdf"] = doc()
    monkeypatch.setattr("sys.argv", ["folio", "read", "a.pdf"])
    assert cli.main() == 0
    assert capsys.readouterr().out == "hello\n"
    assert isinstance(reader.calls[0][0], Path)


ERR = "Tesseract was not found at the configured path"
SUMMARY = "a.pdf: 2 lines found, 1 read, 1 need review, 1 unread pages"
HINT_DOCTOR = "hint: run folio doctor to check the readers"


def failed_page(error=ERR, index=2):
    miss = line(text="", reader="none", status="needs_review", flags=("not_read",), n=2)
    return page([miss], "ocr", "failed", index, error=error)


def broken_doc(filename="a.pdf", error=ERR):
    return doc(filename, [page([line()]), failed_page(error)])


def test_a_page_error_follows_the_summary(reader, capsys):
    reader.table["a.pdf"] = d = broken_doc()
    code, out, err = run(capsys, "read", "a.pdf")
    assert (code, out) == (1, to_txt(d))
    assert err.splitlines() == [SUMMARY, f"a.pdf: page 2: {ERR}", HINT_DOCTOR]


def test_a_page_error_with_out_still_saves_the_file(reader, capsys, tmp_path):
    reader.table["a.pdf"] = d = broken_doc()
    code, _, err = run(capsys, "read", "a.pdf", "--out", str(tmp_path))
    target = tmp_path / "a.txt"
    assert code == 1 and target.read_bytes() == to_txt(d).encode("utf-8")
    assert err.splitlines() == [
        f"{SUMMARY} (saved to {target})",
        f"a.pdf: page 2: {ERR}",
        HINT_DOCTOR,
    ]


def test_only_the_file_with_a_page_error_gets_a_page_line(reader, capsys):
    reader.table["a.pdf"] = doc()
    reader.table["b.pdf"] = doc("b.pdf", [failed_page(index=1)])
    code, _, err = run(capsys, "read", "a.pdf", "b.pdf")
    lines = err.splitlines()
    assert code == 1 and len(lines) == 4
    assert lines[2:] == [f"b.pdf: page 1: {ERR}", HINT_DOCTOR]


def test_the_doctor_hint_is_written_once_for_many_files(reader, capsys):
    reader.table["a.pdf"] = broken_doc()
    reader.table["b.pdf"] = doc(
        "b.pdf", [failed_page("first", 1), page([line()], index=2), failed_page("second", 3)]
    )
    code, _, err = run(capsys, "read", "a.pdf", "b.pdf")
    lines = err.splitlines()
    assert code == 1 and lines.count(HINT_DOCTOR) == 1 and lines[-1] == HINT_DOCTOR
    assert lines[-3:-1] == ["b.pdf: page 1: first", "b.pdf: page 3: second"]


def test_the_hints_come_in_a_fixed_order(reader, capsys):
    server = line(text="", reader="none", status="needs_review", flags=(NEEDS_MODEL_SERVER,))
    reader.table["a.pdf"] = doc("a.pdf", [page([server], "ocr"), failed_page()])
    reader.table["b.pdf"] = SetupNeededError("the OCR model files are missing; run folio setup")
    _, _, err = run(capsys, "read", "a.pdf", "b.pdf")
    assert err.splitlines()[-3:] == [HINT, cli.SETUP_HINT, HINT_DOCTOR]


@pytest.mark.parametrize("error", [None, ""])
def test_no_page_error_prints_nothing_and_exits_zero(reader, capsys, error):
    reader.table["a.pdf"] = doc(pages=[page([line()], error=error)])
    code, _, err = run(capsys, "read", "a.pdf")
    assert code == 0 and err.splitlines() == [
        "a.pdf: 1 lines found, 1 read, 0 need review, 0 unread pages"
    ]


def test_a_page_error_is_shown_without_control_characters(reader, capsys):
    reader.table["a.pdf"] = broken_doc(error="bad\x1bthing")
    _, _, err = run(capsys, "read", "a.pdf")
    assert "a.pdf: page 2: bad?thing" in err.splitlines()


def test_a_failed_save_still_reports_the_page_errors(reader, capsys, tmp_path, monkeypatch):
    reader.table["a.pdf"] = broken_doc()

    def full(*_a):
        raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(os, "replace", full)
    code, _, err = run(capsys, "read", "a.pdf", "--out", str(tmp_path))
    lines = err.splitlines()
    assert code == 1 and lines[0].startswith("error: disk full while writing: ")
    assert lines[1:] == [f"a.pdf: page 2: {ERR}", HINT_DOCTOR]


@pytest.mark.parametrize(("data", "line"), BIG_IMAGE_CASES)
def test_pdf_with_an_oversized_image_is_refused_with_one_line(capsys, tmp_path, data, line):
    path = tmp_path / "a.pdf"
    path.write_bytes(data)
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path), "PATH": ""}
    code, out, err = run(capsys, "read", "--lang", "en", str(path), environ=env)
    assert (code, out) == (1, "")
    assert err.splitlines() == [f"error: {path}: {line}"]
