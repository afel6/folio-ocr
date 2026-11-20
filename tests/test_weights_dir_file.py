"""A weights folder setting that points at a file is refused with one clear line."""

import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from folio_ocr import assets, cli, pipeline
from folio_ocr.errors import SetupNeededError, WeightsError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


@pytest.fixture
def a_file(tmp_path):
    path = tmp_path / "not-a-folder"
    path.write_text("x")
    return path


def line_for(path):
    return f"error: FOLIO_WEIGHTS_DIR points at a file, not a folder: {path}\n"


def run(capsys, argv, environ):
    code = cli.main(argv, environ)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


@pytest.fixture
def reader(monkeypatch):
    table = {}

    def fake(path, *, languages, **_kw):
        result = table[path.name]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(pipeline, "read_path", fake)
    return SimpleNamespace(table=table)


def test_read_with_the_real_pipeline_refuses_a_file_as_weights_folder(capsys, a_file):
    env = {"FOLIO_WEIGHTS_DIR": str(a_file), "PATH": ""}
    argv = ["read", "--lang", "en", str(FIXTURES / "en-invoice-clean.png")]
    assert run(capsys, argv, env) == (1, "", line_for(a_file))


def test_read_turns_a_weights_error_into_the_file_line(reader, capsys, a_file):
    reader.table["a.png"] = WeightsError(
        "the OCR model files cannot be read; check the access rights of the assets folder"
    )
    code, out, err = run(capsys, ["read", "a.png"], {"FOLIO_WEIGHTS_DIR": str(a_file)})
    assert (code, out, err) == (1, "", line_for(a_file))


def test_read_keeps_the_setup_message_for_a_real_folder(reader, capsys, tmp_path):
    reader.table["a.png"] = SetupNeededError("the OCR model files are missing; run folio setup")
    code, _, err = run(capsys, ["read", "a.png"], {"FOLIO_WEIGHTS_DIR": str(tmp_path)})
    expected = (
        "error: a.png: the OCR model files are missing; run folio setup\n"
        "hint: run folio setup to download the model files\n"
    )
    assert (code, err) == (1, expected)


def test_read_repeats_the_line_once_per_input(reader, capsys, a_file):
    for name in ("a.png", "b.png"):
        reader.table[name] = WeightsError("the OCR model files cannot be read; check access")
    code, _, err = run(capsys, ["read", "a.png", "b.png"], {"FOLIO_WEIGHTS_DIR": str(a_file)})
    assert (code, err) == (1, line_for(a_file) * 2)


def test_read_does_not_claim_a_file_when_a_parent_is_a_file(reader, capsys, a_file):
    reader.table["a.png"] = WeightsError("the OCR model files cannot be read; check access")
    code, _, err = run(capsys, ["read", "a.png"], {"FOLIO_WEIGHTS_DIR": str(a_file / "sub")})
    assert code == 1 and "points at a file" not in err


def test_read_of_a_text_layer_pdf_ignores_the_weights_setting(capsys, a_file):
    env = {"FOLIO_WEIGHTS_DIR": str(a_file), "PATH": ""}
    code, out, err = run(capsys, ["read", str(FIXTURES / "en-invoice.pdf")], env)
    assert code == 0 and out and "lines found" in err


@pytest.fixture
def no_network(monkeypatch):
    real = assets.fetch

    def handler(request):
        pytest.fail("setup must not reach the network")

    def guarded(*args, **kwargs):
        return real(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(assets, "fetch", guarded)


def test_setup_refuses_a_file_as_weights_folder(capsys, a_file, no_network):
    code, _, err = run(capsys, ["setup"], {"FOLIO_WEIGHTS_DIR": str(a_file)})
    assert (code, err) == (1, line_for(a_file))


def test_setup_does_not_claim_a_file_when_a_parent_is_a_file(capsys, a_file, no_network):
    code, _, err = run(capsys, ["setup"], {"FOLIO_WEIGHTS_DIR": str(a_file / "sub")})
    assert code == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert "points at a file" not in err
