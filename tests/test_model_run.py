"""Tests for the model-run command line: parsing, error mapping and the session builder."""

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(_P(__file__).resolve().parent))

from pathlib import Path
from types import SimpleNamespace

import accuracy_reads as ar
import heldout
import model_run
import model_run_common as common
import model_run_server as srv
import pytest
from fake_model_server import Answer, FakeModelServer
from model_run_support import FACTS, KEYS, builder, make_session, run

from folio_ocr import model_http
from folio_ocr.errors import ConfigError, FolioError

CONTRACT = ["contract", "--work", "w", "--out", "c.json"]


def failing(exc, monkeypatch):
    def boom(args, build):
        raise exc

    monkeypatch.setitem(model_run.COMMANDS, "contract", boom)


@pytest.mark.parametrize("exc,message", [
    (srv.ModelRunError("one"), "error: one"),
    (heldout.HeldoutError("two"), "error: two"),
    (FolioError("three"), "error: three"),
    (ConfigError("four"), "error: four"),
    (ValueError("five"), "error: five"),
    (model_http.ModelCallError("six"), "error: the model server did not answer: six"),
    (RuntimeError("secret detail"), "error: unexpected failure (RuntimeError)"),
])
def test_errors_are_one_line(capsys, monkeypatch, exc, message):
    failing(exc, monkeypatch)
    rc, out, err = run(capsys, CONTRACT)
    assert rc == 1 and err == message + "\n" and "Traceback" not in out + err


def test_os_error_names_the_file(capsys, monkeypatch):
    failing(PermissionError(13, "denied", "x.json"), monkeypatch)
    rc, _, err = run(capsys, CONTRACT)
    assert rc == 1 and err == "error: permission denied: x.json\n"


def test_errors_are_ascii(capsys, monkeypatch):
    failing(ValueError("café ع"), monkeypatch)
    _, _, err = run(capsys, CONTRACT)
    assert err == "error: caf? ?\n"


def test_usage_errors_exit_2(capsys):
    with pytest.raises(SystemExit) as stop:
        model_run.main(["contract", "--out", "c.json"], builder(make_session()))
    assert stop.value.code == 2
    capsys.readouterr()


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_jobs_must_be_at_least_one(capsys, jobs):
    rc, _, err = run(capsys, [*CONTRACT, "--jobs", jobs])
    assert rc == 1 and err == "error: --jobs must be at least 1\n"


def test_unwritable_results_folder_is_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FOLIO_MODEL_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("FOLIO_MODEL_NAME", "m:8b")
    monkeypatch.setattr(srv, "contract_checks", lambda config, **kw: {
        "checks": [], "passed": True, "facts": None})

    def deny(path, record):
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr(ar, "write_json", deny)
    rc, _, err = run(capsys, ["contract", "--work", tmp_path, "--out", tmp_path / "c.json"])
    assert rc == 1 and err.startswith("error: permission denied") and err.count("\n") == 1


# ---------------------------------------------------------------- build_session


class FakeModel:
    def __init__(self, config, post):
        self.config, self.post = config, post
        self.events = []

    def self_test(self):
        self.events.append("self_test")
        return SimpleNamespace(passed=self.ok, detail="the Arabic line differed")

    def fit(self):
        self.events.append("fit")

    def server(self):
        return self.info


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setenv("FOLIO_MODEL_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("FOLIO_MODEL_NAME", "m:8b")
    events = []
    state = {"ok": True, "info": object(), "model": None}

    def make(config, post):
        model = FakeModel(config, post)
        model.ok, model.info = state["ok"], state["info"]
        state["model"] = model
        return model

    monkeypatch.setattr(model_run, "ModelReader", make)
    monkeypatch.setattr(model_run, "find_tesseract", lambda settings: Path("tess.exe"))
    monkeypatch.setattr(model_run, "decided_version", lambda tess: "5.3.0")
    monkeypatch.setattr(model_run, "warm_up", lambda config: events.append("warm"))
    monkeypatch.setattr(model_run, "server_facts", lambda config, t: dict(FACTS))
    monkeypatch.setattr(model_run, "model_keys", lambda info, name, version: dict(KEYS))
    return state, events


def test_build_session_warms_checks_and_keys(tmp_path, wired):
    state, events = wired
    session = model_run.build_session(tmp_path)
    assert events == ["warm"] and state["model"].events == ["self_test", "fit"]
    assert session.keys == KEYS and session.facts == FACTS and session.tesseract_version == "5.3.0"
    assert session.model is state["model"] and session.tesseract == Path("tess.exe")
    assert callable(session.reader_factory)


def test_build_session_decides_the_tesseract_version_once(tmp_path, monkeypatch, wired):
    calls, keyed = [], []

    def decided(tess):
        calls.append(tess)
        return "5.3.0" if len(calls) == 1 else "9.9.9"

    def keys(info, name, version):
        keyed.append(version)
        return dict(KEYS)

    monkeypatch.setattr(model_run, "decided_version", decided)
    monkeypatch.setattr(model_run, "model_keys", keys)
    session = model_run.build_session(tmp_path)
    assert len(calls) == 1
    assert keyed == [session.tesseract_version]


def test_build_session_without_warm_up(tmp_path, wired):
    state, events = wired
    model_run.build_session(tmp_path, warm=False)
    assert events == [] and state["model"].events == []


@pytest.mark.parametrize("case,message", [
    ("no-url", "FOLIO_MODEL_URL is not set; set it to the model server's /v1 URL"),
    ("no-tess", "Tesseract was not found; set FOLIO_TESSERACT_PATH"),
    ("self-test", "the model self-test failed: the Arabic line differed"),
    ("no-info", "the model server gives no calibration key (version or digest missing)"),
])
def test_build_session_refusals(tmp_path, monkeypatch, wired, case, message):
    state, _ = wired
    if case == "no-url":
        monkeypatch.delenv("FOLIO_MODEL_URL")
    if case == "no-tess":
        monkeypatch.setattr(model_run, "find_tesseract", lambda settings: None)
    if case == "self-test":
        state["ok"] = False
    if case == "no-info":
        state["info"] = None
    with pytest.raises(srv.ModelRunError) as stop:
        model_run.build_session(tmp_path)
    assert str(stop.value) == message


def test_model_config_helper_needs_a_url(monkeypatch):
    monkeypatch.delenv("FOLIO_MODEL_URL", raising=False)
    with pytest.raises(srv.ModelRunError):
        common.model_config()


# ---------------------------------------------------------------- declared identity


def _non_ollama_env(monkeypatch, url, declare):
    monkeypatch.setenv("FOLIO_MODEL_URL", url)
    monkeypatch.setenv("FOLIO_MODEL_NAME", "acme-ocr-7b")
    if declare:
        monkeypatch.setenv("FOLIO_MODEL_DIGEST", "ab" * 32)
        monkeypatch.setenv("FOLIO_MODEL_SERVER", "vllm 0.9")
    monkeypatch.setattr(model_run, "find_tesseract", lambda settings: Path("tess.exe"))
    monkeypatch.setattr(model_run, "decided_version", lambda tess: "5.3.0")


def test_build_session_keys_a_declared_identity_on_a_server_without_identity_routes(
        tmp_path, monkeypatch):
    with FakeModelServer(Answer(status=404, body=b"{}")) as server:
        _non_ollama_env(monkeypatch, server.url, declare=True)
        session = model_run.build_session(tmp_path, warm=False)
    assert session.facts["identity"] == "declared"
    assert set(session.keys) == {"model-en", "model-ar"}
    assert all("|declared:vllm-0.9|" in key for key in session.keys.values())


def test_build_session_refuses_a_server_without_routes_or_declaration(tmp_path, monkeypatch):
    with FakeModelServer(Answer(status=404, body=b"{}")) as server:
        _non_ollama_env(monkeypatch, server.url, declare=False)
        with pytest.raises(srv.ModelRunError, match="^no calibration key can be formed: ") as stop:
            model_run.build_session(tmp_path, warm=False)
    assert "\n" not in str(stop.value)
