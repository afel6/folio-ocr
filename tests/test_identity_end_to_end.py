"""Where the model identity comes from and which thresholds apply, as `folio doctor` and
`GET /api/health` report them through their real entry points."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_model_routing import DIGEST, native_answers, server

from folio_ocr import cli, model_pages, model_setup
from folio_ocr.config import DeclaredIdentity, Settings
from folio_ocr.model_reader import ModelConfig, ModelReader
from folio_ocr.model_self_test_lines import LINES
from folio_ocr.web.app import create_app

HEX = "ABCDEF0123456789" * 4
NO_ROUTES = (
    "info model identity: none: the server reports no model identity; "
    "set FOLIO_MODEL_DIGEST and FOLIO_MODEL_SERVER to declare it"
)
DIFFERS = (
    "info model identity: none: FOLIO_MODEL_DIGEST or FOLIO_MODEL_SERVER "
    "differs from what the server reports"
)
TEXTS = {line.languages: line.text for line in LINES}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("FOLIO_")]:
        monkeypatch.delenv(name)


def env_for(tmp_path, url=None, **extra) -> dict:
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path / "w"), "PATH": "", **extra}
    if url is not None:
        env |= {"FOLIO_MODEL_URL": url, "FOLIO_MODEL_NAME": "m"}
    return env


def doctor(capsys, env) -> tuple[int, str, str]:
    code = cli.main(["doctor"], env)
    out = capsys.readouterr()
    return code, out.out, out.err


def declaration(version="vllm 0.11.2") -> dict:
    return {"FOLIO_MODEL_DIGEST": HEX, "FOLIO_MODEL_SERVER": version}


def tau_file(tmp_path, key) -> tuple[str, str]:
    entry = {
        "tau": 0.5,
        "method": "logprobs",
        "bad_lines": 200,
        "documents": 40,
        "dev_bound": 0.09,
        "dev_manifest_sha256": "b" * 64,
        "self_test": [
            {"line": "en", "text": TEXTS["en"], "score": 0.99},
            {"line": "ar", "text": TEXTS["ar"], "score": 0.99},
        ],
        "drift_tolerance": 0.02,
    }
    raw = json.dumps({"schema": "folio-taus/1", "entries": {key: entry}}).encode()
    path = tmp_path / "taus.json"
    path.write_bytes(raw)
    return str(path), hashlib.sha256(raw).hexdigest()


def declared_key() -> str:
    info = model_setup.ServerInfo("vllm", "0.11", HEX.lower(), declared=True)
    return model_setup.model_setup(info, "m", "logprobs", model_pages.TEXT_LAYER_PREPROCESSING).key


# doctor


def test_doctor_shows_a_declared_identity_with_a_lower_case_digest(tmp_path, capsys):
    with server() as fake:
        _, out, _ = doctor(capsys, env_for(tmp_path, fake.url, **declaration()))
    assert "info model identity: declared vllm 0.11, digest abcdef012345" in out.splitlines()


def test_doctor_shows_an_identity_the_server_reports(tmp_path, capsys):
    with server(native=native_answers()) as fake:
        _, out, _ = doctor(capsys, env_for(tmp_path, fake.url))
    expected = f"info model identity: server ollama 0.40, digest {DIGEST[:12]}"
    assert expected in out.splitlines()


def test_doctor_says_why_there_is_no_identity_when_the_server_reports_none(tmp_path, capsys):
    with server() as fake:
        _, out, _ = doctor(capsys, env_for(tmp_path, fake.url))
    assert NO_ROUTES in out.splitlines()


def test_doctor_refuses_a_declaration_that_differs_from_the_server(tmp_path, capsys):
    other = {"FOLIO_MODEL_DIGEST": "d" * 64, "FOLIO_MODEL_SERVER": "ollama 0.40.16"}
    with server(native=native_answers()) as fake:
        _, out, _ = doctor(capsys, env_for(tmp_path, fake.url, **other))
    assert DIFFERS in out.splitlines()


@pytest.mark.parametrize(
    ("given", "message"),
    [
        (
            {"FOLIO_MODEL_DIGEST": HEX},
            "FOLIO_MODEL_SERVER must be set too when FOLIO_MODEL_DIGEST is set",
        ),
        (
            {"FOLIO_MODEL_SERVER": "vllm 0.11.2"},
            "FOLIO_MODEL_DIGEST must be set too when FOLIO_MODEL_SERVER is set",
        ),
    ],
    ids=["digest-only", "server-only"],
)
def test_doctor_refuses_a_one_sided_declaration(tmp_path, capsys, given, message):
    code, out, err = doctor(capsys, env_for(tmp_path, "http://127.0.0.1:9/v1", **given))
    assert (code, out, err) == (1, "", f"error: {message}\n")


def test_doctor_fails_on_a_missing_tau_file_without_asking_a_server(tmp_path, capsys):
    missing = str(tmp_path / "absent.json")
    code, out, _ = doctor(capsys, env_for(tmp_path, FOLIO_TAU_FILE=missing))
    assert code == 1
    assert any(ln.startswith("fail model taus: FOLIO_TAU_FILE") for ln in out.splitlines())
    assert "model self-test" not in out


def test_doctor_shows_a_local_tau_file_and_how_many_entries_the_package_shadows(tmp_path, capsys):
    path, sha = tau_file(tmp_path, declared_key())
    _, out, _ = doctor(capsys, env_for(tmp_path, FOLIO_TAU_FILE=path))
    expected = (
        f"info model taus: local file sha256 {sha[:12]}: 1 entries, 0 shadowed by the package table"
    )
    assert expected in out.splitlines()


def test_doctor_says_the_reader_is_uncalibrated_when_the_declared_version_has_a_bar(
    tmp_path, capsys
):
    with server() as fake:
        _, out, _ = doctor(capsys, env_for(tmp_path, fake.url, **declaration("vllm 0.11|x")))
    expected = (
        "info reader model: uncalibrated (the model name or the declared server version cannot "
        "form a calibration key: every line it reads is flagged low_confidence)"
    )
    assert expected in out.splitlines()


# health


def health(tmp_path, url=None, **settings_kw) -> dict:
    settings = Settings(weights_dir=tmp_path / "w", model_url=url, model_name="m", **settings_kw)
    reader = ModelReader(ModelConfig.from_settings(settings)) if url else None
    app = create_app(settings, port=8000, tesseract=None, model=reader)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        deadline = time.monotonic() + 15
        while True:  # the startup self-test runs in the background
            model = client.get("/api/health").json()["model"]
            if url is None or model["identity"] is not None or time.monotonic() > deadline:
                return model
            time.sleep(0.05)


def declared_settings():
    return {"declared_identity": DeclaredIdentity("vllm", "0.11", HEX.lower())}


def test_health_has_no_identity_and_no_taus_without_a_model_server(tmp_path):
    model = health(tmp_path)
    assert model["identity"] is None and model["taus"] == []


def test_health_reports_a_declared_identity_after_startup(tmp_path):
    with server() as fake:
        model = health(tmp_path, fake.url, **declared_settings())
    assert model["identity"] == "declared"


def test_health_reports_no_identity_when_nothing_names_the_model(tmp_path):
    with server() as fake:
        model = health(tmp_path, fake.url)
    assert model["identity"] == "none"


def test_health_lists_a_local_tau_with_the_sha256_of_its_file(tmp_path):
    path, sha = tau_file(tmp_path, declared_key())
    with server() as fake:
        model = health(tmp_path, fake.url, tau_file=Path(path), **declared_settings())
    assert model["taus"] == [{"key": declared_key(), "source": "local", "file_sha256": sha}]
