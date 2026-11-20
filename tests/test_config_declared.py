"""The declared model server identity and the threshold file path read from the environment."""

import os
from pathlib import Path

import pytest

from folio_ocr import cli, model_health
from folio_ocr.config import DeclaredIdentity, Settings
from folio_ocr.errors import ConfigError

HEX = "0123456789abcdef" * 4  # 64 lower case hex digits
DIGEST = "FOLIO_MODEL_DIGEST"
SERVER = "FOLIO_MODEL_SERVER"
TAU = "FOLIO_TAU_FILE"


def build(env):
    return Settings.from_env(env, windows=False)


def declared(server, digest=HEX):
    return build({SERVER: server, DIGEST: digest}).declared_identity


def refused(env, name):
    with pytest.raises(ConfigError) as caught:
        build(env)
    message = str(caught.value)
    assert name in message
    for value in env.values():
        if len(value) >= 6 and value.strip():
            assert value not in message
    return message


@pytest.mark.parametrize(
    ("value", "stored"),
    [(HEX, HEX), (HEX.upper(), HEX), ("aB" * 20, "ab" * 20), ("AB" * 32, "ab" * 32)],
)
def test_digest_of_40_or_64_hex_digits_is_stored_lower_case(value, stored):
    assert declared("vllm 0.11", value).digest == stored


@pytest.mark.parametrize(
    "value",
    [
        HEX[:63], HEX + "a", HEX[:39], HEX[:41], " " + HEX, HEX + " ", HEX + "\n",
        "sha256:" + HEX, HEX[:63] + "g", "٠" * 64, "０" * 64,
    ],
)  # fmt: skip
def test_digest_of_another_shape_is_refused(value):
    refused({DIGEST: value, SERVER: "vllm 0.11"}, DIGEST)


@pytest.mark.parametrize(
    ("server", "kind", "version"),
    [
        ("vllm 0.11.2", "vllm", "0.11"),
        ("vllm v0.11", "vllm", "0.11"),
        ("ollama 0.12.3", "ollama", "0.12"),
        ("vllm latest", "vllm", "latest"),
        ("llama.cpp b4500", "llama.cpp", "b4500"),
        ("my-srv_2 0.11rc1", "my-srv_2", "0.11rc1"),
        ("tgi 3.0.1-rc2", "tgi", "3.0"),
    ],
)
def test_server_is_split_into_kind_and_version(server, kind, version):
    identity = declared(server)
    assert (identity.kind, identity.version) == (kind, version)


@pytest.mark.parametrize(
    "value",
    [
        "latest", "vllm  0.11", "VLLM 0.11", " vllm 0.11", "vllm 0.11 ", "vllm ", "-vllm 0.11",
        ".vllm 0.11", "vllm\t0.11",
    ],
)  # fmt: skip
def test_server_of_another_shape_is_refused(value):
    refused({SERVER: value, DIGEST: HEX}, SERVER)


def test_both_set_gives_the_declared_identity():
    settings = build({SERVER: "vllm 0.11.2", DIGEST: HEX.upper()})
    assert settings.declared_identity == DeclaredIdentity("vllm", "0.11", HEX)


@pytest.mark.parametrize("env", [{}, {SERVER: "", DIGEST: ""}])
def test_neither_set_gives_no_identity(env):
    assert build(env).declared_identity is None


def test_only_the_digest_set_asks_for_the_server():
    refused({DIGEST: HEX}, SERVER)


def test_only_the_server_set_asks_for_the_digest():
    refused({SERVER: "vllm 0.11"}, DIGEST)


def test_an_empty_digest_counts_as_unset():
    refused({SERVER: "vllm 0.11", DIGEST: ""}, DIGEST)


def test_a_malformed_digest_is_reported_even_without_a_server():
    message = refused({DIGEST: HEX[:63]}, DIGEST)
    assert SERVER not in message


def test_tau_file_absolute_path_is_kept_exactly(tmp_path):
    value = str(tmp_path / "taus.json")
    assert build({TAU: value}).tau_file == Path(value)


def test_tau_file_defaults_to_none_and_empty_means_unset():
    assert build({}).tau_file is None
    assert build({TAU: ""}).tau_file is None


@pytest.mark.parametrize("kind", ["bare", "dot", "space"])
def test_tau_file_must_be_absolute(tmp_path, kind):
    value = {
        "bare": "taus.json",
        "dot": "./taus.json",
        "space": " " + str(tmp_path / "taus.json"),
    }[kind]
    refused({TAU: value}, TAU)


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths exist on Windows only")
@pytest.mark.parametrize("value", ["\\taus.json", "C:taus.json"])
def test_tau_file_refuses_paths_without_a_full_root(value):
    refused({TAU: value}, TAU)


def test_doctor_refuses_a_malformed_digest_in_one_line(capsys, tmp_path):
    env = {DIGEST: "ab" * 31 + "a", "FOLIO_WEIGHTS_DIR": str(tmp_path)}
    code = cli.main(["doctor"], env)
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("error: FOLIO_MODEL_DIGEST") and err.count("\n") == 1
    assert env[DIGEST] not in err


@pytest.fixture
def model_checks(monkeypatch):
    seen = []

    def fake(settings):
        seen.append(settings.tau_file)
        return []

    monkeypatch.setattr(model_health, "model_checks", fake)
    return seen


def test_doctor_asks_the_model_checks_when_only_a_tau_file_is_set(model_checks, tmp_path):
    taus = tmp_path / "taus.json"
    cli.main(["doctor"], {TAU: str(taus), "FOLIO_WEIGHTS_DIR": str(tmp_path), "PATH": ""})
    assert model_checks == [taus]


def test_doctor_skips_the_model_checks_without_a_server_or_a_tau_file(model_checks, tmp_path):
    cli.main(["doctor"], {"FOLIO_WEIGHTS_DIR": str(tmp_path), "PATH": ""})
    assert model_checks == []
