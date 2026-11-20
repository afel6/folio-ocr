"""Which threshold judges a model line: identity precedence, the package table before a local
tau file, the self-test fingerprint and the FOLIO_TAU_FILE refusals."""

import hashlib
import json
import os

import pytest
from fake_model_server import Answer
from test_model_routing import DIGEST, native_answers, read_pdf, reader, server, triggers

from folio_ocr import calibration, model_health, model_pages, model_setup, model_taus
from folio_ocr.config import DeclaredIdentity, Settings
from folio_ocr.doctor import Check
from folio_ocr.errors import ConfigError
from folio_ocr.model_calibration import NO_KEY, ModelCalibration, TauSource
from folio_ocr.model_reader import ModelConfig
from folio_ocr.model_self_test_lines import LINES

TEXTS = {line.languages: line.text for line in LINES}
DECL = DeclaredIdentity("vllm", "0.11", "c" * 64)
SERVER_INFO = model_setup.ServerInfo("ollama", "0.40", DIGEST)
DECLARED_INFO = model_setup.ServerInfo("vllm", "0.11", "c" * 64, declared=True)
VERSION = b'{"version":"0.40.1"}'


def key_of(info, method="logprobs") -> str:
    return model_setup.model_setup(info, "m", method, model_pages.TEXT_LAYER_PREPROCESSING).key


def tau_file(tmp_path, key, *, method="logprobs", ar=None, score=0.99, tol=0.02, tau=0.5):
    """Write a tau file holding one entry; returns (path, sha256 of its bytes)."""
    score = None if method == "two_read" else score
    entry = {
        "tau": tau,
        "method": method,
        "bad_lines": 200,
        "documents": 40,
        "dev_bound": 0.09,
        "dev_manifest_sha256": "b" * 64,
        "self_test": [
            {"line": "en", "text": TEXTS["en"], "score": score},
            {"line": "ar", "text": ar or TEXTS["ar"], "score": score},
        ],
        "drift_tolerance": tol,
    }
    raw = json.dumps({"schema": "folio-taus/1", "entries": {key: entry}}).encode()
    path = tmp_path / "taus.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def settings(tmp_path, url, **kw) -> Settings:
    return Settings(weights_dir=tmp_path, model_url=url, model_name="m", **kw)


def by_name(checks, name) -> Check:
    return next(c for c in checks if c.name == name)


# identity precedence


def test_declared_identity_is_used_when_the_server_has_no_identity_routes():
    with server() as fake:
        identity = reader(fake.url, declared=DECL).identity()
    assert identity.source == "declared"
    assert identity.info.version_slot == "declared:vllm-0.11"


def test_no_declaration_and_no_routes_is_no_identity():
    with server() as fake:
        identity = reader(fake.url).identity()
    assert (identity.source, identity.info) == ("none", None)


def test_a_declaration_that_differs_from_the_server_is_refused():
    differing = DeclaredIdentity("ollama", "0.40", "d" * 64)
    with server(native=native_answers()) as fake:
        identity = reader(fake.url, declared=differing).identity()
    assert identity.source == "none" and "differs" in identity.reason


DIFFERING = DeclaredIdentity("ollama", "0.40", "d" * 64)
DIFFERS = "FOLIO_MODEL_DIGEST or FOLIO_MODEL_SERVER differs"


@pytest.mark.parametrize(
    ("declared", "native", "identity", "model"),
    [
        (
            DECL,
            None,
            "declared vllm 0.11, digest cccccccccccc",
            "uncalibrated (no frozen",
        ),
        (
            None,
            None,
            "none: the server reports no model identity; ",
            "uncalibrated (the server reports no model identity; ",
        ),
        (
            DIFFERING,
            native_answers(),
            f"none: {DIFFERS} ",
            f"uncalibrated ({DIFFERS} ",
        ),
    ],
    ids=["declared", "none", "differs"],
)
def test_doctor_shows_where_the_identity_comes_from(tmp_path, declared, native, identity, model):
    with server(native=native) as fake:
        got = model_health.model_checks(settings(tmp_path, fake.url, declared_identity=declared))
    assert by_name(got, "model identity").detail.startswith(identity)
    assert by_name(got, "reader model").detail.startswith(model)


NOT_LISTED = b'{"models":[{"name":"other","digest":"' + b"a" * 64 + b'"}]}'


@pytest.mark.parametrize(
    "native",
    [
        {"/api/version": Answer(delay_s=1.0, body=VERSION), "/api/tags": Answer(body=b"{}")},
        {"/api/version": Answer(status=500, body=b"{}"), "/api/tags": Answer(status=500)},
        {"/api/version": Answer(body=VERSION), "/api/tags": Answer(body=NOT_LISTED)},
    ],
    ids=["timeout", "http-500", "name-not-listed"],
)
def test_identity_trouble_beats_a_declaration(tmp_path, native):
    with server(native=native) as fake:
        got = model_health.model_checks(
            settings(tmp_path, fake.url, declared_identity=DECL, line_timeout_s=0.3)
        )
    assert by_name(got, "model identity").detail.startswith("none: ")
    assert by_name(got, "reader model").detail.startswith("uncalibrated (")


def test_a_declared_version_with_a_bar_cannot_form_a_key(tmp_path):
    declared = DeclaredIdentity("vllm", "a|b", "c" * 64)
    with server() as fake:
        got = model_health.model_checks(settings(tmp_path, fake.url, declared_identity=declared))
    assert by_name(got, "model identity").detail.startswith("declared vllm a|b")
    assert by_name(got, "reader model").detail.startswith(f"uncalibrated ({NO_KEY}: ")


# the package table comes before the local file


def test_the_package_table_wins_over_the_local_file(tmp_path, monkeypatch):
    key = key_of(DECLARED_INFO)
    monkeypatch.setattr(calibration, "TAUS", calibration.checked_table({key: 0.9}))
    path, _ = tau_file(tmp_path, key, tau=0.1)
    with server() as fake:
        rdr = reader(fake.url, declared=DECL, local=model_taus.load(path))
        assert rdr.tau(key) == (0.9, TauSource(key, "package", None))
        rdr.fit()
        rdr.server()
        health = model_health.model_health(settings(tmp_path, fake.url), rdr)
        got = model_health.model_checks(
            settings(tmp_path, fake.url, declared_identity=DECL, tau_file=path)
        )
    assert health.taus == (TauSource(key, "package", None),)
    assert "1 shadowed" in by_name(got, "model taus").detail


# the local file applies


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "server"])
def test_a_local_entry_applies_while_the_model_answers_like_its_calibration(tmp_path, declared):
    key = key_of(DECLARED_INFO if declared else SERVER_INFO)
    path, sha = tau_file(tmp_path, key)
    with server(native=None if declared else native_answers()) as fake:
        rdr = reader(fake.url, declared=DECL if declared else None, local=model_taus.load(path))
        assert rdr.tau(key) == (0.5, TauSource(key, "local", sha))
        assert rdr.calibrated() == "ok"
        health = model_health.model_health(settings(tmp_path, fake.url), rdr)
        doc = read_pdf(rdr)
    assert (health.calibrated, health.identity) == ("ok", "declared" if declared else "server")
    assert health.taus == (TauSource(key, "local", sha),)
    assert triggers(doc)
    assert all("low_confidence" not in x.flags for x in triggers(doc))
    assert doc.engine.calibration == ModelCalibration(key, 0.5, "local", sha)


def test_a_local_threshold_above_the_confidence_flags_the_line(tmp_path):
    key = key_of(DECLARED_INFO)
    path, _ = tau_file(tmp_path, key, tau=0.999)  # above the 0.99 the fake's logprob gives
    with server() as fake:
        doc = read_pdf(reader(fake.url, declared=DECL, local=model_taus.load(path)))
    assert triggers(doc)
    assert all("low_confidence" in x.flags for x in triggers(doc))


def _run(tmp_path, **file_kw):
    key = key_of(DECLARED_INFO)
    path, _ = tau_file(tmp_path, key, **file_kw)
    with server() as fake:
        rdr = reader(fake.url, declared=DECL, local=model_taus.load(path))
        found, state = rdr.tau(key), rdr.calibrated()
        doc = read_pdf(rdr)
        got = model_health.model_checks(
            settings(tmp_path, fake.url, declared_identity=DECL, tau_file=path)
        )
    return found, state, doc, got


def test_a_changed_answer_to_the_self_test_drops_the_local_threshold(tmp_path):
    found, state, doc, got = _run(tmp_path, ar=TEXTS["ar"] + "x")
    assert (found, state) == (None, "uncalibrated")
    assert by_name(got, "model self-test").status == "ok"
    detail = by_name(got, "reader model").detail
    assert detail.startswith("uncalibrated (the model does not answer like its calibration: ")
    assert triggers(doc)
    assert all("low_confidence" in x.flags for x in triggers(doc))
    assert doc.engine.calibration.tau is None


def test_a_score_drift_beyond_the_tolerance_drops_the_local_threshold(tmp_path):
    found, state, _, _ = _run(tmp_path, score=0.5, tol=0.01)
    assert (found, state) == (None, "uncalibrated")


def test_a_score_inside_the_tolerance_keeps_the_local_threshold(tmp_path):
    found, state, _, _ = _run(tmp_path, score=0.98, tol=0.02)
    assert found is not None and state == "ok"


def test_a_two_read_entry_matches_on_text_only(tmp_path):
    key = key_of(DECLARED_INFO, "two_read")
    path, sha = tau_file(tmp_path, key, method="two_read")
    with server() as fake:  # its replies carry logprobs; they are ignored for two_read
        rdr = reader(fake.url, declared=DECL, local=model_taus.load(path))
        assert rdr.tau(key) == (0.5, TauSource(key, "local", sha))
    path, _ = tau_file(tmp_path, key, method="two_read", ar=TEXTS["ar"] + "x")
    with server() as fake:
        assert reader(fake.url, declared=DECL, local=model_taus.load(path)).tau(key) is None


# FOLIO_TAU_FILE


def test_a_bad_tau_file_is_refused_even_without_a_server(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"not json")
    bare = Settings(weights_dir=tmp_path, tau_file=bad)
    with pytest.raises(ConfigError, match="^FOLIO_TAU_FILE") as raised:
        ModelConfig.from_settings(bare)
    assert model_health.model_checks(bare) == [Check("model taus", "fail", str(raised.value))]


def test_a_good_tau_file_without_a_server_is_one_info_line_and_no_request(tmp_path):
    path, sha = tau_file(tmp_path, key_of(DECLARED_INFO))

    def no_reader(_config):
        raise AssertionError("no reader may be made without a server")

    got = model_health.model_checks(Settings(weights_dir=tmp_path, tau_file=path), reader=no_reader)
    assert len(got) == 1 and (got[0].name, got[0].status) == ("model taus", "info")
    assert sha[:12] in got[0].detail and "1 entries" in got[0].detail


def test_loading_a_file_never_changes_the_package_table(tmp_path):
    before = calibration.table_sha256()
    key = key_of(DECLARED_INFO)
    path, _ = tau_file(tmp_path, key)
    with server() as fake:
        rdr = reader(fake.url, declared=DECL, local=model_taus.load(path))
        rdr.tau(key)
        model_health.model_checks(settings(tmp_path, fake.url, tau_file=path))
    assert calibration.table_sha256() == before and key not in calibration.TAUS


def test_the_calibration_variables_are_unset_at_the_start_of_a_test(monkeypatch):
    for name in ("FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER", "FOLIO_TAU_FILE"):
        assert name not in os.environ
    monkeypatch.setenv("FOLIO_TAU_FILE", "/x")
    assert os.environ["FOLIO_TAU_FILE"] == "/x"


def test_doctor_names_the_local_file_when_its_threshold_applies(tmp_path):
    key = key_of(DECLARED_INFO)
    path, sha = tau_file(tmp_path, key)
    with server() as fake:
        got = model_health.model_checks(
            settings(tmp_path, fake.url, declared_identity=DECL, tau_file=path)
        )
    reader_line = by_name(got, "reader model")
    assert (reader_line.status, reader_line.detail) == ("ok", f"ok (local file sha256 {sha[:12]})")


def test_the_tau_file_is_read_once_by_doctor(tmp_path, monkeypatch):
    path, _ = tau_file(tmp_path, key_of(DECLARED_INFO))
    reads = []
    real = model_taus.load
    monkeypatch.setattr(model_taus, "load", lambda p: reads.append(p) or real(p))
    with server() as fake:
        model_health.model_checks(
            settings(tmp_path, fake.url, declared_identity=DECL, tau_file=path)
        )
    model_health.model_checks(settings(tmp_path, None, tau_file=path))
    assert reads == [path, path]  # once per doctor run, with and without a server


def test_a_declared_version_that_is_not_utf8_gives_no_key():
    info = model_setup.ServerInfo("vllm", "0." + chr(0xDC80), "c" * 64, declared=True)
    assert model_setup.model_setup(info, "m", "logprobs", "x") is None
