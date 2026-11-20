"""The model server's health checks and its wiring in `folio doctor` and `folio read`."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_model_server import Answer, FakeModelServer, Seen
from test_model_routing import DIGEST, SELF_TEST, TEXT, _key, native_answers, reply, sent_image

from folio_ocr import calibration, cli, model_pages, model_setup, pipeline
from folio_ocr.config import Settings
from folio_ocr.doctor import Check
from folio_ocr.document import Document, Engine, Source
from folio_ocr.model_health import model_checks
from folio_ocr.model_reader import NAME_MISSING, ModelReader

PDF = Path(__file__).parent / "fixtures" / "ar-adversarial-2.pdf"


def fake_server(*, same_text=False, native=None) -> FakeModelServer:
    native = native or {}

    def answer(seen: Seen) -> Answer:
        if seen.target == "/v1/chat/completions":
            known = SELF_TEST.get(_key(sent_image(seen)))
            return Answer(body=reply(TEXT if same_text or known is None else known))
        return native.get(seen.target, Answer(status=404, body=b"{}"))

    return FakeModelServer(answer)


def settings(tmp_path, url=None, name="m") -> Settings:
    return Settings(weights_dir=tmp_path, model_url=url, model_name=name)


def test_no_server_configured_has_no_checks(tmp_path):
    assert model_checks(settings(tmp_path)) == []


def test_url_without_a_name(tmp_path):
    got = model_checks(settings(tmp_path, "http://127.0.0.1:1/v1", name=None))
    assert got == [Check("model self-test", "fail", NAME_MISSING)]


def test_passing_server_is_uncalibrated(tmp_path):
    with fake_server() as fake:
        test, identity, model = model_checks(settings(tmp_path, fake.url))
    assert (test.name, test.status) == ("model self-test", "ok")
    assert (identity.name, identity.status) == ("model identity", "info")
    assert identity.detail.startswith("none: the server reports no model identity")
    assert (model.name, model.status) == ("reader model", "info")
    assert model.detail.startswith("uncalibrated (the server reports no model identity")


def test_calibrated_server(tmp_path, monkeypatch):
    info = model_setup.ServerInfo("ollama", "0.40", DIGEST)
    key = model_setup.model_setup(info, "m", "logprobs", model_pages.TEXT_LAYER_PREPROCESSING).key
    monkeypatch.setattr(calibration, "TAUS", calibration.checked_table({key: 0.0}))
    with fake_server(native=native_answers()) as fake:
        test, identity, model = model_checks(settings(tmp_path, fake.url))
    assert (test.status, model.name, model.status, model.detail) == (
        "ok",
        "reader model",
        "ok",
        "ok (package table)",
    )
    assert identity == Check("model identity", "info", f"server ollama 0.40, digest {DIGEST[:12]}")


def test_same_text_server_fails_the_self_test(tmp_path):
    with fake_server(same_text=True) as fake:
        test, *rest = model_checks(settings(tmp_path, fake.url))
    assert (test.name, test.status) == ("model self-test", "fail")
    assert [c.name for c in rest] == ["model identity", "reader model"]


def doctor_env(tmp_path, url=None) -> dict:
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path), "PATH": ""}
    if url is not None:
        env |= {"FOLIO_MODEL_URL": url, "FOLIO_MODEL_NAME": "m"}
        env["FOLIO_MODEL_API_KEY"] = "sk-SENTINEL-7"
    return env


def test_doctor_shows_the_model_lines_after_the_model_server_line(tmp_path, capsys):
    with fake_server() as fake:
        cli.main(["doctor"], doctor_env(tmp_path, fake.url))
    cap = capsys.readouterr()
    lines = cap.out.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith("info model server: "))
    assert lines[at + 1].startswith("ok   model self-test: ")
    assert lines[at + 2].startswith("info model identity: none: ")
    assert lines[at + 3].startswith("info reader model: uncalibrated (")
    assert "SENTINEL" not in cap.out + cap.err
    sent = [s for s in fake.seen if s.target == "/v1/chat/completions"]
    assert sent and all(s.headers["authorization"] == "Bearer sk-SENTINEL-7" for s in sent)


def test_doctor_without_a_server_has_no_self_test_line(tmp_path, capsys):
    cli.main(["doctor"], doctor_env(tmp_path))
    assert "model self-test" not in capsys.readouterr().out


@pytest.fixture
def read_calls(monkeypatch):
    calls = []

    def record(path, **kw):
        calls.append((path, kw))
        raise RuntimeError("stop")  # ends the run; the CLI reports it as a generic failure

    monkeypatch.setattr(pipeline, "read_path", record)
    return calls


def test_read_refuses_a_url_without_a_name(tmp_path, capsys, read_calls):
    env = {"FOLIO_WEIGHTS_DIR": str(tmp_path), "FOLIO_MODEL_URL": "http://127.0.0.1:1/v1"}
    code = cli.main(["read", str(PDF)], env)
    err = capsys.readouterr().err
    assert code == 1 and f"error: {NAME_MISSING}" in err and read_calls == []


def test_read_gives_every_file_the_same_reader_built_from_the_environment(
    tmp_path, capsys, monkeypatch
):
    calls = []

    empty = Document(
        id="d",
        source=Source("a.pdf", "0" * 64, "pdf", 0),
        languages="en+ar",
        engine=Engine("none", None, "none", None, None, "untested", None),
        pages=(),
        created_at=datetime.now(UTC),
        has_edits=False,
    )

    def record(path, **kw):
        calls.append(kw)
        return empty

    monkeypatch.setattr(pipeline, "read_path", record)
    env = {
        "FOLIO_WEIGHTS_DIR": str(tmp_path),
        "FOLIO_MODEL_URL": "http://127.0.0.1:1/v1",
        "FOLIO_MODEL_NAME": "m",
        "FOLIO_LINE_TIMEOUT_S": "7",
        "FOLIO_PAGE_DEADLINE_S": "99",
        "FOLIO_TWO_READ": "1",
    }
    cli.main(["read", str(PDF), str(PDF)], env)
    capsys.readouterr()
    first, second = (c["model"] for c in calls)
    assert isinstance(first, ModelReader) and first is second
    c = first.config
    assert (c.url, c.name, c.line_timeout_s, c.page_deadline_s, c.two_read) == (
        "http://127.0.0.1:1/v1",
        "m",
        7.0,
        99.0,
        True,
    )
    assert json.dumps(c.api_key) == "null"
