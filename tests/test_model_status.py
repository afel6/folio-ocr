"""The model server's health record, and that environment proxies never carry model requests."""

import threading
import time

import pytest
from fake_model_server import Answer, FakeModelServer
from PIL import Image
from test_model_routing import DIGEST, closed_port_url, native_answers, reader, server

from folio_ocr import calibration, model_setup
from folio_ocr.config import Settings
from folio_ocr.model_health import ModelHealth, model_health

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY")


def settings(tmp_path, url):
    return Settings(weights_dir=tmp_path, model_url=url, model_name="m", model_api_key="sk-SECRET")


def line_image() -> Image.Image:
    image = Image.new("L", (300, 40), 255)
    image.paste(0, (20, 10, 280, 30))
    return image


def test_environment_proxies_are_never_used(monkeypatch):
    with FakeModelServer() as proxy, server() as fake:
        for name in PROXY_VARIABLES:
            monkeypatch.setenv(name, proxy.url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        model = reader(fake.url)
        assert model.fit() == "pass"
        assert model.read(line_image(), "en", model.now() + 60).error is None
        assert proxy.seen == []
        assert len(fake.seen) == 3


def test_no_server_configured(tmp_path):
    health = model_health(Settings(weights_dir=tmp_path), None)
    assert health == ModelHealth(False, None, None, None, None, "untested", "uncalibrated")


def test_configured_but_not_asked_yet_sends_nothing(tmp_path):
    with server() as fake:
        model = reader(fake.url)
        health = model_health(settings(tmp_path, fake.url), model)
        assert fake.seen == []
    assert (health.configured, health.reachable, health.fit) == (True, None, "untested")
    assert (health.name, health.host, health.leaves_machine) == ("m", "127.0.0.1", False)
    assert health.calibrated == "uncalibrated"


def test_after_a_passing_self_test(tmp_path):
    with server() as fake:
        model = reader(fake.url)
        model.fit()
        health = model_health(settings(tmp_path, fake.url), model)
    assert (health.reachable, health.fit) == (True, "pass")


def test_a_server_that_cannot_be_reached(tmp_path):
    url = closed_port_url()
    model = reader(url)
    model.fit()
    health = model_health(settings(tmp_path, url), model)
    assert (health.reachable, health.fit) == (False, "fail")


@pytest.mark.parametrize(
    ("url", "leaves"),
    [("https://u:SECRET-PW@models.example/v1", True), ("http://localhost:11434/v1", False)],
)
def test_host_only_and_whether_it_leaves_the_machine(tmp_path, url, leaves):
    health = model_health(settings(tmp_path, url), None)
    assert health.leaves_machine is leaves
    assert "SECRET" not in repr(health)
    assert health.host in ("models.example", "localhost")


def test_calibrated_is_ok_only_once_the_server_has_been_asked(tmp_path, monkeypatch):
    info = model_setup.ServerInfo("ollama", "0.40", DIGEST)
    key = model_setup.model_setup(info, "m", "logprobs", "x").key
    monkeypatch.setattr(calibration, "TAUS", calibration.checked_table({key: 0.5}))
    with server(native=native_answers()) as fake:
        model = reader(fake.url)
        config = settings(tmp_path, fake.url)
        assert model_health(config, model).calibrated == "uncalibrated"
        assert fake.seen == []
        assert model.server() == info
        asked = len(fake.seen)
        assert asked > 0
        assert model_health(config, model).calibrated == "ok"
        assert len(fake.seen) == asked


def test_health_does_not_wait_for_a_running_self_test(tmp_path):
    with server() as fake:
        inner = fake.answer_for
        fake.answer_for = lambda seen: Answer(
            **{**inner(seen).__dict__, "delay_s": 1.5 if seen.method == "POST" else 0.0}
        )
        model = reader(fake.url)
        worker = threading.Thread(target=model.fit)
        worker.start()
        time.sleep(0.3)
        started = time.perf_counter()
        health = model_health(settings(tmp_path, fake.url), model)
        took = time.perf_counter() - started
        worker.join()
        after = model_health(settings(tmp_path, fake.url), model)
    assert took < 0.5
    assert health.fit == "untested"
    assert after.fit == "pass"
