import base64
import io
import json
import math
import socket
import threading
import time

import pytest
from fake_model_server import Answer, FakeModelServer, Seen
from PIL import Image, ImageDraw

from folio_ocr import calibration, model_http, model_request, model_setup
from folio_ocr.config import Settings
from folio_ocr.errors import ConfigError
from folio_ocr.model_reader import (
    NAME_MISSING,
    ModelConfig,
    ModelReader,
    SelfTest,
    agreement,
    line_time_limit,
    page_time_limit,
)
from folio_ocr.model_reply import status_error
from folio_ocr.model_self_test_lines import LINES

FAR = 1e12
DIGEST = "ab" * 32


def reply(text, logprobs=None, finish="stop") -> bytes:
    choice = {"message": {"content": text}, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"logprob": v} for v in logprobs]}
    return json.dumps({"choices": [choice]}).encode()


def config(url="http://127.0.0.1:1/v1", **kw) -> ModelConfig:
    fields = {"name": "m", "api_key": None, "line_timeout_s": 5.0, "page_deadline_s": 900.0}
    fields |= {"two_read": False}
    return ModelConfig(url=url, **{**fields, **kw})


def line_image(w=300, h=40) -> Image.Image:
    image = Image.new("L", (w, h), 255)
    ImageDraw.Draw(image).rectangle((10, 10, w - 10, h - 10), fill=20)
    return image


def sent_png(seen: Seen) -> Image.Image:
    url = json.loads(seen.body)["messages"][0]["content"][1]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://u:SENTINEL@127.0.0.1:{port}/v1"


def read(fake, image=None, languages="en", **kw):
    reader = ModelReader(config(fake.url, **kw))
    return reader.read(image or line_image(), languages, reader.now() + FAR)


def test_request_shape_and_auth():
    image = line_image()
    with FakeModelServer(Answer(body=reply("hi"))) as fake:
        read(fake, image, "en", name="mm", api_key="sk-1")
        read(fake, image, "ar", name="mm", api_key="sk-1")
    en, ar = fake.seen
    assert en.method == "POST" and en.target == "/v1/chat/completions"
    assert en.headers["authorization"] == "Bearer sk-1"
    body = json.loads(en.body)
    assert body["model"] == "mm" and body["temperature"] == 0 and body["seed"] == 42
    assert body["logprobs"] is True and body["stream"] is False
    assert body["max_tokens"] == model_request.max_tokens(300, 40, "en")
    assert json.loads(ar.body)["max_tokens"] == model_request.max_tokens(300, 40, "ar")
    assert body["max_tokens"] != json.loads(ar.body)["max_tokens"]
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": model_request.PROMPT}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent_png(en).tobytes() == image.tobytes()


def test_reply_goes_through_parse_reply():
    with FakeModelServer(Answer(body=reply("abc", logprobs=[-0.1, -0.5]))) as fake:
        reading = read(fake)
    assert (reading.text, reading.method) == ("abc", "logprobs")
    assert reading.confidence == pytest.approx(math.exp(-0.5))


def test_line_limit_stops_a_slow_reply():
    with FakeModelServer(Answer(body=b"x" * 200, byte_delay_s=0.05)) as fake:
        began = time.perf_counter()
        reading = read(fake, line_timeout_s=0.4)
    assert reading.error == line_time_limit(0.4)
    assert time.perf_counter() - began < 1.5


def test_page_limit_already_over_sends_nothing():
    with FakeModelServer(Answer(body=reply("x"))) as fake:
        reader = ModelReader(config(fake.url, page_deadline_s=7.0))
        reading = reader.read(line_image(), "en", reader.now() - 1)
    assert reading.error == page_time_limit(7.0)
    assert fake.seen == []


def test_page_limit_binds_before_the_line_limit():
    with FakeModelServer(Answer(body=b"x" * 200, byte_delay_s=0.05)) as fake:
        reader = ModelReader(config(fake.url, line_timeout_s=5.0, page_deadline_s=3.0))
        reading = reader.read(line_image(), "en", reader.now() + 0.3)
    assert reading.error == page_time_limit(3.0)


def _counting_reader(times, **kw):
    calls = []

    def post(target, body, timeout):
        calls.append(timeout)
        return model_http.Reply(200, reply("x"))

    clock = iter(times)
    return ModelReader(config(**kw), post=post, clock=lambda: next(clock)), calls


def test_page_left_of_exactly_zero_sends_nothing():
    reader, calls = _counting_reader([0.0, 100.0], page_deadline_s=9.0)
    assert reader.read(line_image(), "en", 100.0).error == page_time_limit(9.0)
    assert calls == []


def test_line_left_of_exactly_zero_sends_nothing():
    reader, calls = _counting_reader([0.0, 5.0], line_timeout_s=5.0)
    assert reader.read(line_image(), "en", FAR).error == line_time_limit(5.0)
    assert calls == []


def test_timeout_passed_on_is_the_smaller_limit():
    reader, calls = _counting_reader([0.0, 1.0], line_timeout_s=5.0)
    reader.read(line_image(), "en", 3.0)
    assert calls == [2.0]


def test_errors_are_friendly_and_hold_no_secret():
    with FakeModelServer(lambda s: Answer(status=401 if len(fake.seen) == 1 else 500)) as fake:
        first = read(fake, api_key="sk-SENTINEL").error
        second = read(fake, api_key="sk-SENTINEL").error
    assert first == status_error(401) and second == status_error(500)
    refused = ModelReader(config(closed_port_url(), api_key="sk-SENTINEL"))
    gone = refused.read(line_image(), "en", FAR).error
    assert gone == model_http._REFUSED
    for message in (first, second, gone):
        assert "SENTINEL" not in message


def test_two_read_second_image_has_a_margin():
    with FakeModelServer(Answer(body=reply("same text"))) as fake:
        reading = read(fake, two_read=True)
    assert len(fake.seen) == 2
    first, second = sent_png(fake.seen[0]), sent_png(fake.seen[1])
    border = math.ceil(model_request.TWO_READ_MARGIN * first.height)
    assert second.width == first.width + 2 * border
    assert (reading.confidence, reading.method) == (1.0, "two_read")


def test_two_read_disagreement_lowers_confidence():
    texts = ["abcdefghijklmnopqrst", "abcdefghijklmnopqrsx"]
    with FakeModelServer(lambda s: Answer(body=reply(texts[len(fake.seen) - 1]))) as fake:
        reading = read(fake, two_read=True)
    assert reading.text == texts[0]
    assert reading.confidence == pytest.approx(0.95)


def test_two_read_skipped_with_logprobs_or_when_off():
    with FakeModelServer(Answer(body=reply("t", logprobs=[-0.2]))) as fake:
        assert read(fake, two_read=True).method == "logprobs"
    assert len(fake.seen) == 1
    with FakeModelServer(Answer(body=reply("t"))) as fake:
        reading = read(fake, two_read=False)
    assert len(fake.seen) == 1
    assert (reading.method, reading.confidence) == ("none", None)


def test_failed_second_read_keeps_the_first_text_unscored():
    answers = [Answer(body=reply("kept")), Answer(status=500)]
    with FakeModelServer(lambda s: answers[len(fake.seen) - 1]) as fake:
        reading = read(fake, two_read=True)
    assert (reading.text, reading.method, reading.confidence) == ("kept", "none", None)
    assert reading.error is None


def test_agreement():
    assert agreement("abc", "abc") == 1.0
    assert agreement("", "") == 1.0
    assert agreement("", "x") == 0.0
    assert agreement("abcd", "abcx") == 0.75
    assert agreement("ab", "abcdef") == 0.0
    assert agreement(chr(0x661) + chr(0x662), "12") == 1.0


def _line_answer(seen: Seen, change=lambda line: line.text) -> Answer:
    sent = sent_png(seen)
    for line in LINES:
        ref = Image.open(io.BytesIO(line.png)).convert("L")
        if ref.size == sent.size and ref.tobytes() == sent.tobytes():
            return Answer(body=reply(change(line)))
    return Answer(body=reply("unknown"))


def _self_test(answer) -> SelfTest:
    with FakeModelServer(answer) as fake:
        return ModelReader(config(fake.url)).self_test()


def test_self_test_passes_when_both_lines_read_exactly():
    result = _self_test(_line_answer)
    assert result.passed
    assert result.detail == "an English and an Arabic test line were read exactly"


def test_self_test_fails_on_the_wrong_script():
    english = LINES[0].text
    result = _self_test(lambda s: Answer(body=reply(english)))
    assert not result.passed
    assert "Arabic" in result.detail and "English" not in result.detail


def test_self_test_fails_on_western_digits_in_arabic():
    table = {ord(chr(0x660 + i)): str(i) for i in range(10)}
    result = _self_test(lambda s: _line_answer(s, lambda line: line.text.translate(table)))
    assert not result.passed and "Arabic" in result.detail


def test_self_test_folds_runs_of_spaces():
    result = _self_test(lambda s: _line_answer(s, lambda line: line.text.replace(" ", "   ")))
    assert result.passed


def test_self_test_reports_http_401_and_refused():
    assert "HTTP 401" in _self_test(Answer(status=401)).detail
    refused = ModelReader(config(closed_port_url())).self_test()
    assert not refused.passed and "SENTINEL" not in refused.detail


def test_fit_runs_the_self_test_once():
    with FakeModelServer(_line_answer) as fake:
        reader = ModelReader(config(fake.url))
        assert {reader.fit() for _ in range(5)} == {"pass"}
    assert len(fake.seen) == 2
    with FakeModelServer(Answer(status=500)) as fake:
        reader = ModelReader(config(fake.url))
        assert {reader.fit() for _ in range(5)} == {"fail"}
    assert len(fake.seen) == 1  # the English line failed, so the Arabic one is not asked


def test_fit_is_safe_across_threads():
    with FakeModelServer(lambda s: Answer(body=reply("x"), delay_s=0.2)) as fake:
        reader = ModelReader(config(fake.url))
        results = []
        threads = [threading.Thread(target=lambda: results.append(reader.fit())) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results == ["fail", "fail"]
    assert len(fake.seen) == 1


def _ollama(seen: Seen) -> Answer:
    if seen.target == "/api/version":
        return Answer(body=b'{"version":"0.40.16"}')
    tags = {"models": [{"name": "m", "model": "m", "digest": DIGEST}]}
    return Answer(body=json.dumps(tags).encode())


def test_server_is_asked_once_and_calibration_follows_the_table(monkeypatch):
    with FakeModelServer(_ollama) as fake:
        reader = ModelReader(config(fake.url))
        info = model_setup.ServerInfo("ollama", "0.40", DIGEST)
        assert [reader.server() for _ in range(3)] == [info] * 3
        assert len(fake.seen) == 2
        assert reader.calibrated() == "uncalibrated"
        setup = model_setup.model_setup(info, "m", "logprobs", "x")
        table = calibration.checked_table({setup.key: 0.5})
        monkeypatch.setattr(calibration, "TAUS", table)
        assert reader.calibrated() == "ok"


def test_server_that_does_not_answer_is_uncalibrated():
    with FakeModelServer(Answer(status=404)) as fake:
        reader = ModelReader(config(fake.url))
        assert reader.server() is None
        assert reader.calibrated() == "uncalibrated"


def test_config_from_settings(tmp_path):
    assert ModelConfig.from_settings(Settings(weights_dir=tmp_path)) is None
    with pytest.raises(ConfigError) as err:
        ModelConfig.from_settings(Settings(weights_dir=tmp_path, model_url="http://h/v1"))
    assert str(err.value) == NAME_MISSING
    url = "http://u:PASSWORD@127.0.0.1:11434/v1"
    settings = Settings(
        weights_dir=tmp_path,
        model_url=url,
        model_name="m",
        model_api_key="sk-KEY",
        line_timeout_s=3.0,
        page_deadline_s=30.0,
        two_read=True,
    )
    cfg = ModelConfig.from_settings(settings)
    assert (cfg.url, cfg.name, cfg.api_key) == (url, "m", "sk-KEY")
    assert (cfg.line_timeout_s, cfg.page_deadline_s, cfg.two_read) == (3.0, 30.0, True)
    assert "sk-KEY" not in repr(cfg) and "PASSWORD" not in repr(cfg)
    assert cfg.host == "127.0.0.1"


def test_second_read_gets_only_what_is_left_of_the_line_limit():
    # clock: line start, first call, (the first post takes 4 s) second call
    reader, calls = _counting_reader([0.0, 0.0, 4.0], line_timeout_s=10.0, two_read=True)
    reader.read(line_image(), "en", FAR)
    assert calls == [pytest.approx(10.0), pytest.approx(6.0)]


def test_page_limit_caps_the_first_post():
    reader, calls = _counting_reader([0.0, 0.0], line_timeout_s=10.0)
    reader.read(line_image(), "en", 2.0)
    assert calls == [pytest.approx(2.0)]


@pytest.mark.parametrize("field", ["model_url", "model_name", "model_api_key"])
def test_a_value_that_cannot_be_sent_is_a_config_error(tmp_path, field):
    values = {"model_url": "http://127.0.0.1:1/v1", "model_name": "m", "model_api_key": "k"}
    values[field] = values[field] + chr(0xDC80)  # an undecodable environment byte
    settings = Settings(weights_dir=tmp_path, **values)
    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_settings(settings)
    assert str(caught.value).startswith("FOLIO_MODEL_")
    assert chr(0xDC80) not in str(caught.value)
