"""Tests for the laptop model run: exchange capture, reply helpers, server facts, the
reply-contract checks."""

import base64
import io
import json
import socket
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import model_run_server as sm
from fake_model_server import Answer, FakeModelServer

from folio_ocr import model_http
from folio_ocr.config import DeclaredIdentity
from folio_ocr.model_reader import ModelConfig, ModelReader
from folio_ocr.model_self_test_lines import LINES

SAMPLE_A = rb"""{"id":"chatcmpl-281","object":"chat.completion","created":1795114573,"model":"qwen3-vl:8b-instruct-q4_K_M","system_fingerprint":"fp_ollama","choices":[{"index":0,"message":{"role":"assistant","content":"Invoice number: INV-2026-3706"},"finish_reason":"stop","logprobs":{"content":[{"token":"Invoice","logprob":-1.1920930376163597e-7,"bytes":[73,110,118,111,105,99,101]},{"token":" number","logprob":0,"bytes":[32,110,117,109,98,101,114]},{"token":":","logprob":0,"bytes":[58]},{"token":" INV","logprob":-0.000002861027041944908,"bytes":[32,73,78,86]},{"token":"-","logprob":0,"bytes":[45]},{"token":"2","logprob":0,"bytes":[50]},{"token":"0","logprob":0,"bytes":[48]},{"token":"2","logprob":0,"bytes":[50]},{"token":"6","logprob":0,"bytes":[54]},{"token":"-","logprob":0,"bytes":[45]},{"token":"3","logprob":0,"bytes":[51]},{"token":"7","logprob":0,"bytes":[55]},{"token":"0","logprob":0,"bytes":[48]},{"token":"6","logprob":0,"bytes":[54]}]}}],"usage":{"prompt_tokens":1179,"prompt_tokens_details":{"cached_tokens":1178},"completion_tokens":15,"total_tokens":1194}}"""
DIGEST = "70e58e37beb15464e19aa903dedb2cba828b89c0fb186f75ff2738c7347b96eb"
NAME = "qwen3-vl:8b-instruct-q4_K_M"


def config(url: str) -> ModelConfig:
    return ModelConfig(url, NAME, None, 30.0, 60.0, False)


def closed_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def chat_body(content: str, finish: str = "stop", logprobs: bool = True) -> bytes:
    choice = {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
    if logprobs:
        choice["logprobs"] = {"content": [{"token": content or "x", "logprob": -0.01}]}
    return json.dumps({"choices": [choice], "usage": {"prompt_tokens": 5}}).encode()


def pixels(png: bytes):
    image = Image.open(io.BytesIO(png)).convert("L")
    return image.size, image.tobytes()


def make_handler(*, logprobs=True, honor_max_tokens=True, version="0.40.16"):
    known = {pixels(line.png): line.text for line in LINES}

    def handle(seen):
        if seen.target.endswith("/api/version"):
            return Answer(body=json.dumps({"version": version}).encode())
        if seen.target.endswith("/api/tags"):
            models = [{"name": NAME, "model": NAME, "digest": DIGEST}]
            return Answer(body=json.dumps({"models": models}).encode())
        request = json.loads(seen.body)
        url = request["messages"][0]["content"][1]["image_url"]["url"]
        line_text = known.get(pixels(base64.b64decode(url.partition("base64,")[2])))
        if line_text is None:
            return Answer(body=chat_body(""))
        if request["max_tokens"] == 1 and honor_max_tokens:
            return Answer(body=chat_body(line_text[:1], "length", logprobs))
        return Answer(body=chat_body(line_text, "stop", logprobs))

    return handle


# ---- capture ----


def test_capture_records_exchange_and_reader_works():
    with FakeModelServer(make_handler()) as server:
        capture = sm.Capture()
        reader = ModelReader(config(server.url), post=capture.post, get=model_http.get)
        assert reader.self_test().passed
        taken = capture.take()
        assert len(taken) == 2
        assert all(e.status == 200 and e.error is None and e.seconds >= 0 for e in taken)
        assert capture.take() == ()


def test_capture_records_error_and_reraises():
    capture = sm.Capture()
    target = model_http.target(closed_url(), None)
    with pytest.raises(model_http.ModelCallError):
        capture.post(target, b"{}", 2.0)
    (exchange,) = capture.take()
    assert exchange.status is None and exchange.body == b"" and exchange.error
    capture.reset()
    assert capture.take() == ()


def test_capture_records_that_a_call_timed_out():
    def slow(target, body, timeout_s):
        raise model_http.ModelCallError("took too long", timed_out=True)

    capture = sm.Capture(post=slow)
    with pytest.raises(model_http.ModelCallError):
        capture.post(model_http.target("http://127.0.0.1:1", None), b"{}", 2.0)
    (exchange,) = capture.take()
    assert exchange.timed_out is True and exchange.error == "took too long"


def test_capture_is_per_thread():
    with FakeModelServer(Answer(body=b"{}")) as server:
        capture = sm.Capture()
        target = model_http.target(server.url, None)
        counts = {}

        def work(name, n):
            for _ in range(n):
                capture.post(target, b"{}", 5.0)
            counts[name] = len(capture.take())

        threads = [threading.Thread(target=work, args=(k, n)) for k, n in (("a", 1), ("b", 3))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counts == {"a": 1, "b": 3}


# ---- reply helpers ----


def test_reply_helpers_on_sample():
    assert sm.reply_json(SAMPLE_A)["object"] == "chat.completion"
    assert len(sm.logprob_entries(SAMPLE_A)) == 14
    assert sm.finish_reason(SAMPLE_A) == "stop"
    assert sm.usage(SAMPLE_A) == {
        "prompt_tokens": 1179,
        "completion_tokens": 15,
        "cached_tokens": 1178,
    }


@pytest.mark.parametrize(
    "body", [b"", b"\xef\xbb\xbf{}", b"[1]", b"\xff", b"{", b'{"choices": 5}', b'{"choices": []}']
)
def test_reply_helpers_are_total(body):
    assert sm.logprob_entries(body) is None
    assert sm.finish_reason(body) is None
    assert sm.usage(body) == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
    }
    if body in (b"", b"\xef\xbb\xbf{}", b"[1]", b"\xff", b"{"):
        assert sm.reply_json(body) is None


def test_usage_ignores_non_int():
    body = (
        b'{"usage": {"prompt_tokens": "5", "completion_tokens": true, "prompt_tokens_details": 3}}'
    )
    assert sm.usage(body) == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
    }


def test_request_max_tokens():
    assert sm.request_max_tokens(b'{"max_tokens": 7}') == 7
    assert sm.request_max_tokens(b'{"max_tokens": "7"}') is None
    assert sm.request_max_tokens(b"nope") is None


def test_exchange_record_round_trip():
    request = json.dumps({"max_tokens": 9}).encode()
    ex = sm.Exchange(request, 200, SAMPLE_A, 1.5, None)
    rec = sm.exchange_record(ex)
    assert rec["max_tokens"] == 9
    assert rec["content"] == "Invoice number: INV-2026-3706"
    assert rec["finish_reason"] == "stop"
    assert rec["prompt_tokens"] == 1179 and rec["cached_tokens"] == 1178
    assert len(rec["entries"]) == 14
    assert len(rec["request_sha256"]) == 64
    assert sm.exchange_from_record(rec) == ex


def test_exchange_record_of_failed_call():
    ex = sm.Exchange(b"x", None, b"", 0.5, "no answer", timed_out=True)
    rec = sm.exchange_record(ex)
    assert rec["timed_out"] is True
    assert rec["content"] is None and rec["entries"] is None and rec["status"] is None
    assert sm.exchange_from_record(rec) == ex


# ---- server facts ----


def test_ollama_version_and_facts():
    with FakeModelServer(make_handler()) as server:
        assert sm.ollama_version(server.url, None, 5.0) == "0.40.16"
        facts = sm.server_facts(config(server.url), 5.0)
    assert facts == {
        "kind": "ollama",
        "version": "0.40.16",
        "version_major_minor": "0.40",
        "digest": DIGEST,
        "model": NAME,
        "host": "127.0.0.1",
        "identity": "server",
    }


@pytest.mark.parametrize(
    "answer",
    [
        Answer(status=500, body=b'{"version": "1"}'),
        Answer(body=b"nope"),
        Answer(body=b'{"version": ""}'),
        Answer(body=b'{"version": 3}'),
    ],
)
def test_ollama_version_none(answer):
    with FakeModelServer(answer) as server:
        assert sm.ollama_version(server.url, None, 5.0) is None


def test_ollama_version_none_without_v1_and_error_propagates():
    assert sm.ollama_version("http://127.0.0.1:1/other", None, 1.0) is None
    with pytest.raises(model_http.ModelCallError):
        sm.ollama_version(closed_url(), None, 2.0)


def test_server_facts_missing_and_down():
    with (
        FakeModelServer(Answer(body=b"{}")) as server,
        pytest.raises(sm.ModelRunError, match="calibration key"),
    ):
        sm.server_facts(config(server.url), 5.0)
    with pytest.raises(
        sm.ModelRunError, match=r"model server (?:did not answer|refused the connection)"
    ):
        sm.server_facts(config(closed_url()), 2.0)


# ---- warm up ----


def test_warm_up_returns_seconds_and_maps_errors():
    with FakeModelServer(make_handler()) as server:
        assert sm.warm_up(config(server.url), 5.0) >= 0
    with (
        FakeModelServer(Answer(status=503)) as server,
        pytest.raises(sm.ModelRunError, match="HTTP 503"),
    ):
        sm.warm_up(config(server.url), 5.0)
    with pytest.raises(sm.ModelRunError, match="did not answer"):
        sm.warm_up(config(closed_url()), 2.0)


# ---- contract checks ----


def run_checks(server, **kw):
    return sm.contract_checks(config(server.url), timeout_s=10.0, **kw)


def by_name(result):
    return {c["name"]: c for c in result["checks"]}


def test_contract_checks_all_pass():
    with FakeModelServer(make_handler()) as server:
        result = run_checks(server, expect_version="0.40.16", expect_digest=DIGEST)
    assert [c["name"] for c in result["checks"]] == [
        "server facts",
        "logprobs on an image request (English)",
        "logprobs on an image request (Arabic)",
        "truncation flag",
        "self-test",
        "blank image",
    ]
    assert all(c["passed"] for c in result["checks"]), result["checks"]
    assert result["passed"] is True
    assert result["facts"]["version"] == "0.40.16"
    assert by_name(result)["blank image"]["required"] is False


def test_contract_checks_no_logprobs_fails():
    with FakeModelServer(make_handler(logprobs=False)) as server:
        result = run_checks(server)
    checks = by_name(result)
    assert not checks["logprobs on an image request (English)"]["passed"]
    assert result["passed"] is False


def test_contract_checks_ignoring_max_tokens_fails():
    with FakeModelServer(make_handler(honor_max_tokens=False)) as server:
        result = run_checks(server)
    assert not by_name(result)["truncation flag"]["passed"]
    assert result["passed"] is False


def test_contract_checks_version_mismatch_names_values():
    with FakeModelServer(make_handler()) as server:
        result = run_checks(server, expect_version="0.99.0")
    check = by_name(result)["server facts"]
    assert not check["passed"]
    assert "0.99.0" in check["detail"] and "0.40.16" in check["detail"]


def test_contract_checks_digest_mismatch():
    with FakeModelServer(make_handler()) as server:
        result = run_checks(server, expect_digest="0" * 64)
    assert not by_name(result)["server facts"]["passed"]


def test_contract_checks_server_down_does_not_raise():
    result = sm.contract_checks(config(closed_url()), timeout_s=2.0)
    required = [c for c in result["checks"] if c["required"]]
    assert required and not any(c["passed"] for c in required)
    assert result["passed"] is False and result["facts"] is None
    assert all(c["detail"] for c in result["checks"])


def test_logprob_rule_detail():
    line = LINES[0]
    body = chat_body(line.text)
    assert sm.logprob_failure(200, body, line.text) is None
    assert sm.logprob_failure(500, body, line.text)
    assert sm.logprob_failure(200, chat_body(line.text, "length"), line.text)
    assert sm.logprob_failure(200, chat_body("other"), line.text)
    positive = json.loads(body)
    positive["choices"][0]["logprobs"]["content"][0]["logprob"] = 0.5
    assert sm.logprob_failure(200, json.dumps(positive).encode(), line.text)
    assert sm.logprob_failure(200, b"nope", line.text)


# ---- server facts with a declared identity ----

DECLARED_DIGEST = "ab" * 32


def declared_config(url: str) -> ModelConfig:
    return ModelConfig(url, "acme-ocr-7b", None, 30.0, 60.0, False,
                       DeclaredIdentity("vllm", "0.9", DECLARED_DIGEST))


def test_server_facts_take_the_declared_identity_when_the_routes_are_absent():
    with FakeModelServer(Answer(status=404, body=b"{}")) as server:
        facts = sm.server_facts(declared_config(server.url), 5.0)
    assert facts == {
        "kind": "vllm",
        "version": "0.9",
        "version_major_minor": "0.9",
        "digest": DECLARED_DIGEST,
        "model": "acme-ocr-7b",
        "host": "127.0.0.1",
        "identity": "declared",
    }


def test_server_facts_name_the_server_identity_source():
    with FakeModelServer(make_handler()) as server:
        facts = sm.server_facts(config(server.url), 5.0)
    assert facts["identity"] == "server" and facts["version"] == "0.40.16"


def test_server_facts_refuse_without_routes_or_declaration_in_one_line():
    with (
        FakeModelServer(Answer(status=404, body=b"{}")) as server,
        pytest.raises(sm.ModelRunError, match="^no calibration key can be formed: ") as stop,
    ):
        sm.server_facts(config(server.url), 5.0)
    assert "\n" not in str(stop.value)
