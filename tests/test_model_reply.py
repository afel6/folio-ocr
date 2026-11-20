import json
import math
import pathlib

import pytest

from folio_ocr.flags import FORMAT_NOISE, NO_TRANSCRIPTION, TRUNCATED
from folio_ocr.model_reply import (
    UNREADABLE,
    has_format_noise,
    logprob_confidence,
    parse_reply,
    status_error,
)

DATA = pathlib.Path(__file__).parent / "data" / "model_replies"


def fixture(name):
    body = (DATA / f"{name}.json").read_bytes()
    return body, json.loads(body.decode("utf-8"))


def body_of(content="Invoice 12", finish="stop", logprobs=None, **extra):
    choice = {"message": {"content": content, **extra}, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return json.dumps({"choices": [choice]}).encode("utf-8")


def lp(*values):
    return {"content": [{"token": "t", "logprob": v} for v in values]}


def test_en_exact():
    body, _ = fixture("en_exact")
    r = parse_reply(200, body)
    assert r.error is None and r.scores == ()
    assert r.text == "Invoice number: INV-2026-3706"
    assert r.flags == () and r.method == "logprobs"
    assert r.confidence == pytest.approx(math.exp(-0.000002861027041944908))


def test_ar_digits_exact():
    body, data = fixture("ar_digits_exact")
    r = parse_reply(200, body)
    assert r.text == data["choices"][0]["message"]["content"]
    assert r.text.endswith("".join(chr(c) for c in range(0x661, 0x665)))
    assert r.flags == () and r.error is None
    assert r.confidence == pytest.approx(math.exp(-0.2777141332626343), rel=1e-9)
    assert data["usage"]["completion_tokens"] != len(data["choices"][0]["logprobs"]["content"])


def test_mixed_visual_order_kept_raw():
    body, data = fixture("mixed_visual_order")
    r = parse_reply(200, body)
    assert r.text == data["choices"][0]["message"]["content"]
    assert r.flags == ()
    assert r.confidence == pytest.approx(math.exp(-0.30734148621559143))


@pytest.mark.parametrize(("finish", "flags"), [("length", (TRUNCATED,)), ("stop", ()), (None, ())])
def test_truncated(finish, flags):
    r = parse_reply(200, body_of(finish=finish))
    assert r.flags == flags and r.text == "Invoice 12"


def test_finish_reason_missing():
    body = json.dumps({"choices": [{"message": {"content": "a"}}]}).encode()
    assert parse_reply(200, body).flags == ()


def test_content_none_and_missing():
    for raw in (body_of(None), json.dumps({"choices": [{"message": {}}]}).encode()):
        r = parse_reply(200, raw)
        assert r.text == "" and r.flags == (NO_TRANSCRIPTION,) and r.error is None


def test_reasoning_only():
    r = parse_reply(200, body_of("", reasoning="I think..."))
    assert r.flags == (NO_TRANSCRIPTION,)
    r = parse_reply(200, body_of(None, reasoning_content="hmm"))
    assert r.flags == (NO_TRANSCRIPTION,)


def test_whitespace_content_kept():
    r = parse_reply(200, body_of("  "))
    assert r.text == "  " and r.flags == (NO_TRANSCRIPTION,)


@pytest.mark.parametrize(
    "content",
    [
        "<think>hmm</think>Invoice 12",
        "```Invoice```",
        "Here is the text: Invoice 12",
        "Invoice\n12",
        "Invoice\r12",
    ],
)
def test_format_noise(content):
    r = parse_reply(200, body_of(content))
    assert r.flags == (FORMAT_NOISE,) and r.text == content


def test_plain_text_no_flags():
    assert parse_reply(200, body_of("Invoice 12")).flags == ()


def test_flag_order():
    r = parse_reply(200, body_of("<think>", finish="length"))
    assert r.flags == (TRUNCATED, FORMAT_NOISE)


def test_has_format_noise_phrases():
    assert has_format_noise("  SURE thing")
    assert has_format_noise("Transcription: x")
    assert has_format_noise("a </THINK")
    assert not has_format_noise("Sunrise")


@pytest.mark.parametrize("value", [{"content": []}, None, {}, "x", [], {"content": "x"}])
def test_logprobs_absent_or_empty(value):
    assert logprob_confidence(value) is None
    r = parse_reply(200, body_of(logprobs=value) if value is not None else body_of())
    assert r.confidence is None and r.method == "none" and r.error is None


@pytest.mark.parametrize("bad", [True, 0.5, "x", None, float("inf")])
def test_logprobs_bad_entry(bad):
    assert logprob_confidence(lp(-0.1, bad)) is None


def test_logprobs_missing_key_and_non_dict_entry():
    assert logprob_confidence({"content": [{"token": "a"}]}) is None
    assert logprob_confidence({"content": [{"logprob": -0.1}, 3]}) is None


def test_logprobs_nan_in_raw_json():
    raw = b'{"choices":[{"message":{"content":"a"},"logprobs":{"content":[{"logprob":NaN}]}}]}'
    r = parse_reply(200, raw)
    assert r.confidence is None and r.method == "none" and r.text == "a"


def test_logprobs_min_and_int_zero():
    assert logprob_confidence(lp(-0.1, -2.0)) == pytest.approx(math.exp(-2.0))
    assert logprob_confidence(lp(0)) == 1.0


def test_statuses():
    assert status_error(401) != status_error(500)
    for s in (401, 403, 404, 429, 500, 503, 302, 418, 100):
        msg = status_error(s)
        assert "\n" not in msg and str(s) in msg
    assert "FOLIO_MODEL_URL" in status_error(404)
    assert "redirect" in status_error(302)
    assert "busy" in status_error(429)
    assert "credentials" in status_error(403)


def test_status_wins_over_body():
    r = parse_reply(500, body_of())
    assert r.error == status_error(500) and r.text == "" and r.flags == ()
    assert parse_reply(302, body_of()).error == status_error(302)
    assert parse_reply(199, body_of()).error == status_error(199)
    assert parse_reply(300, body_of()).error == status_error(300)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"null",
        b"[]",
        b'{"choices": []}',
        b'{"choices": {}}',
        b'{"choices": [1]}',
        b'{"choices": [{"message": "x"}]}',
        b'{"choices": [{}]}',
        b'{"choices": [{"message": {"content": [{"type": "text", "text": "a"}]}}]}',
        b'{"choices": [{"message": {"content": "a"}, "finish_reason": 5}]}',
        b"\xef\xbb\xbf" + body_of(),
        b"\xff\xfe",
        b"",
    ],
)
@pytest.mark.parametrize("status", [200, 204])
def test_unreadable(body, status):
    r = parse_reply(status, body)
    assert r.error == UNREADABLE and r.text == "" and r.flags == ()
    assert r.confidence is None and r.method == "none" and r.scores == ()


def test_error_never_holds_body():
    body = b"SENTINEL"
    assert "SENTINEL" not in parse_reply(500, body).error
    assert "SENTINEL" not in parse_reply(200, body).error
    assert "SENTINEL" not in parse_reply(200, b'{"choices": [1], "x": "SENTINEL"}').error


@pytest.mark.parametrize(
    "body",
    [b"[" * 6000 + b"]" * 6000, b'{"a":' * 6000 + b"1" + b"}" * 6000],
)
def test_json_nested_too_deep_is_unreadable(body):
    r = parse_reply(200, body)
    assert r.error == UNREADABLE and r.text == ""


def test_a_logprob_too_large_for_a_float_gives_no_confidence():
    logprobs = {"content": [{"token": "a", "logprob": 10**400}]}
    r = parse_reply(200, body_of(logprobs=logprobs))
    assert r.confidence is None and r.method == "none"
