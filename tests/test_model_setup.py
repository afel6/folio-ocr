"""Tests for the model server's self-description and the calibration setup key."""

import json
import socket
import types

import pytest
from fake_model_server import Answer, FakeModelServer

from folio_ocr import calibration, model_request
from folio_ocr.model_reply import ORDER_RULE
from folio_ocr.model_setup import (
    Identity,
    ServerInfo,
    identify,
    is_calibrated,
    matching_keys,
    model_setup,
    server_info,
)

NAME = "qwen3-vl:8b-instruct-q4_K_M"
DIGEST = "70e58e37beb15464e19aa903dedb2cba828b89c0fb186f75ff2738c7347b96eb"
INFO = ServerInfo("ollama", "0.40", DIGEST)


def _tags(*entries):
    return {"models": list(entries)}


def _entry(name=NAME, digest=DIGEST):
    return {"name": name, "model": name, "digest": digest}


def _router(version=None, tags=None, version_status=200, tags_raw=None):
    version = {"version": "0.40.16"} if version is None else version
    tags = _tags(_entry()) if tags is None else tags

    def answer(seen):
        if seen.target == "/api/version":
            return Answer(status=version_status, body=json.dumps(version).encode())
        if seen.target == "/api/tags":
            return Answer(body=tags_raw if tags_raw is not None else json.dumps(tags).encode())
        return Answer(status=404)

    return answer


def _info_with(name=NAME, **kwargs):
    with FakeModelServer(_router(**kwargs)) as server:
        return server_info(server.url, None, name, 5.0)


def test_server_info_reads_version_and_digest():
    with FakeModelServer(_router()) as server:
        info = server_info(server.url, None, NAME, 5.0)
    assert info == INFO
    assert [(s.method, s.target, s.body) for s in server.seen] == [
        ("GET", "/api/version", b""),
        ("GET", "/api/tags", b""),
    ]


def test_server_info_sends_the_key():
    with FakeModelServer(_router()) as server:
        server_info(server.url, "sk-1", NAME, 5.0)
    assert [s.headers["authorization"] for s in server.seen] == ["Bearer sk-1"] * 2


def test_server_info_needs_a_v1_base_url_and_sends_nothing():
    with FakeModelServer(_router()) as server:
        assert server_info(server.url[:-2] + "v2", None, NAME, 5.0) is None
    assert server.seen == []


def test_none_when_version_is_missing_or_bad():
    assert _info_with(version_status=404) is None
    assert _info_with(version={"version": "garbage"}) is None
    assert _info_with(version={"version": 40}) is None
    assert _info_with(version=[1]) is None


def test_none_when_the_model_is_not_listed():
    assert _info_with(tags=_tags(_entry("other:1"))) is None
    assert _info_with(tags={"models": "x"}) is None
    assert _info_with(tags=[1]) is None


def test_exact_name_match_is_required():
    assert _info_with(name="qwen3-vl") is None


def test_none_when_listed_twice_with_different_digests():
    assert _info_with(tags=_tags(_entry(), _entry(digest="0" * 64))) is None


def test_same_digest_listed_twice_is_fine():
    assert _info_with(tags=_tags(_entry(), _entry())) == INFO


@pytest.mark.parametrize("digest", [DIGEST.upper(), DIGEST[:63], 12345, None])
def test_none_when_digest_is_not_64_lowercase_hex(digest):
    assert _info_with(tags=_tags(_entry(digest=digest))) is None


def test_none_when_tags_is_not_json():
    assert _info_with(tags_raw=b"not json") is None


def test_none_when_the_port_refuses():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert server_info(f"http://127.0.0.1:{port}/v1", None, NAME, 2.0) is None


def test_setup_key_is_exact():
    setup = model_setup(INFO, NAME, "logprobs", "abc")
    preprocessing = calibration.preprocessing_hash(
        {"page": "abc", **model_request.crop_constants(), "order_rule": ORDER_RULE}
    )
    args = " ".join(model_request.arguments())
    head = f"model|{NAME}|{DIGEST}|logprobs:exp-min-logprob|ollama-0.40"
    expected = f"{head}|{args}|{preprocessing}"
    assert setup.key == expected


def test_two_read_key_differs_only_in_aggregation():
    a = model_setup(INFO, NAME, "logprobs", "abc").key.split("|")
    b = model_setup(INFO, NAME, "two_read", "abc").key.split("|")
    assert b[3] == "two_read:agreement"
    assert a[:3] + a[4:] == b[:3] + b[4:]


def test_no_key_cases():
    assert model_setup(INFO, NAME, "none", "abc") is None
    assert model_setup(None, NAME, "logprobs", "abc") is None
    assert model_setup(INFO, "a|b", "logprobs", "abc") is None


def test_key_follows_page_preprocessing_and_prompt(monkeypatch):
    base = model_setup(INFO, NAME, "logprobs", "abc").key
    assert model_setup(INFO, NAME, "logprobs", "abd").key != base
    monkeypatch.setattr(model_request, "PROMPT_SHA256", "f" * 64)
    assert model_setup(INFO, NAME, "logprobs", "abc").key != base


def _table(*keys):
    return calibration.checked_table(dict.fromkeys(keys, 0.5))


def test_is_calibrated_false_for_empty_table_and_no_info():
    assert is_calibrated(INFO, NAME, _table()) is False
    key = model_setup(INFO, NAME, "logprobs", "abc").key
    assert is_calibrated(None, NAME, _table(key)) is False


def test_is_calibrated_true_for_exact_key_and_other_preprocessing():
    key = model_setup(INFO, NAME, "logprobs", "abc").key
    assert is_calibrated(INFO, NAME, _table(key)) is True
    other = model_setup(INFO, NAME, "logprobs", "zzz").key
    assert is_calibrated(INFO, NAME, _table(other)) is True


def test_is_calibrated_false_when_digest_version_or_name_differ():
    table = _table(model_setup(INFO, NAME, "logprobs", "abc").key)
    assert is_calibrated(ServerInfo("ollama", "0.40", "1" * 64), NAME, table) is False
    assert is_calibrated(ServerInfo("ollama", "0.41", DIGEST), NAME, table) is False
    assert is_calibrated(INFO, "qwen3-vl", table) is False


def test_is_calibrated_defaults_to_the_frozen_table(monkeypatch):
    monkeypatch.setattr(calibration, "TAUS", _table())  # the package table may hold this model
    assert is_calibrated(INFO, NAME) is False
    key = model_setup(INFO, NAME, "logprobs", "abc").key
    monkeypatch.setattr(calibration, "TAUS", _table(key))
    assert is_calibrated(INFO, NAME) is True


def test_none_when_the_version_body_is_nested_too_deep():
    deep = b"[" * 6000 + b"]" * 6000
    with FakeModelServer(Answer(body=deep)) as server:
        assert server_info(server.url, None, NAME, 5.0) is None


@pytest.mark.parametrize("digest", [[], {}, [DIGEST], {"a": DIGEST}])
def test_a_list_or_dict_digest_is_no_digest_and_does_not_raise(digest):
    assert _info_with(tags=_tags(_entry(digest=digest))) is None


def test_one_digest_among_entries_with_a_bad_one_is_none():
    assert _info_with(tags=_tags(_entry(), _entry(digest=[]))) is None


# ---- identity: the server first, a declaration only when its routes are absent ----

GOLDEN_HEAD = f"model|{NAME}|{DIGEST}|"
GOLDEN_ARGS = (
    "prompt_sha256=9f48cab462e9a79d9f1c8d27e81cb8b9cbdf1b8dddcd0b5f2ec6a12e4696c6d1 "
    "temperature=0 seed=42 logprobs=true stream=false max_tokens_base=16 "
    "max_tokens_latin=3 max_tokens_arabic=5"
)
GOLDEN_TEXT_LAYER = "ec18aef6d7ef9a142dc4b6f80e1f0456cfa9bcf8ec8c3e5b22c3df926d44e2cb"
GOLDEN_X = "60f11c11e7c2ba1ed022aa15ef0f1d41ba2299d491cc8627c6a292cbe5f989c4"
DIFFERS = "FOLIO_MODEL_DIGEST or FOLIO_MODEL_SERVER differs from what the server reports"
NOT_LISTED = "the server does not list FOLIO_MODEL_NAME exactly once with a sha256 digest"


def test_released_keys_are_pinned_byte_for_byte():
    from folio_ocr import model_pages

    text = model_pages.TEXT_LAYER_PREPROCESSING
    tail = f"ollama-0.40|{GOLDEN_ARGS}|"
    logprobs = model_setup(INFO, NAME, "logprobs", text).key
    assert logprobs == f"{GOLDEN_HEAD}logprobs:exp-min-logprob|{tail}{GOLDEN_TEXT_LAYER}"
    two_read = model_setup(INFO, NAME, "two_read", text).key
    assert two_read == f"{GOLDEN_HEAD}two_read:agreement|{tail}{GOLDEN_TEXT_LAYER}"
    page_x = model_setup(INFO, NAME, "logprobs", "x").key
    assert page_x == f"{GOLDEN_HEAD}logprobs:exp-min-logprob|{tail}{GOLDEN_X}"


def _declared(kind="ollama", version="0.40", digest=DIGEST):
    return types.SimpleNamespace(kind=kind, version=version, digest=digest)


def _identify(server_or_url, declared=None, timeout_s=5.0, key=None):
    url = getattr(server_or_url, "url", server_or_url)
    return identify(url, key, NAME, timeout_s, declared)


def _routes(version=(200, b'{"version": "0.40.16"}'), tags=None):
    tags = (200, json.dumps(_tags(_entry())).encode()) if tags is None else tags

    def answer(seen):
        status, body = {"/api/version": version, "/api/tags": tags}.get(seen.target, (404, b""))
        return Answer(status=status, body=body)

    return answer


def test_a_declared_key_never_equals_the_server_key():
    declared = ServerInfo("ollama", "0.40", DIGEST, declared=True)
    assert declared.version_slot == "declared:ollama-0.40"
    assert INFO.version_slot == "ollama-0.40"
    a = model_setup(declared, NAME, "logprobs", "abc").key
    b = model_setup(INFO, NAME, "logprobs", "abc").key
    assert a != b
    assert a.split("|")[4] == "declared:ollama-0.40"
    assert is_calibrated(declared, NAME, _table(a)) is True
    assert is_calibrated(INFO, NAME, _table(a)) is False
    assert is_calibrated(declared, NAME, _table(b)) is False
    assert matching_keys(declared, NAME, [a, b]) == (a,)
    assert matching_keys(INFO, NAME, [a, b]) == (b,)


def test_matching_keys_picks_only_this_model_digest_version_and_arguments(monkeypatch):
    mine = [model_setup(INFO, NAME, m, p).key for m in ("logprobs", "two_read") for p in "ab"]
    other_digest = model_setup(ServerInfo("ollama", "0.40", "1" * 64), NAME, "logprobs", "a").key
    other_version = model_setup(ServerInfo("ollama", "0.41", DIGEST), NAME, "logprobs", "a").key
    other_name = model_setup(INFO, "qwen3-vl", "logprobs", "a").key
    with monkeypatch.context() as patch:
        patch.setattr(model_request, "PROMPT_SHA256", "f" * 64)
        other_args = model_setup(INFO, NAME, "logprobs", "a").key
    keys = [*mine, other_digest, other_version, other_name, other_args, "junk"]
    assert matching_keys(INFO, NAME, keys) == tuple(mine)
    assert matching_keys(None, NAME, keys) == ()


def test_routes_present_and_matching_declaration_gives_a_server_key():
    with FakeModelServer(_routes()) as server:
        result = _identify(server, _declared())
    assert result == Identity(INFO, "server", None)
    assert result.info.declared is False


@pytest.mark.parametrize(
    "declared",
    [_declared(digest="2" * 64), _declared(version="0.41"), _declared(kind="vllm")],
)
def test_routes_present_and_differing_declaration_gives_no_key(declared):
    with FakeModelServer(_routes()) as server:
        result = _identify(server, declared)
    assert result == Identity(None, "none", DIFFERS)


def test_both_routes_404_uses_the_declaration():
    with FakeModelServer(Answer(status=404)) as server:
        result = _identify(server, _declared())
    assert result == Identity(ServerInfo("ollama", "0.40", DIGEST, declared=True), "declared", None)


def test_a_declared_forty_hex_digest_is_used_as_given():
    digest = "a1b2c3d4e5" * 4
    with FakeModelServer(Answer(status=404)) as server:
        result = _identify(server, _declared(kind="vllm", version="0.6", digest=digest))
    assert result.info == ServerInfo("vllm", "0.6", digest, declared=True)
    assert result.info.version_slot == "declared:vllm-0.6"


def test_both_routes_404_without_declaration_asks_for_one():
    with FakeModelServer(Answer(status=404)) as server:
        result = _identify(server)
    assert result == Identity(
        None,
        "none",
        "the server reports no model identity; set FOLIO_MODEL_DIGEST and FOLIO_MODEL_SERVER "
        "to declare it",
    )


def test_a_base_url_without_v1_uses_the_declaration_and_sends_nothing():
    with FakeModelServer(_routes()) as server:
        result = _identify(server.url[:-2] + "v2", _declared())
    assert result.source == "declared"
    assert result.info.declared is True
    assert server.seen == []


def test_one_missing_route_is_no_key_even_with_a_declaration():
    reason = "the server answered its model identity routes with HTTP 404/200"
    with FakeModelServer(_routes(version=(404, b""))) as server:
        assert _identify(server, _declared()) == Identity(None, "none", reason)
    reason = "the server answered its model identity routes with HTTP 200/404"
    with FakeModelServer(_routes(tags=(404, b""))) as server:
        assert _identify(server, _declared()) == Identity(None, "none", reason)


@pytest.mark.parametrize("which", ["version", "tags"])
def test_a_server_error_is_no_key_even_with_a_declaration(which):
    with FakeModelServer(_routes(**{which: (500, b"SENTINEL")})) as server:
        result = _identify(server, _declared())
    expected = "500/200" if which == "version" else "200/500"
    assert result == Identity(
        None, "none", f"the server answered its model identity routes with HTTP {expected}"
    )


def test_a_timeout_is_no_key_even_with_a_declaration():
    with FakeModelServer(Answer(delay_s=1.0, body=b"{}")) as server:
        result = _identify(server, _declared(), timeout_s=0.1)
    assert result.info is None and result.source == "none"
    assert result.reason.startswith("the server's model identity could not be read: ")


@pytest.mark.parametrize("tags", [_tags(), _tags(_entry(), _entry(digest="3" * 64))])
def test_a_model_missing_or_listed_twice_is_no_key_even_with_a_declaration(tags):
    with FakeModelServer(_routes(tags=(200, json.dumps(tags).encode()))) as server:
        result = _identify(server, _declared())
    assert result == Identity(None, "none", NOT_LISTED)


def test_an_unreadable_version_is_no_key_even_with_a_declaration():
    with FakeModelServer(_routes(version=(200, b'{"version": "garbage"}'))) as server:
        result = _identify(server, _declared())
    assert result == Identity(None, "none", "the server reported a version this tool cannot read")


def test_server_info_is_the_identity_info():
    with FakeModelServer(_routes()) as server:
        assert server_info(server.url, None, NAME, 5.0, declared=_declared()) == INFO
    with FakeModelServer(Answer(status=404)) as server:
        declared = server_info(server.url, None, NAME, 5.0, declared=_declared())
        assert declared == ServerInfo("ollama", "0.40", DIGEST, declared=True)
        assert server_info(server.url, None, NAME, 5.0) is None
    with FakeModelServer(_routes(tags=(500, b""))) as server:
        assert server_info(server.url, None, NAME, 5.0, declared=_declared()) is None


def test_no_reason_leaks_the_url_key_or_a_reply_body():
    bad = [
        _routes(version=(500, b"SENTINEL")),
        _routes(tags=(200, b"SENTINEL")),
        _routes(version=(200, b"SENTINEL")),
        Answer(status=404, body=b"SENTINEL"),
        Answer(delay_s=1.0),
    ]
    for answer in bad:
        with FakeModelServer(answer) as server:
            result = _identify(server, timeout_s=0.1, key="Bearer-secret-key")
        assert result.reason is not None
        for leak in ("SENTINEL", str(server.port), "127.0.0.1", "Bearer", "secret"):
            assert leak not in result.reason
    refused = identify("http://127.0.0.1:9/v1", "k", NAME, 1.0)
    assert refused.reason is not None and "127.0.0.1" not in refused.reason
