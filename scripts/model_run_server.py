"""Talking to the configured model server (Ollama identity routes or a declared identity) for
the laptop model run: raw reply capture, server facts and reply-contract checks."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from folio_ocr import model_http, model_request, model_setup
from folio_ocr.model_reader import ModelConfig, ModelReader
from folio_ocr.model_reply import parse_reply
from folio_ocr.model_self_test_lines import LINES, SelfTestLine

_LABELS = {"en": "English", "ar": "Arabic"}


NO_KEY = "the model server gives no calibration key (version or digest missing)"


class ModelRunError(Exception):
    """A friendly one-line error the command line prints as "error: <msg>"."""


def _no_answer(exc: model_http.ModelCallError) -> ModelRunError:
    return ModelRunError(f"the model server did not answer: {exc}")


@dataclass(frozen=True)
class Exchange:
    request: bytes
    status: int | None
    body: bytes
    seconds: float
    error: str | None
    timed_out: bool = False


class Capture:
    """A `post` hook that records every request and reply, per thread."""

    def __init__(
        self,
        post: Callable[..., model_http.Reply] = model_http.post,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._post = post
        self._clock = clock
        self._local = threading.local()

    def _list(self) -> list[Exchange]:
        if not hasattr(self._local, "items"):
            self._local.items = []
        return self._local.items

    def post(self, target: model_http.Target, body: bytes, timeout_s: float) -> model_http.Reply:
        started = self._clock()
        try:
            reply = self._post(target, body, timeout_s)
        except model_http.ModelCallError as exc:
            seconds = self._clock() - started
            self._list().append(Exchange(body, None, b"", seconds, str(exc), exc.timed_out))
            raise
        self._list().append(Exchange(body, reply.status, reply.body, self._clock() - started, None))
        return reply

    def reset(self) -> None:
        self._local.items = []

    def take(self) -> tuple[Exchange, ...]:
        taken = tuple(self._list())
        self.reset()
        return taken


def reply_json(body: bytes) -> dict | None:
    """The body as a strict UTF-8 JSON object, else None."""
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _choice(body: bytes) -> dict:
    data = reply_json(body)
    choices = data.get("choices") if data else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def logprob_entries(body: bytes) -> list | None:
    logprobs = _choice(body).get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    return content if isinstance(content, list) else None


def finish_reason(body: bytes) -> str | None:
    value = _choice(body).get("finish_reason")
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def usage(body: bytes) -> dict:
    data = reply_json(body)
    used = data.get("usage") if data else None
    used = used if isinstance(used, dict) else {}
    details = used.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "prompt_tokens": _int_or_none(used.get("prompt_tokens")),
        "completion_tokens": _int_or_none(used.get("completion_tokens")),
        "cached_tokens": _int_or_none(details.get("cached_tokens")),
    }


def request_max_tokens(request: bytes) -> int | None:
    data = reply_json(request)
    return _int_or_none(data.get("max_tokens")) if data else None


def exchange_record(ex: Exchange) -> dict:
    message = _choice(ex.body).get("message")
    content = message.get("content") if isinstance(message, dict) else None
    used = usage(ex.body)
    return {
        "request_b64": base64.b64encode(ex.request).decode("ascii"),
        "request_sha256": hashlib.sha256(ex.request).hexdigest(),
        "max_tokens": request_max_tokens(ex.request),
        "status": ex.status,
        "body_b64": base64.b64encode(ex.body).decode("ascii"),
        "content": content if isinstance(content, str) else None,
        "finish_reason": finish_reason(ex.body),
        "entries": logprob_entries(ex.body),
        **used,
        "seconds": ex.seconds,
        "error": ex.error,
        "timed_out": ex.timed_out,
    }


def exchange_from_record(rec: dict) -> Exchange:
    return Exchange(
        base64.b64decode(rec["request_b64"]),
        rec["status"],
        base64.b64decode(rec["body_b64"]),
        rec["seconds"],
        rec["error"],
        rec.get("timed_out", False),
    )


def ollama_version(
    url: str,
    api_key: str | None,
    timeout_s: float,
    *,
    get: Callable[..., model_http.Reply] = model_http.get,
) -> str | None:
    """The full version string of the server's /api/version; ModelCallError propagates."""
    target = model_http.native_target(url, api_key, "version")
    if target is None:
        return None
    reply = get(target, timeout_s)
    if not 200 <= reply.status < 300:
        return None
    data = reply_json(reply.body)
    version = data.get("version") if data else None
    return version if isinstance(version, str) and version else None


def server_facts(
    config: ModelConfig, timeout_s: float, *, get: Callable[..., model_http.Reply] = model_http.get
) -> dict:
    """The model identity the run is keyed on: the server's own, else the declared one."""
    try:
        identity = model_setup.identify(
            config.url, config.api_key, config.name, timeout_s, config.declared, get=get
        )
        info = identity.info
        if info is None:
            raise ModelRunError(f"no calibration key can be formed: {identity.reason}")
        version = info.version
        if identity.source == "server":
            version = ollama_version(config.url, config.api_key, timeout_s, get=get) or version
    except model_http.ModelCallError as exc:
        raise _no_answer(exc) from None
    return {
        "kind": info.kind,
        "version": version,
        "version_major_minor": info.version,
        "digest": info.digest,
        "model": config.name,
        "host": config.host,
        "identity": identity.source,
    }


def _line_request(config: ModelConfig, line: SelfTestLine, tokens: int | None = None) -> bytes:
    width, height = Image.open(io.BytesIO(line.png)).size
    budget = model_request.max_tokens(width, height, line.languages) if tokens is None else tokens
    return model_request.request_body(config.name, line.png, budget)


def warm_up(
    config: ModelConfig,
    timeout_s: float = 180.0,
    *,
    post: Callable[..., model_http.Reply] = model_http.post,
) -> float:
    """Send the English test line once so the model is loaded; the seconds it took."""
    target = model_http.target(config.url, config.api_key)
    started = time.perf_counter()
    try:
        reply = post(target, _line_request(config, LINES[0]), timeout_s)
    except model_http.ModelCallError as exc:
        raise _no_answer(exc) from None
    if not 200 <= reply.status < 300:
        raise ModelRunError(f"the model server answered HTTP {reply.status}")
    return time.perf_counter() - started


def _entry_failure(entry: object) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("token"), str):
        return True
    value = entry.get("logprob")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return True
    return not math.isfinite(value) or value > 0


def logprob_failure(status: int, body: bytes, expected: str) -> str | None:
    """The first reply-contract rule an image reply breaks, or None when it keeps them all."""
    reading = parse_reply(status, body)
    if status != 200:
        return f"the status was {status}, not 200"
    if reading.error is not None:
        return "the reply could not be read"
    entries = logprob_entries(body)
    if not entries:
        return "the reply holds no logprob entries"
    if any(_entry_failure(entry) for entry in entries):
        return "a logprob entry is malformed or positive"
    if finish_reason(body) != "stop":
        return "the reply did not finish with stop"
    if reading.confidence is None:
        return "no confidence could be formed from the logprobs"
    if " ".join(reading.text.split()) != expected:
        return "the reply text differs from the test line"
    return None


def _check(name: str, required: bool, run: Callable[[], tuple[bool, str]]) -> dict:
    try:
        passed, detail = run()
    except model_http.ModelCallError as exc:
        passed, detail = False, f"no answer: {exc}"
    except ModelRunError as exc:
        passed, detail = False, str(exc)
    return {"name": name, "required": required, "passed": passed, "detail": detail}


def _facts_check(facts: dict | None, expect_version: str | None, expect_digest: str | None):
    for key, expected in (("version", expect_version), ("digest", expect_digest)):
        if expected is not None and facts[key] != expected:
            return False, f"{key} is {facts[key]}, expected {expected}"
    return True, f"{facts['kind']} {facts['version']}, digest {facts['digest']}"


def _line_check(config, line, post) -> tuple[bool, str]:
    target = model_http.target(config.url, config.api_key)
    reply = post(target, _line_request(config, line), 180.0)
    failure = logprob_failure(reply.status, reply.body, line.text)
    return (True, "logprobs and text are as expected") if failure is None else (False, failure)


def _truncation_check(config, post) -> tuple[bool, str]:
    target = model_http.target(config.url, config.api_key)
    reply = post(target, _line_request(config, LINES[0], 1), 180.0)
    reading = parse_reply(reply.status, reply.body)
    if reading.error is not None:
        return False, "the reply could not be read"
    flagged = "truncated" in reading.flags
    return flagged, "truncation is flagged" if flagged else "truncation was not flagged"


def _blank_check(config, post) -> tuple[bool, str]:
    out = io.BytesIO()
    Image.new("L", (200, 40), 255).save(out, format="PNG")
    body = model_request.request_body(
        config.name, out.getvalue(), model_request.max_tokens(200, 40, "en")
    )
    reply = post(model_http.target(config.url, config.api_key), body, 180.0)
    reading = parse_reply(reply.status, reply.body)
    if reading.error is not None:
        return True, f"no reading: {reading.error}"
    return True, f"{len(reading.text)} characters, flags {reading.flags}"


def contract_checks(
    config: ModelConfig,
    *,
    expect_version: str | None = None,
    expect_digest: str | None = None,
    post: Callable[..., model_http.Reply] = model_http.post,
    get: Callable[..., model_http.Reply] = model_http.get,
    timeout_s: float = 180.0,
) -> dict:
    """Run the reply-contract checks; a failing or unreachable server fails checks, never raises."""
    holder: dict = {}

    def facts() -> tuple[bool, str]:
        holder["facts"] = server_facts(config, timeout_s, get=get)
        return _facts_check(holder["facts"], expect_version, expect_digest)

    def send(target: model_http.Target, body: bytes, _timeout: float) -> model_http.Reply:
        return post(target, body, timeout_s)

    checks = [_check("server facts", True, facts)]
    for line in LINES:
        name = f"logprobs on an image request ({_LABELS[line.languages]})"
        checks.append(_check(name, True, lambda line=line: _line_check(config, line, send)))
    checks.append(_check("truncation flag", True, lambda: _truncation_check(config, send)))

    def self_test() -> tuple[bool, str]:
        result = ModelReader(config, post=post, get=get).self_test()
        return result.passed, result.detail

    checks.append(_check("self-test", True, self_test))
    checks.append(_check("blank image", False, lambda: _blank_check(config, send)))
    return {
        "checks": checks,
        "passed": all(c["passed"] for c in checks if c["required"]),
        "facts": holder.get("facts"),
    }
