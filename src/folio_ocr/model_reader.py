"""Reads text line images on the configured model server, with a self-test of the server."""

from __future__ import annotations

import dataclasses
import io
import math
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image

from folio_ocr import model_calibration, model_http, model_request, model_setup, model_taus
from folio_ocr.config import Settings
from folio_ocr.errors import ConfigError
from folio_ocr.metrics import char_errors
from folio_ocr.model_reply import ModelReading, parse_reply
from folio_ocr.model_self_test_lines import LINES
from folio_ocr.model_setup import Declared, Identity, ServerInfo
from folio_ocr.model_taus import LocalTaus, SelfTestReply
from folio_ocr.normalize import norm_v2

NAME_MISSING = "FOLIO_MODEL_NAME must be set when FOLIO_MODEL_URL is set"
_LABELS = {"en": "English", "ar": "Arabic"}


def line_time_limit(seconds: float) -> str:
    return f"the model server did not answer within {seconds:g} s (FOLIO_LINE_TIMEOUT_S)"


def page_time_limit(seconds: float) -> str:
    return f"the page time limit of {seconds:g} s ran out (FOLIO_PAGE_DEADLINE_S)"


def _sendable(value: str) -> bool:
    """False for text that cannot be written as UTF-8 (a lone surrogate, which is how an
    undecodable environment byte arrives)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _model_variables(settings: Settings) -> tuple[tuple[str, str | None], ...]:
    return (
        ("FOLIO_MODEL_URL", settings.model_url),
        ("FOLIO_MODEL_NAME", settings.model_name),
        ("FOLIO_MODEL_API_KEY", settings.model_api_key),
    )


@dataclass(frozen=True)
class ModelConfig:
    url: str = field(repr=False)  # may hold user:password
    name: str
    api_key: str | None = field(repr=False)
    line_timeout_s: float
    page_deadline_s: float
    two_read: bool
    declared: Declared | None = None  # FOLIO_MODEL_DIGEST + FOLIO_MODEL_SERVER
    local: LocalTaus | None = None  # the FOLIO_TAU_FILE thresholds

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.url).hostname or ""

    @classmethod
    def from_settings(
        cls, settings: Settings, local: LocalTaus | None = None
    ) -> ModelConfig | None:
        """The model settings, or None without a server. `local` = the FOLIO_TAU_FILE
        thresholds when the caller has already read them; otherwise the file is read and
        checked here (once per run), even when no server is configured."""
        if local is None and settings.tau_file is not None:
            local = model_taus.load(settings.tau_file)
        if settings.model_url is None:
            return None
        if settings.model_name is None:
            raise ConfigError(NAME_MISSING)
        for name, value in _model_variables(settings):
            if value is not None and not _sendable(value):
                raise ConfigError(f"{name} holds a character that cannot be sent to the server")
        return cls(
            settings.model_url,
            settings.model_name,
            settings.model_api_key,
            settings.line_timeout_s,
            settings.page_deadline_s,
            settings.two_read,
            settings.declared_identity,
            local,
        )


@dataclass(frozen=True)
class SelfTest:
    passed: bool
    detail: str  # one line, never the URL, key or the model's reply text
    replies: tuple[SelfTestReply, ...] = ()  # the answers, matched against local thresholds


def agreement(first: str, second: str) -> float:
    """How closely two readings of one line match, 0..1, after normalisation."""
    a, b = norm_v2(first), norm_v2(second)
    if not a:
        return 1.0 if not b else 0.0
    return max(0.0, 1 - char_errors(a, b).edits / len(a))


class ModelReader:
    """Reads line images on the configured model server; one per run or per server process.
    The self-test and the server's identity are asked once, at first use; safe across threads."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        post: Callable[..., model_http.Reply] = model_http.post,
        get: Callable[..., model_http.Reply] = model_http.get,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._target = model_http.target(config.url, config.api_key)
        self._post = post
        self._get = get
        self._clock = clock
        self._lock = threading.Lock()
        self._fit: Literal["pass", "fail"] | None = None
        self._replies: tuple[SelfTestReply, ...] = ()
        self._identity: Identity | None = None
        self._reachable: bool | None = None

    def now(self) -> float:
        return self._clock()

    def _call(
        self, image: Image.Image, languages: str, line_started: float, page_ends_at: float
    ) -> ModelReading:
        now = self._clock()
        line_left = self.config.line_timeout_s - (now - line_started)
        page_left = page_ends_at - now
        if page_left <= 0:
            return ModelReading("", (), error=page_time_limit(self.config.page_deadline_s))
        if line_left <= 0:
            return ModelReading("", (), error=line_time_limit(self.config.line_timeout_s))
        tokens = model_request.max_tokens(image.width, image.height, languages)
        body = model_request.request_body(self.config.name, model_request.png(image), tokens)
        try:
            reply = self._post(self._target, body, min(line_left, page_left))
        except model_http.ModelCallError as exc:
            self._reachable = False
            if not exc.timed_out:
                return ModelReading("", (), error=str(exc))
            message = (
                page_time_limit(self.config.page_deadline_s)
                if page_left < line_left
                else line_time_limit(self.config.line_timeout_s)
            )
            return ModelReading("", (), error=message)
        self._reachable = True
        return parse_reply(reply.status, reply.body)

    def read(self, image: Image.Image, languages: str, page_ends_at: float) -> ModelReading:
        """One line; the line time limit covers both reads when a second read is made."""
        line_started = self._clock()
        first = self._call(image, languages, line_started, page_ends_at)
        if first.error or first.method == "logprobs" or not self.config.two_read:
            return first
        altered = model_request.altered(image)
        second = self._call(altered, languages, line_started, page_ends_at)
        if second.error:
            return dataclasses.replace(first, confidence=None, method="none")
        return dataclasses.replace(
            first, confidence=agreement(first.text, second.text), method="two_read"
        )

    def self_test(self) -> SelfTest:
        """Read an English and an Arabic test line; each must come back exactly."""
        replies: list[SelfTestReply] = []
        for line in LINES:
            label = _LABELS[line.languages]
            image = Image.open(io.BytesIO(line.png))
            reading = self._call(image, line.languages, self._clock(), math.inf)
            if reading.error:
                detail = f"the {label} test line could not be read: {reading.error}"
                return SelfTest(False, detail, tuple(replies))
            replies.append(SelfTestReply(line.languages, reading.text, reading.confidence))
            if " ".join(reading.text.split()) != line.text:
                detail = f"the {label} test line came back different from its text"
                return SelfTest(False, detail, tuple(replies))
        return SelfTest(
            True, "an English and an Arabic test line were read exactly", tuple(replies)
        )

    def fit(self) -> Literal["pass", "fail"]:
        with self._lock:
            if self._fit is None:
                test = self.self_test()
                self._replies = test.replies
                self._fit = "pass" if test.passed else "fail"
            return self._fit

    def identity(self) -> Identity:
        """Where the model's digest and the server's version come from; asked once."""
        with self._lock:
            if self._identity is None:
                self._identity = model_setup.identify(
                    self.config.url,
                    self.config.api_key,
                    self.config.name,
                    self.config.line_timeout_s,
                    self.config.declared,
                    get=self._get,
                )
            return self._identity

    def server(self) -> ServerInfo | None:
        return self.identity().info

    def fingerprint(self) -> tuple[SelfTestReply, ...]:
        """The self-test answers of this run (runs the self-test once if needed)."""
        self.fit()
        return self._replies

    def tau(self, key: str) -> tuple[float, model_calibration.TauSource] | None:
        """The frozen threshold for exactly `key` and where it comes from, or None."""
        return model_calibration.tau_for(key, self.config.local, self.fingerprint())

    def calibration(self) -> tuple[Literal["uncalibrated", "ok"], str | None]:
        """("ok", None) when a threshold applies to this model, else ("uncalibrated", why)."""
        identity = self.identity()
        replies = self.fingerprint()
        return model_calibration.status(identity, self.config.name, self.config.local, replies)

    def calibrated(self) -> Literal["uncalibrated", "ok"]:
        return self.calibration()[0]

    def reachable(self) -> bool | None:
        """Whether the last line request got an HTTP reply; None before the first request."""
        return self._reachable

    def fit_so_far(self) -> Literal["pass", "fail", "untested"]:
        """The self-test result once it has run; never runs it and never waits for a running
        self-test (reading one attribute needs no lock)."""
        fit = self._fit
        return "untested" if fit is None else fit

    def calibrated_so_far(self) -> Literal["uncalibrated", "ok"]:
        """Whether a threshold is frozen for this setup, from what the server has said so far;
        never asks the server and never waits for it."""
        identity = self._identity
        if identity is None:
            return "uncalibrated"
        local, replies = self.config.local, self._replies
        return model_calibration.status(identity, self.config.name, local, replies)[0]

    def identity_so_far(self) -> Identity | None:
        """The identity once it has been asked; never asks it."""
        return self._identity
