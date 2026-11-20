"""Health of the configured model server: its self-test, where its identity comes from and
whether a confidence threshold applies to it. `folio doctor` shows these after the model
server line; the health report shows `model_health`."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from folio_ocr import calibration, model_calibration, model_taus
from folio_ocr.config import Settings, is_loopback
from folio_ocr.doctor import Check
from folio_ocr.errors import ConfigError
from folio_ocr.model_calibration import TauSource
from folio_ocr.model_reader import ModelConfig, ModelReader
from folio_ocr.model_setup import Identity

UNCALIBRATED = (
    "uncalibrated (no frozen confidence threshold for this model setup yet: every line it "
    "reads is flagged low_confidence)"
)


def _uncalibrated(reason: str | None) -> str:
    if reason is None or reason == model_calibration.NO_TAU:
        return UNCALIBRATED
    return f"uncalibrated ({reason}: every line it reads is flagged low_confidence)"


def _identity_detail(identity: Identity) -> str:
    info = identity.info
    if info is None:
        return f"none: {identity.reason}"
    return f"{identity.source} {info.kind} {info.version}, digest {info.digest[:12]}"


def _taus_check(local: model_taus.LocalTaus) -> Check:
    shadowed = len(model_taus.shadowed(local, calibration.TAUS))
    detail = f"local file sha256 {local.sha256[:12]}: {len(local.entries)} entries, {shadowed}"
    return Check("model taus", "info", detail + " shadowed by the package table")


def _calibrated(identity: Identity, config: ModelConfig, replies) -> str:
    """`ok` with where the applying threshold comes from."""
    source = model_calibration.applying(identity, config.name, config.local, replies)
    if source is None or source.file_sha256 is None:
        return "ok (package table)"
    return f"ok (local file sha256 {source.file_sha256[:12]})"


def _server_checks(model: ModelReader) -> list[Check]:
    test = model.self_test()
    identity = model.identity()
    config = model.config
    state, reason = model_calibration.status(identity, config.name, config.local, test.replies)
    reader_line = (
        Check("reader model", "ok", _calibrated(identity, config, test.replies))
        if state == "ok"
        else Check("reader model", "info", _uncalibrated(reason))
    )
    return [
        Check("model self-test", "ok" if test.passed else "fail", test.detail),
        Check("model identity", "info", _identity_detail(identity)),
        reader_line,
    ]


def model_checks(
    settings: Settings, *, reader: Callable[[ModelConfig], ModelReader] = ModelReader
) -> list[Check]:
    """The model lines of `folio doctor`: `model taus` when FOLIO_TAU_FILE is set, then, with a
    model server, `model self-test` (ok | fail), `model identity` (server | declared | none)
    and `reader model` (ok | uncalibrated, with the reason). Sends the two built-in test lines
    and asks the server for its identity."""
    try:
        local = None if settings.tau_file is None else model_taus.load(settings.tau_file)
    except ConfigError as exc:
        return [Check("model taus", "fail", str(exc))]
    try:
        config = ModelConfig.from_settings(settings, local)
    except ConfigError as exc:
        return [Check("model self-test", "fail", str(exc))]
    taus = [] if local is None else [_taus_check(local)]
    return taus if config is None else taus + _server_checks(reader(config))


@dataclass(frozen=True)
class ModelHealth:
    """The model server as the health report shows it."""

    configured: bool
    reachable: bool | None  # the last line request got an HTTP reply; None before any request
    name: str | None
    host: str | None  # host name only, never the URL, user, password or key
    leaves_machine: bool | None  # the host is not a loopback address
    fit: Literal["pass", "fail", "untested"]  # the self-test result once it has run
    calibrated: Literal["uncalibrated", "ok"]
    identity: Literal["server", "declared", "none"] | None = None  # None until it was asked
    taus: tuple[TauSource, ...] = ()  # the thresholds held for this model, with their source


def model_health(settings: Settings, reader: ModelReader | None) -> ModelHealth:
    """What is known about the configured model server, without sending anything: `reader` is
    the process's ModelReader (None when no server is configured or none was made yet)."""
    if settings.model_url is None:
        return ModelHealth(False, None, None, None, None, "untested", "uncalibrated")
    host = settings.model_host or ""
    identity = None if reader is None else reader.identity_so_far()
    taus: tuple[TauSource, ...] = ()
    if identity is not None and reader is not None:
        taus = model_calibration.sources(identity, reader.config.name, reader.config.local)
    return ModelHealth(
        configured=True,
        reachable=None if reader is None else reader.reachable(),
        name=settings.model_name,
        host=host,
        leaves_machine=not is_loopback(host),
        fit="untested" if reader is None else reader.fit_so_far(),
        calibrated="uncalibrated" if reader is None else reader.calibrated_so_far(),
        identity=None if identity is None else identity.source,
        taus=taus,
    )
