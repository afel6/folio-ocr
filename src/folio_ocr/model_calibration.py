"""Which frozen threshold judges a model line, and where it comes from.

The package table (`calibration.TAUS`) is looked up first. A company's own local tau file
(FOLIO_TAU_FILE) is used only for keys the package does not hold, and only while this run's
self-test answers match those recorded with the entry."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from folio_ocr import calibration, model_setup, model_taus
from folio_ocr.document import Calibration
from folio_ocr.model_setup import Identity
from folio_ocr.model_taus import LocalTaus, SelfTestReply

NO_KEY = "the model name or the declared server version cannot form a calibration key"
NOT_LIKE_CALIBRATION = "the model does not answer like its calibration"
NO_TAU = "no frozen confidence threshold for this model setup yet"


@dataclass(frozen=True)
class TauSource:
    key: str
    source: Literal["package", "local"]
    file_sha256: str | None  # the local file's sha256; None for the package table


@dataclass(frozen=True)
class ModelCalibration(Calibration):
    """The threshold that judged a model document, with where it came from."""

    source: Literal["package", "local"] | None = None
    file_sha256: str | None = None


def tau_for(
    key: str, local: LocalTaus | None, replies: Sequence[SelfTestReply]
) -> tuple[float, TauSource] | None:
    """The threshold for exactly `key`, or None when none applies."""
    package = calibration.lookup(key, calibration.TAUS)
    if package is not None:
        return package, TauSource(key, "package", None)
    entry = None if local is None else local.entries.get(key)
    if entry is None or not model_taus.matches(entry, replies):
        return None
    return entry.tau, TauSource(key, "local", local.sha256)


def record(key: str, found: tuple[float, TauSource] | None) -> ModelCalibration:
    """The Engine record of a model document judged under `key`."""
    if found is None:
        return ModelCalibration(key, None)
    tau, source = found
    return ModelCalibration(key, tau, source.source, source.file_sha256)


def sources(identity: Identity, model_name: str, local: LocalTaus | None) -> tuple[TauSource, ...]:
    """Every threshold held for this model's identity (any method and page preparation): the
    package's, then the local file's that the package does not shadow."""
    package = model_setup.matching_keys(identity.info, model_name, calibration.TAUS)
    found = [TauSource(key, "package", None) for key in package]
    if local is not None:
        shadowed = set(model_taus.shadowed(local, calibration.TAUS))
        mine = model_setup.matching_keys(identity.info, model_name, local.entries)
        found += [TauSource(k, "local", local.sha256) for k in mine if k not in shadowed]
    return tuple(found)


def applying(
    identity: Identity,
    model_name: str,
    local: LocalTaus | None,
    replies: Sequence[SelfTestReply],
) -> TauSource | None:
    """A threshold that applies to this model: a package one, else a local one whose
    self-test answers match this run's."""
    held = sources(identity, model_name, local)
    package = [s for s in held if s.source == "package"]
    if package:
        return package[0]
    if local is None:
        return None
    return next((s for s in held if model_taus.matches(local.entries[s.key], replies)), None)


def status(
    identity: Identity,
    model_name: str,
    local: LocalTaus | None,
    replies: Sequence[SelfTestReply],
) -> tuple[Literal["uncalibrated", "ok"], str | None]:
    """("ok", None) when a threshold applies to this model, else ("uncalibrated", why)."""
    if identity.info is None:
        return "uncalibrated", identity.reason
    if model_setup.model_setup(identity.info, model_name, "logprobs", "") is None:
        return "uncalibrated", NO_KEY
    if applying(identity, model_name, local, replies) is not None:
        return "ok", None
    held_locally = any(s.source == "local" for s in sources(identity, model_name, local))
    return "uncalibrated", NOT_LIKE_CALIBRATION if held_locally else NO_TAU
