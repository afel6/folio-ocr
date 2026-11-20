"""The local tau file: confidence thresholds a company froze for its own model.

The file (FOLIO_TAU_FILE, schema `folio-taus/1`) is kept apart from the package table
`calibration.TAUS`. Each entry holds the threshold, how it was measured, and the answers
the model gave to a short self-test; an entry applies only while the model still answers
the self-test the same way.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from folio_ocr import calibration
from folio_ocr.errors import ConfigError

SCHEMA = "folio-taus/1"
METHODS = ("logprobs", "two_read")
MIN_BAD_LINES = 200
MIN_DOCUMENTS = 40
SELF_TEST_LINES = ("en", "ar")

_NUMBER_FIELDS = ("tau", "dev_bound", "drift_tolerance")
_WHOLE_FIELDS = ("bad_lines", "documents")
_TEXT_FIELDS = ("method", "dev_manifest_sha256")


@dataclass(frozen=True)
class SelfTestReply:
    """One answer of the model self-test."""

    line: str
    text: str
    score: float | None


@dataclass(frozen=True)
class TauEntry:
    key: str
    tau: float
    method: str
    bad_lines: int
    documents: int
    dev_bound: float
    dev_manifest_sha256: str
    self_test: tuple[SelfTestReply, ...]
    drift_tolerance: float


@dataclass(frozen=True)
class LocalTaus:
    sha256: str
    entries: Mapping[str, TauEntry]


def _refuse(reason: str) -> ConfigError:
    return ConfigError(f"FOLIO_TAU_FILE is not a valid {SCHEMA} file: {reason}")


def _no_repeats(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("a JSON key repeats")
        result[key] = value
    return result


def _no_constant(_name: str) -> Any:
    raise ValueError("a non-finite number")


def _is_number(value: object) -> bool:
    """A JSON number that is a finite float (1e999 and huge whole numbers are not)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reply(item: object, method: str) -> SelfTestReply:
    if not isinstance(item, dict):
        raise _refuse("a self_test line is not an object")
    if "score" not in item:
        raise _refuse("a self_test line has no score")
    line, text, score = item.get("line"), item.get("text"), item["score"]
    if not isinstance(line, str) or not isinstance(text, str):
        raise _refuse("a self_test line or text is not a string")
    if score is None:
        if method != "two_read":
            raise _refuse("a self_test score is null")
    elif not _is_number(score):
        raise _refuse("a self_test score is not a number")
    return SelfTestReply(line, text, None if score is None else float(score))


def _self_test(raw: object, method: str) -> tuple[SelfTestReply, ...]:
    if not isinstance(raw, list):
        raise _refuse("self_test is not a list")
    replies = [_reply(item, method) for item in raw]
    by_line = {r.line: r for r in replies}
    if len(replies) != len(SELF_TEST_LINES) or set(by_line) != set(SELF_TEST_LINES):
        raise _refuse("self_test must hold exactly one en and one ar line")
    return tuple(by_line[line] for line in SELF_TEST_LINES)


def _check_fields(raw: dict[str, Any]) -> None:
    for name in (*_NUMBER_FIELDS, *_WHOLE_FIELDS, *_TEXT_FIELDS, "self_test"):
        if name not in raw:
            raise _refuse(f"{name} is missing")
    for name in _NUMBER_FIELDS:
        if not _is_number(raw[name]):
            raise _refuse(f"{name} is not a number (or is a boolean where a number is expected)")
    for name in _WHOLE_FIELDS:
        if not _is_whole(raw[name]):
            raise _refuse(f"{name} is not a whole number")
    for name in _TEXT_FIELDS:
        if not isinstance(raw[name], str):
            raise _refuse(f"{name} is not a string")
    if raw["method"] not in METHODS:
        raise _refuse("method is not logprobs or two_read")
    if raw["bad_lines"] < MIN_BAD_LINES:
        raise _refuse(f"bad_lines is below {MIN_BAD_LINES}")
    if raw["documents"] < MIN_DOCUMENTS:
        raise _refuse(f"documents is below {MIN_DOCUMENTS}")


def _entry(key: str, raw: object) -> TauEntry:
    if not key.startswith("model|"):
        raise _refuse("an entry key does not start with model|")
    if not isinstance(raw, dict):
        raise _refuse("an entry is not an object")
    _check_fields(raw)
    return TauEntry(
        key=key,
        tau=raw["tau"],
        method=raw["method"],
        bad_lines=raw["bad_lines"],
        documents=raw["documents"],
        dev_bound=raw["dev_bound"],
        dev_manifest_sha256=raw["dev_manifest_sha256"],
        self_test=_self_test(raw["self_test"], raw["method"]),
        drift_tolerance=raw["drift_tolerance"],
    )


def _decode(data: bytes) -> object:
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_no_repeats, parse_constant=_no_constant
        )
    except UnicodeDecodeError:
        raise _refuse("it is not UTF-8") from None
    except RecursionError:
        raise _refuse("it is nested too deeply") from None
    except ValueError as error:
        reason = str(error) if str(error) in ("a JSON key repeats", "a non-finite number") else None
        raise _refuse(reason or "it is not JSON") from None


def parse(data: bytes) -> LocalTaus:
    """Parse and check the bytes of a tau file; raises ConfigError when it is refused."""
    document = _decode(data)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise _refuse("the schema field is missing or different")
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, dict):
        raise _refuse("entries is missing or not an object")
    entries = {key: _entry(key, raw) for key, raw in raw_entries.items()}
    try:
        calibration.checked_table({key: e.tau for key, e in entries.items()})
    except (ValueError, OverflowError):
        raise _refuse("a tau is outside 0 to 1 or not finite") from None
    return LocalTaus(hashlib.sha256(data).hexdigest(), MappingProxyType(entries))


def load(path: Path) -> LocalTaus:
    """Read and parse the tau file at `path`."""
    try:
        data = path.read_bytes()
    except OSError:
        raise ConfigError("FOLIO_TAU_FILE cannot be read") from None
    return parse(data)


def matches(entry: TauEntry, replies: Sequence[SelfTestReply]) -> bool:
    """True when this run's self-test answers agree with the recorded ones."""
    by_line = {r.line: r for r in replies}
    for recorded in entry.self_test:
        reply = by_line.get(recorded.line)
        if reply is None or reply.text != recorded.text:
            return False
        if entry.method == "logprobs":
            if reply.score is None or recorded.score is None:
                return False
            if abs(reply.score - recorded.score) > entry.drift_tolerance:
                return False
    return True


def shadowed(local: LocalTaus, package: Mapping[str, float]) -> tuple[str, ...]:
    """File keys that the package table also holds, in file order."""
    return tuple(key for key in local.entries if key in package)
