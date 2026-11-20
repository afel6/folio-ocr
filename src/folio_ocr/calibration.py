"""Frozen confidence thresholds per OCR reader setup, and the low-confidence decision.

A line is flagged low-confidence unless a threshold (tau) is frozen for the exact
reader setup and every word score is usable and at least tau. Missing
thresholds and unusable scores both fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

LOW_CONFIDENCE = "low_confidence"


def checked_table(table: Mapping[str, float]) -> Mapping[str, float]:
    """Validate a setup-key -> tau table and return a read-only copy."""
    for key, tau in table.items():
        if not isinstance(key, str) or not key:
            raise ValueError("threshold table keys must be non-empty strings")
        if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
            raise ValueError("threshold values must be finite and within 0 to 1")
    return MappingProxyType(dict(table))


# Setup key -> frozen tau. No threshold is frozen yet; calibration fills it later.
TAUS: Mapping[str, float] = checked_table({"tesseract|lstm|ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896+8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba|min|5.4|-l ara+eng --psm 13 --oem 1|2603f2951dfdde7bf673852f7cce32910a508d22b466bcbf30f977868e2e0b91": 0.968})


def table_sha256(table: Mapping[str, float] = TAUS) -> str:
    """Hex sha256 of the canonical JSON form of a threshold table."""
    text = json.dumps(
        dict(sorted(table.items())), separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def version_major_minor(version: str) -> str:
    """Reduce a version such as "v0.9.0" or "5.4.0.2024" to "major.minor"."""
    match = re.fullmatch(r"v?([0-9]+)\.([0-9]+)(\..*)?", version)
    if match is None:
        raise ValueError("version must look like major.minor[.more]")
    return f"{match.group(1)}.{match.group(2)}"


def preprocessing_hash(constants: Mapping[str, str | int | float]) -> str:
    """Hex sha256 of the preprocessing constants, independent of insertion order."""
    text = json.dumps(
        constants, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Setup:
    """Everything a threshold depends on; `key` identifies it exactly."""

    reader: str
    architecture: str
    weights_sha256: str
    aggregation: str
    version: str
    arguments: tuple[str, ...]
    preprocessing: str

    def __post_init__(self) -> None:
        required = (
            self.reader,
            self.architecture,
            self.weights_sha256,
            self.aggregation,
            self.version,
            self.preprocessing,
        )
        if not all(required):
            raise ValueError("setup components must not be empty")
        for part in (*required, *self.arguments):
            if "|" in part or "\n" in part:
                raise ValueError("setup components must not contain '|' or a newline")

    @property
    def key(self) -> str:
        return "|".join(
            [
                self.reader,
                self.architecture,
                self.weights_sha256,
                self.aggregation,
                self.version,
                " ".join(self.arguments),
                self.preprocessing,
            ]
        )


def lookup(key: str, table: Mapping[str, float] = TAUS) -> float | None:
    """Return the frozen tau for exactly this setup key, or None."""
    return table.get(key)


def usable(score: float) -> bool:
    """True for a finite score within 0 to 1."""
    return math.isfinite(score) and 0.0 <= score <= 1.0


def line_confidence(scores: Sequence[float]) -> float | None:
    """Lowest score of a line, or None when there is none or any is unusable."""
    if not scores or not all(usable(s) for s in scores):
        return None
    return min(scores)


def is_low_confidence(scores: Sequence[float], tau: float | None) -> bool:
    """Fail closed: no tau, no usable scores, or a score below tau means low."""
    confidence = line_confidence(scores)
    return tau is None or confidence is None or confidence < tau
