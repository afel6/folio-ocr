"""Samples of Unicode data files, one folder per Unicode version under tests/data/unicode;
tests read the folder of the running Python's Unicode version."""

import functools
import unicodedata
from pathlib import Path

import pytest

SAMPLES = Path(__file__).parent / "data" / "unicode"


def sample_path(name: str, version: str | None = None) -> Path:
    """The sample `name` for Unicode `version` (the running Python's when None); fails the
    calling test when there is none."""
    version = unicodedata.unidata_version if version is None else version
    path = SAMPLES / version / name
    if not path.is_file():
        pytest.fail(
            f"no {name} sample for Unicode {version}: add its source sha256 values to "
            "SHA256 in scripts/sample_bidi_tests.py, then run that script with this Python",
            pytrace=False,
        )
    return path


def _levels(field: str) -> tuple[int | None, ...]:
    return tuple(None if v == "x" else int(v) for v in field.split())


@functools.cache
def character_cases() -> tuple[tuple[str, int, int, tuple[int | None, ...], tuple[int, ...]], ...]:
    """Cases of the character sample: text, direction, paragraph level, levels, visual order."""
    cases = []
    for line in (
        sample_path("bidi_character_test_sample.txt").read_text(encoding="utf-8").splitlines()
    ):
        if line.startswith("#") or not line:
            continue
        codes, direction, para, levels, order = line.split(";")
        text = "".join(chr(int(cp, 16)) for cp in codes.split())
        cases.append(
            (text, int(direction), int(para), _levels(levels), tuple(map(int, order.split())))
        )
    return tuple(cases)
