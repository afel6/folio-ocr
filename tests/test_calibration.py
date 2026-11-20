"""Tests for the frozen confidence thresholds and the low-confidence decision."""

import dataclasses
import math

import pytest

from folio_ocr.calibration import (
    LOW_CONFIDENCE,
    TAUS,
    Setup,
    checked_table,
    is_low_confidence,
    line_confidence,
    lookup,
    preprocessing_hash,
    table_sha256,
    usable,
    version_major_minor,
)

BASE = Setup("onnxtr", "db_resnet50", "abc123", "min", "0.9", (), "pp1")
TABLE = {BASE.key: 0.6}
COMPONENTS = ["reader", "architecture", "weights_sha256", "aggregation", "version", "preprocessing"]


def test_flag_code():
    assert LOW_CONFIDENCE == "low_confidence"


@pytest.mark.parametrize(
    ("scores", "low"),
    [
        ((0.9, 0.7), False),
        ((0.9, 0.59), True),
        ((0.6,), False),
        ((), True),
        ((0.9, -0.01), True),
        ((0.9, math.nan), True),
        ((0.9, math.inf), True),
        ((0.9, 1.0000001), True),
    ],
)
def test_low_confidence_with_tau(scores, low):
    assert is_low_confidence(scores, lookup(BASE.key, TABLE)) is low


def test_unusable_score_low_even_with_zero_tau():
    assert is_low_confidence((0.9, -1.0), 0.0) is True
    assert is_low_confidence((0.9, math.nan), 0.0) is True


def test_no_tau_is_low():
    assert is_low_confidence((1.0,), None) is True


def test_line_confidence_and_usable():
    assert line_confidence((0.9, 0.7)) == 0.7
    assert line_confidence(()) is None
    assert line_confidence((0.5, math.nan)) is None
    assert usable(0.0) and usable(1.0)
    assert not usable(-0.0001) and not usable(math.inf)


def test_lookup_exact():
    assert lookup(BASE.key, TABLE) == 0.6
    assert lookup("other", TABLE) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"reader": "tesseract"},
        {"architecture": "db_mobilenet"},
        {"weights_sha256": "def456"},
        {"aggregation": "mean"},
        {"version": "0.10"},
        {"arguments": ("--psm", "6")},
        {"preprocessing": "pp2"},
    ],
)
def test_each_component_changes_lookup(changes):
    other = dataclasses.replace(BASE, **changes)
    assert other.key != BASE.key
    assert lookup(other.key, TABLE) is None


def test_key_layout_literal():
    s = Setup("tesseract", "lstm", "abc", "min", "5.4", ("--psm", "6"), "pp")
    assert s.key == "tesseract|lstm|abc|min|5.4|--psm 6|pp"
    assert BASE.key == "onnxtr|db_resnet50|abc123|min|0.9||pp1"


@pytest.mark.parametrize("bad", ["a|b", "a\nb"])
@pytest.mark.parametrize("field", COMPONENTS)
def test_forbidden_characters_in_component(field, bad):
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, **{field: bad})


@pytest.mark.parametrize("bad", ["a|b", "a\nb"])
def test_forbidden_characters_in_argument(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, arguments=("--psm", bad))


@pytest.mark.parametrize("field", COMPONENTS)
def test_empty_component(field):
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, **{field: ""})


def test_preprocessing_hash():
    a = preprocessing_hash({"x": 1, "y": "z"})
    assert a == preprocessing_hash({"y": "z", "x": 1})
    assert a != preprocessing_hash({"x": 2, "y": "z"})
    # json writes 1 and 1.0 differently, so the hashes differ.
    assert preprocessing_hash({"x": 1}) != preprocessing_hash({"x": 1.0})
    with pytest.raises(ValueError):
        preprocessing_hash({"x": math.nan})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.9.0", "0.9"), ("v0.9.0", "0.9"), ("5.4.0.20240606", "5.4"), ("5.3", "5.3")],
)
def test_version_major_minor(raw, expected):
    assert version_major_minor(raw) == expected


@pytest.mark.parametrize("raw", ["", "5", "abc", "v", "5.x", "٣.٤"])
def test_version_rejected(raw):
    with pytest.raises(ValueError):
        version_major_minor(raw)


@pytest.mark.parametrize(
    "table", [{"a": math.nan}, {"a": -0.1}, {"a": 1.5}, {"": 0.5}, {"a": math.inf}]
)
def test_checked_table_rejects(table):
    with pytest.raises(ValueError):
        checked_table(table)


def test_checked_table_read_only_copy():
    src = {"a": 0.5}
    out = checked_table(src)
    src["a"] = 0.9
    assert out["a"] == 0.5
    with pytest.raises(TypeError):
        out["b"] = 0.1  # type: ignore[index]


# sha256 of the frozen tau table (canonical JSON); it changes only with a new calibration, which
# also records it in the results JSON. The empty table hashes to EMPTY_TABLE_SHA256.
FROZEN_TABLE_SHA256 = "e02f6fe92a8ee259efa213be0bf781b67a724322501153be8b6a2df3a6b6003a"
EMPTY_TABLE_SHA256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def test_taus_read_only_and_valid():
    assert checked_table(TAUS) == TAUS
    with pytest.raises(TypeError):
        TAUS["a"] = 0.5  # type: ignore[index]


def test_table_sha256_pinned():
    assert table_sha256(TAUS) == FROZEN_TABLE_SHA256
    assert table_sha256() == FROZEN_TABLE_SHA256
    assert table_sha256({}) == EMPTY_TABLE_SHA256
    assert table_sha256({"a": 0.5}) != EMPTY_TABLE_SHA256
