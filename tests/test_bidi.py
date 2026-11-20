"""Level resolution and reordering checked against samples of the Unicode conformance files."""

import functools
import unicodedata
from collections import Counter

import pytest
from unicode_samples import _levels, character_cases, sample_path

from folio_ocr import bidi


@functools.cache
def _class_cases():
    cases = []
    levels = order = None
    for line in sample_path("bidi_test_sample.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("@Levels:"):
            levels = _levels(line.split(":", 1)[1])
        elif line.startswith("@Reorder:"):
            order = tuple(int(i) for i in line.split(":", 1)[1].split())
        elif line and not line.startswith("#"):
            classes, bitset = line.split(";")
            for bit, base in ((2, 0), (4, 1)):
                if int(bitset) & bit:
                    cases.append((tuple(classes.split()), base, levels, order))
    return tuple(cases)


def test_samples_match_the_running_unicode_version_and_cover_the_weak_types():
    header = sample_path("bidi_character_test_sample.txt").read_text(encoding="utf-8")
    assert f"/{unicodedata.unidata_version}/" in header
    seen = Counter()
    for text, *_ in character_cases():
        seen.update({bidi.bidi_class(c) for c in text})
    for classes, *_ in _class_cases():
        seen.update(set(classes))
    for name in ("L", "R", "AL", "EN", "AN", "ES", "ET", "CS", "NSM", "BN", "S", "WS", "ON"):
        assert seen[name] >= 100, name
    assert len(character_cases()) > 2000 and len(_class_cases()) > 4000


def test_character_cases_resolve_and_reorder_as_unicode_says():
    failures = []
    for text, direction, para, levels, order in character_cases():
        assert para == direction
        got = bidi.levels(text, direction)
        if got != levels or bidi.visual_order(got) != order:
            failures.append(" ".join(f"{ord(c):04X}" for c in text))
    assert failures == []


def test_class_cases_resolve_and_reorder_as_unicode_says():
    failures = []
    for classes, base, levels, order in _class_cases():
        got = bidi.resolve_levels(classes, base)
        if got != levels or bidi.visual_order(got) != order:
            failures.append((" ".join(classes), base))
    assert failures == []


def test_canonically_equivalent_brackets_pair():
    # U+2329 and U+3009 pair through their canonical equivalents (BD16); unpaired, the
    # closing bracket would take the LTR embedding level 0
    text = chr(0x05D0) + chr(0x2329) + chr(0x05D1) + chr(0x3009)
    assert bidi.levels(text, 0) == (1, 1, 1, 1)


def test_explicit_formatting_is_detected():
    assert bidi.has_explicit_formatting("ab" + chr(0x202B) + "c")
    assert bidi.has_explicit_formatting(chr(0x2069))
    assert not bidi.has_explicit_formatting("abc" + chr(0x200F))


@pytest.mark.parametrize("classes", [("L", "RLE", "R"), ("LRI", "L", "PDI")])
def test_explicit_classes_are_refused(classes):
    with pytest.raises(ValueError):
        bidi.resolve_levels(classes, 0)


@pytest.mark.parametrize("base", [-1, 2, True])
def test_base_level_must_be_zero_or_one(base):
    with pytest.raises(ValueError):
        bidi.levels("abc", base)


@pytest.mark.parametrize(
    ("cp", "expected"),
    [
        (0x0378, "L"), (0x05FF, "R"), (0x07B2, "AL"), (0xFE75, "AL"), (0x10D3A, "AL"),
        (0x10EC0, "AL"), (0x10F5A, "AL"), (0x10D40, "R"), (0x1E8C5, "R"), (0x1EC70, "AL"),
        (0x1ED50, "R"), (0x1EF00, "R"), (0x20C1, "ET"), (0x2065, "BN"), (0xFDD0, "BN"),
        (0xFFFF, "BN"), (0x10FFFF, "BN"), (0xE1000, "L"), (0x40000, "L"),
    ],
    ids=lambda v: f"{v:04X}" if isinstance(v, int) else v,
)  # fmt: skip
def test_unassigned_code_points_take_their_default_class(cp, expected):
    assert unicodedata.bidirectional(chr(cp)) == ""
    assert bidi.bidi_class(chr(cp)) == expected


LONG_NAMES = {
    "Left_To_Right": "L",
    "Right_To_Left": "R",
    "Arabic_Letter": "AL",
    "European_Terminator": "ET",
    "Boundary_Neutral": "BN",
}


def _derived_classes():
    """Bidi class of every code point as the derived-class sample states it."""
    classes = ["L"] * 0x110000
    missing = data = 0
    for line in sample_path("derived_bidi_class_unassigned.txt").read_text("utf-8").splitlines():
        if line.startswith("# @missing:"):
            span, long_name = (x.strip() for x in line.split(":", 1)[1].split(";"))
            assert long_name in LONG_NAMES, long_name
            cls, missing = LONG_NAMES[long_name], missing + 1
        elif line and not line.startswith("#"):
            span, rest = (x.strip() for x in line.split(";", 1))
            cls, data = rest.split("#")[0].strip(), data + 1
        else:
            continue
        first, _, last = span.partition("..")
        classes[int(first, 16) : int(last or first, 16) + 1] = [cls] * (
            int(last or first, 16) - int(first, 16) + 1
        )
    assert missing and data
    return classes


def test_every_unassigned_code_point_takes_its_derived_bidi_class():
    classes = _derived_classes()
    wrong = [
        f"{cp:04X}"
        for cp in range(0x110000)
        if not unicodedata.bidirectional(chr(cp)) and bidi.bidi_class(chr(cp)) != classes[cp]
    ]
    assert not wrong, wrong[:20]


def test_empty_text_has_no_levels():
    assert bidi.levels("", 1) == ()
    assert bidi.visual_order(()) == ()
