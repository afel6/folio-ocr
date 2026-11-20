"""The development variants share no text, fonts, layouts or seeds with the fixtures or held-out sets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import dev_fonts as df
import dev_text as dt
import fixture_text as ft
import make_dev_variants as mdv
import make_fixtures as mf

HELD_OUT = sorted((HERE.parent / "eval" / "heldout").glob("*.json"))


def ranges_overlap(a_first: int, a_last: int, b_first: int, b_last: int) -> bool:
    """True when the inclusive ranges a_first..a_last and b_first..b_last share a value."""
    return a_first <= b_last and b_first <= a_last


def test_ranges_overlap_edge_cases():
    assert ranges_overlap(0, 10, 10, 20)  # touching ends share a value
    assert ranges_overlap(10, 20, 0, 10)
    assert ranges_overlap(0, 100, 40, 50)  # nested
    assert ranges_overlap(40, 50, 0, 100)
    assert ranges_overlap(5, 5, 5, 5)  # single point
    assert ranges_overlap(0, 10, 5, 5)
    assert not ranges_overlap(0, 9, 10, 20)  # adjacent, no shared value
    assert not ranges_overlap(11, 20, 0, 10)
    assert not ranges_overlap(5, 5, 6, 6)
    assert ranges_overlap(0, 10, 3, 14)  # partial


def _fixture_strings() -> set[str]:
    pools = [
        ft.EN.values(),
        ft.AR.values(),
        ft.EN_NAMES,
        ft.AR_NAMES,
        ft.EN_COMPANIES,
        ft.AR_COMPANIES,
        ft.EN_ITEMS,
        ft.AR_ITEMS,
        ft.MONTHS,
    ]
    return {s for pool in pools for s in pool}


def test_dev_strings_differ_from_fixture_strings():
    fixture = _fixture_strings()
    shared = {s for strings in dt.POOLS.values() for s in strings} & fixture
    assert not shared


def test_dev_font_hashes_differ_from_fixture_fonts():
    fixture = {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (HERE / "fixtures" / "fonts").glob("*.ttf")
    }
    assert fixture
    assert not set(df.font_sha256s().values()) & fixture


def test_dev_templates_differ_from_fixture_layouts():
    assert not set(dt.TEMPLATES) & set(ft.LAYOUTS)


def test_fixture_seed_is_outside_dev_range():
    assert mf.DEFAULT_SEED not in mdv.DEV_SEED_RANGE


def test_held_out_manifest_exists():
    assert HELD_OUT, "no held-out manifest in eval/heldout/"


@pytest.mark.parametrize("path", HELD_OUT, ids=lambda p: p.name)
def test_held_out_seeds_avoid_dev_and_builtin_seeds(path):
    seeds = json.loads(path.read_text(encoding="utf-8"))["seeds"]
    first, last = seeds["first"], seeds["last"]
    excluded = json.loads(path.read_text(encoding="utf-8")).get("excluded_seeds", [])
    assert not ranges_overlap(first, last, mdv.DEV_SEED_RANGE.start, mdv.DEV_SEED_RANGE.stop - 1)
    assert not (first <= mf.DEFAULT_SEED <= last) or mf.DEFAULT_SEED in excluded


def test_held_out_manifests_do_not_overlap_each_other():
    spans = [(p.name, json.loads(p.read_text(encoding="utf-8"))["seeds"]) for p in HELD_OUT]
    for i, (name_a, a) in enumerate(spans):
        for name_b, b in spans[i + 1 :]:
            assert not ranges_overlap(a["first"], a["last"], b["first"], b["last"]), (name_a, name_b)
