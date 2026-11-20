"""The model dev manifest (dev-4) extends dev-3 and stays clear of the held-out seeds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import make_dev_variants as mdv

EVAL = HERE.parent / "eval"
DEV3 = json.loads((EVAL / "dev" / "dev-3.json").read_bytes())
DEV4 = json.loads((EVAL / "dev" / "dev-4.json").read_bytes())


def test_dev4_seeds_are_one_to_three_hundred_inside_the_dev_range():
    assert DEV4["seeds"] == {"first": 1, "last": 300}
    assert DEV4["seeds"]["first"] in mdv.DEV_SEED_RANGE
    assert DEV4["seeds"]["last"] in mdv.DEV_SEED_RANGE


def test_dev4_pins_the_same_generator_and_fonts_as_dev3():
    assert DEV4["generator_files"] == DEV3["generator_files"]
    assert DEV4["fonts"] == DEV3["fonts"]


def test_dev4_starts_with_the_dev3_documents():
    assert DEV4["documents"][:450] == DEV3["documents"]


def test_dev4_seeds_overlap_no_held_out_batch():
    for path in sorted((EVAL / "heldout").glob("*.json")):
        seeds = json.loads(path.read_bytes())["seeds"]
        assert DEV4["seeds"]["last"] < seeds["first"] or seeds["last"] < DEV4["seeds"]["first"]
