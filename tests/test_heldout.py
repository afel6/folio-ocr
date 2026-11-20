"""Tests for the held-out batch module."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from fixture_checks import assert_same_ground_truth, assert_same_png

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import heldout as ho

FIXTURES = HERE / "fixtures"
COMMITTED = {e["file"]: e for e in json.loads((FIXTURES / "manifest.json").read_text("utf-8"))["fixtures"]}


@pytest.fixture
def manifest():
    return ho.build_manifest(1000, 1004, "unit")


@pytest.mark.parametrize(("token", "layout"), [("en", "invoice"), ("ar", "letter")])
def test_builtin_seed_images_match_committed_fixtures(token, layout):
    images = ho.held_out_images(ho.BUILTIN_SEED, token, layout)
    assert [i.condition for i in images] == ["clean", "scan", "photo"]
    for img in images:
        if img.name.endswith(".png"):
            assert_same_png(img.data, (FIXTURES / img.name).read_bytes())
        else:
            assert hashlib.sha256(img.data).hexdigest() == COMMITTED[img.name]["sha256"]
        gt = json.loads((FIXTURES / COMMITTED[img.name]["gt"]).read_text("utf-8"))
        assert_same_ground_truth(img.gt, gt)
        assert img.unit == f"{ho.BUILTIN_SEED}-{token}-{layout}"


def test_text_sha256_ignores_crlf_but_file_sha256_does_not(tmp_path):
    lf, crlf = tmp_path / "a.py", tmp_path / "b.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert ho.text_sha256(lf) == ho.text_sha256(crlf)
    assert ho.file_sha256(lf) != ho.file_sha256(crlf)


def test_verify_generator_passes_on_fresh_manifest(manifest):
    ho.verify_generator(manifest)


def test_verify_generator_names_altered_file(manifest):
    bad = {**manifest, "generator_files": {**manifest["generator_files"], "scripts/fixture_text.py": "0" * 64}}
    with pytest.raises(ho.HeldoutError, match="fixture_text.py"):
        ho.verify_generator(bad)


def test_verify_generator_names_file_missing_from_manifest(manifest):
    files = {k: v for k, v in manifest["generator_files"].items() if k != "scripts/fixture_pages.py"}
    with pytest.raises(ho.HeldoutError, match="fixture_pages.py"):
        ho.verify_generator({**manifest, "generator_files": files})


def test_verify_generator_names_extra_font(manifest):
    fonts = {**manifest["fonts"], "tests/fixtures/fonts/Extra.ttf": "0" * 64}
    with pytest.raises(ho.HeldoutError, match="Extra.ttf"):
        ho.verify_generator({**manifest, "fonts": fonts})


def test_verify_generator_names_altered_font(manifest):
    name = ho.FONT_FILES[0]
    with pytest.raises(ho.HeldoutError, match=Path(name).name):
        ho.verify_generator({**manifest, "fonts": {**manifest["fonts"], name: "0" * 64}})


@pytest.mark.parametrize(
    ("first", "last", "name"),
    [(ho.BUILTIN_SEED - 1, ho.BUILTIN_SEED + 1, "n"), (5, 4, "n"), (-1, 3, "n"), (1, 2, "")],
)
def test_build_manifest_refuses(first, last, name):
    with pytest.raises(ho.HeldoutError):
        ho.build_manifest(first, last, name)


def test_manifest_fields(manifest):
    assert manifest["schema"] == ho.SCHEMA
    assert manifest["excluded_seeds"] == [ho.BUILTIN_SEED]
    assert manifest["seeds"] == {"first": 1000, "last": 1004}
    assert manifest["conditions"] == list(ho.CONDITIONS)
    assert manifest["resampling"]["rng_seed"] == 20261119
    assert "scripts/make_fixtures.py" in manifest["generator_files"]


def test_seeds_inclusive(manifest):
    assert list(ho.seeds(manifest)) == [1000, 1001, 1002, 1003, 1004]


def test_seeds_skip_excluded(manifest):
    got = ho.seeds({**manifest, "excluded_seeds": [1002]})
    assert list(got) == [1000, 1001, 1003, 1004]


def test_write_manifest_roundtrip_and_no_overwrite(tmp_path, manifest):
    path = tmp_path / "unit.json"
    sha = ho.write_manifest(manifest, path)
    assert ho.load_manifest(path) == (manifest, sha)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ho.HeldoutError, match="unit.json already exists"):
        ho.write_manifest(manifest, path)


def test_load_manifest_rejects_bad_json_and_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(ho.HeldoutError, match="bad.json"):
        ho.load_manifest(bad)
    bad.write_text('{"schema": "other/1"}', encoding="utf-8")
    with pytest.raises(ho.HeldoutError, match="bad.json"):
        ho.load_manifest(bad)


@pytest.mark.parametrize(("bad", "docs", "expected"), [(99, 40, False), (150, 29, False), (100, 30, True)])
def test_stop_reached_needs_both_minimums(bad, docs, expected):
    assert ho.stop_reached(ho.Tally(bad, docs)) is expected


def test_add_counts_only_documents_with_bad_lines():
    t0 = ho.Tally()
    t1 = ho.add(ho.add(t0, 3), 0)
    assert t0 == ho.Tally()
    assert t1 == ho.Tally(3, 1)


def test_graded_attempts_matches_hash_and_raises_on_bad_json(tmp_path):
    (tmp_path / "a.json").write_text('{"heldout_manifest_sha256": "aa"}', encoding="utf-8")
    (tmp_path / "b.json").write_text('{"heldout_manifest_sha256": "bb"}', encoding="utf-8")
    assert ho.graded_attempts(tmp_path, "aa") == (tmp_path / "a.json",)
    assert ho.graded_attempts(tmp_path / "missing", "aa") == ()
    (tmp_path / "c.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ho.HeldoutError, match="c.json"):
        ho.graded_attempts(tmp_path, "aa")


def test_start_attempt_refuses_second_grade_of_same_batch(tmp_path):
    out = tmp_path / "new" / "results"
    record = {"heldout_manifest_sha256": "a" * 64, "state": "started"}
    path = ho.start_attempt(out, record)
    assert json.loads(path.read_text("utf-8")) == record
    assert path.name.startswith(f"grade-{'a' * 12}-")
    with pytest.raises(ho.HeldoutError, match="already been graded"):
        ho.start_attempt(out, record)
    other = ho.start_attempt(out, {"heldout_manifest_sha256": "b" * 64})
    assert other.exists()


def test_finish_attempt_replaces_content(tmp_path):
    path = ho.start_attempt(tmp_path, {"heldout_manifest_sha256": "c" * 64, "state": "started"})
    done = {"heldout_manifest_sha256": "c" * 64, "state": "done"}
    ho.finish_attempt(path, done)
    assert json.loads(path.read_text("utf-8")) == done
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def _write(tmp_path, name, first, last):
    return ho.write_manifest(ho.build_manifest(first, last, name), tmp_path / f"{name}.json")


@pytest.mark.parametrize(("first", "last"), [(1004, 1010), (990, 1000), (1001, 1002), (900, 1100)])
def test_write_manifest_refuses_a_range_that_overlaps_another_manifest(tmp_path, first, last):
    _write(tmp_path, "early", 1000, 1004)
    with pytest.raises(ho.HeldoutError, match="overlaps an earlier held-out batch"):
        _write(tmp_path, "late", first, last)
    assert [p.name for p in tmp_path.iterdir()] == ["early.json"]


def test_write_manifest_refuses_the_same_range_under_a_new_name(tmp_path):
    _write(tmp_path, "early", 1000, 1004)
    with pytest.raises(ho.HeldoutError, match="a re-tune needs a new seed range"):
        _write(tmp_path, "renamed", 1000, 1004)


def test_write_manifest_accepts_a_disjoint_range(tmp_path):
    _write(tmp_path, "early", 1000, 1004)
    _write(tmp_path, "late", 1005, 1009)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["early.json", "late.json"]


def test_start_attempt_refuses_a_range_that_overlaps_another_manifest(tmp_path):
    _write(tmp_path, "early", 1000, 1004)
    later = tmp_path / "late.json"
    ho.write_manifest(ho.build_manifest(1005, 1009, "late"), later)
    record = {"heldout_manifest_sha256": "a" * 64, "seeds": {"first": 1004, "last": 1008}}
    with pytest.raises(ho.HeldoutError, match="overlaps an earlier held-out batch"):
        ho.start_attempt(tmp_path / "res", record, later)
    ok = {"heldout_manifest_sha256": "a" * 64, "seeds": {"first": 1005, "last": 1009}}
    assert ho.start_attempt(tmp_path / "res", ok, later).exists()


def test_start_attempt_refuses_a_range_that_overlaps_a_recorded_attempt(tmp_path):
    res = tmp_path / "res"
    ho.start_attempt(res, {"heldout_manifest_sha256": "a" * 64, "seeds": {"first": 10, "last": 20}})
    with pytest.raises(ho.HeldoutError, match="a re-tune needs a new seed range"):
        ho.start_attempt(res, {"heldout_manifest_sha256": "b" * 64, "seeds": {"first": 20, "last": 30}})
    other = {"heldout_manifest_sha256": "b" * 64, "seeds": {"first": 21, "last": 30}}
    assert ho.start_attempt(res, other).exists()


def test_a_results_file_that_cannot_be_read_is_reported_by_its_own_reason(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")

    def refuse(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    with pytest.raises(ho.HeldoutError, match=r"cannot read attempt file a.json: Permission denied"):
        ho.graded_attempts(tmp_path, "aa")


@pytest.mark.parametrize("text", ['{"schema": "folio-heldout/1"}', '{"schema": "folio-heldout/1", "seeds": {"first": 1}}'])
def test_load_manifest_names_missing_seeds(tmp_path, text):
    path = tmp_path / "m.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ho.HeldoutError, match="m.json is missing seeds first and last"):
        ho.load_manifest(path)
