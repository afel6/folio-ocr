"""Remembering file verdicts inside a `remembered` block."""

import hashlib
import os

import pytest

from folio_ocr import assets

GOOD = b"good bytes"
BAD = b"evil bytes"  # same size as GOOD


@pytest.fixture
def asset():
    return assets.Asset(
        "t/a.bin",
        "https://assets.example/a.bin",
        hashlib.sha256(GOOD).hexdigest(),
        len(GOOD),
        "MIT",
    )


@pytest.fixture
def hashed(monkeypatch):
    calls = []
    real = hashlib.sha256

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(assets.hashlib, "sha256", counting)
    return calls


def put(tmp_path, data):
    path = tmp_path / "t" / "a.bin"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def reasons(asset, tmp_path):
    return [s.reason for s in assets.verify([asset], tmp_path)]


def test_second_verify_in_a_block_does_not_hash(asset, tmp_path, hashed):
    put(tmp_path, GOOD)
    with assets.remembered({}):
        assert reasons(asset, tmp_path) == ["ok"]
        assert reasons(asset, tmp_path) == ["ok"]
    assert len(hashed) == 1


def test_changed_file_is_hashed_again(asset, tmp_path, hashed):
    path = put(tmp_path, GOOD)
    with assets.remembered({}):
        assert reasons(asset, tmp_path) == ["ok"]
        t = path.stat().st_mtime_ns
        path.write_bytes(BAD)
        os.utime(path, ns=(t, t + 1_000_000_000))
        assert reasons(asset, tmp_path) == ["sha256 mismatch"]
    assert len(hashed) == 2


def test_no_block_hashes_every_time(asset, tmp_path, hashed):
    put(tmp_path, GOOD)
    reasons(asset, tmp_path)
    reasons(asset, tmp_path)
    assert len(hashed) == 2


def test_wrong_size_is_never_hashed_or_stored(asset, tmp_path, hashed):
    put(tmp_path, b"short")
    verdicts = {}
    with assets.remembered(verdicts):
        assert reasons(asset, tmp_path) == ["wrong size"]
    assert hashed == [] and verdicts == {}


def test_missing_is_not_stored(asset, tmp_path):
    verdicts = {}
    with assets.remembered(verdicts):
        assert reasons(asset, tmp_path) == ["missing"]
    assert verdicts == {}


def test_nothing_leaks_after_the_block(asset, tmp_path, hashed):
    put(tmp_path, GOOD)
    with assets.remembered({}):
        reasons(asset, tmp_path)
    reasons(asset, tmp_path)
    assert len(hashed) == 2
