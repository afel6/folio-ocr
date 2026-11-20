"""Each Unicode version has its own folder of samples, and tests read the running version's."""

import sys
import unicodedata
from pathlib import Path

import pytest
from unicode_samples import SAMPLES, sample_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sample_bidi_tests as sb

SOURCES = {
    "bidi_character_test_sample.txt": "BidiCharacterTest.txt",
    "bidi_test_sample.txt": "BidiTest.txt",
    "derived_bidi_class_unassigned.txt": "DerivedBidiClass.txt",
}


@pytest.mark.parametrize("name", SOURCES)
@pytest.mark.parametrize("version", ["15.0.0", "15.1.0"])
def test_each_version_selects_its_own_sample(version, name):
    path = sample_path(name, version)
    assert path.parent == SAMPLES / version
    comments = {
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")
    }
    assert f"# Sample of {sb.source_url(version, SOURCES[name])}" in comments
    assert f"# Source sha256: {sb.SHA256[version][SOURCES[name]]}" in comments


@pytest.mark.parametrize("version", ["15.0.0", "15.1.0"])
def test_the_running_version_is_the_default(monkeypatch, version):
    monkeypatch.setattr(unicodedata, "unidata_version", version)
    for name in SOURCES:
        assert sample_path(name) == SAMPLES / version / name


def test_a_version_without_samples_fails_with_one_line():
    with pytest.raises(BaseException) as info:
        sample_path("bidi_test_sample.txt", "14.0.0")
    assert type(info.value) is pytest.fail.Exception
    assert "14.0.0" in str(info.value)
    assert "\n" not in str(info.value)
    assert info.value.pytrace is False


def test_samples_stay_small():
    files = [p for p in SAMPLES.glob("*/*") if p.is_file()]
    assert len(files) >= 6
    assert all(p.stat().st_size <= 200_000 for p in files)
