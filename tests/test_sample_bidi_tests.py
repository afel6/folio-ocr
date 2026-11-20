from __future__ import annotations

import hashlib
import os
import stat
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sample_bidi_tests as sb


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path):
    paths = []
    for name, text in (("c.txt", "# c\n"), ("b.txt", "# b\n"), ("d.txt", "# d\n")):
        path = tmp_path / name
        path.write_text(text)
        paths.append(path)
    return paths


def _match(monkeypatch, version, c, b, d):
    table = {
        "BidiCharacterTest.txt": _sha(c),
        "BidiTest.txt": _sha(b),
        "DerivedBidiClass.txt": _sha(d),
    }
    monkeypatch.setattr(sb, "SHA256", {**sb.SHA256, version: table})


def _run(capsys, *args, version=None):
    extra = {"unicode_version": version} if version else {}
    code = sb.main([str(a) for a in args], **extra)
    return code, capsys.readouterr().err


def test_missing_input(tmp_path, capsys):
    gone = [tmp_path / n for n in ("no.txt", "no2.txt", "no3.txt")]
    code, err = _run(capsys, *gone, "--out", tmp_path, version="15.1.0")
    assert code == 1 and "not found" in err and err.count("\n") == 1


def test_input_is_folder(tmp_path, capsys):
    code, err = _run(capsys, tmp_path, tmp_path, tmp_path, "--out", tmp_path, version="15.1.0")
    assert code == 1 and err.startswith("error:") and err.count("\n") == 1


def test_wrong_sha(tmp_path, capsys):
    c, b, d = _inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    code, err = _run(capsys, c, b, d, "--out", out, version="15.1.0")
    assert code == 1 and err.startswith("error: cannot sample: ")


def test_unknown_unicode_version_is_refused(tmp_path, capsys):
    c, b, d = _inputs(tmp_path)
    out = tmp_path / "out"
    code, err = _run(capsys, c, b, d, "--out", out, version="14.0.0")
    assert code == 1
    assert err == "error: no source hashes for Unicode 14.0.0 (known: 15.0.0, 15.1.0)\n"
    assert not out.exists()


@pytest.mark.parametrize("v", ["15.0.0", "15.1.0"])
def test_each_version_writes_into_its_own_folder(tmp_path, capsys, monkeypatch, v):
    c, b, d = _inputs(tmp_path)
    _match(monkeypatch, v, c, b, d)
    root = tmp_path / "unicode"
    root.mkdir()
    monkeypatch.setattr(sb, "SAMPLES", root)
    code, err = _run(capsys, c, b, d, version=v)
    assert code == 0 and err == ""
    names = {
        "bidi_character_test_sample.txt",
        "bidi_test_sample.txt",
        "derived_bidi_class_unassigned.txt",
    }
    assert {p.name for p in (root / v).iterdir()} == names
    assert [p.name for p in root.iterdir()] == [v]
    character = (root / v / "bidi_character_test_sample.txt").read_text(encoding="utf-8")
    derived = (root / v / "derived_bidi_class_unassigned.txt").read_text(encoding="utf-8")
    assert f"# Sample of {sb.source_url(v, 'BidiCharacterTest.txt')}" in character
    assert sb.source_url(v, "DerivedBidiClass.txt") in derived
    assert "ucd/extracted/" in derived


def test_the_running_unicode_version_is_the_default(tmp_path, capsys, monkeypatch):
    running = unicodedata.unidata_version
    c, b, d = _inputs(tmp_path)
    _match(monkeypatch, running, c, b, d)
    root = tmp_path / "unicode"
    root.mkdir()
    monkeypatch.setattr(sb, "SAMPLES", root)
    code, err = _run(capsys, c, b, d)
    assert code == 0 and err == ""
    assert [p.name for p in root.iterdir()] == [running]


def test_files_of_another_version_are_refused(tmp_path, capsys, monkeypatch):
    c, b, d = _inputs(tmp_path)
    _match(monkeypatch, "15.1.0", c, b, d)
    out = tmp_path / "out"
    out.mkdir()
    code, err = _run(capsys, c, b, d, "--out", out, version="15.0.0")
    assert code == 1 and err.startswith("error: cannot sample: ")
    assert "Unicode 15.0.0" in err and err.count("\n") == 1


def test_out_is_file(tmp_path, capsys, monkeypatch):
    c, b, d = _inputs(tmp_path)
    version = "15.1.0"
    _match(monkeypatch, version, c, b, d)
    f = tmp_path / "f.txt"
    f.write_text("x")
    code, err = _run(capsys, c, b, d, "--out", f, version=version)
    assert code == 1 and err == f"error: not a folder: {f}\n"


def test_sample_name_taken_by_a_folder(tmp_path, capsys, monkeypatch):
    c, b, d = _inputs(tmp_path)
    version = "15.1.0"
    _match(monkeypatch, version, c, b, d)
    out = tmp_path / "out"
    (out / "bidi_character_test_sample.txt").mkdir(parents=True)
    code, err = _run(capsys, c, b, d, "--out", out, version=version)
    assert code == 1
    assert err == f"error: is a folder: {out / 'bidi_character_test_sample.txt'}\n"


def test_read_only_sample_is_reported(tmp_path, capsys, monkeypatch):
    c, b, d = _inputs(tmp_path)
    version = "15.1.0"
    _match(monkeypatch, version, c, b, d)
    out = tmp_path / "out"
    out.mkdir()
    p = out / "bidi_character_test_sample.txt"
    p.write_text("old")
    os.chmod(p, stat.S_IREAD)
    try:
        code, err = _run(capsys, c, b, d, "--out", out, version=version)
    finally:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    assert code == 1 and err == f"error: permission denied: {p}\n"


@pytest.mark.parametrize("v", ["15.0.0", "15.1.0"])
def test_character_sampler_keeps_plain_cases_and_header(v):
    lines = [
        "# notice",
        "0061;0;0;0;0",  # plain
        "202A 0061;0;0;0;0",  # explicit embedding
        "0061;2;0;0;0",  # direction 2
        "2029;0;0;0;0",  # class B
        "05D0;1;1;1;1",  # plain
    ]
    out = sb.sample_character_test(lines, v)
    assert "# notice" in out
    assert any(sb.source_url(v, "BidiCharacterTest.txt") in x for x in out)
    assert any(sb.SHA256[v]["BidiCharacterTest.txt"] in x for x in out)
    cases = [x for x in out if not x.startswith("#")]
    assert cases == ["0061;0;0;0;0", "05D0;1;1;1;1"]


def test_bidi_test_sampler_keeps_plain_cases_and_header(monkeypatch):
    monkeypatch.setattr(sb, "BIDI_TEST_STRIDE", 1)
    lines = ["@Levels: 0", "@Reorder: 0", "L; 7", "L RLE; 7", "B; 7", "R; 1", "AL; 6"]
    out = sb.sample_bidi_test(lines, "15.1.0")
    assert any(sb.SHA256["15.1.0"]["BidiTest.txt"] in x for x in out)
    cases = [x for x in out if ";" in x and not x.startswith(("#", "@"))]
    assert cases == ["L;6", "AL;6"]


def test_unassigned_sampler_keeps_missing_and_unassigned_lines():
    lines = [
        "# DerivedBidiClass-15.1.0.txt",
        "# Date: x",
        "# @missing: 0000..10FFFF; Left_To_Right",
        "",
        "# @missing: 0590..05FF; Right_To_Left",
        "0590          ; R # Cn       <reserved-0590>",
        "05BE          ; R # Pd       HEBREW PUNCTUATION MAQAF",
        "FDD0..FDEF    ; BN # Cn  [32] <noncharacter-FDD0>..<noncharacter-FDEF>",
        "# Cn in a comment line",
    ]
    out = sb.sample_unassigned_classes(lines, "15.1.0")
    assert f"# Sample of {sb.source_url('15.1.0', 'DerivedBidiClass.txt')}" in out
    assert f"# Source sha256: {sb.SHA256['15.1.0']['DerivedBidiClass.txt']}" in out
    kept = out[next(i for i, x in enumerate(out) if x.startswith("# Kept:")) + 1 :]
    assert kept == [lines[2], lines[4], lines[5], lines[7]]
