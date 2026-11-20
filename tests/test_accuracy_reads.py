"""Tests for scripts/accuracy_reads.py: cells, the two OnnxTR reads, orientation capture, cache."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import accuracy_reads as ar
from test_run_accuracy import make_doc

from folio_ocr import ocr as ocr_module


def _doc(key="K"):
    return make_doc("a", 1, bad=False, conf=0.9, status="auto_accepted", key=key, tau=0.5)


def test_four_cells_with_readers_and_languages():
    assert ar.CELLS == {
        "onnxtr-en-osd": ("onnxtr", "en"),
        "onnxtr-en-vote": ("onnxtr", "en"),
        "tesseract-en": ("tesseract", "en"),
        "tesseract-ar": ("tesseract", "ar"),
    }


def test_expected_keys_per_cell(monkeypatch):
    seen = []
    monkeypatch.setattr(ar, "decided_version", lambda path: "5.3.4")
    monkeypatch.setattr(
        ar, "onnxtr_setup", lambda osd=None: seen.append(osd) or SimpleNamespace(key=f"onnx-{osd}")
    )
    monkeypatch.setattr(ar, "tesseract_setup", lambda v: SimpleNamespace(key=f"tess-{v}"))
    keys = ar.expected_keys(Path("t.exe"))
    assert keys == {
        "onnxtr-en-osd": "onnx-5.3",
        "onnxtr-en-vote": "onnx-None",
        "tesseract-en": "tess-5.3.4",
        "tesseract-ar": "tess-5.3.4",
    }


def test_pipeline_reader_gives_tesseract_only_when_asked(monkeypatch):
    paths = []

    def fake_read_bytes(data, name, **kwargs):
        paths.append((kwargs["languages"], kwargs["tesseract_path"]))
        return _doc()

    monkeypatch.setattr(ar, "read_bytes", fake_read_bytes)
    settings = SimpleNamespace(weights_dir=None, line_timeout_s=1)
    read = ar.pipeline_reader(settings, Path("t.exe"))
    read(b"x", "a.png", "en", True)
    read(b"x", "a.png", "en", False)
    read(b"x", "a.png", "en+ar", True)
    assert paths == [("en", Path("t.exe")), ("en", None), ("en+ar", Path("t.exe"))]


def test_the_two_onnxtr_reads_do_not_share_a_cache_file(tmp_path):
    osd = ar.cache_path(tmp_path, "KEY-osd", "ab" * 32, "en", True)
    vote = ar.cache_path(tmp_path, "KEY-vote", "ab" * 32, "en", False)
    same_key = ar.cache_path(tmp_path, "KEY-osd", "ab" * 32, "en", False)
    assert len({osd, vote, same_key}) == 3


def test_cached_reads_of_one_document_stay_apart(tmp_path):
    calls = []

    def reader(data, name, mode, with_tesseract):
        calls.append(with_tesseract)
        source = "osd" if with_tesseract else "vote"
        return _doc("KEY-osd" if with_tesseract else "KEY-vote"), 1.0, (), ((source, with_tesseract, "pair", None, False),)

    a = ar.cached_read(tmp_path, "KEY-osd", "ab" * 32, b"x", "a.png", "en", True, reader)
    b = ar.cached_read(tmp_path, "KEY-vote", "ab" * 32, b"x", "a.png", "en", False, reader)
    assert calls == [True, False]
    assert ar.orientations_from_cache(a) == (("osd", True, "pair", None, False),)
    assert ar.orientations_from_cache(b) == (("vote", False, "pair", None, False),)
    assert len(list((tmp_path / "reads").rglob("*.json"))) == 2


def test_cache_round_trip_keeps_orientations():
    record = (("vote", True, "four", 0.25, True), ("osd", False, "none", None, False))
    obj = json.loads(json.dumps(ar.document_to_cache(_doc(), 1.0, (), record)))
    assert ar.orientations_from_cache(obj) == record
    assert ar.orientations_from_cache({}) == ()


def test_a_cache_file_without_the_new_orientation_keys_is_read_again(tmp_path):
    old = ar.document_to_cache(_doc(), 1.0, (), ())
    path = ar.cache_path(tmp_path, "K", "ab" * 32, "en", True)
    path.parent.mkdir(parents=True)
    calls = []
    for stale in ({k: v for k, v in old.items() if k != "orientations"},
                  {**old, "orientations": [["vote", True]]}):
        path.write_text(json.dumps(stale), encoding="utf-8")
        calls.clear()

        def reader(data, name, mode, with_tesseract):
            calls.append(1)
            return _doc(), 2.0, (), (("vote", True, "pair", 0.5, False),)

        got = ar.cached_read(tmp_path, "K", "ab" * 32, b"x", "a.png", "en", True, reader)
        assert calls == [1] and got["seconds"] == 2.0
        assert ar.orientations_from_cache(json.loads(path.read_text("utf-8"))) == (
            ("vote", True, "pair", 0.5, False),
        )


def _fake_orientation(tag):
    return lambda models, image, osd: SimpleNamespace(
        source=tag(image), uncertain=image == "u", branch="pair", margin=0.5, error=None
    )


def test_orientation_capture_keeps_threads_apart(monkeypatch):
    monkeypatch.setattr(ocr_module, "page_orientation", _fake_orientation(lambda i: "osd"))
    ar.install_orientation_capture()
    ar.install_orientation_capture()  # wrapping twice must not duplicate entries
    barrier, got = threading.Barrier(2), {}

    def work(tag, images):
        ar.reset_orientations()
        barrier.wait()
        for image in images:
            ocr_module.page_orientation(None, image, None)
        barrier.wait()
        got[tag] = ar.take_orientations()

    a = threading.Thread(target=work, args=("a", ["u", "v"]))
    b = threading.Thread(target=work, args=("b", ["w"]))
    a.start()
    b.start()
    a.join()
    b.join()
    assert got == {
        "a": (("osd", True, "pair", 0.5, False), ("osd", False, "pair", 0.5, False)),
        "b": (("osd", False, "pair", 0.5, False),),
    }


def test_pipeline_reader_returns_orientations_per_read(monkeypatch):
    monkeypatch.setattr(ocr_module, "page_orientation", _fake_orientation(lambda i: "vote"))
    monkeypatch.setattr(
        ar, "read_bytes", lambda data, name, **kw: ocr_module.page_orientation(None, "u", None) and _doc()
    )
    read = ar.pipeline_reader(SimpleNamespace(weights_dir=None, line_timeout_s=1), None)
    expected = (("vote", True, "pair", 0.5, False),)
    assert read(b"x", "a.png", "en", False)[3] == expected
    assert read(b"x", "a.png", "en", False)[3] == expected  # cleared before each read


def test_orientation_capture_passes_the_result_through(monkeypatch):
    result = SimpleNamespace(source="vote", uncertain=False, branch="pair", margin=0.5, error=None)
    monkeypatch.setattr(ocr_module, "page_orientation", lambda models, image, osd: result)
    ar.install_orientation_capture()
    assert ocr_module.page_orientation(None, "i", None) is result


def test_orientation_capture_records_branch_margin_and_failure(monkeypatch):
    shipped = SimpleNamespace(
        source="vote", uncertain=True, branch="four", margin=0.375, error="timed out"
    )
    monkeypatch.setattr(ocr_module, "page_orientation", lambda models, image, osd: shipped)
    ar.install_orientation_capture()
    ar.reset_orientations()
    ocr_module.page_orientation(None, "i", None)
    assert ar.take_orientations() == (("vote", True, "four", 0.375, True),)


# ---------------------------------------------------------------- the setup key of a cached read

SHA = "ab" * 32
ORIENT = (("vote", True, "pair", 0.5, False),)


def _scripted(*keys):
    """A reader that returns one read per key in turn, recording each call."""
    calls = []

    def reader(data, name, mode, with_tesseract):
        calls.append(name)
        return _doc(keys[min(len(calls), len(keys)) - 1]), 1.0, (), ORIENT

    reader.calls = calls
    return reader


def _stale_file(tmp_path, key):
    path = ar.cache_path(tmp_path, "K", SHA, "en", True)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(ar.document_to_cache(_doc(key), 9.0, (), ORIENT)), encoding="utf-8")
    return path


def _read(tmp_path, reader):
    return ar.cached_read(tmp_path, "K", SHA, b"x", "a.png", "en", True, reader)


def test_a_cached_read_of_another_key_is_read_once_more_and_replaced(tmp_path):
    path = _stale_file(tmp_path, "OTHER")
    reader = _scripted("K")
    got = _read(tmp_path, reader)
    assert len(reader.calls) == 1 and got["setup_key"] == "K"
    assert json.loads(path.read_text("utf-8"))["setup_key"] == "K"


def test_a_cached_read_of_another_key_that_stays_another_is_refused_and_kept(tmp_path):
    path = _stale_file(tmp_path, "OTHER")
    before = path.read_bytes()
    reader = _scripted("WRONG")
    with pytest.raises(ValueError, match="WRONG.*K") as err:
        _read(tmp_path, reader)
    assert str(err.value) == ar.key_mismatch("a.png", "WRONG", "K")
    assert len(reader.calls) == 1 and path.read_bytes() == before


def test_a_fresh_read_of_another_key_is_read_once_more(tmp_path):
    reader = _scripted("OTHER", "K")
    got = _read(tmp_path, reader)
    assert len(reader.calls) == 2 and got["setup_key"] == "K"
    assert len(list((tmp_path / "reads").rglob("*.json"))) == 1


def test_a_fresh_read_of_another_key_twice_is_refused_and_not_cached(tmp_path):
    reader = _scripted("OTHER", "OTHER")
    with pytest.raises(ValueError, match="OTHER.*K"):
        _read(tmp_path, reader)
    assert len(reader.calls) == 2 and not list(tmp_path.rglob("*.json"))


def test_a_read_without_calibration_has_no_key(tmp_path):
    assert ar.key_mismatch("a.png", None, "K") == (
        "a.png was read twice with setup key none, not its cell's key K; nothing of it was kept"
    )
