"""Tests for reading the dev documents with the model and joining each reply to its line."""

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import accuracy_eval
import accuracy_reads as ar
import accuracy_stats
import model_run_lines as ml
import model_run_reads as mr
import model_run_server as srv

from folio_ocr import pipeline
from folio_ocr.document import Calibration, Document, Engine, Line, Page, Source
from folio_ocr.model_reply import parse_reply
from folio_ocr.model_setup import ServerInfo

IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
INFO = ServerInfo("ollama", "0.9", "sha256:abc")
AR_TEXT = "مرحبا"


def body_for(text: str, logprobs=(-0.1,)) -> bytes:
    entries = [{"token": "t", "logprob": lp} for lp in logprobs]
    choice = {
        "message": {"content": text},
        "finish_reason": "stop",
        "logprobs": {"content": entries},
    }
    return json.dumps({"choices": [choice], "usage": {"completion_tokens": 3}}).encode()


def exchange(text: str, logprobs=(-0.1,), status=200) -> srv.Exchange:
    req = json.dumps({"max_tokens": 64}).encode()
    return srv.Exchange(req, status, body_for(text, logprobs), 0.5, None)


def record(text: str, logprobs=(-0.1,), status=200) -> dict:
    return srv.exchange_record(exchange(text, logprobs, status))


def conf_of(text: str, logprobs=(-0.1,)) -> float:
    return parse_reply(200, body_for(text, logprobs)).confidence


def line(i, text, *, reader="model", conf=0.9, flags=(), y=None) -> Line:
    top = 30.0 * i if y is None else y
    return Line(f"p1-l{i + 1}", (0.0, top, 100.0, top + 20), text, "ltr", "latin", reader, conf,
                "logprobs", flags, "auto_accepted", text)


def make_page(lines) -> Page:
    return Page(1, 100, 100, None, 0.0, IDENT, "ocr", "ltr", "read", (), (), tuple(lines))


def make_doc(name, lines, key="KEY") -> Document:
    engine = Engine("d", None, "model", None, None, "untested", None, Calibration(key, 0.5))
    return Document("id", Source(name, "0" * 64, "image", 1), "en", engine,
                    (make_page(lines),), datetime(2026, 1, 1, tzinfo=UTC), False)


def gt_for(texts, name="a.png") -> dict:
    lines = [{"text": t, "clean_box": [0, 30 * i, 100, 30 * i + 20]} for i, t in enumerate(texts)]
    return {"file": name, "homography": [list(r) for r in IDENT], "lines": lines}


def read_of(gt, lines, exchanges=(), name="a.png", unit="u1") -> mr.DocRead:
    return mr.DocRead({"file": name, "unit": unit}, gt, make_page(lines), tuple(exchanges), 0.1,
                      "KEY")


# keys and plan

def test_constants():
    assert mr.MODEL_CELLS == {"model-en": ("model", "en"), "model-ar": ("model", "ar")}
    assert mr.MODEL_READS["en+ar"] == (("model-ar", "en+ar", True),)
    assert mr.CELL_LANGUAGES == {"model-en": ("en",), "model-ar": ("ar", "en+ar")}
    assert mr.SCRIPT_GROUP == {"model-en": "en", "model-ar": "ar"}
    assert mr.MODEL_READERS == frozenset({"model"})


def test_model_keys_differ_and_name_the_model():
    keys = mr.model_keys(INFO, "qwen", "5.4.0")
    assert set(keys) == {"model-en", "model-ar"}
    assert keys["model-en"] != keys["model-ar"]
    assert all(k.startswith("model|qwen|sha256:abc|") for k in keys.values())


def test_model_keys_without_a_setup_is_an_error(monkeypatch):
    monkeypatch.setattr(mr, "model_setup", lambda *a: None)
    with pytest.raises(mr.ModelRunError, match="no calibration key"):
        mr.model_keys(INFO, "qwen", "5.4.0")


def test_model_plan_gives_a_copy_of_the_keys():
    keys = {"model-en": "A", "model-ar": "B"}
    plan = mr.model_plan(keys)
    assert plan.cells is mr.MODEL_CELLS and plan.readers == mr.MODEL_READERS
    assert plan.keys(None) == keys and plan.keys(None) is not keys
    assert ar.cell_reads("ar", plan) == (("model-ar", "ar", True),)


def test_replies_path_sits_next_to_the_cache_file(tmp_path):
    cache = ar.cache_path(tmp_path, "K", "ab" * 32, "en+ar", True)
    side = mr.replies_path(tmp_path, "K", "ab" * 32, "en+ar", True)
    assert side.parent == cache.parent and side.name == cache.stem + ".replies.json"


# reader factory

class FakeCapture:
    """Gives `items` on every take, or the next of `sequence` (one list per take) when given."""

    def __init__(self, items, sequence=None):
        self.items, self.resets, self.sequence = tuple(items), 0, list(sequence or ())

    def reset(self):
        self.resets += 1

    def take(self):
        return tuple(self.sequence.pop(0)) if self.sequence else self.items


def stub_read_bytes(monkeypatch, calls, doc):
    def fake(data, name, **kwargs):
        calls.append((data, name, kwargs))
        return doc

    monkeypatch.setattr(mr, "read_bytes", fake)


def test_reader_writes_replies_before_returning(monkeypatch, tmp_path):
    calls = []
    stub_read_bytes(monkeypatch, calls, make_doc("a.png", [line(0, "hi")], "KAR"))
    cap = FakeCapture([exchange("hi")])
    keys = {"model-en": "KEN", "model-ar": "KAR"}
    settings = SimpleNamespace(weights_dir=Path("w"), line_timeout_s=7.0)
    read = mr.model_reader_factory("MODEL", cap, keys, tmp_path)(settings, Path("tess"))
    doc, seconds, boxes, orientations = read(b"data", "a.png", "en+ar", True)
    side = mr.replies_path(tmp_path, "KAR", hashlib.sha256(b"data").hexdigest(), "en+ar", True)
    saved = json.loads(side.read_bytes())
    assert saved == {"exchanges": [srv.exchange_record(exchange("hi"))]}
    assert cap.resets == 1 and seconds >= 0 and boxes == () and orientations == ()
    _, name, kw = calls[0]
    assert name == "a.png" and kw["model"] == "MODEL" and kw["languages"] == "en+ar"
    assert kw["tesseract_path"] == Path("tess") and kw["line_timeout_s"] == 7.0
    assert doc.pages[0].lines[0].text == "hi"


def reader_over(monkeypatch, tmp_path, items):
    stub_read_bytes(monkeypatch, [], make_doc("a.png", [line(0, "hi")], "KAR"))
    settings = SimpleNamespace(weights_dir=Path("w"), line_timeout_s=7.0)
    keys = {"model-en": "KEN", "model-ar": "KAR"}
    return mr.model_reader_factory(None, FakeCapture(items), keys, tmp_path)(settings, None)


def replies_file(tmp_path):
    return mr.replies_path(tmp_path, "KAR", hashlib.sha256(b"d").hexdigest(), "en+ar", False)


@pytest.mark.parametrize("bad", [
    srv.Exchange(b"{}", None, b"", 0.1, "connection refused"),
    srv.Exchange(b"{}", 503, b"", 0.1, None),
])
def test_reader_stops_when_the_server_stops_answering(monkeypatch, tmp_path, bad):
    read = reader_over(monkeypatch, tmp_path, [exchange("hi"), bad])
    with pytest.raises(ValueError, match="stopped answering"):
        read(b"d", "a.png", "en+ar", False)
    assert not replies_file(tmp_path).exists() and not list(tmp_path.rglob("*.json"))


def test_reader_keeps_a_read_whose_lines_timed_out(monkeypatch, tmp_path):
    slow = srv.Exchange(b"{}", None, b"", 9.0, "took too long", timed_out=True)
    read = reader_over(monkeypatch, tmp_path, [slow])
    read(b"d", "a.png", "en+ar", False)
    assert replies_file(tmp_path).exists()


def test_reader_without_tesseract_passes_none(monkeypatch, tmp_path):
    calls = []
    stub_read_bytes(monkeypatch, calls, make_doc("a.png", [], "KEN"))
    settings = SimpleNamespace(weights_dir=Path("w"), line_timeout_s=7.0)
    keys = {"model-en": "KEN", "model-ar": "KAR"}
    read = mr.model_reader_factory(None, FakeCapture([]), keys, tmp_path)(settings, Path("t"))
    read(b"x", "a.png", "en", False)
    assert calls[0][2]["tesseract_path"] is None
    assert mr.replies_path(tmp_path, "KEN", hashlib.sha256(b"x").hexdigest(), "en", False).exists()


def stub_reads(monkeypatch, docs):
    """read_bytes that returns the next of `docs` on each call."""
    queue = list(docs)
    monkeypatch.setattr(mr, "read_bytes", lambda data, name, **kw: queue.pop(0))


def keyed_reader(tmp_path, cap):
    settings = SimpleNamespace(weights_dir=Path("w"), line_timeout_s=7.0)
    keys = {"model-en": "KEN", "model-ar": "KAR"}
    return mr.model_reader_factory(None, cap, keys, tmp_path)(settings, None)


def test_a_read_of_another_key_is_read_again_and_only_the_second_is_written(
        monkeypatch, tmp_path):
    stub_reads(monkeypatch, [make_doc("a.png", [], "OTHER"), make_doc("a.png", [], "KEN")])
    cap = FakeCapture([], sequence=[[exchange("first")], [exchange("second")]])
    doc, *_ = keyed_reader(tmp_path, cap)(b"d", "a.png", "en", False)
    assert doc.engine.calibration.setup_key == "KEN" and cap.resets == 2
    side = mr.replies_path(tmp_path, "KEN", hashlib.sha256(b"d").hexdigest(), "en", False)
    assert json.loads(side.read_bytes()) == {"exchanges": [srv.exchange_record(exchange("second"))]}


def test_a_read_of_another_key_twice_is_refused_and_nothing_is_written(monkeypatch, tmp_path):
    stub_reads(monkeypatch, [make_doc("a.png", [], "OTHER"), make_doc("a.png", [], "WRONG")])
    cap = FakeCapture([exchange("x")])
    with pytest.raises(ValueError, match="a.png") as err:
        keyed_reader(tmp_path, cap)(b"d", "a.png", "en", False)
    assert str(err.value) == ar.key_mismatch("a.png", "WRONG", "KEN")
    assert "KEN" in str(err.value) and not list(tmp_path.rglob("*"))


def test_a_read_without_calibration_is_refused_after_two_reads(monkeypatch, tmp_path):
    bare = make_doc("a.png", [], "KEN")
    bare = Document(bare.id, bare.source, bare.languages,
                    Engine("d", None, "model", None, None, "untested", None, None),
                    bare.pages, bare.created_at, False)
    queue = [bare, bare]
    monkeypatch.setattr(mr, "read_bytes", lambda data, name, **kw: queue.pop(0))
    cap = FakeCapture([])
    with pytest.raises(ValueError, match="setup key none"):
        keyed_reader(tmp_path, cap)(b"d", "a.png", "en", False)
    assert queue == [] and cap.resets == 2 and not list(tmp_path.rglob("*"))


def test_reader_rejects_an_unknown_mode(monkeypatch, tmp_path):
    stub_read_bytes(monkeypatch, [], make_doc("a.png", []))
    settings = SimpleNamespace(weights_dir=Path("w"), line_timeout_s=7.0)
    read = mr.model_reader_factory(None, FakeCapture([]), {}, tmp_path)(settings, None)
    with pytest.raises(ValueError, match="unknown"):
        read(b"x", "a.png", "fr", False)


# dev manifest

def write_dev(root: Path, n=3, languages=("en", "ar", "en+ar")) -> Path:
    docs = []
    for i in range(n):
        data, gt = f"img{i}".encode(), json.dumps(gt_for(["hi"], f"d{i}.png")).encode()
        (root / f"d{i}.png").write_bytes(data)
        (root / f"d{i}.gt.json").write_bytes(gt)
        docs.append({"file": f"d{i}.png", "sha256": hashlib.sha256(data).hexdigest(),
                     "gt": f"d{i}.gt.json", "gt_sha256": hashlib.sha256(gt).hexdigest(),
                     "language": languages[i % len(languages)], "unit": f"u{i}"})
    (root / "manifest.json").write_text(json.dumps({"documents": docs}), encoding="utf-8")
    return root


def test_load_dev_returns_documents_and_manifest_hash(tmp_path):
    write_dev(tmp_path)
    docs, sha = mr.load_dev(tmp_path)
    raw = (tmp_path / "manifest.json").read_bytes()
    assert len(docs) == 3 and sha == hashlib.sha256(raw).hexdigest()


def test_load_dev_refuses_a_changed_file(tmp_path):
    write_dev(tmp_path)
    (tmp_path / "d1.png").write_bytes(b"other")
    with pytest.raises(ValueError, match="d1.png does not match the dev manifest"):
        mr.load_dev(tmp_path)


def test_load_dev_refuses_a_changed_ground_truth(tmp_path):
    write_dev(tmp_path)
    (tmp_path / "d0.gt.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="d0.gt.json does not match"):
        mr.load_dev(tmp_path)


def test_load_dev_names_a_missing_field(tmp_path):
    write_dev(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["documents"][0]["unit"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing unit"):
        mr.load_dev(tmp_path)


def test_load_dev_refuses_a_bad_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("[1]")
    with pytest.raises(ValueError, match="not a valid manifest"):
        mr.load_dev(tmp_path)


def test_dev_docs_for_keeps_manifest_order():
    docs = [{"language": "en", "file": "1"}, {"language": "ar", "file": "2"},
            {"language": "en+ar", "file": "3"}]
    assert [d["file"] for d in mr.dev_docs_for(docs, "model-ar")] == ["2", "3"]
    assert [d["file"] for d in mr.dev_docs_for(docs, "model-en")] == ["1"]


# read_dev_doc

KEYS = {"model-en": "KEN", "model-ar": "KAR"}


def reader_for(monkeypatch, tmp_path, doc, exchanges):
    stub_read_bytes(monkeypatch, [], doc)
    settings = SimpleNamespace(weights_dir=Path("w"), line_timeout_s=7.0)
    return mr.model_reader_factory(None, FakeCapture(exchanges), KEYS, tmp_path / "work")(
        settings, None
    )


def test_read_dev_doc_returns_page_replies_and_gt(monkeypatch, tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    write_dev(dev)
    docs, _ = mr.load_dev(dev)
    work = tmp_path / "work"
    reader = reader_for(monkeypatch, tmp_path, make_doc("d0.png", [line(0, "hi")], "KEN"),
                        [exchange("hi")])
    got = mr.read_dev_doc(work, KEYS, reader, dev, docs[0])
    assert got.doc is docs[0] and got.gt["lines"][0]["text"] == "hi"
    assert got.page.lines[0].text == "hi" and got.exchanges[0]["content"] == "hi"
    assert got.setup_key == "KEN" and got.seconds >= 0


def test_read_dev_doc_missing_replies(monkeypatch, tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    write_dev(dev)
    docs, _ = mr.load_dev(dev)
    work = tmp_path / "work"
    reader = reader_for(monkeypatch, tmp_path, make_doc("d0.png", [line(0, "hi")], "KEN"), [])
    mr.read_dev_doc(work, KEYS, reader, dev, docs[0])
    data = (dev / "d0.png").read_bytes()
    mr.replies_path(work, "KEN", hashlib.sha256(data).hexdigest(), "en", True).unlink()
    with pytest.raises(mr.ModelRunError, match="replies of d0.png are missing"):
        mr.read_dev_doc(work, KEYS, reader, dev, docs[0])


def test_read_dev_doc_damaged_replies(monkeypatch, tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    write_dev(dev)
    docs, _ = mr.load_dev(dev)
    work = tmp_path / "work"
    reader = reader_for(monkeypatch, tmp_path, make_doc("d0.png", [line(0, "hi")], "KEN"), [])
    mr.read_dev_doc(work, KEYS, reader, dev, docs[0])
    data = (dev / "d0.png").read_bytes()
    mr.replies_path(work, "KEN", hashlib.sha256(data).hexdigest(), "en", True).write_text("{")
    with pytest.raises(mr.ModelRunError, match="replies of d0.png are damaged"):
        mr.read_dev_doc(work, KEYS, reader, dev, docs[0])


def test_read_dev_doc_wants_one_page(monkeypatch, tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    write_dev(dev)
    docs, _ = mr.load_dev(dev)
    two = make_doc("d0.png", [], "KEN")
    two = Document(two.id, two.source, two.languages, two.engine, two.pages * 2,
                   two.created_at, False)
    reader = reader_for(monkeypatch, tmp_path, two, [])
    with pytest.raises(ValueError, match="d0.png was read as 2 pages; one image is one page"):
        mr.read_dev_doc(tmp_path / "work", KEYS, reader, dev, docs[0])


# joining

def test_join_same_text_different_confidence_each_gets_its_own():
    a, b = record("x", (-0.1,)), record("x", (-0.5,))
    page = make_page([line(0, "x", conf=conf_of("x", (-0.5,))),
                      line(1, "x", conf=conf_of("x", (-0.1,)))])
    joined = mr.join_replies(page, [a, b])
    assert joined == {0: b, 1: a}


def test_join_uses_each_record_once_in_request_order():
    a, b = record("x"), record("x")
    b = {**b, "seconds": 9.0}
    page = make_page([line(0, "x", conf=conf_of("x")), line(1, "x", conf=conf_of("x"))])
    assert mr.join_replies(page, [a, b]) == {0: a, 1: b}
    assert mr.join_replies(make_page([line(0, "x", conf=conf_of("x"))]), [a, b]) == {0: a}


def test_join_page_order_differs_from_request_order():
    r1, r2, r3 = record("one", (-0.1,)), record("two", (-0.2,)), record("three", (-0.3,))
    page = make_page([line(0, "three", conf=conf_of("three", (-0.3,))),
                      line(1, "one", conf=conf_of("one", (-0.1,))),
                      line(2, "two", conf=conf_of("two", (-0.2,)))])
    assert mr.join_replies(page, [r1, r2, r3]) == {0: r3, 1: r1, 2: r2}


def test_join_ignores_failed_status_and_other_readers_and_unmatched():
    bad = record("x", status=500)
    page = make_page([line(0, "x", conf=conf_of("x")), line(1, "y", reader="onnxtr"),
                      line(2, "z", conf=0.5)])
    assert mr.join_replies(page, [bad, record("y")]) == {}


def test_join_cleans_the_reply_text_like_the_pipeline():
    r = record("a\u200bb")
    cleaned = mr.clean_text("a\u200bb")[0]
    page = make_page([line(0, cleaned, conf=conf_of("a\u200bb"))])
    assert mr.join_replies(page, [r]) == {0: r}


# line records

def test_line_records_fill_measures_for_matched_model_lines():
    gt = gt_for(["abc", "def", AR_TEXT])
    rec = record("abd")
    page_lines = [line(0, "abd", conf=conf_of("abd")), line(1, "def", reader="onnxtr"),
                  line(9, "zzz", y=900.0)]
    read = read_of(gt, page_lines, [rec])
    out = mr.line_records(read, "model-en", frozenset())
    assert [r["gt"] for r in out] == ["abc", "def", AR_TEXT]
    first, second, third = out
    assert first["matched"] and first["reader"] == "model" and first["text"] == "abd"
    assert first["cer"] == pytest.approx(1 / 3) and first["order_error"] is None
    assert first["document"] == "a.png" and first["unit"] == "u1"
    assert first["cell"] == "model-en" and first["script_group"] == "en"
    assert first["classes"] == [] and first["flags"] == []
    assert first["finish_reason"] == "stop" and first["max_tokens"] == 64
    assert first["completion_tokens"] == 3 and first["entries"][0]["logprob"] == -0.1
    assert second["matched"] and second["reader"] == "onnxtr" and second["cer"] is None
    assert second["entries"] is None and second["flags"] == []
    assert not third["matched"] and third["reader"] is None and third["text"] is None
    assert third["cer"] is None and third["classes"] == list(ml.order_classes(AR_TEXT)) == []


def test_line_records_carry_flags_and_classes_of_a_model_line():
    gt = gt_for(["abc ١٢ مر"])
    page_lines = [line(0, "abc", flags=("mixed_order",), conf=0.7)]
    out = mr.line_records(read_of(gt, page_lines), "model-ar", frozenset())[0]
    assert out["flags"] == ["mixed_order"] and out["classes"] == ["arabic+latin"]
    assert out["script_group"] == "ar" and out["entries"] is None


def test_vocabulary_of_collects_norm_words():
    gts = [gt_for(["Hello World"]), gt_for(["world again"])]
    assert mr.vocabulary_of(gts) == frozenset({"Hello", "World", "world", "again"})


# bad lines and the order pass

def test_bad_line_count_counts_model_lines_over_the_error_bound():
    gt = gt_for(["abc", "def", "ghi"])
    lines = [line(0, "abc"), line(1, "xyz"), line(2, "ghi", reader="onnxtr")]
    assert mr.bad_line_count(read_of(gt, lines)) == 1


def order_read(i, n_good=3, bad=False) -> mr.DocRead:
    text = "ab ١٢ مر"
    gt = gt_for([text] * n_good, f"d{i}.png")
    got = ["zz" if bad else text] * n_good
    page_lines = [line(k, t, conf=0.9) for k, t in enumerate(got)]
    return mr.DocRead({"file": f"d{i}.png", "unit": f"u{i}"}, gt, make_page(page_lines), (), 0.1, "K")


def fake_docs(n):
    return [{"file": f"d{i}.png", "unit": f"u{i}", "i": i} for i in range(n)]


def patch_minimums(monkeypatch, lines_min=2, bad_min=1, docs_min=1, cap=300):
    monkeypatch.setattr(ml, "MIN_ORDER_LINES", lines_min)
    monkeypatch.setattr(accuracy_stats, "DEV_MIN_BAD", bad_min)
    monkeypatch.setattr(accuracy_stats, "DEV_MIN_DOCS", docs_min)
    monkeypatch.setattr(accuracy_stats, "DEV_CAP_DOCS", cap)


def test_order_pass_without_the_order_classes_runs_to_the_end(monkeypatch):
    patch_minimums(monkeypatch)
    seen = []

    def read_one(doc):
        seen.append(doc["i"])
        return order_read(doc["i"])

    reads, summary = mr.order_pass(fake_docs(5), read_one, 2)
    assert len(reads) == 5 and sorted(seen) == [0, 1, 2, 3, 4]
    assert summary["stopped_by"] == "dev set end" and summary["documents"] == 5
    assert summary["bad_lines"] == 0 and summary["documents_with_bad_lines"] == 0


def arabic_latin(i, bad) -> mr.DocRead:
    text = "ab مر"
    gt = gt_for([text] * 2, f"d{i}.png")
    got = ["zz" if bad else text] * 2
    return mr.DocRead({"file": f"d{i}.png", "unit": f"u{i}"}, gt,
                      make_page([line(k, t) for k, t in enumerate(got)]), (), 0.1, "K")


def test_order_pass_stops_after_the_batch_that_meets_both_minimums(monkeypatch):
    patch_minimums(monkeypatch)
    monkeypatch.setattr(ml, "ORDER_CLASSES", ("arabic+latin",))
    reads, summary = mr.order_pass(fake_docs(10), lambda d: arabic_latin(d["i"], d["i"] == 0), 3)
    assert len(reads) == 3  # whole batch kept
    assert summary["stopped_by"] == "minimums" and summary["documents"] == 3
    assert summary["bad_lines"] == 2 and summary["documents_with_bad_lines"] == 1
    assert summary["classes"]["arabic+latin"]["lines"] == 6


def test_order_pass_keeps_reading_while_an_order_class_is_short(monkeypatch):
    patch_minimums(monkeypatch, lines_min=1000)
    monkeypatch.setattr(ml, "ORDER_CLASSES", ("arabic+latin",))
    reads, summary = mr.order_pass(fake_docs(10), lambda d: arabic_latin(d["i"], True), 3)
    assert len(reads) == 10 and summary["stopped_by"] == "dev set end"


def test_order_pass_keeps_reading_while_no_document_has_a_bad_line(monkeypatch):
    patch_minimums(monkeypatch)
    monkeypatch.setattr(ml, "ORDER_CLASSES", ("arabic+latin",))
    reads, summary = mr.order_pass(fake_docs(10), lambda d: arabic_latin(d["i"], False), 3)
    assert len(reads) == 10 and summary["stopped_by"] == "dev set end"


def test_order_pass_stops_at_the_cap(monkeypatch):
    patch_minimums(monkeypatch, cap=4)
    reads, summary = mr.order_pass(fake_docs(10), lambda d: order_read(d["i"]), 3)
    assert len(reads) == 4 and summary["stopped_by"] == "cap"


# fixture and Tesseract comparison

def test_fixture_cer_pools_over_ground_truth_lines():
    gt = gt_for(["abcd", "efgh", "ij", "   "])
    lines = [line(0, "abcd"), line(1, "efxh")]
    out = mr.fixture_cer(gt, make_page(lines))
    assert out == {"gt_lines": 3, "matched": 2, "reference_chars": 10, "errors": 3,
                   "cer": pytest.approx(0.3)}
    assert mr.fixture_cer(gt_for(["  "]), make_page([]))["cer"] is None


def test_fixture_cer_uses_final_text():
    gt = gt_for(["abcd"])
    ln = Line("p1-l1", (0.0, 0.0, 100.0, 20.0), "xxxx", "ltr", "latin", "model", 0.9,
              "logprobs", (), "edited", "abcd")
    assert mr.fixture_cer(gt, make_page([ln]))["cer"] == 0.0


def test_fixture_cer_on_a_born_digital_pdf():
    fixtures = HERE / "fixtures"
    data = (fixtures / "en-invoice.pdf").read_bytes()
    gt = json.loads((fixtures / "en-invoice.gt.json").read_bytes())
    doc = pipeline.read_bytes(data, "en-invoice.pdf", languages="en", weights_dir=None,
                              tesseract_path=None, line_timeout_s=30.0, model=None)
    out = mr.fixture_cer(gt, doc.pages[0])
    assert out["cer"] is not None and out["cer"] < 0.005


def test_tesseract_compare_pools_arabic_lines_only():
    gt = gt_for(["abc", AR_TEXT, AR_TEXT + "x"])
    model_page = [line(0, "abc"), line(1, AR_TEXT), line(2, "zz")]
    tess_page = make_page([line(1, AR_TEXT, reader="tesseract")])
    read = read_of(gt, model_page, name="a.png")
    other = read_of(gt, model_page, name="b.png")
    out = mr.tesseract_compare([read, other], {"a.png": tess_page})
    assert out["documents"] == 1 and out["arabic_gt_lines"] == 2
    assert out["model"]["reference_chars"] == out["tesseract"]["reference_chars"]
    assert out["model"]["errors"] == len(norm(AR_TEXT + "x")) - 0 and out["model"]["cer"] > 0
    assert out["tesseract"]["errors"] == len(norm(AR_TEXT + "x"))


def norm(text):
    from folio_ocr.normalize import norm_v2

    return norm_v2(text)


# score checks

def test_auc_counts_ties_as_half():
    assert mr.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert mr.auc([0.1], [0.9]) == 0.0
    assert mr.auc([0.5], [0.5]) == 0.5
    assert mr.auc([0.9, 0.5], [0.5, 0.1]) == pytest.approx(3.5 / 4)
    assert mr.auc([], [0.5]) is None and mr.auc([0.5], []) is None


def agg_rec(cer, logprobs, **over) -> dict:
    base = {"reader": "model", "matched": True, "cer": cer,
            "entries": [{"token": "t", "logprob": lp} for lp in logprobs]}
    return {**base, **over}


def test_aggregation_check_reports_when_another_score_beats_exp_min():
    recs = [agg_rec(0.0, (-0.1, -0.1, -0.1)), agg_rec(0.0, (-0.1,)),
            agg_rec(0.5, (-0.3,)), agg_rec(0.5, (-0.01, -3.0)),
            agg_rec(0.0, (-0.2, -0.2)),
            agg_rec(0.0, ()), agg_rec(None, (-0.1,)), agg_rec(0.0, (-0.1,), reader="none"),
            agg_rec(0.0, (-0.1,), matched=False), agg_rec(0.0, "bad")]
    recs[-1] = {**recs[-1], "entries": ["x"]}
    out = mr.aggregation_check(recs)
    assert (out["lines"], out["good"], out["bad"]) == (5, 3, 2)
    assert out["exp_min"] == pytest.approx(mr.auc(
        [math.exp(-0.1), math.exp(-0.1), math.exp(-0.2)], [math.exp(-0.3), math.exp(-3.0)]))
    assert out["best_other_margin"] == pytest.approx(
        max(out["exp_mean"], out["exp_sum"]) - out["exp_min"])
    assert out["stop"] is (out["best_other_margin"] > 0.02)


def test_aggregation_check_splits_good_and_bad_at_the_accuracy_tool_bound(monkeypatch):
    monkeypatch.setattr(accuracy_eval, "BAD_CER", 0.5)
    out = mr.aggregation_check([agg_rec(0.3, (-0.1,)), agg_rec(0.9, (-0.2,))])
    assert (out["good"], out["bad"]) == (1, 1)


def test_aggregation_check_with_no_bad_lines_has_no_margin():
    out = mr.aggregation_check([agg_rec(0.0, (-0.1,))])
    assert out["exp_min"] is None and out["best_other_margin"] is None and out["stop"] is False


def test_pins_count_lines_off_their_threshold():
    recs = [
        {"cell": "model-en", "reader": "model", "confidence": 0.5, "document": "d", "gt": "g",
         "text": "t"},
        {"cell": "model-en", "reader": "model", "confidence": 0.8, "document": "d", "gt": "g",
         "text": "t"},
        {"cell": "model-en", "reader": "model", "confidence": None},
        {"cell": "model-en", "reader": "onnxtr", "confidence": 0.1},
        {"cell": "model-ar", "reader": "model", "confidence": 0.3, "document": "d", "gt": "g",
         "text": "t"},
    ]
    out = mr.pins(recs, {"model-en": 0.5, "model-ar": None}, 0.001)
    assert out["model-en"]["tau"] == 0.5 and out["model-en"]["eligible"] == 2
    assert out["model-en"]["pinnable"] == 1
    assert out["model-en"]["examples"] == [
        {"document": "d", "gt": "g", "text": "t", "confidence": 0.8}]
    assert out["model-ar"] == {"tau": None, "eligible": 1, "pinnable": 0, "examples": []}


def test_pins_keep_the_first_fifty_examples():
    recs = [{"cell": "model-en", "reader": "model", "confidence": 0.9, "document": "d",
             "gt": "g", "text": str(i)} for i in range(60)]
    out = mr.pins(recs, {"model-en": 0.1}, 0.001)["model-en"]
    assert out["pinnable"] == 60 and len(out["examples"]) == 50
    assert out["examples"][-1]["text"] == "49"
