"""Tests for the step files and the contract, order, drift-sample and drift commands."""

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(_P(__file__).resolve().parent))

import base64
import json
import math
from types import SimpleNamespace

import model_run
import model_run_common as common
import model_run_reads as mr
import model_run_server as srv
import pytest
from fake_model_server import Answer, FakeModelServer
from model_run_support import FACTS, KEYS, builder, make_session, run, step_file, tree

from folio_ocr.model_reply import ORDER_RULE

# ---------------------------------------------------------------- step files


def test_load_step_reads_a_finished_step(tmp_path):
    path = step_file(tmp_path / "o.json", "order", keys=KEYS)
    assert common.load_step(path, "order")["keys"] == KEYS


def test_load_step_missing_file(tmp_path):
    with pytest.raises(srv.ModelRunError, match="the order results are missing or unfinished: o.json"):
        common.load_step(tmp_path / "o.json", "order")


@pytest.mark.parametrize("text", ["not json", "[1]", '{"step": "drift", "status": "finished"}',
                                  '{"step": "order", "status": "started"}'])
def test_load_step_refuses_a_bad_file(tmp_path, text):
    path = tmp_path / "o.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(srv.ModelRunError, match="missing or unfinished"):
        common.load_step(path, "order")


def test_check_keys():
    common.check_keys({"step": "order", "keys": KEYS}, dict(KEYS))
    with pytest.raises(srv.ModelRunError, match="since the order step"):
        common.check_keys({"step": "order", "keys": {"model-en": "x"}}, KEYS)


def test_write_step_adds_common_keys_and_never_the_grade_key(tmp_path):
    out = tmp_path / "sub" / "s.json"
    common.write_step(out, "order", "2026-01-01T00:00:00Z", {"keys": KEYS, "model": FACTS})
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["step"] == "order" and record["status"] == "finished"
    assert record["started_at"] == "2026-01-01T00:00:00Z" and record["finished_at"]
    assert "heldout_manifest_sha256" not in record


def test_write_step_refuses_a_grade_key_and_writes_nothing(tmp_path):
    out = tmp_path / "sub" / "s.json"
    with pytest.raises(ValueError, match="heldout_manifest_sha256"):
        common.write_step(out, "order", "t", {"heldout_manifest_sha256": "ab" * 32})
    assert not out.exists() and not out.parent.exists()


# ---------------------------------------------------------------- contract


def contract_result(passed):
    checks = [
        {"name": "server facts", "required": True, "passed": True, "detail": "ollama 0.9.1"},
        {"name": "self-test", "required": True, "passed": passed, "detail": "d"},
    ]
    return {"checks": checks, "passed": passed, "facts": dict(FACTS)}


def contract_argv(tmp_path, *extra):
    return ["contract", "--work", tmp_path / "w", "--out", tmp_path / "c.json", *extra]


@pytest.fixture
def model_env(monkeypatch):
    monkeypatch.setenv("FOLIO_MODEL_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("FOLIO_MODEL_NAME", "m:8b")


def test_contract_passes_and_writes(tmp_path, capsys, monkeypatch, model_env):
    seen = {}

    def stub(config, **kw):
        seen.update(kw)
        return contract_result(True)

    monkeypatch.setattr(srv, "contract_checks", stub)
    rc, out, err = run(capsys, contract_argv(tmp_path, "--expect-version", "0.9.1"))
    assert rc == 0 and err == ""
    assert seen == {"expect_version": "0.9.1", "expect_digest": None}
    assert "ok   server facts: ollama 0.9.1" in out
    record = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert record["step"] == "contract" and record["passed"] is True


def test_contract_failure_stops_but_writes(tmp_path, capsys, monkeypatch, model_env):
    monkeypatch.setattr(srv, "contract_checks", lambda config, **kw: contract_result(False))
    rc, out, _ = run(capsys, contract_argv(tmp_path))
    assert rc == 1
    assert "FAIL self-test: d" in out
    assert "STOP: a required reply-contract check failed" in out
    assert (tmp_path / "c.json").exists()


def test_contract_needs_the_model_url(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FOLIO_MODEL_URL", raising=False)
    rc, _, err = run(capsys, contract_argv(tmp_path))
    assert rc == 1
    assert err.strip() == "error: FOLIO_MODEL_URL is not set; set it to the model server's /v1 URL"


def test_contract_with_the_server_down(tmp_path, capsys, monkeypatch):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("FOLIO_MODEL_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("FOLIO_MODEL_NAME", "m:8b")
    rc, out, err = run(capsys, contract_argv(tmp_path))
    assert rc == 1 and "Traceback" not in out + err
    assert "STOP: a required reply-contract check failed" in out
    assert str(port) not in out


# ---------------------------------------------------------------- order


def order_argv(tmp_path, **kw):
    return ["order", "--work", tmp_path / "w", "--dev", tmp_path / "dev", "--out",
            tmp_path / "o.json", *kw.get("extra", [])]


def stub_order(monkeypatch, verdict="frozen", classes=None):
    classes = classes or {"mixed": {"status": "ok", "lines": 30, "order_errors": 0,
                                    "unflagged_order_errors": 0}}
    summary = {"classes": classes, "verdict": verdict, "min_lines": 30, "documents": 2,
               "bad_lines": 0, "documents_with_bad_lines": 0, "stopped_by": "minimums"}
    docs = [{"file": "a.png", "language": "ar"}, {"file": "b.png", "language": "en"}]
    seen = {}
    monkeypatch.setattr(mr, "load_dev", lambda dev: (docs, "devsha"))

    def order_pass(ar_docs, read_one, jobs):
        seen["docs"], seen["jobs"] = ar_docs, jobs
        return ("read-a",), summary

    records = [
        {"document": "a.png", "gt": "g", "text": "t", "classes": ["c"], "flags": [], "order_error": True},
        {"document": "a.png", "gt": "g2", "text": "t2", "classes": ["c"], "flags": ["mixed_order"],
         "order_error": True},
        {"document": "a.png", "gt": "g3", "text": "t3", "classes": [], "flags": [], "order_error": False},
    ]
    monkeypatch.setattr(mr, "order_pass", order_pass)
    monkeypatch.setattr(mr, "line_records", lambda read, cell, vocab: records)
    return seen


def test_order_frozen(tmp_path, capsys, monkeypatch):
    seen = stub_order(monkeypatch)
    rc, out, err = run(capsys, order_argv(tmp_path))
    assert rc == 0 and err == ""
    assert [d["file"] for d in seen["docs"]] == ["a.png"]
    assert "mixed: lines 30, order errors 0, unflagged 0 -> ok" in out
    assert f"order rule frozen: {ORDER_RULE}" in out
    rec = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert rec["step"] == "order" and rec["verdict"] == "frozen"
    assert rec["order_rule"] == ORDER_RULE and rec["dev_manifest_sha256"] == "devsha"
    assert rec["keys"] == KEYS and rec["model"] == FACTS
    assert [r["gt"] for r in rec["unflagged_examples"]] == ["g"]
    assert set(rec["unflagged_examples"][0]) == {"document", "gt", "text", "classes", "flags"}


def test_order_unflagged_examples_are_capped_at_twenty(tmp_path, capsys, monkeypatch):
    stub_order(monkeypatch)
    many = [{"document": "a", "gt": str(i), "text": "t", "classes": [], "flags": [],
             "order_error": True} for i in range(30)]
    monkeypatch.setattr(mr, "line_records", lambda read, cell, vocab: many)
    run(capsys, order_argv(tmp_path))
    rec = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert len(rec["unflagged_examples"]) == 20


def test_order_stop_names_classes_and_still_writes(tmp_path, capsys, monkeypatch):
    classes = {"a": {"status": "short", "lines": 3, "order_errors": 0, "unflagged_order_errors": 0},
               "b": {"status": "ok", "lines": 40, "order_errors": 0, "unflagged_order_errors": 0}}
    stub_order(monkeypatch, "stop", classes)
    rc, out, _ = run(capsys, order_argv(tmp_path))
    assert rc == 1
    assert "STOP: the order rule misses or lacks dev evidence for: a" in out
    assert (tmp_path / "o.json").exists()


# ---------------------------------------------------------------- drift sample


def good_exchange(i, status=200, entries=None, confident=True):
    lp = -0.1 if confident else None
    body = {"choices": [{"message": {"content": f"t{i}"}, "finish_reason": "stop",
                         "logprobs": {"content": [{"token": "a", "logprob": lp}] if lp is not None
                                      else []}}]}
    raw = json.dumps(body).encode()
    return {"request_b64": base64.b64encode(f"req{i}".encode()).decode(), "status": status,
            "body_b64": base64.b64encode(raw).decode(),
            "entries": [{"token": "a", "logprob": -0.1}] if entries is None else entries}


def sample_setup(tmp_path, monkeypatch, per_doc):
    """order.json (frozen) plus a dev set whose documents hold `per_doc` exchanges each."""
    docs = [{"file": f"{lang}{i}.png", "language": lang} for lang in ("en", "ar") for i in range(3)]
    monkeypatch.setattr(mr, "load_dev", lambda dev: (docs, "devsha"))
    monkeypatch.setattr(
        mr, "read_dev_doc",
        lambda work, keys, reader, dev, doc: SimpleNamespace(exchanges=per_doc(doc), doc=doc))
    return step_file(tmp_path / "o.json", "order", verdict="frozen", order_rule=ORDER_RULE,
                     keys=KEYS, dev_manifest_sha256="devsha")


def sample_argv(tmp_path, order, *extra):
    return ["drift-sample", "--work", tmp_path / "w", "--dev", tmp_path / "dev", "--order", order,
            "--out", tmp_path / "s.json", *extra]


def test_drift_sample_keeps_usable_exchanges_up_to_the_line_count(tmp_path, capsys, monkeypatch):
    def per_doc(doc):
        return [good_exchange(1), good_exchange(2, status=500), good_exchange(3, entries=[]),
                good_exchange(4, confident=False), good_exchange(5)]

    order = sample_setup(tmp_path, monkeypatch, per_doc)
    rc, _, err = run(capsys, sample_argv(tmp_path, order, "--lines", "3"))
    assert rc == 0, err
    rec = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert rec["step"] == "drift-sample" and rec["keys"] == KEYS
    by_cell = {}
    for r in rec["requests"]:
        by_cell.setdefault(r["cell"], []).append(r)
    assert {c: len(v) for c, v in by_cell.items()} == {"model-en": 3, "model-ar": 3}
    assert set(by_cell["model-en"][0]) == {"cell", "document", "request_b64", "status", "body_b64"}
    assert all(r["status"] == 200 for r in rec["requests"])


@pytest.mark.parametrize("patch", [{"verdict": "stop"}, {"order_rule": "other"},
                                   {"keys": {"model-en": "x"}}, {"dev_manifest_sha256": "zzz"}])
def test_drift_sample_refuses_an_unfrozen_or_stale_order(tmp_path, capsys, monkeypatch, patch):
    order = sample_setup(tmp_path, monkeypatch, lambda doc: [good_exchange(1)])
    record = json.loads(order.read_text(encoding="utf-8"))
    order.write_text(json.dumps({**record, **patch}), encoding="utf-8")
    rc, _, err = run(capsys, sample_argv(tmp_path, order))
    assert rc == 1 and err.startswith("error: ") and err.count("\n") == 1
    assert not (tmp_path / "s.json").exists()


def test_drift_sample_without_an_order_file(tmp_path, capsys, monkeypatch):
    sample_setup(tmp_path, monkeypatch, lambda doc: [])
    rc, _, err = run(capsys, sample_argv(tmp_path, tmp_path / "nope.json"))
    assert rc == 1 and "the order results are missing or unfinished" in err
    assert tree(tmp_path) == ["o.json"]


def test_drift_sample_without_order_runs_no_order_gate(tmp_path, capsys, monkeypatch):
    sample_setup(tmp_path, monkeypatch, lambda doc: [good_exchange(1)])
    argv = sample_argv(tmp_path, "unused")
    del argv[argv.index("--order") : argv.index("--order") + 2]
    session = make_session(keys={"model-en": "other-en", "model-ar": "other-ar"})
    rc, out, err = run(capsys, argv, session)
    assert rc == 0, err
    rec = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert rec["keys"] == session.keys and len(rec["requests"]) == 6
    assert "drift sample: 6 requests" in out


def test_calibrate_still_requires_the_order_option(capsys):
    argv = ["calibrate", "--work", "w", "--dev", "d", "--drift", "f", "--heldout", "h",
            "--heldout-sha256", "x", "--out", "o", "--record", "r"]
    with pytest.raises(SystemExit) as stop:
        model_run.main(argv, builder(make_session()))
    assert stop.value.code == 2
    capsys.readouterr()


# ---------------------------------------------------------------- drift


def reply_body(text, logprob):
    return json.dumps({
        "choices": [{"message": {"content": text}, "finish_reason": "stop",
                     "logprobs": {"content": [{"token": "a", "logprob": logprob}]}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2,
                  "prompt_tokens_details": {"cached_tokens": 7}},
    }).encode()


def drift_sample(tmp_path, bodies):
    requests = [{"cell": cell, "document": doc, "request_b64": base64.b64encode(req).decode(),
                 "status": 200, "body_b64": base64.b64encode(body).decode()}
                for cell, doc, req, body in bodies]
    return step_file(tmp_path / "s.json", "drift-sample", keys=KEYS, requests=requests)


def drift_argv(tmp_path, sample, *extra):
    return ["drift", "--work", tmp_path / "w", "--sample", sample, "--out", tmp_path / "d.json",
            *extra]


def test_drift_arithmetic(tmp_path, capsys):
    counts = {}

    def answer(seen):
        n = counts[seen.body] = counts.get(seen.body, 0) + 1
        if seen.body == b"A":
            return Answer(body=reply_body("same", -0.1 * n))
        return Answer(body=reply_body(f"text {n}", -0.2))

    sample = drift_sample(tmp_path, [
        ("model-en", "a.png", b"A", reply_body("same", -0.5)),
        ("model-ar", "b.png", b"B", reply_body("text 0", -0.2)),
    ])
    with FakeModelServer(answer) as server:
        session = make_session(server.url)
        rc, out, err = run(capsys, drift_argv(tmp_path, sample, "--repeats", "2"), session)
    assert rc == 0, err
    rec = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert rec["step"] == "drift" and rec["keys"] == KEYS
    assert rec["rounds"] == ["fresh", "repeat", "fresh-again"] and rec["repeats"] == 2
    a, b = rec["lines"]
    assert a["confidences"] == pytest.approx([math.exp(-0.5)] + [math.exp(-0.1 * n) for n in (1, 2, 3, 4)])
    assert a["texts_differ"] is False and b["texts_differ"] is True
    assert b["cached_tokens"] and len(a["seconds"]) == 4
    spread = math.exp(-0.1) - math.exp(-0.5)
    assert rec["tolerance"] == pytest.approx(spread)
    assert rec["max_spread_by_cell"]["model-en"] == pytest.approx(spread)
    assert rec["max_spread_by_cell"]["model-ar"] == pytest.approx(0.0)
    assert rec["text_changes"] == 1 and rec["first_request_seconds"] >= 0
    assert f"drift tolerance {spread:.4f} over 2 lines; text changes 1" in out
    assert len(server.seen) == 8


def test_drift_uses_no_warm_up(tmp_path, capsys):
    sample = drift_sample(tmp_path, [("model-en", "a.png", b"A", reply_body("x", -0.1))])
    with FakeModelServer(Answer(body=reply_body("x", -0.1))) as server:
        build = builder(make_session(server.url))
        rc = model_run.main([str(a) for a in drift_argv(tmp_path, sample)], build)
    assert rc == 0 and build.calls[0][1] is False


def test_drift_server_error_is_an_invalid_measurement(tmp_path, capsys):
    sample = drift_sample(tmp_path, [("model-en", "a.png", b"A", reply_body("x", -0.1))])
    with FakeModelServer(Answer(status=500, body=b"{}")) as server:
        rc, _, err = run(capsys, drift_argv(tmp_path, sample), make_session(server.url))
    assert rc == 1
    assert err.startswith("error: the model server did not answer during the repeated reads")
    assert not (tmp_path / "d.json").exists()


def test_drift_server_down(tmp_path, capsys):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{s.getsockname()[1]}/v1"
    sample = drift_sample(tmp_path, [("model-en", "a.png", b"A", reply_body("x", -0.1))])
    rc, _, err = run(capsys, drift_argv(tmp_path, sample), make_session(url))
    assert rc == 1 and "during the repeated reads" in err and err.count("\n") == 1
    assert not (tmp_path / "d.json").exists()


def test_drift_refuses_a_sample_of_other_keys(tmp_path, capsys):
    sample = drift_sample(tmp_path, [])
    record = json.loads(sample.read_text(encoding="utf-8"))
    sample.write_text(json.dumps({**record, "keys": {"model-en": "x"}}), encoding="utf-8")
    rc, _, err = run(capsys, drift_argv(tmp_path, sample))
    assert rc == 1 and "keys differ" in err
    assert not (tmp_path / "d.json").exists()


def test_drift_with_no_lines_has_zero_tolerance(tmp_path, capsys):
    sample = drift_sample(tmp_path, [])
    rc, out, _ = run(capsys, drift_argv(tmp_path, sample))
    assert rc == 0 and "drift tolerance 0.0000 over 0 lines; text changes 0" in out
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["tolerance"] == 0.0
