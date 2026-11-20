"""The first steps of the laptop model run: reply-contract checks, the order rule, and the
repeated-read drift measurement."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import accuracy_reads as ar
import model_run_common as common
import model_run_reads as mr
import model_run_server as srv
from model_run_common import check_keys, load_step, say, write_step
from model_run_server import ModelRunError

from folio_ocr import model_http
from folio_ocr.model_reply import ORDER_RULE, parse_reply

ROUNDS = ("fresh", "repeat", "fresh-again")
UNFLAGGED_EXAMPLES = 20


def _positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"--{name} must be at least 1")


def step_contract(args, build) -> int:
    started = ar.now()
    _, config = common.model_config()
    result = srv.contract_checks(
        config, expect_version=args.expect_version, expect_digest=args.expect_digest
    )
    write_step(Path(args.out), "contract", started, result)
    for check in result["checks"]:
        say(f"{'ok  ' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}")
    if result["passed"]:
        return 0
    say("STOP: a required reply-contract check failed")
    return 1


def _unflagged(records: list[dict]) -> list[dict]:
    wrong = [r for r in records if r["order_error"] is True and "mixed_order" not in r["flags"]]
    fields = ("document", "gt", "text", "classes", "flags")
    return [{k: r[k] for k in fields} for r in wrong[:UNFLAGGED_EXAMPLES]]


def step_order(args, build) -> int:
    started, work, dev = ar.now(), Path(args.work), Path(args.dev)
    docs, dev_sha = mr.load_dev(dev)
    session = build(work)
    reader = session.reader_factory(session.settings, session.tesseract)
    reads, summary = mr.order_pass(
        mr.dev_docs_for(docs, "model-ar"),
        lambda d: mr.read_dev_doc(work, session.keys, reader, dev, d),
        args.jobs,
    )
    records = [r for read in reads for r in mr.line_records(read, "model-ar", frozenset())]
    write_step(Path(args.out), "order", started, {
        "dev_manifest_sha256": dev_sha, "order_rule": ORDER_RULE, "keys": session.keys,
        "model": session.facts, **summary, "unflagged_examples": _unflagged(records),
    })
    for name, row in summary["classes"].items():
        say(f"{name}: lines {row['lines']}, order errors {row['order_errors']}, "
            f"unflagged {row['unflagged_order_errors']} -> {row['status']}")
    if summary["verdict"] == "frozen":
        say(f"order rule frozen: {ORDER_RULE}")
        return 0
    missing = ", ".join(n for n, row in summary["classes"].items() if row["status"] != "ok")
    say(f"STOP: the order rule misses or lacks dev evidence for: {missing}")
    return 1


def _usable(exchange: dict) -> bool:
    """A successful reply with logprob entries and a confidence."""
    status = exchange.get("status")
    entries = exchange.get("entries")
    if not isinstance(status, int) or not 200 <= status <= 299:
        return False
    if not isinstance(entries, list) or not entries:
        return False
    try:
        reading = parse_reply(status, base64.b64decode(exchange["body_b64"]))
    except (KeyError, TypeError, ValueError):
        return False
    return reading.error is None and reading.confidence is not None


def _cell_requests(session, args, cell: str, docs: list[dict], reader) -> list[dict]:
    found: list[dict] = []
    for doc in mr.dev_docs_for(docs, cell):
        read = mr.read_dev_doc(Path(args.work), session.keys, reader, Path(args.dev), doc)
        for ex in read.exchanges:
            if len(found) < args.lines and _usable(ex):
                found.append({"cell": cell, "document": doc["file"],
                              "request_b64": ex["request_b64"], "status": ex["status"],
                              "body_b64": ex["body_b64"]})
        if len(found) >= args.lines:
            break
    return found


def step_drift_sample(args, build) -> int:
    _positive(args.lines, "lines")
    started = ar.now()
    docs, dev_sha = mr.load_dev(Path(args.dev))
    order = None if args.order is None else common.frozen_order(Path(args.order), dev_sha)
    session = build(Path(args.work))
    if order is not None:
        check_keys(order, session.keys)
    reader = session.reader_factory(session.settings, session.tesseract)
    requests = [r for cell in mr.MODEL_CELLS for r in _cell_requests(session, args, cell, docs, reader)]
    write_step(Path(args.out), "drift-sample", started, {"keys": session.keys, "requests": requests})
    say(f"drift sample: {len(requests)} requests")
    return 0


@dataclass(frozen=True)
class Replay:
    confidence: float
    text: str
    cached_tokens: int | None
    seconds: float


def _replay(target: model_http.Target, request: bytes, timeout_s: float) -> Replay:
    started = time.perf_counter()
    try:
        reply = model_http.post(target, request, timeout_s)
    except model_http.ModelCallError as exc:
        raise ModelRunError(f"the model server did not answer during the repeated reads: {exc}") from None
    seconds = time.perf_counter() - started
    if not 200 <= reply.status < 300:
        raise ModelRunError(
            f"the model server did not answer during the repeated reads: HTTP {reply.status}")
    reading = parse_reply(reply.status, reply.body)
    if reading.error is not None or reading.confidence is None:
        raise ModelRunError("the model server did not answer during the repeated reads: "
                            "a reply held no confidence")
    return Replay(reading.confidence, reading.text, srv.usage(reply.body)["cached_tokens"], seconds)


def _original(sample_request: dict) -> tuple[bytes, float, str]:
    try:
        body = base64.b64decode(sample_request["body_b64"])
        reading = parse_reply(sample_request["status"], body)
        request = base64.b64decode(sample_request["request_b64"])
    except (KeyError, TypeError, ValueError):
        raise ModelRunError("the drift sample is damaged; make it again") from None
    if reading.error is not None or reading.confidence is None:
        raise ModelRunError("the drift sample holds a reply without a confidence; make it again")
    return request, reading.confidence, reading.text


def _schedule(count: int, repeats: int) -> list[int]:
    """Request indexes in reading order: each once, each `repeats` times, each once again."""
    return [*range(count), *(i for i in range(count) for _ in range(repeats)), *range(count)]


def _line(sample_request: dict, confidence: float, text: str, replays: list[Replay]) -> dict:
    confidences = [confidence, *(r.confidence for r in replays)]
    return {
        "cell": sample_request["cell"], "document": sample_request["document"],
        "confidences": confidences, "cached_tokens": [r.cached_tokens for r in replays],
        "seconds": [r.seconds for r in replays],
        "texts_differ": len({text, *(r.text for r in replays)}) > 1,
    }


def step_drift(args, build) -> int:
    _positive(args.repeats, "repeats")
    started = ar.now()
    sample = load_step(Path(args.sample), "drift-sample")
    session = build(Path(args.work), warm=False)
    check_keys(sample, session.keys)
    target = model_http.target(session.config.url, session.config.api_key)
    originals = [_original(r) for r in sample["requests"]]
    replays: list[list[Replay]] = [[] for _ in originals]
    first = None
    for i in _schedule(len(originals), args.repeats):
        replay = _replay(target, originals[i][0], args.timeout_s)
        replays[i].append(replay)
        first = replay.seconds if first is None else first
    lines = [_line(r, o[1], o[2], rep) for r, o, rep in zip(sample["requests"], originals, replays)]
    by_cell: dict[str, float] = {}
    for line in lines:
        spread = max(line["confidences"]) - min(line["confidences"])
        by_cell[line["cell"]] = max(by_cell.get(line["cell"], 0.0), spread)
    tolerance = max(by_cell.values(), default=0.0)
    changes = sum(line["texts_differ"] for line in lines)
    write_step(Path(args.out), "drift", started, {
        "keys": session.keys, "rounds": list(ROUNDS), "repeats": args.repeats, "lines": lines,
        "tolerance": tolerance, "max_spread_by_cell": by_cell, "text_changes": changes,
        "first_request_seconds": first})
    say(f"drift tolerance {tolerance:.4f} over {len(lines)} lines; text changes {changes}")
    return 0
