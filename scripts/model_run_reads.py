"""Reading the dev documents with the model: cells, keys, the reader that keeps each reply, the
per-line records joined to those replies, and the checks run over them."""

from __future__ import annotations

import base64
import binascii
import collections
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import accuracy_reads as ar
import accuracy_stats as stats
import model_run_lines as lines
from accuracy_dev import stop_reason
from accuracy_eval import IOU_THRESHOLD, page_outcome
from model_run_scores import aggregation_check, auc, pins  # noqa: F401
from model_run_server import NO_KEY, ModelRunError, exchange_record

from folio_ocr import metrics
from folio_ocr.calibration import version_major_minor
from folio_ocr.charsets import is_arabic_letter
from folio_ocr.document import Line, Page
from folio_ocr.flags import clean_text
from folio_ocr.model_reply import parse_reply
from folio_ocr.model_setup import ServerInfo, model_setup
from folio_ocr.normalize import norm_v2
from folio_ocr.ocr_setup import onnxtr_setup, tesseract_setup
from folio_ocr.pipeline import read_bytes

MODEL_CELLS = {"model-en": ("model", "en"), "model-ar": ("model", "ar")}
MODEL_READS = {
    "en": (("model-en", "en", True),),
    "ar": (("model-ar", "ar", True),),
    "en+ar": (("model-ar", "en+ar", True),),
}
CELL_LANGUAGES = {"model-en": ("en",), "model-ar": ("ar", "en+ar")}
SCRIPT_GROUP = {"model-en": "en", "model-ar": "ar"}
MODEL_READERS = frozenset({"model"})
MEASURE_KEYS = tuple(lines.line_measures("", "", frozenset()))


def model_keys(info: ServerInfo, name: str, tesseract_version: str) -> dict[str, str]:
    """Cell -> the calibration key of the model's logprob reads on that cell's page set-up."""
    onnxtr = onnxtr_setup(version_major_minor(tesseract_version)).preprocessing
    pages = {"model-en": onnxtr, "model-ar": tesseract_setup(tesseract_version).preprocessing}
    keys = {}
    for cell, preprocessing in pages.items():
        setup = model_setup(info, name, "logprobs", preprocessing)
        if setup is None:
            raise ModelRunError(NO_KEY)
        keys[cell] = setup.key
    return keys


def model_plan(keys: Mapping[str, str]) -> ar.CellPlan:
    return ar.CellPlan(MODEL_CELLS, MODEL_READS, MODEL_READERS, lambda _tess: dict(keys))


def replies_path(work: Path, key: str, doc_sha: str, mode: str, with_tesseract: bool) -> Path:
    """The file holding the captured replies next to the cached read they belong to."""
    cache = ar.cache_path(work, key, doc_sha, mode, with_tesseract)
    return cache.with_name(cache.stem + ".replies.json")


def _write_replies(path: Path, exchanges: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ar.write_json(path, {"exchanges": [exchange_record(e) for e in exchanges]})


def _refuse_a_stopped_server(exchanges: Sequence[Any]) -> None:
    """A call that failed for a reason other than a line time-out means the server went away."""
    for ex in exchanges:
        failed = ex.error is not None and not ex.timed_out
        bad_status = ex.status is not None and not 200 <= ex.status < 300
        if failed or bad_status:
            raise ValueError(
                "the model server stopped answering during a read; start it and run again"
            )


def _doc_key(doc: Any) -> str | None:
    calibration = doc.engine.calibration
    return None if calibration is None else calibration.setup_key


def model_reader_factory(model: Any, capture: Any, keys: Mapping[str, str], work: Path):
    """`factory(settings, tess)` gives a reader that reads through `model` and writes the
    captured replies of the read it returns to disk first. A document whose setup key is not
    its cell's is read once more; if it still differs nothing of it is written."""

    def factory(settings: Any, tess: Path | None):
        ar.install_split_capture()
        ar.install_orientation_capture()

        def one_read(data: bytes, name: str, mode: str, with_tesseract: bool) -> tuple:
            ar.reset_split_boxes()
            ar.reset_orientations()
            capture.reset()
            start = time.perf_counter()
            doc = read_bytes(
                data,
                name,
                languages=mode,
                weights_dir=settings.weights_dir,
                tesseract_path=tess if with_tesseract else None,
                line_timeout_s=settings.line_timeout_s,
                model=model,
            )
            seconds = time.perf_counter() - start
            exchanges = capture.take()
            _refuse_a_stopped_server(exchanges)
            return doc, seconds, exchanges, ar.take_split_boxes(), ar.take_orientations()

        def read(data: bytes, name: str, mode: str, with_tesseract: bool) -> tuple:
            if mode not in MODEL_READS:
                raise ValueError(f"unknown reading mode {mode!r}")
            key = keys[MODEL_READS[mode][0][0]]
            for _ in range(2):
                doc, seconds, exchanges, boxes, orientations = one_read(
                    data, name, mode, with_tesseract
                )
                found = _doc_key(doc)
                if found == key:
                    sha = hashlib.sha256(data).hexdigest()
                    _write_replies(replies_path(work, key, sha, mode, with_tesseract), exchanges)
                    return doc, seconds, boxes, orientations
            raise ValueError(ar.key_mismatch(name, found, key))

        return read

    return factory


def _check_dev_file(dev: Path, doc: Any, name_field: str, hash_field: str) -> None:
    for needed in (name_field, hash_field, "language", "unit"):
        if not isinstance(doc, dict) or needed not in doc:
            raise ValueError(f"a dev manifest document is missing {needed}")
    _verified_bytes(dev, doc, name_field, hash_field)


def _verified_bytes(dev: Path, doc: dict, name_field: str, hash_field: str) -> bytes:
    data = (dev / doc[name_field]).read_bytes()
    if hashlib.sha256(data).hexdigest() != doc[hash_field]:
        raise ValueError(f"{doc[name_field]} does not match the dev manifest")
    return data


def load_dev(dev: Path) -> tuple[list[dict], str]:
    """The dev manifest's documents, each checked against its hashes, and the manifest's hash."""
    raw = (dev / "manifest.json").read_bytes()
    try:
        docs = json.loads(raw)["documents"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("the dev manifest is not a valid manifest") from None
    for doc in docs:
        _check_dev_file(dev, doc, "file", "sha256")
        _check_dev_file(dev, doc, "gt", "gt_sha256")
    return docs, hashlib.sha256(raw).hexdigest()


def dev_docs_for(docs: Sequence[dict], cell: str) -> list[dict]:
    return [d for d in docs if d["language"] in CELL_LANGUAGES[cell]]


@dataclass(frozen=True)
class DocRead:
    doc: dict
    gt: dict
    page: Page
    exchanges: tuple[dict, ...]
    seconds: float
    setup_key: str | None


def _load_replies(path: Path, name: str) -> tuple[dict, ...]:
    try:
        saved = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise ModelRunError(
            f"the captured replies of {name} are missing; delete its cached read and run again"
        ) from None
    except ValueError:
        saved = None
    items = saved.get("exchanges") if isinstance(saved, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ModelRunError(
            f"the captured replies of {name} are damaged; delete its cached read and run again"
        )
    return tuple(items)


def read_dev_doc(work: Path, keys: Mapping[str, str], reader, dev: Path, doc: dict) -> DocRead:
    """Read one dev document (from the cache when it was read before) with its replies."""
    data = (dev / doc["file"]).read_bytes()
    cell, mode, with_tess = MODEL_READS[doc["language"]][0]
    sha = hashlib.sha256(data).hexdigest()
    obj = ar.cached_read(work, keys[cell], sha, data, doc["file"], mode, with_tess, reader)
    pages = ar.pages_from_cache(obj)
    if len(pages) != 1:
        raise ValueError(f"{doc['file']} was read as {len(pages)} pages; one image is one page")
    exchanges = _load_replies(replies_path(work, keys[cell], sha, mode, with_tess), doc["file"])
    gt = json.loads((dev / doc["gt"]).read_bytes())
    return DocRead(doc, gt, pages[0], exchanges, obj["seconds"], obj["setup_key"])


def _reply_key(rec: dict) -> tuple[str, float | None] | None:
    """(cleaned text, confidence) the pipeline made of a successful reply; None for others."""
    status = rec.get("status")
    if not isinstance(status, int) or not 200 <= status <= 299:
        return None
    try:
        reading = parse_reply(status, base64.b64decode(rec["body_b64"]))
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None
    return None if reading.error is not None else (clean_text(reading.text)[0], reading.confidence)


def join_replies(page: Page, exchanges: Sequence[dict]) -> dict[int, dict]:
    """Line index -> the reply record the model line was formed from; each record serves one
    line, the earliest request first."""
    waiting: dict[Any, collections.deque] = collections.defaultdict(collections.deque)
    for rec in exchanges:
        key = _reply_key(rec)
        if key is not None:
            waiting[key].append(rec)
    joined = {}
    for i, ln in enumerate(page.lines):
        queue = waiting.get((ln.text, ln.confidence)) if ln.reader == "model" else None
        if queue:
            joined[i] = queue.popleft()
    return joined


def _homography(gt: Mapping) -> Any:
    return tuple(tuple(float(v) for v in row) for row in gt["homography"])


def matched_lines(gt: Mapping, page: Page) -> dict[int, tuple[int, Line]]:
    """GT line index -> (page line index, page line) for the lines that match by box."""
    boxes = metrics.gt_line_boxes(
        [g["clean_box"] for g in gt["lines"]], _homography(gt), page.transform
    )
    pairs = metrics.match_lines(boxes, [ln.box for ln in page.lines], IOU_THRESHOLD)
    return {g: (j, page.lines[j]) for g, j in pairs}


def _empty_measures() -> dict:
    return {k: None for k in MEASURE_KEYS}


def _record(read: DocRead, cell: str, vocabulary: frozenset[str], g: int, hit, joined) -> dict:
    gt_text = read.gt["lines"][g]["text"]
    ln = hit[1] if hit else None
    is_model = ln is not None and ln.reader == "model"
    reply = joined.get(hit[0]) if is_model else None
    reply = reply or {}
    measures = lines.line_measures(gt_text, ln.text, vocabulary) if is_model else _empty_measures()
    return {
        "document": read.doc["file"],
        "unit": read.doc["unit"],
        "cell": cell,
        "script_group": SCRIPT_GROUP[cell],
        "gt": gt_text,
        "classes": list(lines.order_classes(gt_text)),
        "matched": ln is not None,
        "reader": ln.reader if ln else None,
        "text": ln.text if ln else None,
        "flags": list(ln.flags) if is_model else [],
        "confidence": ln.confidence if ln else None,
        "status": ln.status if ln else None,
        "entries": reply.get("entries"),
        "finish_reason": reply.get("finish_reason"),
        "max_tokens": reply.get("max_tokens"),
        "completion_tokens": reply.get("completion_tokens"),
        **measures,
    }


def line_records(read: DocRead, cell: str, vocabulary: frozenset[str]) -> list[dict]:
    """One record per ground-truth line, in order, with the measures of its model reply."""
    hits = matched_lines(read.gt, read.page)
    joined = join_replies(read.page, read.exchanges)
    n = len(read.gt["lines"])
    return [_record(read, cell, vocabulary, g, hits.get(g), joined) for g in range(n)]


def vocabulary_of(gts: Sequence[Mapping]) -> frozenset[str]:
    return frozenset(w for gt in gts for g in gt["lines"] for w in norm_v2(g["text"]).split())


def bad_line_count(read: DocRead) -> int:
    outcome = page_outcome(read.gt, read.page, readers=MODEL_READERS)
    return stats.unit_counts(stats.Unit(read.doc["unit"], (outcome,)), 0.0)[0]


def _classes_full(records: Sequence[dict]) -> bool:
    model = [r for r in records if r["reader"] == "model"]
    return all(
        sum(name in r["classes"] for r in model) >= lines.MIN_ORDER_LINES
        for name in lines.ORDER_CLASSES
    )


def order_pass(docs: Sequence[dict], read_one: Callable[[dict], DocRead], jobs: int):
    """Read the Arabic dev documents in batches of `jobs` until every order class holds enough
    model lines and the bad-line minimums hold, or the cap or the end of the set is reached."""
    reads: list[DocRead] = []
    records: list[dict] = []
    bad = with_bad = 0
    stopped = "dev set end"
    step = max(jobs, 1)
    for start in range(0, len(docs), step):
        batch = docs[start : start + min(step, stats.DEV_CAP_DOCS - start)]
        for read in ar.parallel_map(read_one, batch, jobs):
            n = bad_line_count(read)
            bad, with_bad = bad + n, with_bad + (n > 0)
            reads.append(read)
            records.extend(line_records(read, "model-ar", frozenset()))
        if stop_reason(bad, with_bad, len(reads)) == "minimums" and _classes_full(records):
            stopped = "minimums"
            break
        if len(reads) >= stats.DEV_CAP_DOCS:
            stopped = "cap"
            break
    summary = {
        **lines.order_summary(records),
        "documents": len(reads),
        "bad_lines": bad,
        "documents_with_bad_lines": with_bad,
        "stopped_by": stopped,
    }
    return tuple(reads), summary


def _hyp(hit, attr: str) -> str:
    return "" if hit is None else norm_v2(getattr(hit[1], attr))


def fixture_cer(gt: Mapping, page: Page) -> dict:
    """Character error rate of the page's final text over the ground-truth lines."""
    hits = matched_lines(gt, page)
    n = matched = chars = errors = 0
    for g, gt_line in enumerate(gt["lines"]):
        ref = norm_v2(gt_line["text"])
        if not ref:
            continue
        n += 1
        matched += g in hits
        counts = metrics.char_errors(ref, _hyp(hits.get(g), "final_text"))
        chars, errors = chars + counts.reference_length, errors + counts.edits
    return {"gt_lines": n, "matched": matched, "reference_chars": chars, "errors": errors,
            "cer": errors / chars if chars else None}


def _pool(rows: Sequence[dict]) -> dict:
    pooled = lines.pooled(rows)
    return {k: pooled[k] for k in ("reference_chars", "errors", "cer")}


def tesseract_compare(model_reads: Sequence[DocRead], tesseract_pages: Mapping[str, Page]) -> dict:
    """Pooled error rate of the model's and Tesseract's text over the Arabic ground-truth lines
    of every document both read."""
    model_rows: list[dict] = []
    tess_rows: list[dict] = []
    docs = 0
    for read in model_reads:
        page = tesseract_pages.get(read.doc["file"])
        if page is None:
            continue
        docs += 1
        ours, theirs = matched_lines(read.gt, read.page), matched_lines(read.gt, page)
        for g, gt_line in enumerate(read.gt["lines"]):
            if not any(is_arabic_letter(c) for c in gt_line["text"]):
                continue
            model_rows.append({"gt": gt_line["text"], "text": ours[g][1].text if g in ours else ""})
            tess_rows.append({"gt": gt_line["text"], "text": theirs[g][1].text if g in theirs else ""})
    return {"documents": docs, "arabic_gt_lines": len(model_rows),
            "model": _pool(model_rows), "tesseract": _pool(tess_rows)}
