"""Reading documents for the accuracy run: cells, the pipeline reader and the on-disk read cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import accuracy_eval
import accuracy_stats as stats
import numpy as np

from folio_ocr import calibration
from folio_ocr import ocr as ocr_module
from folio_ocr.document import Box, Document, Line, Page
from folio_ocr.export import to_json
from folio_ocr.ocr_setup import onnxtr_setup, tesseract_setup
from folio_ocr.pipeline import read_bytes
from folio_ocr.tesseract import decided_version

# cell -> (reader, document language group); the two OnnxTR cells read the same English
# documents, one with Tesseract present (page orientation by OSD), one without (by vote)
CELLS = {
    "onnxtr-en-osd": ("onnxtr", "en"),
    "onnxtr-en-vote": ("onnxtr", "en"),
    "tesseract-en": ("tesseract", "en"),
    "tesseract-ar": ("tesseract", "ar"),
}
# document language -> (cell, `--lang` mode, Tesseract path given) reads
_READS = {
    "en": (
        ("onnxtr-en-osd", "en", True),
        ("onnxtr-en-vote", "en", False),
        ("tesseract-en", "en+ar", True),
    ),
    "ar": (("tesseract-ar", "ar", True),),
    "en+ar": (("tesseract-ar", "en+ar", True),),
}

# What the pipeline recorded about the way up of a page:
# (source "osd" | "vote", uncertain, branch "pair" | "four" | "none", vote margin | None,
#  Tesseract's orientation answer failed)
PageOrientation = tuple[str, bool, str, float | None, bool]
# (image bytes, file name, --lang mode, Tesseract path given)
#   -> (document, seconds, split-word boxes, one orientation per page in call order)
Reader = Callable[
    [bytes, str, str, bool],
    tuple[Document, float, tuple[Box, ...], tuple[PageOrientation, ...]],
]

_captured = threading.local()  # per thread: split-word boxes and page orientations seen since reset


def reset_split_boxes() -> None:
    """Forget the split-word boxes this thread has collected."""
    _captured.boxes = []


def take_split_boxes() -> tuple[Box, ...]:
    """The split-word boxes this thread has collected since its last reset."""
    return tuple(getattr(_captured, "boxes", ()))


def install_split_capture() -> None:
    """Wrap `folio_ocr.ocr.recognise_words` (once) so it records the boxes of split words."""
    inner = ocr_module.recognise_words
    if getattr(inner, "records_split_words", False):
        return

    def recognise_words(models, rgb, boxes):
        words = inner(models, rgb, boxes)
        found = [tuple(w.box) for w in words if getattr(w, "split", False)]
        _captured.boxes = [*getattr(_captured, "boxes", ()), *found]
        return words

    recognise_words.records_split_words = True  # type: ignore[attr-defined]
    ocr_module.recognise_words = recognise_words


def reset_orientations() -> None:
    """Forget the page orientations this thread has collected."""
    _captured.orientations = []


def take_orientations() -> tuple[PageOrientation, ...]:
    """The page orientations this thread has collected since its last reset."""
    return tuple(getattr(_captured, "orientations", ()))


def install_orientation_capture() -> None:
    """Wrap `folio_ocr.ocr.page_orientation` (once) so it records what decided each page."""
    inner = ocr_module.page_orientation
    if getattr(inner, "records_orientation", False):
        return

    def page_orientation(*args, **kwargs):
        result = inner(*args, **kwargs)
        _captured.orientations = [
            *getattr(_captured, "orientations", ()),
            (
                result.source,
                bool(result.uncertain),
                result.branch,
                result.margin,
                result.error is not None,
            ),
        ]
        return result

    page_orientation.records_orientation = True  # type: ignore[attr-defined]
    ocr_module.page_orientation = page_orientation


def pipeline_reader(settings: Any, tesseract_path: Path | None) -> Reader:
    """A reader that runs the real pipeline, times each read and collects its split-word boxes
    and page orientations; Tesseract is given to the pipeline only when the read asks for it."""
    install_split_capture()
    install_orientation_capture()

    def read(data: bytes, name: str, mode: str, with_tesseract: bool) -> tuple:
        reset_split_boxes()
        reset_orientations()
        start = time.perf_counter()
        doc = read_bytes(
            data,
            name,
            languages=mode,
            weights_dir=settings.weights_dir,
            tesseract_path=tesseract_path if with_tesseract else None,
            line_timeout_s=settings.line_timeout_s,
        )
        return doc, time.perf_counter() - start, take_split_boxes(), take_orientations()

    return read


def expected_keys(tesseract_path: Path | None) -> dict[str, str]:
    """Cell -> the calibration setup key this install reads it with; the cells that need
    Tesseract are left out when there is none."""
    keys = {"onnxtr-en-vote": onnxtr_setup().key}
    if tesseract_path is not None:
        version = decided_version(tesseract_path)
        keys["onnxtr-en-osd"] = onnxtr_setup(calibration.version_major_minor(version)).key
        keys["tesseract-en"] = keys["tesseract-ar"] = tesseract_setup(version).key
    return keys


@dataclass(frozen=True)
class CellPlan:
    """Which cells a run measures and how: cell -> (reader, document language); document
    language -> (cell, `--lang` mode, Tesseract path given) reads; the `Line.reader` values
    counted as OCR-read; Tesseract path -> cell -> expected setup key."""

    cells: Mapping[str, tuple[str, str]]
    reads: Mapping[str, tuple[tuple[str, str, bool], ...]]
    readers: frozenset[str]
    keys: Callable[[Path | None], dict[str, str]]


DEFAULT_PLAN = CellPlan(CELLS, _READS, accuracy_eval.OCR_READERS,
                        lambda path: expected_keys(path))  # looked up at call time


def cell_reads(
    language: str, plan: CellPlan = DEFAULT_PLAN
) -> tuple[tuple[str, str, bool], ...]:
    """(cell, `--lang` mode, Tesseract path given) reads a document of `language` gets."""
    if language not in plan.reads:
        raise ValueError(f"unknown document language {language!r}")
    return plan.reads[language]


def parallel_map[T, R](fn: Callable[[T], R], items: Iterable[T], jobs: int) -> list[R]:
    """`fn` over `items` on `jobs` threads; results come back in input order."""
    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items))


def document_to_cache(
    doc: Document,
    seconds: float,
    split_boxes: Sequence[Box] = (),
    orientations: Sequence[PageOrientation] = (),
) -> dict:
    """The JSON-ready record of one read."""
    calib = doc.engine.calibration
    return {
        "seconds": seconds,
        "setup_key": None if calib is None else calib.setup_key,
        "tau": None if calib is None else calib.tau,
        "split_boxes": [list(b) for b in split_boxes],
        "orientations": [list(o) for o in orientations],
        "pages": json.loads(to_json(doc, indent=None))["pages"],
    }


def split_boxes_from_cache(obj: dict) -> tuple[Box, ...]:
    """The split-word boxes of a cached read (none for a read cached without them)."""
    return tuple(_box(b) for b in obj.get("split_boxes", ()))


def orientations_from_cache(obj: dict) -> tuple[PageOrientation, ...]:
    """The page orientations of a cached read (none for a read cached without them)."""
    return tuple(
        (str(src), bool(unsure), str(branch), None if margin is None else float(margin), bool(failed))
        for src, unsure, branch, margin, failed in obj.get("orientations", ())
    )


def _has_orientations(obj: dict) -> bool:
    """Whether a cached read carries the current record of every page's orientation."""
    return "orientations" in obj and all(len(o) == 5 for o in obj["orientations"])


def _box(value: Sequence[float]) -> tuple[float, ...]:
    return tuple(value)


def _line(obj: dict) -> Line:
    return Line(**{**obj, "box": _box(obj["box"]), "flags": tuple(obj["flags"])})


def pages_from_cache(obj: dict) -> tuple[Page, ...]:
    """The pages of a cached read, with tuples for boxes, transform and flags."""
    return tuple(
        Page(
            **{
                **p,
                "transform": tuple(_box(row) for row in p["transform"]),
                "flags": tuple(p["flags"]),
                "unread_regions": tuple(_box(b) for b in p["unread_regions"]),
                "lines": tuple(_line(ln) for ln in p["lines"]),
            }
        )
        for p in obj["pages"]
    )


def cache_path(
    work: Path, expected_key: str, doc_sha: str, mode: str, with_tesseract: bool
) -> Path:
    """Where the read of document `doc_sha` under `mode` is kept for this setup, this tau
    table and this use of Tesseract."""
    material = f"{expected_key}|{calibration.table_sha256()}|{int(with_tesseract)}"
    digest = hashlib.sha256(material.encode()).hexdigest()
    return Path(work) / "reads" / digest[:16] / f"{doc_sha}-{mode.replace('+', '_')}.json"


def _write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def key_mismatch(name: str, found: str | None, expected: str) -> str:
    """One line naming a document read twice with another setup key than its cell's."""
    return (f"{name} was read twice with setup key {'none' if found is None else found}, "
            f"not its cell's key {expected}; nothing of it was kept")


def _usable_cache(path: Path) -> dict | None:
    """The cached read at `path` when it carries the current orientation record, else None;
    a damaged file is refused."""
    try:
        cached = json.loads(path.read_bytes())
        return cached if _has_orientations(cached) else None
    except FileNotFoundError:
        return None
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"the cached read {path.name} is damaged; delete it and run again") from None


def _keyed_read(
    expected_key: str, tries: int, data: bytes, name: str, mode: str, with_tesseract: bool,
    reader: Reader,
) -> dict:
    """A read whose setup key is `expected_key`, trying at most `tries` times; a read of
    another key is never returned."""
    found = None
    for _ in range(tries):
        doc, seconds, split_boxes, orientations = reader(data, name, mode, with_tesseract)
        obj = document_to_cache(doc, seconds, split_boxes, orientations)
        found = obj["setup_key"]
        if found == expected_key:
            return obj
    raise ValueError(key_mismatch(name, found, expected_key))


def cached_read(
    work: Path,
    expected_key: str,
    doc_sha: str,
    data: bytes,
    name: str,
    mode: str,
    with_tesseract: bool,
    reader: Reader,
) -> dict:
    """The cached read when it carries `expected_key`, else a fresh read written to the cache.
    A cached read of another key counts as the first read, so a document is read at most twice
    to get the key; a read of another key is never written nor returned (ValueError)."""
    path = cache_path(work, expected_key, doc_sha, mode, with_tesseract)
    cached = _usable_cache(path)
    if cached is not None and cached.get("setup_key") == expected_key:
        return cached
    tries = 1 if cached is not None else 2
    obj = _keyed_read(expected_key, tries, data, name, mode, with_tesseract, reader)
    _write_atomic(path, obj)
    return obj


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} %"


def package(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def pages_block(seconds: Sequence[float]) -> dict:
    if not seconds:
        return {"count": 0, "seconds_p50": None, "seconds_p90": None}
    p50, p90 = (float(q) for q in np.percentile(seconds, [50, 90]))
    return {"count": len(seconds), "seconds_p50": p50, "seconds_p90": p90}


def finished(record: dict, cells: list, seconds: Sequence[float]) -> dict:
    return {**record, "status": "finished", "finished_at": now(), "cells": cells,
            "pages": pages_block(seconds)}


def interval(i: stats.Interval) -> dict:
    return {"point": i.point, "low": i.low, "high": i.high, "bound_95": i.bound_95,
            "resamples": stats.RESAMPLES, "unit": "document"}


def write_json(path: Path, record: dict) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".accuracy-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write((json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
