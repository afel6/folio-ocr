"""Held-out batch for grading OCR confidence thresholds.

Documents come from the current fixture generator with seeds other than its built-in one.
The manifest pins the generator and font hashes; an attempt file records each grading run
so one batch is graded once.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import make_fixtures
import numpy as np
from fixture_images import (
    DPI,
    PHOTO_QUALITY,
    SCAN_QUALITY,
    jpeg_bytes,
    photo_condition,
    png_bytes,
    scan_condition,
)
from fixture_output import document, gt_dict, json_bytes, lines_json, rendered

SCHEMA = "folio-heldout/1"
BUILTIN_SEED = make_fixtures.DEFAULT_SEED
ROOT = Path(__file__).resolve().parents[1]
GENERATOR_FILES: tuple[str, ...] = (
    "scripts/make_fixtures.py",
    *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "scripts").glob("fixture_*.py"))),
)
FONT_FILES: tuple[str, ...] = tuple(
    p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "tests/fixtures/fonts").glob("*.ttf"))
)
CONDITIONS = ("clean", "scan", "photo")
DOCUMENTS = (
    ("en", "en", "invoice"),
    ("en", "en", "letter"),
    ("ar", "ar", "invoice"),
    ("ar", "ar", "letter"),
    ("mixed", "en+ar", "invoice"),
    ("mixed", "en+ar", "letter"),
)
MIN_BAD_LINES = 100
MIN_DOCUMENTS = 30
STOP_RULE = (
    "per cell, seeds in increasing order; the cell takes no more seeds after the first seed at "
    "which it holds >= 100 bad lines from >= 30 documents with a bad line; at the last seed it "
    "stops anyway"
)
OVERLAP = "this seed range overlaps an earlier held-out batch; a re-tune needs a new seed range"
ALREADY_GRADED = "this held-out batch has already been graded; a new grade needs a new seed range"


class HeldoutError(Exception):
    """A held-out batch problem; the message is one friendly line."""


def text_sha256(path: Path) -> str:
    """sha256 of the file with every CRLF read as LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 of the exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(first: int, last: int, name: str) -> dict:
    """The manifest of the batch for seeds first..last, pinning the generator and fonts."""
    if first < 0 or last < first:
        raise HeldoutError("the seed range must run from a non-negative first seed to a last seed")
    if first <= BUILTIN_SEED <= last:
        raise HeldoutError("the seed range must not contain the built-in fixture seed")
    if not name:
        raise HeldoutError("the held-out batch needs a name")
    return {
        "schema": SCHEMA,
        "name": name,
        "generator": "scripts/make_fixtures.py --seed",
        "generator_files": {f: text_sha256(ROOT / f) for f in GENERATOR_FILES},
        "fonts": {f: file_sha256(ROOT / f) for f in FONT_FILES},
        "seeds": {"first": first, "last": last},
        "excluded_seeds": [BUILTIN_SEED],
        "documents": [list(d) for d in DOCUMENTS],
        "conditions": list(CONDITIONS),
        "cells": {
            "onnxtr-en": {"tokens": ["en"], "languages": "en"},
            "tesseract-en": {"tokens": ["en"], "languages": "en+ar"},
            "tesseract-ar": {"tokens": ["ar", "mixed"], "languages": "per document"},
        },
        "stop_rule": STOP_RULE,
        "min_bad_lines": MIN_BAD_LINES,
        "min_documents": MIN_DOCUMENTS,
        "resampling": {
            "unit": "generated document (seed, token, layout) with its clean, scan and photo images",
            "resamples": 2000,
            "rng_seed": 20261119,
        },
    }


def write_manifest(manifest: dict, path: Path) -> str:
    """Create the manifest file (never overwriting) and return its sha256."""
    path = Path(path)
    if not path.exists():
        _refuse_overlap_with_manifests(manifest["seeds"], path)
    try:
        with path.open("xb") as fh:
            fh.write(json_bytes(manifest))
    except FileExistsError:
        raise HeldoutError(f"held-out manifest {path.name} already exists") from None
    except OSError as exc:
        raise HeldoutError(f"cannot write held-out manifest {path.name}: {exc.strerror}") from exc
    return file_sha256(path)


def load_manifest(path: Path) -> tuple[dict, str]:
    """The manifest and the sha256 of its file bytes."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HeldoutError(f"cannot read held-out manifest {path.name}: {exc.strerror}") from exc
    try:
        manifest = json.loads(raw)
    except ValueError:
        raise HeldoutError(f"held-out manifest {path.name} is not valid JSON") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise HeldoutError(f"held-out manifest {path.name} does not have schema {SCHEMA}")
    if _span(manifest.get("seeds")) is None:
        raise HeldoutError(f"held-out manifest {path.name} is missing seeds first and last")
    return manifest, hashlib.sha256(raw).hexdigest()


def _span(seeds: object) -> tuple[int, int] | None:
    """(first, last) of a {first, last} seed record; None when either end is not an integer."""
    if not isinstance(seeds, dict):
        return None
    first, last = seeds.get("first"), seeds.get("last")
    ints = all(isinstance(v, int) and not isinstance(v, bool) for v in (first, last))
    return (first, last) if ints else None


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _refuse_overlap_with_manifests(seeds: dict, path: Path) -> None:
    """Raise HeldoutError when another manifest file beside `path` has an overlapping range."""
    mine = _span(seeds)
    if mine is None or not path.parent.is_dir():
        return
    for other in sorted(path.parent.glob("*.json")):
        if other.name == path.name:
            continue
        theirs = _span(load_manifest(other)[0]["seeds"])
        if _overlaps(mine, theirs):
            raise HeldoutError(OVERLAP)


def _check_files(kind: str, recorded: dict, current: tuple[str, ...], digest) -> None:
    for rel in sorted({*recorded, *current}):
        if rel not in recorded:
            raise HeldoutError(f"{kind} {rel} is not in the held-out manifest")
        if rel not in current:
            raise HeldoutError(f"{kind} {rel} is in the held-out manifest but missing on disk")
        if digest(ROOT / rel) != recorded[rel]:
            raise HeldoutError(f"{kind} {rel} changed since the held-out manifest was written")


def verify_generator(manifest: dict) -> None:
    """Raise HeldoutError naming the first generator or font file that differs from the manifest."""
    _check_files("generator file", manifest.get("generator_files", {}), GENERATOR_FILES, text_sha256)
    _check_files("font file", manifest.get("fonts", {}), FONT_FILES, file_sha256)


def seeds(manifest: dict) -> tuple[int, ...]:
    """The seeds first..last inclusive, in increasing order, without the excluded seeds."""
    span, skip = manifest["seeds"], set(manifest.get("excluded_seeds", []))
    return tuple(s for s in range(span["first"], span["last"] + 1) if s not in skip)


@dataclass(frozen=True)
class HeldoutImage:
    name: str
    data: bytes
    gt: dict
    condition: str
    unit: str


def held_out_images(seed: int, token: str, layout: str) -> tuple[HeldoutImage, ...]:
    """The clean, scan and photo images of one document with ground truth, built in memory."""
    language = next(lang for tok, lang, lay in DOCUMENTS if tok == token and lay == layout)
    stem, unit = f"{token}-{layout}", f"{seed}-{token}-{layout}"
    doc, placed = document(token, language, layout, seed)
    _, clean, boxes = rendered(placed, "naskh" if layout == "letter" else "amiri")
    rng = make_fixtures._image_rng(seed, stem)
    scan, h_scan = scan_condition(clean, rng)
    photo, h_photo = photo_condition(clean, rng)
    meta = (language, layout, doc.direction)
    rows = (
        ("clean", "png", png_bytes(clean), clean.size, np.eye(3)),
        ("scan", "jpg", jpeg_bytes(scan, SCAN_QUALITY), scan.size, h_scan),
        ("photo", "jpg", jpeg_bytes(photo, PHOTO_QUALITY), photo.size, h_photo),
    )
    images = []
    for condition, ext, data, size, homography in rows:
        name = f"{stem}-{condition}.{ext}"
        lines = lines_json(placed, boxes, homography, size)
        gt = {"file": name, **gt_dict(condition, meta, size, homography, lines), "dpi": DPI}
        images.append(HeldoutImage(name, data, json.loads(json_bytes(gt)), condition, unit))
    return tuple(images)


@dataclass(frozen=True)
class Tally:
    bad_lines: int = 0
    documents_with_bad: int = 0


def add(t: Tally, bad_lines: int) -> Tally:
    """The tally after one more document with `bad_lines` bad lines."""
    return Tally(t.bad_lines + bad_lines, t.documents_with_bad + (1 if bad_lines > 0 else 0))


def stop_reached(t: Tally) -> bool:
    return t.bad_lines >= MIN_BAD_LINES and t.documents_with_bad >= MIN_DOCUMENTS


def _attempt_records(results_dir: Path) -> tuple[tuple[Path, object], ...]:
    """Every results file in `results_dir` with its parsed JSON."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return ()
    found = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            found.append((path, json.loads(path.read_bytes())))
        except OSError as exc:
            raise HeldoutError(f"cannot read attempt file {path.name}: {exc.strerror}") from exc
        except ValueError:
            raise HeldoutError(f"attempt file {path.name} cannot be read as JSON") from None
    return tuple(found)


def graded_attempts(results_dir: Path, manifest_sha: str) -> tuple[Path, ...]:
    """Every attempt file in `results_dir` recorded against this manifest hash."""
    return tuple(
        path
        for path, record in _attempt_records(results_dir)
        if isinstance(record, dict) and record.get("heldout_manifest_sha256") == manifest_sha
    )


def _refuse_overlap_with_attempts(results_dir: Path, mine: tuple[int, int]) -> None:
    for _, record in _attempt_records(results_dir):
        theirs = _span(record.get("seeds")) if isinstance(record, dict) else None
        if theirs is not None and _overlaps(mine, theirs):
            raise HeldoutError(OVERLAP)


def start_attempt(results_dir: Path, record: dict, manifest_path: Path | None = None) -> Path:
    """Write the attempt record for a batch that has not been graded; return its path.

    A record with `seeds` {first, last} is refused when that range overlaps another manifest
    beside `manifest_path` or the seeds of any attempt already recorded."""
    results_dir, sha = Path(results_dir), record["heldout_manifest_sha256"]
    if graded_attempts(results_dir, sha):
        raise HeldoutError(ALREADY_GRADED)
    mine = _span(record.get("seeds"))
    if mine is not None:
        if manifest_path is not None:
            _refuse_overlap_with_manifests(record["seeds"], Path(manifest_path))
        _refuse_overlap_with_attempts(results_dir, mine)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"grade-{sha[:12]}-{stamp}.json"
    try:
        with path.open("xb") as fh:
            fh.write(json_bytes(record))
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise HeldoutError(f"cannot write attempt file {path.name}: {exc.strerror}") from exc
    return path


def finish_attempt(path: Path, record: dict) -> None:
    """Atomically replace the attempt file with the finished record."""
    path = Path(path)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".attempt-", suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(json_bytes(record))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)
        raise HeldoutError(f"cannot write attempt file {path.name}: {exc.strerror}") from exc
