"""Reports whether the frozen OCR constants still hold on this machine's CPU: recomputes the page
orientation vote, Tesseract's orientation answer and line recall on the development pages with the
shipped code, and compares them with a reference file made on another machine."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from accuracy_eval import turn_wrong
from PIL import Image, ImageOps
from script_errors import describe_os_error

from folio_ocr import ocr_setup, onnxtr_models, orientation, tesseract
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.geometry import rotation, straighten
from folio_ocr.ingest import page_pixels
from folio_ocr.metrics import compose, map_box, match_lines
from folio_ocr.ocr_words import Word, find_words, group_lines

SCHEMA = "folio-dev-constants/1"
NO_TESSERACT = "Tesseract was not found; set FOLIO_TESSERACT_PATH"
RECALL_TOLERANCE = 0.005
OSD_TIMEOUT_S = 60.0
FIRST_UNITS = 10


class CheckError(Exception):
    """A problem reported as one line."""


# ---------------------------------------------------------------- one page

def _off_upright(w: int, h: int, angle: float, homography) -> bool:
    """Whether the page turned by `angle` is more than 45 degrees from upright."""
    return turn_wrong(homography, rotation(w, h, angle)[0])


def _true_quarter(w: int, h: int, homography) -> int | None:
    for q in (0, 90, 180, 270):
        if not _off_upright(w, h, q, homography):
            return q
    return None


@functools.cache
def _environment() -> tuple[Path, Path, onnxtr_models.Models]:
    """Tesseract, the tessdata folder and the OnnxTR models, loaded once per process."""
    settings = Settings.from_env(os.environ)
    binary = find_tesseract(settings)
    if binary is None:
        raise CheckError(NO_TESSERACT)
    return binary, settings.weights_dir / "tessdata", onnxtr_models.load(settings.weights_dir)


def _osd_field(answer, failure: str | None):
    if failure is not None:
        return f"failure: {failure}"
    return None if answer is None else [answer.quarter, answer.confidence]


def _recall(models, image, gt, homography, quarter: int | None) -> tuple[int | None, int | None]:
    """Lines matched and ground-truth lines after straightening by the true quarter plus skew."""
    if quarter is None:
        return None, None
    rgb = np.asarray(image.convert("RGB"))
    first = onnxtr_models.detect(models, rgb)
    _, found = orientation._turned(models, image, rgb, first, quarter)
    skew = orientation._skew(found) if len(found.boxes) else 0
    flat, matrix = straighten(image, (quarter + skew) % 360)
    gray, flat_rgb = flat.convert("L"), np.asarray(flat.convert("RGB"))
    boxes = find_words(models, gray, flat_rgb, found if skew == 0 else None)
    if not boxes:
        return None, None
    lines = group_lines(gray, [Word(b, "", math.nan) for b in boxes], "ltr")
    full = compose(matrix, homography)
    gt_boxes = [map_box(full, tuple(ln["clean_box"])) for ln in gt["lines"]]
    matched = match_lines(gt_boxes, [ln.box for ln in lines], 0.5)
    return len(matched), len(gt_boxes)


def _vote_fields(models, image, homography, w: int, h: int) -> dict:
    vote = orientation.page_orientation(models, image)
    return {
        "vote_branch": vote.branch, "vote_margin": vote.margin, "vote_angle": vote.angle,
        "vote_uncertain": vote.uncertain,
        "vote_wrong": _off_upright(w, h, vote.angle, homography),
    }


def _final_fields(models, image, homography, w: int, h: int, osd_call) -> dict:
    final = orientation.page_orientation(models, image, osd=osd_call)
    return {
        "final_source": final.source, "final_angle": final.angle,
        "final_uncertain": final.uncertain,
        "final_wrong": _off_upright(w, h, final.angle, homography),
    }


def page_check(path: Path, gt: dict) -> dict:
    """The vote, Tesseract's answer, the final choice and the line recall for one page."""
    binary, tessdata, models = _environment()
    with Image.open(path) as opened:
        image = page_pixels(ImageOps.exif_transpose(opened))
    homography = gt["homography"]
    w, h = image.size
    png = tesseract.page_png(image)
    answer, failure = None, None
    try:
        answer = tesseract.osd(binary, tessdata, png, OSD_TIMEOUT_S)
    except tesseract.TesseractFailure as exc:
        failure = str(exc)
    quarter = _true_quarter(w, h, homography)

    def osd_call(_image):
        if failure is not None:
            raise tesseract.TesseractFailure(failure)
        return answer

    matched, total = _recall(models, image, gt, homography, quarter)
    return {
        "true_quarter": quarter,
        **_vote_fields(models, image, homography, w, h),
        "osd": _osd_field(answer, failure),
        **_final_fields(models, image, homography, w, h, osd_call),
        "recall_matched": matched, "recall_total": total,
    }


# ---------------------------------------------------------------- summary and verdict

def _share(matched: int, total: int) -> float | None:
    return matched / total if total else None


def _vote_branch(pages: Sequence[dict], branch: str) -> dict:
    mine = [p for p in pages if p["vote_branch"] == branch]
    wrong = [p for p in mine if p["vote_wrong"]]
    margins = [p["vote_margin"] for p in wrong if p["vote_margin"] is not None]
    return {
        "pages": len(mine), "wrong": len(wrong),
        "largest_wrong_margin": max(margins, default=None),
        "wrong_unflagged": sum(not p["vote_uncertain"] for p in wrong),
    }


def _osd_counts(pages: Sequence[dict]) -> dict:
    out = {"right": 0, "no_answer": 0, "failure": 0, "wrong": 0}
    for p in pages:
        osd = p["osd"]
        if osd is None:
            out["no_answer"] += 1
        elif isinstance(osd, str):
            out["failure"] += 1
        else:
            out["right" if osd[0] == p["true_quarter"] else "wrong"] += 1
    return out


def _osd_quarter(osd):
    return osd[0] if isinstance(osd, list) else osd


def _differs(pages: Sequence[dict], old: dict, field: str, pick: Callable) -> dict:
    """Pages whose `field` differs from the reference page of the same unit (information only)."""
    units = [p["unit"] for p in pages
             if p["unit"] in old and pick(p[field]) != pick(old[p["unit"]][field])]
    return {"count": len(units), "first": units[:FIRST_UNITS]}


def summarise(pages: Sequence[dict], reference: dict | None) -> dict:
    """Counts over the page records; differences are measured against `reference` when given."""
    old = {} if reference is None else reference["pages"]
    matched = sum(p["recall_matched"] or 0 for p in pages)
    total = sum(p["recall_total"] or 0 for p in pages)
    wrong = [p for p in pages if p["final_wrong"]]
    return {
        "pages": len(pages),
        "vote": {b: _vote_branch(pages, b) for b in ("pair", "four", "none")},
        "osd": _osd_counts(pages),
        "osd_used_wrong": sum(p["final_source"] == "osd" for p in wrong),
        "final": {"wrong": len(wrong),
                  "wrong_unflagged": sum(not p["final_uncertain"] for p in wrong)},
        "recall": {"matched": matched, "total": total, "share": _share(matched, total)},
        "vote_angle_differs": _differs(pages, old, "vote_angle", float),
        "osd_differs": _differs(pages, old, "osd", _osd_quarter),
        "manifest_differs": len({p["unit"] for p in pages} ^ set(old)) if old else 0,
    }


def verdict(summary: dict, reference: dict) -> list[str]:
    """One line per problem; an empty list means the constants hold."""
    problems = []
    for branch, v in summary["vote"].items():
        if v["wrong_unflagged"]:
            problems.append(f"{v['wrong_unflagged']} wrong vote picks not flagged ({branch})")
    if summary["osd_used_wrong"]:
        problems.append(f"{summary['osd_used_wrong']} wrong pages decided by the osd answer")
    if summary["final"]["wrong_unflagged"]:
        problems.append(f"{summary['final']['wrong_unflagged']} final wrong turns not flagged")
    now, was = summary["recall"]["share"], reference["summary"]["recall"]["share"]
    if now is not None and was is not None and abs(now - was) > RECALL_TOLERANCE:
        problems.append(f"line recall share {now:.4f} differs from the reference {was:.4f}")
    if summary["manifest_differs"]:
        problems.append(f"{summary['manifest_differs']} dev pages differ from the reference manifest")
    return problems


# ---------------------------------------------------------------- command line

def check_pages(tasks: Sequence[tuple[Path, dict]], jobs: int, check: Callable | None = None):
    """Records of every (path, ground truth) task in task order, serial or over `jobs` processes."""
    check = check or page_check
    paths, gts = [t[0] for t in tasks], [t[1] for t in tasks]
    if jobs == 1:
        results = [check(p, g) for p, g in zip(paths, gts, strict=True)]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(check, paths, gts))
    return [{"unit": g["unit"], **r} for g, r in zip(gts, results, strict=True)]


def _frozen() -> dict:
    return {
        "uncertain_margin_pair": orientation.UNCERTAIN_MARGIN_PAIR,
        "uncertain_margin_four": orientation.UNCERTAIN_MARGIN_FOUR,
        "osd_floor": orientation.OSD_FLOOR,
        "gap_factor": ocr_setup.OCR_GAP_FACTOR,
    }


def _load_json(raw: bytes, what: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError:
        raise CheckError(f"the {what} is not valid JSON") from None
    if not isinstance(data, dict):
        raise CheckError(f"the {what} is not a JSON object")
    return data


def _load_reference(path: Path, manifest_sha: str) -> dict:
    ref = _load_json(path.read_bytes(), "reference file")
    if ref.get("dev_manifest_sha256") != manifest_sha:
        raise CheckError("the dev manifest differs from the one the reference was made on")
    if ref.get("frozen") != _frozen():
        raise CheckError("the frozen constants differ from the reference's")
    if not isinstance(ref.get("summary"), dict) or not isinstance(ref.get("pages"), dict):
        raise CheckError("the reference file lacks its summary or pages")
    return ref


def _tasks(dev: Path, raw: bytes) -> list[tuple[Path, dict]]:
    try:
        docs = _load_json(raw, "dev manifest")["documents"]
        return [
            (dev / d["file"], _load_json((dev / d["gt"]).read_bytes(), "ground truth file"))
            for d in docs
        ]
    except (KeyError, TypeError):
        raise CheckError("the dev manifest is not a valid manifest") from None


def _write(path: Path, data: dict) -> None:
    """Write `data` as JSON through a temporary file in the same folder."""
    fd, tmp = tempfile.mkstemp(dir=path.resolve().parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    os.replace(tmp, path)


def _run(args: argparse.Namespace) -> int:
    if args.jobs < 1:
        raise CheckError("--jobs must be at least 1")
    if not Path(args.out).resolve().parent.is_dir():
        raise CheckError("the folder for --out does not exist")
    dev = Path(args.dev)
    raw = (dev / "manifest.json").read_bytes()
    ref = _load_reference(Path(args.reference), hashlib.sha256(raw).hexdigest())
    _environment()
    pages = check_pages(_tasks(dev, raw), args.jobs)
    summary = summarise(pages, ref)
    problems = verdict(summary, ref)
    _write(Path(args.out), {"schema": SCHEMA, "frozen": _frozen(), "summary": summary,
                            "pages": {p["unit"]: p for p in pages}, "problems": problems})
    print(json.dumps(summary, indent=1))
    print("\n".join(f"STOP: {p}" for p in problems) or "constants hold")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dev", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except CheckError as exc:
        message = str(exc)
    except OSError as exc:
        message = describe_os_error(exc, args.dev)
    except Exception as exc:  # noqa: BLE001 - one line for the user, never a traceback
        message = f"unexpected failure ({type(exc).__name__})"
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
