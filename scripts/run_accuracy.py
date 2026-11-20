"""Measures OCR confidence thresholds without a model server: calibrate on the development
set, grade a held-out batch once, or report the built-in fixtures apart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import accuracy_dev as ad
import accuracy_reads as ar
import accuracy_stats as stats
import heldout
from accuracy_eval import PageOutcome, auto_at, page_outcome, status_agrees
from heldout import HeldoutError
from script_errors import describe_os_error

from folio_ocr import calibration, orientation
from folio_ocr.charsets import is_arabic_letter
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.errors import FolioError
from folio_ocr.tesseract import decided_version, tessdata_problem

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA = "folio-accuracy/1"
NO_TESSERACT = "Tesseract was not found; set FOLIO_TESSERACT_PATH"
STATUS_DIFFERS = "the pipeline status differs from the threshold rule"


@dataclass(frozen=True)
class Context:
    reader: ar.Reader
    tesseract: Path
    keys: dict[str, str]
    work: Path
    jobs: int
    plan: ar.CellPlan = ar.DEFAULT_PLAN


@dataclass(frozen=True)
class ImageRead:
    outcome: PageOutcome
    setup_key: str | None
    tau: float | None
    seconds: float


@dataclass(frozen=True)
class CellState:
    units: tuple[stats.Unit, ...] = ()
    tally: heldout.Tally = field(default_factory=heldout.Tally)
    setup_keys: frozenset = frozenset()
    tau_missing: bool = False
    seconds: tuple[float, ...] = ()
    seeds: tuple[int, ...] = ()
    closed: bool = False
    stopped_by: str | None = None


def _orientation_constants() -> dict:
    return {
        "uncertain_margins": {
            "pair": orientation.UNCERTAIN_MARGIN_PAIR,
            "four": orientation.UNCERTAIN_MARGIN_FOUR,
        },
        "osd_floor": orientation.OSD_FLOOR,
    }


def _base_record(
    mode: str, tess: Path | None, started_at: str, jobs: int, **sha: str | None
) -> dict:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "status": "started",
        "started_at": started_at,
        "finished_at": None,
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "jobs": jobs,  # reading threads of this run (information only, in no setup key)
        },
        "versions": {
            "folio_ocr": ar.package("folio-ocr"),
            "onnxtr": ar.package("onnxtr"),
            "onnxruntime": ar.package("onnxruntime"),
            "tesseract": None if tess is None else decided_version(tess),
        },
        "tau_table_sha256": calibration.table_sha256(),
        **_orientation_constants(),
        "heldout_manifest_sha256": sha.get("heldout"),
        "dev_manifest_sha256": sha.get("dev"),
        "cells": [],
        "pages": None,
    }


def _context(args: argparse.Namespace, factory: Callable | None, plan: ar.CellPlan) -> Context:
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    settings = Settings.from_env(os.environ)
    tess = find_tesseract(settings)
    if tess is None:
        raise FolioError(NO_TESSERACT)
    decided_version(tess)  # the version is decided before any read, whatever the plan
    problem = tessdata_problem(settings.weights_dir)
    if problem is not None:
        raise FolioError(problem)
    reader = (factory or ar.pipeline_reader)(settings, tess)
    return Context(reader, tess, plan.keys(tess), Path(args.work), args.jobs, plan)


# ---------------------------------------------------------------- one read

def _read_image(
    ctx: Context, read: tuple[str, str, bool], name: str, data: bytes, gt: dict
) -> ImageRead:
    cell, mode, with_tesseract = read
    sha = hashlib.sha256(data).hexdigest()
    obj = ar.cached_read(ctx.work, ctx.keys[cell], sha, data, name, mode, with_tesseract, ctx.reader)
    pages = ar.pages_from_cache(obj)
    if len(pages) != 1:
        raise ValueError(f"{name} was read as {len(pages)} pages; one image is one page")
    orientations = ar.orientations_from_cache(obj)
    source, _, branch, margin, failed = (
        orientations[0] if orientations else (None, False, None, None, False)
    )
    outcome = page_outcome(
        gt, pages[0], ar.split_boxes_from_cache(obj), source, branch, margin, failed,
        readers=ctx.plan.readers,
    )
    return ImageRead(outcome, obj["setup_key"], obj["tau"], obj["seconds"])


# ---------------------------------------------------------------- graded cells

def _absorb(state: CellState, unit: stats.Unit, reads: Sequence[ImageRead], seed: int | None,
            key: str) -> CellState:
    frozen = calibration.lookup(key)
    for r in reads:
        checked = frozen is not None and r.setup_key == key
        if checked and not all(status_agrees(line, frozen) for line in r.outcome.lines):
            raise ValueError(STATUS_DIFFERS)
    bad = stats.unit_counts(unit, 0.0)[0]
    return replace(
        state,
        units=(*state.units, unit),
        tally=heldout.add(state.tally, bad),
        setup_keys=state.setup_keys | {r.setup_key for r in reads},
        tau_missing=state.tau_missing or any(r.tau is None for r in reads),
        seconds=(*state.seconds, *(r.seconds for r in reads)),
        seeds=state.seeds if seed is None else (*state.seeds, seed),
    )


def _read_group(ctx: Context, task: tuple) -> list[ImageRead]:
    read, images = task
    return [_read_image(ctx, read, i.name, i.data, i.gt) for i in images]


def _process(ctx: Context, states: dict, groups: Sequence[tuple], seed: int | None) -> dict:
    """Read every (language, images) group under each cell still open; return the new states."""
    tasks = [
        (read, images)
        for lang, images in groups
        for read in ar.cell_reads(lang, ctx.plan)
        if not states[read[0]].closed
    ]
    results = ar.parallel_map(lambda t: _read_group(ctx, t), tasks, ctx.jobs)
    out = dict(states)
    for ((cell, _mode, _with), images), reads in zip(tasks, results, strict=True):
        unit = stats.Unit(images[0].unit, tuple(r.outcome for r in reads))
        out[cell] = _absorb(out[cell], unit, reads, seed, ctx.keys[cell])
    return out


def _flags_every_line(units: Sequence[stats.Unit], tau: float | None) -> bool:
    """Whether `tau` is above every usable confidence of the OCR-read lines (or none is usable)."""
    if tau is None:
        return False
    scores = [ln.confidence for u in units for p in u.pages for ln in p.lines
              if ln.confidence is not None]
    return not scores or tau > max(scores)


def _cell_record(
    cell: str, state: CellState, keys: dict[str, str], plan: ar.CellPlan = ar.DEFAULT_PLAN
) -> dict:
    reader, language = plan.cells[cell]
    key = keys[cell]
    frozen = calibration.lookup(key)
    tau = 1.0 if frozen is None else frozen
    summary = stats.cell_summary(state.units, tau)
    interval = stats.missed_interval(state.units, tau)
    verdict, reason = stats.grade_verdict(
        summary,
        interval,
        key_ok=state.setup_keys <= {key},
        tau_present=not state.tau_missing and frozen is not None,
    )
    return {"cell": cell, "reader": reader, "document_language": language, "setup_key": key,
            "tau": frozen, "status": verdict, "reason": reason, **summary,
            **_orientation_constants(),
            "flags_every_line": _flags_every_line(state.units, frozen),
            "missed_error_rate": ar.interval(interval)}


FLAGS_EVERY_LINE = " (flags every line)"


def _cell_line(rec: dict) -> str:
    line = _cell_result(rec)
    return line + FLAGS_EVERY_LINE if rec["flags_every_line"] else line


def _cell_result(rec: dict) -> str:
    if rec["status"] not in ("pass", "fail"):
        return f"{rec['cell']}: {rec['reason']}"
    m = rec["missed_error_rate"]
    low, high = m["low"] * 100, m["high"] * 100
    return (f"{rec['cell']}: {rec['status']} {ar.pct(m['point'])} (95 % CI {low:.1f}-{high:.1f} %), "
            f"{rec['bad_lines']} bad lines from {rec['documents_with_bad_lines']} documents")


def _close(states: dict, last: bool) -> dict:
    out = {}
    for cell, st in states.items():
        if st.closed:
            out[cell] = st
        elif heldout.stop_reached(st.tally):
            out[cell] = replace(st, closed=True, stopped_by="minimums")
        elif last:
            out[cell] = replace(st, closed=True, stopped_by="last seed")
        else:
            out[cell] = st
    return out


def _grade_cells(states: dict, keys: dict[str, str], plan: ar.CellPlan) -> list[dict]:
    cells = []
    for cell, st in states.items():
        used = [st.seeds[0], st.seeds[-1]] if st.seeds else None
        cells.append({**_cell_record(cell, st, keys, plan), "seeds_used": used,
                      "stopped_by": st.stopped_by})
    return cells


def _seed_groups(seed: int, states: dict, plan: ar.CellPlan = ar.DEFAULT_PLAN) -> list[tuple]:
    open_cells = {c for c, st in states.items() if not st.closed}
    return [
        (lang, heldout.held_out_images(seed, token, layout))
        for token, lang, layout in heldout.DOCUMENTS
        if open_cells & {c for c, _, _ in ar.cell_reads(lang, plan)}
    ]


def _report(cells: list[dict]) -> None:
    for rec in cells:
        print(_cell_line(rec))


def cmd_grade(args: argparse.Namespace, factory: Callable | None, plan: ar.CellPlan) -> int:
    manifest, sha = heldout.load_manifest(Path(args.heldout))
    heldout.verify_generator(manifest)
    ctx = _context(args, factory, plan)
    record = {**_base_record("grade", ctx.tesseract, ar.now(), ctx.jobs, heldout=sha),
              "seeds": manifest["seeds"]}
    path = heldout.start_attempt(Path(args.results_dir), record, Path(args.heldout))
    order = heldout.seeds(manifest)
    states = {cell: CellState() for cell in plan.cells}
    for seed in order:
        if all(st.closed for st in states.values()):
            break
        states = _process(ctx, states, _seed_groups(seed, states, plan), seed)
        states = _close(states, seed == order[-1])
    cells = _grade_cells(states, ctx.keys, plan)
    seconds = [s for st in states.values() for s in st.seconds]
    heldout.finish_attempt(path, ar.finished(record, cells, seconds))
    _report(cells)
    print(path)
    return 0


# ---------------------------------------------------------------- fixtures

def _fixture_images(token: str, layout: str, recorded: dict[str, str]) -> tuple:
    images = []
    for condition in heldout.CONDITIONS:
        name = f"{token}-{layout}-{condition}.{'png' if condition == 'clean' else 'jpg'}"
        data = (FIXTURES / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != recorded.get(name):
            raise ValueError(f"{name} does not match the fixture manifest")
        gt_name = f"{token}-{layout}-{condition}.gt.json"
        gt_bytes = (FIXTURES / gt_name).read_bytes()
        if hashlib.sha256(gt_bytes).hexdigest() != recorded.get(gt_name):
            raise ValueError(f"{gt_name} does not match the fixture manifest")
        gt = json.loads(gt_bytes)
        images.append(heldout.HeldoutImage(name, data, gt, condition, f"{token}-{layout}"))
    return tuple(images)


def _out_folder(out: str) -> Path:
    """The results file path, its folder created (with parents) before any page is read."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _split_word_share(cells: list[dict]) -> float | None:
    """Share of OCR-read lines holding a split word in the OSD OnnxTR cell (None without it)."""
    for cell in cells:
        if cell["cell"] == "onnxtr-en-osd":
            return stats.rate(cell["split_word_lines"], cell["ocr_read_lines"])
    return None


def cmd_fixtures(args: argparse.Namespace, factory: Callable | None, plan: ar.CellPlan) -> int:
    out = _out_folder(args.out)
    ctx = _context(args, factory, plan)
    entries = json.loads((FIXTURES / "manifest.json").read_bytes())["fixtures"]
    recorded = {e["file"]: e["sha256"] for e in entries}
    recorded.update({e["gt"]: e["gt_sha256"] for e in entries})
    groups = [(lang, _fixture_images(token, layout, recorded))
              for token, lang, layout in heldout.DOCUMENTS]
    record = _base_record("fixtures", ctx.tesseract, ar.now(), ctx.jobs)
    states = _process(ctx, {cell: CellState() for cell in plan.cells}, groups, None)
    cells = [_cell_record(cell, st, ctx.keys, plan) for cell, st in states.items()]
    ar.write_json(out, ar.finished(record, cells, [s for st in states.values() for s in st.seconds]))
    _report(cells)
    print(args.out)
    return 0


# ---------------------------------------------------------------- calibrate

def _load_dev(dev: Path) -> tuple[list[dict], str]:
    raw = (dev / "manifest.json").read_bytes()
    try:
        docs = json.loads(raw)["documents"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("the dev manifest is not a valid manifest") from None
    for doc in docs:
        _check_dev_file(dev, doc, "file", "sha256")
        _check_dev_file(dev, doc, "gt", "gt_sha256")
    return docs, hashlib.sha256(raw).hexdigest()


def _check_dev_file(dev: Path, doc: dict, name_field: str, hash_field: str) -> None:
    """Raise ValueError when a dev document lacks a field or its file differs from its hash."""
    for needed in (name_field, hash_field, "language", "unit"):
        if not isinstance(doc, dict) or needed not in doc:
            raise ValueError(f"a dev manifest document is missing {needed}")
    _verified_bytes(dev, doc, name_field, hash_field)


def _verified_bytes(dev: Path, doc: dict, name_field: str, hash_field: str) -> bytes:
    """The bytes of a dev file, after checking them against the manifest's hash."""
    data = (dev / doc[name_field]).read_bytes()
    if hashlib.sha256(data).hexdigest() != doc[hash_field]:
        raise ValueError(f"{doc[name_field]} does not match the dev manifest")
    return data


def _dev_read(ctx: Context, dev: Path, task: tuple) -> tuple[ImageRead, dict]:
    doc, *reading = task
    gt = json.loads(_verified_bytes(dev, doc, "gt", "gt_sha256"))
    data = _verified_bytes(dev, doc, "file", "sha256")
    read = _read_image(ctx, tuple(reading), doc["file"], data, gt)
    return read, gt


def _accepted(outcome: object, tau: float) -> bool:
    if outcome.kind == "matched":
        return auto_at(outcome.line, tau)
    return outcome.kind == "grouping" and all(auto_at(c, tau) for c in outcome.covers)


def _arabic_share(ctx: Context, dev: Path, docs: list[dict], setup: tuple, tau: float | None,
                  limit: int | None) -> dict:
    """How many Arabic lines `setup` (cell, mode, with Tesseract) auto accepts when it reads the
    first `limit` Arabic and mixed dev documents (all when None) as English."""
    tasks = [(d, *setup) for d in docs if d["language"] in ("ar", "en+ar")][:limit]
    reads = ar.parallel_map(lambda t: _dev_read(ctx, dev, t), tasks, ctx.jobs)
    pairs = [(g, o) for read, gt in reads
             for g, o in zip(gt["lines"], read.outcome.gt, strict=True)
             if any(is_arabic_letter(ch) for ch in g["text"])]
    out = {"documents": len(tasks), "arabic_gt_lines": len(pairs), "auto_accepted": None,
           "share": None}
    if tau is None:
        return {**out, "reason": "no OnnxTR tau"}
    accepted = sum(_accepted(o, tau) for _, o in pairs)
    return {**out, "auto_accepted": accepted, "share": accepted / len(pairs) if pairs else None}


def _arabic_in_en(ctx: Context, dev: Path, docs: list[dict], taus: dict, limit: int | None) -> dict:
    """Arabic lines auto accepted under `--lang en`, for the vote setup (no Tesseract) and the OSD
    setup (with Tesseract), each at its own tau; `share` is the higher of the two."""
    setups = {"vote": ("onnxtr-en-vote", "en", False), "osd": ("onnxtr-en-osd", "en", True)}
    out = {name: _arabic_share(ctx, dev, docs, setup, taus[name], limit)
           for name, setup in setups.items()}
    shares = [m["share"] for m in out.values() if m["share"] is not None]
    return {**out, "share": max(shares, default=None)}


def _arabic_limit(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        limit = int(text)
    except ValueError:
        limit = 0
    if limit < 1:
        raise ValueError("--arabic-in-en-docs must be an integer of at least 1")
    return limit


NO_ONNXTR_CELLS = {"skipped": "the plan has no OnnxTR cells"}
_ONNXTR_CELLS = ("onnxtr-en-vote", "onnxtr-en-osd")


def _arabic_measure(ctx: Context, dev: Path, docs: list[dict], taus: dict,
                    limit: int | None) -> dict:
    """The Arabic-in-English measure, or a note that it was skipped when the plan lacks the two
    OnnxTR cells."""
    if not all(c in ctx.plan.cells for c in _ONNXTR_CELLS):
        return NO_ONNXTR_CELLS
    by_setup = {"vote": taus.get(ctx.keys["onnxtr-en-vote"]),
                "osd": taus.get(ctx.keys["onnxtr-en-osd"])}
    return _arabic_in_en(ctx, dev, docs, by_setup, limit)


def _arabic_line(measure: dict) -> str:
    if "skipped" in measure:
        return f"arabic in en: {measure['skipped']}"
    def pct(m: dict) -> str:
        return "not measured" if m["share"] is None else ar.pct(m["share"])

    return f"arabic in en auto accepted: vote {pct(measure['vote'])}, osd {pct(measure['osd'])}"


def _calibrated_cell(cell: str, dev: ad.DevCell, choice: stats.TauChoice, keys: dict,
                     taus: dict, plan: ar.CellPlan = ar.DEFAULT_PLAN) -> dict:
    reader, language = plan.cells[cell]
    key = keys[cell]
    tau = taus.get(key)
    at, units = (1.0 if tau is None else tau), dev.units
    return {"cell": cell, "reader": reader, "document_language": language, "setup_key": key,
            "tau": tau, "cell_tau": choice.tau, "status": choice.status, "reason": choice.reason,
            "documents_used": len(units), "stopped_by": dev.stopped_by,
            **_orientation_constants(),
            **stats.cell_summary(units, at), "flags_every_line": _flags_every_line(units, tau),
            "missed_error_rate": ar.interval(stats.missed_interval(units, at))}


def _calibrate_line(cell: str, choice: stats.TauChoice, tau: float | None, flags: bool) -> str:
    if choice.status != "calibrated":
        return f"{cell}: {choice.reason}"
    line = (f"{cell}: tau {tau:.3f} (cell tau {choice.tau:.3f}), dev bound "
            f"{ar.pct(choice.interval.bound_95)}, {choice.bad_lines} bad lines")
    return line + FLAGS_EVERY_LINE if flags else line


def cmd_calibrate(args: argparse.Namespace, factory: Callable | None, plan: ar.CellPlan) -> int:
    limit = _arabic_limit(args.arabic_in_en_docs)
    out = _out_folder(args.out)
    ctx, dev = _context(args, factory, plan), Path(args.dev)
    docs, dev_sha = _load_dev(dev)
    record = _base_record("calibrate", ctx.tesseract, ar.now(), ctx.jobs, dev=dev_sha)
    per_cell = ad.collect_dev(docs, lambda t: _dev_read(ctx, dev, t)[0], ctx.jobs, plan)
    choices = {cell: stats.choose_cell_tau(c.units) for cell, c in per_cell.items()}
    taus = stats.setup_taus([(ctx.keys[c], choices[c]) for c in plan.cells])
    cells = [_calibrated_cell(c, per_cell[c], choices[c], ctx.keys, taus, plan) for c in plan.cells]
    share = _split_word_share(cells)
    record = {**record, "taus": taus, "split_word_share_en": share}
    if args.arabic_in_en:
        measure = _arabic_measure(ctx, dev, docs, taus, limit)
        record = {**record, "measures": {"arabic_in_en": measure}}
    seconds = [s for c in per_cell.values() for s in c.seconds]
    ar.write_json(out, ar.finished(record, cells, seconds))
    for c in cells:
        print(_calibrate_line(c["cell"], choices[c["cell"]], c["tau"], c["flags_every_line"]))
    if share is not None:
        print(f"onnxtr-en split-word lines: {ar.pct(share)}")
    if args.arabic_in_en:
        print(_arabic_line(measure))
    print(f"TAUS = {json.dumps(taus, sort_keys=True)}")
    return 0


# ---------------------------------------------------------------- command line

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, func: Callable) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--jobs", type=int, default=1)
        p.set_defaults(func=func)
        return p

    cal = command("calibrate", cmd_calibrate)
    cal.add_argument("--dev", required=True)
    cal.add_argument("--out", required=True)
    cal.add_argument("--arabic-in-en", action="store_true")
    cal.add_argument("--arabic-in-en-docs", default=None)
    grade = command("grade", cmd_grade)
    grade.add_argument("--heldout", required=True)
    grade.add_argument("--results-dir", default=str(ROOT / "eval" / "results"))
    fix = command("fixtures", cmd_fixtures)
    fix.add_argument("--out", required=True)
    return parser


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(
    argv: list[str] | None = None,
    reader_factory: Callable | None = None,
    plan: ar.CellPlan = ar.DEFAULT_PLAN,
) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args, reader_factory, plan)
    except (HeldoutError, FolioError, ValueError) as exc:
        return _fail(str(exc))
    except OSError as exc:
        return _fail(describe_os_error(exc, args.work))
    except Exception as exc:  # noqa: BLE001 - the user sees one line, never a traceback
        return _fail(f"unexpected failure ({type(exc).__name__})")


if __name__ == "__main__":
    sys.exit(main())
