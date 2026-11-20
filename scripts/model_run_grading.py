"""The steps that calibrate, measure, grade and read the fixtures with the model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import accuracy_reads as ar
import heldout
import model_run_common as common
import model_run_lines as ml
import model_run_reads as mr
import model_run_vram as vram
import run_accuracy
from model_run_common import check_keys, load_step, say, write_step
from model_run_server import ModelRunError

from folio_ocr import calibration, pipeline

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DIFFERS = "the held-out manifest differs from its recorded sha256; stop"


def _fixed_heldout(path: Path, expected: str) -> str:
    """The held-out manifest's sha256, which must equal `expected`; the generator is verified."""
    manifest, sha = heldout.load_manifest(path)
    if sha != expected.lower():
        raise ModelRunError(DIFFERS)
    heldout.verify_generator(manifest)
    return sha


def _vram(work: Path, name: str, smi: str | None) -> vram.VramMonitor:
    work.mkdir(parents=True, exist_ok=True)
    return vram.VramMonitor(work / name, vram.nvidia_smi_command(smi))


def _ungraded_heldout(args) -> str:
    """The fixed held-out manifest's sha256, once it is known to have no grade yet."""
    sha = _fixed_heldout(Path(args.heldout), args.heldout_sha256)
    if heldout.graded_attempts(Path(args.results_dir), sha):
        raise ModelRunError(
            "this held-out batch already has a grade; model calibration must come before it")
    return sha


def _peak_fields(log: Path) -> dict:
    """The peak GPU memory of a step, or why it could not be read."""
    try:
        return {"peak_vram_mib": vram.peak_mib(log)}
    except (ModelRunError, OSError) as exc:
        return {"peak_vram_mib": None, "vram_error": _one_line(exc)}


def step_calibrate(args, build) -> int:
    started, work, out = ar.now(), Path(args.work), Path(args.out)
    common.refuse_local_setup("the package calibration")
    _, dev_sha = mr.load_dev(Path(args.dev))
    order = common.frozen_order(Path(args.order), dev_sha)
    drift = load_step(Path(args.drift), "drift")
    sha = _ungraded_heldout(args)
    session = build(work)
    check_keys(order, session.keys)
    check_keys(drift, session.keys)
    argv = ["calibrate", "--dev", str(args.dev), "--work", str(work), "--out", str(out),
            "--jobs", str(args.jobs)]
    with _vram(work, "vram-calibrate.csv", args.nvidia_smi):
        rc = run_accuracy.main(argv, reader_factory=session.reader_factory,
                               plan=mr.model_plan(session.keys))
    if rc != 0:
        return rc
    _ungraded_heldout(args)  # the held-out batch is still the fixed one and still ungraded
    cal, cal_sha = common.read_json_file(out, "calibration")
    taus = cal.get("taus") or {}
    if cal.get("dev_manifest_sha256") != dev_sha:
        raise ModelRunError("the dev manifest changed during the calibration; nothing was recorded")
    peak = _peak_fields(work / "vram-calibrate.csv")
    write_step(Path(args.record), "calibrate", started, {
        "keys": session.keys, "model": session.facts,
        "calibration": {"path": str(out), "sha256": cal_sha}, "taus": taus,
        "heldout_fixed": {"path": str(args.heldout), "sha256": sha},
        "dev_manifest_sha256": dev_sha, "drift_tolerance": drift["tolerance"],
        **peak, "tau_table_sha256_at_calibration": calibration.table_sha256(calibration.TAUS)})
    say(f"MODEL TAUS = {json.dumps(taus, sort_keys=True)}")
    if not taus:
        say("STOP: no model cell was calibrated; do not grade")
    if "vram_error" in peak:
        raise ModelRunError(peak["vram_error"])
    return 0 if taus else 1


def _file_ref(record: dict, field: str) -> tuple[Path, str]:
    ref = record.get(field)
    if not isinstance(ref, dict) or "path" not in ref or "sha256" not in ref:
        raise ModelRunError(f"the {record.get('step')} results are damaged: no {field}")
    return Path(ref["path"]), ref["sha256"]


def _tesseract_pages(session, work: Path, dev: Path, reads) -> dict:
    """The page Tesseract alone makes of each document, from the cache when read before."""
    reader = ar.pipeline_reader(session.settings, session.tesseract)
    key = ar.expected_keys(session.tesseract)["tesseract-ar"]
    pages = {}
    for read in reads:
        name = read.doc["file"]
        data = (dev / name).read_bytes()
        mode = mr.MODEL_READS[read.doc["language"]][0][1]
        obj = ar.cached_read(work, key, common.sha256_bytes(data), data, name, mode, True, reader)
        pages[name] = ar.pages_from_cache(obj)[0]
    return pages


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_measures(measures: dict, aggregation: dict, compare: dict, pins: dict) -> None:
    for group, m in measures.items():
        say(f"{group}: cer {ar.pct(m['cer'])}, cer_with_tashkeel {ar.pct(m['cer_with_tashkeel'])}, "
            f"truncated_share {ar.pct(m['truncated_share'])}, "
            f"digit_script_kept_share {ar.pct(m['digit_script_kept_share'])}")
    say("aggregation auc: " + ", ".join(
        f"{k} {_fmt(aggregation[k])}" for k in ("exp_min", "exp_mean", "exp_sum")))
    say(f"model cer {ar.pct(compare['model']['cer'])} vs tesseract cer "
        f"{ar.pct(compare['tesseract']['cer'])}")
    for cell, pin in pins.items():
        say(f"{cell}: pinnable {pin['pinnable']} of {pin['eligible']} lines")


def step_measure(args, build) -> int:
    started, work, dev = ar.now(), Path(args.work), Path(args.dev)
    record = load_step(Path(args.record), "calibrate")
    cal = common.read_json_checked(*_file_ref(record, "calibration"), "calibration")
    docs, dev_sha = mr.load_dev(dev)
    if dev_sha != record.get("dev_manifest_sha256"):
        raise ModelRunError("the dev set changed since the calibrate step; stop")
    session = build(work)
    check_keys(record, session.keys)
    used = {c["cell"]: c["documents_used"] for c in cal["cells"]}
    reader = session.reader_factory(session.settings, session.tesseract)
    reads = {
        cell: ar.parallel_map(lambda d: mr.read_dev_doc(work, session.keys, reader, dev, d),
                              mr.dev_docs_for(docs, cell)[: used.get(cell, 0)], args.jobs)
        for cell in mr.MODEL_CELLS
    }
    vocabulary = mr.vocabulary_of([r.gt for rs in reads.values() for r in rs])
    records = [rec for cell, rs in reads.items() for r in rs
               for rec in mr.line_records(r, cell, vocabulary)]
    taus = {cell: record["taus"].get(session.keys[cell]) for cell in mr.MODEL_CELLS}
    compare = mr.tesseract_compare(reads["model-ar"], _tesseract_pages(session, work, dev, reads["model-ar"]))
    measures, aggregation = ml.measure_summary(records), mr.aggregation_check(records)
    pins = mr.pins(records, taus, record["drift_tolerance"])
    write_step(Path(args.out), "measure", started, {
        "keys": session.keys, "measures": measures, "order_after_calibration": ml.order_summary(records),
        "aggregation": aggregation, "pins": pins, "tesseract_compare": compare,
        "dev_lines": records})
    _print_measures(measures, aggregation, compare, pins)
    if aggregation["stop"]:
        say("STOP: another logprob aggregation separates good and bad dev lines better; "
            "report before grading")
        return 1
    return 0


def _refuse_ungradable(record: dict) -> None:
    if not record.get("taus"):
        raise ModelRunError("no model cell was calibrated; there is nothing to grade")
    if any(calibration.lookup(k, calibration.TAUS) != v for k, v in record["taus"].items()):
        raise ModelRunError("the tau table does not hold the calibrated model taus; fill them in first")


def _one_line(exc: Exception) -> str:
    return common.ascii_only(" ".join(str(exc).split()) or type(exc).__name__)


def _grade_fields(results: Path, sha: str, rc: int, vram_log: Path) -> tuple[dict, str | None]:
    """The grade step's result fields and the first error met while collecting them."""
    fields: dict = {**_peak_fields(vram_log), "grade": None, "exit_code": rc}
    errors: list[str] = [fields["vram_error"]] if "vram_error" in fields else []
    try:
        attempts = heldout.graded_attempts(results, sha)
        if rc == 0 and not attempts:
            raise ModelRunError("the grade file was not found in the results folder")
        if attempts:
            newest = max(attempts, key=lambda p: p.name)
            fields["grade"] = {"path": str(newest), "sha256": common.file_sha256(newest)}
    except (ModelRunError, heldout.HeldoutError, OSError, ValueError) as exc:
        fields["grade_error"] = _one_line(exc)
        errors.append(fields["grade_error"])
    failed = rc != 0 or "grade_error" in fields
    fields["status"] = "failed" if failed else "finished"
    return fields, (errors[0] if errors else None)


def step_grade(args, build) -> int:
    started, work, results = ar.now(), Path(args.work), Path(args.results_dir)
    common.refuse_local_setup("the grade")
    record = load_step(Path(args.record), "calibrate")
    _, recorded = _file_ref(record, "heldout_fixed")
    sha = _fixed_heldout(Path(args.heldout), args.heldout_sha256)
    if sha != recorded:
        raise ModelRunError(DIFFERS)
    _refuse_ungradable(record)
    if heldout.graded_attempts(results, sha):
        raise ModelRunError(heldout.ALREADY_GRADED)
    session = build(work)
    check_keys(record, session.keys)
    argv = ["grade", "--heldout", str(args.heldout), "--work", str(work), "--results-dir",
            str(results), "--jobs", str(args.jobs)]
    with _vram(work, "vram-grade.csv", args.nvidia_smi):
        rc = run_accuracy.main(argv, reader_factory=session.reader_factory,
                               plan=mr.model_plan(session.keys))
    fields, error = _grade_fields(results, sha, rc, work / "vram-grade.csv")
    write_step(Path(args.out), "grade", started, {"keys": session.keys, **fields})
    if error is not None:
        raise ModelRunError(error)
    return rc


def _fixture_files() -> list[tuple[str, bytes, dict]]:
    """(file, bytes, ground truth) of every fixture in the manifest, each checked by its hash."""
    entries = json.loads((FIXTURES / "manifest.json").read_bytes())["fixtures"]
    found = []
    for entry in sorted(entries, key=lambda e: e["file"]):
        data = (FIXTURES / entry["file"]).read_bytes()
        if common.sha256_bytes(data) != entry["sha256"]:
            raise ValueError(f"{entry['file']} does not match the fixture manifest")
        gt = json.loads((FIXTURES / f"{Path(entry['file']).stem}.gt.json").read_bytes())
        found.append((entry["file"], data, gt))
    return found


def _languages(name: str) -> str:
    return {"en-": "en", "ar-": "ar", "mixed-": "en+ar"}.get(name.split("-")[0] + "-", "en+ar")


def _read_fixture(session, item: tuple[str, bytes, dict]) -> tuple[dict, float]:
    name, data, gt = item
    languages = _languages(name)
    started = time.perf_counter()
    doc = pipeline.read_bytes(
        data, name, languages=languages, weights_dir=session.settings.weights_dir,
        tesseract_path=session.tesseract, line_timeout_s=session.settings.line_timeout_s,
        model=session.model)
    seconds = time.perf_counter() - started
    record = {"languages": languages, "seconds": seconds, "pages": len(doc.pages),
              "origin": doc.pages[0].origin if doc.pages else None}
    if len(doc.pages) != 1:
        return {**record, "cer": None, "error": f"read as {len(doc.pages)} pages"}, seconds
    return {**record, **mr.fixture_cer(gt, doc.pages[0])}, seconds


def step_fixtures(args, build) -> int:
    started, work = ar.now(), Path(args.work)
    items = _fixture_files()
    session = build(work)
    with _vram(work, "vram-fixtures.csv", args.nvidia_smi):
        done = ar.parallel_map(lambda item: _read_fixture(session, item), items, args.jobs)
    results = {item[0]: rec for item, (rec, _) in zip(items, done)}
    peak = _peak_fields(work / "vram-fixtures.csv")
    write_step(Path(args.out), "fixtures", started, {
        "keys": session.keys, "model": session.facts, "fixtures": results,
        "pages": ar.pages_block([s for _, s in done]),
        **peak})
    for name, rec in results.items():
        say(f"{name}: cer {ar.pct(rec['cer'])}")
    if "vram_error" in peak:
        raise ModelRunError(peak["vram_error"])
    return 0
