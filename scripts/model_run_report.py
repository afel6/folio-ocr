"""The final step: join every step file into one results file."""

from __future__ import annotations

from pathlib import Path

import accuracy_reads as ar
import model_run_common as common
from model_run_common import check_keys, load_step, say

from folio_ocr import calibration, model_request

SCHEMA = "folio-accuracy/1"


def _inputs(files: dict[str, Path]) -> dict:
    return {name: {"path": path.name, "sha256": common.file_sha256(path)}
            for name, path in files.items()}


def _same_keys(records: dict[str, dict]) -> dict[str, str]:
    """The keys every step recorded, which must all be equal."""
    keys = records["record"]["keys"]
    for name, rec in records.items():
        if name != "contract":
            check_keys(rec, keys)
    return keys


def _model(facts: dict, keys: dict) -> dict:
    return {"name": facts["model"], "host": facts["host"], "kind": facts["kind"],
            "setup_keys": keys}


def _peak(vram: dict) -> int | float | None:
    """The largest recorded peak, or None when no step recorded one."""
    seen = [v for v in vram.values() if v is not None]
    return max(seen) if seen else None


def _vram_errors(records: dict[str, dict]) -> dict[str, str]:
    """The VRAM log error of every step that carries one, by step name."""
    return {rec["step"]: rec["vram_error"] for rec in records.values() if "vram_error" in rec}


def _body(r: dict[str, dict], cal: dict, grade: dict, keys: dict) -> dict:
    facts = r["record"]["model"]
    vram = {"calibrate": r["record"]["peak_vram_mib"], "grade": r["grade_record"]["peak_vram_mib"],
            "fixtures": r["fixtures"]["peak_vram_mib"]}
    drift = r["drift"]
    return {
        "host": grade["host"], "versions": grade["versions"],
        "ollama_version": facts["version"], "model_digest": facts["digest"],
        "prompt_sha256": model_request.PROMPT_SHA256, "model": _model(facts, keys),
        "cells": grade["cells"], "calibration_cells": cal["cells"], "taus": r["record"]["taus"],
        "tau_table_sha256": calibration.table_sha256(calibration.TAUS),
        "tau_table_sha256_at_calibration": r["record"]["tau_table_sha256_at_calibration"],
        # null: this summary is not a grade attempt; the graded manifest is named apart
        "heldout_manifest_sha256": None,
        "graded_heldout_manifest_sha256": grade["heldout_manifest_sha256"],
        "dev_manifest_sha256": cal["dev_manifest_sha256"],
        "cer_per_fixture": {f: v["cer"] for f, v in r["fixtures"]["fixtures"].items()},
        "pages": grade["pages"],
        "pages_by_step": {"calibrate": cal["pages"], "grade": grade["pages"],
                          "fixtures": r["fixtures"]["pages"]},
        "peak_vram_mib": _peak(vram), "vram_by_step": vram, "vram_errors": _vram_errors(r),
        "contract": {"checks": r["contract"]["checks"], "passed": r["contract"]["passed"]},
        "order": {k: r["order"][k] for k in
                  ("classes", "verdict", "order_rule", "documents", "stopped_by")},
        "drift": {"tolerance": drift["tolerance"], "max_spread_by_cell": drift["max_spread_by_cell"],
                  "text_changes": drift["text_changes"], "rounds": drift["rounds"],
                  "repeats": drift["repeats"], "lines": len(drift["lines"])},
        **{k: r["measure"][k] for k in ("pins", "measures", "tesseract_compare", "aggregation",
                                        "order_after_calibration", "dev_lines")},
    }


def step_report(args, build) -> int:
    paths = {"contract": args.contract, "order": args.order, "drift": args.drift,
             "record": args.record, "measure": args.measure, "grade_record": args.grade_record,
             "fixtures": args.fixtures}
    steps = {"contract": "contract", "order": "order", "drift": "drift", "record": "calibrate",
             "measure": "measure", "grade_record": "grade", "fixtures": "fixtures"}
    records = {name: load_step(Path(path), steps[name]) for name, path in paths.items()}
    keys = _same_keys(records)
    cal_path, cal_sha = Path(records["record"]["calibration"]["path"]), \
        records["record"]["calibration"]["sha256"]
    grade_path, grade_sha = Path(records["grade_record"]["grade"]["path"]), \
        records["grade_record"]["grade"]["sha256"]
    cal = common.read_json_checked(cal_path, cal_sha, "calibration")
    grade = common.read_json_checked(grade_path, grade_sha, "grade")
    files = {"contract": Path(args.contract), "order": Path(args.order),
             "drift": Path(args.drift), "calibrate": Path(args.record),
             "measure": Path(args.measure), "grade": Path(args.grade_record),
             "fixtures": Path(args.fixtures), "calibration": cal_path, "grade_results": grade_path}
    out = Path(args.out)
    result = {
        "schema": SCHEMA, "mode": "model-run", "status": "finished",
        "started_at": records["contract"]["started_at"], "finished_at": ar.now(),
        **_body(records, cal, grade, keys), "inputs": _inputs(files), "label": args.label,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    ar.write_json(out, result)
    say(f"results: {out}")
    say(f"sha256: {common.file_sha256(out)}")
    return 0

