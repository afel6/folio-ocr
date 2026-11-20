"""The local-calibrate command: a company calibrates its own model on its own development set and
the thresholds go into a local tau file (schema folio-taus/1). The package tau table is never
read or written here."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import model_run_common as common
import model_run_reads as mr
import run_accuracy
from model_run_common import check_keys, load_step, say
from model_run_server import ModelRunError

from folio_ocr import model_taus
from folio_ocr.errors import ConfigError

DEV_SCHEMA = "folio-dev/1"
NOT_DEV = ("the --dev folder holds no development manifest (schema folio-dev/1); "
           "a held-out manifest is never calibration input")
TAU_REFUSED = "the tau file exists but is refused; fix or move it first"


def _check_dev_manifest(dev: Path) -> str:
    """The sha256 of the development manifest, which must be one; other OS errors propagate."""
    try:
        data = (dev / "manifest.json").read_bytes()
        manifest = json.loads(data)
    except (FileNotFoundError, UnicodeDecodeError, ValueError, RecursionError):
        raise ModelRunError(NOT_DEV) from None
    if not isinstance(manifest, dict) or manifest.get("schema") != DEV_SCHEMA:
        raise ModelRunError(NOT_DEV)
    return common.sha256_bytes(data)


def _existing_entries(path: Path) -> dict:
    """The raw entries of the tau file at `path`; {} when there is none, an error when refused."""
    if not path.exists():
        return {}
    try:
        data = path.read_bytes()
        model_taus.parse(data)
    except (OSError, ConfigError):
        raise ModelRunError(TAU_REFUSED) from None
    return json.loads(data)["entries"]


def _entry(session, cal: dict, cell: dict, tau: float, tolerance: float) -> dict:
    replies = session.model.fingerprint()
    return {
        "tau": tau, "method": "logprobs", "bad_lines": cell["bad_lines"],
        "documents": cell["documents_with_bad_lines"],
        "dev_bound": cell["missed_error_rate"]["bound_95"],
        "dev_manifest_sha256": cal["dev_manifest_sha256"],
        "self_test": [{"line": r.line, "text": r.text, "score": r.score} for r in replies],
        "drift_tolerance": tolerance,
    }


def _read_calibration(path: Path) -> dict:
    cal, _ = common.read_json_file(path, "calibration")
    if not isinstance(cal.get("cells"), list):
        raise ModelRunError(f"the calibration file is missing or damaged: {Path(path).name}")
    return cal


def _new_entries(session, cal: dict, tolerance: float) -> dict[str, tuple[str, dict]]:
    """Cell -> (key, entry) of every calibrated cell with a tau; says why any other gets none."""
    taus = cal.get("taus") or {}
    found: dict[str, tuple[str, dict]] = {}
    for cell in cal["cells"]:
        name = cell.get("cell")
        key = session.keys.get(name)
        if cell.get("status") == "calibrated" and key is not None and taus.get(key) is not None:
            found[name] = (key, _entry(session, cal, cell, taus[key], tolerance))
        else:
            say(f"{name}: {cell.get('reason') or 'no tau'}; no entry written")
    return found


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".taus-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _merged_bytes(existing: dict, new: dict[str, dict]) -> bytes:
    document = {"schema": model_taus.SCHEMA, "entries": {**existing, **new}}
    try:
        text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    except ValueError:
        raise ModelRunError("the new tau file would be refused: a number is not finite") from None
    data = (text + "\n").encode("utf-8")
    try:
        model_taus.parse(data)
    except ConfigError as exc:
        raise ModelRunError(f"the new tau file would be refused: {exc}") from None
    return data


def step_local_calibrate(args, build) -> int:
    dev, tau_file = Path(args.dev), Path(args.tau_file)
    dev_sha = _check_dev_manifest(dev)
    drift = load_step(Path(args.drift), "drift")
    _existing_entries(tau_file)  # refuse a bad tau file before any model is read
    session = build(Path(args.work))
    check_keys(drift, session.keys)
    out = Path(args.out)
    rc = run_accuracy.main(
        ["calibrate", "--dev", str(dev), "--work", str(args.work), "--out", str(out),
         "--jobs", str(args.jobs)],
        reader_factory=session.reader_factory, plan=mr.model_plan(session.keys))
    if rc != 0:
        return rc
    cal = _read_calibration(out)
    if cal.get("dev_manifest_sha256") != dev_sha:
        raise ModelRunError("the dev manifest changed during the calibration; nothing was written")
    new = _new_entries(session, cal, drift["tolerance"])
    if not new:
        say("STOP: no model cell was calibrated; the tau file is unchanged")
        return 1
    data = _merged_bytes(_existing_entries(tau_file), dict(new.values()))
    _write_atomically(tau_file, data)
    for name, (_, entry) in new.items():
        say(f"{name}: tau {entry['tau']} written")
    say(f"tau file: {tau_file} sha256 {common.sha256_bytes(data)}")
    return 0
