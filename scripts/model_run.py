"""Runs the laptop measurement of a vision model as this tool's line reader, one command per
step: contract, order, drift-sample, drift, calibrate, local-calibrate, measure, grade,
fixtures and report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import model_run_common as common
import model_run_grading as grading
import model_run_local as local
import model_run_report as report
import model_run_steps as steps
from heldout import HeldoutError
from model_run_reads import model_keys, model_reader_factory
from model_run_server import NO_KEY, Capture, ModelRunError, server_facts, warm_up
from script_errors import describe_os_error

from folio_ocr import model_http
from folio_ocr.config import find_tesseract
from folio_ocr.errors import FolioError
from folio_ocr.model_reader import ModelConfig, ModelReader
from folio_ocr.tesseract import decided_version

ROOT = Path(__file__).resolve().parents[1]
NO_TESSERACT = "Tesseract was not found; set FOLIO_TESSERACT_PATH"
Builder = Callable[..., "Session"]


@dataclass(frozen=True)
class Session:
    """Everything a step that reads needs: settings, the model reader, its calibration keys and
    the factory of readers that keep the model's replies."""

    settings: Any
    config: ModelConfig
    tesseract: Path
    tesseract_version: str
    model: ModelReader
    capture: Capture
    keys: dict[str, str]
    facts: dict
    reader_factory: Callable


def build_session(work: Path, *, warm: bool = True) -> Session:
    """Connect to the configured model server; `warm` loads the model and runs its self-test."""
    settings, config = common.model_config()
    tess = find_tesseract(settings)
    if tess is None:
        raise ModelRunError(NO_TESSERACT)
    capture = Capture()
    model = ModelReader(config, post=capture.post)
    if warm:
        warm_up(config)
        test = model.self_test()
        if not test.passed:
            raise ModelRunError(f"the model self-test failed: {test.detail}")
        model.fit()
    facts = server_facts(config, 30.0)
    info = model.server()
    if info is None:
        raise ModelRunError(NO_KEY)
    version = decided_version(tess)
    keys = model_keys(info, config.name, version)
    factory = model_reader_factory(model, capture, keys, Path(work))
    return Session(settings, config, tess, version, model, capture, keys, facts,
                   factory)


COMMANDS: dict[str, Callable[..., int]] = {
    "contract": steps.step_contract,
    "order": steps.step_order,
    "drift-sample": steps.step_drift_sample,
    "drift": steps.step_drift,
    "calibrate": grading.step_calibrate,
    "local-calibrate": local.step_local_calibrate,
    "measure": grading.step_measure,
    "grade": grading.step_grade,
    "fixtures": grading.step_fixtures,
    "report": report.step_report,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, *required: str, **optional: dict) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--jobs", type=int, default=1)
        for flag in required:
            p.add_argument(f"--{flag}", required=True)
        for flag, options in optional.items():
            p.add_argument(f"--{flag.replace('_', '-')}", **options)
        return p

    smi = {"nvidia_smi": {"default": None}}
    command("contract", "out", expect_version={"default": None}, expect_digest={"default": None})
    command("order", "dev", "out")
    command("drift-sample", "dev", "out", order={"default": None},
            lines={"type": int, "default": 40})
    command("drift", "sample", "out", repeats={"type": int, "default": 3},
            timeout_s={"type": float, "default": 180.0})
    command("calibrate", "dev", "order", "drift", "heldout", "heldout-sha256", "out", "record",
            results_dir={"default": str(ROOT / "eval" / "results")}, **smi)
    command("local-calibrate", "dev", "drift", "tau-file", "out")
    command("measure", "dev", "record", "out")
    command("grade", "heldout", "heldout-sha256", "record", "results-dir", "out", **smi)
    command("fixtures", "out", **smi)
    command("report", "contract", "order", "drift", "record", "measure", "grade-record",
            "fixtures", "out", label={"default": None})
    return parser


def _fail(message: str) -> int:
    print(common.ascii_only(f"error: {message}"), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None, session_builder: Builder = build_session) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.jobs < 1:
            raise ValueError("--jobs must be at least 1")
        return COMMANDS[args.command](args, session_builder)
    except (ModelRunError, HeldoutError, FolioError, ValueError) as exc:
        return _fail(str(exc))
    except model_http.ModelCallError as exc:
        return _fail(f"the model server did not answer: {exc}")
    except OSError as exc:
        return _fail(describe_os_error(exc, args.work))
    except Exception as exc:  # noqa: BLE001 - the user sees one line, never a traceback
        return _fail(f"unexpected failure ({type(exc).__name__})")


if __name__ == "__main__":
    sys.exit(main())
