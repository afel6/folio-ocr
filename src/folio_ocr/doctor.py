"""Checks that the tool can run on this machine."""

import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from folio_ocr import ingest, ocr_setup, pdf_client
from folio_ocr.assets import MANIFEST, Asset, verify
from folio_ocr.config import Settings, find_tesseract, is_loopback
from folio_ocr.errors import PdfReaderError

_TESSERACT_TIMEOUT_S = 10
_SETUP_REASONS = {"missing", "wrong size", "sha256 mismatch"}
_VERSION = re.compile(r"tesseract\s+v?([0-9][^\s]*)", re.IGNORECASE)


@dataclass(frozen=True)
class Check:
    name: str
    status: Literal["ok", "fail", "info"]
    detail: str


def _python(version_info: tuple[int, ...]) -> Check:
    text = ".".join(str(n) for n in version_info)
    if version_info >= (3, 12):
        return Check("python", "ok", text)
    return Check("python", "fail", f"{text}; folio-ocr needs Python 3.12 or newer")


def _pdfium() -> Check:
    try:
        import pypdfium2

        return Check("pdfium", "ok", str(pypdfium2.PDFIUM_INFO.version))
    except (ImportError, OSError):
        return Check("pdfium", "fail", "cannot load pdfium")


def _pdf_reader() -> Check:
    """Whether a file reader process starts with its memory and time limits."""
    try:
        pdf_client.self_check()
    except PdfReaderError as err:
        return Check("pdf reader", "fail", f"{err.reason}; PDFs and images cannot be read")
    limits = f"{ingest.PDF_MEMORY_MB} MB and {ingest.PDF_SECONDS} s per file"
    return Check("pdf reader", "ok", f"starts with its limits ({limits})")


def _assets(settings: Settings, assets: Sequence[Asset]) -> list[Check]:
    result = []
    for status in verify(assets, settings.weights_dir):
        name = f"asset {status.asset.name}"
        if status.ok:
            result.append(Check(name, "ok", "verified (sha256)"))
        elif status.reason in _SETUP_REASONS:
            result.append(Check(name, "fail", f"{status.reason}; run folio setup"))
        else:
            advice = "check the access rights of the assets folder"
            result.append(Check(name, "fail", f"{status.reason}; {advice}"))
    return result


def _once(
    runner: Callable[..., subprocess.CompletedProcess],
) -> Callable[..., subprocess.CompletedProcess]:
    """`runner`, except that each command runs at most once per doctor run: later calls get the
    same answer, or the same error, so the tesseract and reader checks share one `--version`."""
    seen: dict[tuple, subprocess.CompletedProcess | subprocess.TimeoutExpired | OSError] = {}

    def run_once(argv, *args, **kwargs):
        key = tuple(argv)
        if key not in seen:
            try:
                seen[key] = runner(argv, *args, **kwargs)
            except (subprocess.TimeoutExpired, OSError) as error:
                seen[key] = error
        outcome = seen[key]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return run_once


def _tesseract(path: Path | None, runner: Callable[..., subprocess.CompletedProcess]) -> Check:
    if path is None:
        return Check(
            "tesseract",
            "info",
            "not found (optional: install Tesseract or set FOLIO_TESSERACT_PATH "
            "to read Arabic without a model server)",
        )
    return Check("tesseract", *_tesseract_status(path, runner))


def _tesseract_status(
    path: Path, runner: Callable[..., subprocess.CompletedProcess]
) -> tuple[Literal["ok", "fail"], str]:
    try:
        done = runner(
            [str(path), "--version"],
            capture_output=True,
            timeout=_TESSERACT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "fail", f"did not answer within {_TESSERACT_TIMEOUT_S} s: {path}"
    except OSError:
        return "fail", f"cannot run {path}"
    if done.returncode != 0:
        return "fail", f"exited with code {done.returncode}: {path}"
    text = (done.stdout + done.stderr).decode("utf-8", errors="replace")
    found = _VERSION.search(text)
    if found is None:
        return "fail", f"unrecognised version output: {path}"
    return "ok", f"{found.group(1)} at {path}"


def _model_server(settings: Settings) -> Check:
    if settings.model_url is None:
        return Check("model server", "info", "not configured (set FOLIO_MODEL_URL to use one)")
    host = settings.model_host or ""
    name = settings.model_name or "(FOLIO_MODEL_NAME not set)"
    detail = f"configured: model {name} at host {host}"
    if not is_loopback(host):
        detail += " (leaves this machine)"
    return Check("model server", "info", detail)


_READER_DETAIL = {
    "missing": "missing",
    "bad_data": "bad_data (its files or program cannot be used; see the lines above)",
    "uncalibrated": "uncalibrated (no frozen confidence threshold yet: every line it reads "
    "is flagged low_confidence)",
    "ok": "ok",
}


_ONNXTR_STATUS = {"missing": "fail", "bad_data": "fail", "uncalibrated": "info", "ok": "ok"}
_TESSERACT_STATUS = {"missing": "info", "bad_data": "fail", "uncalibrated": "info", "ok": "ok"}


def _readers(
    settings: Settings,
    binary: Path | None,
    runner: Callable[..., subprocess.CompletedProcess],
) -> list[Check]:
    """One line per OCR reader: missing | bad_data | uncalibrated | ok. OnnxTR is required;
    Tesseract is optional, so its missing state is information."""
    onnxtr = ocr_setup.onnxtr_status(settings.weights_dir, binary, runner=runner)
    tess = ocr_setup.tesseract_status(settings.weights_dir, binary, runner=runner)
    return [
        Check("reader onnxtr", _ONNXTR_STATUS[onnxtr], _READER_DETAIL[onnxtr]),
        Check("reader tesseract", _TESSERACT_STATUS[tess], _READER_DETAIL[tess]),
    ]


def checks(
    settings: Settings,
    *,
    version_info: tuple[int, ...] = tuple(sys.version_info[:3]),
    environ: Mapping[str, str] = os.environ,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    assets: Sequence[Asset] = MANIFEST,
    model_checks: Callable[[Settings], list[Check]] | None = None,
    pdf_reader: Callable[[], Check] = _pdf_reader,
) -> list[Check]:
    """Run every check and return the results in display order. `model_checks` asks the
    configured model server (self-test, calibration); its lines follow the model server line."""
    runner = _once(runner)
    binary = find_tesseract(settings, environ)
    return [
        _python(version_info),
        _pdfium(),
        pdf_reader(),
        Check("assets folder", "info", str(settings.weights_dir)),
        *_assets(settings, assets),
        _tesseract(binary, runner),
        *_readers(settings, binary, runner),
        _model_server(settings),
        *([] if model_checks is None else model_checks(settings)),
    ]


def run(settings: Settings, write: Callable[[str], None], **kwargs) -> int:
    """Write one line per check; 1 when any check fails, else 0."""
    results = checks(settings, **kwargs)
    for c in results:
        write(f"{c.status:<5}{c.name}: {c.detail}")
    return 1 if any(c.status == "fail" for c in results) else 0
