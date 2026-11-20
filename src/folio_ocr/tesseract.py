"""Read one line crop with the Tesseract 5 command-line program."""

from __future__ import annotations

import errno
import io
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from folio_ocr import assets
from folio_ocr.document import Box
from folio_ocr.errors import FolioError
from folio_ocr.reading import Reading

LANGUAGES = "ara+eng"
ARGUMENTS = ("--psm", "13", "--oem", "1")  # one raw text line, the LSTM engine
PADDING_PX = 6
OSD_ARGUMENTS = ("--psm", "0")  # orientation and script detection only

_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
)
_NOT_TSV = "Tesseract returned output that is not TSV"
_ROTATE = re.compile(r"^Rotate: (0|90|180|270)$", re.MULTILINE)
_OSD_DEGREES = re.compile(r"^Orientation in degrees: (0|90|180|270)$", re.MULTILINE)
_OSD_CONFIDENCE = re.compile(r"^Orientation confidence: (\S+)$", re.MULTILINE)
_VERSION = re.compile(r"tesseract\s+v?([0-9][^\s]*)", re.IGNORECASE)


class TesseractFailure(FolioError):
    """Tesseract could not be started, ran too long, failed, or answered outside its format."""


class TesseractTimeout(TesseractFailure):
    """Tesseract did not finish within its time limit."""


def _require_absolute(binary: Path) -> None:
    if not binary.is_absolute():
        raise TesseractFailure("the Tesseract path must be absolute")


def command(binary: Path, tessdata_dir: Path) -> list[str]:
    _require_absolute(binary)
    return [
        str(binary),
        "stdin",
        "stdout",
        "--tessdata-dir",
        str(tessdata_dir),
        "-l",
        LANGUAGES,
        *ARGUMENTS,
        "-c",
        "tessedit_create_tsv=1",
    ]


def osd_command(binary: Path, tessdata_dir: Path) -> list[str]:
    _require_absolute(binary)
    return [str(binary), "stdin", "stdout", "--tessdata-dir", str(tessdata_dir), *OSD_ARGUMENTS]


@dataclass(frozen=True)
class Osd:
    quarter: int  # degrees counterclockwise (0/90/180/270) that turn the page upright
    confidence: float  # Tesseract's orientation confidence


def page_png(image: Image.Image) -> bytes:
    """The whole page as a grey PNG."""
    out = io.BytesIO()
    image.convert("L").save(out, format="PNG")
    return out.getvalue()


def line_png(image: Image.Image, box: Box, padding: int = PADDING_PX) -> bytes:
    """The line's pixels, grown by `padding`, clamped to the image, as a grey PNG."""
    x0 = max(0, math.floor(box[0] - padding))
    y0 = max(0, math.floor(box[1] - padding))
    x1 = min(image.width, math.ceil(box[2] + padding))
    y1 = min(image.height, math.ceil(box[3] + padding))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the line box is empty")
    out = io.BytesIO()
    image.crop((x0, y0, x1, y1)).convert("L").save(out, format="PNG")
    return out.getvalue()


def _start_failure(exc: OSError) -> TesseractFailure:
    if exc.errno == errno.ENOENT:
        return TesseractFailure("Tesseract was not found at the configured path")
    if exc.errno in (errno.EACCES, errno.EPERM) or isinstance(exc, PermissionError):
        return TesseractFailure("Tesseract cannot be started: permission denied")
    if exc.errno == errno.ENOEXEC or getattr(exc, "winerror", None) in (193, 216):
        return TesseractFailure(
            "the configured Tesseract file is not a program this system can run"
        )
    return TesseractFailure("Tesseract could not be started")


def _execute(
    argv: list[str],
    input_: bytes | None,
    timeout_s: float,
    runner: Callable[..., Any],
    clock,
    task: str = "finish a line",
) -> subprocess.CompletedProcess:
    too_slow = TesseractTimeout(f"Tesseract did not {task} within {timeout_s:g} s")
    started = clock()
    try:
        done = runner(argv, input=input_, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        raise too_slow from None
    except OSError as exc:
        raise _start_failure(exc) from None
    if clock() - started >= timeout_s:
        raise too_slow
    return done


def run(
    binary: Path,
    tessdata_dir: Path,
    png: bytes,
    timeout_s: float,
    *,
    runner: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
) -> bytes:
    """Tesseract's TSV output for one PNG; raises TesseractFailure."""
    done = _execute(command(binary, tessdata_dir), png, timeout_s, runner, clock)
    if done.returncode != 0:
        raise TesseractFailure(f"Tesseract failed (exit code {done.returncode})")
    return done.stdout


_BAD_OSD = "Tesseract returned a page orientation report this tool cannot read"
# Tesseract's answer for a page with too few characters: exit 1, nothing on stdout.
_NO_ANSWER_LINE = b"Too few characters. Skipping this page"


def _quarter_report(text: str) -> tuple[int, float]:
    """(Rotate, confidence) of an OSD report whose two orientation lines agree."""
    rotate, shown = _ROTATE.search(text), _OSD_CONFIDENCE.search(text)
    degrees = _OSD_DEGREES.search(text)
    if rotate is None or shown is None or degrees is None:
        raise TesseractFailure(_BAD_OSD)
    try:
        confidence = float(shown.group(1))
    except ValueError:
        raise TesseractFailure(_BAD_OSD) from None
    if not math.isfinite(confidence) or confidence < 0:
        raise TesseractFailure(_BAD_OSD)
    if (int(rotate.group(1)) + int(degrees.group(1))) % 360 != 0:
        raise TesseractFailure(_BAD_OSD)
    return int(rotate.group(1)), confidence


def parse_osd(stdout: bytes) -> Osd:
    """The page turn and confidence in Tesseract's OSD report. Raises TesseractFailure when the
    report is unreadable or its Rotate and Orientation lines do not sum to 0 mod 360. Text lines
    end with LF or CRLF."""
    try:
        text = stdout.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        raise TesseractFailure(_BAD_OSD) from None
    rotate, confidence = _quarter_report(text)
    return Osd((360 - rotate) % 360, confidence)


def _is_no_answer(done: subprocess.CompletedProcess) -> bool:
    lines = done.stderr.replace(b"\r\n", b"\n").split(b"\n")
    return done.returncode == 1 and done.stdout == b"" and _NO_ANSWER_LINE in lines


def osd(
    binary: Path,
    tessdata_dir: Path,
    png: bytes,
    timeout_s: float,
    *,
    runner: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
) -> Osd | None:
    """Tesseract's orientation answer for one page PNG; None only when Tesseract says the page
    has too few characters. Every failure raises TesseractFailure."""
    done = _execute(
        osd_command(binary, tessdata_dir),
        png,
        timeout_s,
        runner,
        clock,
        "report the page orientation",
    )
    if _is_no_answer(done):
        return None
    if done.returncode != 0:
        raise TesseractFailure(f"Tesseract failed (exit code {done.returncode})")
    return parse_osd(done.stdout)


def _row(line: str) -> tuple[int, float, str]:
    fields = line.split("\t", 11)
    if len(fields) != 12:
        raise TesseractFailure(_NOT_TSV)
    numbers = fields[:10]
    if not all(f.isascii() and f.isdigit() for f in numbers[:6]):
        raise TesseractFailure(_NOT_TSV)
    try:
        conf = float(fields[10])
    except ValueError:
        raise TesseractFailure(_NOT_TSV) from None
    return int(fields[0]), conf, fields[11]


def parse_tsv(stdout: bytes) -> Reading:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise TesseractFailure("Tesseract returned text that is not UTF-8") from None
    # Rows end with LF (CRLF on Windows); split on those only, so a word holding another line
    # break character (form feed, U+2028, ...) stays in its row.
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if not lines or lines[0] != _HEADER:
        raise TesseractFailure(_NOT_TSV)
    words = [_row(line) for line in lines[1:] if line]
    scores = tuple(conf / 100 for level, conf, _ in words if level == 5)
    spoken = [t for level, _, t in words if level == 5 and t.strip()]
    return Reading(" ".join(spoken), scores)


def read_line(
    binary: Path,
    tessdata_dir: Path,
    png: bytes,
    timeout_s: float,
    *,
    runner: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
) -> Reading:
    """Run and parse; a failure comes back as a Reading carrying the error."""
    try:
        return parse_tsv(run(binary, tessdata_dir, png, timeout_s, runner=runner, clock=clock))
    except TesseractFailure as failure:
        return Reading("", (), error=str(failure))


def tesseract_version(
    binary: Path, timeout_s: float = 10.0, *, runner: Callable[..., Any] = subprocess.run
) -> str:
    _require_absolute(binary)
    done = _execute(
        [str(binary), "--version"], None, timeout_s, runner, time.perf_counter, "report its version"
    )
    shown = (done.stdout + done.stderr).decode("utf-8", errors="replace")
    found = _VERSION.search(shown)
    if found is None:
        raise TesseractFailure("Tesseract did not report its version")
    return found.group(1)


# What each program file answered in this process: (version, None), or (None, the line saying why
# it gave none). Keyed by (path, size, mtime_ns), so a replaced program is asked again.
_DECIDED: dict[tuple[str, int, int], tuple[str | None, str | None]] = {}
# ponytail: one lock for every program file; per-file locks if several are ever checked at once
_DECIDED_LOCK = threading.Lock()


def _version_asked_twice_if_slow(binary: Path) -> str:
    try:
        return tesseract_version(binary)
    except TesseractTimeout:
        return tesseract_version(binary)


def _decide(binary: Path) -> tuple[str | None, str | None]:
    try:
        return _version_asked_twice_if_slow(binary), None
    except TesseractFailure as failure:
        return None, str(failure)


def decided_version(binary: Path) -> str:
    """The version of the Tesseract at `binary`, decided once in this process: a check that runs
    out of time is asked once more, then its answer, or its failure, holds for every later caller
    until the program file changes. Raises TesseractFailure with the line saying why."""
    found = shutil.which(str(binary)) or str(binary)  # the file run (Windows may add ".exe")
    try:
        info = Path(found).stat()
    except OSError:
        return _version_asked_twice_if_slow(binary)  # no program file: no check can run it
    key = (found, info.st_size, info.st_mtime_ns)
    with _DECIDED_LOCK:
        if key not in _DECIDED:
            _DECIDED[key] = _decide(binary)
        version, failure = _DECIDED[key]
    if failure is not None:
        raise TesseractFailure(failure)
    return version


def tessdata_problem(weights_dir: Path) -> str | None:
    """None when the language data is present and intact, else one line saying what to do."""
    problem = assets.files_problem((*assets.TESSDATA, assets.TESSERACT_OSD), weights_dir)
    return None if problem is None else _DATA_PROBLEM[problem]


_DATA_PROBLEM = {
    "missing": "Tesseract language data is missing; run folio setup",
    "cannot read": "Tesseract language data cannot be read; check the access rights of the "
    "assets folder",
    "damaged": "Tesseract language data is damaged; run folio setup",
}
