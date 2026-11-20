"""The `folio` command line: read documents, fetch the assets, check this machine.

Output and messages go through `_emit`, which writes UTF-8 bytes so a pipe on Windows keeps
the text and the line ends as they are (argparse writes `--help` itself). Every error is one
line `error: <message>`, except a page that could not be read, which gets one line
`<path>: page <n>: <reason>`; a reader that closes the pipe early ends the run without one.
"""

from __future__ import annotations

import argparse
import errno
import os
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from folio_ocr import assets, doctor, model_health, pipeline
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.document import Document
from folio_ocr.errors import FolioError, SetupNeededError, WeightsError
from folio_ocr.export import review_line_count, to_docx, to_json, to_txt
from folio_ocr.flags import NEEDS_MODEL_SERVER
from folio_ocr.model_reader import ModelConfig, ModelReader

HINT = "hint: use --lang en for English-only pages, or install Tesseract, or set FOLIO_MODEL_URL"
SETUP_HINT = "hint: run folio setup to download the model files"
DOCTOR_HINT = "hint: run folio doctor to check the readers"
WEIGHTS_DIR_IS_A_FILE = "FOLIO_WEIGHTS_DIR points at a file, not a folder: {}"
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)}
_HIDDEN = {"Cc", "Cf", "Cs", "Zl", "Zp"}
_OS_MESSAGES = {
    errno.ENOENT: "not found",
    errno.EISDIR: "is a folder, not a file",
    errno.EACCES: "permission denied",
    errno.EPERM: "permission denied",
    errno.ENOTDIR: "part of the path is not a folder",
    errno.EEXIST: "not a folder",
    errno.ENOSPC: "disk full while writing",
    errno.ENAMETOOLONG: "name too long",
}
if hasattr(errno, "EDQUOT"):
    _OS_MESSAGES[errno.EDQUOT] = "disk full while writing"


class _UsageError(Exception):
    """The command line was wrong; the message is one line for the user."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


class _StdoutError(Exception):
    """Standard output could not be written: a closed pipe or a full disk."""

    def __init__(self, cause: OSError) -> None:
        super().__init__(cause.errno)
        self.cause = cause


def _emit(stream, text: str) -> None:
    """Write `text` to `stream` as UTF-8 bytes when it has a byte buffer."""
    try:
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            stream.write(text)
            stream.flush()
            return
        stream.flush()
        buffer.write(text.encode("utf-8"))
        buffer.flush()
    except OSError as exc:
        if stream is sys.stdout:
            raise _StdoutError(exc) from exc
        raise


def _shown(text: str) -> str:
    """`text` with every control, format, surrogate and line-separator character as '?'."""
    return "".join("?" if unicodedata.category(c) in _HIDDEN else c for c in text)


def describe_os_error(exc: OSError, shown: str) -> str:
    """One friendly line for a file system error, chosen by its error number only."""
    return f"{_OS_MESSAGES.get(exc.errno, 'cannot read or write')}: {shown}"


def _fail(message: str) -> int:
    _emit(sys.stderr, f"error: {message}\n")
    return 1


def _build_parser() -> _Parser:
    parser = _Parser(prog="folio", description="Read printed English and Arabic documents.")
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read", help="read documents into text")
    read.add_argument("files", nargs="+", metavar="FILE")
    read.add_argument("--lang", choices=["en", "ar", "en+ar"], default="en+ar")
    read.add_argument("--out", metavar="DIR")
    read.add_argument("--format", choices=["txt", "docx", "json"], default="txt")
    commands.add_parser("setup", help="download the model files")
    commands.add_parser("doctor", help="check that the tool can run here")
    serve = commands.add_parser("serve", help="start the local web service")
    serve.add_argument("--allow-remote", action="store_true")
    return parser


@dataclass
class _Run:
    """What one `folio read` has done so far."""

    settings: Settings
    tesseract: Path | None  # the Tesseract program found once for the whole run
    lang: str
    fmt: str
    out: Path | None
    inputs: list[Path]
    many: bool
    written: list[Path] = field(default_factory=list)
    docs: list[Document] = field(default_factory=list)
    failed: bool = False
    needs_setup: bool = False
    model: ModelReader | None = None  # one per run: the self-test runs once

    def fail(self, message: str) -> None:
        self.failed = True
        _fail(message)


def _stem(doc: Document) -> str:
    stem = Path(doc.source.filename).stem or "document"
    return f"_{stem}" if stem.split(".")[0].upper() in _RESERVED else stem


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _target(run: _Run, doc: Document) -> Path:
    assert run.out is not None
    stem, n = _stem(doc), 1
    while True:
        name = f"{stem}.{run.fmt}" if n == 1 else f"{stem}-{n}.{run.fmt}"
        candidate = run.out / name
        if not any(_same_file(candidate, p) for p in (*run.inputs, *run.written)):
            return candidate
        n += 1


def _render(doc: Document, fmt: str) -> bytes:
    if fmt == "docx":
        return to_docx(doc)
    if fmt == "json":
        return (to_json(doc) + "\n").encode("utf-8")
    return to_txt(doc).encode("utf-8")


def _write_atomic(target: Path, data: bytes) -> None:
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False
        ) as handle:
            temp = Path(handle.name)
            handle.write(data)
        os.replace(temp, target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _save(run: _Run, doc: Document) -> Path | None:
    target = _target(run, doc)
    try:
        _write_atomic(target, _render(doc, run.fmt))
    except OSError as exc:
        run.fail(describe_os_error(exc, _shown(str(target))))
        return None
    run.written.append(target)
    return target


def _summary(arg: str, doc: Document, saved: Path | None) -> str:
    found = sum(len(p.lines) for p in doc.pages)
    read = sum(x.reader != "none" for p in doc.pages for x in p.lines)
    unread = sum(p.status != "read" for p in doc.pages)
    text = f"{_shown(arg)}: {found} lines found, {read} read, "
    text += f"{review_line_count(doc)} need review, {unread} unread pages"
    return text + (f" (saved to {_shown(str(saved))})" if saved else "") + "\n"


def _unread_areas(count: int) -> str:
    if count == 1:
        return "1 area was not read (see --format json for their boxes)"
    return f"{count} areas were not read (see --format json for their boxes)"


def _page_notes(arg: str, doc: Document) -> str:
    """Per page, in page order: a line naming why its lines could not be read, then a line
    saying how many of its areas were not read; a page with neither gets no line."""
    text = ""
    for p in doc.pages:
        if p.error:
            text += f"{_shown(arg)}: page {p.index}: {_shown(p.error)}\n"
        if p.unread_regions:
            text += f"{_shown(arg)}: page {p.index}: {_unread_areas(len(p.unread_regions))}\n"
    return text


def _has_page_error(doc: Document) -> bool:
    return any(p.error for p in doc.pages)


def _print_doc(run: _Run, arg: str, doc: Document) -> None:
    if run.fmt == "json":
        _emit(sys.stdout, to_json(doc, indent=None) + "\n")
        return
    if run.many:
        _emit(sys.stdout, f"==> {_shown(arg)} <==\n")
    _emit(sys.stdout, to_txt(doc))


def _is_not_a_folder(path: Path) -> bool:
    """True when `path` exists and is not a folder; False when it is a folder or cannot be
    looked up (missing, no access), whose own messages then apply."""
    try:
        return not stat.S_ISDIR(os.stat(path).st_mode)
    except OSError:
        return False


def _read_one(run: _Run, arg: str) -> None:
    shown = _shown(arg)
    try:
        doc = pipeline.read_path(
            Path(arg),
            languages=run.lang,
            weights_dir=run.settings.weights_dir,
            tesseract_path=run.tesseract,
            line_timeout_s=run.settings.line_timeout_s,
            model=run.model,
        )
    except FolioError as exc:
        if isinstance(exc, WeightsError) and _is_not_a_folder(run.settings.weights_dir):
            run.fail(WEIGHTS_DIR_IS_A_FILE.format(_shown(str(run.settings.weights_dir))))
            return
        run.fail(f"{shown}: {exc}")
        run.needs_setup = run.needs_setup or isinstance(exc, SetupNeededError)
        return
    except OSError as exc:
        run.fail(describe_os_error(exc, shown))
        return
    run.docs.append(doc)
    saved = None
    if run.out is None:
        _print_doc(run, arg, doc)
    else:
        saved = _save(run, doc)
    if run.out is None or saved is not None:
        _emit(sys.stderr, _summary(arg, doc, saved))
    _emit(sys.stderr, _page_notes(arg, doc))
    run.failed = run.failed or _has_page_error(doc)


def _needs_model_server(doc: Document) -> bool:
    return any(
        p.origin == "ocr" and p.lines and all(NEEDS_MODEL_SERVER in x.flags for x in p.lines)
        for p in doc.pages
    )


def _make_out(out: Path) -> int | None:
    try:
        out.mkdir(exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return _fail(f"folder not found: {_shown(str(out.parent))}")
        return _fail(describe_os_error(exc, _shown(str(out))))
    return None


def _cmd_read(args: argparse.Namespace, settings: Settings, environ: Mapping[str, str]) -> int:
    config = ModelConfig.from_settings(settings)  # refuses a server without a model name
    out = Path(args.out) if args.out else None
    if out is not None and (code := _make_out(out)) is not None:
        return code
    tesseract = find_tesseract(settings, environ)
    model = None if config is None else ModelReader(config)
    run = _Run(settings, tesseract, args.lang, args.format, out, [Path(f) for f in args.files],
               len(args.files) > 1, model=model)  # fmt: skip
    for arg in args.files:
        _read_one(run, arg)
    if any(_needs_model_server(d) for d in run.docs):
        _emit(sys.stderr, HINT + "\n")
    if run.needs_setup:
        _emit(sys.stderr, SETUP_HINT + "\n")
    if any(_has_page_error(d) for d in run.docs):
        _emit(sys.stderr, DOCTOR_HINT + "\n")
    return 1 if run.failed else 0


def _cmd_setup(settings: Settings) -> int:
    try:
        assets.fetch(
            assets.MANIFEST,
            settings.weights_dir,
            report=lambda m: _emit(sys.stderr, m + "\n"),
        )
    except OSError as exc:
        shown = _shown(str(settings.weights_dir))
        if isinstance(exc, FileExistsError) and _is_not_a_folder(settings.weights_dir):
            return _fail(WEIGHTS_DIR_IS_A_FILE.format(shown))
        return _fail(describe_os_error(exc, shown))
    _emit(sys.stderr, f"assets ready in {_shown(str(settings.weights_dir))}\n")
    return 0


def _dispatch(argv: Sequence[str], environ: Mapping[str, str]) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help
        return exc.code if isinstance(exc.code, int) else 0
    if args.command == "serve":
        from folio_ocr.web.server import serve  # the web stack loads only for this command

        return serve(Settings.from_env(environ, allow_remote=args.allow_remote), environ)
    if args.command == "read" and args.format == "docx" and not args.out:
        parser.error("--format docx needs --out DIR")
    settings = Settings.from_env(environ)
    if args.command == "read":
        return _cmd_read(args, settings, environ)
    if args.command == "setup":
        return _cmd_setup(settings)
    asks_model = settings.model_url is not None or settings.tau_file is not None
    model = {"model_checks": model_health.model_checks} if asks_model else {}
    return doctor.run(
        settings, lambda text: _emit(sys.stdout, _shown(text) + "\n"), environ=environ, **model
    )


def _stdout_failed(exc: _StdoutError) -> int:
    """Point standard output at the null device, so the exit flush cannot fail again; a
    closed pipe (EPIPE, or EINVAL on Windows) ends quietly, anything else gets one line."""
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass  # standard output has no file descriptor left to redirect
    if exc.cause.errno in (errno.EPIPE, errno.EINVAL):
        return 1
    return _fail(describe_os_error(exc.cause, "standard output"))


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Run the command line; returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    environ = os.environ if environ is None else environ
    try:
        return _dispatch(argv, environ)
    except _UsageError as exc:
        return _fail(_shown(str(exc).replace("\n", " ")))
    except FolioError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        _emit(sys.stderr, "error: interrupted\n")
        return 130
    except _StdoutError as exc:
        return _stdout_failed(exc)
    except Exception as exc:  # noqa: BLE001 - last resort: one generic line, never the message
        return _fail(f"unexpected failure ({type(exc).__name__}); please report it")
