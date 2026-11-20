"""Open a user's PDF or image in a reader process with limits the operating system enforces.

One process per opened file (`folio_ocr.pdf_worker`), started as `python -P -m ...` without
folio's own settings in its environment. Its committed memory is capped at PDF_MEMORY_MB
(Windows: a job object set up before it runs; Linux: RLIMIT_DATA it sets before it says it is
ready, so before any file byte is sent) and its work at PDF_SECONDS: a timer kills it when the
time runs out and stays set until a reply is read in full, and the time spent is checked again
when the reply is complete. Only time inside a request counts, not the wait for a slot, the
reading of pages afterwards, or the start of the process, which has a limit of its own of
PDF_SECONDS (no file byte is sent before it says it is ready, so no file can slow it).
At most MAX_WORKERS processes live at once; more callers wait.
This bounds what a file can cost; it is not a sandbox for code run inside PDFium."""

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress

from PIL import Image

from folio_ocr import ingest, pdf_frames, pdf_replies
from folio_ocr.errors import FolioError, InputError, PdfReaderError
from folio_ocr.pdf_replies import PdfPage

MAX_WORKERS = 2
_HEADER_LIMIT = 8 * 2**20  # a reply's JSON: about 48,000 characters of text on one page
_SLOTS = threading.BoundedSemaphore(MAX_WORKERS)
_live = 0
_LIVE_LOCK = threading.Lock()


def live_workers() -> int:
    """Reader processes alive in this process right now."""
    with _LIVE_LOCK:
        return _live


def _count_live(change: int) -> None:
    global _live
    with _LIVE_LOCK:
        _live += change


def limit_line(kind: str) -> str:
    what = "the PDF" if kind == "pdf" else "the image"
    return f"{what} could not be read within folio's memory and time limits; it was not read"


def _command() -> list[str]:
    return [sys.executable, "-P", "-m", "folio_ocr.pdf_worker", str(ingest.PDF_MEMORY_MB)]


def _environment() -> dict[str, str]:
    """The environment without folio's settings, so the model server key never reaches it."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("FOLIO_")}


class _Worker:
    """One reader process. `close()` ends it on every path."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._left = ingest.PDF_SECONDS
        self._ended = threading.Event()
        self._memory_hit = False
        self._job = None
        self._proc: subprocess.Popen | None = None
        self.pid: object = None
        self._closing = threading.Lock()  # the timer's kill never meets close()
        self._closed = False

    def start(self) -> None:
        self._spawn()
        with self._timed(lambda: PdfReaderError(self.kind, "the reader process did not start")):
            ack = pdf_frames.read_header(self._proc.stdout, _HEADER_LIMIT)
        self.pid = ack.get("pid")  # the interpreter's own id (on Windows not the launcher's)
        if ack.get("ready") is not True:
            raise PdfReaderError(self.kind, "the reader process could not limit its memory")
        self._left = ingest.PDF_SECONDS  # the file's work is counted from here

    def _spawn(self) -> None:
        flags = 0
        try:
            if sys.platform == "win32":
                from folio_ocr.job_object import CREATE_SUSPENDED, JobObject

                self._job = JobObject(ingest.PDF_MEMORY_MB * 2**20)
                flags = CREATE_SUSPENDED
            elif not sys.platform.startswith("linux"):
                raise PdfReaderError(self.kind, "this system offers no memory limit folio can use")
        except OSError as err:
            raise PdfReaderError(self.kind, f"{err.strerror} (error {err.errno})") from err
        if not sys.executable:
            raise PdfReaderError(self.kind, "the Python program is not known")
        try:
            self._proc = subprocess.Popen(
                _command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=_environment(),
                creationflags=flags,
            )
        except (OSError, ValueError) as err:
            raise PdfReaderError(self.kind, "the reader process could not be started") from err
        if self._job is not None:
            try:
                self._job.adopt(self._proc.pid)
            except OSError as err:
                raise PdfReaderError(self.kind, f"{err.strerror} (error {err.errno})") from err

    def _expire(self) -> None:
        """Timer: the time is up, so the process is killed and the pending read ends."""
        self._ended.set()
        self._kill()

    def _kill(self) -> None:
        with self._closing:
            if self._closed:
                return
            if self._job is not None:
                self._job.terminate()
            if self._proc is not None:
                self._proc.kill()

    def _limits_hit(self) -> bool:
        if self._job is not None and not self._memory_hit:
            self._memory_hit = self._job.memory_limit_hit()
        return self._memory_hit or self._ended.is_set() or self._left < 0

    @contextmanager
    def _timed(self, failure: Callable[[], FolioError]) -> Iterator[None]:
        if self._left <= 0:
            raise failure()
        started = time.monotonic()
        timer = threading.Timer(self._left, self._expire)
        timer.daemon = True
        timer.start()
        try:
            yield
        except InputError as err:  # a refusal, unless a limit made the file fail
            if self._limits_hit():
                raise failure() from err
            raise
        except (pdf_frames.Ended, OSError) as err:
            raise failure() from err
        finally:
            timer.cancel()
            self._left -= time.monotonic() - started
        if self._limits_hit():
            raise failure()

    def request(self, header: dict, read: Callable[[dict], object], data: bytes | None = None):
        """Send one request and build its result from the reply with `read`."""
        with self._timed(lambda: InputError(limit_line(self.kind))):
            pdf_frames.write(self._proc.stdin, header, data)
            reply = pdf_frames.read_header(self._proc.stdout, _HEADER_LIMIT)
            if "refused" in reply:
                raise InputError(pdf_replies.refusal(reply["refused"]))
            return read(reply)

    def pixels(self, size: tuple[int, int], mode: str) -> Image.Image:
        """The bytes message that follows a reply, as an image of `size` in `mode`."""
        data = pdf_frames.read_data(self._proc.stdout, pdf_replies.byte_count(size, mode))
        return Image.frombytes(mode, size, data)

    def close(self) -> None:
        with self._closing:
            self._closed = True
            if self._job is not None:
                self._job.close()
            if self._proc is None:
                return
            self._proc.kill()
        self._proc.wait()
        for pipe in (self._proc.stdin, self._proc.stdout):
            with suppress(OSError):  # the process is gone; its pipe may refuse a last flush
                pipe.close()


class OpenFile:
    """A file opened and checked in a reader process: `count` pages or frames."""

    def __init__(self, worker: _Worker, data: bytes) -> None:
        self._worker = worker
        self._sizes: tuple[tuple[int, int], ...] = ()
        if worker.kind == "pdf":
            self._sizes = worker.request(
                {"op": "open", "kind": "pdf"}, pdf_replies.page_sizes, data
            )
            self.count = len(self._sizes)
        else:
            self.count = worker.request(
                {"op": "open", "kind": worker.kind}, pdf_replies.frame_count, data
            )

    def _index(self, index: int) -> int:
        if not 0 <= index < self.count:
            raise ValueError(f"page {index + 1} is not in the file")
        return index

    def _render_size(self, reply: dict, index: int) -> tuple[int, int]:
        size = pdf_replies.image_size(reply, "L")
        if size != self._sizes[index]:
            raise pdf_frames.Ended("the reader sent a render of another size")
        return size

    def page(self, index: int) -> PdfPage:
        """PDF page `index` (from 0): its render at RENDER_DPI, text layer and image boxes."""

        def read(reply: dict) -> PdfPage:
            lines, images = pdf_replies.text_lines(reply), pdf_replies.image_boxes(reply)
            size = self._render_size(reply, index)
            return PdfPage(self._worker.pixels(size, "L"), lines, images)

        return self._worker.request({"op": "page", "index": self._index(index)}, read)

    def render(self, index: int) -> Image.Image:
        """PDF page `index` (from 0) rendered in gray at RENDER_DPI."""

        def read(reply: dict) -> Image.Image:
            return self._worker.pixels(self._render_size(reply, index), "L")

        return self._worker.request({"op": "render", "index": self._index(index)}, read)

    def frame(self, index: int) -> Image.Image:
        """Image frame `index` (from 0), decoded, EXIF applied, in mode L or RGB."""

        def read(reply: dict) -> Image.Image:
            mode = pdf_replies.frame_mode(reply)
            info = pdf_replies.frame_info(reply, mode)
            image = self._worker.pixels(pdf_replies.image_size(reply, mode), mode)
            image.info.update(info)
            return image

        return self._worker.request({"op": "frame", "index": self._index(index)}, read)


@contextmanager
def _running(kind: str) -> Iterator[_Worker]:
    """A started reader process in a free slot (callers wait for one); ended on every path."""
    with _SLOTS:
        worker = _Worker(kind)
        _count_live(1)
        try:
            worker.start()
            yield worker
        finally:
            worker.close()
            _count_live(-1)


@contextmanager
def opened(data: bytes, kind: str) -> Iterator[OpenFile]:
    """Open `data` (of `kind` from `ingest.sniff`) in a reader process, waiting for a free slot.
    The process ends when the block ends. InputError: the file is refused or broke a limit;
    PdfReaderError: no reader process with limits could be started."""
    with _running(kind) as worker:
        yield OpenFile(worker, data)


def self_check() -> None:
    """Start a reader process with its limits and end it; PdfReaderError says why it failed."""
    with _running("pdf"):
        pass
