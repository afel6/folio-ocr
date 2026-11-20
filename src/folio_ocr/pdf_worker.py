"""The file reader process: every PDFium call and every image decode on a user's file runs here,
never in the folio process (`folio_ocr.pdf_client` starts it, one per opened file).

It answers a closed list of requests: open (the file checks), page (render + text layer),
render, frame (an image frame) and nothing else; any other request ends it. Replies are plain
data (see `pdf_frames`). On Linux it caps its own data segment before it says it is ready;
on Windows folio's job object caps it before it runs."""

import base64
import os
import sys

import pypdfium2 as pdfium

from folio_ocr import ingest, pdf_frames
from folio_ocr.document import TextChar
from folio_ocr.errors import InputError
from folio_ocr.text_layer import image_boxes, page_chars, text_lines

_REQUEST_LIMIT = 4096
_PR_SET_PDEATHSIG = 1


def _limit_self(memory_mb: int) -> bool:
    """Cap this process's memory (Linux) and end it with its parent. Windows: the job object
    folio put this process in already holds the cap, so there is nothing to do here."""
    if sys.platform == "win32":
        return True
    if not sys.platform.startswith("linux"):
        return False
    import ctypes
    import resource
    import signal

    cap = memory_mb * 2**20
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) == 0
    except (OSError, ValueError, AttributeError):
        return False


def _damaged(index: int) -> InputError:
    return InputError(ingest.page_damaged(index + 1))


def _render(page: pdfium.PdfPage):
    bitmap = page.render(scale=ingest.RENDER_DPI / 72, grayscale=True)
    try:
        return bitmap.to_pil().copy()
    finally:
        bitmap.close()


def _box(box) -> list[float] | None:
    return None if box is None else list(box)


def _char(char: TextChar) -> list:
    return [ord(char.text), _box(char.box), char.unmapped, _box(char.tight_box), char.invisible]


def _with_page(state: dict, index: int, use):
    doc = state["pdf"]
    try:
        page = doc[index]
    except pdfium.PdfiumError as err:
        raise _damaged(index) from err
    try:
        return use(page)
    except pdfium.PdfiumError as err:
        raise _damaged(index) from err
    finally:
        page.close()


def _page_reply(page: pdfium.PdfPage) -> tuple[dict, bytes]:
    image = _render(page)
    chars = page_chars(page, ingest.RENDER_DPI)
    try:  # needed only when the page keeps its text layer; folio decides that
        images = [list(box) for box in image_boxes(page, ingest.RENDER_DPI)]
    except pdfium.PdfiumError:
        images = None
    position = {id(char): i for i, char in enumerate(chars)}
    lines = [[position[id(char)] for char in line] for line in text_lines(chars)]
    header = {"mode": image.mode, "size": list(image.size), "images": images, "lines": lines}
    return {**header, "chars": [_char(c) for c in chars]}, image.tobytes()


def _render_reply(page: pdfium.PdfPage) -> tuple[dict, bytes]:
    image = _render(page)
    return {"mode": image.mode, "size": list(image.size)}, image.tobytes()


def _open(state: dict, request: dict, data: bytes) -> tuple[dict, None]:
    kind = request["kind"]
    if kind == "pdf":
        doc = state["pdf"] = ingest.open_pdf(data)
        sizes = [list(doc.get_page_size(i)) for i in range(len(doc))]
        return {"count": len(sizes), "sizes": sizes}, None
    image = state["image"] = ingest.open_image(data, kind)
    return {"count": ingest.frame_count(image)}, None


def _frame(state: dict, index: int) -> tuple[dict, bytes]:
    frame = ingest.load_frame(state["image"], index)
    icc = frame.info.get("icc_profile")
    transparency = frame.info.get("transparency")
    header = {
        "mode": frame.mode,
        "size": list(frame.size),
        "icc_profile": None if icc is None else base64.b64encode(icc).decode("ascii"),
        "transparency": list(transparency) if isinstance(transparency, tuple) else transparency,
    }
    return header, frame.tobytes()


def _answer(state: dict, request: dict, stdin) -> tuple[dict, bytes | None] | None:
    """The reply to one request, or None for a request outside the closed list."""
    op, index = request.get("op"), request.get("index")
    if op == "open" and not state:
        return _open(state, request, bytes(pdf_frames.read_any(stdin, ingest.MAX_UPLOAD_BYTES)))
    if type(index) is not int:
        return None
    if op == "page" and "pdf" in state:
        return _with_page(state, index, _page_reply)
    if op == "render" and "pdf" in state:
        return _with_page(state, index, _render_reply)
    if op == "frame" and "image" in state:
        return _frame(state, index)
    return None


def _private_stdout():
    """A private copy of stdout for replies; fd 1 then points at the null device, so nothing a
    library prints can mix with a reply."""
    out = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(out, os.O_BINARY)
        msvcrt.setmode(0, os.O_BINARY)
    return os.fdopen(out, "wb")


def main(argv: list[str]) -> int:
    out = _private_stdout()
    stdin = sys.stdin.buffer
    ready = _limit_self(int(argv[0]))
    pdf_frames.write(out, {"ready": ready, "pid": os.getpid()})
    state: dict = {}
    while ready:
        try:
            request = pdf_frames.read_header(stdin, _REQUEST_LIMIT)
        except pdf_frames.Ended:
            return 0
        try:
            reply = _answer(state, request, stdin)
        except InputError as err:
            reply = {"refused": str(err)}, None
        if reply is None:
            return 2
        pdf_frames.write(out, *reply)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
