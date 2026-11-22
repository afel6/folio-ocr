"""A stand-in file reader process that misbehaves in one named way (argv[1]); run as a script.
It speaks the framing of folio_ocr.pdf_frames. Behaviours that act on "open" misbehave there;
the others answer "open" for one 72 x 72 pt page and misbehave on the "page" request."""

import os
import struct
import sys
import time

from folio_ocr import ingest, pdf_client, pdf_frames, pdf_worker

OPEN_OK = {"count": 1, "sizes": [[72.0, 72.0]]}
BOX = [0.0, 0.0, 1.0, 1.0]
DAMAGED_1 = "page 1 of the PDF could not be read; it may be damaged"
DAMAGED_21 = "page 21 of the PDF could not be read; it may be damaged"
PIXELS = 300 * 300  # 72 pt at 300 dpi


def page_header(chars=()):
    chars = list(chars)
    lines = [list(range(len(chars)))] if chars else []
    return {"mode": "L", "size": [300, 300], "images": [], "chars": chars, "lines": lines}


def raw(out, data: bytes) -> None:
    out.write(data)
    out.flush()


def on_open(behaviour, out):
    if behaviour == "sleep_forever":
        time.sleep(600)
    if behaviour == "slow_open":
        time.sleep(0.5)
    refusals = {
        "unknown_refusal": "anything else",
        "page_21_refusal": DAMAGED_21,
        "known_refusal": DAMAGED_1,
    }
    if behaviour in refusals:
        pdf_frames.write(out, {"refused": refusals[behaviour]})
    elif behaviour == "huge_header":
        raw(out, struct.pack(">I", pdf_client._HEADER_LIMIT + 1))
    elif behaviour == "not_json":
        raw(out, struct.pack(">I", 5) + b"nope!")
    elif behaviour == "crash_mid_reply":
        raw(out, struct.pack(">I", 100) + b'{"count"')
        os._exit(3)
    else:
        pdf_frames.write(out, OPEN_OK)


def on_page(behaviour, out):
    if behaviour == "sleep_on_page":
        time.sleep(600)
    if behaviour == "stall_in_pixels":  # a good header, then half the pixels, then nothing
        pdf_frames.write(out, page_header())
        raw(out, struct.pack(">I", PIXELS) + bytes(PIXELS // 2))
        time.sleep(600)
    if behaviour == "wrong_render_size":
        pdf_frames.write(out, {**page_header(), "size": [10, 10]}, bytes(100))
    elif behaviour == "wrong_byte_count":
        pdf_frames.write(out, page_header(), bytes(PIXELS - 1))
    elif behaviour == "bad_char":
        pdf_frames.write(out, page_header([[0x110000, BOX, False, None, False]]), bytes(PIXELS))
    elif behaviour == "bool_char":
        pdf_frames.write(out, page_header([[True, BOX, False, None, False]]), bytes(PIXELS))
    else:
        pdf_frames.write(out, page_header(), bytes(PIXELS))


def main(behaviour: str) -> int:
    out = sys.stdout.buffer
    if behaviour == "exit_before_ready":
        return 5
    if behaviour == "sleep_before_ready":
        with open(sys.argv[2], "w") as pid_file:
            pid_file.write(str(os.getpid()))
        time.sleep(30)
        return 0
    ready = behaviour != "not_ready" and pdf_worker._limit_self(ingest.PDF_MEMORY_MB)
    pdf_frames.write(
        out, {"ready": ready, "pid": os.getpid(), "has_key": "FOLIO_MODEL_API_KEY" in os.environ}
    )
    if not ready:
        time.sleep(30)
        return 0
    while True:
        try:
            request = pdf_frames.read_header(sys.stdin.buffer, 4096)
        except pdf_frames.Ended:
            return 0
        if request["op"] == "open":
            pdf_frames.read_any(sys.stdin.buffer, 2**26)
            on_open(behaviour, out)
        else:
            on_page(behaviour, out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
