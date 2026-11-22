"""A stand-in for the Tesseract program, driven by environment variables.

FAKE_TESSERACT_LOG   file that gets one JSON line per run: argv, and
                     stdin_len, stdin_sha256, pid for a line run
FAKE_TESSERACT_TSV   file whose bytes are printed (default: data/tesseract/en-line.tsv)
FAKE_TESSERACT_EXIT  exit code (default 0)
FAKE_TESSERACT_SLEEP seconds to wait before printing"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_TSV = Path(__file__).resolve().parent / "data" / "tesseract" / "en-line.tsv"
VERSION_TEXT = "tesseract v5.4.0.20240606\n leptonica-1.84.1\n"


def _log(entry: dict) -> None:
    log = os.environ.get("FAKE_TESSERACT_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def main() -> None:
    if sys.argv[1:2] == ["--version"]:
        _log({"argv": ["--version"]})
        sys.stdout.write(VERSION_TEXT)
        return
    data = sys.stdin.buffer.read()
    _log(
        {
            "argv": sys.argv[1:],
            "stdin_len": len(data),
            "stdin_sha256": hashlib.sha256(data).hexdigest(),
            "pid": os.getpid(),
        }
    )
    time.sleep(float(os.environ.get("FAKE_TESSERACT_SLEEP", "0")))
    tsv = Path(os.environ.get("FAKE_TESSERACT_TSV", DEFAULT_TSV))
    sys.stdout.buffer.write(tsv.read_bytes())
    sys.stdout.buffer.flush()
    sys.exit(int(os.environ.get("FAKE_TESSERACT_EXIT", "0")))


if __name__ == "__main__":
    main()
