"""Capture the Tesseract program's TSV output for the first ground-truth line of the clean
English and Arabic invoice fixtures.

Usage: python scripts/capture_tesseract_tsv.py OUTPUT_FOLDER
Writes en-line.tsv and ar-line.tsv, byte for byte as the program printed them."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from script_errors import describe_os_error

from folio_ocr import assets
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.errors import FolioError
from folio_ocr.tesseract import line_png, run

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
TIMEOUT_S = 60.0


def capture(name: str, binary: Path, tessdata: Path, out_dir: Path) -> None:
    ground_truth = json.loads((FIXTURES / f"{name}-invoice-clean.gt.json").read_text("utf-8"))
    with Image.open(FIXTURES / f"{name}-invoice-clean.png") as image:
        png = line_png(image, tuple(ground_truth["lines"][0]["box"]))
    (out_dir / f"{name}-line.tsv").write_bytes(run(binary, tessdata, png, TIMEOUT_S))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: capture_tesseract_tsv.py OUTPUT_FOLDER\n")
        return 1
    out_dir = Path(argv[1])
    try:
        settings = Settings.from_env(os.environ)
        binary = find_tesseract(settings)
        if binary is None:
            sys.stderr.write("Tesseract was not found; set FOLIO_TESSERACT_PATH\n")
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("en", "ar"):
            capture(name, binary, settings.weights_dir / assets.TESSDATA_DIR, out_dir)
    except FolioError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except OSError as exc:
        sys.stderr.write(f"{describe_os_error(exc, out_dir)}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
