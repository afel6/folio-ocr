"""Checks for the built-in printed lines the model self-test reads."""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fixture_checks import assert_same_png
from PIL import Image

from folio_ocr.config import Settings, find_tesseract
from folio_ocr.model_self_test_lines import LINES

HERE = Path(__file__).resolve().parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ENGLISH = "Invoice number: INV-2026-3706"
ARABIC_CODE_POINTS = [
    *(0x0631, 0x0642, 0x0645, 0x20),
    *(0x0627, 0x0644, 0x0641, 0x0627, 0x062A, 0x0648, 0x0631, 0x0629, 0x20),
    *(0x0661, 0x0662, 0x0663, 0x0664),
]
PINNED_SHA256 = {
    "en": "0042ed00951ce2b057bc38f8effec78fde244da15c4c776e529188e53d4ce82a",
    "ar": "ab2f3602520f86a30526f36be76c6709466f8a6b19855dbcc29019455966093b",
}


def test_two_lines_english_then_arabic_with_exact_texts():
    assert [line.languages for line in LINES] == ["en", "ar"]
    assert LINES[0].text == ENGLISH
    assert [ord(c) for c in LINES[1].text] == ARABIC_CODE_POINTS


@pytest.mark.parametrize("line", LINES, ids=lambda line: line.languages)
def test_each_line_is_a_grey_png_of_line_shape(line):
    assert line.png.startswith(PNG_SIGNATURE)
    img = Image.open(io.BytesIO(line.png))
    assert img.mode == "L"
    assert 40 <= img.height <= 120
    assert img.width > img.height * 3


@pytest.mark.parametrize("line", LINES, ids=lambda line: line.languages)
def test_sha256_field_matches_bytes_and_pin(line):
    assert line.sha256 == hashlib.sha256(line.png).hexdigest()
    assert line.sha256 == PINNED_SHA256[line.languages]


def test_regenerated_lines_match_committed_png_content():
    pytest.importorskip("fpdf")
    pytest.importorskip("uharfbuzz")
    sys.path.insert(0, str(HERE.parent / "scripts"))
    import make_self_test_lines as gen

    rendered = gen.render_lines()
    assert rendered.keys() == {line.languages for line in LINES}
    for line in LINES:
        assert_same_png(rendered[line.languages], line.png)


def _tesseract_read(png: bytes, lang: str) -> str:
    settings = Settings.from_env(os.environ)
    program = find_tesseract(settings)
    tessdata = settings.weights_dir / "tessdata"
    if (
        program is None
        or not (tessdata / f"{'ara' if lang == 'ara' else 'eng'}.traineddata").is_file()
    ):
        pytest.skip("Tesseract or its language data is not available")
    done = subprocess.run(
        [
            str(program),
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(tessdata),
            "-l",
            lang,
            "--psm",
            "7",
        ],
        input=png,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0
    return re.sub(r"\s+", " ", done.stdout.decode("utf-8")).strip()


def test_tesseract_reads_the_english_line_back_exactly():
    assert _tesseract_read(LINES[0].png, "eng") == LINES[0].text


def test_tesseract_reads_the_arabic_words_back_in_order():
    # Shaped, joined, right-to-left words: the first three code points are one word, then the next word.
    words = LINES[1].text.rsplit(" ", 1)[0]
    assert _tesseract_read(LINES[1].png, "ara").startswith(words)


@pytest.mark.xfail(
    reason="Tesseract's Arabic model misreads the Arabic-Indic digits and adds direction marks",
    strict=False,
)
def test_tesseract_reads_the_whole_arabic_line_back_exactly():
    assert _tesseract_read(LINES[1].png, "ara") == LINES[1].text
