"""Pages read by OCR: PDF pages whose text layer is not used, and image files.

The page is turned upright and deskewed, its words are found and grouped into lines, and each
line is read by the document's reader (OnnxTR for English, Tesseract for Arabic and mixed pages
when installed). With no reader, lines keep their boxes and wait for a model server. Ink outside
every line box is reported as unread regions."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from folio_ocr import calibration, tesseract
from folio_ocr.charsets import line_script
from folio_ocr.coverage import uncovered_ink
from folio_ocr.document import Line, Page, line_id
from folio_ocr.flags import (
    NEEDS_MODEL_SERVER,
    NOT_READ,
    ORIENTATION_UNCERTAIN,
    UNREAD_REGIONS,
    clean_text,
    line_flags,
    review_status,
)
from folio_ocr.geometry import straighten
from folio_ocr.ocr_setup import OcrSetup
from folio_ocr.ocr_words import LineWords, Word, find_words, group_lines, recognise_words
from folio_ocr.onnxtr_models import Detection
from folio_ocr.orientation import page_orientation
from folio_ocr.reading import Reading
from folio_ocr.text_layer_check import majority_direction, text_direction

if TYPE_CHECKING:
    from folio_ocr.model_pages import ModelPage


def _read(ocr: OcrSetup, gray: Image.Image, line: LineWords) -> Reading | None:
    """What the document's reader makes of `line`; None when nobody can read it."""
    if ocr.kind == "onnxtr":
        text = " ".join(w.text for w in line.words if w.text.strip())
        return Reading(text, tuple(w.score for w in line.words))
    if ocr.kind == "none":
        return None
    if ocr.problem is not None or ocr.tesseract is None or ocr.tessdata is None:
        return Reading("", (), error=ocr.problem or "Tesseract is not set up")
    png = tesseract.line_png(gray, line.box)
    return tesseract.read_line(ocr.tesseract, ocr.tessdata, png, ocr.timeout_s)


def _line(
    ocr: OcrSetup,
    n: tuple[int, int],
    line: LineWords,
    reading: Reading | None,
    page: tuple[str, str, tuple[str, ...]],
) -> Line:
    """Line n = (page number, line number) from its reading; `page` = (direction, languages,
    flags every line of the page carries)."""
    direction, languages, extra = page
    if reading is None or reading.error is not None:
        flags = tuple(dict.fromkeys((NEEDS_MODEL_SERVER if reading is None else NOT_READ, *extra)))
        text, reader, confidence, method = "", "none", None, "none"
    else:
        text, replaced = clean_text(reading.text)
        low = calibration.is_low_confidence(reading.scores, ocr.tau)
        own = (*line_flags(text, languages), *replaced)
        own += (calibration.LOW_CONFIDENCE,) if low else ()
        flags = tuple(dict.fromkeys((*own, *extra)))
        reader, method = ocr.kind, "reader"
        confidence = calibration.line_confidence(reading.scores)
    return Line(
        id=line_id(*n),
        box=line.box,
        text=text,
        direction=text_direction(text, direction),
        script=line_script(text),
        reader=reader,
        confidence=confidence,
        confidence_method=method,
        flags=flags,
        status=review_status(flags, "auto_accepted"),
        final_text=text,
    )


def _page_error(
    ocr: OcrSetup, readings: Sequence[Reading | None], orientation_error: str | None = None
) -> str | None:
    """The page's one-line error: the reader's problem, else the orientation error followed by
    the count of lines not read, else the first line error with the count when several."""
    errors = [r.error for r in readings if r is not None and r.error is not None]
    if ocr.problem is not None or (not errors and orientation_error is None):
        return ocr.problem
    if orientation_error is not None:
        count = len(errors)
        lines = "" if not count else f"; {count} line{'' if count == 1 else 's'} not read"
        return orientation_error + lines
    return errors[0] if len(errors) == 1 else f"{errors[0]} ({len(errors)} lines not read)"


def _model_read_nothing(model: "ModelPage | None", lines: Sequence[Line]) -> bool:
    """True when the model server read the page and answered none of its lines."""
    return model is not None and bool(lines) and all(NOT_READ in line.flags for line in lines)


def _osd(ocr: OcrSetup):
    """Page image -> Tesseract's orientation answer, or None when the document does not use OSD."""
    if not ocr.osd or ocr.tesseract is None or ocr.tessdata is None:
        return None
    binary, tessdata, timeout_s = ocr.tesseract, ocr.tessdata, ocr.timeout_s
    return lambda img: tesseract.osd(binary, tessdata, tesseract.page_png(img), timeout_s)


def _words(
    ocr: OcrSetup, page_img: Image.Image, detection: Detection | None
) -> tuple[Image.Image, list[Word]]:
    gray = page_img.convert("L")
    rgb = np.asarray(page_img.convert("RGB"))
    boxes = find_words(ocr.models, gray, rgb, detection)
    if ocr.kind == "onnxtr":
        return gray, recognise_words(ocr.models, rgb, boxes)
    return gray, [Word(box, "", math.nan) for box in boxes]


def ocr_page(
    page_no: int,
    image: Image.Image,
    *,
    ocr: OcrSetup,
    languages: str,
    render_dpi: int | None,
    reasons: tuple[str, ...] = (),
    invalid_lines: int | None = None,
    model: "ModelPage | None" = None,
) -> Page:
    """The page `image` (post-EXIF, mode L or RGB) read by OCR. `reasons` are the text-layer
    reasons that sent a PDF page here; `invalid_lines` their line count. With `model`, every
    line is read by the model server instead of the document's OCR reader."""
    read, build = (_read, _line) if model is None else (model.read, model.line)
    orientation = page_orientation(ocr.models, image, _osd(ocr))
    page_img, matrix = straighten(image, orientation.angle)
    gray, words = _words(ocr, page_img, orientation.detection)
    found = group_lines(gray, words, "ltr")
    readings = {line.words: read(ocr, gray, line) for line in found}
    direction = majority_direction(r.text for r in readings.values() if r is not None)
    extra = (ORIENTATION_UNCERTAIN,) if orientation.uncertain else ()
    ordered = found if direction == "ltr" else group_lines(gray, words, direction)
    lines = tuple(
        build(ocr, (page_no, k), line, readings[line.words], (direction, languages, extra))
        for k, line in enumerate(ordered, start=1)
    )
    unread = uncovered_ink(gray, [line.box for line in lines])
    return Page(
        index=page_no,
        width=page_img.width,
        height=page_img.height,
        render_dpi=render_dpi,
        rotation_applied=orientation.angle,
        transform=matrix,
        origin="ocr",
        direction=direction,
        status="failed" if ocr.problem is not None or _model_read_nothing(model, lines) else "read",
        flags=(*reasons, *extra, *((UNREAD_REGIONS,) if unread else ())),
        unread_regions=unread,
        lines=lines,
        invalid_lines=invalid_lines,
        error=_page_error(ocr, list(readings.values()), orientation.error),
    )
