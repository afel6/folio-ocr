"""Read a PDF or image file into a Document: the one entry point of the CLI and the API.

Settings arrive as plain arguments; nothing here reads the environment or the network."""

import errno
import hashlib
import os
import stat
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from folio_ocr.coverage import has_ink
from folio_ocr.document import Document, Engine, Page, Source
from folio_ocr.errors import InputError
from folio_ocr.flags import LANGUAGES
from folio_ocr.ingest import (
    MAX_UPLOAD_BYTES,
    PDFIUM_LOCK,
    RENDER_DPI,
    check_size,
    frame_count,
    load_frame,
    open_image,
    open_pdf,
    sanitise_filename,
    sniff,
)
from folio_ocr.model_pages import ModelDocument, ModelPage
from folio_ocr.model_reader import ModelReader
from folio_ocr.ocr import ocr_page
from folio_ocr.ocr_setup import OcrSetup, engine, prepare
from folio_ocr.text_layer import read_text_layer
from folio_ocr.text_layer_check import check_text_layer
from folio_ocr.text_layer_page import text_layer_page


def _check_languages(languages: str) -> None:
    if languages not in LANGUAGES:
        raise InputError("languages must be en, ar or en+ar")


def read_path(
    path: Path,
    *,
    languages: str,
    weights_dir: Path | None = None,
    tesseract_path: Path | None = None,
    line_timeout_s: float = 60.0,
    model: ModelReader | None = None,
) -> Document:
    """Read the file at `path`. Its size is checked before any byte is read. OSError from
    the file system propagates; a folder raises IsADirectoryError on every platform."""
    info = path.stat()
    if stat.S_ISDIR(info.st_mode):
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
    check_size(info.st_size)
    with path.open("rb") as handle:
        data = handle.read(MAX_UPLOAD_BYTES + 1)
    return read_bytes(
        data,
        path.name,
        languages=languages,
        weights_dir=weights_dir,
        tesseract_path=tesseract_path,
        line_timeout_s=line_timeout_s,
        model=model,
    )


def read_bytes(
    data: bytes,
    filename: str,
    *,
    languages: str,
    weights_dir: Path | None = None,
    tesseract_path: Path | None = None,
    line_timeout_s: float = 60.0,
    model: ModelReader | None = None,
) -> Document:
    """Read an uploaded file. A refused file raises InputError with one friendly line; missing
    or damaged OCR model files raise WeightsError when a page needs OCR. With `model`, the
    model server reads every OCR line and re-reads text-layer lines whose order or content
    failed validation."""
    _check_languages(languages)
    check_size(len(data))
    kind = sniff(data)
    reader = _Reader(languages, weights_dir, tesseract_path, line_timeout_s, model)
    pages = _read_pdf(data, languages, reader) if kind == "pdf" else _read_image(data, kind, reader)
    source = Source(
        filename=sanitise_filename(filename),
        sha256=hashlib.sha256(data).hexdigest(),
        kind="pdf" if kind == "pdf" else "image",
        page_count=len(pages),
    )
    return Document(
        id=uuid.uuid4().hex,
        source=source,
        languages=languages,
        engine=reader.engine(),
        pages=pages,
        created_at=datetime.now(UTC),
        has_edits=False,
    )


class _Reader:
    """Reads OCR pages; the reader setup (model files, Tesseract) is resolved at the first OCR
    page, so a document whose pages all keep their text layer needs no model files."""

    def __init__(
        self,
        languages: str,
        weights_dir: Path | None,
        tesseract: Path | None,
        timeout_s: float,
        model: ModelReader | None = None,
    ) -> None:
        self.languages = languages
        self._settings = (weights_dir, tesseract, timeout_s)
        self.setup: OcrSetup | None = None
        self.model = None if model is None else ModelDocument(model, languages)

    def page(self, pending: "_Pending") -> Page:
        if self.setup is None:
            self.setup = prepare(self.languages, *self._settings)
        if self.model is None:
            return self._ocr_page(pending, self.setup)
        # The model reads every line, so a problem of the OCR reader does not fail the page.
        ocr = replace(self.setup, problem=None)
        return self._ocr_page(pending, ocr, model=self.model.ocr_page(ocr))

    def _ocr_page(self, pending: "_Pending", ocr: OcrSetup, **model: ModelPage) -> Page:
        return ocr_page(
            pending.page_no,
            pending.image,
            ocr=ocr,
            languages=self.languages,
            render_dpi=pending.render_dpi,
            reasons=pending.reasons,
            invalid_lines=pending.invalid_lines,
            **model,
        )

    def kept(self, kept: "_Kept") -> Page:
        """A page kept from its text layer; the model re-reads its doubtful lines."""
        if self.model is None:
            return kept.page
        return self.model.text_page().reread(kept.page, kept.image)

    def engine(self) -> Engine:
        base = engine(self.setup)
        return base if self.model is None else self.model.engine(base)


@dataclass(frozen=True)
class _Kept:
    """A page kept from its text layer, with its render for the lines the model re-reads."""

    page: Page
    image: Image.Image


@dataclass(frozen=True)
class _Pending:
    """A page that goes to OCR, rendered and waiting to be read."""

    page_no: int
    image: Image.Image
    render_dpi: int | None
    reasons: tuple[str, ...] = ()
    invalid_lines: int | None = None


def _read_pdf(data: bytes, languages: str, reader: _Reader) -> tuple[Page, ...]:
    """Page by page: a page kept from its text layer is built under PDFIUM_LOCK; a page sent to
    OCR is only rendered there and read after the lock is released, before the next page is
    rendered (one render in memory at a time). Every PDFium call holds the lock."""
    doc = open_pdf(data)
    try:
        with PDFIUM_LOCK:
            count = len(doc)
        pages = []
        for i in range(count):
            with PDFIUM_LOCK:
                found = _read_pdf_page(doc, i, languages)
            pages.append(reader.page(found) if isinstance(found, _Pending) else reader.kept(found))
        return tuple(pages)
    finally:
        with PDFIUM_LOCK:
            doc.close()


def _render(page: pdfium.PdfPage) -> Image.Image:
    """Grayscale render at RENDER_DPI that owns its pixels, so no pdfium bitmap outlives the
    lock."""
    bitmap = page.render(scale=RENDER_DPI / 72, grayscale=True)
    try:
        return bitmap.to_pil().copy()
    finally:
        bitmap.close()


def _read_pdf_page(doc: pdfium.PdfDocument, i: int, languages: str) -> _Kept | _Pending:
    """Page i + 1: kept from its text layer, or rendered for OCR (R4 routing)."""
    damaged = f"page {i + 1} of the PDF could not be read; it may be damaged"
    try:
        page = doc[i]
    except pdfium.PdfiumError as err:
        raise InputError(damaged) from err
    try:
        image = _render(page)
        segments = read_text_layer(page, RENDER_DPI)
        report = check_text_layer(segments, has_ink=has_ink(image))
        if report.to_ocr:
            return _Pending(i + 1, image, RENDER_DPI, report.reasons, report.invalid_lines)
        return _Kept(text_layer_page(i + 1, page, image, segments, report, languages), image)
    except pdfium.PdfiumError as err:
        raise InputError(damaged) from err
    finally:
        page.close()


def _read_image(data: bytes, kind: str, reader: _Reader) -> tuple[Page, ...]:
    """Frames are decoded and read one at a time, so at most one frame is held in memory."""
    image = open_image(data, kind)
    try:
        return tuple(
            reader.page(_Pending(i + 1, load_frame(image, i), None))
            for i in range(frame_count(image))
        )
    finally:
        image.close()


def read_page(
    data: bytes,
    page_no: int,
    *,
    languages: str,
    weights_dir: Path | None = None,
    tesseract_path: Path | None = None,
    line_timeout_s: float = 60.0,
    model: ModelReader | None = None,
) -> tuple[Page, Engine]:
    """Read page `page_no` (from 1) of an uploaded file the way `read_bytes` reads it, with the
    engine record `read_bytes` would give a document holding only that page. ValueError when
    the file has no such page."""
    _check_languages(languages)
    check_size(len(data))
    kind = sniff(data)
    reader = _Reader(languages, weights_dir, tesseract_path, line_timeout_s, model)
    if kind == "pdf":
        page = _one_pdf_page(data, page_no, languages, reader)
    else:
        page = _one_image_page(data, kind, page_no, reader)
    return page, reader.engine()


def _check_page_no(page_no: int, count: int) -> None:
    if not 1 <= page_no <= count:
        raise ValueError(f"page {page_no} is not in the file")


def _one_pdf_page(data: bytes, page_no: int, languages: str, reader: _Reader) -> Page:
    doc = open_pdf(data)
    try:
        with PDFIUM_LOCK:
            count = len(doc)
        _check_page_no(page_no, count)
        with PDFIUM_LOCK:
            found = _read_pdf_page(doc, page_no - 1, languages)
        return reader.page(found) if isinstance(found, _Pending) else reader.kept(found)
    finally:
        with PDFIUM_LOCK:
            doc.close()


def _frame(data: bytes, kind: str, page_no: int) -> Image.Image:
    """Frame `page_no` (from 1) of an image file, decoded; ValueError when it has no such frame."""
    image = open_image(data, kind)
    try:
        _check_page_no(page_no, frame_count(image))
        return load_frame(image, page_no - 1)
    finally:
        image.close()


def _one_image_page(data: bytes, kind: str, page_no: int, reader: _Reader) -> Page:
    return reader.page(_Pending(page_no, _frame(data, kind, page_no), None))


def page_image(data: bytes, page_no: int) -> Image.Image:
    """Page `page_no` (from 1) as the readers get it, before straightening: the gray render at
    RENDER_DPI of a PDF page, or the decoded image frame (EXIF applied, mode L or RGB)."""
    kind = sniff(data)
    if kind != "pdf":
        return _frame(data, kind, page_no)
    doc = open_pdf(data)
    with PDFIUM_LOCK:
        try:
            _check_page_no(page_no, len(doc))
            page = doc[page_no - 1]
            try:
                return _render(page)
            finally:
                page.close()
        except pdfium.PdfiumError as err:
            raise InputError(
                f"page {page_no} of the PDF could not be read; it may be damaged"
            ) from err
        finally:
            doc.close()
