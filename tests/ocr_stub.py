"""Stand-ins for the OCR step, so ingest, routing and threading tests run without model files.

The stub reads a page the way a document with no reader would: no lines, status "unread", the
whole image as one unread region. It records every call it receives."""

from dataclasses import dataclass, field

import pytest

from folio_ocr import ingest
from folio_ocr.document import IDENTITY, Page
from folio_ocr.ocr_setup import OcrSetup


@dataclass(frozen=True)
class PageCall:
    page_no: int
    size: tuple[int, int]
    reasons: tuple[str, ...]
    invalid_lines: int | None
    pdfium_locked: bool


@dataclass
class OcrCalls:
    pages: list[PageCall] = field(default_factory=list)
    prepared: list[tuple] = field(default_factory=list)  # languages, folder, tesseract, timeout
    setup: OcrSetup | None = None  # what prepare returns


def install(monkeypatch: pytest.MonkeyPatch) -> OcrCalls:
    dummy = OcrSetup("none", None, None, None, 60.0, None, None, None)  # type: ignore[arg-type]
    calls = OcrCalls(setup=dummy)

    def prepare(languages, weights_dir, tesseract_path, timeout_s):
        calls.prepared.append((languages, weights_dir, tesseract_path, timeout_s))
        return dummy

    def ocr_page(page_no, image, *, ocr, languages, render_dpi, reasons=(), invalid_lines=None):
        calls.pages.append(
            PageCall(
                page_no, image.size, tuple(reasons), invalid_lines, ingest.PDFIUM_LOCK.locked()
            )
        )
        width, height = image.size
        return Page(
            index=page_no,
            width=width,
            height=height,
            render_dpi=render_dpi,
            rotation_applied=0.0,
            transform=IDENTITY,
            origin="ocr",
            direction="rtl",
            status="unread",
            flags=(*reasons, "unread_regions"),
            unread_regions=((0.0, 0.0, float(width), float(height)),),
            lines=(),
            invalid_lines=invalid_lines,
        )

    monkeypatch.setattr("folio_ocr.pipeline.prepare", prepare)
    monkeypatch.setattr("folio_ocr.pipeline.ocr_page", ocr_page)
    return calls


@pytest.fixture
def ocr_stub(monkeypatch):
    return install(monkeypatch)


@pytest.fixture(scope="module")
def ocr_stub_module():
    with pytest.MonkeyPatch.context() as mp:
        yield install(mp)
