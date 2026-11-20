"""Reading an uploaded file from a multipart form and making its unread document."""

import hashlib
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from folio_ocr import ingest, ocr_setup
from folio_ocr.document import IDENTITY, Document, Page, Source
from folio_ocr.errors import InputError
from folio_ocr.flags import LANGUAGES

FORM_OVERHEAD = 64 * 1024  # multipart boundaries, part headers and the languages part
BODY_LIMIT = ingest.MAX_UPLOAD_BYTES + FORM_OVERHEAD
NOT_MULTIPART = "send the file as multipart/form-data, in a part named file"
NO_FILE = "no file was sent; choose a file to upload"
TWO_FILES = "send one file at a time"
BAD_FORM = "the upload could not be read; try again"
BAD_LANGUAGES = "languages must be en, ar or en+ar"


@dataclass(frozen=True)
class Upload:
    filename: str
    data: bytes = field(repr=False)
    languages: str | None  # None when the part is missing or sent more than once


async def read_body(
    chunks: AsyncIterable[bytes], content_length: str | None, limit: int = BODY_LIMIT
) -> bytes:
    """The request body; refuses a body over `limit` as soon as that is known."""
    declared = content_length is not None and content_length.isascii() and content_length.isdigit()
    if declared and int(content_length) > limit:
        ingest.check_size(int(content_length))
    parts: list[bytes] = []
    total = 0
    async for chunk in chunks:
        total += len(chunk)
        if total > limit:
            ingest.check_size(total)
        parts.append(chunk)
    return b"".join(parts)


def _boundary(content_type: str) -> bytes:
    media, options = parse_options_header(content_type)
    boundary = options.get(b"boundary", b"")
    if media.lower() != b"multipart/form-data" or not boundary:
        raise InputError(NOT_MULTIPART)
    return boundary


@dataclass(frozen=True)
class _Part:
    headers: dict[bytes, bytes]  # header names lower-cased
    data: bytes = field(repr=False)


def _split_parts(boundary: bytes, body: bytes) -> list[_Part]:
    """The parts of a multipart body, in memory; the part bytes are never altered."""
    parts: list[_Part] = []
    state: dict = {"headers": {}, "data": [], "field": b"", "value": b"", "ended": False}

    def on_part_begin() -> None:
        state.update(headers={}, data=[])

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]

    def on_header_end() -> None:
        state["headers"][state["field"].lower()] = state["value"]
        state.update(field=b"", value=b"")

    def on_part_data(data: bytes, start: int, end: int) -> None:
        state["data"].append(data[start:end])

    def on_part_end() -> None:
        parts.append(_Part(state["headers"], b"".join(state["data"])))

    def on_end() -> None:
        state["ended"] = True

    callbacks = {
        "on_part_begin": on_part_begin,
        "on_header_field": on_header_field,
        "on_header_value": on_header_value,
        "on_header_end": on_header_end,
        "on_part_data": on_part_data,
        "on_part_end": on_part_end,
        "on_end": on_end,
    }
    try:
        parser = MultipartParser(boundary, callbacks)
        parser.write(body)
        parser.finalize()
    except (MultipartParseError, ValueError) as exc:
        raise InputError(BAD_FORM) from exc
    if not state["ended"]:
        raise InputError(BAD_FORM)
    return parts


def _named(part: _Part) -> tuple[bytes, bytes | None]:
    """The form field name and file name (None for a plain field) of a part."""
    disposition = part.headers.get(b"content-disposition")
    if disposition is None:
        raise InputError(BAD_FORM)
    _, options = parse_options_header(disposition)
    name = options.get(b"name")
    if name is None:
        raise InputError(BAD_FORM)
    return name, options.get(b"filename")


def parse_form(content_type: str, body: bytes) -> Upload:
    """The one file and the languages value of a multipart upload form."""
    named = [(_named(p), p) for p in _split_parts(_boundary(content_type), body)]
    files = [p for (name, fname), p in named if name == b"file" and fname is not None]
    if not files:
        raise InputError(NO_FILE)
    if len(files) > 1:
        raise InputError(TWO_FILES)
    languages = [p for (name, fname), p in named if name == b"languages" and fname is None]
    (only,) = files
    filename = _named(only)[1] or b""
    value = languages[0].data.decode("utf-8", errors="replace") if len(languages) == 1 else None
    return Upload(
        filename=filename.decode("utf-8", errors="replace"), data=only.data, languages=value
    )


def unread_page(index: int) -> Page:
    return Page(
        index=index,
        width=0,
        height=0,
        render_dpi=None,
        rotation_applied=0.0,
        transform=IDENTITY,
        origin="ocr",
        direction="ltr",
        status="unread",
        flags=(),
        unread_regions=(),
        lines=(),
    )


def _page_count(data: bytes, kind: str) -> int:
    if kind == "pdf":
        with ingest.PDFIUM_LOCK:
            doc = ingest.open_pdf(data)
            try:
                return len(doc)
            finally:
                doc.close()
    img = ingest.open_image(data, kind)
    try:
        return ingest.frame_count(img)
    finally:
        img.close()


def new_document(upload: Upload) -> Document:
    """A document with one unread page per page of the file; runs the file checks."""
    if upload.languages not in LANGUAGES:
        raise InputError(BAD_LANGUAGES)
    data = upload.data
    ingest.check_size(len(data))
    kind = ingest.sniff(data)
    count = _page_count(data, kind)
    return Document(
        id=uuid.uuid4().hex,
        source=Source(
            filename=ingest.sanitise_filename(upload.filename),
            sha256=hashlib.sha256(data).hexdigest(),
            kind="pdf" if kind == "pdf" else "image",
            page_count=count,
        ),
        languages=upload.languages,
        engine=ocr_setup.engine(None),
        pages=tuple(unread_page(i) for i in range(1, count + 1)),
        created_at=datetime.now(UTC),
        has_edits=False,
    )
