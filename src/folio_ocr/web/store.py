"""The open documents of the local web service, kept in memory, with their page reads."""

import dataclasses
import threading
from dataclasses import dataclass, field

from folio_ocr import ocr_setup
from folio_ocr.document import Document, Engine, Line, Page
from folio_ocr.model_pages import swap_line

MAX_DOCUMENTS = 3
NO_READER = ocr_setup.engine(None)

DOCUMENT_GONE = "no such document; it may have been deleted or closed to make room"
PAGE_GONE = "no such page in this document"
LINE_GONE = "no such line in this document"
BUSY = "a page is being read; try again when it finishes"
FULL = "three documents are open and the oldest has edits; export it, then delete it to make room"


class ApiError(Exception):
    """A request the service refuses: an HTTP status and one line for the user."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Entry:
    data: bytes = field(repr=False)  # the uploaded file
    doc: Document


def page_number(text: str, page_count: int) -> int:
    """`text` as a page number from 1 to `page_count`: plain ASCII digits, no leading zero."""
    ok = text.isascii() and text.isdigit() and not text.startswith("0")
    ok = ok and len(text) <= len(str(page_count))  # int() refuses very long digit strings
    if not ok or int(text) > page_count:
        raise ApiError(404, PAGE_GONE)
    return int(text)


def keep_reviewed(current: Page, read: Page) -> Page:
    """`read`, with every line a person edited or confirmed in `current` kept."""
    old = {ln.id: ln for ln in current.lines}
    lines = [swap_line(old[ln.id], ln) if ln.id in old else ln for ln in read.lines]
    seen = {ln.id for ln in read.lines}
    kept = [
        ln for ln in current.lines if ln.status in ("edited", "confirmed") and ln.id not in seen
    ]
    return dataclasses.replace(read, lines=(*lines, *kept))


class Store:
    """The open documents, oldest first, in memory only; every method holds one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[Entry] = []
        # The running read: (document id, page number, the page status before it began).
        self._read: tuple[str, int, str] | None = None

    def _index(self, doc_id: str) -> int:
        for i, e in enumerate(self._entries):
            if e.doc.id == doc_id:
                return i
        raise ApiError(404, DOCUMENT_GONE)

    def _set_page(self, i: int, n: int, page: Page) -> Entry:
        e = self._entries[i]
        pages = (*e.doc.pages[: n - 1], page, *e.doc.pages[n:])
        self._entries[i] = dataclasses.replace(e, doc=dataclasses.replace(e.doc, pages=pages))
        return self._entries[i]

    def add(self, data: bytes, doc: Document) -> None:
        with self._lock:
            if len(self._entries) >= MAX_DOCUMENTS:
                if self._entries[0].doc.has_edits:
                    raise ApiError(409, FULL)
                del self._entries[0]
            self._entries.append(Entry(data, doc))

    def documents(self) -> list[Document]:
        with self._lock:
            return [e.doc for e in self._entries]

    def get(self, doc_id: str) -> Entry:
        with self._lock:
            return self._entries[self._index(doc_id)]

    def page(self, doc_id: str, n: str) -> tuple[Entry, Page]:
        with self._lock:
            entry = self._entries[self._index(doc_id)]
            return entry, entry.doc.pages[page_number(n, len(entry.doc.pages)) - 1]

    def begin_read(self, doc_id: str, n: str) -> tuple[Entry, int]:
        with self._lock:
            i = self._index(doc_id)
            doc = self._entries[i].doc
            no = page_number(n, len(doc.pages))
            if self._read is not None:
                raise ApiError(409, BUSY)
            page = doc.pages[no - 1]
            self._read = (doc_id, no, page.status)
            return self._set_page(i, no, dataclasses.replace(page, status="reading")), no

    def finish_read(self, page: Page, engine: Engine) -> Page:
        with self._lock:
            if self._read is None:
                raise RuntimeError("no page read is running")
            doc_id, no, _ = self._read
            self._read = None
            i = self._index(doc_id)
            merged = keep_reviewed(self._entries[i].doc.pages[no - 1], page)
            entry = self._set_page(i, no, merged)
            if engine != NO_READER:
                doc = dataclasses.replace(entry.doc, engine=engine)
                self._entries[i] = dataclasses.replace(entry, doc=doc)
            return merged

    def abort_read(self) -> None:
        with self._lock:
            read, self._read = self._read, None
            if read is None:
                return
            doc_id, no, before = read
            try:
                i = self._index(doc_id)
            except ApiError:
                return
            page = self._entries[i].doc.pages[no - 1]
            if page.status == "reading":
                self._set_page(i, no, dataclasses.replace(page, status=before))

    def edit_line(self, doc_id: str, line_id: str, final_text: str) -> Line:
        with self._lock:
            i = self._index(doc_id)
            doc = self._entries[i].doc
            for pn, page in enumerate(doc.pages, 1):
                for k, line in enumerate(page.lines):
                    if line.id == line_id:
                        status = "confirmed" if final_text == line.text else "edited"
                        new = dataclasses.replace(line, final_text=final_text, status=status)
                        lines = (*page.lines[:k], new, *page.lines[k + 1 :])
                        entry = self._set_page(i, pn, dataclasses.replace(page, lines=lines))
                        doc = dataclasses.replace(entry.doc, has_edits=True)
                        self._entries[i] = dataclasses.replace(entry, doc=doc)
                        return new
            raise ApiError(404, LINE_GONE)

    def delete(self, doc_id: str) -> None:
        with self._lock:
            del self._entries[self._index(doc_id)]

    def clear(self) -> None:
        with self._lock:
            self._entries = []
