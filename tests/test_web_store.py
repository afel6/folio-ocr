"""The in-memory store of open documents behind the local web service."""

import dataclasses
from datetime import UTC, datetime

import pytest

from folio_ocr.document import Document, Engine, Line, Page, Source
from folio_ocr.web import store as st
from folio_ocr.web.store import NO_READER, ApiError, Store


def mk_line(id_: str, text: str = "t", status: str = "needs_review", final: str | None = None):
    return Line(
        id=id_,
        box=(0.0, 0.0, 1.0, 1.0),
        text=text,
        direction="ltr",
        script="latin",
        reader="onnxtr",
        confidence=0.5,
        confidence_method="reader",
        flags=(),
        status=status,
        final_text=text if final is None else final,
    )


def mk_page(index: int, lines=(), status: str = "unread") -> Page:
    return Page(
        index=index,
        width=10,
        height=10,
        render_dpi=None,
        rotation_applied=0.0,
        transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        origin="ocr",
        direction="ltr",
        status=status,
        flags=(),
        unread_regions=(),
        lines=tuple(lines),
    )


def mk_doc(doc_id: str, pages: int = 2, lines_per_page: int = 2, status="unread") -> Document:
    pgs = tuple(
        mk_page(
            i,
            [mk_line(f"p{i}-l{k}", f"text{i}{k}") for k in range(1, lines_per_page + 1)],
            status,
        )
        for i in range(1, pages + 1)
    )
    return Document(
        id=doc_id,
        source=Source("a.pdf", "0" * 64, "pdf", pages),
        languages="en",
        engine=NO_READER,
        pages=pgs,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        has_edits=False,
    )


def other_engine() -> Engine:
    return dataclasses.replace(NO_READER, reader="tesseract", detector="x")


def ids(store: Store) -> list[str]:
    return [d.id for d in store.documents()]


def gone(excinfo, status: int, message: str) -> None:
    assert excinfo.value.status == status
    assert str(excinfo.value) == message


@pytest.mark.parametrize("text", ["1", "20"])
def test_page_number_accepts(text):
    assert st.page_number(text, 20) == int(text)


@pytest.mark.parametrize(
    "text", ["0", "21", "01", "+1", " 1", "1 ", "١", "", "1.0", "abc", "9" * 5000]
)
def test_page_number_rejects(text):
    with pytest.raises(ApiError) as e:
        st.page_number(text, 20)
    gone(e, 404, st.PAGE_GONE)


def test_four_documents_evict_the_oldest():
    s = Store()
    for n in "abc":
        s.add(b"", mk_doc(n))
    s.add(b"", mk_doc("d"))
    assert ids(s) == ["b", "c", "d"]


def test_edited_oldest_blocks_the_fourth():
    s = Store()
    for n in "abc":
        s.add(b"", mk_doc(n))
    s.edit_line("a", "p1-l1", "changed")
    with pytest.raises(ApiError) as e:
        s.add(b"", mk_doc("d"))
    gone(e, 409, st.FULL)
    assert ids(s) == ["a", "b", "c"]


def test_confirmed_oldest_blocks_the_fourth():
    s = Store()
    for n in "abc":
        s.add(b"", mk_doc(n))
    s.edit_line("a", "p1-l1", "text11")
    with pytest.raises(ApiError) as e:
        s.add(b"", mk_doc("d"))
    assert e.value.status == 409


def test_all_edited_blocks_the_fourth():
    s = Store()
    for n in "abc":
        s.add(b"", mk_doc(n))
        s.edit_line(n, "p1-l1", "changed")
    with pytest.raises(ApiError):
        s.add(b"", mk_doc("d"))
    assert ids(s) == ["a", "b", "c"]


def test_younger_edits_do_not_block_eviction():
    s = Store()
    for n in "abc":
        s.add(b"", mk_doc(n))
    s.edit_line("b", "p1-l1", "changed")
    s.edit_line("c", "p1-l1", "changed")
    s.add(b"", mk_doc("d"))
    assert ids(s) == ["b", "c", "d"]


def test_edit_status_rules():
    s = Store()
    s.add(b"", mk_doc("a"))
    same = s.edit_line("a", "p1-l1", "text11")
    assert (same.status, same.final_text) == ("confirmed", "text11")
    assert s.get("a").doc.has_edits is True
    first = s.edit_line("a", "p1-l2", "new")
    assert first.status == "edited"
    assert s.edit_line("a", "p1-l2", "new").status == "edited"
    assert s.edit_line("a", "p1-l2", "text12").status == "confirmed"
    stored = s.get("a").doc.pages[0].lines[1]
    assert s.edit_line("a", "p1-l2", "text12") == stored


def test_edit_finds_line_on_any_page_without_mutating_old_objects():
    s = Store()
    doc = mk_doc("a")
    s.add(b"", doc)
    s.edit_line("a", "p2-l2", "zzz")
    assert s.get("a").doc.pages[1].lines[1].final_text == "zzz"
    assert doc.pages[1].lines[1].final_text == "text22"
    assert doc.has_edits is False


def test_edit_unknown_document_or_line():
    s = Store()
    s.add(b"", mk_doc("a"))
    with pytest.raises(ApiError) as e:
        s.edit_line("nope", "p1-l1", "x")
    gone(e, 404, st.DOCUMENT_GONE)
    with pytest.raises(ApiError) as e:
        s.edit_line("a", "p9-l9", "x")
    gone(e, 404, st.LINE_GONE)


def test_get_page_errors_check_document_first():
    s = Store()
    s.add(b"data", mk_doc("a"))
    with pytest.raises(ApiError) as e:
        s.page("nope", "99")
    gone(e, 404, st.DOCUMENT_GONE)
    with pytest.raises(ApiError) as e:
        s.page("a", "3")
    gone(e, 404, st.PAGE_GONE)
    entry, page = s.page("a", "2")
    assert entry.data == b"data" and page.index == 2


def test_second_begin_read_is_busy():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.add(b"", mk_doc("b"))
    s.begin_read("a", "1")
    for args in (("a", "1"), ("a", "2"), ("b", "1")):
        with pytest.raises(ApiError) as e:
            s.begin_read(*args)
        gone(e, 409, st.BUSY)


def test_begin_read_404_before_busy():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    with pytest.raises(ApiError) as e:
        s.begin_read("zz", "1")
    assert e.value.status == 404
    with pytest.raises(ApiError) as e:
        s.begin_read("a", "9")
    gone(e, 404, st.PAGE_GONE)


def test_reading_page_keeps_lines():
    s = Store()
    s.add(b"", mk_doc("a"))
    entry, n = s.begin_read("a", "2")
    assert n == 2
    page = entry.doc.pages[1]
    assert page.status == "reading"
    assert page.lines == mk_doc("a").pages[1].lines
    assert s.get("a").doc.pages[1].status == "reading"
    assert s.get("a").doc.pages[0].status == "unread"


def test_finish_read_replaces_lines_but_keeps_edits_made_meanwhile():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.edit_line("a", "p1-l1", "mine")
    new = mk_page(1, [mk_line("p1-l1", "other"), mk_line("p1-l2", "fresh")], "read")
    merged = s.finish_read(new, NO_READER)
    lines = s.get("a").doc.pages[0].lines
    assert merged == s.get("a").doc.pages[0]
    assert (lines[0].final_text, lines[0].status) == ("mine", "edited")
    assert (lines[1].text, lines[1].status) == ("fresh", "needs_review")
    assert s.get("a").doc.pages[0].status == "read"


def test_finish_read_appends_reviewed_line_missing_from_new_page():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.edit_line("a", "p1-l1", "mine")
    new = mk_page(1, [mk_line("p1-l7", "x"), mk_line("p1-l8", "y")], "read")
    merged = s.finish_read(new, NO_READER)
    assert [ln.id for ln in merged.lines] == ["p1-l7", "p1-l8", "p1-l1"]


def test_keep_reviewed_appends_in_current_order():
    cur = mk_page(
        1,
        [
            mk_line("a", status="confirmed"),
            mk_line("b"),
            mk_line("c", status="edited"),
            mk_line("d", status="edited"),
        ],
    )
    read = mk_page(1, [mk_line("d", "new"), mk_line("e")], "read")
    out = st.keep_reviewed(cur, read)
    assert [ln.id for ln in out.lines] == ["d", "e", "a", "c"]
    assert out.lines[0] == cur.lines[3]
    assert out.status == "read"


def test_abort_restores_previous_status():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.abort_read()
    assert s.get("a").doc.pages[0].status == "unread"
    s.begin_read("a", "1")
    s.finish_read(mk_page(1, [], "read"), NO_READER)
    s.begin_read("a", "1")
    s.abort_read()
    assert s.get("a").doc.pages[0].status == "read"


def test_abort_without_read_is_noop_and_slot_frees_after_finish():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.abort_read()
    s.begin_read("a", "1")
    s.finish_read(mk_page(1, [], "read"), NO_READER)
    s.begin_read("a", "2")


def test_finish_without_read_is_a_programming_error():
    s = Store()
    s.add(b"", mk_doc("a"))
    with pytest.raises(RuntimeError):
        s.finish_read(mk_page(1), NO_READER)


def test_abort_leaves_page_alone_when_no_longer_reading():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.delete("a")
    s.add(b"", mk_doc("a"))
    s.abort_read()
    assert s.get("a").doc.pages[0].status == "unread"
    s.begin_read("a", "1")


def test_abort_after_delete_is_a_no_op_and_frees_the_slot():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.delete("a")
    s.abort_read()
    assert ids(s) == []
    s.add(b"", mk_doc("b"))
    s.begin_read("b", "1")


def test_finish_after_delete_is_404_and_frees_slot():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.delete("a")
    with pytest.raises(ApiError) as e:
        s.finish_read(mk_page(1, [], "read"), other_engine())
    gone(e, 404, st.DOCUMENT_GONE)
    assert ids(s) == []
    s.add(b"", mk_doc("b"))
    s.begin_read("b", "1")


def test_eviction_during_read_makes_finish_404():
    s = Store()
    for n in "abc":
        s.add(b"", mk_doc(n))
    s.begin_read("a", "1")
    s.add(b"", mk_doc("d"))
    with pytest.raises(ApiError) as e:
        s.finish_read(mk_page(1, [], "read"), NO_READER)
    assert e.value.status == 404


def test_engine_rules():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.begin_read("a", "1")
    s.finish_read(mk_page(1, [], "read"), other_engine())
    assert s.get("a").doc.engine == other_engine()
    s.begin_read("a", "2")
    s.finish_read(mk_page(2, [], "read"), NO_READER)
    assert s.get("a").doc.engine == other_engine()


def test_delete_unknown_is_404_and_clear_empties():
    s = Store()
    s.add(b"", mk_doc("a"))
    s.add(b"", mk_doc("b"))
    with pytest.raises(ApiError) as e:
        s.delete("zz")
    gone(e, 404, st.DOCUMENT_GONE)
    s.delete("a")
    assert ids(s) == ["b"]
    s.clear()
    assert s.documents() == []
    with pytest.raises(ApiError) as e:
        s.get("b")
    gone(e, 404, st.DOCUMENT_GONE)


def test_messages_are_the_specified_lines():
    assert st.MAX_DOCUMENTS == 3
    assert st.DOCUMENT_GONE == "no such document; it may have been deleted or closed to make room"
    assert st.PAGE_GONE == "no such page in this document"
    assert st.LINE_GONE == "no such line in this document"
    assert st.BUSY == "a page is being read; try again when it finishes"
    assert st.FULL == (
        "three documents are open and the oldest has edits; export it, then delete it to make room"
    )
