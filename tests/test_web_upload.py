import asyncio
import hashlib
import tempfile
from datetime import UTC

import pytest
from PIL import Image
from sample_files import encrypted, pdf_bytes, png_header, save, tiff

from folio_ocr import ocr_setup
from folio_ocr.errors import InputError
from folio_ocr.web import upload
from folio_ocr.web.upload import BODY_LIMIT, Upload, new_document, parse_form, read_body

CT = "multipart/form-data; boundary=B"
PNG = save(Image.new("L", (8, 8)), "PNG")


def part(name, value, filename=None):
    disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
    return b"--B\r\nContent-Disposition: " + disp.encode() + b"\r\n\r\n" + value + b"\r\n"


def form(data=PNG, filename="x.png", languages=b"en"):
    body = part("file", data, filename) if data is not None else b""
    if languages is not None:
        body += part("languages", languages)
    return body + b"--B--\r\n"


def upload_of(data, languages="en", filename="x.png"):
    return Upload(filename=filename, data=data, languages=languages)


def test_parse_form_reads_file_and_languages():
    up = parse_form(CT, form())
    assert (up.filename, up.data, up.languages) == ("x.png", PNG, "en")
    assert "data=" not in repr(up)


def test_parse_form_keeps_languages_untouched_and_ignores_other_parts():
    body = part("other", b"1") + part("other", b"2", "y.txt") + form(languages=b" en ")
    up = parse_form(CT, body)
    assert up.languages == " en "
    assert up.data == PNG


def test_languages_missing_or_repeated_is_none():
    assert parse_form(CT, form(languages=None)).languages is None
    body = form(languages=b"en")[: -len(b"--B--\r\n")] + part("languages", b"ar") + b"--B--\r\n"
    assert parse_form(CT, body).languages is None


@pytest.mark.parametrize(
    ("ctype", "body", "message"),
    [
        ("application/json", b"{}", upload.NOT_MULTIPART),
        ("multipart/form-data", form(), upload.NOT_MULTIPART),
        (CT, form(data=None), upload.NO_FILE),
        (CT, part("file", PNG) + b"--B--\r\n", upload.NO_FILE),
        (
            CT,
            form()[: -len(b"--B--\r\n")] + part("file", PNG, "z.png") + b"--B--\r\n",
            upload.TWO_FILES,
        ),
    ],
)
def test_parse_form_refusals(ctype, body, message):
    with pytest.raises(InputError) as err:
        parse_form(ctype, body)
    assert str(err.value) == message


@pytest.mark.parametrize("body", [b"garbage", form()[:60], b"--B\r\nbad header\r\n\r\nx"])
def test_parse_form_damaged_body_is_input_error(body):
    try:
        up = parse_form(CT, body)
    except InputError:
        return
    pytest.fail(f"accepted {up!r}")


def test_non_utf8_filename_is_replaced():
    body = b'--B\r\nContent-Disposition: form-data; name="file"; filename="a\xff.png"\r\n\r\n'
    body += (
        PNG + b'\r\n--B\r\nContent-Disposition: form-data; name="languages"\r\n\r\nen\r\n--B--\r\n'
    )
    assert "�" in parse_form(CT, body).filename


def test_file_content_that_looks_like_a_part_header_is_untouched():
    content = (
        PNG
        + b'Content-Disposition: form-data; name="x"; filename="a\xff.png"\r\n\r\n'
        + b"\r\n--\r\n--Bx\r\n--"
    )
    body = b'--B\r\nContent-Disposition: form-data; name="file"; filename="a\xff.png"\r\n\r\n'
    body += content + b"\r\n--B--\r\n"
    up = parse_form(CT, body)
    assert up.data == content
    assert up.filename == "a�.png"


@pytest.mark.parametrize(
    "body",
    [
        b'--B\r\nContent-Disposition: form-data; filename="x.png"\r\n\r\nabc\r\n--B--\r\n',
        b"--B\r\nContent-Type: text/plain\r\n\r\nabc\r\n--B--\r\n",
        form()[: -len(b"--B--\r\n")],
        form()[: -len(b"\r\n--B--\r\n")] + b"\r\n",
    ],
)
def test_incomplete_or_nameless_parts_are_bad_form(body):
    with pytest.raises(InputError) as err:
        parse_form(CT, body)
    assert str(err.value) == upload.BAD_FORM


def test_unsafe_filename_is_sanitised():
    doc = new_document(upload_of(PNG, filename="../../etc/pa\r\nss<>.pdf"))
    name = doc.source.filename
    assert not any(c in name for c in "/\\\r\n<>")
    assert not name.startswith(".")


def test_document_record():
    doc = new_document(upload_of(PNG))
    assert doc.source.sha256 == hashlib.sha256(PNG).hexdigest()
    assert (doc.source.kind, doc.source.page_count) == ("image", 1)
    assert doc.created_at.tzinfo is UTC
    assert doc.engine == ocr_setup.engine(None)
    assert doc.has_edits is False
    assert doc.languages == "en"
    assert len(doc.id) == 32
    (page,) = doc.pages
    assert page == upload.unread_page(1)
    assert (page.status, page.lines, page.width) == ("unread", (), 0)


@pytest.mark.parametrize("lang", ["en", "ar", "en+ar"])
def test_languages_accepted(lang):
    assert new_document(upload_of(PNG, lang)).languages == lang


@pytest.mark.parametrize("lang", [None, "EN", " en", "en ", "fr", ""])
def test_languages_refused(lang):
    with pytest.raises(InputError) as err:
        new_document(upload_of(PNG, lang))
    assert str(err.value) == upload.BAD_LANGUAGES


def test_pdf_pages():
    doc = new_document(upload_of(pdf_bytes([(612, 792)] * 20), filename="a.pdf"))
    assert doc.source.kind == "pdf"
    assert len(doc.pages) == 20
    assert all(p.status == "unread" and p.lines == () for p in doc.pages)
    assert [p.index for p in doc.pages] == list(range(1, 21))


@pytest.mark.parametrize(
    "data",
    [
        pdf_bytes([(612, 792)] * 21),
        pdf_bytes([(14400, 14400)]),
        encrypted(),
        encrypted("u"),
        b"hello",
        tiff([(8, 8)] * 21),
        png_header(10000, 6000),
    ],
)
def test_refused_files(data):
    with pytest.raises(InputError):
        new_document(upload_of(data))


def test_tiff_frames_counted():
    assert new_document(upload_of(tiff([(8, 8)] * 3))).source.page_count == 3


def padded_png(size):
    return PNG + bytes(size - len(PNG))


def test_file_of_exactly_25_mb_stays_in_memory(monkeypatch):
    import python_multipart.multipart as mp

    def fail(*a, **k):
        raise AssertionError("temporary file made")

    for name in ("NamedTemporaryFile", "mkstemp", "TemporaryFile"):
        monkeypatch.setattr(tempfile, name, fail)
    assert mp.tempfile is tempfile
    data = padded_png(26_214_400)
    up = parse_form(CT, form(data))
    assert len(up.data) == 26_214_400
    doc = new_document(up)
    assert doc.source.page_count == 1


def test_file_one_byte_over_is_refused():
    up = parse_form(CT, form(padded_png(26_214_401)))
    with pytest.raises(InputError, match="larger than 25 MB"):
        new_document(up)


class Chunks:
    def __init__(self, *chunks):
        self.chunks = chunks
        self.touched = False

    def __aiter__(self):
        self.touched = True
        return self._gen()

    async def _gen(self):
        for c in self.chunks:
            yield c


def test_read_body_refuses_by_content_length_before_reading():
    chunks = Chunks(b"x")
    with pytest.raises(InputError, match="larger than 25 MB"):
        asyncio.run(read_body(chunks, "26279937"))
    assert not chunks.touched


def test_read_body_refuses_streamed_excess():
    with pytest.raises(InputError, match="larger than 25 MB"):
        asyncio.run(read_body(Chunks(b"a" * BODY_LIMIT, b"b"), None))


def test_read_body_returns_exactly_the_limit():
    assert asyncio.run(read_body(Chunks(b"a" * 10, b"b" * (BODY_LIMIT - 10)), "5")) == (
        b"a" * 10 + b"b" * (BODY_LIMIT - 10)
    )


def test_read_body_ignores_a_non_numeric_length():
    assert asyncio.run(read_body(Chunks(b"ab", b"c"), "abc")) == b"abc"
