"""The 50-megapixel cap on the images a PDF draws: every image's header size, and its mask's,
is checked before any page is drawn or any image is decoded."""

import time
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
import pytest
from sample_files import PAGE_IMAGE_LINE as PAGE_IMAGE
from sample_files import PDF_IMAGE_LINE as PDF_IMAGE
from sample_files import UNCHECKED_LINE as UNCHECKED
from sample_files import assert_within_a_second, image_pdf, pdf_stream, pdfium_image_sizes, raw_pdf

from folio_ocr.errors import InputError
from folio_ocr.ingest import MAX_PIXELS, open_pdf
from folio_ocr.pdf_images import Declared, largest_declared_image

FIXTURES = Path(__file__).parent / "fixtures"
BOMB = (20000, 20000)
BOMB_SIZE = b"/Width 20000 /Height 20000"


def refusal(data):
    with pytest.raises(InputError) as info:
        open_pdf(data)
    return str(info.value)


def opens(data):
    open_pdf(data).close()
    return True


@pytest.fixture
def no_drawing(monkeypatch):
    """Fail the test if anything draws a page or decodes an image."""

    def boom(*_args, **_kwargs):
        raise AssertionError("must not draw a page or decode an image")

    monkeypatch.setattr(pdfium.PdfPage, "render", boom)
    for name in (
        "FPDF_RenderPageBitmap",
        "FPDF_RenderPageBitmapWithMatrix",
        "FPDFImageObj_GetBitmap",
        "FPDFImageObj_GetRenderedBitmap",
        "FPDFImageObj_GetImageDataDecoded",
    ):
        monkeypatch.setattr(pdfium_raw, name, boom)


# ---- refusals ----


@pytest.mark.parametrize("place", ["page", "form", "inline"])
def test_page_image_over_the_cap_is_refused_before_drawing(no_drawing, place):
    assert refusal(image_pdf(*BOMB, place=place)) == PAGE_IMAGE


@pytest.mark.parametrize("place", ["annot", "pattern"])
def test_image_drawn_outside_the_page_content_is_refused(no_drawing, place):
    assert refusal(image_pdf(*BOMB, place=place)) == PDF_IMAGE


@pytest.mark.parametrize("key", ["SMask", "Mask"])
def test_mask_over_the_cap_is_refused(no_drawing, key):
    assert refusal(image_pdf(0, 0, mask=(key, *BOMB))) == PDF_IMAGE


def test_page_size_cap_still_comes_first(no_drawing):
    """An oversized page is refused on its size alone, before any page is loaded."""
    data = raw_pdf(
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 14400 14400] >>",
    )
    assert refusal(data) == "page 1 is too large to read (over 50 megapixels at 300 dpi)"


# ---- the boundary: exactly 50 MP passes, one pixel more is refused ----


def test_boundary_numbers():
    assert 5000 * 10000 == MAX_PIXELS and 57 * 877193 == MAX_PIXELS + 1


@pytest.mark.parametrize("place", ["page", "form", "inline", "annot", "pattern"])
def test_image_of_exactly_50_megapixels_passes(place):
    assert opens(image_pdf(5000, 10000, place=place))


@pytest.mark.parametrize(
    ("place", "line"),
    [("page", PAGE_IMAGE), ("inline", PAGE_IMAGE), ("annot", PDF_IMAGE)],
)
def test_one_pixel_over_50_megapixels_is_refused(place, line):
    assert refusal(image_pdf(57, 877193, place=place)) == line


@pytest.mark.parametrize("key", ["SMask", "Mask"])
def test_mask_boundary(key):
    assert opens(image_pdf(0, 0, mask=(key, 5000, 10000)))
    assert refusal(image_pdf(0, 0, mask=(key, 57, 877193))) == PDF_IMAGE


def test_every_fixture_pdf_still_opens():
    pdfs = sorted(FIXTURES.glob("*.pdf"))
    assert pdfs and all(opens(p.read_bytes()) for p in pdfs)


# ---- the stream dictionaries are read the way PDFium reads them ----

READ_ALIKE = {
    "plain": b"/Width 300 /Height 200",
    "escaped names": b"/W#69dth 300 /H#65ight 200",
    "comment before a value": b"/Width %c\n 300 /Height 200",
    "indirect value": b"/Width 6 0 R /Height 200",
    "string in a key position": b"( /Width 300 /Height 200 )",
    "hex string in a key position": b"< /Width 300 /Height 200 >",
    "last of two equal keys": b"/Width 999 /Height 200 /Width 300",
    "real number": b"/Width 300.9 /Height 200",
    "zero-padded number": b"/Width 0000000000300 /Height 200",
    "zero-padded reference": b"/Width 0000000000006 0 R /Height 200",
    "name inside an array": b"/Decode [0 1 /Width 999] /Width 300 /Height 200",
    "empty name takes a value": b"/ ( /X <) /Width 300 /Height 200 /Y >",
    "empty name before a size": b"/ /Width 999 /Width 300 /Height 200",
    "empty name in a nested dictionary": b"/D << / (>>) /Width 999 >> /Width 300 /Height 200",
}


@pytest.mark.parametrize("size", READ_ALIKE.values(), ids=READ_ALIKE.keys())
def test_sizes_are_read_as_pdfium_reads_them(size):
    data = image_pdf(0, 0, size=size, objects=[b"300"])
    assert pdfium_image_sizes(data) == [(300, 200)]
    assert largest_declared_image(data) == Declared(300 * 200, True)


def declared(data):
    return largest_declared_image(data).largest


def obj(number, body):
    return b"%d 0 obj\n%s\nendobj\n" % (number, body)


def test_only_stream_dictionaries_count():
    assert declared(obj(1, b"<< /Width 20000 /Height 20000 >>")) == 0


def test_a_nested_dictionary_is_not_the_stream_size():
    stream = pdf_stream(b"/DecodeParms << /Width 20000 /Height 20000 >> /Width 10 /Height 10")
    assert declared(obj(1, stream)) == 100


def test_stream_contents_are_not_read():
    stream = pdf_stream(b"", b"<< /Width 20000 /Height 20000 >>\nstream\n")
    assert declared(obj(1, stream)) == 0


def test_a_stream_hidden_in_a_string_cannot_hide_the_next_object():
    """A fake stream inside a string, whose /Length lands on the next object's endstream,
    must not skip that object."""
    real = obj(2, pdf_stream(b"/Width 20000 /Height 20000", b"x"))
    head = b"1 0 obj\n(<< /Length %010d >>\nstream\n)\nendobj\n"
    contents_start = len(head % 0) - len(b")\nendobj\n")
    fake_length = len(head % 0) + real.index(b"endstream") - contents_start
    data = head % fake_length + real
    assert data[contents_start + fake_length :].startswith(b"endstream")
    assert declared(data) == 20000 * 20000


def test_a_reference_to_a_number_object_is_followed():
    data = obj(1, pdf_stream(b"/Width 2 0 R /Height 20000")) + obj(2, b"20000")
    assert declared(data) == 20000 * 20000


def test_a_zero_padded_size_is_read_at_its_value():
    padded = b"00000000020000"
    assert declared(obj(1, pdf_stream(b"/Width %s /Height 20000" % padded))) == 20000 * 20000
    data = obj(1, pdf_stream(b"/Width 2 0 R /Height 20000")) + obj(2, padded)
    assert declared(data) == 20000 * 20000


def test_negative_or_zero_sizes_hold_no_pixels():
    data = obj(1, pdf_stream(b"/Width -20000 /Height -20000")) + obj(2, pdf_stream(b"/Width 0"))
    assert declared(data) == 0


# ---- sizes that cannot be read are refused with their own line ----


def test_a_size_whose_object_is_missing_cannot_be_checked(no_drawing):
    data = image_pdf(0, 0, size=b"/Width 9 0 R /Height 200")
    assert largest_declared_image(data) == Declared(0, False)
    assert refusal(data) == UNCHECKED


def test_a_measured_image_over_the_cap_wins_over_an_unread_size():
    data = obj(1, pdf_stream(b"/Width 9 0 R /Height 1")) + obj(2, pdf_stream(BOMB_SIZE))
    assert largest_declared_image(data) == Declared(20000 * 20000, False)


def test_a_size_that_is_a_reference_to_a_reference_cannot_be_checked():
    data = obj(1, pdf_stream(b"/Width 2 0 R /Height 1")) + obj(2, b"3 0 R") + obj(3, b"2 0 R")
    assert largest_declared_image(data) == Declared(0, False)


# ---- reading hostile files stays bounded ----

HOSTILE = {
    "dict opens": b"<<",
    "parens": b"(",
    "brackets": b"[",
    "names": b"/A ",
    "numbers": b"0 ",
    "size keys": b"/Width 1 ",
    "escaped size keys": b"/W#69dth 1 ",
    "strings": b"<< /A (",
    "nested strings": b"<< /A ((",
    "nested arrays": b"<< /A [[",
    "nested dicts": b"<< /A <<",
    "array values": b"/A [] ",
    "objects": b"1 0 obj << /A [",
    "streams": b">> stream\n",
    "referenced sizes": b"<< /Width 1 0 R >> stream\nendstream ",
    "obj keywords": b"1 0 obj (",
    "empty objects": b"obj<<>>",
    "headers in a comment": b"%1 0 obj ",
    "a comment after obj": b"obj %",
}


@pytest.mark.parametrize("unit", HOSTILE.values(), ids=HOSTILE.keys())
def test_reading_a_hostile_file_is_bounded(unit):
    """Each read ends well within a second with a verdict: a clean pass or "cannot tell"."""
    data = b"%PDF-1.7\n1 0 obj\n<<" + unit * (2 * 1024 * 1024 // len(unit))
    start = time.perf_counter()
    result = largest_declared_image(data)
    assert_within_a_second(start)
    assert result.largest == 0


# ---- the reading budgets ----

SOUP = b"%PDF-1.7\n" + b"7 0 obj << " * 100_000 + b">>"


def many_sizes(count):
    return b"1 0 obj\n<<" + b"/Width 1 " * count + b">>\nstream\n"


def test_a_dictionary_within_the_step_budget_is_read_completely():
    assert largest_declared_image(many_sizes(5_000)).complete is True


def test_a_dictionary_beyond_the_step_budget_cannot_be_checked():
    assert largest_declared_image(many_sizes(20_000)).complete is False


def test_overlapping_object_dictionaries_beyond_the_byte_budget_cannot_be_checked():
    assert largest_declared_image(SOUP).complete is False


def test_a_budget_overrun_inside_an_unused_stream_refuses_the_pdf(no_drawing):
    data = image_pdf(300, 200, objects=[pdf_stream(b"", SOUP)])
    assert largest_declared_image(data).complete is False
    assert refusal(data) == UNCHECKED
