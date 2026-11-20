"""Hostile PDFs: reference chains and cycles, lying /Length values, damaged cross-reference
tables, truncated files, odd number spellings, deep nesting and sizes held in object streams
all end in a known one-line refusal or a clean open, and the size reader always returns."""

import re
import time

import pytest
from sample_files import PAGE_IMAGE_LINE as PAGE_IMAGE
from sample_files import PDF_IMAGE_LINE as PDF_IMAGE
from sample_files import UNCHECKED_LINE as UNCHECKED
from sample_files import (
    _image_stream,
    _placed,
    assert_within_a_second,
    image_pdf,
    objstm_pdf,
    pdf_stream,
    pdfium_image_sizes,
    raw_pdf,
)

from folio_ocr.errors import InputError
from folio_ocr.ingest import open_pdf
from folio_ocr.pdf_images import Declared, largest_declared_image

DAMAGED = "the PDF could not be opened; it may be damaged"
BOMB = 20000 * 20000


def read(data):
    """The reader's verdict, which must come back within a second."""
    start = time.perf_counter()
    result = largest_declared_image(data)
    assert_within_a_second(start)
    return result


def verdict(data):
    """What open_pdf does: "open", or the refusal line."""
    start = time.perf_counter()
    try:
        open_pdf(data).close()
        return "open"
    except InputError as err:
        return str(err)
    finally:
        assert_within_a_second(start)


def sized(width):
    return image_pdf(0, 0, size=b"/Width %s /Height 200" % width, objects=[b"300"])


# ---- references ----


def test_a_reference_chain_200_deep_cannot_be_checked():
    chain = [b"%d 0 R" % (n + 1) for n in range(7, 207)] + [b"300"]
    data = image_pdf(0, 0, size=b"/Width 6 0 R /Height 200", objects=[b"7 0 R", *chain])
    assert read(data) == Declared(0, False)
    assert verdict(data) == UNCHECKED


def test_cyclic_references_cannot_be_checked():
    data = image_pdf(0, 0, size=b"/Width 6 0 R /Height 200", objects=[b"7 0 R", b"6 0 R"])
    assert read(data) == Declared(0, False)
    assert verdict(data) == UNCHECKED


# ---- /Length ----


@pytest.mark.parametrize("length", [b"99999999999999999999", b"1000000000000", b"-5", b"99 0 R"])
def test_a_lying_length_does_not_hide_an_oversized_image(length):
    data = image_pdf(0, 0, size=b"/Width 20000 /Height 20000 /Length " + length)
    assert read(data).largest == BOMB
    assert verdict(data) == PAGE_IMAGE


# ---- damaged files ----


def zero_offsets(data):
    start = data.rindex(b"xref\n0 ")
    return data[:start] + re.sub(rb"\d{10}", b"0000000000", data[start:])


def chop_xref(data):
    return data[: data.rindex(b"xref\n0 ")]


def test_zeroed_offsets_still_find_the_oversized_image():
    data = zero_offsets(image_pdf(20000, 20000))
    assert read(data).largest == BOMB
    assert verdict(data) == PAGE_IMAGE


def test_a_missing_table_is_refused_as_damaged_even_with_an_oversized_image():
    data = chop_xref(image_pdf(20000, 20000))
    assert read(data).largest == BOMB
    assert verdict(data) == DAMAGED


def test_damage_to_a_small_image_never_gives_an_image_line():
    small = image_pdf(300, 200)
    assert verdict(zero_offsets(small)) == "open"
    assert verdict(chop_xref(small)) == DAMAGED


@pytest.mark.parametrize("cut", ["25", "50", "75", "endstream"])
def test_a_truncated_file_is_refused_as_damaged(cut):
    data = image_pdf(20000, 20000)
    end = data.index(b"endstream") if cut == "endstream" else len(data) * int(cut) // 100
    read(data[:end])
    assert verdict(data[:end]) == DAMAGED


# ---- number spellings: PDFium's view against the reader's ----

PASSES = {
    "1e999": b"1e999",
    "negative zero": b"-0",
    "plus point five": b"+.5",
    "two points": b"1.2.3",
    "string": b"(300)",
}


@pytest.mark.parametrize("width", PASSES.values(), ids=PASSES.keys())
def test_a_size_pdfium_reads_as_zero_pixels_passes(width):
    data = sized(width)
    assert pdfium_image_sizes(data)[0][0] <= 1
    assert read(data).largest < 1000
    assert verdict(data) == "open"


BEYOND_32_BITS = {
    "past 2**32": (b"4294967296", 0),
    "twenty digits": (b"99999999999999999999", 0),
    "signed past 2**31": (b"+2147483648", 0),
    "unsigned past 2**31": (b"2147483648", 2147483648),
}


@pytest.mark.parametrize(("width", "seen"), BEYOND_32_BITS.values(), ids=BEYOND_32_BITS.keys())
def test_an_integer_width_beyond_32_bits_holds_no_pixels(width, seen):
    """PDFium reads such a width as 0, or as a negative number it reports unsigned."""
    data = sized(width)
    assert pdfium_image_sizes(data) == [(seen, 200)]
    assert read(data) == Declared(0, True)
    assert verdict(data) == "open"


def test_a_real_width_beyond_32_bits_is_saturated_and_refused():
    data = sized(b"99999999999.0")
    assert pdfium_image_sizes(data) == [(2**31 - 1, 200)]
    assert read(data) == Declared((2**31 - 1) * 200, True)
    assert verdict(data) == PAGE_IMAGE


def test_an_oversized_height_with_a_valid_width_is_refused_by_both():
    data = image_pdf(0, 0, size=b"/Width 300 /Height 2000000")
    assert pdfium_image_sizes(data) == [(300, 2000000)]
    assert verdict(data) == PAGE_IMAGE


# ---- nesting ----


@pytest.mark.parametrize("unit", [b"[", b"<<", b"("], ids=["arrays", "dictionaries", "strings"])
def test_deep_nesting_is_read_without_recursion_errors(unit):
    result = read(b"%PDF-1.7\n<< /A " + unit * 100_000)
    assert result.largest == 0


# ---- sizes the reader cannot see ----


def test_a_size_held_in_an_object_stream_cannot_be_checked():
    data = objstm_pdf()
    assert pdfium_image_sizes(data) == [(300, 200)]
    assert read(data) == Declared(0, False)
    assert verdict(data) == UNCHECKED


# ---- dictionaries and numbers placed to mislead the reader ----


def smask_pdf(smask_size, *objects):
    """A page drawing a 10 x 10 image whose /SMask is object 7 (a stream stating `smask_size`),
    with `objects` placed as object 6."""
    image = _image_stream(b"/Width 10 /Height 10", None, b" /SMask 7 0 R")
    return raw_pdf(*_placed("page", image, [*objects, _image_stream(smask_size, None)]))


def test_a_string_holding_an_open_hex_string_does_not_hide_the_mask():
    data = smask_pdf(b"/Width 20000 /Height 20000", b"(<< /A <)")
    assert pdfium_image_sizes(data) == [(10, 10)]
    assert read(data) == Declared(BOMB, True)
    assert verdict(data) == PDF_IMAGE


@pytest.mark.parametrize("number", [b"4294967295", b"-1"])
def test_a_width_that_is_no_reference_leaves_the_earlier_width(number):
    mask = _image_stream(b"/Width 20000 /Height 20000 /Width " + number + b" 0 R", None)
    data = image_pdf(0, 0, size=b"/Width 10 /Height 10 /SMask 6 0 R", objects=[mask, b"1"])
    assert pdfium_image_sizes(data) == [(10, 10)]
    assert read(data) == Declared(BOMB, True)
    assert verdict(data) == PDF_IMAGE


def test_a_size_in_an_object_stream_next_to_a_page_image_still_gives_the_page_line():
    data = objstm_pdf(b"/Width 7 0 R /Height 20000", b"20000") + b"7 0 obj 1 endobj\n"
    assert pdfium_image_sizes(data) == [(20000, 20000)]
    assert read(data) == Declared(0, False)
    assert verdict(data) == PAGE_IMAGE


def test_a_width_of_5000_digits_is_zero_pixels():
    data = sized(b"1" * 5000)
    assert pdfium_image_sizes(data) == [(0, 200)]
    assert read(data) == Declared(0, True)
    assert verdict(data) == "open"


def test_a_reference_of_5000_digits_cannot_be_checked():
    data = sized(b"1" * 5000 + b" 0 R")
    assert read(data) == Declared(0, False)
    assert verdict(data) == UNCHECKED


def test_an_object_header_of_5000_digits_after_the_file_does_not_stop_the_read():
    data = image_pdf(300, 200) + b"\n" + b"1" * 5000 + b" 0 obj\n"
    assert read(data) == Declared(300 * 200, True)
    assert verdict(data) == "open"


def test_a_width_that_is_a_reference_to_object_zero_is_zero_pixels():
    data = sized(b"0 0 R")
    assert pdfium_image_sizes(data) == [(0, 200)]
    assert read(data) == Declared(0, True)
    assert verdict(data) == "open"


NESTED_STREAM = b"/N << /Length 16 >> stream\nX endstream >> Y\nendstream junk"


def test_a_stream_nested_in_a_dictionary_cannot_be_checked():
    data = smask_pdf(
        b"/Width 10 /Height 10 " + NESTED_STREAM + b" /Width 16000 /Height 16000", b"null"
    )
    assert read(data).complete is False
    assert verdict(data) == UNCHECKED


NESTED_EMPTY_NAME = b"/D << / (>>) /Width 999 >> /Width %d /Height %d"


def test_an_empty_name_in_a_nested_dictionary_does_not_hide_the_mask():
    data = smask_pdf(NESTED_EMPTY_NAME % (20000, 20000), b"null")
    assert read(data) == Declared(BOMB, True)
    assert verdict(data) == PDF_IMAGE


def mid_number_pdf():
    """An xref entry for object 7 that points one byte into the header "17 0 obj"."""
    image = _image_stream(b"/Width 10 /Height 10", None, b" /SMask 6 0 R")
    mask = _image_stream(b"/Width 7 0 R /Height 20000", None)
    out, offsets = bytearray(b"%PDF-1.7\n"), []
    for number, body in enumerate(_placed("page", image, [mask]), 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    out += b"7 0 obj\n1\nendobj\n"
    offsets.append(len(out) + 1)
    out += b"17 0 obj\n20000\nendobj\n"
    xref, size = len(out), len(offsets) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(out)


def test_an_offset_pointing_into_a_header_number_defines_the_shorter_object():
    data = mid_number_pdf()
    assert read(data) == Declared(400_000_000, True)
    assert verdict(data) == PDF_IMAGE


def test_a_long_word_after_many_dictionaries_closing_on_one_end_is_read_quickly():
    soup = b"1 0 obj <<" * 200 + b">>" + b"x" * 20_000_000
    data = image_pdf(300, 200, objects=[pdf_stream(b"", soup)])
    assert read(data) == Declared(60000, True)
    assert verdict(data) == "open"


def reference_size_pdf(unused=b""):
    size = b"/Width 6 0 R /Height 20000"
    return image_pdf(0, 0, size=size, objects=[b"300", pdf_stream(b"", unused)])


def test_a_reference_size_with_an_obj_keyword_lacking_a_plain_header_cannot_be_checked():
    assert read(reference_size_pdf(b"x9 0 obj")).complete is False
    control = read(reference_size_pdf())
    assert control.complete is True
    assert control.largest == 6_000_000
