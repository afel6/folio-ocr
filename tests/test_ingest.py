import io
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
import pytest
from PIL import Image, ImageFile
from sample_files import encrypted, pdf_bytes, png_header, save, tiff

from folio_ocr import ingest
from folio_ocr.errors import InputError
from folio_ocr.ingest import (
    MAX_PIXELS,
    MAX_UPLOAD_BYTES,
    check_size,
    frame_count,
    load_frame,
    open_image,
    open_pdf,
    page_pixels,
    render_size,
    sanitise_filename,
    sniff,
)
from folio_ocr.pipeline import read_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "en-invoice.pdf"
ENCRYPTED = "the PDF is encrypted; remove its password and try again"
DAMAGED = "the PDF could not be opened; it may be damaged"
DECODE = "the image could not be decoded; it may be damaged"
IMG_BIG = "the image is too large to read (over 50 megapixels)"


# ---- sniff / size / filename ----


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"%PDF-1.7", "pdf"),
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff\xe0", "jpeg"),
        (b"II*\x00", "tiff"),
        (b"MM\x00*", "tiff"),
        (b"II+\x00", "tiff"),
        (b"MM\x00+", "tiff"),
        (b"RIFF\x01\x02\x03\x04WEBPVP8 ", "webp"),
    ],
)
def test_sniff_accepts(data, kind):
    assert sniff(data) == kind


@pytest.mark.parametrize(
    "data",
    [b"hello", b"GIF89a", b"%PDF", b" %PDF-1.7", b"RIFF\x00\x00\x00\x00WAVE", b""],
)
def test_sniff_refuses(data):
    with pytest.raises(InputError, match="not a PDF, PNG, JPEG, TIFF or WebP"):
        sniff(data)


def test_check_size_boundary():
    check_size(MAX_UPLOAD_BYTES)
    with pytest.raises(InputError, match="larger than 25 MB"):
        check_size(MAX_UPLOAD_BYTES + 1)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("C:\\Users\\x\\inv.pdf", "inv.pdf"),
        ("../../etc/passwd", "passwd"),
        ("a\x00b\u202e.pdf", "a_b_.pdf"),
        ("fac<t>ure?.pdf", "fac_t_ure_.pdf"),
        ("  ..report.pdf. ", "report.pdf"),
        ("فاتورة.pdf", "فاتورة.pdf"),
        ("a\ud800b.pdf", "a_b.pdf"),
        ("x\x85y\x7f.pdf", "x_y_.pdf"),
        ("...", "document"),
        ("", "document"),
        ("dir/", "document"),
        ("a" * 300, "a" * 255),
        ("a\u061cb.pdf", "a_b.pdf"),
    ],
)
def test_sanitise_filename(name, expected):
    assert sanitise_filename(name) == expected


# ---- PDF ----


def _open_ok(data):
    doc = open_pdf(data)
    n = len(doc)
    doc.close()
    return n


@pytest.mark.parametrize("rotation", [0, 90])
def test_render_size_matches_pdfium(rotation):
    doc = pdfium.PdfDocument(FIXTURE.read_bytes())
    page = doc[0]
    if rotation:
        page.set_rotation(rotation)
    bitmap = page.render(scale=300 / 72)
    assert (bitmap.width, bitmap.height) == render_size(*page.get_size())
    bitmap.close()
    page.close()
    doc.close()


def test_open_pdf_fixture():
    assert _open_ok(FIXTURE.read_bytes()) >= 1


def test_open_pdf_page_count_cap():
    assert _open_ok(pdf_bytes([(100, 100)] * 20)) == 20
    with pytest.raises(InputError, match="more than 20 pages"):
        open_pdf(pdf_bytes([(100, 100)] * 21))


@pytest.mark.parametrize("user", ["u", ""])
def test_open_pdf_encrypted(user):
    with pytest.raises(InputError) as info:
        open_pdf(encrypted(user))
    assert str(info.value) == ENCRYPTED


def test_unsupported_security_handler_is_reported_as_encrypted(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise pdfium.PdfiumError("unsupported", err_code=pdfium_raw.FPDF_ERR_SECURITY)

    monkeypatch.setattr(pdfium, "PdfDocument", refuse)
    with pytest.raises(InputError) as info:
        open_pdf(b"%PDF-1.7")
    assert str(info.value) == ENCRYPTED


def test_open_pdf_damaged():
    with pytest.raises(InputError) as info:
        open_pdf(b"%PDF-1.7\nnot really a pdf at all")
    assert str(info.value) == DAMAGED


def test_oversized_page_refused_without_rendering(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("must not render or load a page")

    monkeypatch.setattr(pdfium.PdfPage, "render", boom)
    monkeypatch.setattr(pdfium_raw, "FPDF_RenderPageBitmap", boom)
    monkeypatch.setattr(pdfium.PdfDocument, "get_page", boom)
    monkeypatch.setattr(pdfium.PdfDocument, "__getitem__", boom)
    with pytest.raises(InputError, match=r"page 1 is too large to read \(over 50 megapixels"):
        open_pdf(pdf_bytes([(14400, 14400)]))


def test_page_cap_boundary():
    under, over = 1697.0, 1697.5
    assert render_size(under, under)[0] ** 2 <= MAX_PIXELS < render_size(over, over)[0] ** 2
    assert _open_ok(pdf_bytes([(under, under)])) == 1
    with pytest.raises(InputError, match="page 1 is too large"):
        open_pdf(pdf_bytes([(over, over)]))


def test_page_cap_names_the_page():
    with pytest.raises(InputError, match="page 2 is too large"):
        open_pdf(pdf_bytes([(100, 100), (14400, 14400)]))


def test_refusal_closes_document(monkeypatch):
    closed = []
    original = pdfium.PdfDocument.close

    def spy(self):
        closed.append(True)
        original(self)

    monkeypatch.setattr(pdfium.PdfDocument, "close", spy)
    with pytest.raises(InputError):
        open_pdf(pdf_bytes([(100, 100)] * 21))
    assert closed


# ---- images ----


def test_png_pixel_cap_boundary():
    ok = save(Image.new("1", (7071, 7071)), "PNG")
    open_image(ok, "png").close()
    with pytest.raises(InputError) as info:
        open_image(save(Image.new("1", (7072, 7072)), "PNG"), "png")
    assert str(info.value) == IMG_BIG


def test_tiff_second_frame_oversized():
    with pytest.raises(InputError) as info:
        open_image(tiff([(10, 10), (7072, 7072)]), "tiff")
    assert str(info.value) == IMG_BIG


def test_truncated_png_refused_with_decode_message():
    data = save(Image.effect_noise((200, 200), 80), "PNG")
    img = open_image(data[: len(data) // 2], "png")
    with pytest.raises(InputError) as info:
        load_frame(img, 0)
    assert str(info.value) == DECODE


def test_unreadable_header_refused():
    with pytest.raises(InputError) as info:
        open_image(b"\x89PNG\r\n\x1a\nbroken", "png")
    assert str(info.value) == DECODE


def test_tiff_frames():
    img = open_image(tiff([(10, 10), (20, 10), (30, 10)]), "tiff")
    assert frame_count(img) == 3
    assert load_frame(img, 2).size == (30, 10)
    assert load_frame(img, 0).size == (10, 10)


def test_animated_webp_is_one_page():
    frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    data = save(frames[0], "WEBP", save_all=True, append_images=frames[1:])
    img = open_image(data, "webp")
    assert getattr(img, "n_frames", 1) == 2
    assert frame_count(img) == 1


def test_jpeg_exif_orientation_applied_and_frame_independent():
    exif = Image.Exif()
    exif[274] = 6
    data = save(Image.new("RGB", (40, 20), "white"), "JPEG", exif=exif)
    img = open_image(data, "jpeg")
    frame = load_frame(img, 0)
    img.close()
    del img, data
    assert frame.size == (20, 40)
    assert frame.getpixel((0, 0)) == (255, 255, 255)


def test_plain_frame_is_a_copy_independent_of_source():
    img = open_image(save(Image.new("RGB", (4, 4), "red"), "PNG"), "png")
    frame = load_frame(img, 0)
    img.close()
    assert frame.getpixel((1, 1)) == (255, 0, 0)


def test_module_constants():
    assert (ingest.MAX_PDF_PAGES, ingest.MAX_PIXELS, ingest.RENDER_DPI) == (20, 50_000_000, 300)
    assert ingest.MAX_UPLOAD_BYTES == 25 * 1024 * 1024


@pytest.mark.parametrize("side", [8000, 20000])
def test_crafted_header_refused_from_the_header_alone(monkeypatch, side):
    def no_decode(*_a, **_k):
        raise AssertionError("pixels were decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", no_decode)
    data = png_header(side, side)
    assert len(data) < 100
    with pytest.raises(InputError) as info:
        open_image(data, "png")
    assert str(info.value) == IMG_BIG


def test_tiff_with_twenty_frames_is_read_and_twenty_one_refused():
    img = open_image(tiff([(4, 4)] * 20), "tiff")
    assert frame_count(img) == 20
    assert img.tell() == 0
    with pytest.raises(InputError) as info:
        open_image(tiff([(4, 4)] * 21), "tiff")
    assert str(info.value) == "the TIFF has more than 20 pages"


def test_frames_after_the_first_are_ignored_for_a_webp():
    frames = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8), "blue")]
    data = save(frames[0], "WEBP", save_all=True, append_images=frames[1:])
    assert load_frame(open_image(data, "webp"), 0).getpixel((0, 0)) == (0, 0, 0)


def _reads_or_refuses(data, name):
    """Reading either gives a Document or raises InputError; any other error escapes."""
    try:
        read_bytes(data, name, languages="en")
    except InputError:
        pass


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_truncated_two_frame_tiff_never_raises_another_error(ocr_stub):
    noise = [Image.effect_noise(size, 60) for size in ((30, 20), (40, 10))]
    data = save(noise[0], "TIFF", save_all=True, append_images=noise[1:])
    for cut in range(1, len(data)):
        _reads_or_refuses(data[:cut], "x.tif")


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_truncated_single_picture_never_raises_another_error(fmt, ocr_stub):
    data = save(Image.effect_noise((40, 30), 60), fmt)
    for cut in range(1, len(data), 7):
        _reads_or_refuses(data[:cut], "x." + fmt.lower())


def test_sanitised_name_is_cut_before_trailing_space_and_dot_are_stripped():
    name = sanitise_filename("a" * 253 + " .pdf")
    assert len(name) <= 255
    assert not name.endswith((" ", ".")) and not name.startswith((" ", "."))


# page_pixels: every decoded frame becomes 8-bit gray or RGB as it looks on white paper
def test_transparent_black_is_white_paper():
    for mode in ("RGBA", "LA"):
        frame = Image.new(mode, (4, 3), (0, 0) if mode == "LA" else (0, 0, 0, 0))
        out = page_pixels(frame)
        assert out.mode == "RGB" and out.getextrema() == ((255, 255),) * 3


def test_half_transparent_ink_is_blended_over_white():
    out = page_pixels(Image.new("RGBA", (2, 2), (0, 0, 0, 128)))
    assert out.getpixel((0, 0)) == (127, 127, 127)


def test_palette_with_a_transparent_colour_is_laid_over_white():
    frame = Image.new("P", (2, 1), 0)
    frame.putpalette([0, 0, 0, 10, 20, 30])
    frame.putpixel((1, 0), 1)
    frame.info["transparency"] = 0
    out = page_pixels(frame)
    assert (out.getpixel((0, 0)), out.getpixel((1, 0))) == ((255, 255, 255), (10, 20, 30))


def test_sixteen_bit_gray_is_scaled_not_clipped():
    frame = Image.new("I;16", (3, 1))
    for x, v in enumerate((0, 32768, 65535)):
        frame.putpixel((x, 0), v)
    out = page_pixels(frame)
    assert out.mode == "L" and [out.getpixel((x, 0)) for x in range(3)] == [0, 127, 255]


def test_thirty_two_bit_gray_with_16_bit_values_is_scaled():
    frame = Image.new("I", (2, 1))
    frame.putpixel((1, 0), 65535)
    out = page_pixels(frame)
    assert [out.getpixel((x, 0)) for x in range(2)] == [0, 255]


def test_eight_bit_values_in_a_32_bit_frame_keep_their_level():
    frame = Image.new("I", (1, 1), 200)
    assert page_pixels(frame).getpixel((0, 0)) == 200


def test_other_modes_become_l_or_rgb():
    assert page_pixels(Image.new("1", (2, 2), 1)).mode == "L"
    assert page_pixels(Image.new("CMYK", (2, 2), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
    assert page_pixels(Image.new("P", (2, 2), 0)).mode == "RGB"


def test_gray_and_rgb_frames_are_unchanged():
    for mode in ("L", "RGB"):
        frame = Image.new(mode, (2, 2))
        assert page_pixels(frame) is frame


def test_decoded_rgba_png_frame_is_rgb():
    buffer = io.BytesIO()
    Image.new("RGBA", (5, 5), (0, 0, 0, 0)).save(buffer, format="PNG")
    img = Image.open(io.BytesIO(buffer.getvalue()))
    assert load_frame(img, 0).mode == "RGB"
