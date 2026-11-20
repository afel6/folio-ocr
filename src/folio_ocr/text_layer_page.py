"""A page read from the embedded text layer of a PDF page (review rule R4): the text is kept
as extracted and never repaired; doubtful lines are flagged and unread ink is reported."""

from collections.abc import Sequence

from PIL import Image

from folio_ocr.charsets import line_script
from folio_ocr.coverage import uncovered_ink
from folio_ocr.document import IDENTITY, Box, Line, Page, TextChar, line_id
from folio_ocr.flags import UNREAD_REGIONS, clean_text, line_flags, review_status
from folio_ocr.ingest import RENDER_DPI
from folio_ocr.text_layer import line_box, line_text
from folio_ocr.text_layer_check import LineCheck, TextLayerCheck


def text_layer_line(
    page_no: int, k: int, chars: Sequence[TextChar], check: LineCheck, languages: str
) -> Line:
    """Line `k` of page `page_no`: its R4 flags, `invalid_text` when a lone surrogate became
    U+FFFD, and the review flags of its text."""
    text, replaced = clean_text(line_text(chars))
    flags = tuple(dict.fromkeys((*check.flags, *replaced, *line_flags(text, languages))))
    return Line(
        id=line_id(page_no, k),
        box=line_box(chars),
        text=text,
        direction=check.direction,
        script=line_script(text),
        reader="text_layer",
        confidence=None,
        confidence_method="none",
        flags=flags,
        status=review_status(flags, "auto_accepted"),
        final_text=text,
    )


def text_layer_page(
    page_no: int,
    images: Sequence[Box],
    image: Image.Image,
    segments: Sequence[Sequence[TextChar]],
    report: TextLayerCheck,
    languages: str,
) -> Page:
    """The page whose text layer is kept. `image` is its render at RENDER_DPI, which is also the
    straightened frame: text-layer pages are never deskewed, so the transform is identity.
    `images` (the boxes of the images drawn on the page, from `text_layer.image_boxes`) and ink
    outside every line box are unread regions."""
    pairs = zip(segments, report.lines, strict=True)
    lines = tuple(
        text_layer_line(page_no, k, chars, check, languages)
        for k, (chars, check) in enumerate(pairs, start=1)
    )
    ink = uncovered_ink(image, [*(line.box for line in lines), *images])
    unread = tuple(sorted((*images, *ink), key=lambda b: (b[1], b[0])))
    return Page(
        index=page_no,
        width=image.width,
        height=image.height,
        render_dpi=RENDER_DPI,
        rotation_applied=0.0,
        transform=IDENTITY,
        origin="text_layer",
        direction=report.direction,
        status="read",
        flags=(*report.reasons, *((UNREAD_REGIONS,) if unread else ())),
        unread_regions=unread,
        lines=lines,
    )
