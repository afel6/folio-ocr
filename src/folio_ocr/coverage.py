"""Find dark pixels on a page image that no known text box covers.

The dark surround of a photographed page (the table around the paper) is not ink on the page.
A patch of dark pixels joined SURROUND_CONNECTIVITY-way is surround, and is left out with any ink
joined to it, when all three hold:
- its bounding box spans at least SURROUND_SPAN of the image's width and of its height;
- it covers less than SURROUND_SHARE of the image's pixels, so a dark page background stays ink;
- one of its pixels lies in the outermost SURROUND_EDGE_PX rows or columns of the image, so a
  printed frame or form grid inside a light margin stays ink (the band is wider than one pixel
  because straightening a photo resamples its corners and can leave the turned ring a pixel
  short of the canvas edge).
A ring round the paper qualifies, so does a U, or an L of two bars along two edges (the paper
pushed into a corner of the photo); a bar along one edge does not, nor two bars on opposite
edges that never meet."""

import math
from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from folio_ocr.document import Box

INK_LEVEL = 128
SURROUND_CONNECTIVITY = 8
SURROUND_SPAN = 0.9
SURROUND_SHARE = 0.5
SURROUND_EDGE_PX = 3
# Named in the OCR setup key: how the ink outside every box is found.
METHOD = (
    f"ink-tiles-8-way;without-dark-surround;surround-{SURROUND_CONNECTIVITY}-way"
    f";span>={SURROUND_SPAN};share<{SURROUND_SHARE};edge<{SURROUND_EDGE_PX}px"
)


def _ink_mask(image: Image.Image) -> Image.Image:
    """Return a new mask image: 255 where the pixel is ink, 0 elsewhere."""
    return image.convert("L").point(lambda v: 255 if v < INK_LEVEL else 0)


def _surround_labels(stats: np.ndarray, width: int, height: int) -> np.ndarray:
    """The labels (row numbers of `stats`, 0 = background) of the patches that are surround."""
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    wide, tall = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    near_edge = (
        (left < SURROUND_EDGE_PX)
        | (top < SURROUND_EDGE_PX)
        | (left + wide > width - SURROUND_EDGE_PX)
        | (top + tall > height - SURROUND_EDGE_PX)
    )
    spans = (wide >= SURROUND_SPAN * width) & (tall >= SURROUND_SPAN * height)
    small = stats[:, cv2.CC_STAT_AREA] < SURROUND_SHARE * width * height
    found = np.flatnonzero(spans & small & near_edge)
    return found[found > 0]


def _without_surround(mask: Image.Image) -> Image.Image:
    """`mask` (255 = ink) without the dark surround described in the module docstring."""
    ink = np.asarray(mask) > 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=SURROUND_CONNECTIVITY
    )
    surround = _surround_labels(stats, mask.width, mask.height)
    if surround.size == 0:
        return mask
    kept = ink & ~np.isin(labels, surround)
    return Image.fromarray(np.where(kept, 255, 0).astype(np.uint8))


def has_ink(image: Image.Image) -> bool:
    """Say whether any pixel of the image is dark enough to count as ink."""
    return image.convert("L").getextrema()[0] < INK_LEVEL


def _check(margin: float, tile: int) -> None:
    """Raise ValueError unless margin is a finite number >= 0 and tile a positive int."""
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError("margin must be a finite number >= 0")  # noqa: TRY004
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite number >= 0")
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise ValueError("tile must be a positive integer")


def _tile_bounds(mask: Image.Image, tile: int) -> dict[tuple[int, int], Box]:
    """Map each square holding ink to the tight bounds of its ink, in page pixels."""
    width, height = mask.size
    found: dict[tuple[int, int], Box] = {}
    for row, y in enumerate(range(0, height, tile)):
        for col, x in enumerate(range(0, width, tile)):
            bbox = mask.crop((x, y, min(x + tile, width), min(y + tile, height))).getbbox()
            if bbox is not None:
                found[(col, row)] = (x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3])
    return found


_NEIGHBOURS = tuple((dc, dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1) if (dc, dr) != (0, 0))


def _regions(found: dict[tuple[int, int], Box]) -> list[Box]:
    """Join squares that touch (8-neighbourhood) and return each group's bounds."""
    seen: set[tuple[int, int]] = set()
    regions: list[Box] = []
    for start, first in found.items():
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        x0, y0, x1, y1 = first
        while queue:
            col, row = queue.popleft()
            for dc, dr in _NEIGHBOURS:
                nxt = (col + dc, row + dr)
                if nxt in found and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
                    b = found[nxt]
                    x0, y0 = min(x0, b[0]), min(y0, b[1])
                    x1, y1 = max(x1, b[2]), max(y1, b[3])
        regions.append((float(x0), float(y0), float(x1), float(y1)))
    return regions


def uncovered_ink(
    image: Image.Image,
    covered: Sequence[Box],
    *,
    margin: float = 2.0,
    tile: int = 32,
) -> tuple[Box, ...]:
    """Return boxes around ink lying outside every covered box (grown by margin).

    The dark surround of a photographed page is not ink (see the module docstring). Ink in
    squares of `tile` pixels that touch, diagonals included, forms one region; regions are
    sorted by (y0, x0).
    """
    _check(margin, tile)
    mask = _without_surround(_ink_mask(image))
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in covered:
        left, top = math.floor(x0 - margin), math.floor(y0 - margin)
        right, bottom = math.ceil(x1 + margin) - 1, math.ceil(y1 + margin) - 1
        if right >= left and bottom >= top:
            draw.rectangle((left, top, right, bottom), fill=0)
    if mask.getbbox() is None:
        return ()
    return tuple(sorted(_regions(_tile_bounds(mask, tile)), key=lambda b: (b[1], b[0])))
