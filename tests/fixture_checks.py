"""Content checks for fixtures regenerated on a different build or CPU."""

import json
from io import BytesIO

import numpy as np
from PIL import Image


def assert_same_png(actual: bytes, expected: bytes) -> None:
    """Compare PNG content independently of its compressed byte stream."""
    with Image.open(BytesIO(actual)) as actual_image, Image.open(BytesIO(expected)) as pinned:
        assert actual_image.format == pinned.format == "PNG"
        assert actual_image.mode == pinned.mode
        assert actual_image.size == pinned.size
        assert actual_image.tobytes() == pinned.tobytes(), "decoded PNG pixels changed"
        assert actual_image.getpalette() == pinned.getpalette()
        assert actual_image.info == pinned.info


def assert_same_ground_truth(actual: dict, expected: dict) -> None:
    """Keep text and boxes exact, allowing only matrix round-off."""
    regenerated, committed = actual.copy(), expected.copy()
    actual_matrix, expected_matrix = regenerated.pop("homography"), committed.pop("homography")
    assert json.dumps(regenerated, sort_keys=True) == json.dumps(committed, sort_keys=True)
    np.testing.assert_allclose(
        actual_matrix, expected_matrix, rtol=0, atol=1e-10, equal_nan=False, strict=True
    )
