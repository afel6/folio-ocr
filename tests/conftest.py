import pytest
from ocr_stub import ocr_stub, ocr_stub_module  # noqa: F401  (fixtures shared by the test files)

# Settings that would change which confidence threshold judges a model line; a test sets them
# itself when it needs them.
_MODEL_CALIBRATION_VARIABLES = ("FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER", "FOLIO_TAU_FILE")


@pytest.fixture(autouse=True)
def _no_model_calibration_settings(monkeypatch):
    for name in _MODEL_CALIBRATION_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _tesseract_version_decided_per_test(monkeypatch):
    """Each test decides the Tesseract version anew, so a check that ran out of time in one test
    is not kept for the tests after it."""
    from folio_ocr import tesseract

    monkeypatch.setattr(tesseract, "_DECIDED", {})
