"""The shared error classes: each one shows the user exactly one safe line."""

import pytest

from folio_ocr.errors import (
    ConfigError,
    FolioError,
    InputError,
    ModelServerError,
    SetupNeededError,
    WeightsError,
)

KINDS = (InputError, ConfigError, WeightsError, ModelServerError)


@pytest.mark.parametrize("kind", KINDS)
def test_every_error_is_a_folio_error(kind):
    assert issubclass(kind, FolioError)
    assert issubclass(FolioError, Exception)
    assert kind.__doc__


@pytest.mark.parametrize("kind", (FolioError, *KINDS))
def test_str_is_the_one_line_message(kind):
    error = kind("the file is not a PDF or an image")
    assert str(error) == "the file is not a PDF or an image"
    assert error.args == ("the file is not a PDF or an image",)


def test_the_five_classes_are_distinct():
    assert len({FolioError, *KINDS}) == 5
    for kind in KINDS:
        assert not any(issubclass(kind, other) for other in KINDS if other is not kind)


def test_setup_needed_is_a_weights_error_with_a_one_line_message():
    assert issubclass(SetupNeededError, WeightsError)
    assert issubclass(SetupNeededError, FolioError)
    assert SetupNeededError.__doc__
    error = SetupNeededError("the OCR model files are missing; run folio setup")
    assert str(error) == "the OCR model files are missing; run folio setup"
    assert error.args == ("the OCR model files are missing; run folio setup",)


@pytest.mark.parametrize(
    "message",
    ["", "   ", "two\nlines", "ends with a break\n", "carriage\rreturn", "para graph"],
)
def test_a_message_must_be_one_non_empty_line(message):
    with pytest.raises(ValueError):
        FolioError(message)


def test_a_message_must_be_text():
    with pytest.raises(ValueError):
        InputError(b"bytes")
