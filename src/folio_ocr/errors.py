"""Errors the tool reports to its user. `str(error)` is one line that is safe to show:
it names what went wrong and what to do, never a path, a stack trace or document text."""


class FolioError(Exception):
    """Base of every error shown to the user; the message is exactly one non-empty line."""

    def __init__(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip() or message.splitlines() != [message]:
            raise ValueError("an error message is one non-empty line of text")
        super().__init__(message)


class InputError(FolioError):
    """The input file was refused: its type, size, page count, encryption, pixel count or
    damage, or an unknown `languages` value.
    The HTTP API answers 400."""


class ConfigError(FolioError):
    """A setting is missing or invalid (environment variables, a configured path)."""


class WeightsError(FolioError):
    """The OCR model files are missing, damaged or unreadable, or an asset download failed."""


class SetupNeededError(WeightsError):
    """The model files are missing or fail their pinned size or sha256; `folio setup` fetches them."""


class ModelServerError(FolioError):
    """The configured model server failed or answered outside its contract.
    The HTTP API answers 502."""
