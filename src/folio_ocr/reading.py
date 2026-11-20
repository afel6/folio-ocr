"""What an OCR reader made of one line of a page."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    text: str  # the words of the line in reading order, joined by single spaces
    scores: tuple[float, ...]  # one score per word, scaled to 0-1, kept exactly as reported
    error: str | None = None  # one line saying why the line could not be read; text is then ""
