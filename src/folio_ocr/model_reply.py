"""What one answer of an OpenAI-compatible chat server says about a transcribed text line."""

import json
import math
from dataclasses import dataclass
from typing import Literal

from folio_ocr.charsets import is_arabic_letter, is_latin_letter
from folio_ocr.flags import FORMAT_NOISE, NO_TRANSCRIPTION, TRUNCATED
from folio_ocr.reading import Reading

UNREADABLE = "the model server sent a reply this tool cannot read"

# Openers a chatty model puts before the transcription itself.
CHATTER_PREFIXES = (
    "here is",
    "here's",
    "here are",
    "sure",
    "certainly",
    "the text",
    "the image",
    "the transcription",
    "transcription:",
    "text:",
    "output:",
)


@dataclass(frozen=True)
class ModelReading(Reading):
    """What the model server made of one line image."""

    flags: tuple[str, ...] = ()  # reply-contract flags
    confidence: float | None = None  # exp(min logprob) over the returned entries, 0..1
    method: Literal["logprobs", "two_read", "none"] = "none"


def status_error(status: int) -> str:
    """One friendly line for a non-success HTTP status; never holds the reply body."""
    if status in (401, 403):
        return f"the model server refused the API key or credentials (HTTP {status})"
    if status == 404:
        return (
            "the model server has no such endpoint or model (HTTP 404); "
            "check FOLIO_MODEL_URL (usually ending in /v1) and FOLIO_MODEL_NAME"
        )
    if status == 429:
        return "the model server is busy (HTTP 429)"
    if 500 <= status <= 599:
        return f"the model server failed (HTTP {status})"
    if 300 <= status <= 399:
        return (
            f"the model server redirected the request (HTTP {status}); "
            "set FOLIO_MODEL_URL to the final address"
        )
    return f"the model server answered HTTP {status}"


def has_format_noise(text: str) -> bool:
    """True when the text is not a bare one-line transcription."""
    folded = text.casefold()
    return (
        "<think" in folded
        or "</think" in folded
        or "```" in text
        or "\n" in text
        or "\r" in text
        or text.lstrip().casefold().startswith(CHATTER_PREFIXES)
    )


# The reading-order rule for model replies, named in the calibration setup key: a reply with
# Arabic and Latin letters is flagged MIXED_ORDER, because the order of its runs is not checked.
MIXED_ORDER = "mixed_order"
ORDER_RULE = "mixed_order=arabic+latin-letters"


def has_mixed_letters(text: str) -> bool:
    """True when the text holds at least one Arabic letter and at least one Latin letter."""
    return any(is_arabic_letter(c) for c in text) and any(is_latin_letter(c) for c in text)


def _usable_logprob(entry: object) -> float | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("logprob")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number) or number > 0:
        return None
    return number


def logprob_confidence(logprobs: object) -> float | None:
    """exp of the lowest logprob over the returned entries, or None when unusable."""
    if not isinstance(logprobs, dict):
        return None
    entries = logprobs.get("content")
    if not isinstance(entries, list) or not entries:
        return None
    values = [_usable_logprob(entry) for entry in entries]
    if any(v is None for v in values):
        return None
    return math.exp(min(values))


def _error(message: str) -> ModelReading:
    return ModelReading("", (), error=message)


def _first_choice(body: bytes) -> dict | None:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):  # nesting too deep to parse
        return None
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    if not isinstance(message.get("content"), (str, type(None))):
        return None
    if not isinstance(choice.get("finish_reason"), (str, type(None))):
        return None
    return choice


def parse_reply(status: int, body: bytes) -> ModelReading:
    """Turn one HTTP answer into a reading; any failure gives a one-line error."""
    if not 200 <= status <= 299:
        return _error(status_error(status))
    choice = _first_choice(body)
    if choice is None:
        return _error(UNREADABLE)
    text = choice["message"].get("content") or ""
    flags = []
    if choice.get("finish_reason") == "length":
        flags.append(TRUNCATED)
    if text.strip() == "":
        flags.append(NO_TRANSCRIPTION)
    if has_format_noise(text):
        flags.append(FORMAT_NOISE)
    confidence = logprob_confidence(choice.get("logprobs"))
    return ModelReading(
        text,
        (),
        flags=tuple(flags),
        confidence=confidence,
        method="none" if confidence is None else "logprobs",
    )
