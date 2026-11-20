"""Lines read by the model server: every detected line of an OCR page, and the text-layer
lines whose order or content cannot be trusted. Each reading becomes a line with its flags."""

import dataclasses
from typing import Literal, Protocol

from PIL import Image

from folio_ocr import calibration, model_calibration, model_request, model_setup
from folio_ocr.charsets import line_script
from folio_ocr.document import Calibration, Engine, Line, Page, line_id
from folio_ocr.flags import (
    MODEL_SELF_TEST_FAILED,
    NOT_READ,
    TEXT_LAYER_INVALID,
    VISUAL_ORDER,
    clean_text,
    line_flags,
    review_status,
)
from folio_ocr.ingest import RENDER_DPI
from folio_ocr.model_reply import MIXED_ORDER, ModelReading, has_mixed_letters
from folio_ocr.ocr_setup import OcrSetup, onnxtr_setup
from folio_ocr.ocr_words import LineWords
from folio_ocr.reading import Reading
from folio_ocr.text_layer_check import text_direction

REREAD_TRIGGERS = (VISUAL_ORDER, TEXT_LAYER_INVALID)
TEXT_LAYER_PREPROCESSING = calibration.preprocessing_hash(
    {"page": "text_layer", "render_dpi": RENDER_DPI}
)
TWO_READ_FLOOR = 0.98  # an agreement score below this is never accepted
_Found = tuple[float, model_calibration.TauSource] | None
NO_PIXELS = "the line box holds no pixels of the page"


class _Config(Protocol):
    name: str
    host: str
    page_deadline_s: float


class LineReader(Protocol):
    config: _Config

    def now(self) -> float: ...
    def fit(self) -> Literal["pass", "fail"]: ...
    def server(self) -> model_setup.ServerInfo | None: ...
    def tau(self, key: str) -> tuple[float, model_calibration.TauSource] | None: ...
    def read(self, image: Image.Image, languages: str, page_ends_at: float) -> ModelReading: ...


def ocr_preprocessing(setup: OcrSetup) -> str:
    """The page preparation a model line on an OCR page was judged under."""
    return (setup.setup if setup.setup is not None else onnxtr_setup(None)).preprocessing


def swap_line(current: Line, reread: Line) -> Line:
    """`reread` unless a person has edited or confirmed `current`."""
    return current if current.status in ("edited", "confirmed") else reread


def is_low_confidence(reading: ModelReading, tau: float | None) -> bool:
    """Fail closed: no frozen threshold for the reading's own method and setup (`tau` None), or
    a score below it."""
    scores = () if reading.confidence is None else (reading.confidence,)
    low = calibration.is_low_confidence(scores, tau)
    floor = reading.method == "two_read" and (
        reading.confidence is None or reading.confidence < TWO_READ_FLOOR
    )
    return low or floor


class _State:
    """What the model did in one document."""

    def __init__(self) -> None:
        self.asked = False
        self.ocr_read = False
        self.calibration: Calibration | None = None


class ModelDocument:
    """The model's part in reading one document."""

    def __init__(self, reader: LineReader, languages: str) -> None:
        self._reader = reader
        self._languages = languages
        self._state = _State()

    def _page(self, preprocessing: str) -> "ModelPage":
        ends_at = self._reader.now() + self._reader.config.page_deadline_s
        return ModelPage(self._reader, self._languages, self._state, preprocessing, ends_at)

    def ocr_page(self, setup: OcrSetup) -> "ModelPage":
        return self._page(ocr_preprocessing(setup))

    def text_page(self) -> "ModelPage":
        return self._page(TEXT_LAYER_PREPROCESSING)

    def engine(self, base: Engine) -> Engine:
        if not self._state.asked:
            return base
        config = self._reader.config
        return dataclasses.replace(
            base,
            reader="model" if self._state.ocr_read else base.reader,
            model_name=config.name,
            model_host=config.host,
            model_fit=self._reader.fit(),
            prompt_sha256=model_request.PROMPT_SHA256,
            calibration=self._state.calibration,
        )


class ModelPage:
    def __init__(
        self,
        reader: LineReader,
        languages: str,
        state: _State,
        preprocessing: str,
        ends_at: float,
    ) -> None:
        self._reader = reader
        self._languages = languages
        self._state = state
        self._preprocessing = preprocessing
        self._ends_at = ends_at

    def _ask(self, gray: Image.Image, line: LineWords | Line) -> ModelReading:
        self._state.asked = True
        self._reader.fit()
        try:
            crop = model_request.crop(gray, line.box)
        except ValueError:
            return ModelReading("", (), error=NO_PIXELS)
        return self._reader.read(crop, self._languages, self._ends_at)

    def read(self, ocr: OcrSetup, gray: Image.Image, line: LineWords) -> ModelReading:
        self._state.ocr_read = True
        return self._ask(gray, line)

    def _failed_flags(self) -> tuple[str, ...]:
        return (MODEL_SELF_TEST_FAILED,) if self._reader.fit() == "fail" else ()

    def _unread(self, extra: tuple[str, ...]) -> tuple[str, ...]:
        return (NOT_READ, *self._failed_flags(), *extra)

    def _remember(self, setup: calibration.Setup | None, found: _Found) -> None:
        """The document's calibration is that of its first line with a setup."""
        if setup is not None and self._state.calibration is None:
            self._state.calibration = model_calibration.record(setup.key, found)

    def _flags(
        self,
        reading: ModelReading,
        text: str,
        replaced: tuple[str, ...],
        tau: float | None,
    ) -> tuple:
        return (
            *reading.flags,
            *line_flags(text, self._languages),
            *replaced,
            *((MIXED_ORDER,) if has_mixed_letters(text) else ()),
            *((calibration.LOW_CONFIDENCE,) if is_low_confidence(reading, tau) else ()),
        )

    def _model_line(
        self,
        n: tuple[int, int],
        box: tuple[float, float, float, float],
        reading: Reading | None,
        page: tuple[str, str, tuple[str, ...]],
    ) -> Line:
        direction, _, extra = page
        if reading is None or reading.error is not None:
            flags = tuple(dict.fromkeys(self._unread(extra)))
            return Line(
                id=line_id(*n),
                box=box,
                text="",
                direction=text_direction("", direction),
                script=line_script(""),
                reader="none",
                confidence=None,
                confidence_method="none",
                flags=flags,
                status=review_status(flags, "auto_accepted"),
                final_text="",
            )
        text, replaced = clean_text(reading.text)
        setup = model_setup.model_setup(
            self._reader.server(), self._reader.config.name, reading.method, self._preprocessing
        )
        found = None if setup is None else self._reader.tau(setup.key)
        self._remember(setup, found)
        own = self._flags(reading, text, replaced, None if found is None else found[0])
        flags = tuple(dict.fromkeys((*own, *self._failed_flags(), *extra)))
        return Line(
            id=line_id(*n),
            box=box,
            text=text,
            direction=text_direction(text, direction),
            script=line_script(text),
            reader="model",
            confidence=reading.confidence,
            confidence_method=reading.method,
            flags=flags,
            status=review_status(flags, "auto_accepted"),
            final_text=text,
        )

    def line(
        self,
        ocr: OcrSetup,
        n: tuple[int, int],
        line: LineWords,
        reading: Reading | None,
        page: tuple[str, str, tuple[str, ...]],
    ) -> Line:
        return self._model_line(n, line.box, reading, page)

    def _reread_line(self, page: Page, gray: Image.Image, line: Line) -> tuple[Line, str | None]:
        if line.status in ("edited", "confirmed") or not set(line.flags) & set(REREAD_TRIGGERS):
            return line, None
        reading = self._ask(gray, line)
        if reading.error is not None:
            flags = tuple(dict.fromkeys((*line.flags, *self._unread(()))))
            failed = dataclasses.replace(
                line, flags=flags, status=review_status(flags, line.status)
            )
            return failed, reading.error
        new = self._model_line(
            (page.index, 1), line.box, reading, (page.direction, self._languages, ())
        )
        flags = tuple(dict.fromkeys((*line.flags, *new.flags)))
        done = dataclasses.replace(
            new, id=line.id, flags=flags, status=review_status(flags, line.status)
        )
        return done, None

    def reread(self, page: Page, image: Image.Image) -> Page:
        gray = image.convert("L")
        results = [self._reread_line(page, gray, line) for line in page.lines]
        errors = [e for _, e in results if e is not None]
        error = page.error
        if errors:
            message = errors[0] + (f" ({len(errors)} lines not read)" if len(errors) > 1 else "")
            error = message if page.error is None else f"{page.error}; {message}"
        return dataclasses.replace(page, lines=tuple(r for r, _ in results), error=error)
