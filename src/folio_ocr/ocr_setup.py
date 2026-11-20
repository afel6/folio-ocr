"""The OCR reader a document's pages are read with, chosen once per document: OnnxTR for English,
Tesseract for Arabic and mixed pages when it is installed, else nobody (lines wait for a model
server). Also the reader's calibration setup key and frozen threshold. Tesseract's version, which
the key holds, is decided once per program file and process (tesseract.decided_version), so every
document of a run gets the same key."""

import hashlib
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from folio_ocr import calibration, coverage, onnxtr_models, orientation, tesseract
from folio_ocr.assets import (
    ONNXTR,
    ONNXTR_DETECTOR,
    ONNXTR_RECOGNISER,
    TESSDATA,
    TESSDATA_DIR,
    TESSERACT_OSD,
    Asset,
    files_problem,
)
from folio_ocr.document import Calibration, Engine
from folio_ocr.onnxtr_models import Models
from folio_ocr.tesseract import TesseractFailure

# Lines are cut where the gap between word boxes exceeds this many times the row's median box
# height (the value with the best line recall on the development pages, which hold tables whose
# columns stand 1.2 to 2.5 box heights apart).
OCR_GAP_FACTOR = 1.7
LINE_INK_MARGIN = 2.0  # a line box shrinks to the ink found within this margin of its words
RESCUE_REGIONS = 16  # unread ink regions searched again for words, per page
# A region is searched again inside a window grown on each side by the larger of these.
WINDOW_MIN_PX = 64
WINDOW_HEIGHTS = 2  # times the region's height
_DESKEW = "onnxtr-0.9-estimate_orientation"
_ORIENTATION_VOTE = "long-box-sideways;crnn-vote"
_ORIENTATION_OSD = "long-box-sideways;tesseract-osd;crnn-vote-fallback"


@dataclass(frozen=True)
class OcrSetup:
    kind: Literal["onnxtr", "tesseract", "none"]
    models: Models
    tesseract: Path | None  # absolute path of the program when kind is tesseract
    tessdata: Path | None
    timeout_s: float  # time limit for one Tesseract line or page orientation report
    problem: str | None  # one line: why Tesseract cannot read any line of this document
    setup: calibration.Setup | None  # None when nobody reads the lines or Tesseract cannot start
    tau: float | None
    osd: bool = False  # the way up of each page is asked of Tesseract's orientation report


def _orientation_terms(osd_version: str | None) -> dict[str, str | float]:
    """The preprocessing entries that describe how the way up of a page is decided."""
    if osd_version is None:
        return {"orientation": _ORIENTATION_VOTE}
    return {
        "orientation": _ORIENTATION_OSD,
        "osd_sha256": TESSERACT_OSD.sha256,
        "osd_arguments": " ".join(tesseract.OSD_ARGUMENTS),
        "osd_floor": orientation.OSD_FLOOR,
        "osd_tesseract": osd_version,
    }


def _preprocessing(osd_version: str | None, **extra: str | float) -> str:
    return calibration.preprocessing_hash(
        {
            "detector": onnxtr_models.DETECTOR,
            "detector_sha256": ONNXTR_DETECTOR.sha256,
            "gap_factor": OCR_GAP_FACTOR,
            "line_ink_margin": LINE_INK_MARGIN,
            "rescue_regions": RESCUE_REGIONS,
            "window_min_px": WINDOW_MIN_PX,
            "window_heights": WINDOW_HEIGHTS,
            "sideways_share": orientation.SIDEWAYS_SHARE,
            "long_ratio": orientation.LONG_RATIO,
            "vote_words": orientation.VOTE_WORDS,
            "vote_quantile": orientation.VOTE_QUANTILE,
            "uncertain_margin_pair": orientation.UNCERTAIN_MARGIN_PAIR,
            "uncertain_margin_four": orientation.UNCERTAIN_MARGIN_FOUR,
            "coverage": coverage.METHOD,
            "deskew": _DESKEW,
            **_orientation_terms(osd_version),
            "scale": 1,
            **extra,
        }
    )


def onnxtr_setup(osd_version: str | None = None) -> calibration.Setup:
    """The calibration setup of the OnnxTR word recogniser as installed; `osd_version` is the
    major.minor of the Tesseract that reports page orientation, None for the recogniser vote."""
    return calibration.Setup(
        reader="onnxtr",
        architecture=onnxtr_models.RECOGNISER,
        weights_sha256=ONNXTR_RECOGNISER.sha256,
        aggregation=onnxtr_models.AGGREGATION,
        version=calibration.version_major_minor(version("onnxtr")),
        arguments=(),
        preprocessing=_preprocessing(
            osd_version,
            split_ratio=onnxtr_models.SPLIT_RATIO,
            split_target=onnxtr_models.SPLIT_TARGET,
            split_overlap=onnxtr_models.SPLIT_OVERLAP,
        ),
    )


def tesseract_setup(tesseract_version: str) -> calibration.Setup:
    """The calibration setup of Tesseract `tesseract_version` with the pinned language data."""
    version = calibration.version_major_minor(tesseract_version)
    return calibration.Setup(
        reader="tesseract",
        architecture="lstm",
        weights_sha256="+".join(asset.sha256 for asset in TESSDATA),
        aggregation="min",
        version=version,
        arguments=("-l", tesseract.LANGUAGES, *tesseract.ARGUMENTS),
        preprocessing=_preprocessing(version, padding_px=tesseract.PADDING_PX, crop_mode="L"),
    )


def _tesseract(weights_dir: Path, binary: Path) -> tuple[str | None, calibration.Setup | None]:
    """(problem, setup) of Tesseract at `binary`; the problem is one line when it cannot read.
    The version check has its own time limit; the line time limit applies to lines only."""
    problem = tesseract.tessdata_problem(weights_dir)
    if problem is not None:
        return problem, None
    try:
        return None, tesseract_setup(tesseract.decided_version(binary))
    except TesseractFailure as failure:
        return str(failure), None
    except ValueError:
        return "Tesseract reported a version this tool does not understand", None


Runner = Callable[..., subprocess.CompletedProcess]


def _version(binary: Path, runner: Runner | None) -> str:
    """The version of the Tesseract at `binary`: the one this process decided for the program
    (`runner` None), or the answer of `runner` to `tesseract --version`."""
    if runner is None:
        return tesseract.decided_version(binary)
    return tesseract.tesseract_version(binary, runner=runner)


def _osd_version(
    weights_dir: Path | None, binary: Path | None, *, runner: Runner | None = None
) -> str | None:
    """major.minor of the Tesseract at `binary` when it can report page orientation for an
    English document (it runs, its version is known and the pinned osd file is intact), else None."""
    if binary is None or weights_dir is None:
        return None
    if files_problem((TESSERACT_OSD,), weights_dir) is not None:
        return None
    try:
        return calibration.version_major_minor(_version(binary, runner))
    except (TesseractFailure, ValueError):
        return None


def prepare(
    languages: str, weights_dir: Path | None, tesseract_path: Path | None, timeout_s: float
) -> OcrSetup:
    """The reader for this document's OCR pages. Raises WeightsError when the OCR model files
    are not set up (every OCR page needs the detector)."""
    models = onnxtr_models.load(weights_dir)  # refuses a missing folder
    if languages == "en":
        found = _osd_version(weights_dir, tesseract_path)
        setup = onnxtr_setup(found)
        if found is None or tesseract_path is None or weights_dir is None:
            return OcrSetup("onnxtr", models, None, None, timeout_s, None, setup, _tau(setup))
        tessdata = weights_dir / TESSDATA_DIR
        return OcrSetup(
            "onnxtr", models, tesseract_path, tessdata, timeout_s, None, setup, _tau(setup), True
        )
    if tesseract_path is None or weights_dir is None:
        return OcrSetup("none", models, None, None, timeout_s, None, None, None)
    problem, setup = _tesseract(weights_dir, tesseract_path)
    tau = None if setup is None else _tau(setup)
    tessdata = weights_dir / TESSDATA_DIR
    return OcrSetup(
        "tesseract",
        models,
        tesseract_path,
        tessdata,
        timeout_s,
        problem,
        setup,
        tau,
        problem is None,
    )


def _tau(setup: calibration.Setup) -> float | None:
    return calibration.lookup(setup.key)


ReaderStatus = Literal["missing", "bad_data", "uncalibrated", "ok"]


def _files_status(assets: tuple[Asset, ...], weights_dir: Path) -> ReaderStatus | None:
    problem = files_problem(assets, weights_dir)
    if problem is None:
        return None
    return "missing" if problem == "missing" else "bad_data"


def onnxtr_status(
    weights_dir: Path, tesseract_path: Path | None = None, *, runner: Runner | None = None
) -> ReaderStatus:
    """missing / bad_data (model files), uncalibrated (no frozen tau for the setup a read with
    `tesseract_path` would use; None reads by the recogniser vote) or ok. `runner` runs
    `tesseract --version`, so a caller can answer it once for several checks; without it the
    version this process decided for the program is used, the one every read uses."""
    problem = _files_status(ONNXTR, weights_dir)
    if problem is not None:
        return problem
    setup = onnxtr_setup(_osd_version(weights_dir, tesseract_path, runner=runner))
    return "uncalibrated" if _tau(setup) is None else "ok"


def tesseract_status(
    weights_dir: Path, binary: Path | None, *, runner: Runner | None = None
) -> ReaderStatus:
    """missing (program or language data), bad_data (language data, or a version this tool
    cannot read), uncalibrated (no frozen tau for its setup) or ok."""
    if binary is None:
        return "missing"
    problem = _files_status((*TESSDATA, TESSERACT_OSD), weights_dir)
    if problem is not None:
        return problem
    try:
        setup = tesseract_setup(_version(binary, runner))
    except TesseractFailure:
        return "missing"
    except ValueError:
        return "bad_data"
    return "uncalibrated" if _tau(setup) is None else "ok"


def engine(ocr: OcrSetup | None) -> Engine:
    """The engine record of a document; `ocr` is None when no page needed OCR."""
    if ocr is None:
        return Engine("none", None, "none", None, None, "untested", None)
    files = [*ONNXTR, *(TESSDATA if ocr.kind == "tesseract" else ())]
    digest = hashlib.sha256("".join(f"{a.name} {a.sha256}\n" for a in files).encode("ascii"))
    record = None if ocr.setup is None else Calibration(ocr.setup.key, ocr.tau)
    return Engine(
        detector=onnxtr_models.DETECTOR,
        weights_sha256=digest.hexdigest(),
        reader=ocr.kind,
        model_name=None,
        model_host=None,
        model_fit="untested",
        prompt_sha256=None,
        calibration=record,
    )
