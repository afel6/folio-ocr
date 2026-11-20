"""Choosing the OCR reader of a document and routing pages to it, tested with doubles."""

import dataclasses
import hashlib
import io
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from sample_files import pdf_bytes, tiff

from folio_ocr import ocr_setup, onnxtr_models, orientation, tesseract
from folio_ocr.assets import ONNXTR_DETECTOR, TESSDATA
from folio_ocr.document import Calibration, Engine
from folio_ocr.errors import WeightsError
from folio_ocr.ocr_setup import OcrSetup, engine, onnxtr_setup, prepare
from folio_ocr.pipeline import read_bytes, read_path
from folio_ocr.tesseract import TesseractFailure

FIXTURES = Path(__file__).parent / "fixtures"
BINARY = Path("/opt/tesseract")
ARA_SHA = "ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896"
ENG_SHA = "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba"
RECOGNISER_SHA = "743599aa33cb4d254e14f3fe013f1c2a49eaff1329a7ec451844a0caa4f84bc5"


@pytest.fixture(autouse=True)
def doubles(monkeypatch):
    monkeypatch.setattr(onnxtr_models, "load", lambda folder: "models")
    monkeypatch.setattr(ocr_setup, "version", lambda _name: "0.9.0")
    monkeypatch.setattr(ocr_setup, "_tau", lambda setup: 0.5)


def tesseract_works(monkeypatch, version="5.4.0.20240606", problem=None):
    monkeypatch.setattr(tesseract, "tessdata_problem", lambda folder: problem)
    monkeypatch.setattr(tesseract, "tesseract_version", lambda binary, runner=None: version)


# --- prepare -------------------------------------------------------------------------------


def test_english_is_read_by_onnxtr(tmp_path):
    result = prepare("en", tmp_path, BINARY, 7.5)
    assert result.kind == "onnxtr" and result.models == "models"
    assert result.setup == onnxtr_setup()
    assert result.tau == 0.5
    assert (result.tesseract, result.tessdata, result.problem, result.timeout_s) == (
        None,
        None,
        None,
        7.5,
    )


@pytest.mark.parametrize("languages", ["ar", "en+ar"])
def test_arabic_without_tesseract_has_no_reader(tmp_path, languages):
    result = prepare(languages, tmp_path, None, 7.5)
    assert result.kind == "none" and result.models == "models"
    assert result.setup is None and result.tau is None and result.problem is None


def test_missing_model_files_stop_before_any_reader_is_chosen(tmp_path, monkeypatch):
    def refuse(folder):
        raise WeightsError("the OCR model files are missing; run folio setup")

    monkeypatch.setattr(onnxtr_models, "load", refuse)
    with pytest.raises(WeightsError):
        prepare("ar", tmp_path, None, 7.5)


def test_missing_language_data_is_a_problem_not_an_error(tmp_path):
    result = prepare("ar", tmp_path, BINARY, 7.5)
    assert result.kind == "tesseract"
    assert result.problem == "Tesseract language data is missing; run folio setup"
    assert result.setup is None and result.tau is None
    assert (result.tesseract, result.tessdata) == (BINARY, tmp_path / "tessdata")


def test_a_failed_version_check_is_a_problem(tmp_path, monkeypatch):
    def fail(binary):
        raise TesseractFailure("Tesseract did not report its version")

    tesseract_works(monkeypatch)
    monkeypatch.setattr(tesseract, "tesseract_version", fail)
    result = prepare("ar", tmp_path, BINARY, 7.5)
    assert result.problem == "Tesseract did not report its version"
    assert result.setup is None


def test_a_version_that_cannot_be_reduced_is_a_problem(tmp_path, monkeypatch):
    tesseract_works(monkeypatch, version="five")
    result = prepare("ar", tmp_path, BINARY, 7.5)
    assert result.problem == "Tesseract reported a version this tool does not understand"
    assert result.setup is None and result.tau is None


def test_a_working_tesseract_gets_a_setup_keyed_by_its_files_and_arguments(tmp_path, monkeypatch):
    tesseract_works(monkeypatch)
    result = prepare("en+ar", tmp_path, BINARY, 7.5)
    assert result.problem is None and result.tau == 0.5
    assert result.setup.key.startswith(
        f"tesseract|lstm|{ARA_SHA}+{ENG_SHA}|min|5.4|-l ara+eng --psm 13 --oem 1|"
    )


def test_the_onnxtr_setup_key_names_the_recogniser_and_its_version():
    assert onnxtr_setup().key.startswith(
        f"onnxtr|crnn_vgg16_bn|{RECOGNISER_SHA}|min;split=min|0.9||"
    )


def test_the_three_frozen_setup_keys_are_pinned():
    """The calibration table's taus belong to exactly these keys; changing one voids them."""
    from folio_ocr.ocr_setup import tesseract_setup

    models = "onnxtr|crnn_vgg16_bn|" + RECOGNISER_SHA + "|min;split=min|0.9||"
    assert onnxtr_setup().key == (
        models + "c208ceaf3e66097f3a0ddb74dc228fd37d83039c7cb1e27d733c8bcd2c415f9f"
    )
    assert onnxtr_setup("5.4").key == (
        models + "8650601719ff75586c215ed13693e058729e220d8fb1342c0278850cf68989aa"
    )
    assert tesseract_setup("5.4.0.20240606").key == (
        f"tesseract|lstm|{ARA_SHA}+{ENG_SHA}|min|5.4|-l ara+eng --psm 13 --oem 1|"
        "2603f2951dfdde7bf673852f7cce32910a508d22b466bcbf30f977868e2e0b91"
    )


# --- what the preprocessing hash depends on -----------------------------------------------


def changed(monkeypatch, name, value):
    monkeypatch.setattr(ocr_setup, name, value)
    return onnxtr_setup().preprocessing


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OCR_GAP_FACTOR", 1.4),
        ("LINE_INK_MARGIN", 3.0),
        ("RESCUE_REGIONS", 8),
        ("WINDOW_MIN_PX", 32),
        ("WINDOW_HEIGHTS", 3),
        ("ONNXTR_DETECTOR", dataclasses.replace(ONNXTR_DETECTOR, sha256="0" * 64)),
    ],
)
def test_the_preprocessing_hash_follows_each_constant(monkeypatch, name, value):
    before = onnxtr_setup().preprocessing
    assert onnxtr_setup().preprocessing == before
    assert changed(monkeypatch, name, value) != before


@pytest.mark.parametrize(("name", "value"), [("SPLIT_RATIO", 9), ("SPLIT_TARGET", 5)])
def test_the_preprocessing_hash_follows_the_crop_split(monkeypatch, name, value):
    before = onnxtr_setup().preprocessing
    monkeypatch.setattr(onnxtr_models, name, value)
    assert onnxtr_setup().preprocessing != before


def test_the_preprocessing_hash_follows_the_split_overlap(monkeypatch):
    before = onnxtr_setup().preprocessing
    monkeypatch.setattr(onnxtr_models, "SPLIT_OVERLAP", 0.4)
    assert onnxtr_setup().preprocessing != before


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SIDEWAYS_SHARE", 0.6),
        ("VOTE_QUANTILE", 0.5),
        ("VOTE_WORDS", 20),
        ("UNCERTAIN_MARGIN_PAIR", 0.2),
        ("UNCERTAIN_MARGIN_FOUR", 0.2),
    ],
)
def test_the_preprocessing_hash_follows_each_orientation_constant(monkeypatch, name, value):
    before = onnxtr_setup().preprocessing
    monkeypatch.setattr(orientation, name, value)
    assert onnxtr_setup().preprocessing != before


# --- orientation method: Tesseract OSD or the recogniser vote ------------------------------


def osd_file(monkeypatch, problem=None):
    monkeypatch.setattr(ocr_setup, "files_problem", lambda assets, folder: problem)


def test_english_uses_osd_only_with_a_known_version_and_an_intact_osd_file(tmp_path, monkeypatch):
    osd_file(monkeypatch)
    tesseract_works(monkeypatch)
    chosen = prepare("en", tmp_path, BINARY, 7.5)
    assert chosen.kind == "onnxtr" and chosen.osd is True
    assert (chosen.tesseract, chosen.tessdata) == (BINARY, tmp_path / "tessdata")
    assert chosen.setup == onnxtr_setup("5.4")


def test_english_without_tesseract_votes(tmp_path, monkeypatch):
    osd_file(monkeypatch)
    chosen = prepare("en", tmp_path, None, 7.5)
    assert chosen.osd is False and chosen.setup == onnxtr_setup()
    assert (chosen.tesseract, chosen.tessdata) == (None, None)


def test_english_with_an_unknown_tesseract_version_votes(tmp_path, monkeypatch):
    osd_file(monkeypatch)
    tesseract_works(monkeypatch, version="five")
    chosen = prepare("en", tmp_path, BINARY, 7.5)
    assert chosen.osd is False and chosen.setup == onnxtr_setup()
    version_is(monkeypatch, TesseractFailure("Tesseract did not report its version"))
    assert prepare("en", tmp_path, BINARY, 7.5).osd is False


@pytest.mark.parametrize("problem", ["missing", "damaged", "cannot read"])
def test_english_with_a_bad_osd_file_votes(tmp_path, monkeypatch, problem):
    osd_file(monkeypatch, problem)
    tesseract_works(monkeypatch)
    chosen = prepare("en", tmp_path, BINARY, 7.5)
    assert chosen.osd is False and chosen.setup == onnxtr_setup()


def test_the_osd_file_is_the_one_checked(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(
        ocr_setup, "files_problem", lambda assets, folder: asked.append((assets, folder))
    )
    tesseract_works(monkeypatch)
    prepare("en", tmp_path, BINARY, 7.5)
    assert asked == [((ocr_setup.TESSERACT_OSD,), tmp_path)]


def test_the_two_onnxtr_setups_have_different_keys():
    assert onnxtr_setup().key != onnxtr_setup("5.4").key
    assert onnxtr_setup("5.4").key != onnxtr_setup("5.3").key


def test_the_orientation_text_names_the_method():
    assert onnxtr_setup().preprocessing != onnxtr_setup("5.4").preprocessing
    assert ocr_setup._ORIENTATION_VOTE == "long-box-sideways;crnn-vote"
    assert ocr_setup._ORIENTATION_OSD == "long-box-sideways;tesseract-osd;crnn-vote-fallback"


@pytest.mark.parametrize("languages", ["ar", "en+ar"])
def test_a_tesseract_setup_always_uses_osd(tmp_path, monkeypatch, languages):
    tesseract_works(monkeypatch)
    chosen = prepare(languages, tmp_path, BINARY, 7.5)
    assert chosen.kind == "tesseract" and chosen.osd is True
    assert chosen.setup == ocr_setup.tesseract_setup("5.4.0")
    assert chosen.setup.preprocessing != onnxtr_setup().preprocessing


def test_a_tesseract_problem_leaves_no_osd_to_run(tmp_path):
    chosen = prepare("ar", tmp_path, BINARY, 7.5)
    assert chosen.problem is not None and chosen.osd is False


def test_no_reader_has_no_osd(tmp_path):
    assert prepare("ar", tmp_path, None, 7.5).osd is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SIDEWAYS_SHARE", 0.6),
        ("LONG_RATIO", 2.0),
        ("UNCERTAIN_MARGIN_PAIR", 0.2),
        ("UNCERTAIN_MARGIN_FOUR", 0.2),
    ],
)
def test_the_osd_keys_follow_the_orientation_constants(monkeypatch, name, value):
    for make in (lambda: onnxtr_setup("5.4").key, lambda: ocr_setup.tesseract_setup("5.4").key):
        before = make()
        monkeypatch.setattr(orientation, name, value)
        assert make() != before
        monkeypatch.undo()


def test_the_vote_key_ignores_the_osd_floor(monkeypatch):
    before = onnxtr_setup().key
    monkeypatch.setattr(orientation, "OSD_FLOOR", 9.0)
    assert onnxtr_setup().key == before


def altered(monkeypatch, make, module, name, value):
    """`make()` with `module.name` set to `value` for the call only."""
    with monkeypatch.context() as patch:
        patch.setattr(module, name, value)
        return make()


NO_OSD_SHA = dataclasses.replace(ocr_setup.TESSERACT_OSD, sha256="0" * 64)


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        (orientation, "OSD_FLOOR", 9.0),
        (ocr_setup, "TESSERACT_OSD", NO_OSD_SHA),
        (tesseract, "OSD_ARGUMENTS", ("--psm", "1")),
    ],
)
def test_both_osd_keys_follow_the_floor_the_file_and_the_arguments(
    monkeypatch, module, name, value
):
    for make in (lambda: onnxtr_setup("5.4").key, lambda: ocr_setup.tesseract_setup("5.4").key):
        assert altered(monkeypatch, make, module, name, value) != make()


def test_the_osd_key_follows_the_tesseract_version():
    assert onnxtr_setup("5.5").key != onnxtr_setup("5.4").key


# --- engine record -------------------------------------------------------------------------


def test_no_ocr_means_no_engine():
    assert engine(None) == Engine("none", None, "none", None, None, "untested", None)
    assert engine(None).calibration is None


def digest(*assets):
    text = "".join(f"{a.name} {a.sha256}\n" for a in assets)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def test_the_onnxtr_engine_records_its_files_and_calibration():
    setup = onnxtr_setup()
    found = engine(OcrSetup("onnxtr", None, None, None, 5.0, None, setup, 0.5))
    assert (found.detector, found.reader) == ("fast_base", "onnxtr")
    assert found.calibration == Calibration(setup.key, 0.5)
    assert found.weights_sha256 == digest(*ocr_setup.ONNXTR)
    assert (found.model_name, found.model_host, found.model_fit) == (None, None, "untested")


def test_the_tesseract_engine_also_records_the_language_data(tmp_path, monkeypatch):
    tesseract_works(monkeypatch)
    chosen = prepare("ar", tmp_path, BINARY, 5.0)
    found = engine(chosen)
    assert found.reader == "tesseract" and found.detector == "fast_base"
    assert found.weights_sha256 == digest(*ocr_setup.ONNXTR, *TESSDATA)
    assert found.weights_sha256 != engine(prepare("en", tmp_path, BINARY, 5.0)).weights_sha256
    assert found.calibration == Calibration(chosen.setup.key, 0.5)


def test_an_engine_without_a_setup_has_no_calibration():
    found = engine(OcrSetup("tesseract", None, BINARY, None, 5.0, "x", None, None))
    assert found.calibration is None and found.reader == "tesseract"


def test_a_reader_with_no_threshold_records_none(tmp_path):
    chosen = dataclasses.replace(prepare("en", tmp_path, None, 5.0), tau=None)
    assert engine(chosen).calibration == Calibration(chosen.setup.key, None)


# --- the pipeline and the OCR step ---------------------------------------------------------


def two_scanned_frames():
    return tiff([(10, 10), (20, 10)])


def mixed_pdf():
    """A scanned page followed by a page with a text layer."""
    merged = pdfium.PdfDocument.new()
    for name in ("ar-invoice-scanned.pdf", "en-invoice.pdf"):
        src = pdfium.PdfDocument(str(FIXTURES / name))
        merged.import_pages(src)
        src.close()
    buf = io.BytesIO()
    merged.save(buf)
    merged.close()
    return buf.getvalue()


def test_ocr_pages_are_read_after_pdfium_is_released(ocr_stub):
    doc = read_bytes(mixed_pdf(), "m.pdf", languages="en+ar")
    scanned, text = doc.pages
    assert [(c.page_no, c.pdfium_locked) for c in ocr_stub.pages] == [(1, False)]
    assert scanned.origin == "ocr" and scanned.index == 1
    assert ocr_stub.pages[0].reasons == ("empty_with_ink",)
    assert ocr_stub.pages[0].invalid_lines == 0
    alone = read_bytes((FIXTURES / "en-invoice.pdf").read_bytes(), "a.pdf", languages="en+ar")
    assert text.origin == "text_layer" and text.index == 2
    assert [ln.text for ln in text.lines] == [ln.text for ln in alone.pages[0].lines]
    assert (text.width, text.height) == (alone.pages[0].width, alone.pages[0].height)


def test_a_document_of_text_layer_pages_needs_no_model_files(ocr_stub, monkeypatch):
    def refuse(*_args):
        raise WeightsError("the OCR model files are not set up; run folio setup")

    monkeypatch.setattr("folio_ocr.pipeline.prepare", refuse)
    doc = read_bytes(pdf_bytes([(100, 100)] * 3), "t.pdf", languages="en")
    assert len(doc.pages) == 3 and ocr_stub.pages == []
    assert doc.engine == engine(None)
    doc = read_path(FIXTURES / "en-invoice.pdf", languages="en", weights_dir=None)
    assert doc.engine == engine(None)


def test_the_reader_is_chosen_once_per_document(ocr_stub):
    doc = read_bytes(two_scanned_frames(), "s.tiff", languages="en")
    assert len(doc.pages) == 2
    assert len(ocr_stub.prepared) == 1
    assert [c.page_no for c in ocr_stub.pages] == [1, 2]
    assert doc.engine == engine(ocr_stub.setup)


def test_missing_model_files_stop_a_document_that_needs_ocr(ocr_stub, monkeypatch):
    def refuse(*_args):
        raise WeightsError("the OCR model files are missing; run folio setup")

    monkeypatch.setattr("folio_ocr.pipeline.prepare", refuse)
    with pytest.raises(WeightsError) as info:
        read_bytes(two_scanned_frames(), "s.tiff", languages="en")
    assert str(info.value) == "the OCR model files are missing; run folio setup"


def test_read_bytes_hands_its_settings_to_the_reader_choice(ocr_stub, tmp_path):
    read_bytes(
        two_scanned_frames(),
        "s.tiff",
        languages="en+ar",
        weights_dir=tmp_path,
        tesseract_path=BINARY,
        line_timeout_s=7.5,
    )
    assert ocr_stub.prepared == [("en+ar", tmp_path, BINARY, 7.5)]


def test_read_path_hands_its_settings_to_the_reader_choice(ocr_stub, tmp_path):
    target = tmp_path / "s.tiff"
    target.write_bytes(two_scanned_frames())
    read_path(
        target, languages="ar", weights_dir=tmp_path, tesseract_path=BINARY, line_timeout_s=9.0
    )
    read_path(target, languages="ar")
    assert ocr_stub.prepared == [("ar", tmp_path, BINARY, 9.0), ("ar", None, None, 60.0)]


# --- reader status (folio doctor) -----------------------------------------------------------


def files_say(monkeypatch, problem):
    monkeypatch.setattr(ocr_setup, "files_problem", lambda assets, folder: problem)


def version_is(monkeypatch, result):
    def check(binary, **_kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(tesseract, "tesseract_version", check)


def tau_only_for(monkeypatch, setup):
    monkeypatch.setattr(ocr_setup, "_tau", lambda found: 0.5 if found.key == setup.key else None)


@pytest.mark.parametrize(
    ("problem", "status"),
    [("missing", "missing"), ("cannot read", "bad_data"), ("damaged", "bad_data")],
)
def test_onnxtr_status_follows_the_model_files(tmp_path, monkeypatch, problem, status):
    files_say(monkeypatch, problem)
    assert ocr_setup.onnxtr_status(tmp_path) == status


def test_onnxtr_status_asks_about_the_onnxtr_files_in_the_given_folder(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(
        ocr_setup, "files_problem", lambda assets, folder: asked.append((assets, folder))
    )
    ocr_setup.onnxtr_status(tmp_path)
    assert asked == [(ocr_setup.ONNXTR, tmp_path)]


def test_onnxtr_status_uncalibrated_then_ok(tmp_path, monkeypatch):
    files_say(monkeypatch, None)
    monkeypatch.setattr(ocr_setup, "_tau", lambda setup: None)
    assert ocr_setup.onnxtr_status(tmp_path) == "uncalibrated"
    tau_only_for(monkeypatch, onnxtr_setup())
    assert ocr_setup.onnxtr_status(tmp_path) == "ok"


def test_tesseract_status_without_the_program_is_missing(tmp_path, monkeypatch):
    files_say(monkeypatch, "damaged")
    assert ocr_setup.tesseract_status(tmp_path, None) == "missing"


@pytest.mark.parametrize(
    ("problem", "status"),
    [("missing", "missing"), ("cannot read", "bad_data"), ("damaged", "bad_data")],
)
def test_tesseract_status_follows_the_language_data(tmp_path, monkeypatch, problem, status):
    files_say(monkeypatch, problem)
    version_is(monkeypatch, "5.4.0")
    assert ocr_setup.tesseract_status(tmp_path, BINARY) == status


def test_tesseract_status_asks_about_the_language_data_and_the_orientation_file(
    tmp_path, monkeypatch
):
    asked = []
    monkeypatch.setattr(
        ocr_setup, "files_problem", lambda assets, folder: asked.append((assets, folder))
    )
    version_is(monkeypatch, "5.4.0")
    ocr_setup.tesseract_status(tmp_path, BINARY)
    assert asked == [((*ocr_setup.TESSDATA, ocr_setup.TESSERACT_OSD), tmp_path)]


def test_tesseract_status_a_failed_version_check_is_missing(tmp_path, monkeypatch):
    files_say(monkeypatch, None)
    version_is(monkeypatch, TesseractFailure("did not run"))
    assert ocr_setup.tesseract_status(tmp_path, BINARY) == "missing"


@pytest.mark.parametrize("shown", ["tesseract vX", "five", "5"])
def test_tesseract_status_an_unreadable_version_is_bad_data(tmp_path, monkeypatch, shown):
    files_say(monkeypatch, None)
    version_is(monkeypatch, shown)
    assert ocr_setup.tesseract_status(tmp_path, BINARY) == "bad_data"


def test_tesseract_status_uncalibrated_then_ok_for_the_exact_key(tmp_path, monkeypatch):
    files_say(monkeypatch, None)
    version_is(monkeypatch, "5.4.0.20240606")
    monkeypatch.setattr(ocr_setup, "_tau", lambda setup: None)
    assert ocr_setup.tesseract_status(tmp_path, BINARY) == "uncalibrated"
    tau_only_for(monkeypatch, ocr_setup.tesseract_setup("5.4.0"))
    assert ocr_setup.tesseract_status(tmp_path, BINARY) == "ok"
    version_is(monkeypatch, "5.3.0")  # another version has another key
    assert ocr_setup.tesseract_status(tmp_path, BINARY) == "uncalibrated"


def test_tesseract_status_passes_its_runner_to_the_version_check(tmp_path, monkeypatch):
    files_say(monkeypatch, None)
    seen = []
    monkeypatch.setattr(
        tesseract, "tesseract_version", lambda b, runner=None: seen.append(runner) or "5.4.0"
    )
    runner = object()
    ocr_setup.tesseract_status(tmp_path, BINARY, runner=runner)
    assert seen == [runner]


def test_onnxtr_status_reports_the_setup_a_read_with_that_tesseract_would_use(
    tmp_path, monkeypatch
):
    files_say(monkeypatch, None)
    tesseract_works(monkeypatch)
    tau_only_for(monkeypatch, onnxtr_setup("5.4"))
    assert ocr_setup.onnxtr_status(tmp_path, BINARY) == "ok"
    assert ocr_setup.onnxtr_status(tmp_path, None) == "uncalibrated"
    assert ocr_setup.onnxtr_status(tmp_path) == "uncalibrated"
    tau_only_for(monkeypatch, onnxtr_setup())
    assert ocr_setup.onnxtr_status(tmp_path) == "ok"
    assert ocr_setup.onnxtr_status(tmp_path, BINARY) == "uncalibrated"


def test_onnxtr_status_with_an_unusable_tesseract_uses_the_vote_setup(tmp_path, monkeypatch):
    files_say(monkeypatch, None)
    tesseract_works(monkeypatch, version="five")
    tau_only_for(monkeypatch, onnxtr_setup())
    assert ocr_setup.onnxtr_status(tmp_path, BINARY) == "ok"


def test_onnxtr_status_asks_the_given_runner_for_the_tesseract_version(tmp_path, monkeypatch):
    files_say(monkeypatch, None)
    monkeypatch.setattr(ocr_setup, "files_problem", lambda assets, folder: None)
    seen = []
    monkeypatch.setattr(
        tesseract, "tesseract_version", lambda b, runner=None: seen.append(runner) or "5.4.0"
    )
    runner = object()
    ocr_setup.onnxtr_status(tmp_path, BINARY, runner=runner)
    assert seen == [runner]
