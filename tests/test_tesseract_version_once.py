"""The Tesseract version is decided once per program file in a process, so every document of a
run gets the same reader setup, even when a version check runs out of time on a busy machine."""

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import accuracy_reads as ar

from folio_ocr import ocr_setup, onnxtr_models, tesseract
from folio_ocr.ocr_setup import onnxtr_setup, prepare, tesseract_setup

ANSWER = b"tesseract 5.4.0.20240606\n leptonica-1.84.1\n"
NEWER = b"tesseract 5.5.0\n"
REAL_VERSION = tesseract.tesseract_version
TOO_SLOW = "Tesseract did not report its version within 10 s"


class FakeVersion:
    """`tesseract --version`: each call takes the next answer ("timeout" or the bytes printed);
    the last answer repeats."""

    def __init__(self, *answers, pause_s=0.0):
        self.answers, self.calls, self.pause_s = list(answers), 0, pause_s
        self._lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        assert argv[1:] == ["--version"]
        with self._lock:
            self.calls += 1
            answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        time.sleep(self.pause_s)
        if answer == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, answer, b"")


@pytest.fixture
def program(tmp_path, monkeypatch):
    """A program file whose `--version` the test answers; model and language files pass."""
    binary = tmp_path / "tesseract.exe"
    binary.write_bytes(b"a program")
    binary.chmod(0o755)  # a program file the system would run
    monkeypatch.setattr(onnxtr_models, "load", lambda folder: "models")
    monkeypatch.setattr(ocr_setup, "version", lambda _name: "0.9.0")
    monkeypatch.setattr(ocr_setup, "files_problem", lambda assets, folder: None)
    monkeypatch.setattr(tesseract, "tessdata_problem", lambda folder: None)
    monkeypatch.setattr(ocr_setup, "_tau", lambda setup: None)
    return binary


def answering(monkeypatch, fake: FakeVersion) -> FakeVersion:
    def version(binary, timeout_s=10.0, runner=None):
        return REAL_VERSION(binary, timeout_s, runner=fake)

    monkeypatch.setattr(tesseract, "tesseract_version", version)
    return fake


def english_key(folder: Path, binary: Path) -> str:
    return prepare("en", folder, binary, 7.5).setup.key


def test_a_first_check_that_times_out_leaves_every_document_one_key(program, tmp_path, monkeypatch):
    fake = answering(monkeypatch, FakeVersion("timeout", ANSWER))
    keys = {english_key(tmp_path, program) for _ in range(3)}
    assert keys == {onnxtr_setup("5.4").key}
    assert prepare("ar", tmp_path, program, 7.5).setup == tesseract_setup("5.4.0")
    assert fake.calls == 2  # the timed-out check and its one retry, then never again


def test_a_check_that_always_fails_makes_every_document_vote_and_says_why(
    program, tmp_path, monkeypatch
):
    fake = answering(monkeypatch, FakeVersion("timeout"))
    english = [prepare("en", tmp_path, program, 7.5) for _ in range(3)]
    assert {r.setup.key for r in english} == {onnxtr_setup().key}
    assert not any(r.osd for r in english)
    arabic = prepare("ar", tmp_path, program, 7.5)
    assert arabic.problem == TOO_SLOW and arabic.setup is None and arabic.osd is False
    assert ocr_setup.tesseract_status(tmp_path, program) == "missing"
    assert fake.calls == 2


def test_a_failed_decision_holds_even_when_a_later_check_would_answer(
    program, tmp_path, monkeypatch
):
    answering(monkeypatch, FakeVersion("timeout", "timeout", ANSWER))
    keys = {english_key(tmp_path, program) for _ in range(4)}
    assert keys == {onnxtr_setup().key}


def test_only_a_check_that_ran_out_of_time_is_asked_again(program, tmp_path, monkeypatch):
    fake = answering(monkeypatch, FakeVersion(b"no version here", ANSWER))
    assert {english_key(tmp_path, program) for _ in range(3)} == {onnxtr_setup().key}
    assert prepare("ar", tmp_path, program, 7.5).problem == "Tesseract did not report its version"
    assert fake.calls == 1


def test_a_changed_program_file_is_asked_again(program, tmp_path, monkeypatch):
    fake = answering(monkeypatch, FakeVersion(ANSWER, NEWER))
    first = english_key(tmp_path, program)
    assert english_key(tmp_path, program) == first == onnxtr_setup("5.4").key
    info = program.stat()
    os.utime(program, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
    assert english_key(tmp_path, program) == onnxtr_setup("5.5").key
    assert fake.calls == 2


def test_a_program_file_of_another_size_is_asked_again(program, tmp_path, monkeypatch):
    fake = answering(monkeypatch, FakeVersion(ANSWER, NEWER))
    assert english_key(tmp_path, program) == english_key(tmp_path, program)
    assert english_key(tmp_path, program) == onnxtr_setup("5.4").key
    info = program.stat()
    program.write_bytes(b"another, longer program")
    os.utime(program, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert english_key(tmp_path, program) == onnxtr_setup("5.5").key
    assert fake.calls == 2


def test_documents_read_at_once_share_one_check(program, tmp_path, monkeypatch):
    fake = answering(monkeypatch, FakeVersion("timeout", ANSWER, pause_s=0.05))
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = set(pool.map(lambda _: english_key(tmp_path, program), range(8)))
    assert keys == {onnxtr_setup("5.4").key}
    assert fake.calls == 2


def test_the_runner_expects_the_keys_the_reads_get_after_a_timed_out_check(
    program, tmp_path, monkeypatch
):
    answering(monkeypatch, FakeVersion("timeout", ANSWER))
    expected = ar.expected_keys(program)
    assert english_key(tmp_path, program) == expected["onnxtr-en-osd"]
    assert prepare("ar", tmp_path, program, 7.5).setup.key == expected["tesseract-ar"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows runs a program named without .exe")
def test_a_program_named_without_its_exe_suffix_is_decided_once(program, tmp_path, monkeypatch):
    fake = answering(monkeypatch, FakeVersion("timeout", ANSWER))
    named = program.with_suffix("")
    assert {english_key(tmp_path, named) for _ in range(3)} == {onnxtr_setup("5.4").key}
    assert fake.calls == 2
