"""Checks that on Windows a Tesseract program folio starts ends when folio ends (hard kill included),
that a program the job cannot take is stopped with a one-line error, and that a program that
already ended before it joined the job still gives its answer."""

import json
import os
import subprocess
import sys
import threading
import time

import pytest
from fake_tesseract_install import install_fake_tesseract
from pid_tools import pid_alive

from folio_ocr import job_object, tesseract

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects only")

CHILD = """
import os, sys, threading, time
from pathlib import Path
from folio_ocr import tesseract

program, work = Path(sys.argv[1]), Path(sys.argv[2])
threading.Thread(
    target=tesseract.run, args=(program, work, b"x", 120), daemon=True
).start()
log = work / "log.jsonl"
deadline = time.monotonic() + 40
while time.monotonic() < deadline:
    if log.exists() and log.read_text(encoding="utf-8").strip():
        break
    time.sleep(0.05)
os._exit(1)
"""


def test_a_running_tesseract_ends_when_folio_dies(tmp_path):
    program = install_fake_tesseract(tmp_path / "bin" / "tesseract.exe")
    log = tmp_path / "log.jsonl"
    env = {**os.environ, "FAKE_TESSERACT_LOG": str(log), "FAKE_TESSERACT_SLEEP": "60"}
    pid = None
    try:
        done = subprocess.run(
            [sys.executable, "-c", CHILD, str(program), str(tmp_path)],
            env=env,
            timeout=60,
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 1, done.stderr[-500:]
        pid = json.loads(log.read_text(encoding="utf-8").splitlines()[0])["pid"]
        assert not pid_alive(pid, patience=5.0)
    finally:
        if pid is None and log.exists() and log.read_text(encoding="utf-8").strip():
            pid = json.loads(log.read_text(encoding="utf-8").splitlines()[0])["pid"]
        if pid is not None and pid_alive(pid, patience=0.0):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False
            )


def test_a_program_folio_cannot_put_in_its_job_is_stopped_with_one_line(tmp_path, monkeypatch):
    program = install_fake_tesseract(tmp_path / "bin" / "tesseract.exe")
    monkeypatch.setenv("FAKE_TESSERACT_SLEEP", "60")

    def refuse(pid):
        raise OSError(8, "the Windows job object could not take the process")

    monkeypatch.setattr(job_object, "end_with_folio", refuse)
    started = time.monotonic()
    reading = tesseract.read_line(program, tmp_path, b"x", 60)
    assert reading.error == "Tesseract could not be started"
    assert time.monotonic() - started < 25


def test_a_program_that_ended_before_it_joined_the_job_still_answers(monkeypatch):
    def late(pid):
        assert not pid_alive(pid, patience=10)
        raise OSError(5, "gone")

    monkeypatch.setattr(job_object, "end_with_folio", late)
    done = tesseract._run_ending_with_folio(
        [sys.executable, "-c", "print('hi')"],
        input=None,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0
    assert done.stdout.strip() == b"hi"


def _run_a_sleeping_program_and_expect_the_stop():
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        tesseract._run_ending_with_folio(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            input=None,
            capture_output=True,
            timeout=30,
            check=False,
        )
    assert time.monotonic() - started < 20


def test_any_other_failure_while_it_runs_ends_the_program(monkeypatch):
    def stop(self, *args, **kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(subprocess.Popen, "communicate", stop)
    _run_a_sleeping_program_and_expect_the_stop()


def test_any_other_failure_joining_the_job_ends_the_program(monkeypatch):
    def stop(pid):
        raise RuntimeError("stop")

    monkeypatch.setattr(job_object, "end_with_folio", stop)
    _run_a_sleeping_program_and_expect_the_stop()


@pytest.mark.parametrize("input_size", [0, 128 * 1024], ids=["output", "input"])
def test_a_tesseract_timeout_stops_its_descendants(tmp_path, input_size):
    marker = tmp_path / "descendant"
    launcher = """
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         stdout=sys.stdout, stderr=sys.stderr)
Path(sys.argv[1]).write_text(str(child.pid))
child.wait()
"""
    timed_out = threading.Event()

    def run():
        with pytest.raises(subprocess.TimeoutExpired):
            tesseract._run_ending_with_folio(
                [sys._base_executable, "-c", launcher, str(marker)],
                input=b"x" * input_size if input_size else None,
                capture_output=True, timeout=2, check=False,
            )
        timed_out.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert timed_out.wait(7), "the deadline must stop a process holding an OCR pipe"
        assert marker.exists(), "the stand-in did not start its descendant before the deadline"
        assert not pid_alive(int(marker.read_text()))
    finally:
        if marker.exists() and pid_alive(pid := int(marker.read_text()), patience=0):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
        worker.join(timeout=5)
