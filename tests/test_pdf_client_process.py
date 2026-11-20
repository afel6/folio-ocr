"""The reader process client: memory limit, reaping, start failures and the closed request list."""

import os
import subprocess
import sys

import pytest
from pid_tools import pid_alive
from sample_files import pdf_bytes

from folio_ocr import ingest, pdf_client, pdf_frames, pipeline
from folio_ocr.errors import InputError, PdfReaderError

LIMIT = pdf_client.limit_line("pdf")
PDF = pdf_bytes([(612, 792)])
START_LINE = "PDFs cannot be read here: folio could not start its PDF reader with limits"
SUPPORTED = sys.platform == "win32" or sys.platform.startswith("linux")
HUGE_PAGE = pdf_bytes([(1690, 1690)])  # about 49 MP at 300 dpi

CHILD = """
import os, sys
from folio_ocr import pdf_client
with pdf_client.opened(bytes.fromhex(sys.argv[1]), "pdf") as f:
    print(f._worker.pid, flush=True)
    os._exit(1)
"""


BUSY_CHILD = """
import os, sys, threading, time
sys.path.insert(0, sys.argv[2])
from pid_tools import fake_command
from folio_ocr import pdf_client
pdf_client._command = lambda: fake_command("sleep_on_page")
with pdf_client.opened(b"x", "pdf") as f:
    mid = threading.Event()
    def ask():
        mid.set()
        f.page(0)
    threading.Thread(target=ask, daemon=True).start()
    mid.wait()
    time.sleep(1)
    print(f._worker.pid, flush=True)
    os._exit(1)
"""


@pytest.mark.skipif(not SUPPORTED, reason="needs a memory limit")
def test_a_page_over_the_memory_limit_gives_the_limit_line(monkeypatch):
    monkeypatch.setattr(ingest, "PDF_MEMORY_MB", 100)
    with pytest.raises(InputError) as err:
        pipeline.page_image(HUGE_PAGE, 1)
    # Linux has no memory-limit message: PDFium may report the failed bitmap as a damaged page
    allowed = {LIMIT} if sys.platform == "win32" else {LIMIT, ingest.page_damaged(1)}
    assert str(err.value) in allowed
    assert pdf_client.live_workers() == 0


@pytest.mark.skipif(not SUPPORTED, reason="needs a memory limit")
def test_the_same_page_renders_under_the_normal_limit():
    assert pipeline.page_image(HUGE_PAGE, 1).size == ingest.render_size(1690, 1690)


def test_an_exception_in_the_block_ends_the_reader():
    pid = None
    with pytest.raises(RuntimeError), pdf_client.opened(PDF, "pdf") as file:
        pid = file._worker.pid
        raise RuntimeError("boom")
    assert not pid_alive(pid) and pdf_client.live_workers() == 0


def test_an_interrupt_in_the_block_ends_the_reader():
    pid = None
    with pytest.raises(KeyboardInterrupt), pdf_client.opened(PDF, "pdf") as file:
        pid = file._worker.pid
        raise KeyboardInterrupt
    assert not pid_alive(pid) and pdf_client.live_workers() == 0


def test_the_reader_ends_when_folio_dies():
    done = subprocess.run(
        [sys.executable, "-c", CHILD, PDF.hex()],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=os.getcwd(),
    )
    assert done.returncode == 1, done.stderr[-300:]
    assert not pid_alive(int(done.stdout.split()[0]))


def test_a_busy_reader_ends_when_folio_dies():
    here = os.path.dirname(os.path.abspath(__file__))
    done = subprocess.run(
        [sys.executable, "-c", BUSY_CHILD, PDF.hex(), here],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=os.getcwd(),
    )
    assert done.returncode == 1, done.stderr[-300:]
    assert not pid_alive(int(done.stdout.split()[0]), patience=5.0)


def test_the_reader_imports_no_heavy_library():
    code = (
        "import sys, folio_ocr.pdf_worker; "
        "print(sorted(m for m in ('numpy','onnxtr','cv2','onnxruntime') if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-P", "-c", code], check=False, capture_output=True, text=True, timeout=60
    )
    assert done.stdout.strip() == "[]", done.stderr[-300:]


def test_python_is_started_without_the_working_folder_on_its_path(monkeypatch, tmp_path):
    (tmp_path / "pypdfium2.py").write_text("raise RuntimeError('planted')\n")
    (tmp_path / "folio_ocr").mkdir()
    (tmp_path / "folio_ocr" / "__init__.py").write_text("raise RuntimeError('planted')\n")
    monkeypatch.chdir(tmp_path)
    with pdf_client.opened(PDF, "pdf") as file:
        assert file.count == 1


def assert_start_failure(err):
    assert str(err.value) == START_LINE and err.value.reason
    assert pdf_client.live_workers() == 0


def test_a_process_that_cannot_start_is_one_line(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError(2, "no such file")

    monkeypatch.setattr(pdf_client.subprocess, "Popen", refuse)
    with pytest.raises(PdfReaderError) as err, pdf_client.opened(PDF, "pdf"):
        pass
    assert_start_failure(err)


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_a_job_object_that_cannot_be_made_is_one_line(monkeypatch):
    from folio_ocr import job_object

    def refuse(*args, **kwargs):
        raise OSError(5, "the Windows job object could not be created")

    monkeypatch.setattr(job_object, "JobObject", refuse)
    with pytest.raises(PdfReaderError) as err, pdf_client.opened(PDF, "pdf"):
        pass
    assert_start_failure(err)


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows only")
def test_a_process_the_job_cannot_take_is_one_line(monkeypatch):
    from folio_ocr import job_object

    def refuse(self, pid):
        raise OSError(5, "the Windows job object could not take the reader process")

    monkeypatch.setattr(job_object.JobObject, "adopt", refuse)
    with pytest.raises(PdfReaderError) as err, pdf_client.opened(PDF, "pdf"):
        pass
    assert_start_failure(err)


@pytest.mark.parametrize(
    ("header", "index_note"),
    [({"op": "bogus"}, "unknown op"), ({"op": "page", "index": 0}, "page before open"),
     ({"op": "render", "index": "0"}, "index is a string")],
)  # fmt: skip
def test_a_request_outside_the_closed_list_ends_the_reader(header, index_note):
    with pdf_client._running("pdf") as worker:
        pdf_frames.write(worker._proc.stdin, header)
        assert worker._proc.wait(5) == 2, index_note
        with pytest.raises(pdf_frames.Ended):
            pdf_frames.read_header(worker._proc.stdout, 100)
    assert pdf_client.live_workers() == 0
