"""The reader process client against a stand-in reader that misbehaves, and the real reader."""

import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from pid_tools import fake_command, pid_alive
from sample_files import pdf_bytes

from folio_ocr import ingest, job_object, pdf_client, pdf_frames, pipeline
from folio_ocr.errors import InputError, PdfReaderError

LIMIT = pdf_client.limit_line("pdf")
PDF = pdf_bytes([(612, 792)])


def use_fake(monkeypatch, behaviour):
    monkeypatch.setattr(pdf_client, "_command", lambda: fake_command(behaviour))


def short_file_budget(monkeypatch, seconds):
    """The reader starts with the normal budget; the file's work then gets `seconds`."""
    start = pdf_client._Worker.start

    def started(self):
        start(self)
        self._left = seconds

    monkeypatch.setattr(pdf_client._Worker, "start", started)


def open_and_page(data=b"x"):
    with pdf_client.opened(data, "pdf") as file:
        file.page(0)


@pytest.mark.parametrize("behaviour", ["exit_before_ready", "not_ready"])
def test_a_reader_that_does_not_get_ready_is_a_start_failure(monkeypatch, behaviour):
    use_fake(monkeypatch, behaviour)
    with pytest.raises(PdfReaderError) as err:
        open_and_page()
    assert err.value.reason
    assert pdf_client.live_workers() == 0


@pytest.mark.parametrize(
    "behaviour",
    ["sleep_on_page", "wrong_render_size", "wrong_byte_count", "bad_char", "bool_char"],
)
def test_a_bad_page_reply_gives_the_limit_line(monkeypatch, behaviour):
    use_fake(monkeypatch, behaviour)
    short_file_budget(monkeypatch, 2)
    with pytest.raises(InputError) as err:
        open_and_page()
    assert str(err.value) == LIMIT
    assert pdf_client.live_workers() == 0


@pytest.mark.parametrize(
    "behaviour",
    ["unknown_refusal", "page_21_refusal", "huge_header", "not_json", "crash_mid_reply"],
)
def test_a_bad_open_reply_gives_the_limit_line(monkeypatch, behaviour):
    use_fake(monkeypatch, behaviour)
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "pdf"):
        pass
    assert str(err.value) == LIMIT
    assert pdf_client.live_workers() == 0


def test_a_known_refusal_reaches_the_caller_unchanged(monkeypatch):
    use_fake(monkeypatch, "known_refusal")
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "pdf"):
        pass
    assert str(err.value) == "page 1 of the PDF could not be read; it may be damaged"
    assert pdf_client.live_workers() == 0


def test_a_reader_that_never_answers_open_runs_out_of_time(monkeypatch):
    use_fake(monkeypatch, "sleep_forever")
    short_file_budget(monkeypatch, 1)
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "pdf"):
        pass
    assert str(err.value) == LIMIT
    assert pdf_client.live_workers() == 0


@pytest.mark.parametrize("behaviour", ["sleep_on_page", "stall_in_pixels"])
def test_time_limit_kills_the_reader_within_a_few_seconds(monkeypatch, behaviour):
    # stall_in_pixels: the timer stays set until the pixels of a reply are read in full
    use_fake(monkeypatch, behaviour)
    short_file_budget(monkeypatch, 1)
    pid = None
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "pdf") as file:
        pid = file._worker.pid
        started = time.monotonic()
        file.page(0)
    assert str(err.value) == LIMIT
    assert time.monotonic() - started <= 3
    assert isinstance(pid, int) and not pid_alive(pid)


def test_the_time_is_checked_again_when_the_reply_completes(monkeypatch):
    use_fake(monkeypatch, "slow_open")
    short_file_budget(monkeypatch, 1)
    # two readings for the start, then the request: the clock jumps while its reply is read
    clock = iter([0.0, 0.1, 0.0, 5.0])
    monkeypatch.setattr(pdf_client, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "pdf"):
        pass
    assert str(err.value) == LIMIT


def test_only_two_readers_live_at_once(monkeypatch):
    use_fake(monkeypatch, "slow_open")
    results, seen, done = [], [], threading.Event()

    def sample():
        while not done.is_set():
            seen.append(pdf_client.live_workers())
            time.sleep(0.01)

    def work():
        try:
            with pdf_client.opened(b"x", "pdf") as file:
                results.append(file.count)
        except Exception as err:  # noqa: BLE001 - recorded and asserted below
            results.append(err)

    sampler = threading.Thread(target=sample)
    sampler.start()
    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    done.set()
    sampler.join(5)
    assert results == [1, 1, 1, 1]
    assert max(seen) == 2
    assert pdf_client.live_workers() == 0


def test_the_real_reader_opens_a_pdf_and_renders_a_page():
    with pdf_client.opened(PDF, "pdf") as file:
        page = file.page(0)
        assert file.count == 1
        assert page.image.size == ingest.render_size(612, 792)
        assert file.render(0).size == page.image.size
        with pytest.raises(ValueError, match="not in the file"):
            file.page(1)
    assert pdf_client.live_workers() == 0


def test_a_refusal_from_the_real_reader_is_shown():
    with pytest.raises(InputError) as err, pdf_client.opened(b"%PDF-1.4 junk", "pdf"):
        pass
    assert str(err.value) in ingest.REFUSALS


def test_start_environment_has_no_folio_setting(monkeypatch):
    monkeypatch.setenv("FOLIO_MODEL_API_KEY", "x")
    monkeypatch.setenv("FOLIO_MODEL_URL", "http://example.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-key")
    monkeypatch.setenv("PYTHONPATH", "untrusted-import-path")
    assert not [k for k in pdf_client._environment() if k.upper().startswith("FOLIO_")]
    real, calls = subprocess.Popen, []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pdf_client.subprocess, "Popen", spy)
    with pdf_client.opened(PDF, "pdf"):
        pass
    ((args, kwargs),) = calls
    expected = [sys.executable, "-P", "-m", "folio_ocr.pdf_worker", str(ingest.PDF_MEMORY_MB)]
    assert args[0] == expected
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert not [k for k in kwargs["env"] if k.upper().startswith("FOLIO_")]
    assert not {"OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "PYTHONPATH"} & kwargs["env"].keys()


def test_the_reader_does_not_see_the_model_key(monkeypatch):
    monkeypatch.setenv("FOLIO_MODEL_API_KEY", "x")
    use_fake(monkeypatch, "slow_open")
    seen, real = [], pdf_frames.read_header

    def spy(stream, limit):
        reply = real(stream, limit)
        seen.append(reply)
        return reply

    monkeypatch.setattr(pdf_client.pdf_frames, "read_header", spy)
    with pdf_client._running("pdf"):
        pass
    assert seen[0]["has_key"] is False


def test_the_time_budget_adds_up_across_requests_to_one_worker(monkeypatch):
    # A clock that moves 300 s at every timed step, against a 1000 s budget. The start has a
    # budget of its own; then the open leaves 700, the first page 400, the second page 100, so
    # the third page is out of time although each step alone is far under the budget. The
    # real timer is set for the budget left, so it never fires here.
    monkeypatch.setattr(ingest, "PDF_SECONDS", 1000.0)
    now = [0.0]

    def tick():
        now[0] += 300.0
        return now[0]

    monkeypatch.setattr(pdf_client, "time", SimpleNamespace(monotonic=tick))
    with pdf_client.opened(PDF, "pdf") as file:
        file.page(0)
        file.page(0)
        with pytest.raises(InputError) as err:
            file.page(0)
    assert str(err.value) == LIMIT
    assert pdf_client.live_workers() == 0


def test_a_slow_start_does_not_use_the_files_time(monkeypatch):
    # The start takes 900 of a 1000 s budget; the open and the page still get a full budget.
    monkeypatch.setattr(ingest, "PDF_SECONDS", 1000.0)
    steps = iter([0.0, 900.0, 900.0, 1100.0, 1100.0, 1300.0])
    monkeypatch.setattr(pdf_client, "time", SimpleNamespace(monotonic=lambda: next(steps)))
    with pdf_client.opened(PDF, "pdf") as file:
        file.page(0)
    assert pdf_client.live_workers() == 0


@pytest.fixture
def memory_limit_after_start(monkeypatch):
    """The job reports its memory limit as hit once the reader has started."""
    state = {"on": False}
    real = pdf_client._Worker.start

    def start(self):
        real(self)
        state["on"] = True

    monkeypatch.setattr(pdf_client._Worker, "start", start)
    monkeypatch.setattr(job_object.JobObject, "memory_limit_hit", lambda self: state["on"])


@pytest.mark.skipif(sys.platform != "win32", reason="the job object is Windows only")
def test_a_job_memory_message_turns_a_good_reply_into_the_limit_line(memory_limit_after_start):
    with pytest.raises(InputError) as err:
        pipeline.page_image(PDF, 1)
    assert str(err.value) == LIMIT
    assert pdf_client.live_workers() == 0


@pytest.mark.skipif(sys.platform != "win32", reason="the job object is Windows only")
def test_a_job_memory_message_turns_a_refusal_into_the_limit_line(
    monkeypatch, memory_limit_after_start
):
    use_fake(monkeypatch, "known_refusal")
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "pdf"):
        pass
    assert str(err.value) == LIMIT
    assert pdf_client.live_workers() == 0


PDF_503 = "PDFs cannot be read here: folio could not start its PDF reader with limits"


def test_a_reader_that_never_gets_ready_runs_out_of_the_start_time(monkeypatch, tmp_path):
    pid_file = tmp_path / "pid"
    monkeypatch.setattr(
        pdf_client, "_command", lambda: [*fake_command("sleep_before_ready"), str(pid_file)]
    )
    monkeypatch.setattr(ingest, "PDF_SECONDS", 1)
    started = time.monotonic()
    with pytest.raises(PdfReaderError) as err, pdf_client.opened(b"x", "pdf"):
        pass
    assert str(err.value) == PDF_503
    assert 0.8 <= time.monotonic() - started <= 3
    assert pdf_client.live_workers() == 0
    # On a busy machine the reader may be killed before it could even write its pid.
    assert not pid_file.exists() or not pid_alive(int(pid_file.read_text()))


def test_a_misbehaving_image_reader_gives_the_image_limit_line(monkeypatch):
    use_fake(monkeypatch, "not_json")
    with pytest.raises(InputError) as err, pdf_client.opened(b"x", "png"):
        pass
    assert str(err.value) == (
        "the image could not be read within folio's memory and time limits; it was not read"
    )
    assert pdf_client.live_workers() == 0
