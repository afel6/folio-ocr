"""Files that are small but costly to read are refused or read inside folio's limits: the
reader process stays under its memory cap and its time, folio's own process barely grows,
and no reader process is left. Each bomb runs in a fresh Python process."""

import json
import subprocess
import sys
from pathlib import Path

import pdf_bombs
import pytest
from pid_tools import pid_alive
from test_run_accuracy import FakeReader, grade_args, patched, run, write_heldout  # noqa: F401

from folio_ocr import ingest, pdf_client
from folio_ocr.errors import InputError

MAX_PARENT_GROWTH_MB = 100
SLACK_SECONDS = 2
TESTS = str(Path(__file__).parent)

CHILD = r"""
import ctypes
import json
import sys
import time
from ctypes import wintypes

sys.path.insert(0, sys.argv[1])
from pid_tools import pid_alive

from folio_ocr import pdf_client, pipeline
from folio_ocr.errors import InputError
from folio_ocr.web.upload import Upload, new_document


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t)
        for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage",
        )
    ]


def committed_mb(peak):
    if sys.platform == "win32":
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get = ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not get(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return (counters.PeakPagefileUsage if peak else counters.PagefileUsage) / 2**20
    import resource

    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kib / 1024


seen = {"peaks": [], "pids": []}
original_close = pdf_client._Worker.close


def close(self):
    if self._job is not None:
        seen["peaks"].append(self._job.peak_bytes() / 2**20)
    seen["pids"].append(self.pid)
    original_close(self)


pdf_client._Worker.close = close


def call(step):
    started = time.perf_counter()
    try:
        step()
    except InputError as err:
        outcome = "refused: " + str(err)
    else:
        outcome = "ok"
    return outcome, time.perf_counter() - started


data = open(sys.argv[2], "rb").read()
before = committed_mb(False)
outcomes, walls = [], []
for step in (
    lambda: new_document(Upload("bomb.pdf", data, "en")),
    lambda: pipeline.page_image(data, 1),
):
    outcome, wall = call(step)
    outcomes.append(outcome)
    walls.append(wall)
    if outcome != "ok":
        break
print(json.dumps({
    "outcomes": outcomes,
    "walls": walls,
    "growth_mb": committed_mb(True) - before,
    "worker_peaks_mb": seen["peaks"],
    "pids": seen["pids"],
    "live": pdf_client.live_workers(),
    "alive": [p for p in seen["pids"] if isinstance(p, int) and pid_alive(p, 3.0)],
}))
"""

BOMBS = {
    "page image": pdf_bombs.page_image_bomb,
    "packed smask": pdf_bombs.packed_smask_bomb,
    "content stream": pdf_bombs.content_stream_bomb,
    "form fan-out": pdf_bombs.form_fanout_bomb,
}


def measure(name, tmp_path):
    data = BOMBS[name]()
    assert len(data) <= 25 * 2**20
    path = tmp_path / "bomb.pdf"
    path.write_bytes(data)
    done = subprocess.run(
        [sys.executable, "-c", CHILD, TESTS, str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3 * ingest.PDF_SECONDS + 120,
    )
    assert done.returncode == 0, done.stderr[-600:]
    return json.loads(done.stdout.splitlines()[-1])


@pytest.mark.parametrize("name", list(BOMBS))
def test_a_bomb_is_refused_or_read_within_the_limits(name, tmp_path, capsys):
    result = measure(name, tmp_path)
    with capsys.disabled():
        print(f"\n{name}: {result}")
    for outcome in result["outcomes"]:
        assert outcome == "ok" or outcome.startswith("refused: ")
    assert max(result["walls"]) <= ingest.PDF_SECONDS + SLACK_SECONDS
    assert result["growth_mb"] < MAX_PARENT_GROWTH_MB
    if sys.platform == "win32":
        assert result["worker_peaks_mb"]
        assert max(result["worker_peaks_mb"]) <= ingest.PDF_MEMORY_MB
    assert result["live"] == 0
    assert result["alive"] == []
    assert not any(pid_alive(pid, 0.2) for pid in result["pids"] if isinstance(pid, int))


def test_limit_refusal_stops_the_accuracy_run(tmp_path, capsys):
    line = pdf_client.limit_line("pdf")

    def reader(data, name, mode, with_tesseract):
        raise InputError(line)

    held = write_heldout(tmp_path, 1000, 1001)
    assert run(grade_args(tmp_path, held), reader) != 0
    err = capsys.readouterr().err
    assert err == f"error: {line}\n"
