"""Refusing an oversized image must not decode it first: the peak memory barely grows."""

import subprocess
import sys
from pathlib import Path

import pytest
from sample_files import PAGE_IMAGE_LINE as PAGE_LINE
from sample_files import PDF_IMAGE_LINE as PDF_LINE

MAX_GROWTH_MB = 100
SIDE = 16000

CODE = r"""
import sys

sys.path.insert(0, sys.argv[1])
from sample_files import image_pdf

from folio_ocr.errors import InputError
from folio_ocr.ingest import open_pdf


def peak_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get = ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not get(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


side, kind, expected = int(sys.argv[2]), sys.argv[3], sys.argv[4]
if kind == "page":
    data = image_pdf(side, side, pixels=True)
else:
    data = image_pdf(0, 0, mask=("SMask", side, side), pixels=True)
before = peak_mb()
try:
    open_pdf(data)
except InputError as error:
    assert str(error) == expected, str(error)
else:
    raise SystemExit("the oversized image was not refused")
print(peak_mb() - before)
"""


@pytest.mark.parametrize(("kind", "line"), [("page", PAGE_LINE), ("smask", PDF_LINE)])
def test_refusing_an_oversized_pdf_image_keeps_peak_memory_flat(kind, line):
    tests_dir = str(Path(__file__).parent)
    done = subprocess.run(
        [sys.executable, "-c", CODE, tests_dir, str(SIDE), kind, line],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert float(done.stdout.split()[-1]) < MAX_GROWTH_MB
