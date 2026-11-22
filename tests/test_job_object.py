"""The Windows job object that caps a reader process's committed memory, and the Linux
data-segment limit the reader sets on itself."""

import subprocess
import sys

import pytest
from pid_tools import pid_alive

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")

CHILD = """
import os, sys
print(os.getpid(), flush=True)
sys.stdin.readline()
try:
    bytearray(200 * 2**20)
    print("allocated", flush=True)
except MemoryError:
    print("memerr", flush=True)
sys.stdin.readline()
"""


@pytest.mark.skipif(not WINDOWS, reason="job objects are Windows only")
def test_a_job_caps_memory_reports_it_and_kills_on_close():
    from folio_ocr.job_object import CREATE_SUSPENDED, JobObject

    job = JobObject(64 * 2**20)
    proc = subprocess.Popen(
        [sys.executable, "-P", "-c", CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        creationflags=CREATE_SUSPENDED,
        text=True,
    )
    try:
        job.adopt(proc.pid)
        pid = int(proc.stdout.readline())
        proc.stdin.write("\n")
        proc.stdin.flush()
        assert proc.stdout.readline().strip() == "memerr"
        assert job.memory_limit_hit() is True
        assert job.memory_limit_hit() is False
        assert job.peak_bytes() > 0
        job.close()
        assert not pid_alive(pid)
    finally:
        job.close()
        proc.kill()
        proc.wait()
        proc.stdin.close()
        proc.stdout.close()


@pytest.mark.skipif(not WINDOWS, reason="job objects are Windows only")
def test_a_job_without_a_memory_cap_only_ends_its_processes_on_close():
    import ctypes

    from folio_ocr import job_object
    from folio_ocr.job_object import JobObject

    job = JobObject(None)
    try:
        assert job._port is None
        limits = job_object._ExtendedLimits()
        job_object._K32.QueryInformationJobObject(
            job._job,
            job_object._EXTENDED_LIMITS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
            None,
        )
        assert limits.Basic.LimitFlags == job_object._LIMIT_KILL_ON_JOB_CLOSE
        assert limits.ProcessMemoryLimit == 0
    finally:
        job.close()


@pytest.mark.skipif(not WINDOWS, reason="job objects are Windows only")
def test_adopting_a_dead_process_raises_oserror():
    from folio_ocr.job_object import JobObject

    done = subprocess.Popen([sys.executable, "-P", "-c", "pass"])
    done.wait()
    job = JobObject(64 * 2**20)
    try:
        with pytest.raises(OSError):
            job.adopt(done.pid)
    finally:
        job.close()


LIMIT_CHILD = """
import resource
from folio_ocr import pdf_worker
assert pdf_worker._limit_self(100) is True
cap = 100 * 2**20
print(resource.getrlimit(resource.RLIMIT_DATA) == (cap, cap), flush=True)
try:
    bytearray(200 * 2**20)
    print("allocated")
except MemoryError:
    print("memerr")
"""


@pytest.mark.skipif(not LINUX, reason="RLIMIT_DATA is Linux only")
def test_the_reader_caps_its_own_data_segment():
    done = subprocess.run(
        [sys.executable, "-P", "-c", LIMIT_CHILD],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.stdout.split() == ["True", "memerr"], done.stderr[-300:]
