"""Helpers for tests that start reader processes: is a process alive, and the stand-in reader."""

import os
import sys
import time
from pathlib import Path

FAKE_WORKER = Path(__file__).with_name("fake_pdf_worker.py")
_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 258


def _alive_now(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def pid_alive(pid: int, patience: float = 5.0) -> bool:
    """True when process `pid` is still running after `patience` seconds of polling."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if not _alive_now(pid):
            return False
        time.sleep(0.05)
    return _alive_now(pid)


def fake_command(behaviour: str) -> list[str]:
    return [sys.executable, "-P", str(FAKE_WORKER), behaviour]
