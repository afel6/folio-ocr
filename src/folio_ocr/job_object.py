"""Windows Job Object that caps the committed memory of the file reader process.

The job is set up before the process runs: the process starts suspended, joins the job, and only
then runs, so the Python interpreter that a virtual environment's python.exe starts as its child
is in the job too. Closing the job kills every process in it, also when folio itself ends.
A process that hits the cap posts a message to the job's completion port."""

import ctypes
import sys
from ctypes import wintypes

_EXTENDED_LIMITS = 9  # JobObjectExtendedLimitInformation
_COMPLETION_PORT = 7  # JobObjectAssociateCompletionPortInformation
_LIMIT_PROCESS_MEMORY = 0x0100
_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_MSG_PROCESS_MEMORY_LIMIT = 9
_MSG_JOB_MEMORY_LIMIT = 10
_SNAP_THREADS = 0x4
_THREAD_SUSPEND_RESUME = 0x2
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_RESUME_FAILED = 0xFFFFFFFF
CREATE_SUSPENDED = 0x4


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("Basic", _BasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _PortLink(ctypes.Structure):
    _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, boolean, ptr = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, ctypes.c_void_p
    signatures = {
        "CreateJobObjectW": (handle, [ptr, wintypes.LPCWSTR]),
        "SetInformationJobObject": (boolean, [handle, ctypes.c_int, ptr, dword]),
        "QueryInformationJobObject": (boolean, [handle, ctypes.c_int, ptr, dword, ptr]),
        "CreateIoCompletionPort": (handle, [handle, handle, ctypes.c_size_t, dword]),
        "GetQueuedCompletionStatus": (
            boolean,
            [handle, ctypes.POINTER(dword), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ptr)]
            + [dword],
        ),
        "OpenProcess": (handle, [dword, boolean, dword]),
        "AssignProcessToJobObject": (boolean, [handle, handle]),
        "TerminateJobObject": (boolean, [handle, wintypes.UINT]),
        "CreateToolhelp32Snapshot": (handle, [dword, dword]),
        "Thread32First": (boolean, [handle, ctypes.POINTER(_ThreadEntry)]),
        "Thread32Next": (boolean, [handle, ctypes.POINTER(_ThreadEntry)]),
        "OpenThread": (handle, [dword, boolean, dword]),
        "ResumeThread": (dword, [handle]),
        "CloseHandle": (boolean, [handle]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(k, name)
        function.restype, function.argtypes = restype, argtypes
    return k


_K32 = _kernel32() if sys.platform == "win32" else None
_INVALID_HANDLE = ctypes.c_void_p(-1).value


def _fail(step: str) -> OSError:
    return OSError(ctypes.get_last_error(), f"the Windows job object could not {step}")


class JobObject:
    """A job whose processes each get at most `memory_bytes` of committed memory."""

    def __init__(self, memory_bytes: int) -> None:
        self._job = _K32.CreateJobObjectW(None, None)
        if not self._job:
            raise _fail("be created")
        self._port = None
        try:
            self._limit(memory_bytes)
            self._port = _K32.CreateIoCompletionPort(_INVALID_HANDLE, None, 0, 1)
            if not self._port:
                raise _fail("get a completion port")
            link = _PortLink(None, self._port)
            if not _K32.SetInformationJobObject(
                self._job, _COMPLETION_PORT, ctypes.byref(link), ctypes.sizeof(link)
            ):
                raise _fail("watch its memory limit")
        except BaseException:
            self.close()
            raise

    def _limit(self, memory_bytes: int) -> None:
        limits = _ExtendedLimits()
        limits.Basic.LimitFlags = _LIMIT_PROCESS_MEMORY | _LIMIT_KILL_ON_JOB_CLOSE
        limits.ProcessMemoryLimit = memory_bytes
        if not _K32.SetInformationJobObject(
            self._job, _EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise _fail("set the memory limit")

    def adopt(self, pid: int) -> None:
        """Put the suspended process `pid` in the job, then let it run."""
        process = _K32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not process:
            raise _fail("open the reader process")
        try:
            if not _K32.AssignProcessToJobObject(self._job, process):
                raise _fail("take the reader process")
        finally:
            _K32.CloseHandle(process)
        _resume(pid)

    def memory_limit_hit(self) -> bool:
        """True when a process of the job has hit its memory limit since the last call."""
        hit = False
        message, key, overlapped = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
        while _K32.GetQueuedCompletionStatus(
            self._port, ctypes.byref(message), ctypes.byref(key), ctypes.byref(overlapped), 0
        ):
            hit = hit or message.value in (_MSG_PROCESS_MEMORY_LIMIT, _MSG_JOB_MEMORY_LIMIT)
        return hit

    def peak_bytes(self) -> int:
        """The largest committed memory any process of the job has used."""
        limits = _ExtendedLimits()
        if not _K32.QueryInformationJobObject(
            self._job, _EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits), None
        ):
            raise _fail("report its memory use")
        return limits.PeakProcessMemoryUsed

    def terminate(self) -> None:
        """Kill every process in the job; the job stays open."""
        if self._job:
            _K32.TerminateJobObject(self._job, 1)

    def close(self) -> None:
        """Kill every process in the job and release it."""
        if self._job:
            self.terminate()
            _K32.CloseHandle(self._job)
            self._job = None
        if self._port:
            _K32.CloseHandle(self._port)
            self._port = None


def _resume(pid: int) -> None:
    """Resume the threads of the suspended process `pid` (its one main thread)."""
    snapshot = _K32.CreateToolhelp32Snapshot(_SNAP_THREADS, 0)
    if snapshot == _INVALID_HANDLE:
        raise _fail("list the reader's threads")
    resumed = 0
    try:
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = _K32.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                resumed += _resume_thread(entry.th32ThreadID)
            more = _K32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        _K32.CloseHandle(snapshot)
    if not resumed:
        raise _fail("start the reader process")


def _resume_thread(thread_id: int) -> int:
    thread = _K32.OpenThread(_THREAD_SUSPEND_RESUME, False, thread_id)
    if not thread:
        return 0
    try:
        return 0 if _K32.ResumeThread(thread) == _RESUME_FAILED else 1
    finally:
        _K32.CloseHandle(thread)
