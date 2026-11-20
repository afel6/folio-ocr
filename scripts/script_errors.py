"""One-line, user-facing descriptions of file-system errors for the dev scripts."""

from __future__ import annotations

import errno
from pathlib import Path


def describe_os_error(exc: OSError, fallback: Path | str) -> str:
    """'<reason>: <path>' for an OSError raised while reading or writing `fallback`."""
    path = exc.filename if exc.filename is not None else fallback
    if isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT:
        reason = "not found"
    elif isinstance(exc, IsADirectoryError) or (
        isinstance(exc, PermissionError) and exc.filename is not None and Path(path).is_dir()
    ):
        reason = "is a folder"  # Windows reports opening a folder as permission denied
    elif isinstance(exc, PermissionError):
        reason = "permission denied"
    elif isinstance(exc, (NotADirectoryError, FileExistsError)):
        reason = "not a folder"
    else:
        reason = "cannot read or write"
    return f"{reason}: {path}"
