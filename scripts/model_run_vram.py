"""GPU memory logging for the laptop model run: an nvidia-smi logger and its peak figure."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from model_run_server import ModelRunError

NVIDIA_SMI_ARGS = ("--query-gpu=memory.used", "--format=csv", "-l", "1")
_NO_SMI = (
    "nvidia-smi was not found; install the NVIDIA driver tools or pass its path with --nvidia-smi"
)
_MIB = re.compile(r"\s*(\d+)\s*MiB\s*")


def nvidia_smi_command(path: str | None) -> tuple[str, ...]:
    return (path or "nvidia-smi", *NVIDIA_SMI_ARGS)


class VramMonitor:
    """Logs GPU memory use to a file for as long as the `with` block runs."""

    def __init__(self, log_path: Path, command: Sequence[str]) -> None:
        self._log_path = log_path
        self._command = tuple(command)
        self._file = None
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        self._file = open(self._log_path, "wb")
        try:
            self._process = subprocess.Popen(
                self._command, stdout=self._file, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            self._file.close()
            raise ModelRunError(_NO_SMI) from None
        except OSError as exc:
            self._file.close()
            raise ModelRunError(
                f"nvidia-smi could not be started ({exc.strerror or type(exc).__name__}); "
                "pass its path with --nvidia-smi"
            ) from None
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        finally:
            if self._file is not None:
                self._file.close()


def peak_mib(log_path: Path) -> int:
    """The highest memory figure in the log, over every sample and GPU."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    values = [int(m.group(1)) for line in text.splitlines() if (m := _MIB.fullmatch(line))]
    if not values:
        raise ModelRunError("nvidia-smi gave no memory samples")
    return max(values)
