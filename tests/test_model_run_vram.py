"""Tests for the GPU memory logger of the laptop model run."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import model_run_vram as vram
from model_run_server import ModelRunError

FAKE_SMI = (
    sys.executable,
    "-c",
    (
        "import time; print('memory.used [MiB]', flush=True); print('7062 MiB', flush=True); "
        "print('6100 MiB', flush=True); time.sleep(30)"
    ),
)


def test_vram_monitor_and_peak(tmp_path):
    log = tmp_path / "vram.csv"
    with vram.VramMonitor(log, FAKE_SMI) as monitor:
        assert monitor is not None
        deadline = time.monotonic() + 20
        while log.stat().st_size < 30 and time.monotonic() < deadline:
            time.sleep(0.05)
    assert vram.peak_mib(log) == 7062


def test_vram_monitor_missing_executable(tmp_path):
    missing = vram.VramMonitor(tmp_path / "v.csv", ("no-such-tool-xyz",))
    with pytest.raises(ModelRunError, match="nvidia-smi was not found"), missing:
        pytest.fail("body must not run")


def test_vram_monitor_keeps_body_exception(tmp_path):
    with pytest.raises(KeyError), vram.VramMonitor(tmp_path / "v.csv", FAKE_SMI):
        raise KeyError("body")


def test_peak_mib_no_samples_and_missing(tmp_path):
    log = tmp_path / "empty.csv"
    log.write_bytes(b"memory.used [MiB]\nnot a number\n")
    with pytest.raises(ModelRunError, match="no memory samples"):
        vram.peak_mib(log)
    with pytest.raises(FileNotFoundError):
        vram.peak_mib(tmp_path / "missing.csv")


def test_peak_mib_takes_max_over_gpus(tmp_path):
    log = tmp_path / "g.csv"
    log.write_bytes(b"memory.used [MiB]\n100 MiB\n900 MiB\n\n300 MiB\n")
    assert vram.peak_mib(log) == 900


def test_vram_monitor_with_a_folder_as_the_command(tmp_path):
    monitor = vram.VramMonitor(tmp_path / "v.csv", (str(tmp_path),))
    with pytest.raises(ModelRunError, match="could not be started") as caught, monitor:
        pass
    message = str(caught.value)
    assert "pass its path with --nvidia-smi" in message and str(tmp_path) not in message
    assert monitor._file.closed


def test_nvidia_smi_command():
    assert vram.nvidia_smi_command(None) == ("nvidia-smi", *vram.NVIDIA_SMI_ARGS)
    assert vram.nvidia_smi_command("C:/x/smi.exe")[0] == "C:/x/smi.exe"
