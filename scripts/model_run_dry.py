"""A dry run of the whole model-run pipeline against a local fake model server.

Usage: python scripts/model_run_dry.py --work <EMPTY-OR-NEW-DIR> [--jobs 2]

The fake server is the one the tests use (tests/fake_model_server.py, reused here). It answers
with canned replies shaped like Ollama's, so every step from contract to report runs end to end
without a GPU or a download. Minimums are lowered in this process only; the results file is
labelled a dry run and is not a measurement of any model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import accuracy_stats
import heldout
import model_run
import model_run_lines
import model_run_vram
from fake_model_server import Answer, FakeModelServer, Seen
from heldout import HeldoutError
from model_run_common import say
from model_run_server import ModelRunError
from PIL import Image
from script_errors import describe_os_error

from folio_ocr import calibration
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.errors import FolioError
from folio_ocr.model_self_test_lines import LINES

MODEL_NAME = "qwen3-vl:8b-instruct-q4_K_M"
MODEL_DIGEST = "70e58e37beb15464e19aa903dedb2cba828b89c0fb186f75ff2738c7347b96eb"
VERSION = "0.40.16"
LABEL = "dry run against a local fake model server; minimums lowered; not a measurement"
ARABIC_TEXT = "رقم الفاتورة"
ENGLISH_TEXT = "Invoice number: INV-2026-3706"
# (token, logprob) of the real Ollama replies for an English and an Arabic line.
ENGLISH = [("Invoice", -1.1920930376163597e-7), (" number", 0), (":", 0),
           (" INV", -0.000002861027041944908), ("-", 0), ("2", 0), ("0", 0), ("2", 0), ("6", 0),
           ("-", 0), ("3", 0), ("7", 0), ("0", 0), ("6", 0)]
ARABIC = [("ر", -0.2777141332626343), ("قم", -0.00006652099546045065),
          (" الف", -0.0032729962840676308), ("ات", -0.000004768382950715022),
          ("ورة", -0.00009763717389432713), (" ", -0.00047062980593182147),
          ("١", -0.0000010728841743912199), ("٢", -0.00012255465844646096),
          ("٣", -0.00015987243386916816), ("٤", -0.00007093204476404935)]
_COUNTER = itertools.count()
_LOCK = threading.Lock()


def _grey(png: bytes) -> tuple[tuple[int, int], bytes]:
    image = Image.open(io.BytesIO(png)).convert("L")
    return image.size, image.tobytes()


SELF_TEST = [(_grey(line.png), line) for line in LINES]


def _json_reply(payload: dict) -> Answer:
    return Answer(body=json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _entries(pairs: list[tuple[str, float]], factor: float) -> list[dict]:
    return [{"token": t, "logprob": p * factor, "bytes": list(t.encode("utf-8"))}
            for t, p in pairs]


def _completion(text: str, pairs: list[tuple[str, float]], finish: str, factor: float) -> Answer:
    arabic = pairs is ARABIC
    prompt, completion = (1151, 15) if arabic else (1179, 15)
    return _json_reply({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1795114573,
        "model": MODEL_NAME, "system_fingerprint": "fp_ollama",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish,
                     "logprobs": {"content": _entries(pairs, factor)}}],
        "usage": {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": prompt - 1},
                  "completion_tokens": completion, "total_tokens": prompt + completion}})


def _request_image(body: dict) -> bytes:
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    return base64.b64decode(url.partition("base64,")[2])


def _route(body: dict) -> tuple[str, list[tuple[str, float]], str]:
    """The (text, logprob pairs, finish reason) for one decoded chat request."""
    tokens = body["max_tokens"]
    if tokens == 1:
        return "Invoice", ENGLISH[:1], "length"
    size, pixels = _grey(_request_image(body))
    for (line_size, line_pixels), line in SELF_TEST:
        if size == line_size and pixels == line_pixels:
            return line.text, ARABIC if line.languages != "en" else ENGLISH, "stop"
    if set(pixels) == {255}:
        return "", [], "stop"
    if tokens == 16 + math.ceil(3 * size[0] / size[1]):
        return ENGLISH_TEXT, ENGLISH, "stop"
    return ARABIC_TEXT, ARABIC[:5], "stop"


def answer(seen: Seen) -> Answer:
    """The fake server's reply to one request."""
    if seen.method == "GET" and seen.target.endswith("/api/version"):
        return _json_reply({"version": VERSION})
    if seen.method == "GET" and seen.target.endswith("/api/tags"):
        return _json_reply({"models": [{"name": MODEL_NAME, "model": MODEL_NAME,
                                        "digest": MODEL_DIGEST}]})
    with _LOCK:
        n = next(_COUNTER)
    text, pairs, finish = _route(json.loads(seen.body))
    return _completion(text, pairs, finish, 1 + 0.02 * (n % 3))


class DryRunError(Exception):
    """A dry-run failure that is shown to the user as one line."""


STUB_SMI = (
    "import time\nwhile True:\n"
    "    print('memory.used [MiB]', flush=True)\n"
    "    print('1234 MiB', flush=True)\n"
    "    time.sleep(0.5)\n"
)


def _prepare_data(work: Path) -> tuple[Path, Path, str]:
    """The dev slice, the held-out manifest and its sha256."""
    dev = work / "dev"
    work.mkdir(parents=True, exist_ok=True)
    dev.mkdir()
    done = subprocess.run(
        [sys.executable, "make_dev_variants.py", "--first", "90101", "--count", "2",
         "--out", str(dev)], cwd=ROOT / "scripts", capture_output=True, check=False)
    if done.returncode != 0:
        raise DryRunError(f"the dev slice could not be made (exit {done.returncode})")
    manifest = work / "heldout" / "dry-run.json"
    manifest.parent.mkdir(parents=True)
    (work / "results").mkdir()
    sha = heldout.write_manifest(heldout.build_manifest(9000001, 9000001, "dry-run"), manifest)
    return dev, manifest, sha


def _lower_minimums() -> dict[tuple[object, str], object]:
    """Set every minimum to its dry-run value; the originals, to restore afterwards."""
    wanted = [(accuracy_stats, "DEV_MIN_BAD", 1), (accuracy_stats, "DEV_MIN_DOCS", 1),
              (accuracy_stats, "TEST_MIN_BAD", 1), (accuracy_stats, "TEST_MIN_DOCS", 1),
              (heldout, "MIN_BAD_LINES", 1), (heldout, "MIN_DOCUMENTS", 1),
              (model_run_lines, "MIN_ORDER_LINES", 0)]
    saved = {(mod, name): getattr(mod, name) for mod, name, _ in wanted}
    for mod, name, value in wanted:
        setattr(mod, name, value)
    return saved


def _restore(saved: dict[tuple[object, str], object]) -> None:
    for (mod, name), value in saved.items():
        setattr(mod, name, value)


def _fill_taus(taus: dict[str, float]) -> None:
    """Make the in-process tau table hold `taus`, as the laptop's table edit would."""
    table = calibration.checked_table({**calibration.TAUS, **taus})
    original_lookup, original_sha = calibration.lookup, calibration.table_sha256
    calibration.TAUS = table
    calibration.lookup = lambda key, t=None: original_lookup(key, table if t is None else t)
    calibration.table_sha256 = lambda t=None: original_sha(table if t is None else t)


def _stub_nvidia_smi() -> Callable | None:
    """Replace the VRAM reader by a stub when the PC has no nvidia-smi; the original, if so."""
    if shutil.which("nvidia-smi") is not None:
        return None
    original = model_run_vram.nvidia_smi_command
    model_run_vram.nvidia_smi_command = lambda path=None: (sys.executable, "-c", STUB_SMI)
    return original


def _label() -> str:
    return LABEL + ("; VRAM from a stub" if shutil.which("nvidia-smi") is None else "")


def _steps(work: Path, dev: Path, manifest: Path, sha: str, jobs: int) -> list[tuple[str, list]]:
    steps, cache, results = work / "steps", str(work / "cache"), work / "results"
    j = ["--jobs", str(jobs)]

    def out(name: str) -> str:
        return str(steps / f"{name}.json")

    base = ["--work", cache]
    return [
        ("contract", ["contract", *base, "--out", out("contract"), "--expect-version", VERSION,
                      "--expect-digest", MODEL_DIGEST]),
        ("order", ["order", *base, *j, "--dev", str(dev), "--out", out("order")]),
        ("drift-sample", ["drift-sample", *base, "--dev", str(dev), "--order", out("order"),
                          "--out", out("drift-sample"), "--lines", "5"]),
        ("drift", ["drift", *base, "--sample", out("drift-sample"), "--out", out("drift"),
                   "--repeats", "2", "--timeout-s", "30"]),
        ("calibrate", ["calibrate", *base, *j, "--dev", str(dev), "--order", out("order"),
                       "--drift", out("drift"), "--heldout", str(manifest),
                       "--heldout-sha256", sha, "--out", str(results / "calibrate-model-dry.json"),
                       "--record", out("calibrate"), "--results-dir", str(results)]),
        ("measure", ["measure", *base, *j, "--dev", str(dev), "--record", out("calibrate"),
                     "--out", out("measure")]),
        ("grade", ["grade", *base, "--jobs", "1", "--heldout", str(manifest),
                   "--heldout-sha256", sha, "--record", out("calibrate"),
                   "--results-dir", str(results), "--out", out("grade")]),
        ("fixtures", ["fixtures", *base, *j, "--out", out("fixtures")]),
        ("report", ["report", *base, "--contract", out("contract"), "--order", out("order"),
                    "--drift", out("drift"), "--record", out("calibrate"),
                    "--measure", out("measure"), "--grade-record", out("grade"),
                    "--fixtures", out("fixtures"),
                    "--out", str(results / "model-run-dry.json"), "--label", _label()]),
    ]


def _run_steps(plan: list[tuple[str, list]], work: Path) -> None:
    for name, argv in plan:
        if name == "grade":
            record = json.loads((work / "steps" / "calibrate.json").read_bytes())
            _fill_taus(record["taus"])
        started = time.perf_counter()
        code = model_run.main(argv)
        say(f"-- {name}: {time.perf_counter() - started:.1f} s")
        sys.stdout.flush()
        if code != 0:
            raise DryRunError(f"the {name} step failed (exit {code})")


def _run(work: Path, jobs: int) -> int:
    if find_tesseract(Settings.from_env(os.environ)) is None:
        raise DryRunError(model_run.NO_TESSERACT)
    dev, manifest, sha = _prepare_data(work)
    _run_steps(_steps(work, dev, manifest, sha, jobs), work)
    report = work / "results" / "model-run-dry.json"
    raw = report.read_bytes()
    say(f"report: {report}")
    say(f"sha256: {hashlib.sha256(raw).hexdigest()}")
    say("keys: " + ", ".join(sorted(json.loads(raw))))
    return 0


_CLEARED = ("FOLIO_MODEL_API_KEY", "FOLIO_TWO_READ", "FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER",
            "FOLIO_TAU_FILE")
_ENVIRONMENT = ("FOLIO_MODEL_URL", "FOLIO_MODEL_NAME", *_CLEARED)


def _save_environment() -> dict[str, str | None]:
    """The values (None when unset) of every variable the dry run sets or clears."""
    return {name: os.environ.get(name) for name in _ENVIRONMENT}


def _restore_environment(saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _set_environment(url: str) -> None:
    os.environ["FOLIO_MODEL_URL"] = url
    os.environ["FOLIO_MODEL_NAME"] = MODEL_NAME
    for name in _CLEARED:
        os.environ.pop(name, None)


def _refusal(args: argparse.Namespace) -> str | None:
    work = Path(args.work)
    if work.exists() and any(work.iterdir()):
        return "the work folder is not empty; give a new or empty folder"
    return "--jobs must be at least 1" if args.jobs < 1 else None


def _execute(args: argparse.Namespace) -> int:
    refusal = _refusal(args)
    if refusal is not None:
        raise DryRunError(refusal)
    saved_env = _save_environment()
    saved_cal = (calibration.TAUS, calibration.lookup, calibration.table_sha256)
    saved_smi, saved_min = _stub_nvidia_smi(), _lower_minimums()
    try:
        with FakeModelServer(answer) as fake:
            _set_environment(fake.url)
            return _run(Path(args.work), args.jobs)
    finally:
        calibration.TAUS, calibration.lookup, calibration.table_sha256 = saved_cal
        _restore(saved_min)
        if saved_smi is not None:
            model_run_vram.nvidia_smi_command = saved_smi
        _restore_environment(saved_env)


def _fail(message: str) -> int:
    print(f"error: {message}".encode("ascii", "replace").decode(), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        return _execute(args)
    except (DryRunError, ModelRunError, HeldoutError, FolioError, ValueError) as exc:
        return _fail(str(exc))
    except OSError as exc:
        return _fail(describe_os_error(exc, args.work))
    except Exception as exc:  # noqa: BLE001 - the user sees one line, never a traceback
        return _fail(f"unexpected failure ({type(exc).__name__})")


if __name__ == "__main__":
    sys.exit(main())
