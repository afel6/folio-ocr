"""What every model-run step shares: the step files and their checks, console output, and the
model settings read from the environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import accuracy_reads as ar
from model_run_server import ModelRunError

from folio_ocr.config import Settings
from folio_ocr.model_reader import ModelConfig
from folio_ocr.model_reply import ORDER_RULE

NO_URL = "FOLIO_MODEL_URL is not set; set it to the model server's /v1 URL"
GRADE_KEY = "heldout_manifest_sha256"  # a step file must never hold this key as a top-level key


def say(text: str) -> None:
    """Print one console line as plain ASCII."""
    print(ascii_only(text))


def ascii_only(text: str) -> str:
    return text.encode("ascii", "replace").decode()


def model_config() -> tuple[Settings, ModelConfig]:
    settings = Settings.from_env(os.environ)
    config = ModelConfig.from_settings(settings)
    if config is None:
        raise ModelRunError(NO_URL)
    return settings, config


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_step(path: Path, step: str, started_at: str, body: dict) -> None:
    """Write a finished step file atomically as UTF-8 JSON."""
    record = {"step": step, "status": "finished", "started_at": started_at,
              "finished_at": ar.now(), **body}
    if GRADE_KEY in record:
        raise ValueError(f"a step file must not hold {GRADE_KEY}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ar.write_json(path, record)


def load_step(path: Path, step: str) -> dict:
    """The finished step file at `path`, else ModelRunError."""
    path = Path(path)
    unfinished = ModelRunError(f"the {step} results are missing or unfinished: {path.name}")
    try:
        record = json.loads(path.read_bytes())
    except (FileNotFoundError, ValueError) as exc:
        raise unfinished from exc
    if not isinstance(record, dict) or record.get("step") != step or record.get("status") != "finished":
        raise unfinished
    return record


LOCAL_SETUP_VARIABLES = ("FOLIO_MODEL_DIGEST", "FOLIO_MODEL_SERVER", "FOLIO_TAU_FILE")


def refuse_local_setup(what: str) -> None:
    """Refuse a package calibration or grade while a declared identity or a local tau file is
    set: those runs measure the server's own identity and the package table only."""
    named = [name for name in LOCAL_SETUP_VARIABLES if name in os.environ]
    if named:
        raise ModelRunError(f"{what} uses the package setup only and a local calibration is never "
                            f"graded; unset {', '.join(named)}")


def check_keys(record: dict, keys: dict[str, str]) -> None:
    if record.get("keys") != keys:
        step = record.get("step")
        raise ModelRunError(f"the model setup changed since the {step} step (keys differ); stop")


def read_json_file(path: Path, what: str) -> tuple[dict, str]:
    """The JSON object at `path` and the sha256 of the bytes it came from."""
    path = Path(path)
    damaged = ModelRunError(f"the {what} file is missing or damaged: {path.name}")
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (FileNotFoundError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise damaged from exc
    if not isinstance(record, dict):
        raise damaged
    return record, sha256_bytes(raw)


def read_json_checked(path: Path, sha256: str, what: str) -> dict:
    """A JSON file whose bytes must still hash to the recorded `sha256`."""
    record, actual = read_json_file(path, what)
    if actual != sha256:
        raise ModelRunError(f"the {what} file changed since it was recorded: {Path(path).name}")
    return record


def frozen_order(path: Path, dev_sha: str) -> dict:
    """The order step file, which must hold a frozen rule measured on this dev set."""
    order = load_step(path, "order")
    if order.get("verdict") != "frozen" or order.get("order_rule") != ORDER_RULE:
        raise ModelRunError("the order rule is not frozen; run the order step again")
    if order.get("dev_manifest_sha256") != dev_sha:
        raise ModelRunError("the dev set changed since the order step; stop")
    return order
