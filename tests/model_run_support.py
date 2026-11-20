"""Shared helpers for the model-run command tests: a fake session and step-file builders."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import model_run

from folio_ocr.model_reader import ModelConfig

KEYS = {"model-en": "key-en", "model-ar": "key-ar"}
FACTS = {
    "kind": "ollama",
    "version": "0.9.1",
    "version_major_minor": "0.9",
    "digest": "sha256:abc",
    "model": "m:8b",
    "host": "127.0.0.1",
}


def make_session(url: str = "http://127.0.0.1:9/v1", factory=None, keys=None):
    settings = SimpleNamespace(weights_dir=None, line_timeout_s=60.0)
    config = ModelConfig(url, "m:8b", None, 30.0, 60.0, False)
    return model_run.Session(
        settings=settings,
        config=config,
        tesseract=Path("tess.exe"),
        tesseract_version="5.3.0",
        model=SimpleNamespace(),
        capture=SimpleNamespace(),
        keys=dict(KEYS if keys is None else keys),
        facts=dict(FACTS),
        reader_factory=factory or (lambda settings, tess: (lambda *a: None)),
    )


def builder(session):
    """A session builder that hands out `session` and records how it was called."""
    calls: list[tuple[Path, bool]] = []

    def build(work, *, warm=True):
        calls.append((Path(work), warm))
        return session

    build.calls = calls  # type: ignore[attr-defined]
    return build


def run(capsys, argv, session=None):
    """(exit code, stdout, stderr) of one command-line run with an injected session."""
    build = builder(session if session is not None else make_session())
    rc = model_run.main([str(a) for a in argv], build)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def step_file(path: Path, step: str, **body) -> Path:
    record = {
        "step": step,
        "status": "finished",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        **body,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def sha_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))
