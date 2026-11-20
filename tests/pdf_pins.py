"""Byte-level pins of what folio-ocr produces for every fixture PDF: page renders, the PNGs the
web API serves, and the no-server Document JSON. `main()` rewrites tests/data/pdf_pins.json."""

import hashlib
import json
import os
import tempfile
import zlib
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import PIL
import pypdfium2
from fastapi.testclient import TestClient

from folio_ocr import calibration, pipeline
from folio_ocr.config import Settings, find_tesseract
from folio_ocr.export import to_json
from folio_ocr.tesseract import decided_version
from folio_ocr.web.app import create_app

FIXTURES = Path(__file__).parent / "fixtures"
PIN_FILE = Path(__file__).parent / "data" / "pdf_pins.json"
LANGUAGES = ("en", "en+ar")
SETTINGS = Settings.from_env(os.environ)
WEIGHTS = SETTINGS.weights_dir
TESSERACT = find_tesseract(SETTINGS)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_pdfs() -> list[Path]:
    return sorted(FIXTURES.glob("*.pdf"), key=lambda p: p.name)


def page_count(data: bytes) -> int:
    pdf = pypdfium2.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def build_info() -> dict:
    return {
        "pypdfium2": str(pypdfium2.version.PYPDFIUM_INFO),
        "pdfium": str(pypdfium2.PDFIUM_INFO.version),
        "pillow": PIL.__version__,
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
    }


def renders(data: bytes, pages: int) -> dict:
    out = {}
    for n in range(1, pages + 1):
        img = pipeline.page_image(data, n)
        out[str(n)] = {"mode": img.mode, "size": list(img.size), "sha256": sha(img.tobytes())}
    return out


def served_pngs(data: bytes, name: str, pages: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        static = Path(tmp) / "static"
        static.mkdir()
        (static / "index.html").write_text("<p>home</p>")
        app = create_app(
            Settings(weights_dir=Path(tmp)), port=8000, tesseract=None, model=None,
            static_dir=static, log=[].append,
        )  # fmt: skip
        with TestClient(app, base_url="http://127.0.0.1:8000") as c:
            up = c.post(
                "/api/documents",
                files={"file": (name, data, "application/pdf")},
                data={"languages": "en"},
            )
            up.raise_for_status()
            doc = up.json()["document_id"]
            out = {}
            for n in range(1, pages + 1):
                r = c.get(f"/api/documents/{doc}/pages/{n}/image")
                r.raise_for_status()
                out[str(n)] = sha(r.content)
            return out


def document_pin(data: bytes, name: str, languages: str) -> tuple[dict, str | None]:
    doc = pipeline.read_bytes(
        data, name, languages=languages, weights_dir=WEIGHTS, tesseract_path=TESSERACT
    )
    obj = json.loads(to_json(doc))
    weights = (obj.get("engine") or {}).get("weights_sha256")
    for key in ("id", "created_at"):
        obj.pop(key, None)
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    pages = [{"status": p["status"], "origin": p["origin"]} for p in obj["pages"]]
    return {"sha256": sha(blob.encode("utf-8")), "page_count": len(pages), "pages": pages}, weights


def ocr_env_ready() -> bool:
    from folio_ocr import ocr_setup
    from folio_ocr.tesseract import tessdata_problem

    bad = ocr_setup.onnxtr_status(WEIGHTS) in ("missing", "bad_data")
    return not bad and TESSERACT is not None and tessdata_problem(WEIGHTS) is None


@contextmanager
def empty_taus():
    """Run with the empty confidence threshold table the Document pins were made under."""
    with mock.patch.object(calibration, "lookup", lambda key, table=None: {}.get(key)):
        yield


def tesseract_decided() -> str | None:
    return decided_version(TESSERACT) if TESSERACT is not None else None


def compute() -> dict:
    with empty_taus():
        return _compute()


def _compute() -> dict:
    files, weights = {}, {}
    for path in fixture_pdfs():
        data = path.read_bytes()
        pages = page_count(data)
        entry = {
            "renders": renders(data, pages),
            "png": served_pngs(data, path.name, pages),
            "documents": {},
        }
        for lang in LANGUAGES:
            entry["documents"][lang], found = document_pin(data, path.name, lang)
            if found:
                weights.setdefault(lang, found)
        files[path.name] = entry
    ocr_env = {
        "tesseract": tesseract_decided(),
        "weights": weights,
        "taus": calibration.table_sha256({}),
    }
    return {"build": build_info(), "ocr_env": ocr_env, "files": files}


def main() -> None:
    PIN_FILE.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
