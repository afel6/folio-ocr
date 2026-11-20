"""Static checks on the browser UI files: no inline code, no external URLs, safe DOM use."""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "src" / "folio_ocr" / "web" / "static"
SVG_NS = "http://www.w3.org/2000/svg"

FILES = sorted(p for p in STATIC.rglob("*") if p.is_file())
JS_FILES = sorted(STATIC.glob("*.js"))
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")

FORBIDDEN_JS = [
    r"innerHTML",
    r"outerHTML",
    r"insertAdjacentHTML",
    r"document\.write",
    r"DOMParser",
    r"createContextualFragment",
    r"srcdoc",
    r"eval\(",
    r"new\s+Function",
    r"\.style\b",
    r"setAttribute\(\s*['\"]style['\"]",
    r"console\.log",
]
IMPORT_SPEC = re.compile(
    r"""(?:\bimport\b[^'"();]*?\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"]+)\1"""
)


def test_index_has_no_inline_code():
    assert not re.search(r"<script(?![^>]*\ssrc=)", INDEX, re.IGNORECASE)
    assert not re.search(r"<style\b", INDEX, re.IGNORECASE)
    assert not re.search(r"\sstyle\s*=", INDEX, re.IGNORECASE)
    assert not re.search(r"\son[a-z]+\s*=", INDEX, re.IGNORECASE)


def test_index_references_existing_root_paths():
    refs = re.findall(r"""\b(?:src|href)\s*=\s*["']([^"']*)["']""", INDEX, re.IGNORECASE)
    assert refs
    for ref in refs:
        if ref.startswith("#"):
            continue
        assert ref.startswith("/") and not ref.startswith("//"), ref
        assert (STATIC / ref.lstrip("/")).is_file(), ref


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_js_avoids_unsafe_apis(path):
    text = path.read_text(encoding="utf-8")
    for pattern in FORBIDDEN_JS:
        assert not re.search(pattern, text), f"{path.name}: {pattern}"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_js_imports_are_local_modules(path):
    text = path.read_text(encoding="utf-8")
    for _, spec in IMPORT_SPEC.findall(text):
        assert re.fullmatch(r"\./[A-Za-z0-9_-]+\.js", spec), f"{path.name}: {spec}"
        assert (STATIC / spec[2:]).is_file(), f"{path.name}: {spec}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_external_urls(path):
    if path.suffix not in {".html", ".js", ".css"}:
        pytest.skip("not a text asset")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".js":
        text = text.replace(SVG_NS, "")
    assert not re.search(r"https?://", text), path.name
    if path.suffix == ".css":
        assert "@import" not in text
        assert "@font-face" not in text


def test_static_holds_only_shipped_files():
    names = [p.name for p in FILES]
    assert not [n for n in names if n.startswith("test_") or n.endswith("_test.js")]
    assert "package.json" not in names
    assert not [n for n in names if n.endswith(".map")]
    assert not (STATIC / "node_modules").exists()
