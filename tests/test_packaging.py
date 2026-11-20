from importlib.resources import files
from pathlib import Path

SOURCE_STATIC = Path(__file__).resolve().parents[1] / "src" / "folio_ocr" / "web" / "static"


def test_every_web_static_file_ships_with_the_package():
    # CI installs the package as a wheel, so this fails there when the
    # package-data glob in pyproject.toml stops covering a file under web/static.
    installed = files("folio_ocr") / "web" / "static"
    sources = [p.relative_to(SOURCE_STATIC) for p in SOURCE_STATIC.rglob("*") if p.is_file()]
    assert sources
    missing = [p.as_posix() for p in sources if not installed.joinpath(*p.parts).is_file()]
    assert missing == []
