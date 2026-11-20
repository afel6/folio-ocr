"""Pinned fonts for the development-variant documents (SIL Open Font Licence, licences beside them).

Source: github.com/google/fonts at commit cc5661d43cd4f9aad5567033a8e15ca28dc35c86. The files sit
in tests/dev_fonts/ and are checked against their size and SHA-256 before first use.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

DEV_FONT_DIR = Path(__file__).resolve().parents[1] / "tests" / "dev_fonts"


@dataclass(frozen=True)
class DevFont:
    file: str
    size: int
    sha256: str
    script: str


DEV_FONTS: Mapping[str, DevFont] = MappingProxyType(
    {
        "crimson": DevFont(
            "CrimsonText-Regular.ttf",
            107200,
            "48e6c5d5ad1d01599d374ecb817e15890d1feb3b8a3a88e527d44c90389e1f06",
            "latin",
        ),
        "ptsans": DevFont(
            "PT_Sans-Web-Regular.ttf",
            442960,
            "9cc831490532009bae2b3ce0d39c62adfc889060beb421593bfd9d2396d0f10a",
            "latin",
        ),
        "courier": DevFont(
            "CourierPrime-Regular.ttf",
            71188,
            "72f793376f8e2841656bf21d77a5de010f2929bd6956a22ee848ad0c7eb978af",
            "latin",
        ),
        "scheherazade": DevFont(
            "ScheherazadeNew-Regular.ttf",
            331504,
            "794bac8dc9e83d1d620bc471ea694f5f31d0965ce8006490a79dfc51a2d283b3",
            "arabic",
        ),
        "plexarabic": DevFont(
            "IBMPlexSansArabic-Regular.ttf",
            235924,
            "6f611412270a132bbac838da9259d4c68569b4175f3b3b8fa3fa36a30b56dab9",
            "arabic",
        ),
        "lateef": DevFont(
            "Lateef-Regular.ttf",
            241092,
            "fcbd9e0227a4d7238880defa5fd74c012d69351150709e559a8577446145c2fc",
            "arabic",
        ),
    }
)
LATIN_KEYS = tuple(k for k, f in DEV_FONTS.items() if f.script == "latin")
ARABIC_KEYS = tuple(k for k, f in DEV_FONTS.items() if f.script == "arabic")


@cache
def dev_font_path(key: str) -> Path:
    """Path of a dev font after checking its size and SHA-256; ValueError names the file if wrong."""
    font = DEV_FONTS[key]
    path = DEV_FONT_DIR / font.file
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"font file {font.file} cannot be read ({exc.strerror})") from exc
    if len(data) != font.size or hashlib.sha256(data).hexdigest() != font.sha256:
        raise ValueError(f"font file {font.file} does not match its pinned size and hash")
    return path


def font_sha256s() -> dict[str, str]:
    """File name to pinned SHA-256 of every dev font."""
    return {f.file: f.sha256 for f in DEV_FONTS.values()}
