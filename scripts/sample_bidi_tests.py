"""Writes the small samples of the Unicode data files used by the tests.

The full files (BidiCharacterTest.txt, BidiTest.txt and extracted/DerivedBidiClass.txt of the
Unicode version of the running Python) are fetched by hand from unicode.org; this script
checks their sha256 against the table for that version, keeps the cases without explicit
embeddings, overrides, isolates or paragraph separators in a fixed stride, keeps the @missing
lines and unassigned-code-point lines of DerivedBidiClass.txt, and writes them under
tests/data/unicode/<version>.
Usage: python scripts/sample_bidi_tests.py CHARACTER_TEST_FILE BIDI_TEST_FILE
       DERIVED_BIDI_CLASS_FILE [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import unicodedata
from pathlib import Path

from script_errors import describe_os_error

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "BidiCharacterTest.txt": "ucd/BidiCharacterTest.txt",
    "BidiTest.txt": "ucd/BidiTest.txt",
    "DerivedBidiClass.txt": "ucd/extracted/DerivedBidiClass.txt",
}
SHA256 = {
    "15.0.0": {
        "BidiCharacterTest.txt": "3c423c301f7b8dc41b879062cbf01fd1b4ec2ea4826e20d276c44b52129a01b6",
        "BidiTest.txt": "72a7a509dba0e147322c17997fb5159431042ff4a49fa08c7c25ccc1e291bbfe",
        "DerivedBidiClass.txt": "4841f2090c2dbc592d3ce43bb74c2191b3da50fb9a0d00274f1448c202851b02",
    },
    "15.1.0": {
        "BidiCharacterTest.txt": "18507f7ec57ec8155b7771dbc88f3e3201ad08eba4dba70e4e095ef628020fa8",
        "BidiTest.txt": "23413900dddcd2a246440f1f0cf27a3e1e62b3d0aebfa8f44ce241f5950fe956",
        "DerivedBidiClass.txt": "b57884c59a3a5348d86faed39965bbddb4e4493d3c16c1ec378ec26bfb5821be",
    },
}
SAMPLES = ROOT / "tests" / "data" / "unicode"  # one folder per Unicode version
EXPLICIT_CODES = frozenset(range(0x202A, 0x202F)) | frozenset(range(0x2066, 0x206A))
EXPLICIT_CLASSES = frozenset({"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI", "B"})
HAND_WRITTEN_LINES = 1250  # the UAX #9 examples, the misc. cases and the first "()" block
CHARACTER_STRIDE = 61
BIDI_TEST_STRIDE = 13


def source_url(version: str, name: str) -> str:
    """unicode.org address of the Unicode `version` data file `name`."""
    return f"https://www.unicode.org/Public/{version}/{SOURCES[name]}"


def _checked(path: Path, name: str, version: str) -> list[str]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256[version][name]:
        raise ValueError(f"{path.name} is not the Unicode {version} {name}")
    return data.decode("utf-8").splitlines()


def _header(name: str, rule: str, lines: list[str], version: str) -> list[str]:
    notice = [line for line in lines[:4] if line.startswith("#")]
    return [
        *notice,
        f"# Sample of {source_url(version, name)}",
        f"# Source sha256: {SHA256[version][name]}",
        f"# Kept: {rule}",
    ]


def _plain_case(line: str) -> bool:
    codes, direction = line.split(";")[:2]
    points = [int(cp, 16) for cp in codes.split()]
    return direction in ("0", "1") and not any(
        cp in EXPLICIT_CODES or unicodedata.bidirectional(chr(cp)) == "B" for cp in points
    )


def sample_character_test(lines: list[str], version: str) -> list[str]:
    rule = (
        f"direction 0 or 1, no U+202A-202E, U+2066-2069 or class B; every case in the first "
        f"{HAND_WRITTEN_LINES} lines, then every {CHARACTER_STRIDE}th"
    )
    out, seen = _header("BidiCharacterTest.txt", rule, lines, version), 0
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("#") or not _plain_case(line):
            continue
        seen += 1
        if number <= HAND_WRITTEN_LINES or seen % CHARACTER_STRIDE == 0:
            out.append(line)
    return out


def sample_bidi_test(lines: list[str], version: str) -> list[str]:
    rule = (
        "class sequences without explicit classes or B, for LTR or RTL paragraphs "
        f"(bitset bit 2 or 4); every {BIDI_TEST_STRIDE}th"
    )
    out, context, seen = _header("BidiTest.txt", rule, lines, version), [], 0
    for line in lines:
        if line.startswith(("@Levels:", "@Reorder:")):
            context = [*context, line] if line.startswith("@Reorder:") else [line]
            continue
        if not line or line.startswith("#") or ";" not in line:
            continue
        classes, bitset = line.split(";")
        if EXPLICIT_CLASSES & set(classes.split()) or not int(bitset) & 6:
            continue
        seen += 1
        if seen % BIDI_TEST_STRIDE == 0:
            out += [*context, f"{classes};{int(bitset) & 6}"]
            context = []
    return out


def sample_unassigned_classes(lines: list[str], version: str) -> list[str]:
    rule = "the @missing lines and the lines of unassigned code points (general category Cn)"
    kept = [
        line
        for line in lines
        if line.startswith("# @missing:")
        or (line and not line.startswith("#") and line.partition("#")[2].split()[:1] == ["Cn"])
    ]
    return [*_header("DerivedBidiClass.txt", rule, lines, version), *kept]


def write_samples(
    character_file: Path, bidi_file: Path, derived_file: Path, out_dir: Path, version: str
) -> tuple[Path, ...]:
    pairs = (
        (
            "bidi_character_test_sample.txt",
            sample_character_test(
                _checked(character_file, "BidiCharacterTest.txt", version), version
            ),
        ),
        (
            "bidi_test_sample.txt",
            sample_bidi_test(_checked(bidi_file, "BidiTest.txt", version), version),
        ),
        (
            "derived_bidi_class_unassigned.txt",
            sample_unassigned_classes(
                _checked(derived_file, "DerivedBidiClass.txt", version), version
            ),
        ),
    )
    out_dir.mkdir(exist_ok=True)
    paths = []
    for name, lines in pairs:
        path = out_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)


def main(argv: list[str] | None = None, unicode_version: str = unicodedata.unidata_version) -> int:
    parser = argparse.ArgumentParser(description="Sample the Unicode bidi data files.")
    parser.add_argument("character_test", type=Path)
    parser.add_argument("bidi_test", type=Path)
    parser.add_argument("derived_bidi_class", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if unicode_version not in SHA256:
        known = ", ".join(SHA256)
        sys.stderr.write(
            f"error: no source hashes for Unicode {unicode_version} (known: {known})\n"
        )
        return 1
    out = args.out if args.out is not None else SAMPLES / unicode_version
    try:
        write_samples(
            args.character_test, args.bidi_test, args.derived_bidi_class, out, unicode_version
        )
    except OSError as exc:
        sys.stderr.write(f"error: {describe_os_error(exc, out)}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"error: cannot sample: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
