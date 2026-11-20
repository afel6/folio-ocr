"""Records, per character, how Microsoft Edge lays out the must-not-flag lines.

Usage: python scripts/edge_layout.py --work DIR [--out tests/data/edge_must_not_flag_layout.json]
Writes DIR/layout.html, lets Edge lay it out and print it to DIR/layout.pdf, reads the box of
every character back from the text the page prints about itself and writes them to --out, so unit tests can compare against a browser's
bidirectional layout rather than our own.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pypdfium2 as pdfium
from measure_edge import find_edge, print_html
from must_not_flag import LTR, MEASURE_ONLY, RTL
from script_errors import describe_os_error

FONT = "16px Arial"
LINES = (
    [(t, "rtl") for t in RTL] + [(t, "ltr") for t in LTR] + [(t, d) for t, d in MEASURE_ONLY]
)
SCRIPT = (
    "const out = [];\n"
    "for (const p of document.querySelectorAll('p')) {\n"
    "  const node = p.firstChild, boxes = [];\n"
    "  for (let i = 0; i < node.length; i++) {\n"
    "    const r = document.createRange();\n"
    "    r.setStart(node, i); r.setEnd(node, i + 1);\n"
    "    const b = r.getBoundingClientRect();\n"
    "    boxes.push([b.left, b.top, b.right, b.bottom]);\n"
    "  }\n"
    "  out.push({text: p.textContent, dir: p.dir, boxes: boxes});\n"
    "}\n"
    "const esc = String.fromCharCode(92) + 'u';\n"
    "const json = JSON.stringify(out).replace(/[^\\x21-\\x7e]/g, (c) =>\n"
    "  esc + c.charCodeAt(0).toString(16).padStart(4, '0'));\n"
    "document.getElementById('layout').textContent = '@@' + json + '@@';\n"
    "document.querySelectorAll('p').forEach((p) => { p.style.display = 'none'; });\n"
)
MARKED = re.compile(r"@@(.*?)@@")


def build_html(lines: Sequence[tuple[str, str]]) -> str:
    body = "\n".join(f'<p dir="{d}">{html.escape(t)}</p>' for t, d in lines)
    return (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f"<style>p {{ white-space: nowrap; font: {FONT}; margin: 4px 0; }}\n"
        "pre { font: 7px monospace; white-space: pre-wrap; word-break: break-all; }</style>\n"
        f'</head>\n<body>\n{body}\n<pre id="layout"></pre>\n<script>\n{SCRIPT}</script>\n'
        "</body>\n</html>\n"
    )


def _valid_box(box: object) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
    )


def parse_layout(printed: str, lines: Sequence[tuple[str, str]]) -> list[dict]:
    """The layout records in the printed page text; raises RuntimeError when malformed."""
    found = MARKED.search("".join(printed.split()))
    if found is None:
        raise RuntimeError("Microsoft Edge returned no layout data")
    try:
        records = json.loads(found.group(1))
    except json.JSONDecodeError:
        raise RuntimeError("Microsoft Edge returned unreadable layout data") from None
    if not isinstance(records, list) or len(records) != len(lines):
        raise RuntimeError("Microsoft Edge returned the wrong number of lines")
    for rec, (text, direction) in zip(records, lines, strict=True):
        if not isinstance(rec, dict) or rec.get("text") != text or rec.get("dir") != direction:
            raise RuntimeError("Microsoft Edge returned a line that differs from its source")
        boxes = rec.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != len(text):
            raise RuntimeError("Microsoft Edge returned the wrong number of character boxes")
        if not all(_valid_box(b) for b in boxes):
            raise RuntimeError("Microsoft Edge returned an invalid character box")
    return records


def round_records(records: Sequence[dict]) -> list[dict]:
    return [
        {"text": r["text"], "dir": r["dir"],
         "boxes": [[round(v, 2) for v in b] for b in r["boxes"]]}
        for r in records
    ]  # fmt: skip


def edge_version(edge: Path) -> str:
    """'Microsoft Edge <version>' when msedge answers --version, else '... unknown'."""
    try:
        done = subprocess.run([str(edge), "--version"], check=False, timeout=20,
                              capture_output=True)  # fmt: skip
        words = done.stdout.decode("utf-8", "replace").split()
    except (OSError, subprocess.TimeoutExpired):
        words = []
    known = bool(words) and words[-1][:1].isdigit()
    return f"Microsoft Edge {words[-1] if known else 'unknown'}"


def pdf_text(pdf_path: Path) -> str:
    """All text of the printed PDF."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return "".join(page.get_textpage().get_text_range() for page in doc)
    finally:
        doc.close()


def measure(edge: Path, work: Path) -> list[dict]:
    # Edge on Windows writes nothing to stdout for --dump-dom, so the page prints its own data.
    pdf = print_html(edge, work, "layout", build_html(LINES))
    return round_records(parse_layout(pdf_text(pdf), LINES))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Edge's per-character line layout.")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("tests/data/edge_must_not_flag_layout.json"))
    args = parser.parse_args(argv)
    edge = find_edge()
    if edge is None:
        sys.stderr.write("error: Microsoft Edge not found\n")
        return 1
    try:
        args.work.mkdir(parents=True, exist_ok=True)
        records = measure(edge, args.work)
        data = {"source": edge_version(edge), "font": FONT, "lines": records}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    except OSError as exc:
        sys.stderr.write(f"error: {describe_os_error(exc, args.work)}\n")
        return 1
    except pdfium.PdfiumError:  # a RuntimeError, so it comes first
        sys.stderr.write(f"error: cannot read the PDF printed under {args.work}\n")
        return 1
    except RuntimeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
