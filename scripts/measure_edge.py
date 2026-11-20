"""Prints Arabic lines to PDF with the installed Microsoft Edge and reports the text-layer flags.

Usage: python scripts/measure_edge.py --out DIR [--set arabic|must-not-flag]
Set "arabic" (default) writes edge-arabic.html and edge-arabic.pdf under DIR, then prints, for
every line pdfium extracts, its direction, its flags, its NORM_V2 error rate against the closest
source line and the extracted text. Set "must-not-flag" prints the correct lines of
must_not_flag.py as right-to-left, left-to-right and wrapped pages and compares each extracted
line with its source. Measurement only: the exit code does not depend on the flags.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pypdfium2 as pdfium
from must_not_flag import LTR, MEASURE_ONLY, RTL
from script_errors import describe_os_error

from folio_ocr.metrics import char_errors
from folio_ocr.normalize import norm_v2
from folio_ocr.text_layer import line_text, page_chars, segment_lines
from folio_ocr.text_layer_check import check_text_layer, page_direction

EDGE_PATHS = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
)
CER_LIMIT = 0.02
PDF_WAIT_SECONDS = 15  # Edge can return before the PDF is on disk
LINES = (
    "المبلغ الإجمالي 1,250.75 ريال",
    "تم دفع ٣٥٠٫٧٥ ريال بتاريخ ١٢/٠٥/٢٠٢٦",
    "خصم 15% على الطلب رقم 4587",
    "نسبة الإنجاز ٧٥٪ حتى الآن",
    "الموعد 2026-05-12 في المكتب الرئيسي",
    "الاسم: Tomas Okafor",
    "حضر الاجتماع السيد Daniel Mercer من الشركة",
    "الهاتف +966 50 123 4567",
    "العقد رقم CT-2026-3587 يغطي التوريد",
    "تاريخ الاستحقاق 30/06/2026",
)


def build_html(lines: Sequence[str]) -> str:
    body = "\n".join(f"<p>{html.escape(line)}</p>" for line in lines)
    return (
        '<!DOCTYPE html>\n<html lang="ar" dir="rtl">\n<head>\n<meta charset="utf-8">\n'
        '<style>body { font-family: "Arial"; font-size: 16px; }</style>\n'
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def find_edge() -> Path | None:
    for path in EDGE_PATHS:
        if path.exists():
            return path
    found = shutil.which("msedge")
    return Path(found) if found else None


def summarize(
    direction: str, reasons: Sequence[str], rows: Sequence[tuple[Sequence[str], float]]
) -> str:
    """One summary line from the page direction, page reasons and (flags, CER) per line."""
    over = [flags for flags, cer in rows if cer > CER_LIMIT]
    return (
        f"summary: page direction={direction} reasons={','.join(reasons) or 'none'} "
        f"lines={len(rows)} "
        f"visual_order={sum('visual_order' in flags for flags, _ in rows)} "
        f"text_layer_invalid={sum('text_layer_invalid' in flags for flags, _ in rows)} "
        f"cer_over_{CER_LIMIT}={len(over)} of_those_unflagged={sum(not f for f in over)}"
    )


def closest_cer(text: str, sources: Sequence[str]) -> float:
    hyp = norm_v2(text)
    return min(char_errors(norm_v2(src), hyp).rate or 0.0 for src in sources)


def read_lines(pdf_path: Path) -> list[tuple[object, list[tuple[str, object]]]]:
    """Per page: the text-layer check and its (extracted text, checked line) pairs."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        results = []
        for page in doc:
            chars = page_chars(page)
            segs = segment_lines(chars, page_direction(chars))
            check = check_text_layer(segs, has_ink=True)
            results.append((check, [(line_text(s), check.lines[i]) for i, s in enumerate(segs)]))
        return results
    finally:
        doc.close()


def report(pdf_path: Path, sources: Sequence[str]) -> None:
    for number, (check, lines) in enumerate(read_lines(pdf_path), 1):
        rows = []
        for i, (text, line) in enumerate(lines):
            cer = closest_cer(text, sources)
            rows.append((line.flags, cer))
            sys.stdout.write(
                f"{number}.{i} {line.direction} flags={','.join(line.flags) or '-'} "
                f"cer={cer:.3f} {text!a}\n"
            )
        sys.stdout.write(summarize(check.direction, check.reasons, rows) + "\n")


def run_edge(cmd: Sequence[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Runs Edge; raises RuntimeError with a friendly line on a timeout or a non-zero exit."""
    try:
        done = subprocess.run(list(cmd), check=False, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Microsoft Edge did not finish within {timeout} seconds") from None
    except OSError as exc:
        raise RuntimeError(f"cannot start Microsoft Edge: {exc.strerror or exc}") from None
    if done.returncode != 0:
        raise RuntimeError(f"Microsoft Edge exited with code {done.returncode}")
    return done


def wait_for_file(path: Path, seconds: float | None = None) -> bool:
    """True once `path` exists; Edge can return before its output is on disk."""
    deadline = time.monotonic() + (PDF_WAIT_SECONDS if seconds is None else seconds)
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)
    return path.is_file()


def print_html(edge: Path, out: Path, stem: str, page: str) -> Path:
    """Prints the HTML page to OUT/STEM.pdf with Edge; raises RuntimeError on failure."""
    html_path, pdf_path = out / f"{stem}.html", out / f"{stem}.pdf"
    html_path.write_text(page, encoding="utf-8", newline="\n")
    cmd = [
        str(edge), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        f"--user-data-dir={out / 'edge-profile'}", "--no-pdf-header-footer",
        "--print-to-pdf-no-header", f"--print-to-pdf={pdf_path}", html_path.as_uri(),
    ]  # fmt: skip
    pdf_path.unlink(missing_ok=True)
    run_edge(cmd)
    if not wait_for_file(pdf_path):
        raise RuntimeError("Microsoft Edge wrote no PDF")
    return pdf_path


def print_pdf(edge: Path, out: Path) -> Path:
    return print_html(edge, out, "edge-arabic", build_html(LINES))


WRAPPED = "تم تسليم الطلب إلى Tomas Okafor، ثم أرسلت النسخة (copy) إلى الإدارة - Finance - في 12/05/2026 بنسبة 50% من المبلغ."


class Row(NamedTuple):
    kind: str  # must-not-flag or measure-only
    direction: str
    raw_equal: bool
    norm_equal: bool
    cer: float
    flags: tuple[str, ...]
    source: str
    extracted: str

def build_set_html(lines: Sequence[str], direction: str) -> str:
    """One unwrapped paragraph per line inside a page of the given direction."""
    lang = "ar" if direction == "rtl" else "en"
    body = "\n".join(f"<p>{html.escape(line)}</p>" for line in lines)
    return (
        f'<!DOCTYPE html>\n<html lang="{lang}" dir="{direction}">\n<head>\n<meta charset="utf-8">\n'
        '<style>body { font-family: "Arial"; font-size: 16px; } p { white-space: nowrap; }</style>\n'
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def build_wrapped_html() -> str:
    return (
        '<!DOCTYPE html>\n<html lang="ar">\n<head>\n<meta charset="utf-8">\n'
        '<style>body { font-family: "Arial"; font-size: 16px; }</style>\n</head>\n<body>\n'
        f'<p dir="rtl" style="width: 260px">{html.escape(WRAPPED)}</p>\n</body>\n</html>\n'
    )


def mnf_sources(direction: str) -> list[tuple[str, str]]:
    """(source line, kind) of the lines printed on the page of this direction."""
    lines = RTL if direction == "rtl" else LTR
    return [(t, "must-not-flag") for t in lines] + [
        (t, "measure-only") for t, d in MEASURE_ONLY if d == direction
    ]


def match_rows(extracted: Sequence[tuple[str, object]], sources: Sequence[tuple[str, str]]):
    """One Row per extracted line, matched to the source line with the lowest NORM_V2 CER."""
    rows = []
    for text, line in extracted:
        hyp = norm_v2(text)
        cer, src, kind = min(
            ((char_errors(norm_v2(s), hyp).rate or 0.0, s, k) for s, k in sources),
            key=lambda c: c[0],
        )
        rows.append(
            Row(kind, line.direction, text == src, norm_v2(src) == hyp, cer,
                tuple(line.flags), src, text)
        )  # fmt: skip
    return rows


def format_row(row: Row) -> str:
    return (
        f"{row.kind} dir={row.direction} raw_equal={'yes' if row.raw_equal else 'no'} "
        f"norm_equal={'yes' if row.norm_equal else 'no'} cer={row.cer:.3f} "
        f"flags={','.join(row.flags) or '-'} {row.source!a} {row.extracted!a}"
    )


def mnf_summary(label: str, rows: Sequence[Row]) -> str:
    """Counts of lines, lines equal to their source, order flags on those and the rest."""
    equal = [r for r in rows if r.norm_equal]
    unequal = [r for r in rows if not r.norm_equal]
    return (
        f"summary {label}: lines={len(rows)} equal_after_norm={len(equal)} "
        f"visual_order_on_equal={sum('visual_order' in r.flags for r in equal)} "
        f"not_equal={len(unequal)} not_equal_flagged={sum(bool(r.flags) for r in unequal)}"
    )


def report_wrapped(pdf_path: Path) -> None:
    for check, lines in read_lines(pdf_path):
        flagged = [(t, ln) for t, ln in lines if ln.flags]
        sys.stdout.write(f"wrapped: segments={len(lines)} flagged={len(flagged)}\n")
        for text, line in flagged:
            sys.stdout.write(f"wrapped dir={line.direction} flags={','.join(line.flags)} {text!a}\n")


def run_must_not_flag(edge: Path, out: Path) -> None:
    all_rows: list[Row] = []
    for direction in ("rtl", "ltr"):
        sources = mnf_sources(direction)
        page = build_set_html([s for s, _ in sources], direction)
        pdf = print_html(edge, out, f"mnf-{direction}", page)
        rows = [r for _, lines in read_lines(pdf) for r in match_rows(lines, sources)]
        for row in rows:
            sys.stdout.write(format_row(row) + "\n")
        sys.stdout.write(mnf_summary(direction, rows) + "\n")
        all_rows += rows
    sys.stdout.write(mnf_summary("overall", all_rows) + "\n")
    report_wrapped(print_html(edge, out, "mnf-wrapped", build_wrapped_html()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print lines to PDF with Edge and check them.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", dest="line_set", choices=("arabic", "must-not-flag"),
                        default="arabic")  # fmt: skip
    args = parser.parse_args(argv)
    edge = find_edge()
    if edge is None:
        sys.stderr.write("error: Microsoft Edge not found\n")
        return 1
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        if args.line_set == "arabic":
            report(print_pdf(edge, args.out), LINES)
        else:
            run_must_not_flag(edge, args.out)
    except OSError as exc:
        sys.stderr.write(f"error: {describe_os_error(exc, args.out)}\n")
        return 1
    except pdfium.PdfiumError:  # a RuntimeError, so it comes first
        sys.stderr.write(f"error: cannot read the PDF printed under {args.out}\n")
        return 1
    except RuntimeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
