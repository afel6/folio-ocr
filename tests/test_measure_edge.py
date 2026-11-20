from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_edge as me


def test_html_escapes_markup():
    html = me.build_html(["a < b & c"])
    assert "a &lt; b &amp; c" in html
    assert '<html lang="ar" dir="rtl">' in html
    assert '<meta charset="utf-8">' in html
    assert html.count("<p>") == 1


def test_edge_discovery_order(monkeypatch, tmp_path):
    first, second = tmp_path / "a.exe", tmp_path / "b.exe"
    monkeypatch.setattr(me, "EDGE_PATHS", (first, second))
    monkeypatch.setattr(me.shutil, "which", lambda name: str(tmp_path / "w.exe"))
    assert me.find_edge() == Path(str(tmp_path / "w.exe"))
    second.write_text("x")
    assert me.find_edge() == second
    first.write_text("x")
    assert me.find_edge() == first


def test_edge_discovery_none(monkeypatch):
    monkeypatch.setattr(me, "EDGE_PATHS", ())
    monkeypatch.setattr(me.shutil, "which", lambda name: None)
    assert me.find_edge() is None


def test_summary_counts_flags_and_unflagged_errors():
    rows = [((), 0.0), (("visual_order",), 0.5), (("text_layer_invalid",), 0.3), ((), 0.1)]
    text = me.summarize("rtl", ("invisible_text",), rows)
    assert text == (
        "summary: page direction=rtl reasons=invisible_text lines=4 visual_order=1 "
        "text_layer_invalid=1 cer_over_0.02=3 of_those_unflagged=1"
    )


def test_summary_empty():
    assert "lines=0" in me.summarize("ltr", (), [])
    assert "reasons=none" in me.summarize("ltr", (), [])


def _run_main(tmp_path, capsys, monkeypatch, fake_run):
    monkeypatch.setattr(me, "find_edge", lambda: Path("edge.exe"))
    monkeypatch.setattr(me.subprocess, "run", fake_run)
    code = me.main(["--out", str(tmp_path / "out")])
    return code, capsys.readouterr().err


def test_timeout_exits_one(tmp_path, capsys, monkeypatch):
    def fake(*a, **k):
        raise subprocess.TimeoutExpired("edge", 120)

    code, err = _run_main(tmp_path, capsys, monkeypatch, fake)
    assert code == 1 and err.startswith("error:") and err.count("\n") == 1


def test_nonzero_exit_exits_one(tmp_path, capsys, monkeypatch):
    code, err = _run_main(
        tmp_path, capsys, monkeypatch, lambda *a, **k: subprocess.CompletedProcess(a, 3, b"", b"")
    )
    assert code == 1 and err.count("\n") == 1


def test_missing_pdf_exits_one(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(me, "PDF_WAIT_SECONDS", 0)
    code, err = _run_main(
        tmp_path, capsys, monkeypatch, lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b"")
    )
    assert code == 1 and "no PDF" in err and err.count("\n") == 1


def test_unreadable_pdf_exits_one(tmp_path, capsys, monkeypatch):
    def fake(cmd, **k):
        Path(cmd[-2].split("=", 1)[1]).write_bytes(b"not a pdf")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    code, err = _run_main(tmp_path, capsys, monkeypatch, fake)
    assert code == 1 and err.count("\n") == 1
    assert "cannot read the PDF printed under" in err


def test_edge_that_cannot_start_names_edge(tmp_path, capsys, monkeypatch):
    def fake(cmd, **k):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    code, err = _run_main(tmp_path, capsys, monkeypatch, fake)
    assert code == 1 and err.startswith("error: cannot start Microsoft Edge")


def test_edge_not_found(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(me, "find_edge", lambda: None)
    assert me.main(["--out", str(tmp_path)]) == 1
    assert capsys.readouterr().err == "error: Microsoft Edge not found\n"


def test_out_is_a_file(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(me, "find_edge", lambda: Path("edge.exe"))
    target = tmp_path / "file.txt"
    target.write_text("x")
    assert me.main(["--out", str(target)]) == 1
    err = capsys.readouterr().err
    assert "not a folder" in err and err.count("\n") == 1


def test_report_on_real_pdf(tmp_path, capsys, monkeypatch):
    import pypdfium2 as pdfium

    pdf = tmp_path / "t.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    doc.save(pdf)
    me.report(pdf, ["abc"])
    assert "summary:" in capsys.readouterr().out


def test_set_html_directions_and_one_p_per_line():
    page = me.build_set_html(["a < b", "c"], "ltr")
    assert '<html lang="en" dir="ltr">' in page and "a &lt; b" in page
    assert page.count("<p>") == 2 and "white-space: nowrap" in page
    assert '<html lang="ar" dir="rtl">' in me.build_set_html(["x"], "rtl")


def test_mnf_sources_split_by_direction():
    rtl, ltr = me.mnf_sources("rtl"), me.mnf_sources("ltr")
    assert len(rtl) == len(me.RTL) + 3 and len(ltr) == len(me.LTR) + 2
    assert [k for _, k in rtl].count("measure-only") == 3
    assert ("Friday) and send (copy", "measure-only") in ltr


def test_wrapped_html_is_rtl_width_260():
    page = me.build_wrapped_html()
    assert '<p dir="rtl" style="width: 260px">' in page and "Tomas Okafor" in page


class _Line:
    def __init__(self, flags, direction="rtl"):
        self.flags, self.direction = flags, direction


def test_match_rows_and_format():
    sources = [("abc", "must-not-flag"), ("xyz", "measure-only")]
    rows = me.match_rows([("xyz", _Line(())), ("abd", _Line(("visual_order",), "ltr"))], sources)
    assert rows[0].kind == "measure-only" and rows[0].raw_equal and rows[0].norm_equal
    assert rows[1].source == "abc" and not rows[1].norm_equal and rows[1].cer > 0
    assert "raw_equal=no norm_equal=no" in me.format_row(rows[1])
    assert "flags=visual_order" in me.format_row(rows[1])


def _row(norm_equal, flags=()):
    return me.Row("must-not-flag", "rtl", norm_equal, norm_equal, 0.0, flags, "s", "e")


def test_mnf_summary():
    rows = [_row(True), _row(True, ("visual_order",)), _row(False, ("text_layer_invalid",)),
            _row(False)]  # fmt: skip
    assert me.mnf_summary("rtl", rows) == (
        "summary rtl: lines=4 equal_after_norm=2 visual_order_on_equal=1 "
        "not_equal=2 not_equal_flagged=1"
    )
    assert "lines=0" in me.mnf_summary("overall", [])


def test_must_not_flag_set_runs_three_pdfs(tmp_path, capsys, monkeypatch):
    import pypdfium2 as pdfium

    def fake(cmd, **k):
        doc = pdfium.PdfDocument.new()
        doc.new_page(200, 200)
        doc.save(Path(cmd[-2].split("=", 1)[1]))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(me, "find_edge", lambda: Path("edge.exe"))
    monkeypatch.setattr(me.subprocess, "run", fake)
    out = tmp_path / "o"
    assert me.main(["--out", str(out), "--set", "must-not-flag"]) == 0
    text = capsys.readouterr().out
    assert "summary overall: lines=0" in text and "wrapped: segments=0" in text
    assert all((out / f"mnf-{n}.pdf").is_file() for n in ("rtl", "ltr", "wrapped"))
