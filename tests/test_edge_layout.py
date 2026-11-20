from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import edge_layout as el
import measure_edge as me
import pypdfium2 as pdfium

LINES = [("ab<", "rtl"), ("cd", "ltr")]


def _dom(records):
    """Page text as pdfium returns it: the marked JSON broken over lines."""
    text = "@@" + json.dumps(records, separators=(",", ":")).replace(" ", "\\u0020") + "@@"
    return "\n".join(text[i : i + 40] for i in range(0, len(text), 40))


def _good():
    box = [0.0, 0.0, 8.0, 18.0]
    return [{"text": t, "dir": d, "boxes": [box] * len(t)} for t, d in LINES]


def test_html_escapes_and_sets_direction():
    page = el.build_html(LINES)
    assert '<p dir="rtl">ab&lt;</p>' in page and '<p dir="ltr">cd</p>' in page
    assert page.count("<p ") == 2 and 'id="layout"' in page
    assert "white-space: nowrap" in page and "16px Arial" in page


def test_lines_cover_both_directions():
    assert {d for _, d in el.LINES} == {"rtl", "ltr"}


def test_parse_good():
    assert el.parse_layout(_dom(_good()), LINES) == _good()


def _bad(dom):
    try:
        el.parse_layout(dom, LINES)
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("no error")


def test_parse_missing_pre():
    assert "no layout" in _bad("nothing here")


def test_parse_bad_json():
    assert "unreadable" in _bad("@@{@@")


def test_parse_wrong_counts_and_text():
    assert "number of lines" in _bad(_dom(_good()[:1]))
    short = _good()
    short[0]["boxes"] = short[0]["boxes"][:1]
    assert "character boxes" in _bad(_dom(short))
    other = _good()
    other[1]["text"] = "xx"
    assert "differs" in _bad(_dom(other))
    nan = _good()
    nan[0]["boxes"][0] = [0, 0, 1]
    assert "invalid" in _bad(_dom(nan))
    assert "differs" in _bad(_dom(["x", "y"]))


def test_round_records():
    rec = [{"text": "a", "dir": "ltr", "boxes": [[0.123456, 1, 2, 3]]}]
    assert el.round_records(rec)[0]["boxes"] == [[0.12, 1, 2, 3]]


def _fake_run(version=b"Microsoft Edge 1.2.3\n"):
    def fake(cmd, **k):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, version, b"")
        doc = pdfium.PdfDocument.new()
        doc.new_page(200, 200)
        doc.save(Path(cmd[-2].split("=", 1)[1]))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    return fake


def _full():
    return [{"text": t, "dir": d, "boxes": [[0, 0, 1, 1]] * len(t)} for t, d in el.LINES]


def test_main_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "find_edge", lambda: Path("edge.exe"))
    monkeypatch.setattr(me.subprocess, "run", _fake_run())
    monkeypatch.setattr(el, "pdf_text", lambda pdf: _dom(_full()))
    out = tmp_path / "o" / "layout.json"
    assert el.main(["--work", str(tmp_path / "w"), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source"] == "Microsoft Edge 1.2.3" and data["font"] == "16px Arial"
    assert len(data["lines"]) == len(el.LINES)
    assert (tmp_path / "w" / "layout.html").is_file()


def test_version_unknown(monkeypatch):
    monkeypatch.setattr(me.subprocess, "run", _fake_run(version=b""))
    assert el.edge_version(Path("e")) == "Microsoft Edge unknown"

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("e", 20)

    monkeypatch.setattr(me.subprocess, "run", boom)
    assert el.edge_version(Path("e")) == "Microsoft Edge unknown"


def test_main_bad_dom_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(el, "find_edge", lambda: Path("edge.exe"))
    monkeypatch.setattr(me.subprocess, "run", _fake_run())
    monkeypatch.setattr(el, "pdf_text", lambda pdf: "no data")
    assert el.main(["--work", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and err.count("\n") == 1


def test_main_edge_missing_and_timeout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(el, "find_edge", lambda: None)
    assert el.main(["--work", str(tmp_path)]) == 1
    assert "not found" in capsys.readouterr().err
    monkeypatch.setattr(el, "find_edge", lambda: Path("edge.exe"))

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("e", 120)

    monkeypatch.setattr(me.subprocess, "run", boom)
    assert el.main(["--work", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 1
    assert "120 seconds" in capsys.readouterr().err


def test_work_is_a_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(el, "find_edge", lambda: Path("edge.exe"))
    target = tmp_path / "f.txt"
    target.write_text("x")
    assert el.main(["--work", str(target)]) == 1
    err = capsys.readouterr().err
    assert "not a folder" in err and err.count("\n") == 1


def test_out_is_a_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(el, "find_edge", lambda: Path("edge.exe"))
    monkeypatch.setattr(me.subprocess, "run", _fake_run())
    monkeypatch.setattr(el, "pdf_text", lambda pdf: _dom(_full()))
    assert el.main(["--work", str(tmp_path / "w"), "--out", str(tmp_path)]) == 1
    assert capsys.readouterr().err.count("\n") == 1


def test_pdf_text_reads_printed_page(tmp_path):
    pdf = tmp_path / "t.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    doc.save(pdf)
    assert el.pdf_text(pdf).strip() == ""


def test_unreadable_pdf_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(el, "find_edge", lambda: Path("edge.exe"))

    def fake(cmd, **k):
        Path(cmd[-2].split("=", 1)[1]).write_bytes(b"not a pdf")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(me.subprocess, "run", fake)
    assert el.main(["--work", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "cannot read the PDF printed under" in err
