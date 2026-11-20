"""Tests for the pinned downloadable assets: verification and fetching."""

import errno
import gzip
import hashlib
import tempfile

import httpx
import pytest

from folio_ocr import assets
from folio_ocr.assets import (
    MANIFEST,
    ONNXTR,
    TESSDATA,
    TESSDATA_DIR,
    TESSERACT_OSD,
    Asset,
    fetch,
    files_problem,
    verify,
)
from folio_ocr.errors import WeightsError

BODY = b"hello asset"
SHA = hashlib.sha256(BODY).hexdigest()
URL = "https://assets.example/a.bin"
ASSET = Asset("sub/a.bin", URL, SHA, len(BODY), "MIT")


class Server:
    def __init__(self, responder):
        self.calls = 0
        self.responder = responder
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        self.calls += 1
        return self.responder(request)


def serving(content=BODY, status=200, headers=None):
    return Server(lambda r: httpx.Response(status, content=content, headers=headers))


def files(directory):
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


def test_fetch_writes_verified_file(tmp_path):
    msgs = []
    fetch([ASSET], tmp_path, report=msgs.append, transport=serving().transport)
    assert (tmp_path / "sub" / "a.bin").read_bytes() == BODY
    assert verify([ASSET], tmp_path)[0].ok
    assert any("downloading" in m for m in msgs) and any("verified" in m for m in msgs)
    assert files(tmp_path) == ["a.bin"]


def test_second_fetch_makes_no_request(tmp_path):
    fetch([ASSET], tmp_path, transport=serving().transport)
    server, msgs = serving(), []
    fetch([ASSET], tmp_path, report=msgs.append, transport=server.transport)
    assert server.calls == 0
    assert any("already verified" in m for m in msgs)


def test_wrong_content_same_size_is_replaced(tmp_path):
    target = tmp_path / "sub" / "a.bin"
    target.parent.mkdir()
    target.write_bytes(b"x" * len(BODY))
    server = serving()
    fetch([ASSET], tmp_path, transport=server.transport)
    assert server.calls == 1 and target.read_bytes() == BODY


def test_sha_mismatch_leaves_nothing(tmp_path):
    with pytest.raises(WeightsError) as info:
        fetch([ASSET], tmp_path, transport=serving(b"x" * len(BODY)).transport)
    assert "sub/a.bin" in str(info.value) and len(str(info.value).splitlines()) == 1
    assert files(tmp_path) == []


def test_longer_body_refused_and_stops_early(tmp_path):
    chunks = [b"hello ", b"asset", b" more", b" and more"]
    with pytest.raises(WeightsError):
        fetch([ASSET], tmp_path, transport=serving(iter(chunks)).transport)
    assert files(tmp_path) == []


def test_shorter_body_refused(tmp_path):
    with pytest.raises(WeightsError):
        fetch([ASSET], tmp_path, transport=serving(BODY[:-1]).transport)
    assert files(tmp_path) == []


def test_http_404(tmp_path):
    with pytest.raises(WeightsError, match="404"):
        fetch([ASSET], tmp_path, transport=serving(b"", status=404).transport)
    assert files(tmp_path) == []


def test_redirect_to_https_is_followed(tmp_path):
    def responder(request):
        if request.url.host == "assets.example":
            return httpx.Response(302, headers={"Location": "https://cdn.example/b.bin"})
        return httpx.Response(200, content=BODY)

    fetch([ASSET], tmp_path, transport=Server(responder).transport)
    assert (tmp_path / "sub" / "a.bin").read_bytes() == BODY


def test_redirect_to_http_refused(tmp_path):
    def responder(request):
        if request.url.scheme == "https":
            return httpx.Response(302, headers={"Location": "http://cdn.example/b.bin"})
        return httpx.Response(200, content=BODY)

    with pytest.raises(WeightsError, match="not https"):
        fetch([ASSET], tmp_path, transport=Server(responder).transport)
    assert files(tmp_path) == []


@pytest.mark.parametrize(
    ("exc", "text"),
    [(httpx.ConnectError("boom"), "network error"), (httpx.ReadTimeout("slow"), "timed out")],
)
def test_transport_errors(tmp_path, exc, text):
    def responder(request):
        raise exc

    with pytest.raises(WeightsError, match=text) as info:
        fetch([ASSET], tmp_path, transport=Server(responder).transport)
    assert info.value.__cause__ is exc
    assert files(tmp_path) == []


def test_oserror_while_writing_propagates(tmp_path, monkeypatch):
    real = tempfile.NamedTemporaryFile

    class Failing:
        def __init__(self, *a, **k):
            self._f = real(*a, **k)
            self.name = self._f.name

        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self):
            self._f.close()

    monkeypatch.setattr(assets.tempfile, "NamedTemporaryFile", Failing)
    with pytest.raises(OSError) as info:
        fetch([ASSET], tmp_path, transport=serving().transport)
    assert info.value.errno == errno.ENOSPC
    assert files(tmp_path) == []


def test_replace_failure_propagates(tmp_path, monkeypatch):
    def boom(src, dst):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(assets.os, "replace", boom)
    with pytest.raises(PermissionError):
        fetch([ASSET], tmp_path, transport=serving().transport)
    assert files(tmp_path) == []


def test_gzip_body_is_decoded_before_hashing(tmp_path):
    server = serving(gzip.compress(BODY), headers={"Content-Encoding": "gzip"})
    fetch([ASSET], tmp_path, transport=server.transport)
    assert (tmp_path / "sub" / "a.bin").read_bytes() == BODY


def test_mkdir_error_propagates(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    with pytest.raises(OSError):
        fetch([ASSET], blocker / "dir", transport=serving().transport)


def test_stops_at_first_failure(tmp_path):
    second = Asset("b.bin", "https://assets.example/b.bin", SHA, len(BODY), "MIT")
    server = serving(b"x" * len(BODY))
    with pytest.raises(WeightsError):
        fetch([ASSET, second], tmp_path, transport=server.transport)
    assert server.calls == 1


def test_verify_reasons(tmp_path):
    assert verify([ASSET], tmp_path)[0].reason == "missing"
    target = tmp_path / "sub" / "a.bin"
    target.parent.mkdir()
    target.write_bytes(b"short")
    assert verify([ASSET], tmp_path)[0].reason == "wrong size"
    target.write_bytes(b"x" * len(BODY))
    status = verify([ASSET], tmp_path)[0]
    assert (status.ok, status.reason) == (False, "sha256 mismatch")
    target.write_bytes(BODY)
    status = verify([ASSET], tmp_path)[0]
    assert (status.ok, status.reason, status.asset) == (True, "ok", ASSET)
    target.unlink()
    target.mkdir()
    assert verify([ASSET], tmp_path)[0].reason == "cannot read"


def _pair(tmp_path):
    """Two pinned files, `first` and `second`, both intact on disk."""
    pair = [Asset(name, URL, SHA, len(BODY), "MIT") for name in ("first.bin", "second.bin")]
    for asset in pair:
        (tmp_path / asset.name).write_bytes(BODY)
    return pair


def test_files_problem_is_none_when_every_file_is_intact(tmp_path):
    assert files_problem(_pair(tmp_path), tmp_path) is None
    assert files_problem([], tmp_path) is None


@pytest.mark.parametrize("content", [b"short", b"x" * len(BODY)])
def test_files_problem_says_damaged_for_wrong_size_or_bytes(tmp_path, content):
    first, second = _pair(tmp_path)
    (tmp_path / second.name).write_bytes(content)
    assert files_problem([first, second], tmp_path) == "damaged"


def test_files_problem_missing_beats_cannot_read_and_damaged(tmp_path):
    first, second = _pair(tmp_path)
    third = Asset("third.bin", URL, SHA, len(BODY), "MIT")
    (tmp_path / first.name).unlink()
    (tmp_path / first.name).mkdir()  # cannot read
    (tmp_path / second.name).write_bytes(b"short")  # damaged
    assert files_problem([first, second], tmp_path) == "cannot read"
    assert files_problem([second, first, third], tmp_path) == "missing"
    assert files_problem([third, second, first], tmp_path) == "missing"


def test_files_problem_cannot_read_beats_damaged(tmp_path):
    first, second = _pair(tmp_path)
    (tmp_path / first.name).write_bytes(b"short")
    (tmp_path / second.name).unlink()
    (tmp_path / second.name).mkdir()
    assert files_problem([first, second], tmp_path) == "cannot read"
    assert files_problem([second, first], tmp_path) == "cannot read"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "http://assets.example/a.bin"},
        {"url": "https:///a.bin"},
        {"name": ""},
        {"name": "../x"},
        {"name": "/abs"},
        {"name": "a\\b"},
        {"name": "a//b"},
        {"name": "./a"},
        {"sha256": SHA.upper()},
        {"sha256": SHA[:-1]},
        {"size_bytes": 0},
    ],
)
def test_asset_refuses_bad_fields(kwargs):
    fields = {"name": "a.bin", "url": URL, "sha256": SHA, "size_bytes": 5, "licence": "MIT"}
    fields.update(kwargs)
    with pytest.raises(ValueError):
        Asset(**fields)


def test_manifest_contents():
    commit = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
    assert [a.name for a in TESSDATA] == [
        "tessdata/ara.traineddata",
        "tessdata/eng.traineddata",
    ]
    assert [a.size_bytes for a in TESSDATA] == [12_603_724, 15_400_601]
    for a in TESSDATA:
        assert a.url.startswith("https://raw.githubusercontent.com/") and commit in a.url
        assert a.licence == "Apache-2.0" and a.name.startswith(TESSDATA_DIR + "/")


def test_manifest_orientation_data():
    assert TESSERACT_OSD == Asset(
        "tessdata/osd.traineddata",
        "https://raw.githubusercontent.com/tesseract-ocr/tessdata/"
        "13665561c0162cbd8d94d1f8cab50585849de998/osd.traineddata",
        "9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff",
        10_562_727,
        "Apache-2.0",
    )
    assert MANIFEST == (*TESSDATA, TESSERACT_OSD, *ONNXTR)
    # the language data checks and the OCR setup key stay ara + eng
    assert TESSERACT_OSD not in TESSDATA


def test_manifest_onnxtr_models():
    assert MANIFEST[-len(ONNXTR) :] == ONNXTR
    assert [a.name for a in ONNXTR] == [
        "onnxtr/rep_fast_base-1b89ebf9.onnx",
        "onnxtr/crnn_vgg16_bn-743599aa.onnx",
    ]
    assert sum(a.size_bytes for a in ONNXTR) == 105_556_772  # under the 150 MB budget
    for a in ONNXTR:
        file = a.name.split("/")[1]
        assert a.url.startswith("https://github.com/felixdittrich92/OnnxTR/releases/download/v")
        assert a.url.endswith("/" + file) and a.licence == "Apache-2.0"
        assert a.sha256.startswith(file.rsplit("-", 1)[1].removesuffix(".onnx"))
